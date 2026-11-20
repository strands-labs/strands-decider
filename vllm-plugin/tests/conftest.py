"""The server's tests' toy checkpoint (its config files, with Gemma 4's control tokens and codes at
their ids, a few words at made-up ones, and no weights) and a stand-in for vLLM's engine."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from strands_decider import core
from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

from strands_decider_vllm.cli import parser
from strands_decider_vllm.server import build_app
from strands_decider_vllm.settings import settings

SOFT = core.CONTROL_TOKEN_IDS["<|image|>"]
OPEN, CLOSE = core.CONTROL_TOKEN_IDS["<|image>"], core.CONTROL_TOKEN_IDS["<image|>"]
LAYOUTS = {
    "multimodal": {"architectures": ["Gemma4ForConditionalGeneration"], "text_config": {}},
    "decoder": {"architectures": ["Gemma4TextModel"]},
}


def toy_tokenizer():
    specials = {"<pad>": 0, "<eos>": 1, "<unk>": 3, **core.CONTROL_TOKEN_IDS}
    vocab = {**specials, **dict(zip(core.CODES, core.CODE_TOKEN_IDS, strict=True))}
    free = (i for i in range(10, 1000) if i not in vocab.values())
    for w in ["\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]:
        vocab[w] = next(free)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    tok.add_special_tokens([AddedToken(t, special=True) for t in specials])
    return tok


@pytest.fixture
def checkpoint(tmp_path):
    """A directory with the toy checkpoint's config files, for the given inputs and layout."""

    def make(inputs=("text", "image"), layout="multimodal"):
        toy_tokenizer().save(str(tmp_path / "tokenizer.json"))
        files = {
            "decision_config.json": {
                "interface": "strands-decider",
                "format_version": 1,
                "prompt": "strands-chat",
                "inputs": list(inputs),
                "codes": list(core.CODES),
                "token_ids": list(core.CODE_TOKEN_IDS),
                "temperature": 1.0,
                "final_logit_softcapping": 30.0,
                "attention_mode": "prefix_bidir",
                "pooling": "last",
                "max_length_validated": 8192,
                "provenance": {
                    "base_model": "google/gemma-4-E2B-it",
                    "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                },
            },
            "config.json": LAYOUTS[layout],
            "processor_config.json": {
                "image_processor": {
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                }
            },
        }
        for name, content in files.items():
            (tmp_path / name).write_text(json.dumps(content))
        return str(tmp_path)

    return make


@pytest.fixture
def make_settings(checkpoint):
    """The server's settings for the toy checkpoint and the given flags."""

    def make(*flags, inputs=("text", "image"), layout="multimodal"):
        args = ["serve", checkpoint(inputs, layout), "--preprocess-workers", "2", *flags]
        return settings(parser().parse_args(args))

    return make


class FakeEngine:
    """vLLM's AsyncLLM as the server uses it: it expands each <|image|> to one placeholder per 48 x 48
    pixels (or ``wrong_n``), as the plugin's processor does, and gives the allowed ids logprobs."""

    def __init__(self, wrong_n=None, fail=None):
        self.wrong_n, self.fail = wrong_n, fail
        self.calls, self.healthy, self.shut = [], True, False

    async def generate(self, prompt, sp, request_id):
        self.calls.append((prompt, sp))
        if self.fail:
            raise self.fail
        images = iter((prompt.get("multi_modal_data") or {}).get("image", []))
        expanded = []
        for t in prompt["prompt_token_ids"]:
            if t == SOFT:
                w, h = next(images).size
                expanded += [OPEN, *[SOFT] * (self.wrong_n or (w // 48) * (h // 48)), CLOSE]
            else:
                expanded.append(t)
        logprobs = {
            t: SimpleNamespace(logprob=-float(j)) for j, t in enumerate(sp.allowed_token_ids)
        }
        output = SimpleNamespace(logprobs=[logprobs])
        yield SimpleNamespace(prompt_token_ids=expanded, outputs=[output], num_cached_tokens=7)

    async def check_health(self):
        if not self.healthy:
            raise RuntimeError("EngineDeadError")

    def shutdown(self):
        self.shut = True


@pytest.fixture
def server(make_settings):
    """The app for the toy checkpoint and the given flags, its engine and its settings."""

    def make(*flags, inputs=("text", "image"), **engine_kwargs):
        settings, engine = make_settings(*flags, inputs=inputs), FakeEngine(**engine_kwargs)
        app = build_app(settings, lambda: engine)
        return SimpleNamespace(app=app, client=TestClient(app), engine=engine, settings=settings)

    return make
