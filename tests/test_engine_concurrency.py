"""Concurrent requests on one engine get the answers they get alone.

The server answers requests on FastAPI's thread pool, so several `evaluate` calls run at
once on one SystemOneEngine. Option offsets kept on the engine between `_fit` and
`_option_idx` were overwritten by a concurrent request, and the pointer head read another
request's option positions. CPU only: a tiny random torso and a byte-level tokenizer.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.schema import ChoiceQuestion, SystemOneRequest


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["<state> <question> <options> <answer> choice"],
                            trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<eos>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


@pytest.fixture(scope="module")
def engine():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    tok = _tokenizer()
    torso = Qwen3Model(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8))
    cfg = StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16,
                               use_lora=False, torch_dtype="float32", max_length=512)
    model = StrandsDeciderModel(cfg, torso, tok)
    return SystemOneEngine(model, EngineConfig(device="cpu"))


def _requests():
    """Requests whose option positions differ, so a swapped offset list changes the answer."""
    out = []
    for i in range(24):
        opts = [f"option{k}" + "x" * ((i * 7 + k * 3) % 11) for k in range(2 + i % 4)]
        out.append(SystemOneRequest(
            state="s" * (5 + (i * 13) % 40),
            questions={"q": ChoiceQuestion(type="choice", instructions="pick " * (1 + i % 5),
                                           criteria={o: "" for o in opts})}))
    return out


def _probs(resp):
    return resp.answers["q"].model_dump()


def test_concurrent_requests_match_serial(engine, monkeypatch):
    reqs = _requests()
    with torch.inference_mode():
        serial = [_probs(engine.evaluate(r)) for r in reqs]

    # Hold every thread between tokenising and pointing, so they interleave there each run.
    fit = SystemOneEngine._fit
    barrier = threading.Barrier(4, timeout=5)

    def slow_fit(self, *a, **kw):
        out = fit(self, *a, **kw)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        time.sleep(0.01)
        return out

    monkeypatch.setattr(SystemOneEngine, "_fit", slow_fit)

    def run(r):
        with torch.inference_mode():
            return _probs(engine.evaluate(r))

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(run, reqs))
    assert concurrent == serial


def test_engine_keeps_no_per_request_state(engine):
    with torch.inference_mode():
        engine.evaluate(_requests()[0])
    assert not hasattr(engine, "_last_offsets")
