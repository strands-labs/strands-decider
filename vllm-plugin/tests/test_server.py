"""The plugin's server without a GPU: rendering, the collapsed prompt vLLM expands, the image decode, the probabilities
and v1's response format, and the HTTP refusals, against a stand-in for vLLM's engine. Needs the core library
(strands_decider.core, PR #129), tokenizers, Pillow, FastAPI and vLLM (for SamplingParams); skipped without them."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from types import SimpleNamespace

import pytest

core = pytest.importorskip("strands_decider.core")
pytest.importorskip("tokenizers")
PIL = pytest.importorskip("PIL.Image")
pytest.importorskip("fastapi")
pytest.importorskip("vllm")
from fastapi.testclient import TestClient  # noqa: E402

from strands_decider_vllm import (  # noqa: E402
    ARCH,
    ARCH_MULTIMODAL,
    ARCH_UNIFIED,
    PREFIX_END_KEY,
    SERVER_CONFIG_KEY,
    server,
)

IMAGE_OPEN, IMAGE_SOFT, IMAGE_CLOSE = "<|image>", "<|image|>", "<image|>"


def toy_tokenizer():
    """A word-level tokenizer with Gemma 4's control tokens, its image tokens and every code as one token."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words + list(core.CODES)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    specials = ("<bos>", "<|turn>", "<turn|>", "<eos>", IMAGE_OPEN, IMAGE_SOFT, IMAGE_CLOSE)
    tok.add_special_tokens([AddedToken(s, special=True) for s in specials])
    return tok


def png(w, h, color=(200, 30, 30, 255), mode="RGBA"):
    buf = io.BytesIO()
    PIL.new(mode, (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


@pytest.fixture
def checkpoint(tmp_path):
    def make(inputs=("text", "image"), processor=None, prompt="strands-plain"):
        tok = toy_tokenizer()
        tok.save(str(tmp_path / "tokenizer.json"))
        dc = {
            "interface": "strands-decider",
            "format_version": 1,
            "prompt": prompt,
            "inputs": list(inputs),
            "codes": list(core.CODES),
            "token_ids": [tok.token_to_id(c) for c in core.CODES],
            "temperature": 1.0,
            "final_logit_softcapping": 30.0,
            "attention_mode": "prefix_bidir",
            "pooling": "last",
            "max_length_validated": 8192,
            "provenance": {
                "base_model": "google/gemma-4-E2B-it",
                "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            },
        }
        (tmp_path / "decision_config.json").write_text(json.dumps(dc))
        (tmp_path / "config.json").write_text(
            json.dumps({"architectures": ["Gemma4ForConditionalGeneration"], "text_config": {}})
        )
        pc = (
            processor
            if processor is not None
            else {"image_processor": dict(server.PROCESSOR_SETTINGS)}
        )
        (tmp_path / "processor_config.json").write_text(json.dumps(pc))
        return server.Checkpoint.open(str(tmp_path)), tok

    return make


class FakeEngine:
    """vLLM's AsyncLLM as the server uses it: it expands each <|image|> as Gemma 4's processor does (N from the
    image's size, by the specification's rule, or ``wrong_n``), and returns logprobs for the allowed ids."""

    def __init__(self, tok, logits=None, wrong_n=None, refuse=None):
        self.tok, self.logits, self.wrong_n, self.refuse = tok, logits or {}, wrong_n, refuse
        self.calls = []
        self.healthy = True
        self.cached = 7

    async def generate(self, prompt, sp, request_id):
        self.calls.append((prompt, sp))
        if self.refuse:
            raise self.refuse
        images = (prompt.get("multi_modal_data") or {}).get("image", [])
        soft, op, cl = (self.tok.token_to_id(t) for t in (IMAGE_SOFT, IMAGE_OPEN, IMAGE_CLOSE))
        out, i = [], 0
        if images:
            assert prompt["mm_processor_kwargs"] == {"do_resize": False}
        for t in prompt["prompt_token_ids"]:
            if t == soft:
                w, h = images[
                    i
                ].size  # the processor, its resize off, gives one token per 48 x 48 pixels
                n = self.wrong_n or (w // 48) * (h // 48)
                out += [op] + [soft] * n + [cl]
                i += 1
            else:
                out.append(t)
        lp = {
            t: SimpleNamespace(logprob=self.logits.get(j, -float(j)))
            for j, t in enumerate(sp.allowed_token_ids)
        }
        yield SimpleNamespace(
            prompt_token_ids=out,
            outputs=[SimpleNamespace(logprobs=[lp])],
            num_cached_tokens=self.cached,
        )

    async def check_health(self):
        if not self.healthy:
            raise RuntimeError("EngineDeadError")

    def shutdown(self):
        self.shut = True


def decider(make, engine_kwargs=None, **kw):
    ckpt, tok = make(**kw)
    engine = FakeEngine(tok, **(engine_kwargs or {}))
    problem = None
    if "image" in json.loads(ckpt.files["decision_config.json"].read_text())["inputs"]:
        problem = server.processor_problem(ckpt.json("processor_config.json"))
    d = server.Decider(
        engine, ckpt, served_name="m", max_length=8192, prefix_caching=True, image_problem=problem
    )
    return d, engine, tok


QUESTIONS = {
    "color": {
        "type": "choice",
        "instructions": "Which color?",
        "criteria": {"red": None, "blue": None, "green": None},
    },
    "square": {"type": "noul", "instructions": "Is it a square?"},
    "size": {
        "type": "score",
        "instructions": "How large?",
        "criteria": ["small", "medium", "large"],
    },
}


# --- pieces ---------------------------------------------------------------------------------------------------------


def test_collapse_leaves_one_placeholder_per_image():
    ids = (2, 9, 100, 7, 7, 7, 101, 8, 100, 7, 7, 101, 5)
    images = [SimpleNamespace(start=3, end=6), SimpleNamespace(start=9, end=11)]
    assert server.collapse(ids, images, 7) == [2, 9, 7, 8, 7, 5]
    assert server.collapse(ids, [], 7) == list(ids)


def test_probabilities_renormalize_over_the_questions_codes():
    lp = {10: SimpleNamespace(logprob=math.log(0.2)), 11: SimpleNamespace(logprob=math.log(0.6))}
    p = server.probabilities(lp, (10, 11))
    assert p == pytest.approx([0.25, 0.75])
    assert server.probabilities(lp, (10, 11, 12))[2] == 0.0
    with pytest.raises(server.Mismatch):
        server.probabilities({}, (10,))


@pytest.mark.parametrize(
    "settings, ok",
    [
        ({"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280}, True),
        ({"patch_size": 14, "pooling_kernel_size": 3, "max_soft_tokens": 280}, False),
        ({"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 560}, False),
    ],
)
def test_the_image_processor_must_have_the_specifications_settings(settings, ok):
    assert (server.processor_problem({"image_processor": settings}) is None) is ok
    assert server.processor_problem(None) is not None


@pytest.mark.parametrize(
    "base, arch",
    [
        ("Gemma4ForConditionalGeneration", ARCH_MULTIMODAL),
        ("Gemma4UnifiedForConditionalGeneration", ARCH_UNIFIED),
        ("Gemma4TextModel", ARCH),
    ],
)
def test_the_plugin_architecture_follows_the_base(base, arch):
    assert server.plugin_architecture({"architectures": [base]}) == arch
    with pytest.raises(SystemExit):
        server.plugin_architecture({"architectures": ["Qwen3ForCausalLM"]})


@pytest.mark.parametrize(
    "w, h",
    [
        (1, 1),
        (64, 48),
        (300, 40),
        (640, 480),
        (4096, 4096),
        (1, 300),
        (300, 1),
        (1, 16777216),
        (13441, 47),
    ],
)
def test_resized_size_has_the_cores_soft_tokens(w, h):
    W, H = server.resized_size(w, h)
    assert W % 48 == 0 and H % 48 == 0 and W * H <= 645120
    assert (W // 48) * (H // 48) == core.soft_tokens(w, h)


def test_prepare_image_resizes_with_pillows_bicubic():
    data = png(64, 48, (10, 200, 30, 255), mode="RGBA")
    img = server.prepare_image(data, 266)
    expected = server.decode_image(data).resize((912, 672), PIL.Resampling.BICUBIC)
    assert img.size == (912, 672) and img.tobytes() == expected.tobytes()
    with pytest.raises(server.Mismatch, match="not to the core's 265 tokens"):
        server.prepare_image(data, 265)


def test_decode_image_composites_transparency_onto_white():
    img = server.decode_image(png(4, 3, (0, 0, 0, 0)))
    assert img.mode == "RGB" and img.size == (4, 3) and img.getpixel((0, 0)) == (255, 255, 255)


def test_decode_image_applies_the_exif_orientation():
    buf = io.BytesIO()
    exif = PIL.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to show it
    PIL.new("RGB", (8, 4), (10, 20, 30)).save(buf, format="JPEG", exif=exif)
    assert server.decode_image(buf.getvalue()).size == (4, 8)


def test_a_file_pillow_cannot_decode_is_refused():
    # A sound header whose compressed pixel data is garbage: the core's header checks pass, Pillow's decode fails.
    data = bytearray(png(64, 64, (1, 2, 3, 255)))
    start = data.index(b"IDAT") + 4
    data[start : start + 16] = b"\xff" * 16
    with pytest.raises(server.Refused, match="image could not be"):
        server.decode_image(bytes(data))


# --- the request path -----------------------------------------------------------------------------------------------


def test_a_text_request_is_answered_in_v1s_format(checkpoint):
    d, engine, _ = decider(checkpoint)
    out = asyncio.run(d.decide({"state": "A ticket.", "questions": QUESTIONS}))
    assert set(out) == {"model", "answers", "usage"} and out["model"] == "m"
    assert out["answers"]["color"]["choice"] == "red"  # logprobs 0, -1, -2
    assert set(out["answers"]["size"]) >= {"score", "probabilities", "confidence"}
    rendered = core.render(
        {"state": "A ticket.", "questions": QUESTIONS}, d.tokenizer, "strands-plain"
    )
    assert out["usage"] == {
        "input_tokens": rendered.input_tokens,
        "output_tokens": 3,
        "cached_tokens": 21,
    }
    for prompt, sp in engine.calls:
        assert "multi_modal_data" not in prompt
        assert (
            sp.max_tokens == 1
            and sp.temperature == 1.0
            and sp.logprobs == len(sp.allowed_token_ids)
        )
        assert sp.extra_args == {PREFIX_END_KEY: rendered.questions["color"].state_len}


def test_an_image_request_sends_the_collapsed_prompt_the_images_and_their_sha256(checkpoint):
    d, engine, tok = decider(checkpoint)
    files = [png(64, 48), png(300, 40, (0, 0, 255, 255))]
    body = {"state": "Two photos.", "images": [b64(f) for f in files], "questions": QUESTIONS}
    asyncio.run(d.decide(body))
    rendered = core.render(body, d.tokenizer, "strands-plain")
    s = rendered.questions["color"].state_len
    assert [im.tokens for im in rendered.images] == [266, 270]
    for prompt, sp in engine.calls:
        assert prompt["prompt_token_ids"].count(tok.token_to_id(IMAGE_SOFT)) == 2, (
            "one <|image|> per image"
        )
        assert tok.token_to_id(IMAGE_OPEN) not in prompt["prompt_token_ids"]
        # Resized with Pillow to the specification's W x H: 266 = 19 x 14 and 270 = 45 x 6 tokens of 48 pixels.
        assert [im.size for im in prompt["multi_modal_data"]["image"]] == [(912, 672), (2160, 288)]
        assert prompt["multi_modal_uuids"] == {"image": [im.sha256 for im in rendered.images]}
        assert sp.extra_args == {PREFIX_END_KEY: s}, "S counts the expanded image tokens"
    assert rendered.images[0].sha256 == __import__("hashlib").sha256(files[0]).hexdigest()


def test_the_first_question_runs_before_the_others(checkpoint):
    d, engine, _ = decider(checkpoint)
    order = []
    original = engine.generate

    async def generate(prompt, sp, rid):
        order.append(("start", sp.allowed_token_ids[0]))
        async for out in original(prompt, sp, rid):
            await asyncio.sleep(0)
            yield out
        order.append(("end", sp.allowed_token_ids[0]))

    engine.generate = generate
    asyncio.run(d.decide({"state": "x", "questions": QUESTIONS}))
    assert order[:2] == [("start", order[0][1]), ("end", order[0][1])], (
        "the state is cached before the rest"
    )
    assert len(order) == 6


@pytest.fixture
def client(checkpoint):
    def make(engine_kwargs=None, **kw):
        d, engine, _ = decider(checkpoint, engine_kwargs, **kw)

        async def factory():
            return d

        return TestClient(server.build_app(factory)), engine

    return make


def post(c, body):
    r = c.post("/v1/systemone", json=body)
    return r.status_code, r.json()


def test_http_text_and_image_requests_and_health(client):
    c, engine = client()
    with c:
        for path in ("/health", "/ping"):
            r = c.get(path)
            assert r.status_code == 200 and r.json()["inputs"] == ["text", "image"]
        st, out = post(c, {"state": "x", "questions": QUESTIONS})
        assert st == 200 and "latency_ms" in out
        st, out = post(c, {"state": "x", "images": [b64(png(50, 50))], "questions": QUESTIONS})
        assert st == 200 and out["answers"]["square"]["noul"] == pytest.approx(
            math.exp(-1) / (1 + math.exp(-1)), abs=1e-4
        )
        assert (
            c.post("/invocations", json={"state": "x", "questions": QUESTIONS}).status_code == 200
        )
        engine.healthy = False
        assert c.get("/health").status_code == 503
    assert engine.shut


@pytest.mark.parametrize(
    "body, match",
    [
        (
            {"state": "x", "images": [b64(png(8, 8))] * 5, "questions": QUESTIONS},
            "5 images; this server takes at most 4",
        ),
        ({"state": "x", "images": ["@@@"], "questions": QUESTIONS}, "image is not valid base64"),
        (
            {"state": "x", "images": [b64(b"GIF89a" + b"\x00" * 4)], "questions": QUESTIONS},
            "could not be read",
        ),
        ({"state": "x", "images": None, "questions": QUESTIONS}, "images must be a list"),
        ({"questions": QUESTIONS}, "state"),
        ({"state": "x", "questions": {}}, "question"),
    ],
)
def test_bad_requests_get_400_and_the_server_keeps_serving(client, body, match):
    c, _ = client()
    with c:
        st, out = post(c, body)
        assert st == 400 and match in out["detail"], out
        assert post(c, {"state": "x", "questions": QUESTIONS})[0] == 200


def test_images_for_a_text_only_checkpoint_are_refused_before_decoding(client):
    c, engine = client(inputs=("text",))
    with c:
        st, out = post(c, {"state": "x", "images": ["not even base64"], "questions": QUESTIONS})
        assert st == 400 and out["detail"] == (
            'this checkpoint does not take images: its decision_config.json inputs are ["text"]'
        )
        assert engine.calls == []


def test_images_for_another_image_processor_are_refused(client):
    c, _ = client(
        processor={
            "image_processor": {"patch_size": 14, "pooling_kernel_size": 3, "max_soft_tokens": 280}
        }
    )
    with c:
        st, out = post(c, {"state": "x", "images": [b64(png(8, 8))], "questions": QUESTIONS})
        assert st == 400 and "patch_size" in out["detail"]
        assert post(c, {"state": "x", "questions": QUESTIONS})[0] == 200, "text is still served"


def test_an_engine_refusal_is_400(client):
    from vllm.exceptions import VLLMValidationError

    c, _ = client(
        engine_kwargs={"refuse": VLLMValidationError("strands-decider: no", parameter="images")}
    )
    with c:
        st, out = post(c, {"state": "x", "questions": QUESTIONS})
        assert st == 400 and "strands-decider: no" in out["detail"]


def test_an_expansion_that_differs_from_the_render_is_500_never_an_answer(client):
    c, _ = client(engine_kwargs={"wrong_n": 5})
    with c:
        st, out = post(c, {"state": "x", "images": [b64(png(48, 48))], "questions": QUESTIONS})
        assert st == 500 and "not the core's N" in out["detail"]


def test_a_prompt_over_the_validated_length_is_refused(client):
    c, _ = client()
    with c:
        st, out = post(c, {"state": "word " * 9000, "questions": QUESTIONS})
        assert st == 400 and "exceeds the limit of 8192 tokens" in out["detail"]


# --- the engine's settings ------------------------------------------------------------------------------------------


def args(**over):
    a = dict(
        served_model_name="m",
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enforce_eager=False,
        mm_processor_cache_gb=4.0,
        dtype="bfloat16",
        block_size=None,
    )
    a.update(over)
    return SimpleNamespace(**a)


@pytest.mark.parametrize("inputs, image_limit", [(("text", "image"), 4), (("text",), 0)])
def test_engine_args_carry_the_plugins_flags(checkpoint, inputs, image_limit):
    ckpt, _ = checkpoint(inputs=inputs)
    e = server.engine_args(ckpt, args())
    assert e.hf_overrides == {
        "architectures": [ARCH_MULTIMODAL],
        "text_config": {"use_bidirectional_attention": None},
    }
    assert e.limit_mm_per_prompt == {"image": image_limit, "video": 0, "audio": 0}
    assert (e.max_model_len, e.max_num_batched_tokens, e.enable_chunked_prefill) == (
        8192,
        8192,
        False,
    )
    assert (e.dtype, e.logprobs_mode, e.max_logprobs, e.generation_config) == (
        "bfloat16",
        "processed_logprobs",
        255,
        "vllm",
    )
    assert e.prefix_caching_hash_algo == "sha256" and e.prefix_cache_retention_interval is None
    assert e.additional_config == {SERVER_CONFIG_KEY: server.__version__}
    assert (
        server.engine_args(
            ckpt, args(max_model_len=4096, max_num_batched_tokens=16384)
        ).max_num_batched_tokens
        == 16384
    )


def test_prepare_environment_keeps_the_callers_values(monkeypatch):
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.1")
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    server.prepare_environment("127.0.0.1")
    import os

    assert os.environ["VLLM_HOST_IP"] == "10.0.0.1" and os.environ["GLOO_SOCKET_IFNAME"] == "lo"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_the_cli_binds_loopback_by_default():
    from strands_decider_vllm.cli import parser

    a = parser().parse_args(["serve", "/ckpt"])
    assert a.host == "127.0.0.1" and a.enable_prefix_caching
