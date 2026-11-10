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


def decider(make, engine_kwargs=None, max_body=None, **kw):
    ckpt, tok = make(**kw)
    engine = FakeEngine(tok, **(engine_kwargs or {}))
    problem = None
    if "image" in json.loads(ckpt.files["decision_config.json"].read_text())["inputs"]:
        problem = server.processor_problem(ckpt.json("processor_config.json"))
    d = server.Decider(
        engine,
        ckpt,
        served_name="m",
        max_length=8192,
        prefix_caching=True,
        image_problem=problem,
        max_body=max_body,
        preprocess_workers=2,
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
    check = getattr(core, "check_image_processor", None)
    if check is not None:  # the core's own check, when it has one, gives the same verdict
        assert (server.processor_problem({"image_processor": settings}, check) is None) is ok


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
    if hasattr(core, "resized_size"):  # the core's own, which the server uses when the core has it
        assert core.resized_size(w, h) == (W, H)


def test_prepare_image_resizes_with_pillows_bicubic():
    data = png(64, 48, (10, 200, 30, 255), mode="RGBA")
    img = server.prepare_image(data, 266)
    expected = server.decode_image(data).resize((912, 672), PIL.Resampling.BICUBIC)
    assert img.size == (912, 672) and img.tobytes() == expected.tobytes()
    with pytest.raises(server.Mismatch, match="not to the core's 265 tokens"):
        server.prepare_image(data, 265)


def test_the_resize_is_the_widths_pass_then_the_heights():
    # A tall screenshot shrinks its height by more than twice its width: Pillow 13 would do the height first in one
    # call. Section 11.1's rule is the width's pass first, which two calls give on every Pillow.
    img = PIL.effect_noise((1080, 2400), 64).convert("RGB")
    W, H = server.resized_size(1080, 2400)
    want = img.resize((W, 2400), PIL.Resampling.BICUBIC).resize((W, H), PIL.Resampling.BICUBIC)
    assert server.bicubic(img, W, H).tobytes() == want.tobytes()
    assert server.bicubic(img, 1080, 2400) is img, "a pass that changes nothing is left out"
    if hasattr(core, "resize_image"):  # the core's, which the server uses when the core has it
        assert core.resize_image(img, W, H).tobytes() == want.tobytes()
        assert server.require_core().resize_image is core.resize_image


def test_a_decoded_size_the_core_refuses_is_a_refusal():
    def refuse(w, h):
        raise ValueError(f"image could not be decoded: its pixels are {w}x{h}")

    with pytest.raises(server.Refused, match="could not be decoded: its pixels are 64x48"):
        server.prepare_image(png(64, 48), 266, refuse)


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
    def make(engine_kwargs=None, max_body=None, **kw):
        d, engine, _ = decider(checkpoint, engine_kwargs, max_body, **kw)

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
    # AsyncLLM.generate raises VLLMValidationError for a request its engine core refuses (the end-to-end tiers'
    # tests/gpu/engine_checks.py checks this on a real engine).
    from vllm.exceptions import VLLMValidationError

    c, _ = client(
        engine_kwargs={"refuse": VLLMValidationError("strands-decider: no", parameter="images")}
    )
    with c:
        st, out = post(c, {"state": "x", "questions": QUESTIONS})
        assert st == 400 and "strands-decider: no" in out["detail"]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("/srv/private/path: boom"),
        # A ValueError from inside vLLM or the plugin's processor is a server fault, not the client's.
        ValueError("strands-decider: sides are multiples of 48, /srv/private/path"),
        KeyError("internal"),
    ],
)
def test_any_other_failure_is_a_generic_500_and_the_server_keeps_serving(client, error):
    c, engine = client(engine_kwargs={"refuse": error})
    with c:
        r = c.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})
        assert r.status_code == 500 and r.json()["detail"] == "internal server error"
        assert set(r.json()) == {"detail", "error_id"} and "private" not in r.text
        engine.refuse = None
        assert post(c, {"state": "x", "questions": QUESTIONS})[0] == 200


def test_an_expansion_that_differs_from_the_render_is_500_never_an_answer(client):
    c, _ = client(engine_kwargs={"wrong_n": 5})
    with c:
        st, out = post(c, {"state": "x", "images": [b64(png(48, 48))], "questions": QUESTIONS})
        assert st == 500 and out["detail"] == "internal server error" and out["error_id"]


@pytest.mark.parametrize(
    "raw, match",
    [
        ("[]", "must be a JSON object, not list"),
        ('"x"', "must be a JSON object, not str"),
        ("null", "must be a JSON object, not NoneType"),
        ("{", "the body is not JSON"),
        ("[" * 100000 + "]" * 100000, "the body is not JSON"),
    ],
)
def test_a_body_that_is_not_a_json_object_is_400(client, raw, match):
    c, engine = client()
    with c:
        r = c.post("/v1/systemone", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 400 and match in r.json()["detail"], r.text
        assert engine.calls == []


def test_a_body_over_the_limit_is_413_whether_or_not_it_declares_its_length(client):
    c, engine = client(max_body=2000)
    with c:
        big = json.dumps({"state": "x" * 3000, "questions": QUESTIONS})
        r = c.post("/v1/systemone", content=big, headers={"content-type": "application/json"})
        assert r.status_code == 413 and "at most 2000" in r.json()["detail"]
        chunks = (big[i : i + 500].encode() for i in range(0, len(big), 500))  # no Content-Length
        r = c.post("/v1/systemone", content=chunks, headers={"content-type": "application/json"})
        assert r.status_code == 413 and "over 2000" in r.json()["detail"]
        assert engine.calls == []
        assert post(c, {"state": "x", "questions": QUESTIONS})[0] == 200
        assert c.get("/health").json()["max_body_bytes"] == 2000


def test_the_default_body_limit_holds_four_images_and_the_text():
    four = server.MAX_IMAGES * server.base64_size(server.IMAGE_FILE_BYTES)
    assert server.default_max_body(True) == server.TEXT_BYTES + four
    assert round(server.default_max_body(True) / 2**20) == 187
    assert server.default_max_body(False) == server.TEXT_BYTES


def test_ping_answers_while_an_image_is_decoded(checkpoint, monkeypatch):
    import threading
    import time

    import httpx

    d, _, _ = decider(checkpoint)
    entered, release = threading.Event(), threading.Event()
    real = server.prepare_image

    def slow(*args):
        entered.set()
        assert release.wait(20), "the test released the decode"
        return real(*args)

    monkeypatch.setattr(server, "prepare_image", slow)
    app = server.build_app(None)
    app.state.decider = d
    body = {"state": "x", "images": [b64(png(640, 480))], "questions": QUESTIONS}

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            answer = asyncio.ensure_future(c.post("/v1/systemone", json=body))
            assert await asyncio.get_running_loop().run_in_executor(None, entered.wait, 20)
            t = time.perf_counter()
            ping = await asyncio.wait_for(c.get("/ping"), 10)
            waited = time.perf_counter() - t
            decoding = not release.is_set()
            release.set()
            return ping, waited, decoding, await answer

    ping, waited, decoding, answer = asyncio.run(run())
    assert ping.status_code == 200 and decoding and waited < 5, waited
    assert answer.status_code == 200
    d.close()


def test_a_failing_question_cancels_the_others(checkpoint):
    d, engine, _ = decider(checkpoint)
    original, calls, cancelled = engine.generate, [], []

    async def generate(prompt, sp, rid):
        calls.append(rid)
        if len(calls) == 2:
            raise RuntimeError("boom")
        if len(calls) == 3:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(rid)
                raise
        async for out in original(prompt, sp, rid):
            yield out

    engine.generate = generate
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(d.decide({"state": "x", "questions": QUESTIONS}))
    assert cancelled == [calls[2]], "the third question's engine request was cancelled"
    d.close()


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
        max_length=None,
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
    e = server.engine_args(ckpt, args(), 8193)
    assert e.hf_overrides == {
        "architectures": [ARCH_MULTIMODAL],
        "text_config": {"use_bidirectional_attention": None},
    }
    assert e.limit_mm_per_prompt == {"image": image_limit, "video": 0, "audio": 0}
    assert (e.max_model_len, e.max_num_batched_tokens, e.enable_chunked_prefill) == (
        8193,
        8193,
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
        server.engine_args(ckpt, args(max_num_batched_tokens=16384), 4097).max_num_batched_tokens
        == 16384
    )


@pytest.mark.parametrize(
    "over, expected",
    [
        ({}, (8192, 8193)),
        ({"max_length": 4096}, (4096, 4097)),
        ({"max_model_len": 16384}, (8192, 16384)),
        ({"max_length": 1000, "max_model_len": 1001}, (1000, 1001)),
    ],
)
def test_the_context_leaves_room_for_the_answer_token(over, expected):
    # vLLM 0.33 refuses a prompt of exactly max_model_len tokens (tests/gpu/engine_checks.py checks it on a real
    # engine), so the default context is one token more than the longest prompt served.
    assert server.lengths(args(**over), 8192) == expected


@pytest.mark.parametrize(
    "over", [{"max_model_len": 8192}, {"max_length": 4096, "max_model_len": 4096}]
)
def test_a_context_no_longer_than_the_longest_prompt_is_refused_at_startup(over):
    with pytest.raises(SystemExit, match="must be below --max-model-len"):
        server.lengths(args(**over), 8192)


def test_prepare_environment_puts_vllms_sockets_on_loopback_and_keeps_the_callers_values(
    monkeypatch,
):
    import os

    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.1")
    for k in ("GLOO_SOCKET_IFNAME", "VLLM_USE_V2_MODEL_RUNNER", "NCCL_SOCKET_IFNAME"):
        monkeypatch.delenv(k, raising=False)
    server.prepare_environment()  # whatever address the HTTP server binds
    assert os.environ["VLLM_HOST_IP"] == "10.0.0.1"
    assert os.environ["GLOO_SOCKET_IFNAME"] == os.environ["NCCL_SOCKET_IFNAME"] == "lo"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_the_cli_binds_loopback_by_default():
    from strands_decider_vllm.cli import parser

    a = parser().parse_args(["serve", "/ckpt"])
    assert a.host == "127.0.0.1" and a.enable_prefix_caching
    assert (a.max_model_len, a.max_length, a.max_body_mb) == (None, None, None)
    assert a.limit_concurrency == 128 and 1 <= a.preprocess_workers <= 4


def test_reads_stage_argument_is_found_by_name():
    assert server.stage_argument(lambda values, r, c, d, *, stage=None, inputs=None: 0) == "stage"
    assert server.stage_argument(lambda values, r, c, d, *, inputs=None: 0) == "inputs"
    assert server.require_core().stage == {server.stage_argument(core.read): "probabilities"}


def test_a_hub_checkpoint_is_read_at_one_commit(tmp_path, monkeypatch):
    commit = "0123456789abcdef0123456789abcdef01234567"
    snap = tmp_path / "snapshots" / commit
    snap.mkdir(parents=True)
    for n in server.Checkpoint.NAMES:
        (snap / n).write_text("{}")
    calls = []

    def download(repo, name, revision=None):
        calls.append((name, revision))
        return str(snap / name)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    ck = server.Checkpoint.open("org/model", "main")
    assert ck.revision == commit, "the engine is given the commit, not the branch"
    assert calls[0] == ("config.json", "main")
    assert {r for _, r in calls[1:]} == {commit} and len(calls) == len(server.Checkpoint.NAMES)
