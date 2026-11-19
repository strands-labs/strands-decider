"""The plugin's server against a stand-in for vLLM's engine: the prompts it sends, the answers, the
HTTP refusals and the limits on bodies and threads. Needs the core, Pillow, FastAPI and vLLM."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import threading
from types import SimpleNamespace

import pytest

core = pytest.importorskip("strands_decider.core")
PIL = pytest.importorskip("PIL.Image")
pytest.importorskip("vllm")
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from vllm.exceptions import VLLMValidationError  # noqa: E402

from strands_decider_vllm import SERVER_CONFIG_KEY, server  # noqa: E402
from strands_decider_vllm.cli import parser  # noqa: E402

OPEN, SOFT, CLOSE = (core.CONTROL_TOKEN_IDS[t] for t in ("<|image>", "<|image|>", "<image|>"))
QUESTIONS = {
    "color": {"type": "choice", "instructions": "Which?", "criteria": {"red": None, "blue": None}},
    "square": {"type": "noul", "instructions": "Is it a square?"},
    "size": {"type": "score", "instructions": "How large?", "criteria": ["small", "large"]},
}


def toy_tokenizer():
    """Words at made-up ids, with Gemma 4's control tokens and codes at Gemma 4's, as the core checks."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

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
    def make(inputs=("text", "image")):
        toy_tokenizer().save(str(tmp_path / "tokenizer.json"))
        dc = {
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
        }
        files = {
            "decision_config.json": dc,
            "config.json": {"architectures": ["Gemma4ForConditionalGeneration"], "text_config": {}},
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


def settings(path, *flags):
    return server.settings(
        parser().parse_args(["serve", path, "--preprocess-workers", "2", *flags])
    )


class FakeEngine:
    """vLLM's AsyncLLM as the server uses it: it expands each <|image|> to one placeholder per 48 x 48
    pixels (or ``wrong_n``), as the plugin's processor does, and gives the allowed ids logprobs."""

    def __init__(self, wrong_n=None, fail=None):
        self.wrong_n, self.fail, self.healthy, self.calls = wrong_n, fail, True, []

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
        yield SimpleNamespace(
            prompt_token_ids=expanded,
            outputs=[SimpleNamespace(logprobs=[logprobs])],
            num_cached_tokens=7,
        )

    async def check_health(self):
        if not self.healthy:
            raise RuntimeError("EngineDeadError")

    def shutdown(self):
        self.shut = True


def png(w, h, color=(200, 30, 30, 255)):
    buf = io.BytesIO()
    PIL.new("RGBA", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


def raw(body):
    return json.dumps(body).encode()


@pytest.fixture
def decider(checkpoint):
    made = []

    def make(*flags, inputs=("text", "image"), **engine):
        d = server.Decider(FakeEngine(**engine), settings(checkpoint(inputs), *flags))
        made.append(d)
        return d

    yield make
    for d in made:
        d.close()


# --- the engine requests and the answer -----------------------------------------------------------


def test_a_text_request_is_one_engine_request_per_question(decider):
    d = decider()
    body = {"state": "A ticket.", "questions": QUESTIONS}
    out = asyncio.run(d.decide(raw(body)))
    rendered = core.render(core.Request.model_validate(body), d.s.tokenizer)
    assert out["model"] == "strands-decider" and out["answers"]["color"]["choice"] == "red"
    assert out["answers"]["square"]["noul"] == pytest.approx(
        math.exp(-1) / (1 + math.exp(-1)), abs=1e-4
    )
    assert out["usage"] == {
        "input_tokens": rendered.input_tokens,
        "output_tokens": 3,
        "cached_tokens": 21,
    }
    for (prompt, sp), q in zip(d.engine.calls, rendered.questions.values(), strict=True):
        assert prompt == {"prompt_token_ids": list(q.ids)}
        assert (sp.max_tokens, sp.temperature, sp.top_k, sp.logprobs) == (1, 1.0, 0, len(q.codes))
        assert sp.allowed_token_ids == list(q.code_token_ids) and sp.extra_args == {
            "prefix_end": q.state_len
        }


def test_an_image_request_sends_one_placeholder_and_the_resized_pixels_per_image(decider):
    d = decider()
    files = [png(64, 48), png(300, 40, (0, 0, 255, 255))]
    body = {"state": "Two photos.", "images": [b64(f) for f in files], "questions": QUESTIONS}
    asyncio.run(d.decide(raw(body)))
    rendered = core.render(core.Request.model_validate(body), d.s.tokenizer)
    for prompt, sp in d.engine.calls:
        ids = prompt["prompt_token_ids"]
        assert ids.count(SOFT) == 2 and OPEN not in ids
        assert [im.size for im in prompt["multi_modal_data"]["image"]] == [(912, 672), (2160, 288)]
        assert prompt["multi_modal_uuids"] == {
            "image": [hashlib.sha256(f).hexdigest() for f in files]
        }
        assert prompt["mm_processor_kwargs"] == {"do_resize": False}
        assert sp.extra_args == {"prefix_end": rendered.questions["color"].state_len}, (
            "S counts the expansion"
        )
    expected = core.resize_image(server.decode_image(files[0]), 912, 672)
    assert d.engine.calls[0][0]["multi_modal_data"]["image"][0].tobytes() == expected.tobytes()


def test_decode_image_turns_transparency_white_and_follows_exif():
    assert server.decode_image(png(4, 3, (0, 0, 0, 0))).getpixel((0, 0)) == (255, 255, 255)
    buf, exif = io.BytesIO(), PIL.Exif()
    exif[0x0112] = 6  # rotated 90 degrees
    PIL.new("RGB", (8, 4)).save(buf, format="JPEG", exif=exif)
    assert server.decode_image(buf.getvalue()).size == (4, 8)
    broken = bytearray(png(64, 64))  # a sound header, garbage pixels: only the decode can tell
    start = broken.index(b"IDAT") + 4
    broken[start : start + 16] = b"\xff" * 16
    with pytest.raises(server.Refused, match="could not be decoded"):
        server.decode_image(bytes(broken))


def test_the_first_question_runs_before_the_others_and_a_failure_cancels_the_rest(decider):
    d = decider()
    order, answered, cancelled = [], [], []
    original = d.engine.generate

    async def generate(prompt, sp, rid):
        order.append(rid)
        if len(order) == 1:
            await asyncio.sleep(0.05)
            async for out in original(prompt, sp, rid):
                yield out
            answered.append(rid)
            return
        assert answered, "the others start once the first has its answer, and its cached state"
        if len(order) == 2:
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(rid)
            raise
        yield  # an async generator, like vLLM's

    d.engine.generate = generate
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(d.decide(raw({"state": "x", "questions": QUESTIONS})))
    assert cancelled == [order[2]], "the third question's engine request was cancelled"


# --- HTTP -----------------------------------------------------------------------------------------


@pytest.fixture
def client(decider):
    def make(*flags, **kw):
        d = decider(*flags, **kw)

        async def start():
            return d

        return TestClient(server.build_app(start)), d.engine

    return make


def post(c, body):
    r = c.post("/v1/systemone", json=body)
    return r.status_code, r.json()


def test_requests_and_health_over_http(client):
    c, engine = client()
    with c:
        for path in ("/health", "/ping"):
            assert c.get(path).json()["inputs"] == ["text", "image"]
        st, out = post(c, {"state": "x", "images": [b64(png(8, 8))] * 5, "questions": QUESTIONS})
        assert st == 200 and "latency_ms" in out, (
            "any number of images: the prompt's length bounds them"
        )
        assert c.post("/invocations", json={"questions": QUESTIONS}).status_code == 200
        engine.healthy = False
        assert c.get("/ping").status_code == 503
    assert engine.shut


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"state": "x", "images": ["@@@"], "questions": QUESTIONS}, "image is not valid base64"),
        (
            {"state": "x", "images": [b64(b"GIF89a" + b"\0" * 4)], "questions": QUESTIONS},
            "could not be read",
        ),
        (
            {"state": "x", "images": None, "questions": QUESTIONS},
            "images: Input should be a valid array",
        ),
        ({"state": "x", "questions": QUESTIONS, "audio": []}, "audio input is not supported"),
        ({"state": "x", "questions": {}}, "questions: Dictionary should have at least 1 item"),
        ({"state": "word " * 9000, "questions": QUESTIONS}, "exceeds the limit of 8192 tokens"),
        ("[" * 100000 + "]" * 100000, "recursion limit exceeded"),
        ("{", "Invalid JSON"),
        ("[]", "Input should be an object"),
    ],
)
def test_a_bad_request_gets_422_and_the_server_keeps_serving(client, body, detail):
    c, engine = client()
    with c:
        content = body if isinstance(body, str) else json.dumps(body)
        r = c.post("/v1/systemone", content=content, headers={"content-type": "application/json"})
        assert r.status_code == 422 and detail in r.json()["detail"], r.text
        assert engine.calls == [] and post(c, {"state": "x", "questions": QUESTIONS})[0] == 200


def test_images_for_a_text_only_checkpoint_are_refused_before_decoding(client):
    c, engine = client(inputs=("text",))
    with c:
        st, out = post(c, {"state": "x", "images": ["not even base64"], "questions": QUESTIONS})
        assert (st, out["detail"], engine.calls) == (
            422,
            "this checkpoint does not take images",
            [],
        )


@pytest.mark.parametrize(
    "engine, status",
    [
        ({"fail": VLLMValidationError("strands-decider: no", parameter="images")}, 422),
        ({"fail": RuntimeError("/srv/private/path: boom")}, 500),
        (
            {"fail": ValueError("strands-decider: /srv/private/path")},
            500,
        ),  # vLLM's, not the client's
        ({"wrong_n": 5}, 500),  # an expansion that differs from the rendering: never an answer
    ],
)
def test_an_engine_failure_and_the_server_keeps_serving(client, engine, status):
    c, fake = client(**engine)
    with c:
        body = {"state": "x", "images": [b64(png(48, 48))], "questions": QUESTIONS}
        r = c.post("/v1/systemone", json=body)
        assert r.status_code == status and "private" not in r.text
        if status == 500:  # only an error id
            assert set(r.json()) == {"detail", "error_id"}
        else:
            assert "strands-decider: no" in r.json()["detail"]
        fake.fail = fake.wrong_n = None
        assert post(c, {"state": "x", "questions": QUESTIONS})[0] == 200


def test_a_body_over_the_limit_is_413_with_or_without_its_length(client):
    c, engine = client("--max-body-mb", str(2048 / 2**20))
    big = json.dumps({"state": "x" * 3000, "questions": QUESTIONS})
    with c:
        for content in (big, (big[i : i + 500].encode() for i in range(0, len(big), 500))):
            r = c.post(
                "/v1/systemone", content=content, headers={"content-type": "application/json"}
            )
            assert r.status_code == 413 and "over 2048 bytes" in r.json()["detail"]
        assert engine.calls == [] and c.get("/health").json()["max_body_bytes"] == 2048


# --- the limits on buffered bodies and threads ----------------------------------------------------


def test_the_byte_budget_grants_in_arrival_order_and_a_cancelled_or_timed_out_wait_takes_nothing():
    async def run():
        b = server.ByteBudget(100)
        assert await b.take(60) == 60
        big = asyncio.ensure_future(b.take(80))
        await asyncio.sleep(0)
        small = asyncio.ensure_future(b.take(10))  # it would fit, but waits behind the larger one
        await asyncio.sleep(0)
        assert not small.done()
        b.give(60)
        assert await asyncio.wait_for(asyncio.gather(big, small), 5) == [80, 10]
        waiting = asyncio.ensure_future(b.take(50))
        await asyncio.sleep(0)
        waiting.cancel()
        b.give(90)  # drops the cancelled waiter before its task wakes
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await b.take(100)
        with pytest.raises(TimeoutError):
            await b.take(1, timeout=0.05)
        assert b.used == 100 and not b._waiters

    asyncio.run(run())


async def asgi_post(app, chunks, *, content_length=None, then="end"):
    """POST /v1/systemone straight to the ASGI app: ``chunks``, then the body's end, nothing ever
    again ("stall") or a disconnect. (status, body, the budget in use at each receive)."""
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/systemone",
        "raw_path": b"/v1/systemone",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    queue, sent, used = list(chunks), [], []

    async def receive():
        used.append(app.state.decider.budget.used)
        if queue:
            return {
                "type": "http.request",
                "body": queue.pop(0),
                "more_body": bool(queue) or then != "end",
            }
        if then == "disconnect":
            return {"type": "http.disconnect"}
        if then == "stall":
            await asyncio.Event().wait()
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent), used


@pytest.fixture
def budget_app(decider):
    def make(timeout="30"):
        d = decider(
            *("--max-body-mb", "2.5", "--max-buffered-mb", "3", "--body-read-timeout", timeout),
            *("--max-length", "10000000"),
        )
        app = server.build_app(None)
        app.state.decider = d
        return d, app

    return make


def large_body(n=2 * 2**20):
    return json.dumps({"state": "x" * n, "questions": QUESTIONS}).encode()


def test_stalled_or_vanished_clients_give_their_reservations_back(budget_app):
    # Two clients declare large bodies and send nothing more; a third leaves mid-body. Without the
    # timeouts they would hold the budget from every later large body.
    d, app = budget_app(timeout="0.3")

    async def run():
        declared = int(2.5 * 2**20)
        stalled = await asyncio.gather(
            asgi_post(app, [], content_length=declared, then="stall"),
            asgi_post(app, [], content_length=declared, then="stall"),
        )
        gone = await asgi_post(
            app, [b'{"state": "' + b"x" * 2**20], content_length=declared, then="disconnect"
        )
        body = large_body()
        return stalled, gone, await asgi_post(app, [body], content_length=len(body))

    stalled, (gone_status, gone_body, used), later = asyncio.run(run())
    assert sorted(st for st, _, _ in stalled)[0] == 408 and {st for st, _, _ in stalled} <= {
        408,
        503,
    }
    assert gone_status == 400 and b"disconnected" in gone_body and max(used) == int(2.5 * 2**20)
    assert later[0] == 200 and d.budget.used == 0


def test_large_bodies_share_the_budget_and_one_without_a_length_gives_back_what_it_doesnt_use(
    budget_app,
):
    d, app = budget_app()
    seen = []
    render = d.render
    d.render = lambda body: seen.append(d.budget.used) or render(body)
    body = large_body()

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            rs = await asyncio.gather(*(c.post("/v1/systemone", content=body) for _ in range(3)))
        chunked = await asgi_post(app, [body[i : i + 2**19] for i in range(0, len(body), 2**19)])
        return rs, chunked

    rs, (status, _, used) = asyncio.run(run())
    assert [r.status_code for r in rs] == [200] * 3 and 0 < max(seen[:3]) <= 3 * 2**20
    assert status == 200 and used[:3] == [0, 0, 0], "the first MiB is read without a reservation"
    assert int(2.5 * 2**20) in used and seen[3] == len(body), "then the limit, until it's read"
    assert d.budget.used == 0


def test_a_text_request_and_ping_never_wait_behind_image_decodes(decider, monkeypatch):
    d = decider()  # --preprocess-workers 2
    entered, release = threading.Semaphore(0), threading.Event()
    real = server.prepare_image

    def slow(image):
        entered.release()
        assert release.wait(20)
        return real(image)

    monkeypatch.setattr(server, "prepare_image", slow)
    app = server.build_app(None)
    app.state.decider = d
    image = {"state": "x", "images": [b64(png(640, 480))], "questions": QUESTIONS}

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            decodes = [asyncio.ensure_future(c.post("/v1/systemone", json=image)) for _ in range(4)]
            for _ in range(2):  # both decode threads are busy
                assert await asyncio.get_running_loop().run_in_executor(
                    None, entered.acquire, True, 20
                )
            ping = await asyncio.wait_for(c.get("/ping"), 5)
            text = await asyncio.wait_for(
                c.post("/v1/systemone", json={"questions": QUESTIONS}), 10
            )
            blocked = not release.is_set()
            release.set()
            return ping, text, blocked, await asyncio.gather(*decodes)

    ping, text, blocked, decodes = asyncio.run(run())
    assert ping.status_code == text.status_code == 200 and blocked
    assert [r.status_code for r in decodes] == [200] * 4


# --- startup --------------------------------------------------------------------------------------


@pytest.mark.parametrize("inputs, images", [(("text", "image"), 8193), (("text",), 0)])
def test_the_engine_gets_the_plugins_flags(checkpoint, inputs, images):
    e = settings(checkpoint(inputs)).engine_args
    assert e.hf_overrides == {
        "architectures": ["StrandsDeciderForConditionalGeneration"],
        "text_config": {"use_bidirectional_attention": None},
    }
    assert e.limit_mm_per_prompt == {"image": images, "video": 0, "audio": 0}
    assert (e.max_model_len, e.max_num_batched_tokens, e.enable_chunked_prefill) == (
        8193,
        8193,
        False,
    )
    assert (e.logprobs_mode, e.max_logprobs, e.generation_config) == (
        "processed_logprobs",
        255,
        "vllm",
    )
    assert e.prefix_caching_hash_algo == "sha256" and e.prefix_cache_retention_interval is None
    assert e.additional_config == {SERVER_CONFIG_KEY: server.__version__}


@pytest.mark.parametrize(
    "flags, lengths",
    [
        ((), (8192, 8193)),
        (("--max-length", "4096"), (4096, 4097)),
        (("--max-model-len", "16384"), (8192, 16384)),
    ],
)
def test_the_context_leaves_a_position_for_the_answer(checkpoint, flags, lengths):
    # vLLM refuses a prompt that fills the context (tests/gpu/engine_checks.py).
    s = settings(checkpoint(), *flags)
    assert (s.max_length, s.engine_args.max_model_len) == lengths
    with pytest.raises(SystemExit, match="must be below --max-model-len"):
        settings(checkpoint(), "--max-model-len", "8192")


def test_a_hub_checkpoint_is_read_at_one_commit(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / ("0123456789" * 4)
    snapshot.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: calls.append(k) or str(snapshot)
    )
    # The engine is given the commit the config files were read at, not the branch.
    assert server.checkpoint_dir("org/model", "main") == (snapshot, "0123456789" * 4)
    assert calls[0]["revision"] == "main"
    assert set(calls[0]["allow_patterns"]) == set(server.CONFIG_FILES)


def test_the_server_binds_loopback_and_vllms_sockets_stay_there(monkeypatch):
    import os

    from strands_decider_vllm import cli

    a = parser().parse_args(["serve", "/ckpt"])
    assert a.host == "127.0.0.1" and a.limit_concurrency == 128
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.1")  # the caller's value is kept
    for key in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "VLLM_USE_V2_MODEL_RUNNER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(server, "serve", lambda a: None)
    cli.main(["serve", "/ckpt"])
    assert os.environ["VLLM_HOST_IP"] == "10.0.0.1" and os.environ["NCCL_SOCKET_IFNAME"] == "lo"
    assert (
        os.environ["GLOO_SOCKET_IFNAME"] == "lo" and os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    )
