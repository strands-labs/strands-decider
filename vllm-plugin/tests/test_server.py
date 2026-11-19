"""The plugin's server over HTTP against a stand-in for vLLM's engine: the engine requests it sends,
the answers, the refusals, and the threads that keep image decodes from blocking other requests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import threading

import httpx
import pytest
from PIL import Image
from strands_decider import core
from vllm.exceptions import VLLMValidationError

OPEN, SOFT = core.CONTROL_TOKEN_IDS["<|image>"], core.CONTROL_TOKEN_IDS["<|image|>"]
QUESTIONS = {
    "color": {"type": "choice", "instructions": "Which?", "criteria": {"red": None, "blue": None}},
    "square": {"type": "noul", "instructions": "Is it a square?"},
    "size": {"type": "score", "instructions": "How large?", "criteria": ["small", "large"]},
}
TEXT = {"state": "A ticket.", "questions": QUESTIONS}


def png(w, h, color=(200, 30, 30, 255)):
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


def rendered(body, s):
    return core.render(core.Request.model_validate(body), s.settings.tokenizer)


def test_a_text_request_is_one_engine_request_per_question(server):
    s = server()
    with s.client as c:
        for path in ("/health", "/ping"):
            assert c.get(path).json()["inputs"] == ["text", "image"]
        out = c.post("/v1/systemone", json=TEXT).json()
        assert c.post("/invocations", json=TEXT).status_code == 200  # SageMaker's
        s.engine.healthy = False
        assert c.get("/ping").status_code == 503
    assert s.engine.shut
    assert out["model"] == "strands-decider" and out["answers"]["color"]["choice"] == "red"
    noul = math.exp(-1) / (1 + math.exp(-1))
    assert out["answers"]["square"]["noul"] == pytest.approx(noul, abs=1e-4)
    want = rendered(TEXT, s)
    assert out["usage"] == {
        "input_tokens": want.input_tokens,
        "output_tokens": 3,
        "cached_tokens": 21,
    }
    for (prompt, sp), q in zip(s.engine.calls, want.questions.values(), strict=False):
        assert prompt == {"prompt_token_ids": list(q.ids)}
        assert (sp.max_tokens, sp.temperature, sp.top_k, sp.logprobs) == (1, 1.0, 0, len(q.codes))
        assert sp.allowed_token_ids == list(q.code_token_ids)
        assert sp.extra_args == {"prefix_end": q.state_len}


def test_an_image_request_sends_one_placeholder_and_the_resized_pixels_per_image(server):
    s = server()
    files = [png(64, 48), png(300, 40, (0, 0, 255, 255))]
    body = {"state": "Two photos.", "images": [b64(f) for f in files], "questions": QUESTIONS}
    with s.client as c:
        assert c.post("/v1/systemone", json=body).status_code == 200
    state_len = rendered(body, s).questions["color"].state_len
    for prompt, sp in s.engine.calls:
        ids = prompt["prompt_token_ids"]
        assert ids.count(SOFT) == 2 and OPEN not in ids
        assert [im.size for im in prompt["multi_modal_data"]["image"]] == [(912, 672), (2160, 288)]
        assert prompt["multi_modal_uuids"] == {
            "image": [hashlib.sha256(f).hexdigest() for f in files]
        }
        assert prompt["mm_processor_kwargs"] == {"do_resize": False}
        assert sp.extra_args == {"prefix_end": state_len}, "S counts the expansion"
    first = Image.open(io.BytesIO(files[0])).convert("RGB")
    sent = s.engine.calls[0][0]["multi_modal_data"]["image"][0]
    assert sent.tobytes() == core.resize_image(first, 912, 672).tobytes()


def test_images_are_decoded_upright_with_transparency_on_white(server):
    rotated, exif = io.BytesIO(), Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees
    Image.new("RGB", (96, 48)).save(rotated, format="JPEG", exif=exif)
    s = server()
    images = [b64(png(48, 48, (0, 0, 0, 0))), b64(rotated.getvalue())]
    with s.client as c:
        r = c.post("/v1/systemone", json={"images": images, "questions": QUESTIONS})
        assert r.status_code == 200
    clear, upright = s.engine.calls[0][0]["multi_modal_data"]["image"]
    assert clear.getpixel((0, 0)) == (255, 255, 255)
    assert upright.size == core.resized_size(48, 96)


def test_the_first_question_runs_before_the_others_and_a_failure_cancels_the_rest(server):
    s = server()
    order, answered, cancelled = [], [], []
    generate = s.engine.generate

    async def first_alone_then_fail(prompt, sp, rid):
        order.append(rid)
        if len(order) == 1:
            await asyncio.sleep(0.05)
            async for out in generate(prompt, sp, rid):
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

    s.engine.generate = first_alone_then_fail
    with s.client as c:
        assert c.post("/v1/systemone", json=TEXT).status_code == 500
    assert cancelled == [order[2]], "the third question's engine request was cancelled"


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"images": ["@@@"], "questions": QUESTIONS}, "image is not valid base64"),
        ({"images": [b64(b"GIF89a" + b"\0" * 4)], "questions": QUESTIONS}, "could not be read"),
        ({"images": None, "questions": QUESTIONS}, "images: Input should be a valid array"),
        ({"questions": QUESTIONS, "audio": []}, "audio input is not supported"),
        ({"questions": {}}, "questions: Dictionary should have at least 1 item"),
        ({"state": "word " * 9000, "questions": QUESTIONS}, "exceeds the limit of 8192 tokens"),
        ("[" * 100000 + "]" * 100000, "recursion limit exceeded"),
        ("{", "Invalid JSON"),
        ("[]", "Input should be an object"),
    ],
)
def test_a_bad_request_gets_422_and_the_server_keeps_serving(server, body, detail):
    s = server()
    content = body if isinstance(body, str) else json.dumps(body)
    with s.client as c:
        r = c.post("/v1/systemone", content=content, headers={"content-type": "application/json"})
        assert r.status_code == 422 and detail in r.json()["detail"], r.text
        assert s.engine.calls == [] and c.post("/v1/systemone", json=TEXT).status_code == 200


def test_pixels_that_dont_decode_get_422(server):
    broken = bytearray(png(64, 64))  # a sound header, garbage pixels: only the decode can tell
    start = broken.index(b"IDAT") + 4
    broken[start : start + 16] = b"\xff" * 16
    with server().client as c:
        r = c.post("/v1/systemone", json={"images": [b64(bytes(broken))], "questions": QUESTIONS})
    assert r.status_code == 422 and "could not be decoded" in r.json()["detail"]


def test_images_for_a_text_only_checkpoint_are_refused_before_decoding(server):
    s = server(inputs=("text",))
    with s.client as c:
        r = c.post("/v1/systemone", json={"images": ["not even base64"], "questions": QUESTIONS})
    assert (r.status_code, r.json()["detail"]) == (422, "this checkpoint does not take images")
    assert s.engine.calls == []


@pytest.mark.parametrize(
    "engine, status",
    [
        ({"fail": VLLMValidationError("strands-decider: no", parameter="x")}, 422),
        ({"fail": RuntimeError("/srv/private/path: boom")}, 500),
        (
            {"fail": ValueError("strands-decider: /srv/private/path")},
            500,
        ),  # vLLM's, not the client's
        ({"wrong_n": 5}, 500),  # an expansion that differs from the rendering: never an answer
    ],
)
def test_an_engine_failure_and_the_server_keeps_serving(server, engine, status):
    s = server(**engine)
    with s.client as c:
        body = {"images": [b64(png(48, 48))], "questions": QUESTIONS}
        r = c.post("/v1/systemone", json=body)
        assert r.status_code == status and "private" not in r.text
        if status == 500:  # only an error id
            assert set(r.json()) == {"detail", "error_id"}
        s.engine.fail = s.engine.wrong_n = None
        assert c.post("/v1/systemone", json=TEXT).status_code == 200


def test_a_text_request_and_ping_never_wait_behind_image_decodes(server, monkeypatch):
    from strands_decider_vllm import decider

    entered, release = threading.Semaphore(0), threading.Event()
    pixels = decider._pixels

    def slow(image):
        entered.release()
        assert release.wait(20)
        return pixels(image)

    monkeypatch.setattr(decider, "_pixels", slow)
    app = server().app  # --preprocess-workers 2
    image = {"images": [b64(png(640, 480))], "questions": QUESTIONS}

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://test") as c,
        ):
            decodes = [asyncio.ensure_future(c.post("/v1/systemone", json=image)) for _ in range(4)]
            for _ in range(2):  # both decode threads are busy
                acquired = asyncio.get_running_loop().run_in_executor(
                    None, entered.acquire, True, 20
                )
                assert await acquired
            ping = await asyncio.wait_for(c.get("/ping"), 5)
            text = await asyncio.wait_for(c.post("/v1/systemone", json=TEXT), 10)
            blocked = not release.is_set()
            release.set()
            statuses = [r.status_code for r in await asyncio.gather(*decodes)]
        return ping.status_code, text.status_code, blocked, statuses

    assert asyncio.run(run()) == (200, 200, True, [200] * 4)
