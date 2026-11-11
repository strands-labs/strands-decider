"""The AWS Lambda handler (deploy/lambda/function/handler.py), over a stand-in backend.

The backend is the one part replaced: llama.cpp and the image are deploy/lambda's to test.
Everything else, the routes, the status codes and bodies, the settings, the load and the
model prefetch, is the handler's own. The engine under it is tests/test_hidden_engine.py's.
"""

from __future__ import annotations

import base64
import importlib
import io
import itertools
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

from strands_decider.hidden_engine import HiddenStateEngine

FUNCTION = Path(__file__).resolve().parents[1] / "deploy" / "lambda" / "function"
HIDDEN, POINTER = 8, 4
REQUEST = {
    "state": "My renewal failed and I leave on Friday!",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"circulation": "renewals", "billing": "fines"},
        },
    },
}


class FakeBackend:
    """Hidden states drawn from the request's shape, so the same request gets the same ones."""

    calls: ClassVar[list[tuple[list[int], list[list[int]], list[list[int]]]]] = []

    def __init__(self, lib_path: str, model_path: str, threads: int):
        self.args = (lib_path, model_path, threads)

    def forward(self, state, questions, rows):
        FakeBackend.calls.append((state, questions, rows))
        rng = np.random.default_rng(len(state))
        return [
            rng.normal(size=(len(r), HIDDEN)) + len(q) for q, r in zip(questions, rows, strict=True)
        ]


class FakeVisionBackend(FakeBackend):
    """FakeBackend that also reads images, sizing each as the engine does."""

    images: ClassVar[list[list[tuple[int, int]]]] = []

    def __init__(self, lib_path: str, model_path: str, mmproj_path: str, threads: int):
        super().__init__(lib_path, model_path, threads)
        self.mmproj_path = mmproj_path

    def forward_images(self, segments, images, questions, rows):
        FakeVisionBackend.images.append([(w, h) for _, w, h in images])
        state = [t for segment in segments for t in segment]
        return self.forward(state, questions, rows), [w // 32 * (h // 32) + 2 for _, w, h in images]


@pytest.fixture
def model_dir(tmp_path):
    """What the image holds in /opt/model: the GGUF (any bytes here) and the assets."""
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "decider-q8_0.gguf").write_bytes(os.urandom(1000))
    tok = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(assets / "tokenizer.json"))
    rng = np.random.default_rng(0)
    head = {
        "norm.weight": rng.normal(size=HIDDEN),
        "norm.bias": rng.normal(size=HIDDEN),
        "q.weight": rng.normal(size=(POINTER, HIDDEN)),
        "q.bias": rng.normal(size=POINTER),
        "k.weight": rng.normal(size=(POINTER, HIDDEN)),
        "k.bias": rng.normal(size=POINTER),
    }
    save_file({k: v.astype(np.float32) for k, v in head.items()}, str(assets / "head.safetensors"))
    config = {
        "model_name": "tiny-q8_0",
        "pointer_dim": POINTER,
        "max_length": 512,
        "temperature": 1.0,
        "temperature_by_kind": {},
        "ordinal_smoothing": 0.0,
    }
    (assets / "head_config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def load(model_dir, monkeypatch):
    """Import the handler afresh, as a new execution environment does, with `env` set."""

    def _load(**env: str):
        monkeypatch.setenv("DECIDER_MODEL_DIR", str(model_dir))
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE", raising=False)
        monkeypatch.delenv("DECIDER_MAX_REQUEST_TOKENS", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setitem(
            sys.modules,
            "backend",
            types.SimpleNamespace(LlamaBackend=FakeBackend, LlamaVisionBackend=FakeVisionBackend),
        )
        monkeypatch.syspath_prepend(str(FUNCTION))
        monkeypatch.delitem(sys.modules, "handler", raising=False)
        FakeBackend.calls.clear()
        FakeVisionBackend.images.clear()
        return importlib.import_module("handler")

    yield _load
    sys.modules.pop("handler", None)


def url_event(method: str, path: str, body=None, *, raw: str | None = None, b64: bool = False):
    """A Function URL event, as Lambda passes it."""
    text = raw if raw is not None else (json.dumps(body) if body is not None else None)
    if b64 and text is not None:
        text = base64.b64encode(text.encode()).decode()
    return {
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
        "body": text,
        "isBase64Encoded": b64,
    }


def call(handler, event):
    response = handler.handler(event, None)
    assert response["headers"] == {"content-type": "application/json"}
    return response["statusCode"], json.loads(response["body"])


def test_loads_the_model_from_the_image(load, model_dir):
    handler = load()
    backend = handler._engine.result().backend
    lib_path, model_path, threads = backend.args
    assert lib_path == str(FUNCTION / "libdecider.so")
    assert model_path == str(model_dir / "decider-q8_0.gguf")
    assert threads == os.cpu_count()
    assert len(FakeBackend.calls) == 1  # the warm-up, before any request


def test_health_reports_the_model_and_settings(load):
    assert call(load(), url_event("GET", "/health")) == (
        200,
        {
            "status": "ok",
            "model": "tiny-q8_0",
            "max_length": 512,
            "temperature": 1.0,
            "max_request_tokens": None,
            "vision": False,
        },
    )
    handler = load(DECIDER_MAX_REQUEST_TOKENS="24576")
    _, body = call(handler, url_event("GET", "/health"))
    assert body["max_request_tokens"] == 24576


@pytest.mark.parametrize("b64", [False, True], ids=["text", "base64"])
def test_answers_as_the_engine_does(load, model_dir, b64):
    handler = load()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST, b64=b64))
    engine = HiddenStateEngine.from_assets(str(model_dir / "assets"), FakeBackend("", "", 1))
    expected = engine.evaluate(REQUEST).model_dump()
    assert status == 200
    assert body.pop("latency_ms") > 0
    assert body == expected


def test_direct_invocation_returns_the_response(load):
    handler = load()
    response = handler.handler(REQUEST, None)
    assert set(response) == {"model", "answers", "usage", "latency_ms"}
    assert set(response["answers"]) == {"urgent", "team"}
    with pytest.raises(ValidationError):
        handler.handler({"state": "x", "questions": {}}, None)


def test_unknown_routes(load):
    handler = load()
    assert call(handler, url_event("GET", "/nope")) == (404, {"detail": "Not Found"})
    assert call(handler, url_event("GET", "/v1/systemone")) == (
        405,
        {"detail": "Method Not Allowed"},
    )
    assert call(handler, url_event("POST", "/health")) == (405, {"detail": "Method Not Allowed"})


def test_invalid_requests_get_fastapis_422(load):
    handler = load()
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {"state": "x", "questions": {}})
    )
    assert status == 422
    assert [(e["type"], e["loc"]) for e in body["detail"]] == [("too_short", ["body", "questions"])]
    status, body = call(handler, url_event("POST", "/v1/systemone", raw="{nope"))
    assert status == 422
    assert [(e["type"], e["loc"]) for e in body["detail"]] == [("json_invalid", ["body"])]


def test_requests_the_engine_refuses_get_422_before_the_forward(load):
    handler = load(DECIDER_MAX_REQUEST_TOKENS="10")
    FakeBackend.calls.clear()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 422 and "exceeds the limit of 10 tokens per request" in body["detail"]
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": ["aGk="]})
    )
    assert status == 422 and "images are not supported" in body["detail"]
    assert not FakeBackend.calls


def test_on_demand_init_loads_in_the_background(load):
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 200 and set(body["answers"]) == {"urgent", "team"}


def test_a_failed_background_load_exits(load, model_dir, monkeypatch):
    (model_dir / "assets" / "head_config.json").unlink()
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    with pytest.raises(FileNotFoundError):
        handler._engine.result()
    for _ in range(100):  # the done-callback runs just after result() returns
        if exits:
            break
        time.sleep(0.01)
    assert exits == [1]


def test_a_negative_request_limit_fails_the_init_before_the_load(load):
    with pytest.raises(ValueError, match="must be 0 \\(no limit\\) or more, not -5"):
        load(DECIDER_MAX_REQUEST_TOKENS="-5")
    assert not FakeBackend.calls  # refused before the model loaded


def test_a_failed_load_at_import_fails_the_init(load, model_dir):
    (model_dir / "assets" / "head_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        load()


def test_prefetch_reads_every_byte_once(load, model_dir, monkeypatch):
    handler = load()
    read: list[tuple[int, int]] = []
    pread = os.pread

    def recording_pread(fd, n, offset):
        data = pread(fd, n, offset)
        read.append((offset, len(data)))
        return data

    monkeypatch.setattr(os, "pread", recording_pread)
    handler._prefetch(str(model_dir / "decider-q8_0.gguf"), chunk=64, readers=4)
    covered = sorted(read)
    assert covered[0][0] == 0
    assert all(a + n == b for (a, n), (b, _) in itertools.pairwise(covered))
    assert sum(n for _, n in covered) == 1000


def _png(width: int, height: int) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_an_image_with_a_vision_tower_answers_over_images(load, model_dir):
    pytest.importorskip("PIL")
    (model_dir / "mmproj-f16.gguf").write_bytes(os.urandom(500))
    handler = load()
    backend = handler._engine.result().backend
    assert isinstance(backend, FakeVisionBackend)
    assert backend.mmproj_path == str(model_dir / "mmproj-f16.gguf")
    # The image warm-up: 64 px, raised to the processor grid's least area, 256 x 256.
    assert FakeVisionBackend.images == [[(256, 256)]]
    assert call(handler, url_event("GET", "/health"))[1]["vision"] is True
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": [_png(600, 400)]})
    )
    assert status == 200 and set(body["answers"]) == {"urgent", "team"}
    assert FakeVisionBackend.images[-1] == [(448, 288)]  # 448 x 299, on the processor's grid


def test_an_image_without_a_vision_tower_refuses_images(load):
    pytest.importorskip("PIL")
    handler = load()
    assert not isinstance(handler._engine.result().backend, FakeVisionBackend)
    status, body = call(
        handler, url_event("POST", "/v1/systemone", {**REQUEST, "images": [_png(64, 64)]})
    )
    assert status == 422 and "images are not supported" in body["detail"]
    assert not FakeVisionBackend.images


def test_a_fault_in_the_function_is_a_500_with_its_traceback_logged(load, monkeypatch, caplog):
    handler = load()

    def broken_forward(self, state, questions, rows):
        raise RuntimeError("dh_forward failed with 4")

    monkeypatch.setattr(FakeBackend, "forward", broken_forward)
    with caplog.at_level("ERROR", logger="decider"):
        status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert (status, body) == (500, {"detail": "Internal Server Error"})
    assert "dh_forward failed with 4" in caplog.text  # the traceback, in the function's log
    with pytest.raises(RuntimeError, match="dh_forward failed"):
        handler.handler(REQUEST, None)  # a direct invocation fails, as Lambda reports it
