"""The Lambda handler for a model checkpoint with a decision_config.json, over a stand-in for the
llama.cpp backend, which tests/test_backend_llama_cpp.py and tests/test_llama_cpp_shim.py test.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
from core_support import decision_config_fixture, image_b64

from strands_decider.core import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS

DECISION = Path(__file__).resolve().parents[1] / "deploy" / "lambda" / "decision"
HIDDEN_SIZE = 8
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
IMAGE_REQUEST = {**REQUEST, "images": [image_b64("checker-64x48.png")]}


class FakeBackend:
    """LlamaCppBackend's stand-in: hidden states that make the first question's first code and the
    others' second code win, or `fail` raised."""

    def __init__(self, library, model, *, mmproj=None, threads=None):
        self.args = (library, model, mmproj)
        self.calls: list[tuple] = []
        self.fail: Exception | None = None
        self.last_timings = {"state_ms": 1.0}
        self.cpu_backend = "static"

    def forward(self, state, questions, images=()):
        if self.fail is not None:
            raise self.fail
        self.calls.append((state, questions, tuple(images)))
        h = np.zeros((len(questions), HIDDEN_SIZE), dtype=np.float32)
        h[0, 0] = h[1:, 1] = 40.0
        return h


def gemma_ids_tokenizer():
    """A word-level tokenizer with Gemma 4's control tokens and every code at Gemma 4's ids, so that
    it passes `prepare_tokenizer`; other text is a few small ids."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    vocab = {w: i + 10 for i, w in enumerate(words)}
    vocab.update(zip(CODES, CODE_TOKEN_IDS, strict=True))
    vocab.update(CONTROL_TOKEN_IDS)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    tok.add_special_tokens([AddedToken(t, special=True) for t in CONTROL_TOKEN_IDS])
    return tok


def write_checkpoint(
    d: Path, *, inputs=("text", "image"), lazy_ranges=(), max_request_tokens=None
) -> Path:
    """What the image holds in /opt/model, with a readout of identity rows."""
    from safetensors.numpy import save_file

    d.mkdir(parents=True, exist_ok=True)
    gemma_ids_tokenizer().save(str(d / "tokenizer.json"))
    config = {**decision_config_fixture(), "inputs": list(inputs), "max_length_validated": 512}
    (d / "decision_config.json").write_text(json.dumps(config))
    weight = np.zeros((len(CODES), HIDDEN_SIZE), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = 1.0
    save_file({"weight": weight}, str(d / "readout.safetensors"))
    (d / "model.gguf").write_bytes(bytes(range(256)) * 4)
    if "image" in inputs:
        (d / "mmproj.gguf").write_bytes(b"\0" * 100)
    artifact = {"model_name": "tiny-decision-q8_0", "lazy_ranges": list(lazy_ranges)}
    if max_request_tokens is not None:
        artifact["max_request_tokens"] = max_request_tokens
    (d / "artifact.json").write_text(json.dumps(artifact))
    return d


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Import the handler afresh, as a new execution environment does, from `directory` with `env`
    set. The backends it makes are in `load.backends`; posix_fadvise's calls, which macOS doesn't
    have, in `load.dropped`."""
    backends: list[FakeBackend] = []
    dropped: list[tuple[int, int]] = []
    fadvise = lambda fd, offset, length, advice: dropped.append((offset, length))  # noqa: E731
    monkeypatch.setattr(os, "posix_fadvise", fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)

    def backend(*args, **kwargs):
        backends.append(FakeBackend(*args, **kwargs))
        return backends[-1]

    fake = types.ModuleType("strands_decider.backends.llama_cpp")
    fake.LlamaCppBackend = backend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strands_decider.backends.llama_cpp", fake)
    monkeypatch.syspath_prepend(str(DECISION))

    def load(directory: Path | None = None, **env: str):
        directory = directory or write_checkpoint(tmp_path / "model")
        monkeypatch.setenv("DECIDER_MODEL_DIR", str(directory))
        for key in ("AWS_LAMBDA_INITIALIZATION_TYPE", "DECIDER_MAX_REQUEST_TOKENS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delitem(sys.modules, "handler", raising=False)
        dropped.clear()
        return importlib.import_module("handler")

    load.backends = backends  # type: ignore[attr-defined]
    load.dropped = dropped  # type: ignore[attr-defined]
    yield load
    sys.modules.pop("handler", None)


def url_event(method: str, path: str, body=None, *, raw: str | None = None, b64: bool = False):
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
    assert response["headers"]["content-type"] == "application/json"
    return response["statusCode"], json.loads(response["body"])


@pytest.mark.parametrize(
    ("inputs", "mmproj", "warm_ups", "dropped"),
    [
        (["text", "image"], "mmproj.gguf", [0, 1], [(100, 800), (0, 0), (100, 800)]),
        (["text"], None, [0], [(100, 800), (100, 800)]),
    ],
)
def test_the_load_warms_up_then_drops_the_pages_it_is_done_with(
    load, tmp_path, inputs, mmproj, warm_ups, dropped
):
    d = write_checkpoint(tmp_path / "ckpt", inputs=inputs, lazy_ranges=[[100, 900]])
    handler = load(d)
    backend = load.backends[-1]
    assert backend.args == (
        str(DECISION / "libdecision.so"), str(d / "model.gguf"), mmproj and str(d / mmproj)
    )  # fmt: skip
    # A text warm-up, then for a checkpoint that takes images an image one.
    assert [len(images) for _, _, images in backend.calls] == warm_ups
    # The table's range right after the prefetch, before llama.cpp loads; the vision tower's file,
    # whole, once mtmd has read it; the table's range again after the warm-ups.
    assert load.dropped == dropped
    assert call(handler, url_event("GET", "/health"))[1]["inputs"] == inputs


def test_answers(load):
    handler = load()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 200 and body["model"] == "tiny-decision-q8_0" and body["latency_ms"] >= 0
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.0}  # code A, "false", wins
    assert body["answers"]["team"]["choice"] == "billing"  # code B
    state, questions, _ = load.backends[-1].calls[-1]
    assert body["usage"] == {
        "input_tokens": len(state) + sum(map(len, questions)),
        "output_tokens": 2,
    }
    # A base64-encoded body, as a Function URL may send one, with an image; a direct invocation.
    status, with_image = call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST, b64=True))
    assert status == 200 and with_image["answers"] == body["answers"]
    assert len(load.backends[-1].calls[-1][2]) == 1
    assert handler.handler(REQUEST, None)["answers"] == body["answers"]


@pytest.mark.parametrize(
    ("event", "env", "status", "detail"),
    [
        (url_event("GET", "/nope"), {}, 404, "Not Found"),
        (url_event("GET", "/v1/systemone"), {}, 405, "Method Not Allowed"),
        (url_event("POST", "/health"), {}, 405, "Method Not Allowed"),
        (url_event("POST", "/v1/systemone", raw="{not json"), {}, 422, "Invalid JSON"),
        (url_event("POST", "/v1/systemone", {**REQUEST, "state": "word " * 600}), {}, 422, "exceeds the limit of 512"),
        (url_event("POST", "/v1/systemone", REQUEST), {"DECIDER_MAX_REQUEST_TOKENS": "40"}, 422,
         "exceeds the limit of 40 tokens per request"),
    ],
)  # fmt: skip
def test_refusals_come_before_the_backend_runs(load, event, env, status, detail):
    handler = load(**env)
    warm_ups = len(load.backends[-1].calls)
    got_status, body = call(handler, event)
    assert got_status == status and detail in body["detail"]
    assert len(load.backends[-1].calls) == warm_ups


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (ValueError("image could not be decoded: truncated"), 422, "image could not be decoded: truncated"),
        (RuntimeError("sd_forward: the state's decode failed"), 500, "Internal Server Error"),
    ],
)  # fmt: skip
def test_a_backend_error_is_the_requests_only_if_a_value_error(load, error, status, detail):
    handler = load()
    load.backends[-1].fail = error
    assert call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST)) == (
        status,
        {"detail": detail},
    )


@pytest.mark.parametrize(
    ("value", "limit"), [("5000", 5000), (None, 16384), ("", 16384), ("0", None)]
)
def test_health_and_the_request_limit_which_fails_closed(load, value, limit):
    handler = load(**({} if value is None else {"DECIDER_MAX_REQUEST_TOKENS": value}))
    assert call(handler, url_event("GET", "/health")) == (
        200,
        {
            "status": "ok",
            "model": "tiny-decision-q8_0",
            "inputs": ["text", "image"],
            "max_length": 512,
            "max_request_tokens": limit,
        },
    )


@pytest.mark.parametrize(("value", "limit"), [(None, 12288), ("8000", 8000), ("0", None)])
def test_the_build_s_request_limit_lowers_a_higher_one(load, tmp_path, value, limit):
    # E4B's build records 12,288: a 16,384-token request runs it out of memory.
    d = write_checkpoint(tmp_path / "ckpt", max_request_tokens=12288)
    handler = load(d, **({} if value is None else {"DECIDER_MAX_REQUEST_TOKENS": value}))
    assert call(handler, url_event("GET", "/health"))[1]["max_request_tokens"] == limit


def test_an_on_demand_load_answers_once_ready_and_exits_if_it_fails(load, tmp_path, monkeypatch):
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    assert handler.handler(url_event("POST", "/v1/systemone", REQUEST), None)["statusCode"] == 200
    exited = threading.Event()
    monkeypatch.setattr(os, "_exit", lambda code: exited.set() if code == 1 else None)
    broken = write_checkpoint(tmp_path / "broken")
    (broken / "decision_config.json").unlink()
    load(broken, AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    assert exited.wait(10)
