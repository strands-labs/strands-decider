"""The decision checkpoint's AWS Lambda handler (deploy/lambda/decision/handler.py), over a stand-in
backend, and the image build's checks of a checkpoint (deploy/lambda/decision/scripts/build_artifact.py).

The backend is the one part replaced: llama.cpp is tests/test_backend_llama_cpp.py's and
deploy/lambda/decision's to test. Everything else is the handler's own: the load and its validation
of the checkpoint's files, the routes, the status codes and bodies, the limits and the warm-up. The
core under it is tests/test_core_*.py's. A checkpoint whose decision_config.json lacks the
soft-cap, or whose config.json names another, is refused, at load and at build.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from core_support import decision_config_fixture, image_b64, toy_tokenizer

from strands_decider.core import CODES, HIDDEN

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
    """Hidden states that make the first question's first code and the others' second code win."""

    calls: ClassVar[list[tuple[list[int], list[list[int]], tuple]]] = []
    fail: ClassVar[bool] = False
    outputs = HIDDEN

    def __init__(self, library, model, *, mmproj=None, processor_config=None, threads=None):
        self.args = (library, model, mmproj, processor_config, threads)
        self.inputs = ("text", "image") if mmproj else ("text",)
        self.last_timings: dict = {}

    def forward(self, state, questions, images=()):
        if FakeBackend.fail:
            raise RuntimeError("sd_forward failed: the state's decode failed")
        FakeBackend.calls.append((state, questions, tuple(images)))
        self.last_timings = {"state_ms": 1.0, "question_ms": [2.0] * len(questions)}
        if images:
            self.last_timings["images"] = [{"size": [912, 672], "encode_ms": 3.0}] * len(images)
        h = np.zeros((len(questions), HIDDEN_SIZE), dtype=np.float32)
        for i in range(len(questions)):
            h[i, 0 if i == 0 else 1] = 40.0
        return h


def _safetensors(path: Path, weight: np.ndarray) -> None:
    bits = (weight.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    header = json.dumps(
        {"weight": {"dtype": "BF16", "shape": list(weight.shape), "data_offsets": [0, bits.nbytes]}}
    )
    path.write_bytes(struct.pack("<Q", len(header)) + header.encode() + bits.tobytes())


def write_checkpoint(d: Path, *, vision: bool = True, **config) -> Path:
    """What the image holds in /opt/model, with a toy tokenizer and a readout of identity rows."""
    d.mkdir(parents=True, exist_ok=True)
    tok = toy_tokenizer()
    tok.save(str(d / "tokenizer.json"))
    cfg = {
        **decision_config_fixture(),
        "token_ids": [tok.token_to_id(c) for c in CODES],
        "max_length_validated": 512,
        **config,
    }
    for k in [k for k, v in config.items() if v is None]:
        del cfg[k]
    (d / "decision_config.json").write_text(json.dumps(cfg))
    weight = np.zeros((len(CODES), HIDDEN_SIZE), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = 1.0
    _safetensors(d / "readout.safetensors", weight)
    (d / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                }
            }
        )
    )
    (d / "artifact.json").write_text(json.dumps({"model_name": "tiny-decision-q8_0"}))
    (d / "model.gguf").write_bytes(b"\0" * 1000)
    if vision:
        (d / "mmproj.gguf").write_bytes(b"\0" * 100)
    return d


@pytest.fixture
def model_dir(tmp_path):
    return write_checkpoint(tmp_path / "model")


@pytest.fixture
def load(model_dir, monkeypatch):
    """Import the handler afresh, as a new execution environment does, with `env` set."""

    def _load(directory: Path = model_dir, **env: str):
        monkeypatch.setenv("DECIDER_MODEL_DIR", str(directory))
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE", raising=False)
        monkeypatch.delenv("DECIDER_MAX_REQUEST_TOKENS", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        fake = types.ModuleType("strands_decider.backends.llama_cpp")
        fake.LlamaCppBackend = FakeBackend  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "strands_decider.backends.llama_cpp", fake)
        monkeypatch.syspath_prepend(str(DECISION))
        monkeypatch.delitem(sys.modules, "handler", raising=False)
        FakeBackend.calls.clear()
        FakeBackend.fail = False
        return importlib.import_module("handler")

    yield _load
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


def test_loads_the_artifact_and_warms_up_text_and_images(load, model_dir):
    handler = load(DECIDER_MAX_REQUEST_TOKENS="5000")
    model = handler._model.result()
    library, gguf, mmproj, processor_config, _ = model.backend.backend.args
    assert library == str(DECISION / "libdecision.so")
    assert gguf == str(model_dir / "model.gguf") and mmproj == str(model_dir / "mmproj.gguf")
    assert processor_config == str(model_dir / "processor_config.json")
    # A text warm-up, then an image one, before any request.
    assert [len(images) for _, _, images in FakeBackend.calls] == [0, 1]
    assert model.config.final_logit_softcapping == 30.0 and model.readout.softcap == 30.0


def test_health(load):
    assert call(load(DECIDER_MAX_REQUEST_TOKENS="5000"), url_event("GET", "/health")) == (
        200,
        {
            "status": "ok",
            "model": "tiny-decision-q8_0",
            "inputs": ["text", "image"],
            "max_length": 512,
            "max_request_tokens": 5000,
        },
    )


def test_answers_a_text_request(load):
    handler = load()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 200
    assert body["model"] == "tiny-decision-q8_0"
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.0}  # code A, "false", wins
    assert body["answers"]["team"]["choice"] == "billing"  # code B
    assert body["usage"]["output_tokens"] == 2 and body["latency_ms"] >= 0
    (state, questions, images) = FakeBackend.calls[-1]
    assert body["usage"]["input_tokens"] == len(state) + sum(map(len, questions)) and images == ()
    response = handler.handler(url_event("POST", "/v1/systemone", REQUEST), None)
    assert (
        response["headers"]["x-decider-load-wait-ms"]
        and "state_ms" in response["headers"]["x-decider-timings"]
    )
    # A base64-encoded body, as a Function URL may send one.
    assert call(handler, url_event("POST", "/v1/systemone", REQUEST, b64=True))[0] == 200


def test_answers_an_image_request(load):
    handler = load()
    status, body = call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST))
    assert status == 200 and body["answers"]["team"]["choice"] == "billing"
    state, _, (image,) = FakeBackend.calls[-1]
    assert image.tokens == 266 and state[image.start : image.end] == [state[image.start]] * 266


def test_a_direct_invocation(load):
    handler = load()
    payload = handler.handler(REQUEST, None)
    assert payload["answers"]["team"]["choice"] == "billing"
    with pytest.raises(ValueError, match="at least one question"):
        handler.handler({"state": "s", "questions": {}}, None)


@pytest.mark.parametrize(
    ("event", "status", "detail"),
    [
        (url_event("GET", "/nope"), 404, "Not Found"),
        (url_event("GET", "/v1/systemone"), 405, "Method Not Allowed"),
        (url_event("POST", "/health"), 405, "Method Not Allowed"),
        (url_event("POST", "/v1/systemone", raw="{not json"), 422, "the body is not JSON"),
        (url_event("POST", "/v1/systemone", ["a", "list"]), 422, "a request must be an object"),
        (url_event("POST", "/v1/systemone", {"state": "s"}), 422, "at least one question"),
        (url_event("POST", "/v1/systemone", {**REQUEST, "images": [image_b64("checker-64x48.png")] * 5}), 422,
         "5 images; this server takes at most 4"),
        (url_event("POST", "/v1/systemone", {**REQUEST, "images": ["not base64!"]}), 422, "image is not valid base64"),
        (url_event("POST", "/v1/systemone", {**REQUEST, "state": "word " * 600}), 422, "exceeds the limit"),
    ],
)  # fmt: skip
def test_refusals(load, event, status, detail):
    handler = load()
    n_calls = len(FakeBackend.calls)
    got_status, body = call(handler, event)
    assert got_status == status and detail in str(body["detail"])
    assert len(FakeBackend.calls) == n_calls  # refused before the backend ran


def test_the_request_limit(load):
    handler = load(DECIDER_MAX_REQUEST_TOKENS="40")
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 422
    assert "exceeds the limit of 40 tokens per request" in body["detail"]
    assert "(the state's" in body["detail"] and len(FakeBackend.calls) == 2  # the warm-ups only
    with pytest.raises(ValueError, match="DECIDER_MAX_REQUEST_TOKENS"):
        load(DECIDER_MAX_REQUEST_TOKENS="-1")


def test_a_backend_failure_is_a_500(load):
    handler = load()
    FakeBackend.fail = True
    assert call(handler, url_event("POST", "/v1/systemone", REQUEST)) == (
        500,
        {"detail": "Internal Server Error"},
    )


def test_an_artifact_without_a_vision_tower_refuses_images(load, tmp_path):
    from strands_decider.core.request import NO_VISION

    handler = load(write_checkpoint(tmp_path / "text", vision=False))
    assert handler._model.result().health()["inputs"] == ["text"]
    assert call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST)) == (
        422,
        {"detail": NO_VISION},
    )
    handler = load(write_checkpoint(tmp_path / "textonly", inputs=["text"]))
    status, body = call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST))
    assert status == 422 and body["detail"].startswith("this checkpoint does not take images")


def test_a_checkpoint_without_the_soft_cap_is_refused(load, tmp_path):
    with pytest.raises(ValueError, match=r"missing required fields \['final_logit_softcapping'\]"):
        load(write_checkpoint(tmp_path / "nocap", final_logit_softcapping=None))
    assert FakeBackend.calls == []  # refused before the model loads


def test_other_decision_configs_are_refused(load, tmp_path):
    with pytest.raises(ValueError, match="not a Strands Decider decision config"):
        load(write_checkpoint(tmp_path / "pplx", interface=None))
    d = write_checkpoint(tmp_path / "disagree")
    (d / "config.json").write_text(json.dumps({"text_config": {"final_logit_softcapping": 50.0}}))
    with pytest.raises(ValueError, match="inconsistent"):
        load(d)


# ---- the image build's checks ----------------------------------------------------------------


@pytest.fixture
def build_artifact():
    spec = importlib.util.spec_from_file_location(
        "build_artifact", DECISION / "scripts" / "build_artifact.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layout(d: Path, *, text_config=None, **config) -> Path:
    write_checkpoint(d, **config)
    (d / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4ForConditionalGeneration"],
                    "text_config": {"final_logit_softcapping": 30.0} if text_config is None else text_config})
    )  # fmt: skip
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GemmaTokenizer"}))
    return d


def test_the_build_checks_a_good_checkpoint(build_artifact, tmp_path):
    config = build_artifact.check(_layout(tmp_path / "ok"))
    assert config.final_logit_softcapping == 30.0 and config.inputs == ("text", "image")


def test_the_build_refuses_a_checkpoint_without_the_soft_cap(build_artifact, tmp_path):
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        build_artifact.check(_layout(tmp_path / "nocap", final_logit_softcapping=None))
    # config.json without it under text_config, where the multimodal layout keeps it: a reader of
    # config.json's top level would find nothing and skip the cap; the build refuses instead.
    with pytest.raises(SystemExit, match=r"text_config\.final_logit_softcapping is None"):
        build_artifact.check(_layout(tmp_path / "notext", text_config={}))
    with pytest.raises(ValueError, match="inconsistent"):
        build_artifact.check(
            _layout(tmp_path / "differs", text_config={"final_logit_softcapping": 50.0})
        )


def test_the_build_refuses_other_checkpoints(build_artifact, tmp_path):
    d = _layout(tmp_path / "template")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    with pytest.raises(SystemExit, match="no chat template"):
        build_artifact.check(d)
    d = _layout(tmp_path / "processor")
    (d / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "patch_size": 14,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="patch_size"):
        build_artifact.check(d)
    with pytest.raises(ValueError, match="not a Strands Decider decision config"):
        build_artifact.check(_layout(tmp_path / "other", interface=None))
