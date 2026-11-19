"""The decision checkpoint's Lambda handler over a stand-in backend, and the image build's handling
of a checkpoint. The backend is tests/test_backend_llama_cpp.py's and tests/test_llama_cpp_shim.py's
to test, the core under both tests/test_core_*.py's.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
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


def write_checkpoint(d: Path, *, vision: bool = True, lazy_ranges=(), **config) -> Path:
    """What the image holds in /opt/model, with a readout of identity rows."""
    from safetensors.numpy import save_file

    d.mkdir(parents=True, exist_ok=True)
    gemma_ids_tokenizer().save(str(d / "tokenizer.json"))
    cfg = {**decision_config_fixture(), "max_length_validated": 512, **config}
    (d / "decision_config.json").write_text(json.dumps(cfg))
    weight = np.zeros((len(CODES), HIDDEN_SIZE), dtype=np.float32)
    weight[0, 0] = weight[1, 1] = 1.0
    save_file({"weight": weight}, str(d / "readout.safetensors"))
    processor = {"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280}
    (d / "processor_config.json").write_text(json.dumps({"image_processor": processor}))
    (d / "model.gguf").write_bytes(bytes(range(256)) * 4)
    if vision:
        (d / "mmproj.gguf").write_bytes(b"\0" * 100)
    artifact = {"model_name": "tiny-decision-q8_0", "lazy_ranges": list(lazy_ranges)}
    (d / "artifact.json").write_text(json.dumps(artifact))
    return d


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Import the handler afresh, as a new execution environment does, from `directory` with `env`
    set. posix_fadvise's calls, which macOS doesn't have, are recorded in `load.dropped`."""
    dropped: list[tuple[int, int]] = []
    fadvise = lambda fd, offset, length, advice: dropped.append((offset, length))  # noqa: E731
    monkeypatch.setattr(os, "posix_fadvise", fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    fake = types.ModuleType("strands_decider.backends.llama_cpp")
    fake.LlamaCppBackend = FakeBackend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strands_decider.backends.llama_cpp", fake)
    monkeypatch.syspath_prepend(str(DECISION))

    def load(directory: Path | None = None, **env: str):
        monkeypatch.setenv(
            "DECIDER_MODEL_DIR", str(directory or write_checkpoint(tmp_path / "model"))
        )
        for key in ("AWS_LAMBDA_INITIALIZATION_TYPE", "DECIDER_MAX_REQUEST_TOKENS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delitem(sys.modules, "handler", raising=False)
        dropped.clear()
        return importlib.import_module("handler")

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


def backend_of(handler) -> FakeBackend:
    return handler._model.result().llama


def test_the_load_warms_up_then_drops_the_lazy_tables_pages(load, tmp_path):
    d = write_checkpoint(tmp_path / "ckpt", lazy_ranges=[[100, 900]])
    handler = load(d)
    backend = backend_of(handler)
    library = str(DECISION / "libdecision.so")
    assert backend.args == (library, str(d / "model.gguf"), str(d / "mmproj.gguf"))
    # A text warm-up, then an image one.
    assert [len(images) for _, _, images in backend.calls] == [0, 1]
    # The vision tower's file, whole, after mtmd reads it; the table's range after the warm-ups.
    assert load.dropped == [(0, 0), (100, 800)]


def test_answers(load):
    handler = load()
    status, body = call(handler, url_event("POST", "/v1/systemone", REQUEST))
    assert status == 200 and body["model"] == "tiny-decision-q8_0" and body["latency_ms"] >= 0
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.0}  # code A, "false", wins
    assert body["answers"]["team"]["choice"] == "billing"  # code B
    state, questions, _ = backend_of(handler).calls[-1]
    assert body["usage"] == {
        "input_tokens": len(state) + sum(map(len, questions)),
        "output_tokens": 2,
    }
    # A base64-encoded body, as a Function URL may send one, with an image; a direct invocation.
    status, with_image = call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST, b64=True))
    assert status == 200 and with_image["answers"] == body["answers"]
    assert len(backend_of(handler).calls[-1][2]) == 1
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
    warm_ups = len(backend_of(handler).calls)
    got_status, body = call(handler, event)
    assert got_status == status and detail in body["detail"]
    assert len(backend_of(handler).calls) == warm_ups


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (ValueError("image could not be decoded: truncated"), 422, "image could not be decoded: truncated"),
        (RuntimeError("sd_forward: the state's decode failed"), 500, "Internal Server Error"),
    ],
)  # fmt: skip
def test_a_backend_error_is_the_requests_only_if_a_value_error(load, error, status, detail):
    handler = load()
    backend_of(handler).fail = error
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


def test_an_on_demand_load_answers_once_ready_and_exits_if_it_fails(load, tmp_path, monkeypatch):
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    assert handler.handler(url_event("POST", "/v1/systemone", REQUEST), None)["statusCode"] == 200
    exited = threading.Event()
    monkeypatch.setattr(os, "_exit", lambda code: exited.set() if code == 1 else None)
    broken = write_checkpoint(tmp_path / "broken")
    (broken / "decision_config.json").unlink()
    handler = load(broken, AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    assert exited.wait(10) and isinstance(handler._model.exception(), FileNotFoundError)


def test_a_text_only_checkpoint(load, tmp_path):
    handler = load(write_checkpoint(tmp_path / "text", vision=False, inputs=["text"]))
    assert backend_of(handler).args[2] is None  # no vision tower
    assert handler._model.result().health()["inputs"] == ["text"]
    status, body = call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST))
    assert status == 422 and body["detail"].startswith("this checkpoint does not take images")


# ---- the image build ------------------------------------------------------------------------------


@pytest.fixture
def build_artifact(monkeypatch):
    # Loading the script puts the checkout's src/ first on sys.path; a copy is put back after the
    # test, so that later tests read the installed distribution's metadata.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "build_artifact", DECISION / "scripts" / "build_artifact.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_build_checks_the_checkpoint_as_the_function_loads_it(build_artifact, tmp_path):
    d = write_checkpoint(tmp_path / "ckpt")
    assert build_artifact.check(d).inputs == ("text", "image")
    (d / "processor_config.json").unlink()
    with pytest.raises(SystemExit, match=r"has no processor_config\.json"):
        build_artifact.check(d)


def test_the_build_fetches_only_an_s3_checkpoints_own_files(build_artifact, tmp_path, monkeypatch):
    objects = {
        "ckpt/e2b/config.json": b"{}",
        "ckpt/e2b/model.safetensors": b"w",
        "ckpt/e2b/sub/x": b"no",
    }

    class Client:
        def get_paginator(self, name):
            listing = [{"Key": k, "Size": len(v)} for k, v in objects.items()]
            return types.SimpleNamespace(paginate=lambda Bucket, Prefix: [{"Contents": listing}])

        def download_file(self, bucket, key, dest, Config):
            Path(dest).write_bytes(objects[key])

    transfer = types.ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = lambda **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda name: Client()))
    monkeypatch.setitem(sys.modules, "boto3.s3", types.ModuleType("boto3.s3"))
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer)
    d, commit = build_artifact.fetch("s3://b/ckpt/e2b", None, tmp_path)
    assert commit is None
    assert sorted(p.name for p in d.iterdir()) == ["config.json", "model.safetensors"]


def test_the_build_pins_a_hub_checkpoint_and_checks_its_files(
    build_artifact, tmp_path, monkeypatch
):
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    lfs = types.SimpleNamespace(sha256=hashlib.sha256(files["model.safetensors"]).hexdigest())
    siblings = [types.SimpleNamespace(rfilename="config.json", lfs=None),
                types.SimpleNamespace(rfilename="model.safetensors", lfs=lfs)]  # fmt: skip
    downloaded = []

    def snapshot_download(repo, revision, local_dir):
        downloaded.append(revision)
        for n, b in files.items():
            (Path(local_dir) / n).write_bytes(b)

    info = types.SimpleNamespace(sha="c" * 40, siblings=siblings)
    hub = types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(model_info=lambda repo, revision, files_metadata: info),
        snapshot_download=snapshot_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert build_artifact.fetch("org/model", "main", tmp_path / "a")[1] == "c" * 40
    assert downloaded == ["c" * 40]  # the resolved commit, not the branch
    lfs.sha256 = "0" * 64
    with pytest.raises(SystemExit, match="does not have the sha256 the Hub records"):
        build_artifact.fetch("org/model", "main", tmp_path / "b")


def test_the_build_refuses_a_checkpoint_with_another_digest(build_artifact, tmp_path, monkeypatch):
    """The digest is `sha256sum * | sort -k 2 | sha256sum` in the checkpoint's directory."""
    ckpt = write_checkpoint(tmp_path / "ckpt")
    lines = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(ckpt.iterdir())
    )
    assert build_artifact.manifest_digest(ckpt) == hashlib.sha256(lines.encode()).hexdigest()
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "LLAMA_CPP_REV").write_text("f" * 40 + "\n")
    argv = ["build_artifact.py", str(ckpt), str(tmp_path / "out"), "--llama-cpp", str(llama),
            "--quantize", "/bin/false", "--expect-digest", "0" * 64, "--work", str(tmp_path)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="manifest digest is"):
        build_artifact.main()
    assert not (tmp_path / "out").exists()  # refused before anything was converted
