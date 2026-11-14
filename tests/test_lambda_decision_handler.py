"""The decision checkpoint's AWS Lambda handler (deploy/lambda/decision/handler.py), over a stand-in
backend, and the image build's checks of a checkpoint (deploy/lambda/decision/scripts/build_artifact.py).

The backend is the one part replaced: it is tests/test_backend_llama_cpp.py's and, with the real
library, tests/test_llama_cpp_shim.py's to test. Everything else is the handler's own: the load and its validation
of the checkpoint's files, the routes, the status codes and bodies, the limits and the warm-up. The
core under it is tests/test_core_*.py's. A checkpoint whose decision_config.json lacks the
soft-cap, or whose config.json names another, is refused, at load and at build.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import struct
import sys
import threading
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from core_support import decision_config_fixture, image_b64

from strands_decider.backends.llama_cpp import ImageError
from strands_decider.core import CODES, CONTROL_TOKEN_IDS, HIDDEN

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
    fail: ClassVar[Exception | None] = None
    outputs = HIDDEN

    def __init__(self, library, model, *, mmproj=None, processor_config=None, threads=None):
        self.args = (library, model, mmproj, processor_config, threads)
        self.has_vision = mmproj is not None
        self.last_timings: dict = {}

    def forward(self, state, questions, images=()):
        if FakeBackend.fail is not None:
            raise FakeBackend.fail
        FakeBackend.calls.append((state, questions, tuple(images)))
        self.last_timings = {"state_ms": 1.0, "question_ms": [2.0] * len(questions)}
        if images:
            unique = {im.sha256 for im in images}
            self.last_timings["images"] = [{"size": [912, 672], "encode_ms": 3.0}] * len(unique)
            self.last_timings["n_images"] = len(images)
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


def gemma_ids_tokenizer():
    """A word-level tokenizer with Gemma 4's control tokens and every code at Gemma 4's ids (the
    fixture's token_ids), so that it passes the core's check_tokenizer, as load_tokenizer runs
    it; other text is a few small ids."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    taken = set(CONTROL_TOKEN_IDS.values()) | set(decision_config_fixture()["token_ids"])
    free = (i for i in range(1000) if i not in taken)
    vocab = {w: next(free) for w in words}
    vocab.update(zip(CODES, decision_config_fixture()["token_ids"], strict=True))
    vocab.update(CONTROL_TOKEN_IDS)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    tok.add_special_tokens([AddedToken(t, special=True) for t in CONTROL_TOKEN_IDS])
    return tok


def write_checkpoint(d: Path, *, vision: bool = True, **config) -> Path:
    """What the image holds in /opt/model, with a tokenizer at Gemma 4's ids and a readout of
    identity rows."""
    d.mkdir(parents=True, exist_ok=True)
    gemma_ids_tokenizer().save(str(d / "tokenizer.json"))
    cfg = {**decision_config_fixture(), "max_length_validated": 512, **config}
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
    (d / "model.gguf").write_bytes(bytes(range(256)) * 4)
    if vision:
        (d / "mmproj.gguf").write_bytes(b"\0" * 100)
    record_artifact(d)
    return d


CHUNK = 16 << 20


def record_artifact(d: Path, **extra) -> None:
    """artifact.json as the image build writes it: each served file's size and sha256, and the
    GGUFs' chunk digests."""
    files = {}
    for p in sorted(d.iterdir()):
        if p.name in ("artifact.json", "config.json") or not p.is_file():
            continue
        data = p.read_bytes()
        rec = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if p.name.endswith(".gguf"):
            rec |= {
                "chunk_size": CHUNK,
                "chunks": [
                    hashlib.sha256(data[i : i + CHUNK]).hexdigest()
                    for i in range(0, len(data), CHUNK)
                ],
            }
        files[p.name] = rec
    (d / "artifact.json").write_text(
        json.dumps({"model_name": "tiny-decision-q8_0", "files": files, **extra})
    )


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
        fake.ImageError = ImageError  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "strands_decider.backends.llama_cpp", fake)
        monkeypatch.syspath_prepend(str(DECISION))
        monkeypatch.delitem(sys.modules, "handler", raising=False)
        FakeBackend.calls.clear()
        FakeBackend.fail = None
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


@pytest.mark.parametrize(("setting", "dropped"), [(None, True), ("hot", True), ("all", False)])
def test_the_lazy_tables_pages_leave_the_page_cache(load, model_dir, monkeypatch, setting, dropped):
    calls = []
    monkeypatch.setattr(
        handler_os(),
        "posix_fadvise",
        lambda fd, off, n, advice: calls.append((off, n)),
        raising=False,
    )
    monkeypatch.setattr(handler_os(), "POSIX_FADV_DONTNEED", 4, raising=False)
    # One good range, and ones that are empty, past the file's end, or negative, which are skipped:
    # posix_fadvise reads a length of 0 as "to the end of the file".
    bad = [[900, 900], [1000, 5000], [-1, 10], ["a", 9]]
    record_artifact(model_dir, lazy_ranges=[[100, 900], *bad])
    load(**({} if setting is None else {"DECIDER_PAGE_CACHE": setting}))
    # The vision tower's file, whole, after mtmd reads it; the table's range after the warm-ups.
    assert calls == ([(0, 0), (100, 800)] if dropped else [(0, 0)])


def test_a_bad_page_cache_setting_is_refused(load):
    with pytest.raises(ValueError, match="DECIDER_PAGE_CACHE"):
        load(DECIDER_PAGE_CACHE="none")


def handler_os():
    import os

    return os


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
    stateless = {k: v for k, v in REQUEST.items() if k != "state"}  # the state is optional
    assert handler.handler(stateless, None)["answers"].keys() == payload["answers"].keys()
    with pytest.raises(ValueError, match=r"^questions: Dictionary should have at least 1 item"):
        handler.handler({"state": "s", "questions": {}}, None)


@pytest.mark.parametrize(
    ("event", "status", "detail"),
    [
        (url_event("GET", "/nope"), 404, "Not Found"),
        (url_event("GET", "/v1/systemone"), 405, "Method Not Allowed"),
        (url_event("POST", "/health"), 405, "Method Not Allowed"),
        (url_event("POST", "/v1/systemone", raw="{not json"), 422, "the body is not JSON"),
        (url_event("POST", "/v1/systemone", ["a", "list"]), 422, "a request must be a JSON object"),
        (url_event("POST", "/v1/systemone", {"state": "s"}), 422, "questions: Field required"),
        (url_event("POST", "/v1/systemone", {**REQUEST, "audio": []}), 422, "audio input is not supported"),
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


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("sd_forward failed: the state's decode failed"), ValueError("an internal check")],
)
def test_a_backend_failure_is_a_500(load, failure):
    handler = load()
    FakeBackend.fail = failure
    assert call(handler, url_event("POST", "/v1/systemone", REQUEST)) == (
        500,
        {"detail": "Internal Server Error"},
    )


def test_an_image_that_cannot_be_decoded_is_a_422(load):
    handler = load()
    FakeBackend.fail = ImageError("image could not be decoded: broken data stream")
    assert call(handler, url_event("POST", "/v1/systemone", IMAGE_REQUEST)) == (
        422,
        {"detail": "image could not be decoded: broken data stream"},
    )


def test_audio_is_refused(load):
    status, body = call(
        load(), url_event("POST", "/v1/systemone", {**REQUEST, "audio": ["UklGRg=="]})
    )
    assert status == 422 and "audio input is not supported" in body["detail"]


def test_the_request_limit_fails_closed(load):
    assert load()._model.result().health()["max_request_tokens"] == 16384
    assert (
        load(DECIDER_MAX_REQUEST_TOKENS="")._model.result().health()["max_request_tokens"] == 16384
    )
    assert (
        load(DECIDER_MAX_REQUEST_TOKENS="0")._model.result().health()["max_request_tokens"] is None
    )


def test_images_are_counted_with_repeats(load, tmp_path):
    handler = load(write_checkpoint(tmp_path / "long", max_length_validated=2048))
    twice = {**REQUEST, "images": [image_b64("checker-64x48.png")] * 2}
    _, info = handler._evaluate(twice)
    assert info["images"] == 2 and len(info["timings_ms"]["images"]) == 1


def test_a_served_file_that_is_not_the_builds_is_refused(load, tmp_path):
    for name in ("model.gguf", "mmproj.gguf", "readout.safetensors", "tokenizer.json"):
        d = write_checkpoint(tmp_path / name)
        data = bytearray((d / name).read_bytes())
        data[len(data) // 2] ^= 1
        (d / name).write_bytes(bytes(data))
        with pytest.raises(ValueError, match="is not the file the image build wrote"):
            load(d)
    assert FakeBackend.calls == []


def test_a_listed_vision_tower_must_be_there(load, tmp_path):
    d = write_checkpoint(tmp_path / "gone")
    (d / "mmproj.gguf").unlink()
    with pytest.raises(FileNotFoundError):
        load(d)


def test_an_on_demand_load_that_fails_ends_the_environment(load, tmp_path, monkeypatch):
    import os

    exited = threading.Event()
    codes = []

    def _exit(code):
        codes.append(code)
        exited.set()

    monkeypatch.setattr(os, "_exit", _exit)
    handler = load(
        write_checkpoint(tmp_path / "nocap", final_logit_softcapping=None),
        AWS_LAMBDA_INITIALIZATION_TYPE="on-demand",
    )
    assert exited.wait(10) and codes == [1]
    assert isinstance(handler._model.exception(), ValueError)


def test_an_on_demand_load_answers_once_ready(load):
    handler = load(AWS_LAMBDA_INITIALIZATION_TYPE="on-demand")
    response = handler.handler(url_event("POST", "/v1/systemone", REQUEST), None)
    assert (
        response["statusCode"] == 200 and float(response["headers"]["x-decider-load-wait-ms"]) >= 0
    )


def test_an_artifact_without_a_vision_tower_refuses_images(load, tmp_path):
    from strands_decider.core import NO_VISION

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


def test_a_tokenizer_without_the_checkpoints_ids_is_refused(load, tmp_path):
    from core_support import toy_tokenizer

    d = write_checkpoint(tmp_path / "toy")
    toy_tokenizer().save(str(d / "tokenizer.json"))
    record_artifact(d)
    with pytest.raises(ValueError, match="control tokens are not Gemma 4's"):
        load(d)


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


def test_the_build_fetches_an_s3_checkpoint_and_only_its_files(
    build_artifact, tmp_path, monkeypatch
):
    objects = {
        "ckpt/e2b/config.json": b"{}",
        "ckpt/e2b/model.safetensors": b"w" * 10,
        "ckpt/e2b/sub/x": b"no",
    }

    class Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return types.SimpleNamespace(
                paginate=lambda Bucket, Prefix: [
                    {
                        "Contents": [
                            {"Key": k, "Size": len(v)}
                            for k, v in objects.items()
                            if k.startswith(Prefix)
                        ]
                    }
                ]
            )

        def download_file(self, bucket, key, dest, Config):
            assert bucket == "b"
            Path(dest).write_bytes(objects[key])

    fake = types.ModuleType("boto3")
    fake.client = lambda name: Client()  # type: ignore[attr-defined]
    transfer = types.ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = lambda **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)
    monkeypatch.setitem(sys.modules, "boto3.s3", types.ModuleType("boto3.s3"))
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer)
    d, commit = build_artifact.fetch("s3://b/ckpt/e2b", None, tmp_path)
    assert commit is None and sorted(p.name for p in d.iterdir()) == [
        "config.json",
        "model.safetensors",
    ]


def test_the_build_pins_a_hub_checkpoint_and_checks_its_files(
    build_artifact, tmp_path, monkeypatch
):
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    lfs = {"model.safetensors": hashlib.sha256(files["model.safetensors"]).hexdigest()}
    seen = {}

    class Api:
        def model_info(self, repo, revision, files_metadata):
            seen["info"] = (repo, revision)
            siblings = [
                types.SimpleNamespace(
                    rfilename=n, lfs=types.SimpleNamespace(sha256=lfs[n]) if n in lfs else None
                )
                for n in files
            ]
            return types.SimpleNamespace(sha="c" * 40, siblings=siblings)

    def snapshot_download(repo, revision, local_dir):
        seen["download"] = revision
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for n, b in files.items():
            (Path(local_dir) / n).write_bytes(b)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = Api  # type: ignore[attr-defined]
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    _, commit = build_artifact.fetch("org/model", "main", tmp_path / "a")
    assert commit == "c" * 40 and seen == {"info": ("org/model", "main"), "download": "c" * 40}
    lfs["model.safetensors"] = "0" * 64
    with pytest.raises(SystemExit, match="does not have the sha256 the Hub records"):
        build_artifact.fetch("org/model", "main", tmp_path / "b")


def test_the_manifest_digest_is_sha256sum_of_the_files(build_artifact, tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    for n, b in {"b.json": b"2", "a.json": b"1"}.items():
        (d / n).write_bytes(b)
    lines = "".join(
        f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in [("a.json", b"1"), ("b.json", b"2")]
    )
    assert build_artifact.manifest_digest(d) == hashlib.sha256(lines.encode()).hexdigest()


def test_the_build_refuses_a_checkpoint_with_another_digest(build_artifact, tmp_path, monkeypatch):
    ckpt = _layout(tmp_path / "ckpt")
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "LLAMA_CPP_REV").write_text("f" * 40 + "\n")
    assert build_artifact.llama_cpp_rev(llama) == "f" * 40
    argv = ["build_artifact.py", str(ckpt), str(tmp_path / "out"), "--llama-cpp", str(llama),
            "--quantize", "/bin/false", "--expect-digest", "0" * 64]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="manifest digest is"):
        build_artifact.main()
    assert not (tmp_path / "out").exists()  # refused before anything was converted
    (llama / "LLAMA_CPP_REV").unlink()
    with pytest.raises(SystemExit, match="no LLAMA_CPP_REV"):
        build_artifact.llama_cpp_rev(llama)
