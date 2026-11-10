"""A decision checkpoint's System One API as an AWS Lambda function (README.md, Decision checkpoints).

A Function URL request gets GET /health and POST /v1/systemone, with `strands-decider serve`'s
request and response bodies. A direct invocation passes a System One request as the event and
returns the response; an invalid request fails the invocation.

The request is rendered, checked and answered by the torch-free core (`strands_decider.core`,
docs/spec.md), and the decoder runs in llama.cpp (`strands_decider.backends.llama_cpp`): the
GGUF, and for images the vision tower's mmproj, that the image build converted from the
checkpoint. The readout and the soft-cap come from the checkpoint's readout.safetensors and
decision_config.json, which the load validates.

A request may carry at most 4 images, its prompts at most the checkpoint's max_length_validated
tokens (the state, image placeholders included, plus a question), and at most
DECIDER_MAX_REQUEST_TOKENS tokens in all, the state once, then every question: 16,384 unless it is
set, which keeps the slowest request inside the 15-minute timeout on Graviton2, and 0 for no limit.
Each limit is checked before any image is decoded or any token is run. A request that breaks one,
or carries an image that cannot be read, gets a 422; anything else that fails is a 500.

Every file the function serves is checked at load against the sha256 artifact.json records, the
GGUFs chunk by chunk as they are read into the page cache; a mismatch fails the load.

The model loads once per execution environment. On demand it loads in a background thread: Lambda
gives an on-demand init 10 s, and every invocation waits for the load. With provisioned
concurrency it loads at import. SnapStart is not offered for decision checkpoints: it was not
measured, and the page-cache drop after the warm-up assumes the prefetch ran in this environment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import resource
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from strands_decider.backends.llama_cpp import ImageError, LlamaCppBackend
from strands_decider.core import DecisionConfig, Readout, evaluate, load_tokenizer
from strands_decider.core.config import IMAGE

log = logging.getLogger("decider")
log.setLevel(logging.INFO)

MODEL_DIR = os.environ.get("DECIDER_MODEL_DIR", "/opt/model")  # the Dockerfile's, unless set
LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libdecision.so")
ROUTES = {"/health": "GET", "/v1/systemone": "POST"}
# The request limit without DECIDER_MAX_REQUEST_TOKENS (README.md, Decision checkpoints, Limits).
DEFAULT_MAX_REQUEST_TOKENS = 16384
# The chunk the GGUFs are read and checked in: artifact.json's chunk_size.
CHUNK = 16 << 20
# A 64x64 white PNG, for the image warm-up.
_WARM_UP_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAATUlEQVR42u3PQQ0AAAgEILV/5zOFDzdoQCepz6aeEx"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQELi3cqoDfaKuZM4AAAAASUVORK5CYII="
)
_WARM_UP = {"state": "Warm up.", "questions": {"q": {"type": "noul", "instructions": "?"}}}


class RequestError(ValueError):
    """The request's fault: an invalid request, an image that cannot be read, or one over a limit.
    The Function URL answers it with a 422; every other failure is a 500."""


class RequestTooLarge(RequestError):
    pass


class LimitedBackend:
    """The backend, refusing a request that forwards more than `max_request_tokens` tokens (the
    state once, then every question) before it decodes an image or runs a token. None: no limit."""

    def __init__(self, backend: LlamaCppBackend, max_request_tokens: int | None):
        self.backend = backend
        self.max_request_tokens = max_request_tokens
        self.outputs = backend.outputs
        self.inputs = backend.inputs
        self.forwarded = False  # whether the request has been through the decoder

    def forward(
        self, state: list[int], questions: list[list[int]], images: Sequence[Any] = ()
    ) -> Any:
        n = len(state) + sum(len(q) for q in questions)
        if self.max_request_tokens and n > self.max_request_tokens:
            raise RequestTooLarge(
                f"request of {n} tokens (the state's {len(state)} once, then its questions) "
                f"exceeds the limit of {self.max_request_tokens} tokens per request"
            )
        try:
            out = (
                self.backend.forward(state, questions, images)
                if images
                else self.backend.forward(state, questions)
            )
        except ImageError as e:
            raise RequestError(str(e)) from e
        except ValueError as e:  # the backend's own, not the request's
            raise RuntimeError(f"the backend failed: {e}") from e
        self.forwarded = True
        return out


class Model:
    """Everything a request needs, loaded once."""

    def __init__(self, model_dir: str, max_request_tokens: int | None):
        started = time.perf_counter()
        with open(os.path.join(model_dir, "artifact.json"), encoding="utf-8") as f:
            self.artifact = json.load(f)
        files: dict[str, Any] = self.artifact["files"]
        # The small files, checked before they are read; the GGUFs below, as they are read.
        for name, record in files.items():
            if not name.endswith(".gguf"):
                _verify(os.path.join(model_dir, name), record)
        # The checkpoint's own files, validated before the model loads: a decision_config.json
        # without `"interface": "strands-decider"`, or without the soft-cap, is refused here, and
        # so is a tokenizer that does not give the config's code and control token ids.
        self.config = DecisionConfig.load(model_dir)
        self.readout = Readout.load(model_dir, self.config)
        self.tokenizer = load_tokenizer(model_dir, self.config)
        self.name = str(self.artifact["model_name"])
        gguf = os.path.join(model_dir, "model.gguf")
        mmproj = os.path.join(model_dir, "mmproj.gguf")
        # An artifact without a vision tower serves text only; one that lists it must have it.
        vision = IMAGE in self.config.inputs and "mmproj.gguf" in files
        for path in (gguf, mmproj) if vision else (gguf,):
            _prefetch(path, files[os.path.basename(path)])
        log.info("model read in %.1f s", time.perf_counter() - started)
        backend = LlamaCppBackend(
            LIBRARY,
            gguf,
            mmproj=mmproj if vision else None,
            processor_config=os.path.join(model_dir, "processor_config.json") if vision else None,
            threads=os.cpu_count() or 1,
        )
        if vision:
            # mtmd has read the vision tower into its own buffers; its file's pages in the page
            # cache, which Lambda counts as memory used, are not read again.
            _drop_from_page_cache(mmproj)
        self.backend = LimitedBackend(backend, None)
        # The first forward's one-time costs, paid before the snapshot is taken or the first
        # request arrives, and before the request limit, which could refuse even this.
        self.evaluate(_WARM_UP)
        if vision:
            self.evaluate({**_WARM_UP, "images": [_WARM_UP_IMAGE]})
        if _page_cache() == "hot":
            # The prefetch has brought the whole file into this environment, so the rows of the
            # per-layer embedding table that a request reads come back quickly; the table's 2.5 GB
            # (E2B) need not stay in the page cache, which Lambda counts as memory used.
            size = files["model.gguf"]["size"]
            for start, end in self.artifact.get("lazy_ranges", []):
                if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= size:
                    _drop_from_page_cache(gguf, start, end - start)
        self.backend.max_request_tokens = max_request_tokens
        log.info("model ready in %.1f s", time.perf_counter() - started)

    def evaluate(self, request: Any) -> dict[str, Any]:
        """The response to `request`. A RequestError is the request's fault: the core's refusals
        come before the decoder runs, so a ValueError after it is the runtime's (RuntimeError)."""
        self.backend.forwarded = False
        try:
            return evaluate(
                request,
                self.tokenizer,
                self.readout,
                self.config,
                self.backend,
                model_name=self.name,
            )
        except RequestError:
            raise
        except ValueError as e:
            if self.backend.forwarded:
                raise RuntimeError(f"reading the answers failed: {e}") from e
            raise RequestError(str(e)) from e

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.name,
            "inputs": list(self.backend.inputs),
            "max_length": self.config.max_length_validated,
            "max_request_tokens": self.backend.max_request_tokens,
        }


def _verify(path: str, record: dict[str, Any]) -> None:
    """That a served file has the size and sha256 artifact.json records for it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    if os.path.getsize(path) != record["size"] or h.hexdigest() != record["sha256"]:
        raise ValueError(f"{path} is not the file the image build wrote: its sha256 differs")


def _prefetch(path: str, record: dict[str, Any], readers: int = 32) -> None:
    """Read a file into the page cache, many chunks at once, and check each chunk against the
    sha256 artifact.json records for it. Lambda loads an image's files lazily: read in order, as
    llama.cpp reads it, a model takes minutes to load in a new environment. The whole GGUF is
    read, its per-layer embedding table included, whose rows llama.cpp would otherwise read on
    demand, slowly and erratically, from the lazily loaded image."""
    chunks: list[str] = record["chunks"]
    if record.get("chunk_size") != CHUNK:
        raise ValueError(f"{path}: artifact.json's chunk size is not {CHUNK}")
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size != record["size"] or len(chunks) != -(-size // CHUNK):
            raise ValueError(f"{path} is not the file the image build wrote: its size differs")

        def read(i: int) -> bool:
            offset, end = i * CHUNK, min((i + 1) * CHUNK, size)
            h = hashlib.sha256()
            while offset < end:
                block = os.pread(fd, min(end - offset, 4 << 20), offset)
                if not block:
                    return False
                h.update(block)
                offset += len(block)
            return h.hexdigest() == chunks[i]

        with ThreadPoolExecutor(readers) as pool:
            bad = [i for i, ok in enumerate(pool.map(read, range(len(chunks)))) if not ok]
    finally:
        os.close(fd)
    if bad:
        raise ValueError(f"{path} is not the file the image build wrote: chunks {bad[:5]} differ")


def _drop_from_page_cache(path: str, offset: int = 0, length: int = 0) -> None:
    """Drop a file's pages, or `length` bytes of them from `offset` (0: to the end), from the page
    cache. Pages a process has mapped and touched stay."""
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _page_cache() -> str:
    """DECIDER_PAGE_CACHE: "hot" (the default) keeps the model's hot tensors in the page cache and
    drops the lazily read table after the prefetch; "all" keeps the whole file."""
    value = os.environ.get("DECIDER_PAGE_CACHE") or "hot"
    if value not in ("hot", "all"):
        raise ValueError(f'DECIDER_PAGE_CACHE must be "hot" or "all", not {value!r}')
    return value


def _max_request_tokens() -> int | None:
    """DECIDER_MAX_REQUEST_TOKENS: DEFAULT_MAX_REQUEST_TOKENS when unset, None (no limit) for 0."""
    raw = os.environ.get("DECIDER_MAX_REQUEST_TOKENS")
    value = DEFAULT_MAX_REQUEST_TOKENS if raw is None or raw.strip() == "" else int(raw)
    if value < 0:
        raise ValueError(f"DECIDER_MAX_REQUEST_TOKENS must be 0 (no limit) or more, not {value}")
    return value or None


def _load() -> Model:
    _page_cache()  # checked before the model loads
    return Model(MODEL_DIR, _max_request_tokens())


def _exit_if_failed(future: Future) -> None:
    # Every invocation would fail; exit, and Lambda replaces the environment.
    if future.exception() is not None:
        log.error("model load failed", exc_info=future.exception())
        os._exit(1)


if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "on-demand":
    _model: Future = ThreadPoolExecutor(max_workers=1).submit(_load)
    _model.add_done_callback(_exit_if_failed)
else:  # snap-start, provisioned concurrency, or not on Lambda
    _model = Future()
    _model.set_result(_load())


def _cpu_part() -> str:
    """The host CPU, read per request because a snapshot can restore on any: 0xd0c is Graviton2,
    0xd40 Graviton3, 0xd4f Graviton4."""
    try:
        with open("/proc/cpuinfo") as f:
            lines = [line for line in f if line.startswith("CPU part")]
    except OSError:  # not Linux
        return "?"
    return lines[0].split(":")[1].strip() if lines else "?"


def _evaluate(request: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The response, and what the log line and the response headers report about it."""
    started = time.perf_counter()
    model = _model.result()
    waited = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    payload = model.evaluate(request)
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    timings = model.backend.backend.last_timings
    info = {
        "latency_ms": payload["latency_ms"],
        "load_wait_ms": round(waited, 1),
        "input_tokens": payload["usage"]["input_tokens"],
        "questions": payload["usage"]["output_tokens"],
        "images": timings.get("n_images", 0),
        "cpu_part": _cpu_part(),
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "timings_ms": _rounded(timings),
    }
    log.info("request %s", json.dumps(info))
    return payload, info


def _rounded(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 1)
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    return x


def _response(status: int, body: Any, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", **(headers or {})},
        "body": json.dumps(body),
    }


def _http(event: dict[str, Any]) -> dict[str, Any]:
    path, method = event["rawPath"], event["requestContext"]["http"]["method"]
    if path not in ROUTES:
        return _response(404, {"detail": "Not Found"})
    if method != ROUTES[path]:
        return _response(405, {"detail": "Method Not Allowed"})
    if path == "/health":
        return _response(200, _model.result().health(), {"x-decider-cpu-part": _cpu_part()})
    body = event.get("body") or ""
    try:
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        request = json.loads(body)
    except (ValueError, UnicodeDecodeError) as e:
        return _response(422, {"detail": f"the body is not JSON: {e}"})
    try:
        payload, info = _evaluate(request)
    except RequestError as e:  # an invalid request, an image that cannot be read, or over a limit
        log.info("refused: %s", e)
        return _response(422, {"detail": str(e)})
    except Exception:
        log.exception("request failed")
        return _response(500, {"detail": "Internal Server Error"})
    headers = {
        "x-decider-cpu-part": info["cpu_part"],
        "x-decider-load-wait-ms": str(info["load_wait_ms"]),
        "x-decider-timings": json.dumps(info["timings_ms"], separators=(",", ":")),
    }
    return _response(200, payload, headers)


def handler(event: Any, context: Any) -> dict[str, Any]:
    # Two ways in. A Function URL request arrives wrapped in an HTTP event, with a
    # requestContext.http, and gets serve's routes and status codes. A direct invocation passes
    # the System One request itself as the event, and gets the response as the result.
    if isinstance(event, dict) and "http" in event.get("requestContext", {}):
        return _http(event)
    return _evaluate(event)[0]
