"""A decision checkpoint's System One API as an AWS Lambda function (README.md, Decision checkpoints).

A Function URL request gets GET /health and POST /v1/systemone. A direct invocation passes a System
One request as the event and gets the response back. The core renders and reads each request; the
decoder runs in llama.cpp (`strands_decider.backends.llama_cpp`) on the GGUFs the image build wrote.

A ValueError is the request's fault and gets a 422: an invalid request, an image that can't be
read, or a request over a limit, each refused before any token runs. Anything else is a 500.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import resource
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from strands_decider.backends.llama_cpp import LlamaCppBackend
from strands_decider.core import (
    DecisionConfig,
    Request,
    evaluate,
    load_readout,
    load_tokenizer,
)
from strands_decider.core.config import IMAGE

log = logging.getLogger("decider")
log.setLevel(logging.INFO)

MODEL_DIR = os.environ.get("DECIDER_MODEL_DIR", "/opt/model")  # the Dockerfile's, unless set
LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libdecision.so")
ROUTES = {"/health": "GET", "/v1/systemone": "POST"}
# The most tokens a request may forward, the state once plus every question, unless
# DECIDER_MAX_REQUEST_TOKENS sets it (0: no limit). The slowest request measured at this limit,
# 4 images and an 8,000-token state, took 549 s of Lambda's 900 s on Graviton2.
DEFAULT_MAX_REQUEST_TOKENS = 16384
READ_CHUNK = 16 << 20
# A 64x64 white PNG, for the image warm-up.
_WARM_UP_IMAGE = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAATUlEQVR42u3PQQ0AAAgEILV/5zOFDzdoQCepz6aeEx"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQELi3cqoDfaKuZM4AAAAASUVORK5CYII="
)
_WARM_UP = {"state": "Warm up.", "questions": {"q": {"type": "noul", "instructions": "?"}}}


class LimitedBackend:
    """The backend, refusing a request that forwards more than `max_request_tokens` tokens before
    it decodes an image or runs a token. None: no limit."""

    def __init__(self, backend: LlamaCppBackend, max_request_tokens: int | None):
        self.backend = backend
        self.max_request_tokens = max_request_tokens

    def forward(self, state: list[int], questions: list[list[int]], images: Any = ()) -> Any:
        n = len(state) + sum(len(q) for q in questions)
        if self.max_request_tokens and n > self.max_request_tokens:
            raise ValueError(
                f"request of {n} tokens (the state's {len(state)} once, then its questions) "
                f"exceeds the limit of {self.max_request_tokens} tokens per request"
            )
        return self.backend.forward(state, questions, images)


class Model:
    """Everything a request needs, loaded once."""

    def __init__(self, model_dir: str, max_request_tokens: int | None):
        started = time.perf_counter()
        with open(os.path.join(model_dir, "artifact.json"), encoding="utf-8") as f:
            artifact = json.load(f)
        self.name = str(artifact["model_name"])
        self.config = DecisionConfig.load(model_dir)
        self.weight = load_readout(model_dir)
        self.tokenizer = load_tokenizer(model_dir)
        gguf = os.path.join(model_dir, "model.gguf")
        mmproj = os.path.join(model_dir, "mmproj.gguf")
        vision = IMAGE in self.config.inputs
        for path in (gguf, mmproj) if vision else (gguf,):
            _prefetch(path)
        log.info("model read in %.1f s", time.perf_counter() - started)
        backend = LlamaCppBackend(
            LIBRARY,
            gguf,
            mmproj=mmproj if vision else None,
            processor_config=os.path.join(model_dir, "processor_config.json") if vision else None,
        )
        if vision:
            # mtmd has read the vision tower into its own buffers, so its pages in the page
            # cache, which Lambda counts as memory used, aren't needed.
            _drop_from_page_cache(mmproj)
        # The first forward's one-time costs, paid before the first request, with no limit yet.
        self.backend = LimitedBackend(backend, None)
        self.evaluate(Request.model_validate(_WARM_UP))
        if vision:
            self.evaluate(Request.model_validate({**_WARM_UP, "images": [_WARM_UP_IMAGE]}))
        # The per-layer embedding table, 2.5 GB of E2B's file, is read a few rows at a time; after
        # the prefetch those rows come back quickly without the table staying in the page cache.
        for start, end in artifact["lazy_ranges"]:
            _drop_from_page_cache(gguf, start, end - start)
        self.backend.max_request_tokens = max_request_tokens
        log.info("model ready in %.1f s", time.perf_counter() - started)

    def evaluate(self, request: Request) -> dict[str, Any]:
        return evaluate(
            request, self.tokenizer, self.weight, self.config, self.backend, model_name=self.name
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.name,
            "inputs": ["text", "image"] if self.backend.backend.has_vision else ["text"],
            "max_length": self.config.max_length_validated,
            "max_request_tokens": self.backend.max_request_tokens,
        }


def _prefetch(path: str, readers: int = 32) -> None:
    """Read a file into the page cache, many chunks at once. Lambda loads an image's files lazily,
    so read in order, as llama.cpp reads it, a model takes minutes to load in a new environment."""
    fd = os.open(path, os.O_RDONLY)

    def read(start: int) -> None:  # in 4 MiB blocks, each dropped once read
        for offset in range(start, min(start + READ_CHUNK, size), 4 << 20):
            os.pread(fd, 4 << 20, offset)

    try:
        size = os.fstat(fd).st_size
        with ThreadPoolExecutor(readers) as pool:
            list(pool.map(read, range(0, size, READ_CHUNK)))
    finally:
        os.close(fd)


def _drop_from_page_cache(path: str, offset: int = 0, length: int = 0) -> None:
    """Drop a file's pages, or `length` bytes of them from `offset` (0: to the end), from the page
    cache. Pages a process has mapped and touched stay."""
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _load() -> Model:
    raw = os.environ.get("DECIDER_MAX_REQUEST_TOKENS", "").strip()
    limit = int(raw) if raw else DEFAULT_MAX_REQUEST_TOKENS
    return Model(MODEL_DIR, limit or None)


def _exit_if_failed(future: Future) -> None:
    # Every invocation would fail; exit, and Lambda replaces the environment.
    if future.exception() is not None:
        log.error("model load failed", exc_info=future.exception())
        os._exit(1)


# On demand, the model loads in a background thread: Lambda gives an on-demand init 10 s, and every
# invocation waits for the load. With provisioned concurrency it loads at import.
if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "on-demand":
    _model: Future = ThreadPoolExecutor(max_workers=1).submit(_load)
    _model.add_done_callback(_exit_if_failed)
else:
    _model = Future()
    _model.set_result(_load())


def _cpu_part() -> str:
    """The host CPU, which Lambda picks per environment: 0xd0c is Graviton2, 0xd40 Graviton3,
    0xd4f Graviton4."""
    try:
        with open("/proc/cpuinfo") as f:
            return next(line for line in f if line.startswith("CPU part")).split(":")[1].strip()
    except (OSError, StopIteration):
        return "?"


def _evaluate(request: Request) -> dict[str, Any]:
    """The response, logged with the request's timings."""
    started = time.perf_counter()
    model = _model.result()
    waited_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    payload = model.evaluate(request)
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    info = {
        "latency_ms": payload["latency_ms"],
        "load_wait_ms": round(waited_ms, 1),
        "input_tokens": payload["usage"]["input_tokens"],
        "questions": payload["usage"]["output_tokens"],
        "cpu_part": _cpu_part(),
        "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
        "timings_ms": model.backend.backend.last_timings,
    }
    log.info("request %s", json.dumps(info))
    return payload


def _response(status: int, body: Any) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def _http(event: dict[str, Any]) -> dict[str, Any]:
    path, method = event["rawPath"], event["requestContext"]["http"]["method"]
    if path not in ROUTES:
        return _response(404, {"detail": "Not Found"})
    if method != ROUTES[path]:
        return _response(405, {"detail": "Method Not Allowed"})
    if path == "/health":
        return _response(200, _model.result().health())
    body = event.get("body") or ""
    try:
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body)
        return _response(200, _evaluate(Request.model_validate_json(body)))
    except ValueError as e:  # pydantic's ValidationError is one
        log.info("refused: %s", e)
        return _response(422, {"detail": str(e)})
    except Exception:
        log.exception("request failed")
        return _response(500, {"detail": "Internal Server Error"})


def handler(event: Any, context: Any) -> dict[str, Any]:
    # A Function URL request arrives as an HTTP event, with a requestContext.http; a direct
    # invocation passes the System One request itself.
    if isinstance(event, dict) and "http" in event.get("requestContext", {}):
        return _http(event)
    return _evaluate(Request.model_validate(event))
