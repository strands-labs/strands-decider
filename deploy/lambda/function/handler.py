"""strands-decider's System One API as an AWS Lambda function (README.md).

A Function URL request gets `strands-decider serve`'s API: GET /health and POST /v1/systemone,
with its request and response bodies and status codes. A direct invocation passes a System One
request as the event and returns the response; an invalid request fails the invocation.

The model loads once per execution environment. With SnapStart it loads at import, when a
version is published, and new environments restore from the snapshot. Otherwise it loads in a
background thread: Lambda gives an on-demand init 10 s, and an init that overruns is run again,
from the start, inside the first invocation. Every invocation waits for the load.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from backend import LlamaBackend
from pydantic import ValidationError

from strands_decider.hidden_engine import HiddenStateEngine
from strands_decider.schema import SystemOneRequest

log = logging.getLogger("decider")
log.setLevel(logging.INFO)

MODEL_DIR = os.environ.get("DECIDER_MODEL_DIR", "/opt/model")  # the Dockerfile's, unless set
ROUTES = {"/health": "GET", "/v1/systemone": "POST"}


def _prefetch(path: str, chunk: int = 16 << 20, readers: int = 32) -> None:
    """Read a file into the page cache, many ranges at once. Lambda loads an image's files
    lazily: read once, in order, as llama.cpp reads it, the model took from 11 s to 7 minutes to
    load in a new environment, and 11 to 23 s read this way."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size

        def read(offset: int) -> None:
            end = min(offset + chunk, size)
            while offset < end:
                n = len(os.pread(fd, end - offset, offset))
                if n == 0:
                    break
                offset += n

        with ThreadPoolExecutor(readers) as pool:
            list(pool.map(read, range(0, size, chunk)))
    finally:
        os.close(fd)


def _load() -> HiddenStateEngine:
    started = time.perf_counter()
    model = os.path.join(MODEL_DIR, "decider-q8_0.gguf")
    _prefetch(model)
    log.info("model read in %.1f s", time.perf_counter() - started)
    backend = LlamaBackend(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libdecider.so"),
        model,
        os.cpu_count() or 1,
    )
    engine = HiddenStateEngine.from_assets(
        os.path.join(MODEL_DIR, "assets"),
        backend,
        strict_window=os.environ.get("DECIDER_STRICT_WINDOW") == "true",
    )
    # The first forward's one-time costs, paid before the snapshot is taken, and before the
    # request limit, which could refuse even this.
    engine.evaluate(
        {"state": "Warm up.", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    )
    engine.cfg.max_request_tokens = int(os.environ.get("DECIDER_MAX_REQUEST_TOKENS") or 0) or None
    log.info("model ready in %.1f s", time.perf_counter() - started)
    return engine


def _exit_if_failed(future: Future) -> None:
    # Every invocation would fail; exit, and Lambda replaces the environment.
    if future.exception() is not None:
        log.error("model load failed", exc_info=future.exception())
        os._exit(1)


if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "on-demand":
    _engine: Future = ThreadPoolExecutor(max_workers=1).submit(_load)
    _engine.add_done_callback(_exit_if_failed)
else:  # snap-start
    _engine = Future()
    _engine.set_result(_load())


def _cpu_part() -> str:
    """The host CPU, read per request because a snapshot can restore on any: 0xd0c is
    Graviton2, 0xd40 Graviton3, 0xd4f Graviton4."""
    try:
        with open("/proc/cpuinfo") as f:
            lines = [line for line in f if line.startswith("CPU part")]
    except OSError:  # not Linux
        return "?"
    return lines[0].split(":")[1].strip() if lines else "?"


def _evaluate(request: SystemOneRequest) -> dict[str, Any]:
    started = time.perf_counter()
    payload = _engine.result().evaluate(request).model_dump()
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "latency_ms=%s input_tokens=%s cpu_part=%s",
        payload["latency_ms"],
        payload["usage"]["input_tokens"],
        _cpu_part(),
    )
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
    engine = _engine.result()
    if path == "/health":
        return _response(
            200,
            {
                "status": "ok",
                "model": engine.cfg.model_name,
                "max_length": engine.cfg.max_length,
                "temperature": engine.cfg.temperature,
                "strict_window": engine.cfg.strict_window,
                "max_request_tokens": engine.cfg.max_request_tokens,
            },
        )
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    try:
        request = SystemOneRequest.model_validate_json(body)
    except ValidationError as e:  # FastAPI's 422, as `serve` answers
        errors = e.errors(include_url=False, include_context=False)
        return _response(422, {"detail": [{**err, "loc": ["body", *err["loc"]]} for err in errors]})
    try:
        return _response(200, _evaluate(request))
    except ValueError as e:  # over the window or the request limit
        return _response(422, {"detail": str(e)})


def handler(event: Any, context: Any) -> dict[str, Any]:
    if isinstance(event, dict) and "http" in event.get("requestContext", {}):
        return _http(event)
    return _evaluate(SystemOneRequest.model_validate(event))
