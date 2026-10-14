"""The System One API on AWS Lambda, served by llama.cpp (Q8_0) without torch.

Same request and response schema as `strands-decider serve`. Everything heavy (the
GGUF from S3 into memory, the model load, a warm-up forward) runs at import. With
SnapStart that happens once, when a version is published, and new environments
restore from the snapshot. Without SnapStart it runs in a background thread: Lambda
gives an on-demand init 10 s, and re-runs an init that overruns inside the first
request, which would throw away 10 s of the roughly 30 s load on every cold start.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

import loader
from engine import DeciderEngine
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("decider.app")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "strands-decider-2B-qwen3.5-v1-2610-q8_0")
THREADS = int(os.environ.get("THREADS", "0")) or (os.cpu_count() or 1)
# About 5 minutes on a Graviton3 host and 8 on a Graviton2, inside the 15-minute timeout. 0: no limit.
MAX_REQUEST_TOKENS = int(os.environ.get("MAX_REQUEST_TOKENS", "24576"))


def _load() -> tuple[DeciderEngine, float]:
    t0 = time.perf_counter()
    engine = DeciderEngine(
        lib_path=os.path.join(HERE, "libs", "libdecider_hidden.so"),
        gguf_path=loader.download_to_memfd(os.environ["MODEL_BUCKET"], os.environ["MODEL_KEY"]),
        assets_dir=os.path.join(HERE, "assets"),
        threads=THREADS,
        max_request_tokens=MAX_REQUEST_TOKENS or None,
    )
    # Pay the first-forward costs before the snapshot is taken.
    engine.evaluate(
        {
            "state": "warm up",
            "questions": {"q": {"type": "noul", "instructions": "Is this a warm-up?"}},
        },
        MODEL_NAME,
    )
    init_seconds = round(time.perf_counter() - t0, 2)
    log.info("model ready in %.1fs, n_embd=%d, threads=%d", init_seconds, engine.n_embd, THREADS)
    return engine, init_seconds


def _exit_if_failed(future: Future) -> None:
    # A failed background load would fail every request; exit so Lambda replaces the
    # environment, as it would after a failed init.
    if future.exception() is not None:
        log.error("model load failed", exc_info=future.exception())
        os._exit(1)


if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") == "on-demand":
    _ENGINE: Future = ThreadPoolExecutor(max_workers=1).submit(_load)
    _ENGINE.add_done_callback(_exit_if_failed)
else:  # snap-start, provisioned concurrency, or not on Lambda
    _ENGINE = Future()
    _ENGINE.set_result(_load())


def _cpu_part() -> str:
    """CPU part of the host serving this request.

    0xd0c is Graviton2, 0xd40 Graviton3 and 0xd4f Graviton4.

    Read per request, because a SnapStart snapshot can restore on either.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("CPU part"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


app = FastAPI(title="strands-decider on AWS Lambda (llama.cpp)")


@app.get("/health")
def health() -> dict:
    # Also the Lambda Web Adapter's readiness check, so it answers while the model loads.
    if not _ENGINE.done():
        return {"status": "loading", "model": MODEL_NAME, "cpu_part": _cpu_part()}
    _, init_seconds = _ENGINE.result()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "backend": "llama.cpp",
        "quant": "Q8_0",
        "threads": THREADS,
        "init_seconds": init_seconds,
        "max_request_tokens": MAX_REQUEST_TOKENS,
        "cpu_part": _cpu_part(),
    }


@app.post("/v1/systemone")
def systemone(request: dict) -> JSONResponse:
    engine, _ = _ENGINE.result()  # waits out a background load
    started = time.perf_counter()
    try:
        payload = engine.evaluate(request, MODEL_NAME)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    payload["cpu_part"] = _cpu_part()
    log.info(
        "latency_ms=%s cpu_part=%s input_tokens=%s",
        payload["latency_ms"],
        payload["cpu_part"],
        payload["usage"]["input_tokens"],
    )
    return JSONResponse(content=payload)
