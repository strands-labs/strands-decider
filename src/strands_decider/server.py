"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .infer import EngineConfig, SystemOneEngine
from .modeling import StrandsDeciderModel
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None
# FastAPI runs a `def` handler on a thread pool, so two requests in flight reach the engine
# from two threads. One device cannot run two forward passes at once: on MPS the second one
# trips a Metal command-buffer assertion that kills the process. Serialise at the door.
_engine_lock = threading.Lock()


def get_engine() -> SystemOneEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
) -> FastAPI:
    global _engine

    app = FastAPI(
        title="strands-decider System One",
        version="0.1.0",
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    _engine = SystemOneEngine(
        model,
        EngineConfig(
            device=device, use_prefix_cache=use_prefix_cache, model_name=resolved_name,
            strict_window=strict_window, max_batch=max_batch,
        ),
    )

    @app.get("/health")
    def health() -> dict:
        eng = get_engine()
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            "base_model": eng.model.config.base_model,
            "num_slots": eng.model.config.num_slots,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        return JSONResponse(content=evaluate(request))

    return app


def evaluate(request: SystemOneRequest) -> dict:
    """One request through the engine, one at a time, timed; the body of ``POST /v1/systemone``."""
    eng = get_engine()
    started = time.perf_counter()
    try:
        with _engine_lock:
            response = eng.evaluate(request)
    except ValueError as exc:
        # Option count over num_slots, malformed permutation, etc. -- caller error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    payload = response.model_dump()
    payload["latency_ms"] = round(elapsed_ms, 2)
    return payload


def serve(
    checkpoint: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
