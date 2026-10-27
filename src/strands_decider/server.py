"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import __version__
from .infer import EngineConfig, SystemOneEngine, load_mlx, load_neuron
from .modeling import StrandsDeciderModel
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None
_engine_lock = threading.Lock()


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion (issue #9) when
    two overlapping threads hit the same device. CUDA and CPU are thread-safe for
    inference, and MLX serialises on its own scheduler."""
    return str(device).startswith("mps")


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
    vision: bool = False,
    max_batch_tokens: int | None = None,
    neuron_cache: str | None = None,
) -> FastAPI:
    global _engine

    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        strict_window=strict_window,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    if vision:
        if device in ("mlx", "neuron"):
            raise ValueError(f"--vision runs on torch devices (cuda, mps, cpu), not {device}")
        from .vision import load_vision_engine

        _engine = load_vision_engine(checkpoint, config, attn_implementation=attn_implementation)
    elif device == "mlx":
        _engine = load_mlx(checkpoint, config)
    elif device == "neuron":
        neuron = load_neuron(checkpoint, config, cache_dir=neuron_cache)
        # Load every bucket's graph before the server reports healthy, not on a request. That takes
        # minutes per core, so say how far it has got.
        neuron.warm(lambda r: print(json.dumps({"warm": r}), flush=True))
        _engine = neuron
    else:
        model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
        _engine = SystemOneEngine(model, config)

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it.
    @app.get("/health")
    @app.get("/ping")
    def health() -> dict:
        eng = get_engine()
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            "base_model": eng.model.config.base_model,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": eng.cfg.device,
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
        }

    guard: contextlib.AbstractContextManager
    guard = _engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = get_engine()
        started = time.perf_counter()
        try:
            with guard:
                response = eng.evaluate(request)
        except ValueError as exc:
            # Malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


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
    vision: bool = False,
    max_batch_tokens: int | None = None,
    neuron_cache: str | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
        neuron_cache=neuron_cache,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
