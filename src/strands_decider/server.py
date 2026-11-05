"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import AsyncIterator, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel
from .schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion when two overlapping
    threads hit the same device. CUDA and CPU are thread-safe for inference, and the MLX
    engine runs every evaluation on a single worker thread."""
    return str(device).startswith("mps")


def warm_up(engine: SystemOneEngine, max_batch: int, vision: bool = False) -> None:
    """Ask every question kind, then every number of questions up to `max_batch`, once.

    On CUDA the first forward of a shape compiles kernels (Triton's, on a Qwen3.5 torso), once
    for each number of questions, whatever the state's length. A request with more than
    `max_batch` questions runs in chunks of at most `max_batch`, so these shapes cover every
    request after this. With `vision`, one image request too: see `_warm_up_image`.
    """
    state = "A warm-up request, answered before the server accepts requests."
    kinds: dict[str, Question] = {
        "noul": NoulQuestion(instructions="Is this a warm-up?"),
        "choice": ChoiceQuestion(instructions="Pick one.", criteria={"a": None, "b": None}),
        "score": ScoreQuestion(instructions="How warm is it?", criteria=["cold", "warm"]),
    }
    started = time.perf_counter()
    for name, question in kinds.items():
        engine.evaluate(SystemOneRequest(state=state, questions={name: question}))
    engine.evaluate(SystemOneRequest(state=state, questions=kinds))
    for n in range(2, max_batch + 1):
        questions: dict[str, Question] = {f"q{i}": kinds["noul"] for i in range(n)}
        engine.evaluate(SystemOneRequest(state=state, questions=questions))
    if vision:
        _warm_up_image(engine, state, kinds["noul"])
    print(f"[strands-decider] warmed up in {time.perf_counter() - started:.1f} s")


def _warm_up_image(engine: SystemOneEngine, state: str, question: Question) -> None:
    """One image request: the first one initialises the image path; after it, image requests
    reuse the text path's kernels, whatever the number of questions or images. The image is
    small, so the request fits any window.
    """
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
    image = base64.b64encode(buf.getvalue()).decode()
    engine.evaluate(SystemOneRequest(state=state, questions={"q": question}, images=[image]))


def create_app(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    device_map: str | None = None,
    warmup: bool = True,
    build_engine: Callable[[], SystemOneEngine] | None = None,
) -> FastAPI:
    if device_map and (vision or not device.startswith("cuda")):
        raise ValueError("--device-map splits the text torso across CUDA GPUs: use it with cuda")
    if vision and device == "mlx":
        raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )

    def _default_build_engine() -> SystemOneEngine:
        if vision:
            from .vision import load_vision_engine

            return load_vision_engine(checkpoint, config, attn_implementation=attn_implementation)
        if device == "mlx":
            return load_mlx(checkpoint, config)
        model = StrandsDeciderModel.load(
            checkpoint, device_map=device_map, attn_implementation=attn_implementation
        )
        return SystemOneEngine(model, config)

    # Tests inject a stub via `build_engine=...` to skip model loading. Construction
    # still runs under the lifespan, so a half-built engine is still closed.
    build = build_engine or _default_build_engine

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build()
        app.state.engine = engine
        try:
            # Listener binds only after lifespan startup returns, so /health (/ping)
            # fails until this is done.
            if warmup:
                warm_up(engine, max_batch, vision=vision)
            yield
        finally:
            # The MLX engine owns a worker thread (see mlx_engine.MLXEngine); other engines
            # have nothing to close.
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
        lifespan=lifespan,
    )

    engine_lock = threading.Lock()
    guard: contextlib.AbstractContextManager
    guard = engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    # async, like /ping below: FastAPI runs a sync dependency in the worker thread pool too.
    async def get_engine(request: Request) -> SystemOneEngine:
        engine: SystemOneEngine = request.app.state.engine
        return engine

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it. /ping is async: a sync handler would wait for a free
    # thread in the pool that /invocations fills (40 threads), and SageMaker allows a
    # health check 2 s.
    @app.get("/health")
    @app.get("/ping")
    async def health(eng: SystemOneEngine = Depends(get_engine)) -> dict:
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

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(
        request: SystemOneRequest, eng: SystemOneEngine = Depends(get_engine)
    ) -> JSONResponse:
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
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    device_map: str | None = None,
    warmup: bool = True,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
        device_map=device_map,
        warmup=warmup,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
