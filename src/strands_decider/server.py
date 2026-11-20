"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import __version__
from .core import Request as DecisionRequest
from .decision import DecisionEngine, is_model_checkpoint, load_decision_engine
from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel, refuse_token_readout
from .refusals import EngineBusy, RequestRefused
from .schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

logger = logging.getLogger(__name__)


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
    max_concurrency: int | None = None,
    dtype: str | None = None,
    queue_timeout: float = 60.0,
    revision: str | None = None,
    max_length: int | None = None,
) -> FastAPI:
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    # A model checkpoint gets its own app; anything else takes v1's path below, unchanged.
    # Tests that inject a v1 engine (`build_engine`) skip the check.
    if build_engine is None and is_model_checkpoint(checkpoint, revision=revision):
        if device_map:
            raise ValueError(
                "--device-map is only supported for v1 model checkpoints; this one runs on one device"
            )
        if vision:
            warnings.warn(
                "--vision is only supported for v1 model checkpoints; this one takes images when its "
                "decision_config.json inputs list them",
                stacklevel=2,
            )
        engine = load_decision_engine(
            checkpoint,
            device=device,
            dtype=dtype,
            reuse_state=use_prefix_cache,
            max_batch=max_batch,
            max_batch_tokens=max_batch_tokens,
            max_length=max_length,
            model_name=resolved_name,
            max_concurrency=max_concurrency or 1,
            queue_timeout=queue_timeout,
            revision=revision,
        )
        if warmup:  # before uvicorn binds the listener
            _warm_up_decision(engine)
        return _decision_app(engine, checkpoint)
    for flag, value in (("--dtype", dtype), ("--revision", revision), ("--max-length", max_length)):
        if value is not None:
            raise ValueError(f"{flag} is not supported for v1 model checkpoints")
    if device_map and (vision or not device.startswith("cuda")):
        raise ValueError("--device-map splits the text torso across CUDA GPUs: use it with cuda")
    if vision and device == "mlx":
        raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")

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
        refuse_token_readout(model.config)
        return SystemOneEngine(model, config)

    # Tests inject a stub via `build_engine=...` to skip model loading. Construction
    # still runs under the lifespan, so a half-built engine is still closed.
    build = build_engine or _default_build_engine

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build()
        app.state.engine = engine
        # None: requests share anyio's default worker pool (40 threads), as sync handlers do.
        app.state.limiter = anyio.CapacityLimiter(max_concurrency) if max_concurrency else None
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

    def evaluate(eng: SystemOneEngine, request: SystemOneRequest) -> SystemOneResponse:
        with guard:
            return eng.evaluate(request)

    # The engine runs in a worker thread. With --max-concurrency N, at most N requests run at
    # once and the rest wait on the event loop, holding no thread and no GPU memory.
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    async def systemone(
        request: SystemOneRequest, http: Request, eng: SystemOneEngine = Depends(get_engine)
    ) -> JSONResponse:
        started = time.perf_counter()
        try:
            response = await anyio.to_thread.run_sync(
                evaluate, eng, request, limiter=http.app.state.limiter
            )
        except ValueError as exc:
            # Malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def _warm_up_decision(engine: DecisionEngine) -> None:
    """A request with a question of each kind, and one with an image if images are served, so
    that the first requests don't pay for lazy initialisation. A warm-up prompt longer than the
    served length is skipped."""
    state = "A warm-up request, answered before the server accepts requests."
    questions = {
        "noul": {"type": "noul", "instructions": "Is this a warm-up?"},
        "choice": {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"a": None, "b": None},
        },
        "score": {"type": "score", "instructions": "How warm is it?", "criteria": ["cold", "warm"]},
    }
    requests = [{"state": state, "questions": questions}]
    if engine.image_problem is None:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
        image = base64.b64encode(buf.getvalue()).decode("ascii")
        requests.append(
            {"state": state, "questions": {"noul": questions["noul"]}, "images": [image]}
        )
    started = time.perf_counter()
    for body in requests:
        try:
            engine.evaluate(DecisionRequest.model_validate(body))
        except RequestRefused as e:
            logger.warning("a warm-up request was skipped: %s", e)
    logger.debug("warmed up in %.1f s", time.perf_counter() - started)


def _decision_app(engine: DecisionEngine, checkpoint: str) -> FastAPI:
    """A model checkpoint on v1's paths, with v1's responses. The engine bounds its own
    concurrency, after its refusals and with a timeout (503), and its backend takes turns on MPS,
    so the app adds no limiter and no lock."""
    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )
    app.state.engine = engine

    @app.get("/health")
    @app.get("/ping")
    async def health() -> dict:
        return {"status": "ok", "checkpoint": checkpoint, **engine.describe()}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    async def systemone(http: Request) -> JSONResponse:
        try:
            # The core's request model, not v1's: v1's would drop `audio`, which the core
            # refuses, and require a `state`.
            request = DecisionRequest.model_validate_json(await http.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        started = time.perf_counter()
        try:
            response = await anyio.to_thread.run_sync(engine.evaluate, request)
        except RequestRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EngineBusy as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        payload = response.model_dump()
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(content=payload)

    return app
