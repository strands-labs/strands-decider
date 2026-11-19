"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

from __future__ import annotations

import collections
import contextlib
import os
import threading
import time
from collections.abc import AsyncIterator, Callable

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .infer import EngineConfig, PreparedRequest, SystemOneEngine, load_mlx
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


class _Pending:
    __slots__ = ("done", "error", "prepared", "response", "t")

    def __init__(self, prepared: PreparedRequest, t: float):
        self.prepared = prepared
        self.t = t
        self.done = threading.Event()
        self.response: SystemOneResponse | None = None
        self.error: BaseException | None = None


class MicroBatcher:
    """Cross-request micro-batching: one thread owns the device and scores many requests
    per forward.

    Each request is prepared on its own handler thread (rendering, tokenisation, option
    positions: a request the engine refuses raises there and never reaches the queue), then
    queued. The device thread takes the oldest request, waits up to `wait_ms` from its
    arrival for more (not at all once the queue holds a forward's worth, so under load it
    never idles), takes queued requests in arrival order up to the engine's `batch_tokens`
    / `batch_rows`, and scores them with `evaluate_prepared`. Forwards run one at a time,
    so peak memory is bounded by the largest planned forward rather than by the number of
    requests in flight.
    """

    def __init__(self, engine: SystemOneEngine, wait_ms: float = 0.0):
        self.engine = engine
        self.wait_s = max(0.0, float(wait_ms)) / 1000.0
        self._queue: collections.deque[_Pending] = collections.deque()
        self._cv = threading.Condition()
        self.stats = {"cycles": 0, "requests": 0, "max_requests_per_cycle": 0}
        self._stop = False
        self._thread = threading.Thread(
            target=self._loop, name="strands-decider-batcher", daemon=True
        )
        self._thread.start()

    def submit(self, request: SystemOneRequest) -> SystemOneResponse:
        prepared = self.engine.prepare(request)  # ValueError -> the caller's 422
        item = _Pending(prepared, time.monotonic())
        with self._cv:
            self._queue.append(item)
            self._cv.notify()
        item.done.wait()
        if item.error is not None:
            raise item.error
        assert item.response is not None
        return item.response

    def _cost(self, item: _Pending) -> tuple[int, int]:
        p = item.prepared
        if p.rows is not None:
            return (p.tokens, len(p.rows))
        if p.prefix is not None:
            return (p.tokens, len(p.prefix.question_ids))
        return (0, 1)

    def _take(self) -> list[_Pending]:
        cfg = self.engine.cfg
        with self._cv:
            while not self._queue and not self._stop:
                self._cv.wait()
            if self._stop and not self._queue:
                return []
            deadline = self._queue[0].t + self.wait_s
            while True:
                tokens = rows = 0
                for it in self._queue:
                    t, r = self._cost(it)
                    tokens, rows = tokens + t, rows + r
                now = time.monotonic()
                if tokens >= cfg.batch_tokens or rows >= cfg.batch_rows or now >= deadline:
                    break
                self._cv.wait(timeout=deadline - now)
            batch: list[_Pending] = []
            tokens = rows = 0
            while self._queue:
                t, r = self._cost(self._queue[0])
                if batch and (tokens + t > cfg.batch_tokens or rows + r > cfg.batch_rows):
                    break
                batch.append(self._queue.popleft())
                tokens, rows = tokens + t, rows + r
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take()
            if not batch:
                return
            try:
                results = self.engine.evaluate_prepared([it.prepared for it in batch])
            except BaseException as exc:  # the device thread must survive any error
                results = [exc] * len(batch)
            self.stats["cycles"] += 1
            self.stats["requests"] += len(batch)
            self.stats["max_requests_per_cycle"] = max(
                self.stats["max_requests_per_cycle"], len(batch)
            )
            for it, res in zip(batch, results, strict=True):
                if isinstance(res, BaseException):
                    it.error = res
                else:
                    it.response = res
                it.done.set()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()


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
    batching: bool | None = None,
    batch_wait_ms: float = 0.0,
    batch_tune: bool | None = None,
    device_map: str | None = None,
    warmup: bool = True,
    build_engine: Callable[[], SystemOneEngine] | None = None,
    max_concurrency: int | None = None,
) -> FastAPI:
    if device_map and (vision or not device.startswith("cuda")):
        raise ValueError("--device-map splits the text torso across CUDA GPUs: use it with cuda")
    if vision and device == "mlx":
        raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    # One set of forward budgets: --max-batch and --max-batch-tokens bound a forward whether
    # its rows come from one request or from several. Unset, --max-batch-tokens leaves one
    # request's own forwards uncapped as before, and caps shared forwards at 16,384 tokens.
    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
        batch_rows=max_batch,
        batch_tokens=max_batch_tokens or EngineConfig.batch_tokens,
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
        # None: requests share anyio's default worker pool (40 threads), as sync handlers do.
        app.state.limiter = anyio.CapacityLimiter(max_concurrency) if max_concurrency else None

        # Batching defaults on for CUDA, where it was measured; MPS serialises anyway and MLX
        # / vision override the low-level forward so batching would only queue them.
        use_batching = batching
        if use_batching is None:
            use_batching = str(device).startswith("cuda") and engine.can_batch_rows()
        tune_at_startup = batch_tune
        if tune_at_startup is None:
            tune_at_startup = bool(use_batching) and str(device).startswith("cuda")
        if use_batching and tune_at_startup:
            import torch

            with torch.inference_mode():
                tuned = engine.tune_batching()
            print(
                f"[strands-decider] batching tuned: rows per forward by width {tuned}",
                flush=True,
            )
        app.state.batcher = MicroBatcher(engine, batch_wait_ms) if use_batching else None

        try:
            # Listener binds only after lifespan startup returns, so /health (/ping)
            # fails until this is done.
            if warmup:
                warm_up(engine, max_batch, vision=vision)
            yield
        finally:
            if app.state.batcher is not None:
                app.state.batcher.close()
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
    async def health(http: Request, eng: SystemOneEngine = Depends(get_engine)) -> dict:
        batcher = getattr(http.app.state, "batcher", None)
        batching_info = None
        if batcher is not None:
            batching_info = {
                "wait_ms": batch_wait_ms,
                "batch_tokens": eng.cfg.batch_tokens,
                "batch_rows": eng.cfg.batch_rows,
                "rows_by_width": {str(k): v for k, v in (eng.cfg.batch_rows_by_width or {}).items()}
                if eng.cfg.batch_rows_by_width
                else None,
                **batcher.stats,
            }
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
            "batching": batching_info,
        }

    def evaluate(
        eng: SystemOneEngine,
        request: SystemOneRequest,
        batcher: MicroBatcher | None,
    ) -> SystemOneResponse:
        with guard:
            if batcher is not None:
                return batcher.submit(request)
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
                evaluate,
                eng,
                request,
                http.app.state.batcher,
                limiter=http.app.state.limiter,
            )
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
    batching: bool | None = None,
    batch_wait_ms: float = 0.0,
    batch_tune: bool | None = None,
    device_map: str | None = None,
    warmup: bool = True,
    max_concurrency: int | None = None,
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
        batching=batching,
        batch_wait_ms=batch_wait_ms,
        batch_tune=batch_tune,
        device_map=device_map,
        warmup=warmup,
        max_concurrency=max_concurrency,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
