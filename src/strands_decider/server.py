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
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import __version__
from .infer import EngineConfig, PreparedRequest, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None
_engine_lock = threading.Lock()


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion when two overlapping
    threads hit the same device. CUDA and CPU are thread-safe for inference, and the MLX
    engine runs every evaluation on a single worker thread."""
    return str(device).startswith("mps")


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
        return (p.tokens, len(p.rows)) if p.rows is not None else (0, 1)

    def _take(self) -> list[_Pending]:
        cfg = self.engine.cfg
        with self._cv:
            while not self._queue:
                self._cv.wait()
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


class _Pending:
    __slots__ = ("done", "error", "prepared", "response", "t")

    def __init__(self, prepared: PreparedRequest, t: float):
        self.prepared = prepared
        self.t = t
        self.done = threading.Event()
        self.response: SystemOneResponse | None = None
        self.error: BaseException | None = None


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
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
    batching: bool | None = None,
    batch_wait_ms: float = 0.0,
    batch_tune: bool | None = None,
) -> FastAPI:
    global _engine

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # The MLX engine owns a worker thread (see mlx_engine.MLXEngine); other engines
            # have nothing to close.
            close = getattr(_engine, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
        lifespan=lifespan,
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    # One set of forward budgets: --max-batch and --max-batch-tokens bound a forward whether
    # its rows come from one request or from several. Unset, --max-batch-tokens leaves one
    # request's own forwards uncapped, as before, and caps shared forwards at 16,384 tokens.
    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
        batch_rows=max_batch,
        batch_tokens=max_batch_tokens or EngineConfig.batch_tokens,
    )
    if vision:
        if device == "mlx":
            raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")
        from .vision import load_vision_engine

        _engine = load_vision_engine(checkpoint, config, attn_implementation=attn_implementation)
    elif device == "mlx":
        _engine = load_mlx(checkpoint, config)
    else:
        model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
        _engine = SystemOneEngine(model, config)

    # With batching, every device call goes through one micro-batching thread (MicroBatcher):
    # concurrent requests share forwards, and forwards run one at a time. It is the default
    # on CUDA, where it was measured; MPS serialises requests anyway (issue #9), and the MLX
    # and vision engines score each request alone, so batching would only queue them.
    # The tune (on with batching on CUDA) times forwards of growing size at startup and caps
    # shared forwards at the fastest: it pays on short prompts, where small forwards are
    # launch-bound, and keeps long-prompt forwards small, which also keeps answers closer to
    # unbatched ones.
    if batching is None:
        batching = str(device).startswith("cuda") and _engine.can_batch_rows()
    if batch_tune is None:
        batch_tune = bool(batching) and str(device).startswith("cuda")
    if batching and batch_tune:
        import torch

        with torch.inference_mode():
            tuned = _engine.tune_batching()
        print(f"[strands-decider] batching tuned: rows per forward by width {tuned}", flush=True)
    batcher = MicroBatcher(_engine, batch_wait_ms) if batching else None
    app.state.batcher = batcher

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
            "batching": None
            if batcher is None
            else {
                "wait_ms": batch_wait_ms,
                "batch_tokens": eng.cfg.batch_tokens,
                "batch_rows": eng.cfg.batch_rows,
                "rows_by_width": eng.cfg.batch_rows_by_width,
                **batcher.stats,
            },
        }

    guard: contextlib.AbstractContextManager
    guard = _engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = get_engine()
        started = time.perf_counter()
        try:
            if batcher is not None:
                response = batcher.submit(request)
            else:
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
    batching: bool | None = None,
    batch_wait_ms: float = 0.0,
    batch_tune: bool | None = None,
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
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
