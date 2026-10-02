"""The two ways a tool reaches a decider, behind one small interface.

``HttpDecider`` talks to a running ``strands-decider serve`` over ``POST /v1/systemone``,
with ``urllib`` so the tools add no dependency beyond ``strands-agents``. ``LocalDecider``
wraps a ``SystemOneEngine`` in this process, so an agent can carry the model without a
server. Both return the package's own ``SystemOneResponse`` and keep the same tally.

Synchronous and serialised on purpose. The engine owns one device, and two forward passes
in flight at once crash it on MPS (a Metal command-buffer assertion takes the process down),
so each client holds a lock across ``ask``. A Strands agent runs independent tool calls
concurrently, which is exactly when this matters; the ``@tool`` decorator already moves a
plain ``def`` off the event loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..schema import Content, Question, SystemOneRequest, SystemOneResponse

if TYPE_CHECKING:
    from ..infer import SystemOneEngine

ENV_URL = "STRANDS_DECIDER_URL"
"""A running server. Checked before ``ENV_CHECKPOINT``."""

ENV_CHECKPOINT = "STRANDS_DECIDER_CHECKPOINT"
"""A checkpoint to load in this process when no server is named."""

DEFAULT_PORT = 8099
SERVE_HINT = (
    f"strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port {DEFAULT_PORT}"
)


class DeciderUnavailable(RuntimeError):
    """Nothing is answering. The message says how to start something."""


class DeciderEndpoint(Protocol):
    """What the tools need: one state, named typed questions, a typed response."""

    def ask(self, state: Content, questions: Mapping[str, Question]) -> SystemOneResponse: ...


@dataclass
class DeciderUsage:
    """What this process has asked so far. No cost column: the model is yours."""

    calls: int = 0
    questions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms_total: float = 0.0
    last_model: str | None = None
    last_latency_ms: float | None = None
    errors: int = 0

    def record(self, response: SystemOneResponse, latency_ms: float) -> None:
        self.calls += 1
        self.questions += len(response.answers)
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.latency_ms_total += latency_ms
        self.last_model = response.model
        self.last_latency_ms = round(latency_ms, 1)

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["latency_ms_total"] = round(self.latency_ms_total, 1)
        row["latency_ms_mean"] = (
            round(self.latency_ms_total / self.calls, 1) if self.calls else None
        )
        return row

    def reset(self) -> None:
        fresh = DeciderUsage()
        for name, value in asdict(fresh).items():
            setattr(self, name, value)

    def __str__(self) -> str:
        mean = f", {self.latency_ms_total / self.calls:.0f} ms mean" if self.calls else ""
        return (
            f"{self.calls} calls, {self.questions} questions, "
            f"{self.input_tokens} input tokens{mean}"
        )


class HttpDecider:
    """A client for ``strands-decider serve``. One request per ``ask``."""

    def __init__(self, url: str | None = None, *, timeout: float = 120.0) -> None:
        self.url = (url or os.environ.get(ENV_URL) or f"http://127.0.0.1:{DEFAULT_PORT}").rstrip(
            "/"
        )
        # The first request at a new input length pays a one-off shape compile on MPS.
        self.timeout = timeout
        self.usage = DeciderUsage()
        self._lock = threading.Lock()

    def _unavailable(self, exc: Exception) -> DeciderUnavailable:
        return DeciderUnavailable(
            f"no strands-decider at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def ask(self, state: Content, questions: Mapping[str, Question]) -> SystemOneResponse:
        request = SystemOneRequest(state=state, questions=dict(questions))
        body = request.model_dump_json(exclude_none=True).encode()
        http = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        with self._lock:
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(http, timeout=self.timeout) as reply:
                    payload = json.loads(reply.read())
            except urllib.error.HTTPError as exc:
                self.usage.errors += 1
                detail = exc.read().decode(errors="replace")
                raise ValueError(
                    f"the decider refused the request (HTTP {exc.code}): {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                self.usage.errors += 1
                raise self._unavailable(exc) from exc
            response = SystemOneResponse.model_validate(payload)
            latency = float(payload.get("latency_ms") or (time.perf_counter() - started) * 1000)
            self.usage.record(response, latency)
            return response

    def health(self) -> dict[str, Any]:
        """The loaded checkpoint, window and device. Raises if nothing is serving."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as reply:
                data: dict[str, Any] = json.loads(reply.read())
                return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise self._unavailable(exc) from exc

    def __repr__(self) -> str:
        return f"HttpDecider({self.url!r})"


class LocalDecider:
    """The engine in this process. ``load(checkpoint)`` builds one; ``__init__`` adopts one."""

    def __init__(self, engine: SystemOneEngine, *, checkpoint: str | None = None) -> None:
        self.engine = engine
        self.checkpoint = checkpoint
        self.usage = DeciderUsage()
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        checkpoint: str | None = None,
        *,
        device: str | None = None,
        use_prefix_cache: bool = True,
    ) -> LocalDecider:
        """Load a checkpoint (``STRANDS_DECIDER_CHECKPOINT`` when omitted) on the best device."""
        from ..cli import _auto_device
        from ..infer import load_engine

        checkpoint = checkpoint or os.environ.get(ENV_CHECKPOINT)
        if not checkpoint:
            raise DeciderUnavailable(f"no checkpoint given and {ENV_CHECKPOINT} is not set")
        engine = load_engine(
            checkpoint, device=device or _auto_device(), use_prefix_cache=use_prefix_cache
        )
        return cls(engine, checkpoint=checkpoint)

    def ask(self, state: Content, questions: Mapping[str, Question]) -> SystemOneResponse:
        request = SystemOneRequest(state=state, questions=dict(questions))
        with self._lock:
            started = time.perf_counter()
            try:
                response = self.engine.evaluate(request)
            except ValueError:
                self.usage.errors += 1
                raise
            self.usage.record(response, (time.perf_counter() - started) * 1000)
            return response

    def health(self) -> dict[str, Any]:
        """The same rows the server's ``/health`` reports, read off the engine."""
        config = self.engine.model.config
        return {
            "status": "ok",
            "model": self.engine.cfg.model_name,
            "checkpoint": self.checkpoint,
            "base_model": config.base_model,
            "num_slots": config.num_slots,
            "max_length": config.max_length,
            "temperature": config.temperature,
            "device": self.engine.cfg.device,
            "prefix_cache": self.engine.cfg.use_prefix_cache,
        }

    def __repr__(self) -> str:
        return f"LocalDecider({self.checkpoint or self.engine.cfg.model_name!r})"


def from_env() -> HttpDecider | LocalDecider:
    """A server if ``STRANDS_DECIDER_URL`` is set, else a checkpoint from ``STRANDS_DECIDER_CHECKPOINT``.

    Raises:
        DeciderUnavailable: Neither variable is set.
    """
    if os.environ.get(ENV_URL):
        return HttpDecider()
    if os.environ.get(ENV_CHECKPOINT):
        return LocalDecider.load()
    raise DeciderUnavailable(
        f"no decider configured. Set {ENV_URL} to a running server (start one with "
        f"`{SERVE_HINT}`), or {ENV_CHECKPOINT} to a checkpoint to load in this process, "
        "or call strands_decider.tools.configure(HttpDecider(...) | LocalDecider.load(...))."
    )
