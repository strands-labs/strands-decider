"""strands-decider on GPU-accelerated Lambda Managed Instances (README.md).

GPU-accelerated LMI is Event Functions only: this handler takes a direct Invoke payload, or an
event-source record (SQS, SNS, EventBridge), and returns the System One response. No Function
URL, no HTTP routes.

The model loads lazily on the first invocation of each execution environment, not at module
import. If the load fails at import, Lambda marks the version `Failed`/`RuntimeInitError` and
CloudWatch Logs often has nothing, so it is better to let the first invocation carry the import
and surface its traceback in the response.

The engine serializes on a threading.Lock; SystemOneEngine is not thread-safe on the same
device. `PerExecutionEnvironmentMaxConcurrency` > 1 then just queues on the lock, so keep it at
1 unless you add batching upstream.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

log = logging.getLogger("decider")
log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

CHECKPOINT = os.environ.get("DECIDER_CHECKPOINT", "/opt/model/checkpoint")
DEVICE = os.environ.get("DECIDER_DEVICE", "cuda")

_engine: Any = None
_lock = threading.Lock()


def _ensure_engine() -> Any:
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        from strands_decider.infer import load_engine
        from strands_decider.schema import SystemOneRequest

        started = time.perf_counter()
        engine = load_engine(CHECKPOINT, device=DEVICE)
        engine.evaluate(
            SystemOneRequest.model_validate(
                {"state": "Warm up.", "questions": {"q": {"type": "noul", "instructions": "?"}}}
            )
        )
        engine.cfg.max_request_tokens = (
            int(os.environ.get("DECIDER_MAX_REQUEST_TOKENS") or 0) or None
        )
        log.info("engine ready in %.1f s on %s", time.perf_counter() - started, DEVICE)
        _engine = engine
        return _engine


def _request_from(event: Any) -> Any:
    """Accept either a System One request directly, or an event-source wrapper (SQS/SNS records,
    EventBridge detail)."""
    from strands_decider.schema import SystemOneRequest

    if isinstance(event, dict):
        if event.get("Records"):
            record = event["Records"][0]
            body = record.get("body") or record.get("Sns", {}).get("Message")
            if isinstance(body, str):
                return SystemOneRequest.model_validate_json(body)
        if "detail" in event and isinstance(event["detail"], dict):
            return SystemOneRequest.model_validate(event["detail"])
    return SystemOneRequest.model_validate(event)


def handler(event: Any, context: Any) -> dict[str, Any]:
    engine = _ensure_engine()
    request = _request_from(event)
    started = time.perf_counter()
    with _lock:
        payload = engine.evaluate(request).model_dump()
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    log.info(
        "latency_ms=%s input_tokens=%s", payload["latency_ms"], payload["usage"]["input_tokens"]
    )
    return payload
