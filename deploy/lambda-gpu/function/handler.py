"""strands-decider on GPU-accelerated Lambda Managed Instances (README.md).

GPU-accelerated LMI is Event Functions only. This handler accepts a direct Invoke payload or
an event-source record (SQS, SNS, EventBridge) and forwards it to the strands-decider inference
server that the image's bootstrap script started during initialization.

One server per execution environment serves every concurrent invocation. Lambda forks
PerExecutionEnvironmentMaxConcurrency worker processes per execution environment, and each
process imports handler.py and connects to the same local server; the model is never loaded
inside a worker. The server handles request-level concurrency in FastAPI, bounded by
`DECIDER_MAX_CONCURRENCY` (`--max-concurrency` on `strands-decider serve`).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("decider")
log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

PORT = os.environ.get("DECIDER_SERVER_PORT", "8000")
BASE_URL = f"http://127.0.0.1:{PORT}"


def _record_body(record: dict[str, Any]) -> bytes:
    """A System One request as JSON bytes, for a single SQS/SNS record."""
    body = record.get("body") or record.get("Sns", {}).get("Message")
    if isinstance(body, str):
        return body.encode()
    if isinstance(body, dict):
        return json.dumps(body).encode()
    raise ValueError(f"record has no usable body: keys={sorted(record)}")


def _request_body(event: Any) -> bytes:
    """A System One request as JSON bytes, for non-Records event shapes."""
    if isinstance(event, (bytes, bytearray)):
        return bytes(event)
    if isinstance(event, str):
        return event.encode()
    if isinstance(event, dict):
        if "detail" in event and isinstance(event["detail"], dict):
            return json.dumps(event["detail"]).encode()
    return json.dumps(event).encode()


def _forward(body: bytes, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{BASE_URL}/v1/systemone",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # 4xx / 5xx: surface the server's error body (serve returns FastAPI's 422 for
        # validation errors and 500 for unexpected faults).
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"strands-decider serve returned {e.code}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"strands-decider serve unreachable at {BASE_URL}: {e.reason}") from e


def handler(event: Any, context: Any) -> dict[str, Any]:
    # Give the server until just before the invocation deadline to answer; no point waiting
    # past the point where Lambda will terminate the invocation anyway.
    remaining = getattr(context, "get_remaining_time_in_millis", lambda: 900_000)() / 1000
    timeout = max(1.0, remaining - 1.0)

    records = event.get("Records") if isinstance(event, dict) else None
    if not records:
        return _forward(_request_body(event), timeout=timeout)

    # SQS/SNS batch: process every record and report per-record failures so successful
    # records are deleted and only the failed ones are retried. Requires ReportBatchItemFailures
    # on the event-source mapping.
    failures: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    for record in records:
        item_id = record.get("messageId") or record.get("messageID")
        try:
            results.append(_forward(_record_body(record), timeout=timeout))
        except Exception:
            log.exception("record %s failed", item_id)
            if item_id:
                failures.append({"itemIdentifier": item_id})
            else:
                raise
    if failures:
        return {"batchItemFailures": failures}
    return {"results": results}
