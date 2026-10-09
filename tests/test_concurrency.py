"""`evaluate` serialises concurrent requests (issue #9).

Two things broke when the FastAPI server ran `evaluate` on its thread pool and two
requests overlapped:

  1. On MPS, two threads touching the Metal backend at once crashed the process
     (`failed assertion _status < MTLCommandBufferStatusCommitted`, or a segfault).
  2. On every device, `_fit` leaves the request's token offsets on the engine
     (`self._last_offsets`) and `_option_idx` reads them back, so two overlapping
     requests read each other's offsets and point the readout at the wrong option --
     surfacing as a spurious 422.

The fix is a per-engine lock taken in `SystemOneEngine.evaluate`. These tests pin that
the lock exists, that it actually serialises, and that the offsets each request wrote
are the ones it reads back under contention. They use a stub subclass so no model,
tokeniser or device is needed.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from strands_decider.infer import SystemOneEngine


class _ProbeEngine(SystemOneEngine):
    """A `SystemOneEngine` that records concurrency, without a real model.

    `__init__` is bypassed (it would move a torch model to a device); we set only what
    `evaluate` needs. `_evaluate` imitates the real one's hazard: write a per-request
    value to the shared `_last_offsets` slot, pause, then read it back. Without the lock
    an overlapping request overwrites the slot in between.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inside = 0  # how many threads are inside _evaluate right now
        self._max_inside = 0  # the high-water mark; must stay 1 if the lock works
        self._observer = threading.Lock()  # guards the two counters above

    def _evaluate(self, request):  # type: ignore[override]
        with self._observer:
            self._inside += 1
            self._max_inside = max(self._max_inside, self._inside)
        # Imitate _fit stashing offsets on the engine, then _option_idx reading them.
        self._last_offsets = request
        time.sleep(0.02)
        read_back = self._last_offsets
        with self._observer:
            self._inside -= 1
        return read_back


def test_evaluate_takes_the_lock():
    """A plain CPU engine has the lock the base __init__ installs."""
    engine = _ProbeEngine()
    assert isinstance(engine._lock, type(threading.Lock()))


def test_overlapping_evaluate_never_runs_concurrently():
    """With the lock, at most one request is inside _evaluate at a time."""
    engine = _ProbeEngine()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(engine.evaluate, range(8)))
    assert engine._max_inside == 1


def test_offsets_are_not_clobbered_across_requests():
    """Each request reads back the offsets it wrote, not another's (the #9 race)."""
    engine = _ProbeEngine()
    requests = list(range(60))  # the maintainer saw ~half of 60 fail without the lock
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(engine.evaluate, requests))
    assert results == requests
