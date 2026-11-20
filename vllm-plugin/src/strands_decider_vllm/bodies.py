"""Request bodies under three limits, so that large, slow or vanished clients can't hold the
server's memory: a size (HTTP 413), a budget of bytes that large bodies buffer at once (503 after a
wait), and a time without a chunk (408)."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request

#: Bodies up to this size are read outside the budget.
_SMALL_BODY = 2**20


class BodyReader:
    def __init__(self, max_body: int, max_buffered: int, timeout: float) -> None:
        self.max_body, self.timeout = max_body, timeout
        self._budget = _Budget(max_buffered)

    @asynccontextmanager
    async def read(self, request: Request) -> AsyncIterator[bytearray]:
        """The body, whose share of the budget is held until the block exits. A body declared over
        1 MiB reserves its size before it's read; one without a length reserves max_body once past
        1 MiB, and gives back what it doesn't use once read."""
        declared = request.headers.get("content-length")
        size = int(declared) if declared is not None else None  # the HTTP parser has checked it
        if size is not None and size > self.max_body:
            raise self._too_large()
        reserved = 0
        try:
            if size is not None and size > _SMALL_BODY:
                reserved = await self._reserve(size)
            body = bytearray()
            async for chunk in self._chunks(request):
                body += chunk
                if len(body) > self.max_body:
                    raise self._too_large()
                if not reserved and len(body) > _SMALL_BODY:
                    reserved = await self._reserve(self.max_body)
            if size is None and reserved > len(body):
                await self._budget.give(reserved - len(body))
                reserved = len(body)
            yield body
        finally:
            if reserved:
                await self._budget.give(reserved)

    async def _reserve(self, n: int) -> int:
        try:
            async with asyncio.timeout(self.timeout):
                await self._budget.take(n)
        except TimeoutError:
            raise HTTPException(
                503, "busy with other large requests; retry", headers={"Retry-After": "5"}
            ) from None
        return n

    async def _chunks(self, request: Request) -> AsyncIterator[bytes]:
        chunks = request.stream()
        while True:
            try:
                async with asyncio.timeout(self.timeout):
                    chunk = await anext(chunks)
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise HTTPException(
                    408, f"no part of the body arrived for {self.timeout:g} s (--body-read-timeout)"
                ) from None
            yield chunk

    def _too_large(self) -> HTTPException:
        return HTTPException(413, f"the body is over {self.max_body} bytes (--max-body-mb)")


class _Budget:
    """Bytes granted in arrival order, so that a large body isn't starved by smaller ones."""

    def __init__(self, capacity: int) -> None:
        self._free = capacity
        self._queue: collections.deque[object] = collections.deque()
        self._changed = asyncio.Condition()

    async def take(self, n: int) -> None:
        async with self._changed:
            turn = object()
            self._queue.append(turn)
            try:
                await self._changed.wait_for(lambda: self._queue[0] is turn and self._free >= n)
                self._free -= n
            finally:  # taken, cancelled or timed out: the next in line may go
                self._queue.remove(turn)
                self._changed.notify_all()

    async def give(self, n: int) -> None:
        async with self._changed:
            self._free += n
            self._changed.notify_all()
