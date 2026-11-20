"""The limits on request bodies, over HTTP: their size, the budget that large bodies share, and the
time a client may take, so that large, slow or vanished clients can't hold the server's memory."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

QUESTIONS = {"q": {"type": "noul", "instructions": "Is it urgent?"}}


def test_a_body_over_the_limit_is_413_with_or_without_its_length(server):
    s = server("--max-body-mb", str(2048 / 2**20))
    big = json.dumps({"state": "x" * 3000, "questions": QUESTIONS})
    with s.client as c:
        for content in (big, (big[i : i + 500].encode() for i in range(0, len(big), 500))):
            r = c.post(
                "/v1/systemone", content=content, headers={"content-type": "application/json"}
            )
            assert r.status_code == 413 and "over 2048 bytes" in r.json()["detail"]
        assert s.engine.calls == [] and c.get("/health").json()["max_body_bytes"] == 2048


async def asgi_post(app, chunks, *, content_length=None, then="end"):
    """POST /v1/systemone straight to the ASGI app: ``chunks``, then the body's end, nothing ever
    again ("stall") or a disconnect. Returns (status, body)."""
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/systemone",
        "raw_path": b"/v1/systemone",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    queue, sent = list(chunks), []

    async def receive():
        if queue:
            more = bool(queue[1:]) or then != "end"
            return {"type": "http.request", "body": queue.pop(0), "more_body": more}
        if then == "disconnect":
            return {"type": "http.disconnect"}
        if then == "stall":
            await asyncio.Event().wait()
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent)


def large_body(n=2 * 2**20):
    return json.dumps({"state": "x" * n, "questions": QUESTIONS}).encode()


@pytest.fixture
def budgeted(server):
    """An app whose budget holds one 2.5 MiB body at a time."""

    def make(timeout="30"):
        flags = ("--max-body-mb", "2.5", "--max-buffered-mb", "1", "--body-read-timeout", timeout)
        return server(*flags, "--max-length", "10000000").app

    return make


def test_stalled_or_vanished_clients_give_their_share_of_the_budget_back(budgeted):
    # Two clients declare large bodies and send nothing more; a third leaves mid-body. Without the
    # timeouts they would hold the budget from every later large body.
    app = budgeted(timeout="0.3")

    async def run():
        declared = int(2.5 * 2**20)
        async with app.router.lifespan_context(app):
            stalled = await asyncio.gather(
                asgi_post(app, [], content_length=declared, then="stall"),
                asgi_post(app, [], content_length=declared, then="stall"),
            )
            gone = await asgi_post(
                app, [b'{"state": "' + b"x" * 2**20], content_length=declared, then="disconnect"
            )
            body = large_body()
            later = await asgi_post(app, [body], content_length=len(body))
        return stalled, gone, later

    stalled, (gone, detail), (later, _) = asyncio.run(run())
    assert sorted(status for status, _ in stalled) == [408, 503], "no chunk, then no room"
    assert gone == 400 and b"disconnected" in detail
    assert later == 200, "the budget is whole again"


def test_large_bodies_take_turns_with_or_without_a_length(budgeted):
    app = budgeted()
    body = large_body()

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://test") as c,
        ):
            declared = await asyncio.gather(
                *(c.post("/v1/systemone", content=body) for _ in range(3))
            )
            chunked = [body[i : i + 2**19] for i in range(0, len(body), 2**19)]
            unknown = await asyncio.gather(*(asgi_post(app, chunked) for _ in range(2)))
        return [r.status_code for r in declared], [status for status, _ in unknown]

    assert asyncio.run(run()) == ([200] * 3, [200] * 2)
