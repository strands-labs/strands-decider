"""The server's request path on a real engine, in process: the context's boundary, and what an engine refusal raises.

    python engine_checks.py serve CKPT --max-length L [any other strands-decider-vllm serve flags]

The engine is built as ``strands-decider-vllm serve`` builds it, from the same flags (``server.settings``), and no
socket is opened. Each check is a PASS/FAIL line:

1. the default context (``--max-model-len``) is ``--max-length`` plus 1;
2. a request rendered to exactly ``--max-length`` tokens is answered (the server's ``Decider``, on the engine);
3. a request one token longer is refused by the server (HTTP 422) before it reaches the engine;
4. vLLM itself refuses a prompt of exactly ``max_model_len`` tokens with ``VLLMValidationError``: the reason the
   default context is one token longer than the longest prompt;
5. a request the engine core refuses (no ``prefix_end``) makes ``AsyncLLM.generate`` raise ``VLLMValidationError``,
   which the server answers with HTTP 422 (``server.refusal``); the engine keeps serving;
6. with a checkpoint that takes images, a burst of image requests through the server's HTTP app (in process, on
   httpx's ASGI transport): vLLM's own image preprocessing (every call of the plugin's multimodal processor, which
   vLLM's renderer makes) runs on a thread other than the event loop's, and ``GET /ping`` keeps answering, timed
   throughout.
"""

from __future__ import annotations

import asyncio
import sys
import uuid

FAILURES: list[str] = []


def check(ok, what):
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILURES.append(what)
    return ok


def request_of_length(d, length: int) -> tuple[dict, object]:
    """A one-question request whose rendered prompt is exactly ``length`` tokens, and its rendering."""
    question = {"q": {"type": "noul", "instructions": "Is the ticket about a refund?"}}

    def render(state: str):
        body = {"state": state, "questions": question}
        r = d.core.render(body, d.tokenizer, d.config.prompt)
        return body, r, len(r.questions["q"].ids)

    words = lambda n: " ".join(["ticket"] * n)  # noqa: E731
    lo, hi = 0, length  # the most words that fit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if render(words(mid))[2] <= length else (lo, mid - 1)
    for tail in ("", ".", " a", ",", "!"):  # top up the last few tokens
        for k in range(0, 8):
            body, r, got = render(words(lo) + tail * k)
            if got == length:
                return body, r
    raise SystemExit(f"could not render a request of exactly {length} tokens")


async def run(d, length: int, max_model_len: int) -> None:
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError

    from strands_decider_vllm import PREFIX_END_KEY, server

    body, rendered = request_of_length(d, length)
    out = await d.decide(body)
    check(
        set(out["answers"]) == {"q"} and out["usage"]["input_tokens"] == length,
        f"a request of exactly --max-length = {length} tokens is answered",
    )
    longer = dict(body, state=body["state"] + " ticket")
    try:
        await d.decide(longer)
        check(False, "a request one token longer is refused by the server")
    except Exception as e:
        check(
            server.refusal(e) and "exceeds the limit" in str(e),
            f"a request one token longer is refused by the server: {type(e).__name__}: {str(e)[:100]}",
        )

    q = rendered.questions["q"]
    ids = list(q.ids)
    sp = dict(
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        logprobs=len(q.code_token_ids),
        allowed_token_ids=list(q.code_token_ids),
    )

    async def generate(prompt_ids, **over):
        params = SamplingParams(**{**sp, "extra_args": {PREFIX_END_KEY: q.state_len}, **over})
        rid = f"check-{uuid.uuid4().hex}"
        async for _ in d.engine.generate({"prompt_token_ids": prompt_ids}, params, rid):
            pass

    full = ids + ids[-1:] * (max_model_len - len(ids))
    try:
        await generate(full)
        check(False, f"vLLM refuses a prompt of exactly max_model_len = {max_model_len} tokens")
    except VLLMValidationError as e:
        check(
            "maximum model length" in str(e),
            f"vLLM refuses a prompt of exactly max_model_len = {max_model_len} tokens: {str(e)[:110]}",
        )
    try:
        await generate(ids, extra_args={})
        check(False, "a request the engine core refuses raises VLLMValidationError")
    except Exception as e:
        check(
            isinstance(e, VLLMValidationError)
            and "strands-decider" in str(e)
            and server.refusal(e),
            f"a request the engine core refuses raises {type(e).__name__} (HTTP 422): {str(e)[:100]}",
        )
    again = await d.decide(body)
    check(again["answers"] == out["answers"], "the engine serves on, with the same answer")


def photo(seed: int, w: int = 1600, h: int = 1200) -> str:
    """A synthetic photo as base64 PNG, different for every seed."""
    import base64
    import io
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(60):
        x, y = rng.randrange(w), rng.randrange(h)
        draw.rectangle(
            (x, y, x + rng.randrange(1, w // 3), y + rng.randrange(1, h // 3)),
            fill=tuple(rng.randrange(256) for _ in range(3)),
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def images_off_the_loop(d, requests: int = 8, per_request: int = 2) -> None:
    import threading
    import time

    import httpx

    from strands_decider_vllm import processing, server

    loop_thread = threading.get_ident()
    calls: list[tuple[int, float]] = []
    cls = processing.DeciderMultiModalProcessor
    original = cls.apply

    def apply(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            calls.append((threading.get_ident(), time.perf_counter() - t))

    cls.apply = apply
    app = server.build_app(None)
    app.state.decider = d
    question = {"q": {"type": "noul", "instructions": "Is the parcel damaged?"}}
    bodies = [
        {
            "state": "Photos of the parcel.",
            "images": [photo(100 * i + k) for k in range(per_request)],
            "questions": question,
        }
        for i in range(requests)
    ]
    pings: list[tuple[float, int]] = []
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://check", timeout=600
        ) as c:
            stop = asyncio.Event()

            async def pinger():
                while not stop.is_set():
                    t = time.perf_counter()
                    r = await c.get("/ping")
                    pings.append((time.perf_counter() - t, r.status_code))
                    await asyncio.sleep(0.005)

            ping = asyncio.ensure_future(pinger())
            rs = await asyncio.gather(*(c.post("/v1/systemone", json=b) for b in bodies))
            stop.set()
            await ping
    finally:
        cls.apply = original
    check(
        [r.status_code for r in rs] == [200] * requests,
        f"{requests} requests of {per_request} new photos each answered over HTTP: "
        f"{sorted({r.status_code for r in rs})}",
    )
    longest = max((dt for _, dt in calls), default=0.0)
    check(
        bool(calls) and all(t != loop_thread for t, _ in calls),
        f"vLLM's image preprocessing ran off the event loop: {len(calls)} processor calls, none on its thread "
        f"(the longest {longest * 1000:.0f} ms)",
    )
    worst = max((dt for dt, _ in pings), default=float("inf"))
    check(
        bool(pings) and all(st == 200 for _, st in pings) and worst < 1.0,
        f"/ping answered {len(pings)} times during the image requests, the slowest in {worst * 1000:.0f} ms",
    )


def main() -> None:
    from strands_decider_vllm import server
    from strands_decider_vllm.cli import parser

    a = parser().parse_args(sys.argv[1:])
    if a.command != "serve" or not a.max_length:
        raise SystemExit("usage: engine_checks.py serve CKPT --max-length L [flags]")
    server.prepare_environment()
    s = server.settings(a)
    max_model_len = s.engine_args.max_model_len
    check(
        max_model_len == a.max_length + 1 if a.max_model_len is None else True,
        f"the default context is --max-length + 1: {max_model_len}",
    )
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(s.engine_args)
    d = server.decider(engine, s, a)
    try:

        async def checks():  # one event loop: AsyncLLM's output handler lives on the loop it started on
            await run(d, a.max_length, max_model_len)
            if not d.takes_images:
                print("SKIP  images off the loop: the checkpoint takes no images", flush=True)
            elif d.image_problem:  # a checkpoint that should take images must not skip the check
                check(
                    False,
                    f"images off the loop: skipped, the server refuses images ({d.image_problem})",
                )
            else:
                await images_off_the_loop(d)

        asyncio.run(checks())
    finally:
        d.close()
        engine.shutdown()
    print(f"engine checks: {'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
