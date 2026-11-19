"""The server's request path on a real engine, in process, with no socket.

    python engine_checks.py [--repeat-tol T] serve CKPT --max-length L [other serve flags]

The engine is built from the flags as ``strands-decider-vllm serve`` builds it. PASS/FAIL lines:

1. the default context is --max-length + 1;
2. a request of exactly --max-length tokens is answered, and one token more is refused (HTTP 422);
3. vLLM itself refuses a prompt that fills the context: why the default context is one longer;
4. a request the engine core refuses raises VLLMValidationError (HTTP 422), the engine serves on,
   and the first request asked again gets its answer within T;
5. for a checkpoint that takes images, image requests over HTTP: vLLM's image preprocessing (the
   plugin's processor, which vLLM's renderer calls) runs off the event loop while /ping answers.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import uuid

FAILURES: list[str] = []


def check(ok, what):
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


def request_of_length(tokenizer, length):
    """A one-question request body whose prompt is exactly ``length`` tokens, and its rendering."""
    from strands_decider import core

    def render(state):
        body = {"state": state, "questions": {"q": {"type": "noul", "instructions": "A refund?"}}}
        rendered = core.render(core.Request.model_validate(body), tokenizer)
        return body, rendered, len(rendered.questions["q"].ids)

    lo, hi = 0, length  # the most words that fit, then a few tokens more
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if render("ticket " * mid)[2] <= length else (lo, mid - 1)
    for tail in ("", ".", " a", ","):
        for k in range(8):
            body, rendered, n = render("ticket " * lo + tail * k)
            if n == length:
                return body, rendered
    raise SystemExit(f"could not render a request of exactly {length} tokens")


async def text_checks(d, length, max_model_len, repeat_tol):
    import json

    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError

    from strands_decider_vllm import server

    body, rendered = request_of_length(d.s.tokenizer, length)
    out = await d.decide(json.dumps(body).encode())
    check(
        out["usage"]["input_tokens"] == length,
        f"a request of --max-length = {length} tokens is answered",
    )
    try:
        await d.decide(json.dumps({**body, "state": body["state"] + " ticket"}).encode())
        check(False, "a request one token longer is refused")
    except server.Refused as e:
        check(
            "exceeds the limit" in str(e), f"a request one token longer is refused: {str(e)[:90]}"
        )

    q = rendered.questions["q"]

    async def generate(ids, extra_args):
        sp = SamplingParams(
            max_tokens=1,
            logprobs=2,
            allowed_token_ids=list(q.code_token_ids),
            extra_args=extra_args,
        )
        async for _ in d.engine.generate(
            {"prompt_token_ids": ids}, sp, f"check-{uuid.uuid4().hex}"
        ):
            pass

    for ids, extra_args, want, what in [
        (
            list(q.ids) + [q.ids[-1]] * (max_model_len - length),
            {"prefix_end": q.state_len},
            "maximum model length",
            f"vLLM refuses a prompt of max_model_len = {max_model_len} tokens",
        ),
        (
            list(q.ids),
            {},
            "strands-decider",
            "a request the engine core refuses raises VLLMValidationError (HTTP 422)",
        ),
    ]:
        try:
            await generate(ids, extra_args)
            check(False, what)
        except VLLMValidationError as e:
            check(want in str(e), f"{what}: {str(e)[:90]}")
    again = await d.decide(json.dumps(body).encode())  # from the prefix cache: other kernels
    dp = abs(again["answers"]["q"]["noul"] - out["answers"]["q"]["noul"])
    check(
        dp <= repeat_tol,
        f"the engine serves on, with the same answer: |dp| {dp:.4f} (bound {repeat_tol})",
    )


def photo(seed):
    import random

    from make_requests import photo  # this directory's

    return photo(random.Random(seed), 1600, 1200, "PNG")


async def images_off_the_loop(d, requests=8, per_request=2):
    import httpx

    from strands_decider_vllm import processing, server

    loop_thread, calls = threading.get_ident(), []
    cls = processing.ResizedImageProcessor
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
            "state": "Photos.",
            "images": [photo(100 * i + k) for k in range(per_request)],
            "questions": question,
        }
        for i in range(requests)
    ]
    pings = []
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
        f"{requests} requests of {per_request} new photos answered over HTTP",
    )
    check(
        bool(calls) and all(t != loop_thread for t, _ in calls),
        f"vLLM's image preprocessing ran off the event loop: {len(calls)} processor calls, none on its thread",
    )
    worst = max((dt for dt, _ in pings), default=float("inf"))
    check(
        bool(pings) and all(st == 200 for _, st in pings) and worst < 1.0,
        f"/ping answered {len(pings)} times meanwhile, the slowest in {worst * 1000:.0f} ms",
    )


def main():
    from strands_decider_vllm.cli import ENVIRONMENT, parser

    for key, value in ENVIRONMENT.items():
        os.environ.setdefault(key, value)
    from vllm.v1.engine.async_llm import AsyncLLM

    from strands_decider_vllm import server

    argv, repeat_tol = sys.argv[1:], 0.02  # ci.sh passes its per-device repeat tolerance
    if argv[:1] == ["--repeat-tol"]:
        repeat_tol, argv = float(argv[1]), argv[2:]
    a = parser().parse_args(argv)
    s = server.settings(a)
    max_model_len = s.engine_args.max_model_len
    check(
        a.max_model_len is not None or max_model_len == s.max_length + 1,
        f"the default context is --max-length + 1: {max_model_len}",
    )
    d = server.Decider(AsyncLLM.from_engine_args(s.engine_args), s)

    async def checks():  # one event loop: AsyncLLM's output handler lives on the loop it started on
        await text_checks(d, s.max_length, max_model_len, repeat_tol)
        if s.takes_images:
            await images_off_the_loop(d)
        else:
            print("SKIP  images off the loop: the checkpoint takes no images", flush=True)

    try:
        asyncio.run(checks())
    finally:
        d.close()
        d.engine.shutdown()
    print(f"engine checks: {'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
