"""Checks of a running ``strands-decider-vllm serve``. Needs Pillow.

- ``images``: text and image requests (one, two, four and five images) are answered; refusals get
  HTTP 422 and the server keeps serving; and the image-content cache test, on exact counts of cached
  tokens: after a request with one photo, one with a different photo of the same size (the same
  token ids) reuses nothing, a further question on the first photo reuses its state's full blocks,
  and repeats reuse all but their last token's. ``--text-only URL`` adds ``textonly``'s check.
- ``textonly``: a server of a text-only checkpoint refuses images and answers text.
- ``answers``: every request of ``--requests`` at ``--concurrency``, each question's probabilities
  to ``--out``; with ``--expect`` (reference.py's output) the agreement, gated.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

FAILURES: list[str] = []
QUESTIONS = {
    "color": {
        "type": "choice",
        "instructions": "What is the dominant color?",
        "criteria": {"red": None, "green": None, "blue": None, "other": None},
    },
    "busy": {"type": "noul", "instructions": "Is the image busy?"},
    "quality": {
        "type": "score",
        "instructions": "How sharp is it?",
        "criteria": ["blurry", "fair", "sharp"],
    },
}


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILURES.append(what)
    return ok


def post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/systemone", data=data, headers={"Content-Type": "application/json"}
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except Exception:
            detail = None
        return e.code, detail, time.perf_counter() - t
    except Exception as e:
        return None, repr(e)[:300], time.perf_counter() - t


def healthy(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def photo(seed, w=640, h=480):
    """A synthetic photo, different for every seed."""
    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randrange(w), rng.randrange(h)
        d.ellipse(
            (x, y, x + rng.randrange(w // 2), y + rng.randrange(h // 2)),
            fill=tuple(rng.randrange(256) for _ in range(3)),
        )
    buf = io.BytesIO()
    img.filter(ImageFilter.GaussianBlur(3)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def vector(answer):
    """An answer's probabilities in code order: a noul's (false, true), the others' as listed."""
    return (
        [1.0 - answer["noul"], answer["noul"]]
        if "noul" in answer
        else list(answer["probabilities"].values())
    )


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def dmax(p, q):
    return max(abs(a - b) for a, b in zip(p, q, strict=True))


def scenario_images(a):
    url = a.url
    st, r, _ = post(url, {"state": "A late delivery.", "questions": QUESTIONS})
    check(st == 200 and set(r["answers"]) == set(QUESTIONS), f"a text request: HTTP {st}")
    for n in (1, 2, 4, 5):
        body = {
            "state": f"{n} photos.",
            "images": [photo(100 + i, 320 + 48 * i, 240) for i in range(n)],
            "questions": QUESTIONS,
        }
        st, r, dt = post(url, body)
        check(
            st == 200 and r["usage"]["input_tokens"] > 250 * n,
            f"{n} images: HTTP {st} in {dt * 1000:.0f} ms",
        )
    good = {"state": "x", "images": [photo(1)], "questions": {"busy": QUESTIONS["busy"]}}
    for name, body, want in [
        (
            "bad base64",
            {"state": "x", "images": ["@@@"], "questions": QUESTIONS},
            "image is not valid base64",
        ),
        (
            "not an image",
            {
                "state": "x",
                "images": [base64.b64encode(b"%PDF-1.4").decode()],
                "questions": QUESTIONS,
            },
            "image could not be read",
        ),
        (
            "images not a list",
            {"state": "x", "images": "abc", "questions": QUESTIONS},
            "images: Input should be a valid array",
        ),
        ("not JSON", b"{not json", "Invalid JSON"),
    ]:
        st, r, _ = post(url, body)
        check(st == 422 and want in str(r), f"{name}: HTTP {st} ({str(r)[:100]})")
        check(healthy(url) and post(url, good)[0] == 200, f"{name}: the server keeps serving")

    # vLLM reuses a prompt's whole blocks of A tokens: an exact repeat of L tokens all but the last
    # token's, floor((L - 1) / A) A; a further question on a cached state of S tokens the state's,
    # floor(S / A) A. A is the block size, or a multiple when the layers' KV pages differ in size.
    text = {"note": "listing photos", "items": [f"item {i} looks fine" for i in range(60)]}
    q1, q2 = {"busy": QUESTIONS["busy"]}, {"color": QUESTIONS["color"]}
    first, other = photo(7001), photo(7002)
    rs = [
        post(url, {"state": text, "images": [im], "questions": q})
        for im, q in [
            (first, q1),
            (other, q1),
            (first, q2),
            (first, q1),
            (other, q1),
            (first, {**q1, **q2}),
        ]
    ]
    if not check(
        all(st == 200 for st, _, _ in rs), f"cache test requests: {[st for st, _, _ in rs]}"
    ):
        return {}
    u = [r["usage"] for _, r, _ in rs]
    L1, L2 = u[0]["input_tokens"], u[2]["input_tokens"]
    S = L1 + L2 - u[5]["input_tokens"]  # input_tokens counts a shared state once
    counts = [u[i]["cached_tokens"] for i in (2, 3, 4, 5)]

    def repeat(n, A):
        return (n - 1) // A * A

    B = a.block_size
    B = next(
        (
            A
            for A in (B, 2 * B, 4 * B)
            if counts == [S // A * A, repeat(L1, A), repeat(L1, A), repeat(L1, A) + repeat(L2, A)]
        ),
        B,
    )
    check(
        u[1]["input_tokens"] == L1 and u[1]["cached_tokens"] == 0,
        f"a different photo of the same size: {u[1]['cached_tokens']} cached tokens (want 0)",
    )
    check(
        u[2]["cached_tokens"] == S // B * B,
        f"a further question: {u[2]['cached_tokens']} cached tokens (want {S // B * B} of S = {S})",
    )
    for i, j, what in [
        (3, 0, "the same photo and question again"),
        (4, 1, "the other photo again"),
    ]:
        d = dmax(vector(rs[i][1]["answers"]["busy"]), vector(rs[j][1]["answers"]["busy"]))
        check(
            u[i]["cached_tokens"] == repeat(L1, B) and d <= a.same_tol,
            f"{what}: {u[i]['cached_tokens']} cached tokens (want {repeat(L1, B)}), |dp| vs cold {d:.2e}",
        )
    check(
        u[5]["cached_tokens"] == repeat(L1, B) + repeat(L2, B),
        f"both questions again: {u[5]['cached_tokens']} cached tokens",
    )
    if a.text_only:
        scenario_textonly(a.text_only)
    check(healthy(url), "server healthy at the end")
    return {"state_tokens": S, "cached": counts}


def scenario_textonly(url):
    st, r, _ = post(url, {"state": "x", "images": ["not even base64"], "questions": QUESTIONS})
    check(
        st == 422 and r == "this checkpoint does not take images",
        f"text-only checkpoint: images refused before decoding (HTTP {st})",
    )
    check(
        post(url, {"state": "x", "questions": QUESTIONS})[0] == 200,
        "text-only checkpoint: text answered",
    )
    return {}


def scenario_answers(a):
    reqs = [json.loads(x) for x in open(a.requests) if x.strip()][: a.n]
    results, todo, lock = {}, list(range(len(reqs))), threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            results[i] = post(a.url, reqs[i])

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [i for i in range(len(reqs)) if results[i][0] == 200]
    check(
        len(ok) == len(reqs),
        f"{len(ok)} of {len(reqs)} requests answered in {wall:.1f} s ({len(reqs) / wall:.2f} requests/s at concurrency {a.concurrency})",
    )
    with open(a.out, "w") as f:
        for i in ok:
            for name, ans in results[i][1]["answers"].items():
                f.write(
                    json.dumps(
                        {
                            "request": i,
                            "question": name,
                            "probabilities": vector(ans),
                            "latency_s": results[i][2],
                        }
                    )
                    + "\n"
                )
    lat = sorted(results[i][2] for i in ok)
    summary = {
        "answered": len(ok),
        "requests_per_s": len(reqs) / wall,
        "p50_s": lat[len(lat) // 2] if lat else None,
    }
    if a.expect:
        ref = {(r["request"], r["question"]): r for r in map(json.loads, open(a.expect))}
        hits, dps, by_images = [], [], {}
        for i in ok:
            for name, ans in results[i][1]["answers"].items():
                p, q = vector(ans), ref[(i, name)]["probabilities"]
                hits.append(argmax(p) == argmax(q))
                dps.append(dmax(p, q))
                by_images.setdefault(ref[(i, name)]["images"], []).append(hits[-1])
        summary.update(
            top1=sum(hits) / len(hits),
            mean_dp=sum(dps) / len(dps),
            max_dp=max(dps),
            top1_by_images={k: sum(v) / len(v) for k, v in sorted(by_images.items())},
        )
        check(
            summary["top1"] >= a.min_top1
            and summary["mean_dp"] <= a.max_mean_dp
            and summary["max_dp"] <= a.max_dp,
            f"reference: top-1 {summary['top1']:.4f} over {len(dps)} questions, mean |dp| {summary['mean_dp']:.4f}, max |dp| {summary['max_dp']:.4f}",
        )
    print(json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenario", choices=["images", "answers", "textonly"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--text-only", help="images: the URL of a text-only checkpoint's server")
    ap.add_argument("--requests")
    ap.add_argument("--expect")
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-top1", type=float, default=0.98)
    ap.add_argument("--max-mean-dp", type=float, default=1.0)
    ap.add_argument("--max-dp", type=float, default=1.0)
    ap.add_argument("--block-size", type=int, default=16, help="images: the KV block size")
    ap.add_argument(
        "--same-tol", type=float, default=0.01, help="images: a cached answer vs its cold one"
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    run = {
        "images": scenario_images,
        "answers": scenario_answers,
        "textonly": lambda a: scenario_textonly(a.url),
    }
    out = run[a.scenario](a)
    if a.out and a.scenario == "images":
        json.dump({**out, "failures": FAILURES}, open(a.out, "w"))
    print(f"{a.scenario}: {'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
