"""GPU checks of the plugin's server (``strands-decider-vllm serve``) against a running instance. Needs Pillow.

- ``images``: text and image requests are answered in v1's format; one image, two images and four images; refusals
  (five images, bad base64, a file that is not an image, a body that is not JSON) get HTTP 400 and the server keeps
  serving; the image-content cache test, on exact counts of cached tokens: after a request with one photo, a request
  with a different photo of the same size, the same text and the same question reuses no cached token, a further
  question on the first photo reuses its state's full blocks, and repeats reuse all but their last token's;
  ``--text-only URL`` checks that a server of a text-only checkpoint refuses images with the
  specification's message and still answers text.
- ``textonly``: only that last check, against ``--url``.
- ``answers``: every request of ``--requests`` (System One requests, images inline) at ``--concurrency``; writes each
  question's code probabilities to ``--out``; with ``--expect`` (a ``reference.py`` output) reports top-1 agreement
  and mean and max |dp| and gates on ``--min-top1``, ``--max-mean-dp`` and ``--max-dp``.

  python server_client.py images --url http://127.0.0.1:8200 --out images.json
  python server_client.py answers --url http://127.0.0.1:8200 --requests req.jsonl --expect ref.jsonl --out ans.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request

FAILURES: list[str] = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILURES.append(what)
    return ok


def post(url, body, timeout=900):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/systemone", data=data, headers={"Content-Type": "application/json"}
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail")
        except Exception:
            detail = None
        return e.code, detail, time.perf_counter() - t
    except Exception as e:
        return None, repr(e)[:300], time.perf_counter() - t


def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as r:
            return r.status
    except Exception as e:
        return repr(e)[:100]


def photo(seed, w=640, h=480, fmt="PNG"):
    """A synthetic photo: smooth color fields and noise, different for every seed, the same size for a given w, h."""
    import random

    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        d.ellipse(
            (x0, y0, x0 + rng.randrange(w // 2), y0 + rng.randrange(h // 2)),
            fill=tuple(rng.randrange(256) for _ in range(3)),
        )
    img = img.filter(ImageFilter.GaussianBlur(3))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


QUESTIONS = {
    "color": {
        "type": "choice",
        "instructions": "What is the dominant color of the first image?",
        "criteria": {"red": None, "green": None, "blue": None, "other": None},
    },
    "busy": {"type": "noul", "instructions": "Is the image busy?"},
    "quality": {
        "type": "score",
        "instructions": "How sharp is it?",
        "criteria": ["blurry", "fair", "sharp"],
    },
}


def vector(answer):
    """The code-order probabilities of one v1 answer: a choice's in request order, a noul's [false, true], a score's
    by level (rounded to 4 decimals, as v1 rounds)."""
    if "noul" in answer:
        return [1.0 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def dmax(p, q):
    return max(abs(a - b) for a, b in zip(p, q, strict=True))


def scenario_images(a):
    url = a.url
    st, r, _ = post(
        url, {"state": "A support ticket about a late delivery.", "questions": QUESTIONS}
    )
    check(
        st == 200 and set(r["answers"]) == set(QUESTIONS),
        f"text request on the image checkpoint: HTTP {st}",
    )
    for n in (1, 2, 4):
        body = {
            "state": f"{n} photos.",
            "images": [photo(100 + i, 320 + 48 * i, 240) for i in range(n)],
            "questions": QUESTIONS,
        }
        st, r, dt = post(url, body)
        ok = (
            st == 200
            and set(r["answers"]) == set(QUESTIONS)
            and r["usage"]["input_tokens"] > 250 * n
        )
        check(
            ok, f"{n} image(s): HTTP {st} in {dt * 1000:.0f} ms, {r['usage'] if st == 200 else r}"
        )
    good = {"state": "x", "images": [photo(1)], "questions": {"busy": QUESTIONS["busy"]}}
    for name, body, want in [
        (
            "five images",
            {"state": "x", "images": [photo(i, 96, 96) for i in range(5)], "questions": QUESTIONS},
            "5 images; this server takes at most 4",
        ),
        (
            "bad base64",
            {"state": "x", "images": ["@@@"], "questions": QUESTIONS},
            "image is not valid base64",
        ),
        (
            "not an image",
            {
                "state": "x",
                "images": [base64.b64encode(b"%PDF-1.4 hello").decode()],
                "questions": QUESTIONS,
            },
            "image could not be read as PNG/JPEG/WEBP/GIF",
        ),
        (
            "images not a list",
            {"state": "x", "images": "abc", "questions": QUESTIONS},
            "images must be a list",
        ),
        ("not JSON", b"{not json", "not JSON"),
    ]:
        st, r, _ = post(url, body)
        check(st == 400 and want in str(r), f"{name}: HTTP {st} ({str(r)[:120]})")
        check(health(url) == 200 and post(url, good)[0] == 200, f"{name}: the server keeps serving")

    # The image-content cache test, on exact counts of cached tokens. Two different photos of the same size have the
    # same token ids. vLLM reuses a prompt's whole blocks of A tokens: for an exact repeat of an L-token prompt all
    # but the last token's, floor((L - 1) / A) A; for a further question on a cached state of S tokens the state's,
    # floor(S / A) A (the plugin keys the blocks inside [0, S) by the state and its images alone). A is the KV block
    # size, or a multiple of it when the model's layers have KV pages of different sizes (vLLM aligns cache hits to
    # their least common multiple: 32 tokens for E2B with 16-token blocks); the counts must fit one A.
    B = a.block_size
    text = {"note": "listing photos", "items": [f"item {i} looks fine" for i in range(60)]}
    q1 = {"busy": QUESTIONS["busy"]}
    q2 = {"color": QUESTIONS["color"]}
    first, other = photo(7001, 640, 480), photo(7002, 640, 480)
    st1, r1, _ = post(url, {"state": text, "images": [first], "questions": q1})
    st2, r2, _ = post(url, {"state": text, "images": [other], "questions": q1})
    st3, r3, _ = post(url, {"state": text, "images": [first], "questions": q2})
    st4, r4, _ = post(url, {"state": text, "images": [first], "questions": q1})
    st5, r5, _ = post(url, {"state": text, "images": [other], "questions": q1})
    # Both questions at once: input_tokens counts the state once, so S = L1 + L2 - L12.
    st6, r6, _ = post(url, {"state": text, "images": [first], "questions": {**q1, **q2}})
    if not check(
        st1 == st2 == st3 == st4 == st5 == st6 == 200,
        f"cache test requests answered: HTTP {[st1, st2, st3, st4, st5, st6]}",
    ):
        return {}
    u = [r["usage"] for r in (r1, r2, r3, r4, r5, r6)]
    L1, L2 = u[0]["input_tokens"], u[2]["input_tokens"]
    S = L1 + L2 - u[5]["input_tokens"]
    counts = [u[i]["cached_tokens"] for i in (2, 3, 4, 5)]

    def expected(A):
        r1_, r2_ = (L1 - 1) // A * A, (L2 - 1) // A * A
        return [S // A * A, r1_, r1_, r1_ + r2_]

    B = next((A for A in (B, 2 * B, 4 * B) if expected(A) == counts), B)
    print(f"      cache hits aligned to {B} tokens; S = {S}, L = {L1} and {L2}", flush=True)
    repeat = lambda n: (n - 1) // B * B  # noqa: E731
    check(
        u[1]["input_tokens"] == L1, "two photos of the same size: the same number of prompt tokens"
    )
    check(
        u[1]["cached_tokens"] == 0,
        f"a different photo of the same size after the first: {u[1]['cached_tokens']} cached tokens (want 0)",
    )
    check(
        u[2]["cached_tokens"] == S // B * B,
        f"the same photo, a further question: {u[2]['cached_tokens']} cached tokens (want the state's full "
        f"blocks, {S // B * B} of S = {S})",
    )
    d4 = dmax(vector(r4["answers"]["busy"]), vector(r1["answers"]["busy"]))
    check(
        u[3]["cached_tokens"] == repeat(L1) and d4 <= a.same_tol,
        f"the same photo and question again: {u[3]['cached_tokens']} cached tokens (want {repeat(L1)}), "
        f"|dp| vs cold {d4:.2e} (tolerance {a.same_tol})",
    )
    d5 = dmax(vector(r5["answers"]["busy"]), vector(r2["answers"]["busy"]))
    check(
        u[4]["cached_tokens"] == repeat(L1) and d5 <= a.same_tol,
        f"the other photo again: {u[4]['cached_tokens']} cached tokens (want {repeat(L1)}), its own answer, "
        f"|dp| vs its cold one {d5:.2e}",
    )
    check(
        u[5]["cached_tokens"] == repeat(L1) + repeat(L2),
        f"both questions again: {u[5]['cached_tokens']} cached tokens (want {repeat(L1) + repeat(L2)})",
    )
    out = {
        "cache": {
            "state_tokens": S,
            "input_tokens": L1,
            "other_photo_cached": u[1]["cached_tokens"],
            "further_question_cached": u[2]["cached_tokens"],
            "repeat_cached": u[3]["cached_tokens"],
            "repeat_dp": d4,
            "other_repeat_dp": d5,
        }
    }
    if a.text_only:
        scenario_textonly(a.text_only)
    check(health(url) == 200, "server healthy at the end")
    return out


def scenario_textonly(url):
    st, r, _ = post(url, {"state": "x", "images": ["not even base64"], "questions": QUESTIONS})
    check(
        st == 400
        and r
        == 'this checkpoint does not take images: its decision_config.json inputs are ["text"]',
        f"text-only checkpoint: images refused before decoding (HTTP {st})",
    )
    check(
        post(url, {"state": "x", "questions": QUESTIONS})[0] == 200,
        "text-only checkpoint: text answered",
    )
    return {}


def scenario_answers(a):
    reqs = [json.loads(x) for x in open(a.requests) if x.strip()][: a.n]
    results: dict[int, tuple] = {}
    lock = threading.Lock()
    todo = list(range(len(reqs)))

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            body = {k: v for k, v in reqs[i].items() if k != "files"}
            results[i] = post(a.url, body)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    ok = [i for i in range(len(reqs)) if results[i][0] == 200]
    check(
        len(ok) == len(reqs),
        f"{len(ok)} of {len(reqs)} requests answered in {wall:.1f} s "
        f"({len(reqs) / wall:.2f} requests/s at concurrency {a.concurrency})",
    )
    lat = sorted(results[i][2] for i in ok)
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
    summary = {
        "requests": len(reqs),
        "answered": len(ok),
        "wall_s": wall,
        "requests_per_s": len(reqs) / wall,
        "p50_s": lat[len(lat) // 2] if lat else None,
        "p90_s": lat[int(len(lat) * 0.9)] if lat else None,
    }
    if a.expect:
        ref = {(r["request"], r["question"]): r for r in map(json.loads, open(a.expect))}
        agree, dps, by_images = 0, [], {}
        for i in ok:
            for name, ans in results[i][1]["answers"].items():
                p, q = vector(ans), ref[(i, name)]["probabilities"]
                hit = argmax(p) == argmax(q)
                agree += hit
                dps.append(dmax(p, q))
                k = ref[(i, name)]["images"]
                by_images.setdefault(k, [0, 0])
                by_images[k][0] += hit
                by_images[k][1] += 1
        top1 = agree / len(dps)
        summary.update(
            {
                "questions": len(dps),
                "top1": top1,
                "mean_dp": sum(dps) / len(dps),
                "max_dp": max(dps),
                "top1_by_images": {k: v[0] / v[1] for k, v in sorted(by_images.items())},
            }
        )
        check(
            top1 >= a.min_top1
            and summary["mean_dp"] <= a.max_mean_dp
            and summary["max_dp"] <= a.max_dp,
            f"reference: top-1 {top1:.4f} over {len(dps)} questions (gate {a.min_top1}), "
            f"mean |dp| {summary['mean_dp']:.4f} (gate {a.max_mean_dp}), max |dp| {summary['max_dp']:.4f} "
            f"(gate {a.max_dp})",
        )
    print(json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenario", choices=["images", "answers", "textonly"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--text-only", help="the URL of a server of a text-only checkpoint (images)")
    ap.add_argument("--requests")
    ap.add_argument("--expect")
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-top1", type=float, default=0.98)
    ap.add_argument("--max-mean-dp", type=float, default=1.0, help="answers: the gate on mean |dp|")
    ap.add_argument("--max-dp", type=float, default=1.0, help="answers: the gate on max |dp|")
    ap.add_argument(
        "--block-size", type=int, default=16, help="the server's KV block size (images)"
    )
    ap.add_argument(
        "--same-tol",
        type=float,
        default=0.01,
        help="images: max |dp| of a cached answer against the same request's cold one",
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    fn = {
        "images": scenario_images,
        "answers": scenario_answers,
        "textonly": lambda a: scenario_textonly(a.url),
    }
    out = fn[a.scenario](a)
    if a.out and a.scenario == "images":
        json.dump({**out, "failures": FAILURES}, open(a.out, "w"))
    print(f"{a.scenario}: {'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
