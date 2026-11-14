#!/usr/bin/env python3
"""Detailed warm-server benchmark across the decider's supported aspects.

Sections (skip with --skip name, repeatable; --only name runs just that one):
  length       question type x state length (short -> near the 3072-token JevBench window)
  qcount       questions per request (1 -> 500): the many-questions-are-cheap path
  options      choice width (2 -> 50 options): the pointer head scales with options
  mixed        mixed-type single request vs three single-type requests
  prefixcache  same state repeated (cache hit) vs distinct states (miss)
  concurrency  parallel HTTP clients (1 / 4 / 16): server throughput
  longstate    3072-token-class state, all three types, steady state

Every request goes over HTTP to a running server (start it with --device mlx).
20 unmeasured warmup requests first. Server-reported input_tokens are printed so
lengths are honest even though padding is char-approximate.

  .venv/bin/python bench/bench_all.py [--url http://127.0.0.1:8000]
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000"


# Deterministic filler, varied per section so no two states are byte-identical
# (the prefix cache keys on the state tokens).
def filler(words, salt):
    out, i = [], 0
    while len(out) < words:
        out.append(
            f"Section {i + salt}. Refunds and payout disputes are retried twice before "
            f"manual review {i}. Agents must confirm identity before changing payout "
            f"details and escalate disputes above ${1000 + i} to finance. "
        )
        i += 1
    text = " ".join(out)
    return text[: words * 6]  # ~6 chars/word average


def q_noul(i=0):
    return {"type": "noul", "instructions": f"Does this convey urgency? (v{i})"}


def q_choice(opts, i=0):
    return {
        "type": "choice",
        "instructions": f"Which applies best? (v{i})",
        "criteria": {o: None for o in opts},
    }


def q_score(rubric, i=0):
    return {"type": "score", "instructions": f"Where does this fall? (v{i})", "criteria": rubric}


RUBRIC3 = ["calm", "frustrated", "depressed"]
CHOICE5 = ["billing", "sales", "retail", "technical", "legal"]


def post(payload, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL + "/v1/systemone", data=body, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def timed(payload):
    """Returns (client wall ms, server latency_ms, input_tokens)."""
    t0 = time.perf_counter()
    r = post(payload)
    return (time.perf_counter() - t0) * 1000, r["latency_ms"], r["usage"]["input_tokens"]


def pct(sorted_vals, p):
    return sorted_vals[min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)]


HEADER = (
    f"{'scenario':<34} {'n':>5} {'p50 ms':>8} {'p95 ms':>8} "
    f"{'max ms':>8} {'srv ms':>10} {'tokens':>7} {'ms/q':>8}"
)


def row(label, wall_ms, server_ms=None, tokens=None, n=None):
    s = sorted(wall_ms)
    extra = f"{server_ms:>10.1f}" if server_ms is not None else f"{'-':>10}"
    tok = f"{tokens:>7}" if tokens else f"{'-':>7}"
    per = f"{statistics.fmean(s) / n:>8.2f}" if n else f"{'-':>8}"
    print(
        f"{label:<34} {len(s):>5} {pct(s, 50):>8.1f} {pct(s, 95):>8.1f} "
        f"{s[-1]:>8.1f} {extra} {tok}{per}"
    )


def sec_length(reps):
    print("\n== state length x type (single question per request) ==")
    print(HEADER)
    for words in (20, 200, 1000, 2000, 3500, 5000):
        state = filler(words, salt=words)
        for kind, mk in (
            ("noul", q_noul),
            ("choice", lambda i: q_choice(CHOICE5, i)),
            ("score", lambda i: q_score(RUBRIC3, i)),
        ):
            try:
                samples = [
                    timed({"state": state, "questions": {kind: mk(0)}}) for _ in range(reps)
                ]
            except urllib.error.HTTPError as e:
                print(f"{f'{kind} @{words}w':<34} HTTP {e.code}")
                continue
            wall = [w for w, _, _ in samples]
            srv = statistics.fmean([s for _, s, _ in samples])
            row(f"{kind} @{words}w", wall, srv, samples[0][2])


def sec_qcount(state_words=400):
    print(f"\n== questions per request (fixed ~{state_words}w state) ==")
    print(f"{'questions':>10} {'srv ms':>10} {'ms/q':>8} {'tokens':>7}")
    state = filler(state_words, salt=7)
    for n in (1, 5, 10, 50, 100, 500):
        qs = {f"q{i}": q_noul(i) for i in range(n)}
        _, srv, tok = timed({"state": state, "questions": qs})
        print(f"{n:>10} {srv:>10.1f} {srv / n:>8.2f} {tok:>7}")


def sec_options(reps=30):
    print("\n== choice width (options per choice question) ==")
    print(HEADER)
    state = filler(200, salt=11)
    pool = [f"opt{i}" for i in range(50)]
    for k in (2, 5, 10, 25, 50):
        opts = pool[:k]
        samples = [
            timed({"state": state, "questions": {"c": q_choice(opts)}}) for _ in range(reps)
        ]
        wall = [w for w, _, _ in samples]
        srv = statistics.fmean([s for _, s, _ in samples])
        row(f"{k} options", wall, srv, samples[0][2])


def sec_mixed():
    print("\n== mixed types: one request vs three ==")
    print(f"{'mode':>10} {'srv ms':>10} {'ms/q':>8}")
    state = filler(400, salt=13)
    trio = {"c": q_choice(CHOICE5), "n": q_noul(), "s": q_score(RUBRIC3)}
    _, srv_one, _ = timed({"state": state, "questions": trio})
    print(f"{'one req':>10} {srv_one:>10.1f} {srv_one / 3:>8.2f}")
    total = 0.0
    for name, q in trio.items():
        _, srv, _ = timed({"state": state, "questions": {name: q}})
        total += srv
    print(f"{'three req':>10} {total:>10.1f} {total / 3:>8.2f}")


def sec_prefixcache(reps=40):
    print("\n== state cache: populate (multi-q), then repeated single-q vs distinct ==")
    print(HEADER)
    same = filler(1000, salt=17)
    # A multi-question request populates the cross-request state cache (no extra forward);
    # single-question requests then reuse the snapshot.
    timed({"state": same, "questions": {"a": q_noul(), "b": q_choice(CHOICE5), "c": q_score(RUBRIC3)}})
    samples = [timed({"state": same, "questions": {"n": q_noul()}}) for _ in range(reps)]
    row(
        "same state, single q (hit)",
        [w for w, _, _ in samples],
        statistics.fmean([s for _, s, _ in samples]),
        samples[0][2],
    )
    walls, srvs, tok = [], [], None
    for i in range(reps):
        w, s, tok = timed(
            {"state": filler(1000, salt=100 + i), "questions": {"n": q_noul()}}
        )
        walls.append(w)
        srvs.append(s)
    row("distinct states (miss)", walls, statistics.fmean(srvs), tok)


def sec_concurrency(total=120):
    print("\n== concurrency (short states, one question each) ==")
    print(f"{'clients':>8} {'total req':>10} {'wall s':>8} {'req/s':>8} {'p50 ms':>8} {'p95 ms':>8}")
    payloads = [
        {"state": filler(50, salt=200 + i), "questions": {"n": q_noul(i)}} for i in range(total)
    ]
    for c in (1, 4, 16):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            lats = list(ex.map(lambda p: timed(p)[0], payloads))
        wall = time.perf_counter() - t0
        s = sorted(lats)
        print(f"{c:>8} {total:>10} {wall:>8.1f} {total / wall:>8.1f} {pct(s, 50):>8.1f} {pct(s, 95):>8.1f}")


def sec_longstate(reps=50):
    print("\n== 3072-token-class state, steady state ==")
    print(HEADER)
    state = filler(3500, salt=23)
    for kind, mk in (
        ("noul", q_noul),
        ("choice", lambda i: q_choice(CHOICE5, i)),
        ("score", lambda i: q_score(RUBRIC3, i)),
    ):
        samples = [timed({"state": state, "questions": {kind: mk(0)}}) for _ in range(reps)]
        wall = [w for w, _, _ in samples]
        srv = statistics.fmean([s for _, s, _ in samples])
        row(kind, wall, srv, samples[0][2])


SECTIONS = {
    "length": lambda: sec_length(reps=30),
    "qcount": sec_qcount,
    "options": sec_options,
    "mixed": sec_mixed,
    "prefixcache": sec_prefixcache,
    "concurrency": sec_concurrency,
    "longstate": sec_longstate,
}


def main():
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args()
    URL = args.url

    for i in range(20):  # warmup
        post({"state": filler(30, salt=i), "questions": {"w": q_noul(i)}})

    names = args.only or [n for n in SECTIONS if n not in args.skip]
    print(f"warm server {URL}; sections: {', '.join(names)}")
    for name in names:
        SECTIONS[name]()
    print("\ndone.")


if __name__ == "__main__":
    main()
