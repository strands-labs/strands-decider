#!/usr/bin/env python3
"""Warm-server benchmark: 1000 tests per question type (noul, choice, score).

Two modes:
  --mode request   1000 HTTP requests, one question each (measures per-request latency)
  --mode question  1 HTTP request carrying 1000 questions of one type
                   (measures per-question cost when the state is read once)

Usage:
  .venv/bin/python bench/bench_types.py [--url http://127.0.0.1:8000] [--n 1000]
      [--mode request|question|both]

Warmup: 20 unmeasured requests first (model + HTTP stack).
"""

import argparse
import json
import statistics
import time
import urllib.request

STATES = [
    "Help! My payouts have been failing for 3 days!",
    "The quarterly report looks good, revenue is up 12%.",
    "sihamba ngokushesha",
    "Meeting moved to Thursday 3pm, please bring the slides.",
    "I cannot log in to my account, the password reset email never arrives.",
]

CHOICE_SETS = [
    ("Which team should handle this?", ["billing", "sales", "retail"]),
    ("What language is this phrase in?", ["English", "Zulu", "Dutch"]),
    ("Is the tone of this message formal or casual?", ["formal", "casual"]),
]

SCORE_RUBRICS = [
    ("How frustrated is the writer?", ["calm", "frustrated", "depressed"]),
    ("How urgent is this message?", ["routine", "soon", "urgent"]),
]


def question(kind, i):
    if kind == "noul":
        return {"type": "noul", "instructions": f"Does this convey urgency? (variant {i % 5})"}
    if kind == "choice":
        crit, opts = CHOICE_SETS[i % len(CHOICE_SETS)]
        return {"type": "choice", "instructions": crit, "criteria": {o: None for o in opts}}
    crit, rubric = SCORE_RUBRICS[i % len(SCORE_RUBRICS)]
    return {"type": "score", "instructions": crit, "criteria": rubric}


def post(url, payload, timeout=300):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/v1/systemone", data=body, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pct(sorted_ms, p):
    idx = min(int(len(sorted_ms) * p / 100), len(sorted_ms) - 1)
    return sorted_ms[idx]


def bench_request_mode(url, n, kind):
    lat = []
    for i in range(n):
        t0 = time.perf_counter()
        post(url, {"state": STATES[i % len(STATES)], "questions": {kind: question(kind, i)}})
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def bench_question_mode(url, n, kind):
    """One request, n questions of one kind. Server's own latency_ms is the honest number."""
    qs = {f"{kind}_{i}": question(kind, i) for i in range(n)}
    resp = post(url, {"state": STATES[0], "questions": qs})
    return resp["latency_ms"], len(resp["answers"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--mode", default="both", choices=["request", "question", "both"])
    args = ap.parse_args()

    for i in range(20):  # warmup
        post(args.url, {"state": STATES[i % len(STATES)], "questions": {"w": question("noul", i)}})

    print(f"benchmark: {args.n} tests per type, warm server {args.url}\n")

    if args.mode in ("request", "both"):
        print("mode: one question per HTTP request")
        print(f"{'type':<8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'max ms':>8} {'mean ms':>8}")
        for kind in ("noul", "choice", "score"):
            s = sorted(bench_request_mode(args.url, args.n, kind))
            print(
                f"{kind:<8} {pct(s, 50):>8.1f} {pct(s, 95):>8.1f} "
                f"{pct(s, 99):>8.1f} {s[-1]:>8.1f} {statistics.fmean(s):>8.1f}"
            )
        print()

    if args.mode in ("question", "both"):
        print("mode: many questions in one HTTP request (state read once)")
        print(f"{'type':<8} {'questions':>9} {'server ms':>10} {'ms/q':>8}")
        for kind in ("noul", "choice", "score"):
            server_ms, answered = bench_question_mode(args.url, args.n, kind)
            print(f"{kind:<8} {answered:>9} {server_ms:>10.1f} {server_ms / answered:>8.2f}")


if __name__ == "__main__":
    main()
