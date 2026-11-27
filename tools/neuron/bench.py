"""Latency on the Neuron backend: warm p50/p95 of one question at several state lengths, of 30
questions over one 1,024-token state, and of the blog's request through the engine; plus the load
time, the first request after load, each graph's first-call (compile or cache-load) time and each
rank's device-memory peak.

  WEIGHTS=<checkpoint dir> STRANDS_NEURON_TP=16 python tools/neuron/bench.py --out results/bench.json [--states 256,1024,...]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

T0 = time.time()

from strands_decider.core import Request  # noqa: E402
from strands_decider.decision import load_decision_engine  # noqa: E402

BLOG = {
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
}


def pct(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    return {
        "p50": round(statistics.median(s), 2),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s) + 0.5) - 1)], 2),
        "min": round(s[0], 2),
        "n": len(s),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default="256,1024,4096,16384,32768")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-length", type=int, default=32768)
    args = ap.parse_args()
    ckpt = os.environ["WEIGHTS"]
    eng = load_decision_engine(ckpt, device="neuron", max_length=args.max_length)
    be = eng.backend
    res: dict = {
        "load_and_warm_s": round(time.time() - T0, 1),
        "backend": be.describe(),
        "warm_s": getattr(be, "warm_s", None),
    }
    t = time.perf_counter()
    eng.evaluate(Request.model_validate(BLOG))
    res["first_request_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    blog = []
    for _ in range(args.reps):
        t = time.perf_counter()
        r = eng.evaluate(Request.model_validate(BLOG))
        blog.append((time.perf_counter() - t) * 1e3)
    res["blog_request_ms"] = pct(blog)
    res["blog_answer"] = r.model_dump()["answers"]

    # Real token ids: the longest golden text state, cut to each length, and one of its questions.
    lines = [json.loads(x) for x in open(Path(ckpt) / "golden_vectors.jsonl")]
    long = max((ln for ln in lines if not ln["images"]), key=lambda ln: len(ln["state_ids"]))
    ids = long["state_ids"]
    q = min((q["question_ids"] for q in long["questions"].values()), key=len)
    one = {}
    for n in map(int, args.states.split(",")):
        s = (ids * (n // len(ids) + 1))[: n - len(q)]
        t = time.perf_counter()
        be.forward(s, [q])
        first = (time.perf_counter() - t) * 1e3
        xs = []
        for _ in range(args.reps if n <= 8192 else max(3, args.reps // 3)):
            t = time.perf_counter()
            be.forward(s, [q])
            xs.append((time.perf_counter() - t) * 1e3)
        one[str(n)] = {
            **pct(xs),
            "first_ms": round(first, 1),
            "prompt_tokens": len(s) + len(q),
            "device_forward_s": round(be.last_timings.get("forward_s", 0), 4),
        }
        print("one question, prompt", n, one[str(n)], flush=True)
    res["one_question_ms"] = one

    qs = [
        x["question_ids"]
        for ln in lines
        if not ln["images"]
        for x in ln["questions"].values()
        if len(x["question_ids"]) <= 64
    ][:30]
    s = ids[:1024]
    t = time.perf_counter()
    be.forward(s, qs)
    first = (time.perf_counter() - t) * 1e3
    xs = []
    for _ in range(args.reps):
        t = time.perf_counter()
        be.forward(s, qs)
        xs.append((time.perf_counter() - t) * 1e3)
    res["thirty_questions_1024_state_ms"] = {
        **pct(xs),
        "first_ms": round(first, 1),
        "question_tokens": sum(map(len, qs)),
    }
    print("30 questions", res["thirty_questions_1024_state_ms"], flush=True)
    st = be.stats()
    res["peak_bytes_per_rank"] = [d.get("max_memory_allocated") for d in st]
    res["peak_reserved_per_rank"] = [d.get("max_memory_reserved") for d in st]
    res["compile_s_rank0"] = st[0]["compile_s"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "compile_s_rank0"}, indent=1))
    be.close()


if __name__ == "__main__":
    main()
