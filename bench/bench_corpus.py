#!/usr/bin/env python3
"""Benchmark over the real corpora in data/synthetic/*_eval.jsonl, through the live server.

Unlike bench_all.py's repeated filler paragraph, these are real documents (claims
adjudication, policies, contracts) with real questions and gold labels, so the run
measures latency on a realistic token distribution and reports accuracy alongside.

Sections:
  replay    one request per record (all misses unless states repeat) -- per-type
            latency, accuracy against the labels
  grouped   records grouped by state, one request per state with all its questions
            (the many-questions path), then a second pass over the same states
            (cross-request state cache hits)
  variants  instruction paraphrases: same state and question re-asked per variant,
            answer consistency and cache-hit latency

The record -> API question mapping lives in `record_questions` / `expected_answer`,
pure functions covered by tests/test_bench_corpus.py.

  .venv/bin/python bench/bench_corpus.py [--url http://127.0.0.1:8000] [--n 300]
"""

import argparse
import glob
import json
import os
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8000"
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

SCORE_RUBRICS = [  # ordered, ascending; score questions are built over real states
    ("How would a reader rate the clarity of this document?", ["confusing", "mixed", "clear"]),
    ("How formal is the register of this document?", ["casual", "neutral", "formal"]),
]


def load_records(limit=None):
    """Eval records from every data/synthetic/*_eval.jsonl, deterministically ordered."""
    records = []
    for path in sorted(glob.glob(os.path.join(DATA, "*_eval.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    if limit and len(records) > limit:
        step = len(records) / limit
        records = [records[int(i * step)] for i in range(limit)]
    return records


def record_questions(record, variant=0):
    """(state, {name: api question}) for one corpus record. `variant` picks an
    instruction paraphrase when the record carries them (variant 0 = the original)."""
    instructions = record["instructions"]
    variants = record.get("instruction_variants") or []
    if variant > 0 and variants:
        instructions = variants[(variant - 1) % len(variants)]
    if record["kind"] == "noul":
        return record["state"], {"q": {"type": "noul", "instructions": instructions}}
    criteria = {opt[0]: (opt[1] if len(opt) > 1 else None) for opt in record["options"]}
    return record["state"], {"q": {
        "type": "choice", "instructions": instructions, "criteria": criteria,
    }}


def expected_answer(record):
    """What a correct answer looks like: choice option text, or noul truthiness."""
    if record["kind"] == "noul":
        return bool(int(record["label"]))
    return record["options"][int(record["label"])][0]


def score_question(record, rubric_idx=0):
    """A score question over the record's real state; no gold label exists for these."""
    criteria, rubric = SCORE_RUBRICS[rubric_idx % len(SCORE_RUBRICS)]
    return record["state"], {"q": {"type": "score", "instructions": criteria, "criteria": rubric}}


def post(payload, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL + "/v1/systemone", data=body, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pct(sorted_vals, p):
    return sorted_vals[min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)]


def check(answer, record):
    """(correct?, confidence) for a corpus record's answer."""
    if record["kind"] == "noul":
        return (answer["noul"] > 0.5) == expected_answer(record), abs(2 * answer["noul"] - 1)
    return answer["choice"] == expected_answer(record), answer.get("confidence", 0.0)


def sec_replay(records):
    print("\n== replay: one request per record ==")
    print(f"{'type':<8} {'n':>5} {'p50 ms':>8} {'p95 ms':>8} {'acc':>6} {'conf|ok':>8} {'conf|bad':>8}")
    lat, ok, conf = {}, {}, {}
    for rec in records:
        state, questions = record_questions(rec)
        t0 = time.perf_counter()
        resp = post({"state": state, "questions": questions})
        ms = (time.perf_counter() - t0) * 1000
        good, c = check(resp["answers"]["q"], rec)
        lat.setdefault(rec["kind"], []).append(ms)
        ok.setdefault(rec["kind"], []).append(good)
        conf.setdefault(rec["kind"], []).append((c, good))
    for kind in sorted(lat):
        s = sorted(lat[kind])
        by_good = {}
        for c, good in conf[kind]:
            by_good.setdefault(good, []).append(c)
        conf_ok = statistics.fmean(by_good[True]) if by_good.get(True) else 0.0
        conf_bad = statistics.fmean(by_good[False]) if by_good.get(False) else 0.0
        print(f"{kind:<8} {len(s):>5} {pct(s, 50):>8.1f} {pct(s, 95):>8.1f} "
              f"{statistics.fmean(ok[kind]):>6.2f} {conf_ok:>8.2f} {conf_bad:>8.2f}")
    # score questions over the same real states (no gold label; latency only)
    s_lat = []
    for rec in records[:50]:
        state, questions = score_question(rec)
        t0 = time.perf_counter()
        post({"state": state, "questions": questions})
        s_lat.append((time.perf_counter() - t0) * 1000)
    s = sorted(s_lat)
    print(f"{'score':<8} {len(s):>5} {pct(s, 50):>8.1f} {pct(s, 95):>8.1f} "
          f"{'-':>6} {'-':>8} {'-':>8}")


def sec_grouped(records):
    print("\n== grouped by state: one request with all of a state's questions ==")
    # Group the FULL corpus, not the sample: a stride-sampled record has a unique state,
    # and single-question requests never populate the state cache (that needs a
    # multi-question request, which is exactly what this section sends).
    by_state = {}
    for rec in load_records():
        by_state.setdefault(rec["state"], []).append(rec)
    groups = [g for g in by_state.values() if len(g) > 1][:100]
    sizes = sorted(len(g) for g in groups)
    print(f"multi-question states: {len(groups)}, questions/state p50 {statistics.median(sizes):.0f}, max {sizes[-1]}")

    for label in ("pass 1 (populates cache)", "pass 2 (state-cache hit)"):
        lat, all_ok, per_q = [], [], []
        for group in groups:
            questions = {f"q{i}": record_questions(rec)[1]["q"] for i, rec in enumerate(group)}
            t0 = time.perf_counter()
            resp = post({"state": group[0]["state"], "questions": questions})
            lat.append((time.perf_counter() - t0) * 1000)
            per_q.append(lat[-1] / len(group))
            for i, rec in enumerate(group):
                good, _ = check(resp["answers"][f"q{i}"], rec)
                all_ok.append(good)
        s = sorted(lat)
        q = sorted(per_q)
        print(f"{label:<28} p50 {pct(s, 50):>8.1f} ms  p95 {pct(s, 95):>8.1f} ms  "
              f"ms/q {pct(q, 50):>6.1f}  acc {statistics.fmean(all_ok):.2f}")


def sec_variants(records):
    print("\n== instruction paraphrases: consistency across variants ==")
    with_variants = [r for r in records if r.get("instruction_variants")]
    if not with_variants:
        print("no records with instruction_variants in the sample")
        return
    consistent, lat, n = 0, 0.0, min(60, len(with_variants))
    for rec in with_variants[:n]:
        answers = []
        for variant in range(1 + min(2, len(rec["instruction_variants"]))):
            state, questions = record_questions(rec, variant=variant)
            t0 = time.perf_counter()
            resp = post({"state": state, "questions": questions})
            lat += (time.perf_counter() - t0) * 1000
            a = resp["answers"]["q"]
            answers.append(a["choice"] if rec["kind"] == "choice" else round(a["noul"] > 0.5))
        consistent += len(set(answers)) == 1
    print(f"records: {n}, same answer across variants: {consistent / n:.2f}, "
          f"mean latency (cache hits): {lat / (n * 3):.1f} ms")


def main():
    global URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--n", type=int, default=300, help="records sampled across the eval files")
    ap.add_argument("--skip", action="append", default=[])
    args = ap.parse_args()
    URL = args.url

    records = load_records(args.n)
    kinds = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{len(records)} records from data/synthetic/*_eval.jsonl, kinds {kinds}")

    for name, fn in (("replay", sec_replay), ("grouped", sec_grouped), ("variants", sec_variants)):
        if name not in args.skip:
            fn(records)
    print("\ndone.")


if __name__ == "__main__":
    main()
