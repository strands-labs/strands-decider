"""Paired bootstrap of two readings in a run.py output dir (default: served minus untrained).

    python evaluation/multimodal/paired.py results [--a served --b untrained] [--n 2000]

For each benchmark: accuracy, Brier and ECE of A minus B, with a 95% interval from resampling
items (the same items for both), and A's blind numbers (the medium removed) beside them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision"))
from metrics import summarise


def _load(out: str, tag: str) -> dict[str, dict[str, Any]]:
    with open(os.path.join(out, f"{tag}.jsonl")) as fh:
        return {r["id"]: r for r in map(json.loads, fh) if r["probs"] is not None}


def _stat(rows: list[dict[str, Any]]) -> dict[str, float]:
    s = summarise(rows)["all"]
    return {"accuracy": s["accuracy"], "brier": s["brier"], "ece": s["ece"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--a", default="served")
    ap.add_argument("--b", default="untrained")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    a, b = _load(args.out, args.a), _load(args.out, args.b)
    blind = _load(args.out, "blind") if os.path.exists(os.path.join(args.out, "blind.jsonl")) else {}
    rng = random.Random(0)
    report: dict[str, Any] = {}
    for bench in sorted({r["bench"] for r in a.values()}):
        ids = sorted(k for k, r in a.items() if r["bench"] == bench and k in b)
        if not ids:  # B has no reading here (the untrained 10-way speech choice)
            continue
        point = {m: _stat([a[i] for i in ids])[m] - _stat([b[i] for i in ids])[m] for m in ("accuracy", "brier", "ece")}
        draws: dict[str, list[float]] = {m: [] for m in point}
        for _ in range(args.n):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            sa, sb = _stat([a[i] for i in sample]), _stat([b[i] for i in sample])
            for m in point:
                draws[m].append(sa[m] - sb[m])
        ci = {m: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]) for m, v in draws.items()}
        report[bench] = {"n": len(ids), args.a: _stat([a[i] for i in ids]), args.b: _stat([b[i] for i in ids]),
                         "diff": {m: [round(point[m], 4), round(ci[m][0], 4), round(ci[m][1], 4)] for m in point}}
        if blind:
            report[bench]["blind"] = summarise([blind[i] for i in ids if i in blind])["all"]
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
