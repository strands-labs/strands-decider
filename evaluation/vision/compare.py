"""Compare run.py runs item by item: the seeds of one arm against a baseline (v19, say).

For each image number of the image-trained models' evaluation (NaturalBench accuracy and
G-Acc, POPE accuracy, the exact Image JevBench preview items right, and the mean
confidence with the image removed), the mean over the arm's runs minus the mean over the
baseline's, with a paired bootstrap 95% interval: the units are resampled with
replacement (NaturalBench by group, so G-Acc stays defined; the others by item) and every
run is scored on the same sample.

    python evaluation/vision/compare.py ARM_RUN [ARM_RUN ...] --vs BASE_RUN [--json out.json]

A run is run.py's output directory (strands-v19.jsonl and strands-v19-blind.jsonl).
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Callable
from typing import Any

Rows = list[dict[str, Any]]


def _read(path: str) -> Rows:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _right(r: dict[str, Any]) -> bool:
    return max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"]


def load(run: str, tag: str = "strands-v19") -> dict[str, Rows]:
    """Rows by benchmark ("<bench>" with the image, "<bench>-blind" without)."""
    out: dict[str, Rows] = {}
    for suffix in ("", "-blind"):
        path = os.path.join(run, f"{tag}{suffix}.jsonl")
        if os.path.exists(path):
            for r in _read(path):
                out.setdefault(r["bench"] + suffix, []).append(r)
    return out


def _all_four(rs: Rows) -> tuple[float, float]:
    """A NaturalBench group counts for G-Acc when all four of its answers are right."""
    return float(len(rs) == 4 and all(map(_right, rs))), float(len(rs) == 4)


# (name, rows it reads, the unit resampled, a unit's (numerator, denominator), "mean" or "sum")
METRICS: list[tuple[str, str, str, Callable[[Rows], tuple[float, float]], str]] = [
    ("naturalbench_acc", "naturalbench", "group", lambda rs: (sum(map(_right, rs)), len(rs)), "mean"),
    ("naturalbench_g_acc", "naturalbench", "group", _all_four, "mean"),
    ("pope_acc", "pope_adversarial", "id", lambda rs: (sum(map(_right, rs)), len(rs)), "mean"),
    ("ijb_exact_right", "ijb_preview", "id", lambda rs: (sum(_right(r) for r in rs if r.get("exact")), 1), "sum"),
    ("blind_conf_naturalbench", "naturalbench-blind", "id", lambda rs: (sum(max(r["probs"]) for r in rs), len(rs)),
     "mean"),
    ("blind_conf_pope", "pope_adversarial-blind", "id", lambda rs: (sum(max(r["probs"]) for r in rs), len(rs)),
     "mean"),
]


def _diff(cells: list[dict[Any, tuple[float, float]]], n_arm: int, how: str, sample: list[Any]) -> float:
    """The arm's mean minus the baseline's on one sample of units (`cells`: arm runs first)."""
    vals = []
    for c in cells:
        num, den = sum(c[k][0] for k in sample), sum(c[k][1] for k in sample)
        vals.append(num if how == "sum" else num / den if den else 0.0)  # den 0: no complete group
    return sum(vals[:n_arm]) / n_arm - sum(vals[n_arm:]) / (len(vals) - n_arm)


def paired_bootstrap(arm: list[dict[str, Rows]], base: list[dict[str, Rows]], resamples: int = 10_000,
                     seed: int = 0) -> dict[str, dict[str, Any]]:
    """Mean over `arm` minus mean over `base` per metric, with a paired bootstrap 95% interval."""
    out: dict[str, dict[str, Any]] = {}
    rng = random.Random(seed)
    for name, bench, unit, per_unit, how in METRICS:
        if not all(bench in run for run in arm + base):
            continue
        cells = []  # per run: {unit: (numerator, denominator)}
        for run in arm + base:
            units: dict[Any, Rows] = {}
            for r in run[bench]:
                units.setdefault(r[unit], []).append(r)
            cells.append({k: per_unit(v) for k, v in units.items()})
        keys = sorted(cells[0], key=str)
        if any(sorted(c, key=str) != keys for c in cells):
            raise ValueError(f"{bench}: the runs do not cover the same items")

        draws = sorted(_diff(cells, len(arm), how, rng.choices(keys, k=len(keys))) for _ in range(resamples))
        out[name] = {"diff": round(_diff(cells, len(arm), how, keys), 4),
                     "ci95": [round(draws[int(0.025 * resamples)], 4), round(draws[int(0.975 * resamples) - 1], 4)]}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Paired comparison of run.py runs (an arm's seeds against a baseline).")
    ap.add_argument("runs", nargs="+", help="the arm's run.py output directories")
    ap.add_argument("--vs", nargs="+", required=True, help="the baseline's run.py output directories")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--json", help="also write the result to this file")
    a = ap.parse_args(argv)
    result = paired_bootstrap([load(r) for r in a.runs], [load(r) for r in a.vs], a.resamples)
    for name, v in result.items():
        print(f"{name:<26} {v['diff']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
