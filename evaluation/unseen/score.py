"""Grade run.py's answers on the unseen-family set with JevBench v1.5's rules.

The same rules evaluation/jevbench/v15_proxy.py applies to JevBench results, here on our
own rows (a LOCAL measure, not a JevBench score):

  - yes/no: P(yes) strictly inside (0.2, 0.8) counts wrong; per task (right - 1/2) / (1/2);
  - choice: per task (right - 1/n) / (1 - 1/n), n options;
  - score: 100 * (1 - mean nMAE / mean nMAE_chance), graded by the expected level;
  - Intelligence: the mean of the three competences.

It also reports the top-probability ECE (10 bins), the yes/no answers inside the band, and
yes/no over-confidence (mean top probability minus accuracy), overall and per family. With
`--vs`, the runs given first are seeds of one arm and are compared row by row with the
`--vs` runs: the mean difference and a paired bootstrap 95% interval over the rows. One set
of 1,050 rows: differences of a few points are within noise; read the interval.

    python evaluation/unseen/score.py reports/unseen/NAME [...] [--vs reports/unseen/BASE ...] [--json out.json]

A run is run.py's output directory (or its predictions.jsonl). Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevbench.v15_proxy import (  # the rules, in one place
    BAND,
    credit,
    level_errors,
    paired_bootstrap,
    score_competence,
    yes_no_right,
)

Row = dict[str, Any]
KINDS = ("noul", "choice", "score")


def _top(p: list[float]) -> int:
    return max(range(len(p)), key=lambda i: p[i])


def right(r: Row) -> bool:
    """The argmax is gold (yes/no: also decisive, as v1.5 counts it)."""
    if r["kind"] == "noul":
        return yes_no_right(r["probs"][1], r["gold"] == 1)
    return _top(r["probs"]) == r["gold"]


def competence(rows: list[Row]) -> tuple[dict[str, float], float]:
    """Competence per question type present (in %), and Intelligence, their mean."""
    per = {}
    for k in KINDS:
        rs = [r for r in rows if r["kind"] == k]
        if not rs:
            continue
        if k == "score":
            per[k] = score_competence([level_errors(r["probs"], r["gold"]) for r in rs])
        else:
            per[k] = 100 * sum(credit(right(r), len(r["probs"])) for r in rs) / len(rs)
    return per, sum(per.values()) / len(per)


def ece(rows: list[Row], bins: int = 10) -> float:
    """Top-probability ECE: confidence the largest probability, right its argmax."""
    cells: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for r in rows:
        conf = max(r["probs"])
        cells[min(bins - 1, int(conf * bins))].append((conf, _top(r["probs"]) == r["gold"]))
    return sum(abs(sum(c for c, _ in x) - sum(k for _, k in x)) for x in cells if x) / max(len(rows), 1)


def in_band(r: Row) -> bool:
    return r["kind"] == "noul" and BAND[0] < r["probs"][1] < BAND[1]


def yes_no_overconfidence(rows: list[Row]) -> float | None:
    """Mean top probability minus accuracy (argmax) over the yes/no rows."""
    yn = [r for r in rows if r["kind"] == "noul"]
    if not yn:
        return None
    return sum(max(r["probs"]) - (_top(r["probs"]) == r["gold"]) for r in yn) / len(yn)


def summary(rows: list[Row]) -> dict[str, Any]:
    per, intelligence = competence(rows)
    over = yes_no_overconfidence(rows)
    out: dict[str, Any] = {
        "n": len(rows), "intelligence": round(intelligence, 1),
        "competence_by_type": {k: round(v, 1) for k, v in per.items()},
        "ece_top": round(ece(rows), 4), "yes_no_in_band": sum(map(in_band, rows)),
        "yes_no_overconfidence": None if over is None else round(over, 3),
    }
    tasks = sorted({r["task"] for r in rows})
    out["by_family"] = {t: {"n": len(rs), "competence": round(competence(rs)[1], 1),
                            "accuracy": round(sum(_top(r["probs"]) == r["gold"] for r in rs) / len(rs), 3)}
                        for t in tasks for rs in [[r for r in rows if r["task"] == t]]}
    return out


def _competence_of(kind: str) -> Callable[[list[Row]], float]:
    return lambda rows: competence(rows)[0].get(kind, 0.0)


# What `--vs` compares: for one run's rows, the number reported.
MEASURES: dict[str, Callable[[list[Row]], float]] = {
    "intelligence": lambda rows: competence(rows)[1],
    **{f"competence_{k}": _competence_of(k) for k in KINDS},
    "ece_top": ece,
}


def load(path: str) -> list[Row]:
    pred = os.path.join(path, "predictions.jsonl") if os.path.isdir(path) else path
    with open(pred, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Grade answers on the unseen-family set (a local measure).")
    ap.add_argument("runs", nargs="+", help="run.py output directories or predictions.jsonl files")
    ap.add_argument("--vs", nargs="*", default=[], help="baseline runs: compare the runs above with these")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--json", help="also write everything to this file")
    a = ap.parse_args(argv)
    loaded = {p: load(p) for p in a.runs + a.vs}
    report: dict[str, Any] = {"runs": {}}
    for p, rows in loaded.items():
        report["runs"][p] = summary(rows)
        print(p, json.dumps(report["runs"][p]))
    if a.vs:
        report["vs"] = paired_bootstrap([loaded[p] for p in a.runs], [loaded[p] for p in a.vs],
                                        resamples=a.resamples, measures=MEASURES, key="id")
        print("runs vs --vs:", json.dumps(report["vs"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
