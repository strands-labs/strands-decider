#!/usr/bin/env python3
"""Shrink a run directory in place.

Drops `requests.jsonl` (one row per request — only latency.csv's aggregates and summary.json's
host info are read anywhere) and minifies `summary.json`. Keeps every row that is actually used
to plot, compare runs or quote numbers.

Request-level data goes if either file has errors: a run with `errors == 0` and `ok == n` on
every row has nothing to debug per-request. If a run has errors, those specific rows are kept —
rerun compact.py after fixing the issue to drop them.

    python benchmarks/compact.py benchmarks/runs/<host-slug>

Safe to re-run: it is idempotent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def run_had_errors(latency_csv: Path) -> bool:
    if not latency_csv.exists():
        return False
    with latency_csv.open() as f:
        for row in csv.DictReader(f):
            if (row.get("errors") or "0") != "0":
                return True
            if row.get("n") and row.get("ok") and row["n"] != row["ok"]:
                return True
    return False


def filter_requests_to_errors(path: Path) -> tuple[int, int, int]:
    before = path.stat().st_size
    kept = 0
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with path.open("w") as f:
        for r in rows:
            if r.get("status") != 200:
                f.write(json.dumps(r, separators=(",", ":")))
                f.write("\n")
                kept += 1
    return before, path.stat().st_size, kept


def compact_summary(path: Path) -> tuple[int, int]:
    before = path.stat().st_size
    with path.open() as f:
        d = json.load(f)
    with path.open("w") as f:
        json.dump(d, f, separators=(",", ":"))
    return before, path.stat().st_size


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: compact.py <run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])

    requests = run_dir / "requests.jsonl"
    if requests.exists():
        before = requests.stat().st_size
        if run_had_errors(run_dir / "latency.csv"):
            _, after, kept = filter_requests_to_errors(requests)
            print(f"requests.jsonl: {before:>8} → {after:>8} bytes  (kept {kept} error rows)")
        else:
            requests.unlink()
            print(f"requests.jsonl: {before:>8} → removed (no errors in latency.csv)")

    summary = run_dir / "summary.json"
    if summary.exists():
        before, after = compact_summary(summary)
        pct = 100.0 * (before - after) / before if before else 0.0
        print(f"summary.json:   {before:>8} → {after:>8} bytes  ({pct:4.1f}% saved)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
