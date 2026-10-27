#!/usr/bin/env python3
"""Shrink a run directory in place.

`decider_bench.py` writes `requests.jsonl` with a few fields that are always null for the
server variants we measure (`roundtrip_ms`, `cpu`) and one that is redundant (`done` = `t` +
`client_ms` / 1000). It also pretty-prints `summary.json`. This script drops the null/redundant
per-row fields and minifies `summary.json`, cutting the run's size by about half with no loss
of information needed to re-aggregate or re-plot.

    python benchmarks/compact.py benchmarks/runs/<host-slug>

Safe to re-run: it is idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


DROP_FIELDS = ("roundtrip_ms", "cpu", "done")


def compact_requests(path: Path) -> tuple[int, int]:
    before = path.stat().st_size
    with path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        for k in DROP_FIELDS:
            r.pop(k, None)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")))
            f.write("\n")
    return before, path.stat().st_size


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
    for name, fn in (("requests.jsonl", compact_requests), ("summary.json", compact_summary)):
        p = run_dir / name
        if not p.exists():
            print(f"skipping {name}: not found")
            continue
        before, after = fn(p)
        pct = 100.0 * (before - after) / before if before else 0.0
        print(f"{name}: {before:>8} → {after:>8} bytes  ({pct:4.1f}% saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
