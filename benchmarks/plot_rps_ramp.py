#!/usr/bin/env python3
"""Open-loop ramp (goodput curve) chart for a single run directory.

Reads `latency.csv` and plots p95 request latency against target rps, one line per
(model, input_length). The knee of the curve is the server's sustainable rps for that
input length under an SLA you choose by inspection. Needs `--rps` rows in the run.

    python benchmarks/plot_rps_ramp.py benchmarks/runs/<host-slug>

Writes `rps_ramp.png` next to the input CSVs.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(run_dir: Path):
    """Return {(model, input_length): [(rps, p95_ms, dps), ...]} sorted by rps."""
    rows = defaultdict(list)
    with (run_dir / "latency.csv").open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != "rps":
                continue
            if row["errors"] and int(row["errors"]) > 0:
                continue
            if not row["p95_ms"] or not row["rps"]:
                continue
            key = (row["model"], int(row["input_length"]))
            rows[key].append(
                (
                    float(row["rps"]),
                    float(row["p95_ms"]),
                    float(row["decisions_per_s"]) if row["decisions_per_s"] else None,
                )
            )
    for k in rows:
        rows[k].sort()
    return rows


def plot(run_dir: Path, out: Path) -> None:
    rows = load(run_dir)
    if not rows:
        sys.exit(f"no --rps rows in {run_dir / 'latency.csv'} — re-run with --rps N,N,N")

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_keys = sorted(rows.keys(), key=lambda k: (k[0], k[1]))
    for i, key in enumerate(sorted_keys):
        model, input_length = key
        pts = rows[key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"{model}, {input_length} input tokens"
        line, = ax.plot(xs, ys, marker="o", label=label)
        y_offset = 6 + i * 11
        for x, y, dps in pts:
            if dps is None:
                continue
            tag = f"{dps:.1f} dps"
            ax.annotate(
                tag, (x, y),
                xytext=(5, y_offset), textcoords="offset points",
                fontsize=7, color=line.get_color(),
            )

    ax.set_xlabel("Target rps (Poisson arrivals)")
    ax.set_ylabel("p95 request latency (ms)")
    ax.set_title(f"Open-loop ramp — p95 vs rps  —  {run_dir.name}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: plot_rps_ramp.py <run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    if not (run_dir / "latency.csv").exists():
        print(f"no latency.csv in {run_dir}", file=sys.stderr)
        return 1
    plot(run_dir, run_dir / "rps_ramp.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
