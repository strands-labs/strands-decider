#!/usr/bin/env python3
"""Enhanced throughput vs input-length chart for a single run directory.

The stock `decider_bench.py plot` shows peak decisions/s per input length — one line per model,
one point per length, max over concurrency. This script reads `latency.csv` from one run and
plots one line per concurrency (closed loop) or per rps (open loop), with each point labelled
by the N that produced it. The result exposes whether throughput climbs with load (batching
helps) or stays flat (batch-of-1 serving).

    python benchmarks/plot_throughput.py benchmarks/runs/<host-slug>

Writes `throughput_by_concurrency.png` next to the input CSVs.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load(run_dir: Path):
    """Return {(model, mode, load_value): [(input_length, decisions_per_s, n), ...]} sorted by input length."""
    rows = defaultdict(list)
    with (run_dir / "latency.csv").open() as f:
        for row in csv.DictReader(f):
            if row["errors"] and int(row["errors"]) > 0:
                continue
            if not row["decisions_per_s"]:
                continue
            model = row["model"]
            mode = row["mode"]  # concurrency or rps
            load_value = int(row["concurrency"] or 0) if mode == "concurrency" else float(row["rps"] or 0)
            key = (model, mode, load_value)
            rows[key].append(
                (
                    int(row["input_length"]),
                    float(row["decisions_per_s"]),
                    float(row["p50_ms"]) if row["p50_ms"] else None,
                )
            )
    for k in rows:
        rows[k].sort()
    return rows


def plot(run_dir: Path, out: Path) -> None:
    rows = load(run_dir)
    if not rows:
        sys.exit(f"no usable rows in {run_dir / 'latency.csv'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_keys = sorted(rows.keys())
    for i, (model, mode, load_value) in enumerate(sorted_keys):
        pts = rows[(model, mode, load_value)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = (
            f"{model}, {load_value} in flight"
            if mode == "concurrency"
            else f"{model}, {load_value} rps"
        )
        line, = ax.plot(xs, ys, marker="o", label=label)
        # Stagger annotations vertically by series order so overlapping concurrencies read.
        y_offset = 6 + i * 11
        for x, y, p50 in pts:
            if p50 is None:
                continue
            tag = f"p50 {int(p50)} ms" if p50 < 1000 else f"p50 {p50 / 1000:.1f} s"
            ax.annotate(
                tag, (x, y),
                xytext=(5, y_offset), textcoords="offset points",
                fontsize=7, color=line.get_color(),
            )

    ax.set_xlabel("Input tokens")
    ax.set_ylabel("Decisions per second")
    ax.set_title(f"Throughput by concurrency, input length  —  {run_dir.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: plot_throughput.py <run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    if not (run_dir / "latency.csv").exists():
        print(f"no latency.csv in {run_dir}", file=sys.stderr)
        return 1
    plot(run_dir, run_dir / "throughput_by_concurrency.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
