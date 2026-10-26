#!/usr/bin/env python3
"""Compute cost per 1M decisions and the latency/price frontier from benchmark runs.

Reads every `runs/*/summary.json` plus `prices.yaml`, writes `figures/cost_per_1m_decisions.png`
and `figures/latency_vs_price_frontier.png`.

Hosts with a null hourly rate (own hardware, or an unpriced deploy target) are skipped.

    python benchmarks/cost.py

Peak-throughput model: cost per 1M decisions = (hourly $) / (peak decisions/s) * 1e6 / 3600.
Assumes 100% utilization — underestimates the real cost of an unsaturated endpoint.

For Lambda the per-request formula is used instead: see prices.yaml.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    sys.exit("prices.yaml needs PyYAML: pip install pyyaml")


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"


def load_prices():
    with (ROOT / "prices.yaml").open() as f:
        return yaml.safe_load(f)


def host_slug_from_run_dir(name):
    """Strip the model suffix to match prices.yaml keys.

    e.g. 'm4-pro-mlx-2b-v21' -> 'm4-pro-mlx', 'sagemaker-ml-g6-xlarge-2b-v21' -> 'sagemaker-ml-g6-xlarge'.
    The model suffix is the last two dash-separated tokens (`<size>-<version>`); everything before is the host.
    """
    return re.sub(r"-[0-9A-Za-z]+-v[0-9]+$", "", name)


def peak_throughput(summary):
    """Highest decisions_per_s across every (input_length, load) cell in a summary."""
    peak = 0.0
    for model_name, m in summary.get("models", {}).items():
        for inp, loads in m.get("ttft", {}).items():
            for load, cell in loads.items():
                v = cell.get("decisions_per_s") or 0.0
                if v > peak:
                    peak = v
    return peak


def representative_p95(summary, target_input_length=1024, target_concurrency=1):
    """p95 ms at a chosen cell, or the first available cell if that one isn't present."""
    for model_name, m in summary.get("models", {}).items():
        ttft = m.get("ttft", {})
        key_input = f"{target_input_length} tokens"
        key_load = f"{target_concurrency} in flight"
        if key_input in ttft and key_load in ttft[key_input]:
            return ttft[key_input][key_load].get("p95_ms")
        # fallback: first cell
        for inp, loads in ttft.items():
            for load, cell in loads.items():
                return cell.get("p95_ms")
    return None


def collect():
    """Return list of (host_slug, run_name, summary) for every run with a summary.json."""
    rows = []
    for run_dir in sorted(RUNS.glob("*/")):
        s = run_dir / "summary.json"
        if not s.exists():
            continue
        with s.open() as f:
            summary = json.load(f)
        rows.append((host_slug_from_run_dir(run_dir.name), run_dir.name, summary))
    return rows


def cost_per_1m(hourly_usd, peak_dps):
    if hourly_usd is None or peak_dps <= 0:
        return None
    return hourly_usd / peak_dps * 1_000_000 / 3600


def plot_cost(rows, prices):
    FIGS.mkdir(exist_ok=True)
    data = []
    for slug, run_name, summary in rows:
        price = (prices.get("hosts") or {}).get(slug) or {}
        hourly = price.get("hourly")
        peak = peak_throughput(summary)
        c = cost_per_1m(hourly, peak)
        if c is not None:
            data.append((run_name, c))

    fig, ax = plt.subplots(figsize=(8, 1 + 0.4 * max(len(data), 1)))
    if not data:
        ax.text(0.5, 0.5,
                "No priced hosts yet.\nFill in hourly rates in prices.yaml and rerun.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        data.sort(key=lambda row: row[1])
        names = [row[0] for row in data]
        costs = [row[1] for row in data]
        ax.barh(names, costs)
        ax.set_xlabel("USD per 1M decisions (peak throughput, 100% utilization)")
        ax.set_title("Cost per 1M decisions")
        for i, c in enumerate(costs):
            ax.text(c, i, f"  ${c:,.2f}", va="center")
    fig.tight_layout()
    out = FIGS / "cost_per_1m_decisions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_frontier(rows, prices):
    FIGS.mkdir(exist_ok=True)
    data = []
    for slug, run_name, summary in rows:
        price = (prices.get("hosts") or {}).get(slug) or {}
        hourly = price.get("hourly")
        p95 = representative_p95(summary)
        if hourly is not None and p95 is not None:
            data.append((run_name, hourly, p95))

    fig, ax = plt.subplots(figsize=(7, 5))
    if not data:
        ax.text(0.5, 0.5,
                "No priced hosts yet.\nFill in hourly rates in prices.yaml and rerun.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for name, h, p in data:
            ax.scatter(h, p)
            ax.annotate(name, (h, p), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("USD per hour")
        ax.set_ylabel("p95 TTFT (ms) at 1024 input tokens, concurrency 1")
        ax.set_title("Latency vs price")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGS / "latency_vs_price_frontier.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows = collect()
    prices = load_prices()
    if not rows:
        print("no runs under benchmarks/runs; measure a host first, see benchmarks/README.md")
        return 1
    plot_cost(rows, prices)
    plot_frontier(rows, prices)
    return 0


if __name__ == "__main__":
    sys.exit(main())
