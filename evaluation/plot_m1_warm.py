"""Rebuild the two M1 warm-run figures from saved benchmark JSON.

Requires Python 3.12 and evaluation/requirements-plots.txt. Run from the repository root:
    python evaluation/plot_m1_warm.py
"""

# Unicode punctuation is intentional in figure labels.
# ruff: noqa: RUF001
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#94A3B8", "#2563EB", "#D28A1F", "#C45558"]
INK, MUTED, GRID = "#12243A", "#58677B", "#E3E9F0"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "svg.fonttype": "none",
        "savefig.facecolor": "#FFFFFF",
    }
)
SOURCES = [
    ("BF16", "m1-bf16-clean", "fp"),
    ("FP16", "m1-fp16-clean", "fp"),
    ("INT8", "m1-fp16-clean", "8"),
    ("INT4", "m1-fp16-clean", "4"),
]
series, provenance, exported = [], {}, []
for name, run, precision in SOURCES:
    path = ROOT / "reports" / run / (precision + ".json")
    data = json.loads(path.read_text())
    rows = data["latency"]
    assert len(rows) == 6 and all(len(r["samples_ms"]) == 5 for r in rows)
    series.append(rows)
    provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in rows:
        exported.append(
            {
                "precision": name,
                "source_run": run,
                "base_dtype": data["torso_dtype"],
                "state_tokens": row["state_tokens_requested"],
                "questions": row["questions"],
                "input_tokens": row["input_tokens"],
                "median_s": row["median_ms"] / 1000,
                "min_s": min(row["samples_ms"]) / 1000,
                "max_s": max(row["samples_ms"]) / 1000,
                "peak_mlx_gib": row["mlx_peak_gib"],
            }
        )
shapes = [(r["state_tokens_requested"], r["questions"]) for r in series[0]]
assert all(
    [(r["state_tokens_requested"], r["questions"]) for r in rows] == shapes for rows in series
)
reduction = np.mean(
    [100 * (1 - f["median_ms"] / b["median_ms"]) for b, f in zip(series[0], series[1], strict=True)]
)

fig, ax = plt.subplots(figsize=(13.6, 8.2))
fig.subplots_adjust(left=0.07, right=0.975, bottom=0.19, top=0.76)
fig.text(0.07, 0.94, "FP16 delivers the lowest latency on this M1", fontsize=23, weight="bold")
fig.text(
    0.07,
    0.888,
    "Strands Decider 2B  /  MacBook Air M1, 16 GB  /  median request latency",
    fontsize=12.5,
    color=MUTED,
)
fig.text(0.975, 0.81, "Lower is better", ha="right", fontsize=11, color=MUTED)
x = np.arange(6)
width = 0.18
for i, ((name, _, _), rows, color) in enumerate(zip(SOURCES, series, COLORS, strict=True)):
    vals = np.array([r["median_ms"] / 1000 for r in rows])
    lo = np.array([min(r["samples_ms"]) / 1000 for r in rows])
    hi = np.array([max(r["samples_ms"]) / 1000 for r in rows])
    positions = x + (i - 1.5) * width
    ax.bar(positions, vals, width=width * 0.89, color=color, label=name, zorder=3)
    ax.errorbar(
        positions,
        vals,
        yerr=[vals - lo, hi - vals],
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=2,
        capthick=0.9,
        zorder=4,
        alpha=0.6,
    )
    for px, value, top in zip(positions, vals, hi, strict=True):
        ax.text(
            px,
            top + 0.15,
            (f"{value:.2f}" if value < 10 else f"{value:.1f}"),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color=INK,
            weight="bold" if name == "FP16" else "normal",
        )
ax.set_ylim(0, 12.1)
ax.set_yticks(np.arange(0, 13, 2))
ax.set_ylabel("Seconds", labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{n:,} tokens\n{q} question" + ("s" if q > 1 else "") for n, q in shapes],
    fontsize=10.5,
    linespacing=1.65,
)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0, pad=10)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(
    loc="lower left",
    bbox_to_anchor=(-0.005, 1.045),
    frameon=False,
    ncol=4,
    handlelength=1.3,
    columnspacing=2.3,
    fontsize=12,
    borderaxespad=0,
)
fig.text(
    0.07,
    0.097,
    f"{reduction:.0f}% lower latency on average for FP16 vs BF16 across these six workloads.",
    fontsize=12,
    weight="bold",
)
fig.text(
    0.07,
    0.057,
    "INT8 and INT4 use the FP16 base. Bars and labels: median of 5 runs; whiskers: observed min–max.",
    fontsize=10,
    color=MUTED,
)
fig.text(
    0.07,
    0.031,
    "Two warmups per shape. AC power. Token counts are state targets; prompt and question overhead is included in timing.",
    fontsize=9.4,
    color=MUTED,
)
for ext in ["png", "svg"]:
    fig.savefig(OUT / ("m1-latency." + ext), dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13.6, 7.7))
fig.subplots_adjust(left=0.12, right=0.975, bottom=0.25, top=0.76)
fig.text(0.07, 0.94, "Quantization reduces peak MLX memory", fontsize=24, weight="bold")
fig.text(
    0.07,
    0.885,
    "Largest workload: 4,000 state tokens + 8 questions  /  lower is better",
    fontsize=12.5,
    color=MUTED,
)
values = [rows[-1]["mlx_peak_gib"] for rows in series]
baseline = values[1]
ax.barh(np.arange(4), values, height=0.54, color=COLORS, zorder=3)
ax.set_yticks(np.arange(4), [s[0] for s in SOURCES], fontsize=13, weight="bold")
ax.invert_yaxis()
ax.set_xlim(0, 6.3)
ax.set_xticks(np.arange(0, 6))
ax.set_xlabel("Peak MLX allocation (GiB)", labelpad=13)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0, pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)
for i, value in enumerate(values):
    ax.text(value + 0.10, i - 0.06, f"{value:.2f} GiB", va="center", fontsize=13, weight="bold")
    note = (
        "Same as FP16"
        if i == 0
        else "Baseline"
        if i == 1
        else f"−{100 * (1 - value / baseline):.0f}% vs FP16"
    )
    ax.text(
        value + 0.10,
        i + 0.16,
        note,
        va="center",
        fontsize=10.5,
        color=COLORS[i] if i > 1 else MUTED,
        weight="bold" if i > 1 else "normal",
    )
savings = [
    np.mean(
        [
            100 * (1 - q["mlx_peak_gib"] / f["mlx_peak_gib"])
            for f, q in zip(series[1], rows, strict=True)
        ]
    )
    for rows in series[2:]
]
fig.text(
    0.07,
    0.13,
    f"Average savings across all six workloads:  INT8 {savings[0]:.0f}%  ·  INT4 {savings[1]:.0f}%",
    fontsize=13,
    weight="bold",
)
fig.text(
    0.07,
    0.075,
    "INT8 and INT4 use the FP16 base. The largest-shape reductions are 23% and 36%, respectively.",
    fontsize=10,
    color=MUTED,
)
fig.text(
    0.07,
    0.043,
    "MLX allocation is not total system RAM, process RSS, or peak memory during model loading.",
    fontsize=10,
    color=MUTED,
)
for ext in ["png", "svg"]:
    fig.savefig(OUT / ("m1-memory." + ext), dpi=220)
plt.close(fig)

with (OUT / "m1-plot-data.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(exported[0]))
    writer.writeheader()
    writer.writerows(exported)
(OUT / "m1-plot-provenance.json").write_text(
    json.dumps(
        {
            "matplotlib": matplotlib.__version__,
            "sha256": provenance,
            "selection": "BF16 baseline from m1-bf16-clean; FP16 and both quantized variants from m1-fp16-clean.",
            "whiskers": "Observed minimum and maximum of five timed runs; not confidence intervals.",
        },
        indent=2,
    )
    + "\n"
)
print("Created latency and memory plots in", OUT)
