"""Figures and derived numbers from decider_bench.py runs, for the benchmark-decider skill. Needs only matplotlib.

    import sys; sys.path.insert(0, "skills/benchmark-decider/scripts")
    import figures as F

    lat = F.load("runs/*")                                 # every row of every run's latency.csv, as dicts
    lat = F.split_name(lat, ["size", "engine"], sep="-")   # factors from model names such as 2b-vllm
    best = F.max_throughput([r for r in lat if r["input_length"]])
    F.facet(best, x="tokens_mean", y="decisions_per_s", row="size", col="host", line="engine",
            failed=F.failed_cells("runs/*", ["size", "engine"]), title="Highest throughput by input length",
            xlabel="input tokens", ylabel="decisions/s", out="throughput_grid.png")

facet() draws one panel per (row, col) value and one line per `line` value, with shared log axes on 1-2-5 ticks,
one legend outside the panels, the same color for a line value in every panel, and a "did not start" note in any
panel whose model failed. Rows are plain dicts, so filter them with list comprehensions.
"""

import csv
import glob
import json
from pathlib import Path

# Public on-demand prices change; check them before quoting a cost (aws.amazon.com/lambda/pricing).
LAMBDA_GB_S = {"arm64": 0.0000133334, "x86_64": 0.0000166667}  # USD per GB-second, us-east-1
LAMBDA_REQUEST = 0.20 / 1e6  # USD per request


def _runs(pattern):
    dirs = [Path(p) for p in sorted(glob.glob(pattern) if isinstance(pattern, str) else pattern)]
    return [d for d in dirs if (d / "summary.json").exists()]


def _num(v):
    if v in ("", None):
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def _read(pattern, name):
    rows = []
    for d in _runs(pattern):
        if (d / name).exists():
            with open(d / name, newline="") as f:
                rows += [
                    {**{k: _num(v) for k, v in r.items()}, "run": d.name} for r in csv.DictReader(f)
                ]
    return rows


def load(pattern):
    """Every row of latency.csv in every run directory matching pattern, with a `run` key."""
    return _read(pattern, "latency.csv")


def load_accuracy(pattern):
    return _read(pattern, "accuracy.csv")


def failures(pattern):
    """[(model, host, error)] for models that did not start."""
    out = []
    for d in _runs(pattern):
        for label, r in json.loads((d / "summary.json").read_text())["models"].items():
            if r.get("error"):
                out.append((label, (r.get("model") or {}).get("host", "unknown"), r["error"]))
    return out


def split_name(rows, columns, sep="-", key="model"):
    """Split model names such as '2b-vllm-fp8' into factor keys; missing parts become ''."""
    for r in rows:
        parts = str(r[key]).split(sep, len(columns) - 1)
        r.update({c: parts[i] if i < len(parts) else "" for i, c in enumerate(columns)})
    return rows


def failed_cells(pattern, columns, sep="-", line=-1):
    """(line value, row value, host) for each model that did not start, from its name split like split_name.
    By default the last factor is the line and the first the row, as in facet(row=columns[0], line=columns[-1])."""
    out = []
    for label, host, _ in failures(pattern):
        parts = (label.split(sep, len(columns) - 1) + [""] * len(columns))[: len(columns)]
        out.append((parts[line], parts[0], host))
    return out


def max_throughput(rows, by=("model", "host", "input")):
    """The row with the highest decisions/s in each group, so the load that gave it is kept."""
    best = {}
    for r in rows:
        if r.get("decisions_per_s") is None:
            continue
        k = tuple(r.get(b) for b in by)
        if k not in best or r["decisions_per_s"] > best[k]["decisions_per_s"]:
            best[k] = r
    return list(best.values())


def _probs(answer):
    if "probabilities" in answer:
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    if "noul" in answer:
        return {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
    return None


def answer_drift(results_a, results_b):
    """How far two runs' answers to the same Decision Index requests differ (kit results.jsonl files).

    Returns the questions compared, argmax agreement, mean and p95 total variation (half the summed
    absolute difference of the two probability vectors), and every flip with run a's margin between its
    own answer and run b's (flips under about 0.02 are near-ties)."""

    def read(path):
        out = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        out[r["run_id"]] = (r.get("response") or {}).get("answers") or {}
        return out

    a, b = read(results_a), read(results_b)
    tvs, flips = [], []
    for rid in sorted(a.keys() & b.keys()):
        for q, qa in a[rid].items():
            pa, pb = _probs(qa), _probs(b[rid].get(q) or {})
            if not pa or not pb:
                continue
            tvs.append(0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb)))
            ta, tb = max(pa, key=pa.get), max(pb, key=pb.get)
            if ta != tb:
                flips.append(
                    {
                        "run_id": rid,
                        "question": q,
                        "a": ta,
                        "b": tb,
                        "margin_a": round(pa[ta] - pa.get(tb, 0), 4),
                    }
                )
    tvs.sort()
    return {
        "questions": len(tvs),
        "agree": len(tvs) - len(flips),
        "tv_mean": sum(tvs) / len(tvs) if tvs else None,
        "tv_p95": tvs[int(0.95 * (len(tvs) - 1))] if tvs else None,
        "flips": sorted(flips, key=lambda f: f["margin_a"]),
    }


def cost_per_million(decisions_per_s, price_per_hour):
    """A provisioned host kept busy: USD per million decisions."""
    return price_per_hour / (3600 * decisions_per_s) * 1e6


def lambda_cost_per_million(seconds, memory_mb=10240, arch="arm64"):
    """Lambda billed per use: USD per million decisions that each run `seconds` in the function."""
    return (memory_mb / 1024 * seconds * LAMBDA_GB_S[arch] + LAMBDA_REQUEST) * 1e6


def break_even_per_s(price_per_hour, lambda_cost_per_million_usd):
    """The sustained decisions/s above which a provisioned host is cheaper than Lambda."""
    return price_per_hour / 3600 / (lambda_cost_per_million_usd / 1e6)


def flops(tokens, gflop_per_token, attn_coeff=0.0):
    """FLOPs of one prefill: 2 x matmul parameters per token, plus attention (attn_coeff x tokens^2)."""
    return gflop_per_token * 1e9 * tokens + attn_coeff * tokens**2


def pct_of_peak(tokens, seconds, peak_tflops, gflop_per_token, attn_coeff=0.0):
    """Sustained rate as a share of the device's nominal peak. Above 100% the measurement is wrong."""
    return 100 * flops(tokens, gflop_per_token, attn_coeff) / seconds / (peak_tflops * 1e12)


def ticks125(lo, hi):
    out, d = [], 10.0 ** int(f"{lo:e}".split("e")[1])
    while d <= hi * 10:
        out += [x for x in (d, 2 * d, 5 * d) if lo <= x <= hi]
        d *= 10
    return out


def _round(x, up):
    t = ticks125(1e-9, 1e12)
    return min([v for v in t if v >= x] or [x]) if up else max([v for v in t if v <= x] or [x])


def _fmt(v):
    if v >= 1000 and v % 1000 == 0:
        return f"{v / 1000:g}k"
    return f"{v:,.0f}" if v >= 1000 else f"{v:g}"


def log_axis(ax, axis, values):
    """Log axis whose limits and ticks are round 1-2-5 values that bracket the data."""
    import matplotlib.ticker

    v = [x for x in values if x == x and x > 0]
    if not v:
        return
    lo, hi = _round(min(v) / 1.15, up=False), _round(max(v) * 1.15, up=True)
    t = ticks125(lo, hi)
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"set_{axis}lim")(lo, hi)
    getattr(ax, f"set_{axis}ticks")(t, [_fmt(x) for x in t])
    getattr(ax, f"{axis}axis").set_minor_locator(matplotlib.ticker.NullLocator())


def facet(
    df,
    x,
    y,
    line,
    row=None,
    col=None,
    out="figure.png",
    title=None,
    xlabel=None,
    ylabel=None,
    yscale=1.0,
    logx=True,
    logy=True,
    style=None,
    failed=(),
    hline=None,
    band=None,
    order=None,
):
    """A grid of panels from rows (dicts): rows by `row`, columns by `col`, one line per `line` value.

    yscale multiplies y (1e-3 turns ms into s). style is an optional column whose values get line styles (for
    example `mode`: concurrency solid, rps dashed). failed is a list of (line value, row value, col value) to
    mark "did not start". hline draws a dotted reference line (for example 1 s). band=(lo, hi) column names
    shade a range around each line (for example p10_ms and p90_ms). order maps a column to its value order.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = order or {}

    def values(c):
        if c is None:
            return [None]
        return order.get(c) or list(dict.fromkeys(r[c] for r in df if r.get(c) is not None))

    rows, cols, lines = values(row), values(col), values(line)
    styles = values(style) if style else [None]
    palette = plt.get_cmap("tab10").colors
    color = {v: palette[i % 10] for i, v in enumerate(lines)}
    dash = {v: d for v, d in zip(styles, ["-", "--", ":", "-."] * 4, strict=False)}
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(4.6 * len(cols) + 2.6, 3.3 * len(rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    xs, ys = [], []
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            ax = axes[i][j]
            sub = [
                r
                for r in df
                if (row is None or r.get(row) == rv) and (col is None or r.get(col) == cv)
            ]
            for lv in lines:
                for sv in styles:
                    s = sorted(
                        (
                            r
                            for r in sub
                            if r.get(line) == lv
                            and (style is None or r.get(style) == sv)
                            and r.get(x) is not None
                            and r.get(y) is not None
                        ),
                        key=lambda r: r[x],
                    )
                    if not s:
                        continue
                    lab = f"{lv}, {sv}" if style and len(styles) > 1 else str(lv)
                    ax.plot(
                        [r[x] for r in s],
                        [r[y] * yscale for r in s],
                        marker="o",
                        ms=4,
                        color=color[lv],
                        ls=dash.get(sv, "-"),
                        label=lab,
                    )
                    if band:
                        lo, hi = [r[band[0]] * yscale for r in s], [r[band[1]] * yscale for r in s]
                        ax.fill_between(
                            [r[x] for r in s], lo, hi, color=color[lv], alpha=0.12, lw=0
                        )
                        ys += lo + hi
                    xs += [r[x] for r in s]
                    ys += [r[y] * yscale for r in s]
            notes = [
                lv
                for lv, r_, c_ in failed
                if (row is None or r_ == rv) and (col is None or c_ == cv)
            ]
            for k, lv in enumerate(notes):
                ax.text(
                    0.5,
                    0.92 - 0.07 * k,
                    f"{lv}: did not start",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                    color="grey",
                )
            if hline is not None:
                ax.axhline(hline, color="grey", lw=0.8, ls=":")
            ax.grid(True, alpha=0.25)
            if i == 0 and col:
                ax.set_title(str(cv), fontsize=9)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel or x)
            if j == 0:
                ax.set_ylabel((f"{rv}\n" if row else "") + (ylabel or y))
    if logx:
        log_axis(axes[0][0], "x", xs)
    if logy:
        log_axis(axes[0][0], "y", ys)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=False):
            handles.setdefault(lab, h)
    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=8,
        )
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
