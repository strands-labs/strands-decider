"""Synthetic charts (our own matplotlib renders, no third-party data).

Bar (vertical/horizontal), grouped bar, multi-series line, pie, and 2x2 multi-panel line
figures with randomised data, labels and styles. Questions read values, extremes,
comparisons and trends; every gold answer is computed from the data that was drawn,
and value questions are asked only when the value is printed on the chart.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random

from common import row, write, yesno

CATS = {
    "region": ["North", "South", "East", "West", "Central", "Coastal", "Highlands", "Metro"],
    "product": ["Laptops", "Phones", "Tablets", "Monitors", "Printers", "Cameras", "Speakers", "Routers"],
    "country": ["Brazil", "Canada", "Egypt", "France", "India", "Japan", "Kenya", "Mexico", "Norway", "Peru"],
    "department": ["Sales", "Support", "R&D", "Finance", "Legal", "Ops", "HR", "Marketing"],
    "fruit": ["Apples", "Bananas", "Cherries", "Grapes", "Lemons", "Mangoes", "Pears", "Plums"],
}
SERIES = ["Model A", "Model B", "Baseline", "Ours", "Group 1", "Group 2", "Control", "Treatment", "North site", "South site"]
METRICS = ["Revenue ($M)", "Units sold", "Share (%)", "Score", "Visitors (k)", "Cost ($k)", "Accuracy (%)",
           "Emissions (kt)", "Orders", "Throughput"]
PALETTES = [None, "tab10", "Set2", "Dark2", "Paired", "viridis"]


def _style(plt, rng):
    plt.rcParams.update({"font.size": rng.choice([9, 10, 11, 12]),
                         "axes.grid": rng.random() < 0.5,
                         "font.family": rng.choice(["DejaVu Sans", "Liberation Sans", "DejaVu Serif"])})


def _colors(rng, n):
    p = rng.choice(PALETTES)
    if p is None:
        return [f"C{i % 10}" for i in range(n)]
    import matplotlib
    cmap = matplotlib.colormaps[p]
    return [cmap(i / max(1, n - 1)) if p == "viridis" else cmap(i % cmap.N) for i in range(n)]


def num_opts(v: float, rng, decimals: int) -> tuple[list[list[str]], int]:
    """Gold + 3 distractors at plausible misreadings (neighbour values, scaled)."""

    def fmt(x: float) -> str:
        return f"{x:.{decimals}f}"

    cands = {fmt(v)}
    n_lo = rng.randint(0, 3)  # gold's rank among the options is uniform
    for f in rng.sample([0.5, 0.6, 0.75, 0.8, 0.9], n_lo) + rng.sample([1.1, 1.2, 1.25, 1.4, 1.5], 3 - n_lo) + [0.7, 1.3, 0.85, 1.15]:
        if len(cands) >= 4:
            break
        cands.add(fmt(v * f))
    vals = sorted(cands, key=float)
    if len(vals) < 4:
        return [], -1
    return [[chr(65 + i), x] for i, x in enumerate(vals)], vals.index(fmt(v))


def make(args):
    idx, out, seed = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    _style(plt, rng)
    kind = rng.choice(["bar", "bar", "hbar", "group", "line", "line", "pie", "panels"])
    name = f"charts/c{idx:06d}.png"
    rows = []
    key = rng.choice(list(CATS))
    metric = rng.choice(METRICS)
    dec = rng.choice([0, 0, 1])
    figsize = (rng.uniform(5, 8), rng.uniform(3.5, 6))
    fig = plt.figure(figsize=figsize, dpi=rng.choice([80, 100, 120]))
    sid = f"chart-{idx}"
    kw = dict(source="charts", images=[name], source_id=sid)
    if kind in ("bar", "hbar"):
        n = rng.randint(4, 7)
        labels = rng.sample(CATS[key], n)
        vals = [round(rng.uniform(5, 100), dec) for _ in labels]
        while len(set(vals)) < n:
            vals = [round(rng.uniform(5, 100), dec) for _ in labels]
        ax = fig.add_subplot(111)
        cols = _colors(rng, n) if rng.random() < 0.5 else [_colors(rng, 1)[0]] * n
        bars = (ax.bar if kind == "bar" else ax.barh)(labels, vals, color=cols)
        show = rng.random() < 0.6
        if show:
            ax.bar_label(bars, fmt=f"%.{dec}f", fontsize=9)
        (ax.set_ylabel if kind == "bar" else ax.set_xlabel)(metric)
        ax.set_title(f"{metric} by {key}")
        opts4 = rng.sample(range(n), min(n, rng.choice([4, 5])))
        hi = max(range(n), key=lambda i: vals[i])
        lo = min(range(n), key=lambda i: vals[i])
        for want, word in ((hi, "highest"), (lo, "lowest")):
            sel = list(opts4) if want in opts4 else [*opts4[:-1], want]
            rng.shuffle(sel)
            sub = [vals[i] for i in sel]
            g = (max if word == "highest" else min)(range(len(sel)), key=lambda j: sub[j])
            rows.append(row("choice", f"Which {key} has the {word} {metric.split(' (')[0].lower()}?",
                            [[labels[i], ""] for i in sel], g, task="chart/extreme", **kw))
        i, j = rng.sample(range(n), 2)
        rows.append(yesno(f"Is the {metric.split(' (')[0].lower()} of {labels[i]} greater than that of {labels[j]}?",
                          vals[i] > vals[j], rng, task="chart/compare", **kw))
        if show:
            i = rng.randrange(n)
            o, g = num_opts(vals[i], rng, dec)
            if o:
                rows.append(row("choice", f"What is the {metric} value for {labels[i]}?", o, g, task="chart/value", **kw))
            i, j = rng.sample(range(n), 2)
            o, g = num_opts(abs(vals[i] - vals[j]), rng, dec)
            if o and abs(vals[i] - vals[j]) > 1:
                rows.append(row("choice", f"What is the difference in {metric} between {labels[i]} and {labels[j]}?",
                                o, g, task="chart/diff", **kw))
    elif kind == "group":
        n = rng.randint(3, 5)
        labels = rng.sample(CATS[key], n)
        ser = rng.sample(SERIES, 2)
        vals = [[round(rng.uniform(5, 100), dec) for _ in labels] for _ in ser]
        ax = fig.add_subplot(111)
        w = 0.38
        for s, (sname, v) in enumerate(zip(ser, vals, strict=True)):
            ax.bar([k + (s - 0.5) * w for k in range(n)], v, w, label=sname)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels)
        ax.legend(loc=rng.choice(["upper left", "upper right", "best"]))
        ax.set_ylabel(metric)
        i = rng.randrange(n)
        rows.append(row("choice", f"In {labels[i]}, which series has the higher {metric.split(' (')[0].lower()}?",
                        [[s, ""] for s in ser], int(vals[1][i] > vals[0][i]), task="chart/group", **kw))
        s = rng.randrange(2)
        cnt = sum(vals[s][k] > vals[1 - s][k] for k in range(n))
        o = [[str(k), ""] for k in range(0, n + 1)][:5]
        if cnt < len(o):
            rows.append(row("choice", f"For how many {key} categories is {ser[s]} higher than {ser[1 - s]}?",
                            o, cnt, task="chart/group_count", **kw))
    elif kind == "line":
        start = rng.randint(1995, 2015)
        years = list(range(start, start + rng.randint(6, 10)))
        k = rng.randint(1, 3)
        ser = rng.sample(SERIES, k)
        data = []
        for _ in ser:
            v, cur = [], rng.uniform(10, 60)
            drift = rng.uniform(-6, 6)
            for _ in years:
                cur = max(1, cur + drift + rng.gauss(0, 6))
                v.append(round(cur, 1))
            data.append(v)
        ax = fig.add_subplot(111)
        for sname, v in zip(ser, data, strict=True):
            ax.plot(years, v, marker=rng.choice(["o", "s", "^", None]), label=sname)
        if k > 1:
            ax.legend()
        ax.set_ylabel(metric)
        ax.set_xlabel("Year")
        s = rng.randrange(k)
        v = data[s]
        peak = years[max(range(len(v)), key=lambda i: v[i])]
        cand = [*rng.sample([y for y in years if y != peak], 3), peak]
        cand.sort()
        sname = f" for {ser[s]}" if k > 1 else ""
        rows.append(row("choice", f"In which year does the {metric.split(' (')[0].lower()}{sname} reach its peak?",
                        [[str(y), ""] for y in cand], cand.index(peak), task="chart/peak", **kw))
        a_, b_ = sorted(rng.sample(range(len(years)), 2))
        if abs(v[b_] - v[a_]) > 4:
            rows.append(yesno(f"Does the {metric.split(' (')[0].lower()}{sname} increase from {years[a_]} to {years[b_]}?",
                              v[b_] > v[a_], rng, task="chart/trend", **kw))
        if k > 1:
            yi = rng.randrange(len(years))
            col = [d[yi] for d in data]
            if sorted(col)[-1] - sorted(col)[-2] > 3:
                rows.append(row("choice", f"Which series has the highest value in {years[yi]}?",
                                [[x, ""] for x in ser], max(range(k), key=lambda i: col[i]), task="chart/line_series", **kw))
    elif kind == "pie":
        n = rng.randint(3, 6)
        labels = rng.sample(CATS[key], n)
        raw = [rng.uniform(5, 50) for _ in labels]
        tot = sum(raw)
        pct = [round(100 * x / tot, 1) for x in raw]
        ax = fig.add_subplot(111)
        show = rng.random() < 0.7
        ax.pie(raw, labels=labels, autopct="%1.1f%%" if show else None, startangle=rng.randint(0, 360))
        ax.set_title(f"Share of {metric.split(' (')[0].lower()} by {key}")
        hi = max(range(n), key=lambda i: pct[i])
        if sorted(pct)[-1] - sorted(pct)[-2] > 2 or show:
            rows.append(row("choice", f"Which {key} has the largest share?", [[x, ""] for x in labels], hi,
                            task="chart/pie_max", **kw))
        i = rng.randrange(n)
        thr = rng.choice([10, 20, 25, 30])
        if abs(pct[i] - thr) > 3 or show:
            rows.append(yesno(f"Is the share of {labels[i]} more than {thr}%?", pct[i] > thr, rng,
                              task="chart/pie_thr", **kw))
    else:  # panels: 2x2 curves, "which panel" questions
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(rng.uniform(7, 10), rng.uniform(5, 7.5)),
                                 dpi=rng.choice([80, 100]))
        xs = list(range(0, rng.choice([50, 100, 500, 1000]) + 1, 5))
        xs = xs[:: max(1, len(xs) // 25)]
        rates = rng.sample([0.2, 0.5, 1.0, 1.6, 2.4, -0.6, -1.2], 4)
        what = rng.choice(["throughput", "temperature", "yield", "population", "signal", "pressure", "voltage"])
        for p, (ax, r) in enumerate(zip(axes.flat, rates, strict=True)):
            base = rng.uniform(10, 40)
            ys = [base + r * 20 * x / xs[-1] + rng.gauss(0, 0.8) for x in xs]
            ax.plot(xs, ys, color=rng.choice(["C0", "C1", "C2", "C3", "k"]))
            ax.set_title(f"({'abcd'[p]})")
            ax.set_xlabel(rng.choice(["Generation", "Step", "Epoch", "Time (s)"]))
            ax.set_ylabel(what)
        fig.tight_layout()
        opts = [[chr(65 + p), f"Panel ({'abcd'[p]})"] for p in range(4)]
        pos = [p for p in range(4) if rates[p] > 0]
        if len(pos) >= 2:
            slow = min(pos, key=lambda p: rates[p])
            rows.append(row("choice", rng.choice([f"Which panel shows the slowest growth in {what}?",
                                                  f"Among the four panels, where does {what} rise least steeply?"]),
                            opts, slow, task="chart/panel", **kw))
            fast = max(range(4), key=lambda p: rates[p])
            rows.append(row("choice", rng.choice([f"Which panel shows the steepest rise in {what}?",
                                                  f"Where does {what} grow fastest?"]),
                            opts, fast, task="chart/panel", **kw))
        neg = [p for p in range(4) if rates[p] < 0]
        if len(neg) == 1:
            rows.append(row("choice", f"Which panel shows the {what} decreasing?", opts, neg[0],
                            task="chart/panel", **kw))
    if kind != "panels":
        fig.tight_layout()
    fig.savefig(os.path.join(out, name))
    plt.close("all")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-charts", type=int, default=2600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "charts"), exist_ok=True)
    jobs = [(i, a.out, a.seed * 1_000_003 + i) for i in range(a.n_charts)]
    with mp.Pool(a.workers) as pool:
        rows = [r for part in pool.imap(make, jobs, chunksize=16) for r in part]
    write(os.path.join(a.out, "charts.jsonl"), rows)


if __name__ == "__main__":
    main()
