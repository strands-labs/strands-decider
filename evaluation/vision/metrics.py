"""Scores for Phase 1: accuracy, Brier, ECE, and NaturalBench's paired accuracies.

Every system emits, per item, a probability over that item's options. Confidence is
the top probability (what a router would threshold on); Brier is the multi-class form
summed over options, so a uniform guess on a yes/no item scores 0.5.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def brier(probs: list[float], gold: int) -> float:
    return sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(probs))


def ece(rows: list[tuple[float, bool]], bins: int = 10) -> float:
    if not rows:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in rows if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(ok for _, ok in sel) / len(sel)
            total += len(sel) / len(rows) * abs(conf - acc)
    return total


def summarise(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """`results` rows: {probs, gold, kind, ...}. Overall and per kind."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by["all"].append(r)
        by[r.get("kind", "?")].append(r)
    out = {}
    for k, rs in by.items():
        pred = [max(range(len(r["probs"])), key=r["probs"].__getitem__) for r in rs]
        ok = [p == r["gold"] for p, r in zip(pred, rs, strict=True)]
        conf = [max(r["probs"]) for r in rs]
        out[k] = {
            "n": len(rs),
            "accuracy": round(sum(ok) / len(rs), 4),
            "brier": round(sum(brier(r["probs"], r["gold"]) for r in rs) / len(rs), 4),
            "ece": round(ece(list(zip(conf, ok, strict=True))), 4),
            "mean_confidence": round(sum(conf) / len(rs), 4),
        }
    return out


def naturalbench_paired(results: list[dict[str, Any]]) -> dict[str, float]:
    """Q-Acc, I-Acc, G-Acc over 2x2 groups.

    Each row carries `group`, `q` (0/1) and `i` (0/1). Q-Acc: a question is right on
    both images. I-Acc: an image is right on both questions. G-Acc: all four right.
    """
    cells: dict[Any, dict[tuple[int, int], bool]] = defaultdict(dict)
    for r in results:
        pred = max(range(len(r["probs"])), key=r["probs"].__getitem__)
        cells[r["group"]][(r["q"], r["i"])] = pred == r["gold"]
    groups = [c for c in cells.values() if len(c) == 4]
    if not groups:
        return {}
    q = sum(c[(qq, 0)] and c[(qq, 1)] for c in groups for qq in (0, 1)) / (2 * len(groups))
    i = sum(c[(0, ii)] and c[(1, ii)] for c in groups for ii in (0, 1)) / (2 * len(groups))
    g = sum(all(c.values()) for c in groups) / len(groups)
    return {"groups": len(groups), "q_acc": round(q, 4), "i_acc": round(i, 4), "g_acc": round(g, 4)}


def image_dependence(with_img: list[dict[str, Any]], blind: list[dict[str, Any]]) -> dict[str, float]:
    """How much the image moves the answer: mean total-variation distance and the
    confidence drop when the image is removed, matched by item id."""
    b = {r["id"]: r for r in blind}
    tv, drop, n = 0.0, 0.0, 0
    for r in with_img:
        if r["id"] in b:
            p, q = r["probs"], b[r["id"]]["probs"]
            tv += 0.5 * sum(abs(x - y) for x, y in zip(p, q, strict=True))
            drop += max(p) - max(q)
            n += 1
    return {"n": n, "mean_tv_distance": round(tv / n, 4) if n else math.nan,
            "mean_confidence_drop": round(drop / n, 4) if n else math.nan}


def score_runs(all_res: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Every block of a run's summary.json, from its systems' result rows (`<tag>.jsonl`)."""
    out: dict[str, Any] = {}
    for tag, rs in all_res.items():
        by_bench: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            by_bench.setdefault(r["bench"], []).append(r)
        out[tag] = {b: summarise(v) for b, v in by_bench.items()}
        if by_bench.get("naturalbench"):
            out[tag]["naturalbench"]["paired"] = naturalbench_paired(by_bench["naturalbench"])
        ijb = by_bench.get("ijb_preview")
        if ijb:
            block = out[tag]["ijb_preview"]
            block["exact_only"] = summarise([r for r in ijb if r.get("exact")])["all"]
            per: dict[str, list[dict[str, Any]]] = {}
            for r in ijb:
                per.setdefault(r["dataset"], []).append(r)
            block["by_dataset"] = {d: summarise(v)["all"] for d, v in sorted(per.items())}
    for tag in list(all_res):
        if not tag.endswith("-blind") and f"{tag}-blind" in all_res:
            out[tag]["image_dependence"] = image_dependence(all_res[tag], all_res[f"{tag}-blind"])
    return out
