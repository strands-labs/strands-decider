"""Convert Strands Decider v2 rows (train.jsonl / dev.jsonl) into training Examples.

    python -m training.data.strands_v2 train.jsonl train_examples.jsonl [--workers 16]

A v2 row is {id, suite, family, state, question: {type, criteria, instructions}, target,
label, source: {..., fixed_order?}}. It becomes an Example the trainer reads, for render
"strands-chat":

  noul    options [false, true] with the row's criteria; target P(true) -> [1 - p, p]
  choice  options in the criteria's order; target the distribution over them
  score   options the levels; the label is the argmax (score rows train with ordinal smoothing)

`label` is the row's label as an option index. `target` is kept only when it is not one-hot, so
a one-hot row trains exactly as before. `source.fixed_order` (a score converted to a choice) is
kept: such rows are never shuffled.

Rows without instructions (the key missing or null: 518 of the 17,480,970 v2 rows) are dropped,
as the reference v2 runs dropped them: their length sidecar marked them not renderable, so the
runs never trained on them (with them dropped, the reference trainer's plan is run-2's, sha
94dee960...). Rows that strands_decider.core refuses to render, rows with images (the Strands
templates are text-only) and lines that are not JSON are dropped and counted too. The file is streamed, so it may
be larger than memory; `--workers` processes convert it in parallel, in order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool
from typing import Any

from strands_decider import core

from .format import Example


def convert(row: dict) -> Example:
    q = row["question"]
    kind, crit, target = q["type"], q.get("criteria"), row["target"]
    options: list[list[Any]]
    dist: list[float] | None
    if kind == "noul":
        crit = crit or {}
        options = [[k, crit.get(k)] for k in ("false", "true")]
        dist = [1.0 - float(target), float(target)]
        label = int(bool(row["label"]))
    elif kind == "choice":
        options = [[k, v] for k, v in crit.items()]
        names = list(crit)
        dist = target if isinstance(target, list) else [float(k == target) for k in names]
        label = names.index(row["label"]) if row["label"] in names else dist.index(max(dist))
    else:
        options = [
            [str(i), c if isinstance(c, str) else core.render_content(c)]
            for i, c in enumerate(crit)
        ]
        dist = None
        label = target.index(max(target)) if isinstance(target, list) else int(target)
    return Example(
        kind=kind,
        state=row["state"],
        instructions=q.get("instructions") or "",
        options=options,
        label=label,
        task=row.get("suite", "unknown"),
        target=None if dist is None or set(dist) <= {0, 1} else [float(p) for p in dist],
        fixed_order=bool(row.get("source", {}).get("fixed_order")),
    )


def convert_line(line: str) -> tuple[str, str | None]:
    """(outcome, Example JSON or None) for one input line."""
    if not line.strip():
        return "blank", None
    try:
        row = json.loads(line)
    except ValueError:
        return "not_json", None
    if row.get("images"):
        return "image_rows", None
    q = dict(row["question"])  # as the reference hands it to the core (autojev spec_question)
    if q.get("instructions") is None:
        return "no_instructions", None
    if q.get("type") == "score" and isinstance(q.get("criteria"), list):
        q["criteria"] = [c if isinstance(c, str) else core.render_content(c) for c in q["criteria"]]
    try:  # the core's refusals; they need no tokenizer
        core.render_text({"state": row["state"], "questions": {"q": q}}, core.CHAT)
    except ValueError:
        return "not_renderable", None
    return "kept", convert(row).to_json()


def main(argv: list[str] | None = None) -> Counter[str]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    counts: Counter[str] = Counter()
    t0 = time.time()
    with open(args.src, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        pool = Pool(args.workers) if args.workers > 1 else None
        results: Iterator[tuple[str, str | None]] = (
            pool.imap(convert_line, fin, chunksize=1000) if pool else map(convert_line, fin)
        )
        for outcome, text in results:
            counts[outcome] += 1
            if text is not None:
                fout.write(text + "\n")
        if pool:
            pool.close()
            pool.join()
    rows = sum(counts.values())
    rate = rows / max(time.time() - t0, 1e-9)
    print(json.dumps({**counts, "rows": rows, "rows_per_s": round(rate)}), file=sys.stderr)
    return counts


if __name__ == "__main__":
    main()
