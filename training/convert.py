"""Decision rows converted into the trainer's rows, with each row's prompt length under the template.

    python -m training.convert --train-files <rows.jsonl[.gz] ...> --dev-file <rows.jsonl[.gz]> \\
        --base <base checkpoint dir> --out <dir> [--prompt strands-chat] [--max-length 16384] [--workers 32]

`--mix DIR` reads the same files from a directory laid out as `train/part-*.jsonl.gz` and `dev.jsonl.gz`. A decision
row, one JSON object a line (training/README.md, "The data"), is

    {id, lane, source, group, type, state, instructions, options: [{key, text}], target: [p, ...], meta}

where `lane` names the set a row came from, `source` its dataset or module, `group` the item it belongs to (rows of one
item share it, and the trainer keeps an item out of training when it is in the dev fold), and `meta` its provenance.

Each becomes a training row (training/targets.py):

    {id, suite, family, state, question: {type, criteria, instructions?}, target, label, source: {...}}

  choice  criteria = {key: text}; target = the option distribution (a list); label = meta.label when it is one of the
          keys, else the most probable key.
  noul    options false and true; criteria = {"false": text, "true": text}; target = P(true); label = meta.label (a
          bool) or P(true) > 0.5.
  score   `--score as-choice` (the default) writes the levels as choice options {level text: None}, marked
          source.fixed_order so the trainer never shuffles them; `--score drop` leaves them out.

`source.input_tokens` is the row's prompt length (state part plus question part) under `--prompt`, rendered by
strands_decider.core with the base's tokenizer, so the trainer's batching and length filter use exact lengths. Rows are
dropped, and counted, when two options share a key, the target is not a distribution over the options, there are more
than 255 options, the template cannot render the row, or the prompt plus a 16-token margin exceeds `--max-length` (for
the dev rows, the prompt itself: the trainer's rule for the rows it scores).

Writes `<out>/train.jsonl`, `<out>/dev.jsonl` (from the dev file, when there is one) and
`<out>/train.jsonl.manifest.json` (the template, the tokenizer's sha256, the counts), which the trainer checks.
CPU only.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import multiprocessing as mp
import os
import time
from typing import cast

from strands_decider.core import MAX_CODES

from . import prompt as prompts
from .batching import LENGTH_MARGIN
from .folds import sha256_file

_STATE: dict = {}


def _init(tokenizer_json: str, prompt: str) -> None:
    from tokenizers import Tokenizer

    _STATE.update(tokenizer=Tokenizer.from_file(tokenizer_json), prompt=prompt)


def convert_row(row: dict, score_mode: str) -> tuple[dict | None, str | None]:
    """(the training row, None), or (None, the reason it is dropped)."""
    kind, opts = row["type"], row["options"]
    try:
        target = [float(t) for t in row["target"]]
    except (TypeError, ValueError):
        return None, "bad_target"
    keys = [o["key"] for o in opts]
    if len(set(keys)) != len(keys):
        return None, "duplicate_option_keys"
    if not 1 <= len(opts) <= MAX_CODES:
        return None, "option_count"
    if (
        len(target) != len(opts)
        or any(not math.isfinite(t) or t < 0 for t in target)
        or abs(sum(target) - 1) > 1e-6
    ):
        return None, "bad_target"
    meta = row.get("meta") or {}
    label = meta.get("label")
    fixed_order = False
    if kind == "choice":
        criteria = {o["key"]: o.get("text") for o in opts}
        if label not in criteria:
            label = keys[max(range(len(keys)), key=target.__getitem__)]
    elif kind == "noul":
        if keys != ["false", "true"]:
            return None, "noul_keys"
        criteria = {"false": opts[0].get("text"), "true": opts[1].get("text")}
        if not isinstance(label, bool):
            label = {"true": True, "false": False}.get(str(label).lower(), target[1] > 0.5)
    elif kind == "score":
        if score_mode == "drop":
            return None, "score_dropped"
        levels = [o["text"] if o.get("text") is not None else o["key"] for o in opts]
        if len(set(levels)) != len(levels):
            return None, "duplicate_score_levels"
        criteria = dict.fromkeys(levels)
        kind, fixed_order = "choice", True
        label = levels[max(range(len(levels)), key=target.__getitem__)]
    else:
        return None, "unknown_type"
    if kind == "choice" and label not in criteria:
        return None, "bad_label"
    question = {"type": kind, "criteria": criteria}
    if row.get("instructions"):
        question["instructions"] = row["instructions"]
    source = {"dataset": row["source"], "lane": row["lane"], "mix_id": row["id"]}
    if meta.get("variant") is not None:
        source["variant"] = meta["variant"]
    if fixed_order:
        source["fixed_order"] = True
        source["mix_type"] = "score"
    example = {
        "id": row["id"],
        "suite": str(meta.get("suite") or row["source"]),
        "family": str(row.get("group") or row["id"]),
        "state": row["state"],
        "question": question,
        "target": target[1] if kind == "noul" else target,
        "label": label,
        "source": source,
    }
    return example, None


def _work(task: tuple[str, str, str, int, int, int]) -> dict:
    path, out_path, score_mode, max_length, margin, limit = task
    tokenizer, prompt = _STATE["tokenizer"], _STATE["prompt"]
    dropped: collections.Counter = collections.Counter()
    ids = []
    opener = gzip.open if path.endswith(".gz") else open
    with (
        opener(path, "rt", encoding="utf-8") as stream,
        open(out_path, "w", encoding="utf-8") as handle,
    ):
        for index, line in enumerate(stream):
            if limit and index >= limit:
                break
            converted, reason = convert_row(json.loads(line), score_mode)
            if reason:
                dropped[reason] += 1
                continue
            example = cast(dict, converted)
            try:
                n = prompts.prompt_tokens(example, tokenizer, prompt)
            except (ValueError, TypeError, KeyError):
                dropped["not_renderable"] += 1
                continue
            if n + margin > max_length:
                dropped["over_max_length"] += 1
                continue
            example["source"]["input_tokens"] = n
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            ids.append(example["id"])
    return {"path": path, "kept": len(ids), "dropped": dict(dropped), "ids": ids}


def run(
    sources: list[str], out: str, args: argparse.Namespace, limit: int, margin: int = LENGTH_MARGIN
) -> dict:
    parts = [f"{out}.part{i:04d}" for i in range(len(sources))]
    tasks = [
        (s, p, args.score, args.max_length, margin, limit)
        for s, p in zip(sources, parts, strict=True)
    ]
    with mp.get_context("spawn").Pool(
        min(args.workers, len(tasks)),
        initializer=_init,
        initargs=(os.path.join(args.base, "tokenizer.json"), args.prompt),
    ) as pool:
        results = pool.map(_work, tasks, chunksize=1)
    seen, duplicates = set(), 0
    for r in results:
        for identifier in r.pop("ids"):
            duplicates += identifier in seen
            seen.add(identifier)
    if duplicates:
        for part in parts:
            os.remove(part)
        raise SystemExit(f"{duplicates} duplicate ids across shards; not writing {out}")
    with open(out, "w", encoding="utf-8") as handle:  # the shards in source order
        for part in parts:
            with open(part, encoding="utf-8") as stream:
                for line in stream:
                    handle.write(line)
            os.remove(part)
    dropped: collections.Counter = collections.Counter()
    for r in results:
        dropped.update(r["dropped"])
    return {
        "rows": sum(r["kept"] for r in results),
        "dropped": dict(dropped),
        "files": len(sources),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mix", help="a directory of decision rows: train/part-*.jsonl.gz and dev.jsonl.gz"
    )
    ap.add_argument(
        "--train-files", nargs="*", help="decision-row files to convert instead of the mix's train/"
    )
    ap.add_argument("--dev-file", help="a decision-row dev file instead of the mix's dev.jsonl.gz")
    ap.add_argument(
        "--base",
        required=True,
        help="the base checkpoint directory whose tokenizer.json measures lengths",
    )
    ap.add_argument(
        "--out", required=True, help="the output directory: train.jsonl, dev.jsonl and the manifest"
    )
    ap.add_argument("--prompt", choices=prompts.TEMPLATES, default="strands-chat")
    ap.add_argument("--score", choices=("as-choice", "drop"), default="as-choice")
    ap.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="rows whose prompt + 16 exceeds this are dropped",
    )
    ap.add_argument("--limit", type=int, default=0, help="rows per source file, for tests")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    began = time.time()
    sources = args.train_files or sorted(
        glob.glob(os.path.join(args.mix or "", "train", "part-*.jsonl.gz"))
    )
    if not sources:
        raise SystemExit("no training files: pass --mix or --train-files")
    os.makedirs(args.out, exist_ok=True)
    train_out = os.path.join(args.out, "train.jsonl")
    tokenizer_json = os.path.join(args.base, "tokenizer.json")
    report = {
        "prompt": args.prompt,
        "tokenizer_sha256": sha256_file(tokenizer_json),
        "score": args.score,
        "max_length": args.max_length,
        "length_margin": LENGTH_MARGIN,
        "train": run(sources, train_out, args, args.limit),
    }
    dev = args.dev_file or (os.path.join(args.mix, "dev.jsonl.gz") if args.mix else None)
    if dev and os.path.exists(dev):
        report["dev"] = run([dev], os.path.join(args.out, "dev.jsonl"), args, 0, margin=0)
    report["train"]["sha256"] = sha256_file(train_out)
    report["seconds"] = round(time.time() - began)
    with open(train_out + ".manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
