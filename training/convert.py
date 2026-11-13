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

v1's examples (training/data/format.py, what `python -m training.data` and its builders write) convert for either head:

    python -m training.convert --examples <examples.jsonl ...> --base <base dir> --out <dir> --prompt numbered \\
        [--force-bos] [--teacher 1.0 <teacher.jsonl ...>] [--ordinal-smoothing 0.1] [--val-fraction 0.03] \\
        [--val-split-seed 0 | --val-files <examples.jsonl ...>] [--max-length 4096]

  ids      `<file name>:<line>`; family = id, suite = dataset = the example's task, lane = the file name.
  noul     criteria {"false": text, "true": text}; target = P(true); label = the bool.
  choice   criteria {name: text}; target = the option distribution; label = the option's name.
  score    a choice whose options are the level indices ("0", "1", ...) with their texts, marked source.fixed_order
           and source.score; the pointer head renders it as v1's score question (training/pointer.py).
  targets  the label's one-hot (a score's smoothed over its neighbouring levels by --ordinal-smoothing, as v1's
           collator smooths it), mixed with each --teacher group's distribution for the rows it covers:
           (label + sum of W * p) / (1 + sum of W). Per row, that is the direction of v1's label loss plus W times
           KL(teacher, student) at 1 / (1 + sum of W) of its size; and every row weighs alike, where v1 gave a
           teacher's rows more (its KL term averaged over a micro-batch's teacher rows only). A teacher file is v1's:
           {"i": row index in the concatenated
           --examples, "probs": [...]} (training/data/teacher.py, replay.py, distill.py); each distribution is
           renormalised. The dev rows keep their label's target.
  dev      --val-files, or v1's split: --val-fraction of each task, chosen with --val-split-seed.

An example's instruction variants go to source.instruction_variants, which the pointer head samples from; its length
is the longest over its phrasings. Examples with a weight other than 1 are refused: the trainer weighs every row alike.
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
from typing import Any, cast

from strands_decider.core import MAX_CODES

from . import pointer
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


def smoothed(label: int, count: int, smoothing: float) -> list[float]:
    """A score's target over its levels: 1 - smoothing on the label, the rest shared by its neighbouring levels, as
    v1's collator makes it (training/data/collate.py)."""
    neighbours = [level for level in (label - 1, label + 1) if 0 <= level < count]
    values = [0.0] * count
    if smoothing <= 0 or not neighbours:
        values[label] = 1.0
        return values
    values[label] = 1.0 - smoothing
    for level in neighbours:
        values[level] = smoothing / len(neighbours)
    return values


def row_from_example(
    example: Any,
    identifier: str,
    lane: str,
    smoothing: float,
    teachers: list[tuple[float, list[float]]],
) -> dict:
    """The trainer's row for a v1 example; `teachers` are (weight, distribution) pairs in canonical option order."""
    names = [name for name, _ in example.options]
    if len(set(names)) != len(names):
        raise ValueError(f"{identifier}: two options share a name")
    if example.kind == "noul" and names != ["false", "true"]:
        raise ValueError(f"{identifier}: a noul's options must be false, then true")
    count = len(names)
    if example.kind == "score":
        target = smoothed(example.label, count, smoothing)
    else:
        target = [float(i == example.label) for i in range(count)]
    total = 1.0
    for weight, distribution in teachers:
        if len(distribution) != count:
            raise ValueError(
                f"{identifier}: a teacher distribution of {len(distribution)} for {count} options"
            )
        mass = sum(distribution)
        if not mass > 0 or any(not math.isfinite(p) or p < 0 for p in distribution):
            raise ValueError(f"{identifier}: a teacher distribution that is not one")
        target = [t + weight * p / mass for t, p in zip(target, distribution, strict=True)]
        total += weight
    target = [t / total for t in target]
    question = {
        "type": "noul" if example.kind == "noul" else "choice",
        "criteria": dict(example.options),
        "instructions": example.instructions,
    }
    source: dict[str, Any] = {"dataset": example.task, "lane": lane}
    if example.kind == "score":
        source["fixed_order"] = True
        source["score"] = True
    if example.instruction_variants:
        source["instruction_variants"] = list(example.instruction_variants)
    return {
        "id": identifier,
        "suite": example.task,
        "family": identifier,
        "state": example.state,
        "question": question,
        "target": target[1] if example.kind == "noul" else target,
        "label": example.label == 1 if example.kind == "noul" else names[example.label],
        "source": source,
    }


def read_examples(paths: list[str]) -> list[tuple[str, str, Any]]:
    """(id, lane, example) for every example of `paths`, in order."""
    from .data.format import read_jsonl

    names = [os.path.basename(path) for path in paths]
    if len(set(names)) != len(names):
        raise SystemExit("two --examples files share a file name, which the row ids are made of")
    out = []
    for path, name in zip(paths, names, strict=True):
        for line, example in enumerate(read_jsonl(path)):
            if example.weight != 1:
                raise SystemExit(
                    f"{path}:{line}: weight {example.weight}; the trainer weighs every row alike"
                )
            out.append((f"{name}:{line}", name, example))
    return out


def read_teachers(groups: list[list[str]], count: int) -> list[list[tuple[float, list[float]]]]:
    """Per example, the (weight, distribution) of every --teacher group that covers it."""
    per_row: list[list[tuple[float, list[float]]]] = [[] for _ in range(count)]
    for group in groups:
        try:
            weight, files = float(group[0]), group[1:]
        except ValueError:
            raise SystemExit(
                f"--teacher {group[0]!r}: give a weight, then the teacher files"
            ) from None
        if weight <= 0 or not files:
            raise SystemExit("--teacher takes a positive weight, then one or more teacher files")
        seen: set[int] = set()
        for path in files:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    index = int(entry["i"])
                    if not 0 <= index < count or index in seen:
                        raise SystemExit(
                            f"{path}: row index {index} is outside --examples or repeated"
                        )
                    seen.add(index)
                    per_row[index].append((weight, [float(p) for p in entry["probs"]]))
    return per_row


def convert_examples(args: argparse.Namespace) -> dict:
    """--examples: v1's examples into the trainer's rows (the module docstring)."""
    from .data.format import split_examples

    examples = read_examples(args.examples)
    teachers = read_teachers(args.teacher or [], len(examples))
    if args.val_files:
        train = list(range(len(examples)))
        held = read_examples(args.val_files)
        if {i for i, _l, _e in held} & {i for i, _l, _e in examples}:
            raise SystemExit(
                "an --examples file and a --val-files file share a file name, which the row ids are made of"
            )
    else:
        # the split shuffles; each example's position finds it again
        position = {id(example): i for i, (_id, _lane, example) in enumerate(examples)}
        kept, dev = split_examples(
            [example for _id, _lane, example in examples],
            val_fraction=args.val_fraction,
            seed=args.val_split_seed,
        )
        train = sorted(position[id(example)] for example in kept)
        held = [examples[i] for i in sorted(position[id(example)] for example in dev)]
    if args.prompt == pointer.PROMPT:
        tokenizer = pointer.load_tokenizer(args.base, args.force_bos)
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(os.path.join(args.base, "tokenizer.json"))

    def write(
        selected: list[tuple[str, str, Any, list]], path: str, margin: int, every: bool
    ) -> dict:
        dropped: collections.Counter = collections.Counter()
        rows = []
        covered = {identifier for identifier, _lane, _example, teachers in selected if teachers}
        for identifier, lane, example, covering in selected:
            try:
                row = row_from_example(example, identifier, lane, args.ordinal_smoothing, covering)
            except ValueError as error:  # a malformed example or teacher row stops the conversion
                raise SystemExit(str(error)) from None
            try:
                if args.prompt == pointer.PROMPT:
                    pointer.prompts(row, every)
            except (ValueError, TypeError, KeyError):
                dropped["not_renderable"] += 1
                continue
            rows.append(row)
        if args.prompt == pointer.PROMPT:
            lengths = pointer.prompt_tokens(rows, tokenizer, every_phrasing=every)
        else:
            lengths = []
            for row in rows:
                try:
                    lengths.append(prompts.prompt_tokens(row, tokenizer, args.prompt))
                except (ValueError, TypeError, KeyError):
                    lengths.append(-1)
        kept = with_teacher = 0
        with open(path, "w", encoding="utf-8") as handle:
            for row, n in zip(rows, lengths, strict=True):
                if n < 0:
                    dropped["not_renderable"] += 1
                    continue
                if n + margin > args.max_length:
                    dropped["over_max_length"] += 1
                    continue
                row["source"]["input_tokens"] = n
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                with_teacher += row["id"] in covered
        return {"rows": kept, "teacher_rows": with_teacher, "dropped": dict(dropped)}

    train_out = os.path.join(args.out, "train.jsonl")
    report = {
        "source": "examples",
        "prompt": args.prompt,
        "tokenizer_sha256": sha256_file(os.path.join(args.base, "tokenizer.json")),
        "force_bos": bool(args.force_bos),
        "ordinal_smoothing": args.ordinal_smoothing,
        "teachers": [
            {"weight": float(g[0]), "files": [os.path.basename(f) for f in g[1:]]}
            for g in args.teacher or []
        ],
        "examples": [os.path.basename(f) for f in args.examples],
        "dev_from": (
            {"files": [os.path.basename(f) for f in args.val_files]}
            if args.val_files
            else {"val_fraction": args.val_fraction, "val_split_seed": args.val_split_seed}
        ),
        "max_length": args.max_length,
        "length_margin": LENGTH_MARGIN,
        "train": write(
            [(*examples[i], teachers[i]) for i in train], train_out, LENGTH_MARGIN, every=True
        ),
        "dev": write(
            [(identifier, lane, example, []) for identifier, lane, example in held],
            os.path.join(args.out, "dev.jsonl"),
            0,
            every=False,
        ),
    }
    report["train"]["sha256"] = sha256_file(train_out)
    return report


def main(argv: list[str] | None = None) -> None:
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
    ap.add_argument(
        "--examples", nargs="+", help="v1 examples to convert instead of decision rows (below)"
    )
    ap.add_argument(
        "--teacher",
        nargs="+",
        action="append",
        metavar="WEIGHT FILE",
        help="--examples: a weight, then teacher files whose distributions are mixed into the targets",
    )
    ap.add_argument("--ordinal-smoothing", type=float, default=0.1, help="--examples: as v1's")
    ap.add_argument("--val-fraction", type=float, default=0.03, help="--examples: as v1's split")
    ap.add_argument("--val-split-seed", type=int, default=0, help="--examples: the split's seed")
    ap.add_argument("--val-files", nargs="+", help="--examples: examples to hold out instead")
    ap.add_argument(
        "--force-bos",
        action="store_true",
        help="--prompt numbered: a BOS on every prompt (Gemma 4 -it)",
    )
    ap.add_argument(
        "--prompt", choices=(*prompts.TEMPLATES, pointer.PROMPT), default="strands-chat"
    )
    ap.add_argument("--score", choices=("as-choice", "drop"), default="as-choice")
    ap.add_argument(
        "--max-length",
        type=int,
        default=16384,
        help="rows whose prompt + 16 exceeds this are dropped",
    )
    ap.add_argument("--limit", type=int, default=0, help="rows per source file, for tests")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)
    began = time.time()
    if args.examples:
        os.makedirs(args.out, exist_ok=True)
        report = convert_examples(args)
        report["seconds"] = round(time.time() - began)
        with open(
            os.path.join(args.out, "train.jsonl.manifest.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        return
    if args.prompt == pointer.PROMPT:
        raise SystemExit(
            "decision rows convert for the token readout; the pointer head converts --examples"
        )
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
