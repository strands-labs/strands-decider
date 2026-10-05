"""A stronger teacher on the yes/no rows, kept where it agrees with gold.

v20 distilled the frozen Qwen3.5-4B where it agrees with the gold label (data/distill.py).
On yes/no questions a larger model read the same way is a clearly better judge: on the
two answer-adequacy evaluation sets Qwen3.5-27B scores 0.744 and 0.884 (balanced) where
the 4B scores 0.645 and 0.810. `label` reads that teacher on every yes/no ("noul") row of
the `--sources` files, exactly as data/teacher.py reads the 4B; `build` keeps a row only
where the teacher's answer is the gold label (`agree`, by distill.agreeing) and lays the kept rows
over a base teacher file -- v14's replay distributions on the multi-step rows -- the new
labels winning on any row both cover.

Indices are into the concatenation of `--train-files`, which must be the experiment
config's `train_files` in order: the output is that config's `teacher_file`.

    python -m strands_decider.data.teacher_yn label --out data/teacher_yn_qwen35-27b_raw.jsonl
    python -m strands_decider.data.teacher_yn build --raw data/teacher_yn_qwen35-27b_raw.jsonl \\
        --base data/replay_v14_multistep.jsonl --out data/teacher_yn_qwen35-27b.jsonl

On N GPUs, run `label` once per GPU with `--num-shards N --shard-index i`, then once with
`--merge` (data/shards.py), as for data/teacher.py. `label --engine vllm` reads the same
distributions through vLLM (teacher.label_vllm) on one GPU, unsharded.

`replay` draws text rows that carry their teacher distribution, for image training to
replay (vision_train's `text_replay_files`, whose `teacher` rows train toward it):

    python -m strands_decider.data.teacher_yn replay --teacher data/teacher_yn_qwen35-27b.jsonl \
        --n-yes-no 2000 --n-other 1000 --out data/replay_teacher_yn.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Iterator
from typing import Any

from . import shards, teacher
from .distill import agreeing
from .format import Example, read_jsonl

TRAIN_FILES = ["data/train_v5.jsonl", "data/multistep_v14.jsonl", "data/generated_v16.jsonl",
               "data/generated_v18.jsonl", "data/adequacy_hs2.jsonl", "data/adequacy_gen.jsonl"]
# Every file but the multi-step rows, which keep v14's replay distributions.
SOURCES = [f for f in TRAIN_FILES if f != "data/multistep_v14.jsonl"]
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"


def _corpus(files: list[str]) -> Iterator[tuple[int, str, Example]]:
    """(index in the concatenation, file, example) for every row of `files`."""
    i = 0
    for f in files:
        for ex in read_jsonl(f):
            yield i, f, ex
            i += 1


def select(files: list[str], sources: list[str]) -> tuple[list[int], list[Example]]:
    """The concatenation index and the example of every yes/no row of `sources`."""
    unknown = set(sources) - set(files)
    if unknown:
        raise ValueError(f"sources not among the train files: {sorted(unknown)}")
    picked = [(i, ex) for i, f, ex in _corpus(files) if f in sources and ex.kind == "noul"]
    return [i for i, _ in picked], [ex for _, ex in picked]


def agree(files: list[str], raw: dict[int, list[float]]) -> tuple[dict[int, list[float]], list[str]]:
    """The labelled rows whose argmax is the gold label, and a per-file report of how
    often the teacher agreed."""
    rows = [(f, ex) for _, f, ex in _corpus(files)]
    kept = {r["i"]: r["probs"] for r in agreeing(
        ({"kind": ex.kind, "label": ex.label, "options": ex.options} for _, ex in rows), raw,
        kinds=("noul",))}
    report = []
    for f in files:
        idx = [i for i, (g, _) in enumerate(rows) if g == f and i in raw]
        if idx:
            agree = [i for i in idx if i in kept]
            conf = sum(kept[i][rows[i][1].label] for i in agree) / max(len(agree), 1)
            report.append(f"{f:<28} labelled {len(idx):>6,}  agree {len(agree) / len(idx):.3f}  "
                          f"mean P(gold) where kept {conf:.3f}")
    return kept, report


def _label(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    sharded = shards.sharded(ap, args)
    if args.engine == "vllm" and args.num_shards > 1:
        ap.error("--engine vllm runs as one process; it does not shard")
    gidx, examples = select(args.train_files, args.sources)
    out = shards.path(args.out, args.shard_index, args.num_shards) if sharded else args.out
    done: dict[int, list[float]] = {}  # by concatenation index; rerunning resumes
    if args.merge:
        done = shards.merge(args.out, args.num_shards)
    elif os.path.exists(out):
        done = shards.read(out)
        print(f"resuming: {len(done):,} rows already labelled")
    if not args.merge:
        print(f"labelling {len(examples):,} yes/no rows of {', '.join(args.sources)} ({args.engine})")
        skip = {j for j, i in enumerate(gidx) if i in done}
        t0 = time.time()
        with open(out, "a", encoding="utf-8") as fh:
            def sink(j: int, p: list[float]) -> None:
                fh.write(json.dumps({"i": gidx[j], "probs": [round(x, 6) for x in p]}) + "\n")
                fh.flush()
                done[gidx[j]] = p
            if args.engine == "vllm":
                teacher.label_vllm(args.model, args.revision, examples, skip=skip, sink=sink)
            else:
                model, tok = teacher.load(args.model, args.revision)
                teacher.label(model, tok, examples, max_batch_tokens=args.max_batch_tokens,
                              max_batch=args.max_batch, skip=skip, sink=sink,
                              num_shards=args.num_shards, shard_index=args.shard_index or 0)
        print(f"labelled in {(time.time() - t0) / 60:.1f} min")
    with open(out, "w", encoding="utf-8") as fh:  # in row order, however many resumes it took
        for i in sorted(done):
            fh.write(json.dumps({"i": i, "probs": [round(x, 6) for x in done[i]]}) + "\n")
    if sharded:
        shards.mark_done(out)


def _build(args: argparse.Namespace) -> None:
    raw = shards.read(args.raw)
    base = shards.read(args.base) if args.base else {}
    kept, report = agree(args.train_files, raw)
    print("\n".join(report))
    merged = {**base, **kept}
    with open(args.out, "w", encoding="utf-8") as fh:
        for i in sorted(merged):
            fh.write(json.dumps({"i": i, "probs": merged[i]}) + "\n")
    print(f"{args.out}: {len(merged):,} rows ({len(base):,} base, {len(kept):,} kept teacher rows, "
          f"{len(kept.keys() & base.keys()):,} base rows replaced)")


def replay_rows(files: list[str], teacher_file: str, n_yes_no: int, n_other: int,
                seed: int = 0) -> list[dict[str, Any]]:
    """`n_yes_no` yes/no rows and `n_other` other rows of `files` that the teacher file
    covers, drawn at random, each as its example with `teacher` (canonical option order)."""
    dist = shards.read(teacher_file)
    yes_no: list[tuple[int, Example]] = []
    other: list[tuple[int, Example]] = []
    for i, _, ex in _corpus(files):
        if i in dist:
            (yes_no if ex.kind == "noul" else other).append((i, ex))
    if n_yes_no > len(yes_no) or n_other > len(other):
        raise ValueError(f"the teacher covers {len(yes_no):,} yes/no and {len(other):,} other rows; "
                         f"asked for {n_yes_no:,} and {n_other:,}")
    rng = random.Random(seed)
    picked = sorted(rng.sample(yes_no, n_yes_no) + rng.sample(other, n_other), key=lambda r: r[0])
    return [{**json.loads(ex.to_json()), "teacher": dist[i]} for i, ex in picked]


def _replay(args: argparse.Namespace) -> None:
    rows = replay_rows(args.train_files, args.teacher, args.n_yes_no, args.n_other, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{args.out}: {len(rows):,} rows ({args.n_yes_no:,} yes/no) with teacher distributions")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="A stronger teacher on the yes/no rows, where it agrees with gold.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="the teacher's distribution on every yes/no row of --sources")
    lb.add_argument("--out", required=True)
    lb.add_argument("--train-files", nargs="+", default=TRAIN_FILES)
    lb.add_argument("--sources", nargs="+", default=SOURCES)
    lb.add_argument("--model", default=MODEL)
    lb.add_argument("--revision", default=REVISION)
    lb.add_argument("--max-batch-tokens", type=int, default=32000)
    lb.add_argument("--max-batch", type=int, default=128)
    lb.add_argument("--engine", choices=("hf", "vllm"), default="hf",
                    help="hf: data/teacher.py's batched forward; vllm: teacher.label_vllm")
    shards.add_args(lb)
    bd = sub.add_parser("build", help="keep the rows that agree with gold, over a base teacher file")
    bd.add_argument("--raw", required=True)
    bd.add_argument("--base", default="data/replay_v14_multistep.jsonl")
    bd.add_argument("--out", required=True)
    bd.add_argument("--train-files", nargs="+", default=TRAIN_FILES)
    rp = sub.add_parser("replay", help="text rows with their teacher distribution, for image training")
    rp.add_argument("--teacher", required=True, help="a teacher file `build` wrote")
    rp.add_argument("--n-yes-no", type=int, default=2000)
    rp.add_argument("--n-other", type=int, default=1000)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--out", required=True)
    rp.add_argument("--train-files", nargs="+", default=TRAIN_FILES)
    args = ap.parse_args(argv)
    if args.cmd == "label":
        _label(args, lb)
    elif args.cmd == "build":
        _build(args)
    else:
        _replay(args)


if __name__ == "__main__":
    main()
