"""Teacher distributions only where the teacher is right (v20).

v12 distilled a frozen Qwen3.5-4B on every short-task row and lost `policy` and
`routing`: the teacher disagrees with the annotators on a quarter of them. v20 keeps its
distributions only on the yes/no and choice rows where its answer matches the gold
label, so the student is only ever pulled toward answers that are right and what it
gains is the teacher's confidence. `score` rows are left out: the teacher agrees with
gold on 39-53% of them.

The output is one teacher file for `training.convert --teacher`: the agreeing rows of
`--teacher` (indices into `--corpus`) followed by every row of each `--append` file
(e.g. the replay distributions on the multi-step rows, already offset past the corpus).
Indices must not overlap.

    python -m training.data.distill --corpus data/train_v5.jsonl \
        --teacher data/teacher_v5_qwen35-4b.jsonl \
        --append data/replay_v14_multistep.jsonl --out data/teacher_v20.jsonl
"""

from __future__ import annotations

import json
from collections.abc import Iterable


def agreeing(
    corpus: Iterable[dict],
    teacher: dict[int, list[float]],
    kinds: tuple[str, ...] = ("noul", "choice"),
) -> list[dict]:
    """The teacher rows whose argmax is the gold label, for rows of the given kinds."""
    out = []
    for i, row in enumerate(corpus):
        p = teacher.get(i)
        if p is None or row["kind"] not in kinds:
            continue
        if len(p) != len(row["options"]):
            raise ValueError(
                f"row {i}: {len(p)} teacher probabilities for {len(row['options'])} options"
            )
        if max(range(len(p)), key=p.__getitem__) == row["label"]:
            out.append({"i": i, "probs": p})
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Teacher distributions where the teacher agrees with gold (v20)."
    )
    ap.add_argument("--corpus", default="data/train_v5.jsonl")
    ap.add_argument("--teacher", default="data/teacher_v5_qwen35-4b.jsonl")
    ap.add_argument("--append", nargs="*", default=["data/replay_v14_multistep.jsonl"])
    ap.add_argument("--out", default="data/teacher_v20.jsonl")
    args = ap.parse_args(argv)

    with open(args.teacher, encoding="utf-8") as fh:
        teacher = {r["i"]: r["probs"] for r in map(json.loads, fh)}
    with open(args.corpus, encoding="utf-8") as fh:
        kept = agreeing((json.loads(line) for line in fh), teacher)
    rows = list(kept)
    for path in args.append:
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    if len({r["i"] for r in rows}) != len(rows):
        raise ValueError("overlapping teacher indices")
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(
        f"{len(kept):,} of {len(teacher):,} teacher rows agree with gold (yes/no and choice); "
        f"{len(rows) - len(kept):,} appended; {len(rows):,} written to {args.out}"
    )


if __name__ == "__main__":
    main()
