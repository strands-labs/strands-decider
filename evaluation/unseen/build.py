"""The unseen-family set: 1,050 questions from four task families no model trains on, plus
RuleTaker at a held-out depth (depths 0-2 are trained on).

No training mixture (data/sources.md) contains the four families, and none contains
RuleTaker's depth 5, so the set measures how a checkpoint does on questions whose format
and subject it never saw, beside JevBench's public tasks, which share families with the
training data:

  StrategyQA      250 yes/no   implicit multi-step questions, the question alone
  RuleTaker d5    250 yes/no   depth-5 rule reasoning; training uses depths 0-2 only
  CommonsenseQA   150 choice   5 options
  ARC-Challenge   150 choice   3-5 options
  STS-B           250 score    sentence similarity on 6 levels (0-5, the gold rounded half to even)

Every download is pinned to the dataset revision the recorded runs used (REVISIONS). The
RuleTaker rows come from `strands-decider data build`'s held-out file
(data/train_v5.holdout.jsonl). One seeded generator draws everything in a fixed order,
so the same inputs give the same rows; the manifest written beside the output records the
revisions, the `datasets` version and the held-out file's sha256.

    pip install "strands-decider[train]"
    python evaluation/unseen/build.py --out data/unseen.jsonl

Rows are `Example`s (data/format.py) with a `task` of `unseen/<family>`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from typing import Any

REVISIONS = {
    "ChilleD/StrategyQA": "705562638fe1d8ca6bb98c66fc8f94d45fda8c83",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "sentence-transformers/stsb": "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308",
}
YES_NO = [["false", "The answer is no."], ["true", "The answer is yes."]]
SIMILARITY = ["completely different meaning", "mostly different, share a topic",
              "not equivalent, share some details", "roughly equivalent, some important details differ",
              "mostly equivalent, minor details differ", "completely equivalent in meaning"]
# The held-out RuleTaker tasks the draw walks over, in file order; every kept row is depth 5,
# which comes first in the file.
RULETAKER = ("ruletaker_d5", "ruletaker_natlang")


def _load(name: str, split: str, config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    return list(load_dataset(name, config, split=split, revision=REVISIONS[name]))


def strategyqa(rows: list[dict[str, Any]], rng: random.Random, n: int = 250) -> list[dict[str, Any]]:
    rng.shuffle(rows)
    return [{"kind": "noul", "state": r["question"], "instructions": "Is the true answer to this question yes?",
             "options": YES_NO, "label": int(bool(r["answer"])), "task": "unseen/strategyqa"}
            for r in rows[:n] if r.get("question") is not None and r.get("answer") is not None]


def ruletaker(lines: list[str], rng: random.Random, n: int = 250) -> list[dict[str, Any]]:
    """Each candidate row is kept with probability 1/2 until `n` are kept; the generator is
    drawn once per candidate, kept or not, so the later families' draws do not depend on n."""
    out = []
    for line in lines:
        r = json.loads(line)
        if r.get("task") in RULETAKER and rng.random() < 0.5 and len(out) < n:
            out.append({**r, "task": "unseen/" + r["task"]})
    return out


def multiple_choice(rows: list[dict[str, Any]], rng: random.Random, instructions: str, task: str,
                    n: int = 150) -> list[dict[str, Any]]:
    """An option text repeated within a question (four CommonsenseQA questions) is kept
    once, where it first appears: a question's options are named by their text, and the
    server reads one option per name."""
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] in labels:
            names = list(dict.fromkeys(texts))
            out.append({"kind": "choice", "state": r["question"], "instructions": instructions,
                        "options": [[t, t] for t in names],
                        "label": names.index(texts[labels.index(r["answerKey"])]), "task": task})
    return out


def stsb(rows: list[dict[str, Any]], rng: random.Random, n: int = 250) -> list[dict[str, Any]]:
    """Gold similarity in [0, 1] (this dataset's scale), times 5, rounded to a level (half to
    even, as Python's round does)."""
    rng.shuffle(rows)
    return [{"kind": "score", "state": f"Sentence A: {r['sentence1']}\nSentence B: {r['sentence2']}",
             "instructions": "How similar in meaning are sentence A and sentence B?",
             "options": [[str(i), d] for i, d in enumerate(SIMILARITY)],
             "label": round(r["score"] * 5), "task": "unseen/stsb"}
            for r in rows[:n]]


def build(ruletaker_file: str, seed: int = 0) -> list[dict[str, Any]]:
    """The whole set, in the order the generator draws it."""
    rng = random.Random(seed)
    rows = strategyqa(_load("ChilleD/StrategyQA", "train"), rng)
    with open(ruletaker_file, encoding="utf-8") as fh:
        rows += ruletaker(fh.readlines(), rng)
    rows += multiple_choice(_load("tau/commonsense_qa", "validation"), rng,
                            "Which answer is most plausible?", "unseen/commonsenseqa")
    rows += multiple_choice(_load("allenai/ai2_arc", "test", "ARC-Challenge"), rng,
                            "Which answer is correct?", "unseen/arc_challenge")
    return rows + stsb(_load("sentence-transformers/stsb", "test"), rng)


def main(argv: list[str] | None = None) -> None:
    import datasets

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ruletaker", default="data/train_v5.holdout.jsonl",
                    help="strands-decider data build's held-out file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = build(a.ruletaker, a.seed)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(a.ruletaker, "rb") as fh:
        held_out = hashlib.sha256(fh.read()).hexdigest()
    manifest = {"rows": len(rows), "by_task": dict(Counter(r["task"] for r in rows)), "seed": a.seed,
                "revisions": REVISIONS, "datasets": datasets.__version__,
                "ruletaker_file": a.ruletaker, "ruletaker_sha256": held_out}
    with open(a.out + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
