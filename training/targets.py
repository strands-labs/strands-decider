"""What a training row asks and what it should answer: the option list, the answer, the [rows, 255] target matrix the
loss compares against, and the shuffling of option order.

training/convert.py writes each row as one question about one state:

    {"id", "suite", "family", "state", "question": {"type", "criteria", "instructions"?}, "target", "label",
     "source": {"dataset", "lane", "input_tokens", "fixed_order"?, ...}}

Two question types reach the trainer. A "choice" names its options in `criteria` ({name: description or null}); a
"noul" is a yes/no question whose options are False and True. By default convert.py turns a score question into a
choice whose levels must stay in order and marks it `source.fixed_order`. `target` takes one of three forms: a list of
probabilities in criteria order, P(true) for a noul, or the name of the right option. `label` is the single answer the
dev metrics check.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence

import torch

from strands_decider.core import MAX_CODES

Label = str | int | bool


def option_labels(question: dict) -> list[Label]:
    """The labels the codes A, B, ... stand for, in order: the criteria names of a choice, False and True for a noul."""
    if question["type"] == "noul":
        return [False, True]
    if question["type"] == "choice":
        return list(question["criteria"])
    raise ValueError(
        f"question type {question['type']!r}: the trainer takes choice and noul rows (training/convert.py "
        "makes a score question a fixed-order choice)"
    )


def answer_label(row: dict) -> Label:
    """The one right answer of a row: its `label`, or its `target` when the target names an option."""
    label = row.get("label", row["target"])
    if not isinstance(label, str | int | bool):
        raise ValueError(
            f"{row['id']}: no single answer (give a label when the target is a distribution)"
        )
    return label


def option_index(labels: Sequence[Label], label: Label) -> int:
    """Where `label` sits in `labels`. The types must agree as well as the values, so True does not find 1."""
    for index, value in enumerate(labels):
        if type(value) is type(label) and value == label:
            return index
    raise ValueError(f"{label!r} is none of the options {list(labels)[:4]}...")


def target_matrix(rows: Sequence[dict], device: torch.device) -> torch.Tensor:
    """A float32 [rows, 255] matrix: row i spreads probability over the first k codes, k being its option count, and
    holds 0 on the rest.

    A target whose probabilities are negative, not finite, of the wrong length or not summing to 1 raises. The training
    index calls this on every row up front, so a bad row fails the run at start-up rather than mid-epoch."""
    values = torch.zeros((len(rows), MAX_CODES), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        labels, target = option_labels(row["question"]), row["target"]
        if isinstance(target, list):
            distribution = target
        elif row["question"]["type"] == "noul":
            positive = float(target)
            distribution = [1.0 - positive, positive]
        else:
            distribution = [float(label == target) for label in labels]
        if (
            len(distribution) != len(labels)
            or any(not math.isfinite(p) or p < 0 for p in distribution)
            or abs(sum(distribution) - 1) > 1e-6
        ):
            raise ValueError(
                f"{row['id']}: the target is not a probability distribution over its options"
            )
        values[index, : len(distribution)] = torch.tensor(distribution, device=device)
    return values


def shuffle_options(rows: Sequence[dict], rng: random.Random) -> list[dict]:
    """New copies of `rows` in which every choice lists its options in a random order and a list target follows them.

    A noul is left alone. A fixed-order row (score levels) keeps its order too, but `rng` still shuffles a list for it
    and throws the result away, so the random stream, and with it the order every later row gets, is the same whichever
    rows are fixed. The code an option is printed with stays its answer, so the shuffled row asks the same question."""
    result = []
    for row in rows:
        new = copy.deepcopy(row)
        if new["question"]["type"] == "choice":
            criteria = new["question"]["criteria"]
            target = new["target"]
            weights = dict(zip(criteria, target, strict=True)) if isinstance(target, list) else None
            items = list(criteria.items())
            rng.shuffle(items)
            if not row["source"].get("fixed_order"):
                new["question"]["criteria"] = dict(items)
                if weights is not None:
                    new["target"] = [weights[key] for key, _ in items]
        result.append(new)
    return result
