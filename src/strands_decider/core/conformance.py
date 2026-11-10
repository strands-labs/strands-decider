"""Conformance on answers (docs/spec.md, Golden vectors and conformance): the two measures and
their bounds, computed one way for every runtime.

A runtime's code probabilities are compared with the reference's, question by question:

    top-1 agreement   the fraction of questions whose most probable code is the reference's
                      (the first in code order on a tie, for both)
    mean |dp|         the mean over questions of the largest absolute difference between the
                      two probabilities of any one of the question's codes

A runtime conforms when, on at least MIN_QUESTIONS questions, its top-1 agreement is at least
the bound and its mean |dp| at most the bound recorded for its weight format (BOUNDS).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# The fewest golden questions a conformance claim rests on: with fewer, a 98% bound says
# little (a runtime at 98.8% fails a 200-question sample about one time in seven).
MIN_QUESTIONS = 1000
TOP1 = 0.98
MEAN_ABS_DP = 0.02
# Each weight format's bounds, (top-1 agreement at least, mean |dp| at most), as docs/spec.md
# records them. The quantized ones were measured at E2B only and hold there.
BOUNDS: dict[str, tuple[float, float]] = {
    "bf16": (TOP1, MEAN_ABS_DP),
    "mlx-8bit": (TOP1, MEAN_ABS_DP),
    "gguf-q8_0": (TOP1, MEAN_ABS_DP),
}


@dataclass(frozen=True)
class Agreement:
    """The two measures over `questions` questions."""

    questions: int
    top1: float
    mean_abs_dp: float

    def conforms(
        self,
        *,
        top1: float = TOP1,
        mean_abs_dp: float = MEAN_ABS_DP,
        min_questions: int = MIN_QUESTIONS,
    ) -> bool:
        """Whether these measures meet the bounds, on enough questions."""
        return (
            self.questions >= min_questions
            and self.top1 >= top1
            and self.mean_abs_dp <= mean_abs_dp
        )


def _row(key: Hashable, values: Any) -> list[float]:
    row = [float(v) for v in values]
    if not row or not all(math.isfinite(v) for v in row):
        raise ValueError(f"question {key!r}: probabilities must be finite, and at least one")
    return row


def _top1(row: Sequence[float]) -> int:
    return max(range(len(row)), key=lambda i: (row[i], -i))  # the first on a tie


def agreement(
    probabilities: Mapping[Hashable, Sequence[float]],
    reference: Mapping[Hashable, Sequence[float]],
) -> Agreement:
    """Top-1 agreement and mean |dp| of a runtime's code probabilities against the
    reference's. Both map the same question keys to a question's probabilities, one per code
    in code order (the question's k); ValueError if the keys or a question's k differ."""
    if set(probabilities) != set(reference):
        missing = sorted(map(repr, set(reference) - set(probabilities)))[:5]
        extra = sorted(map(repr, set(probabilities) - set(reference)))[:5]
        raise ValueError(f"the questions differ: missing {missing}, not in the reference {extra}")
    if not reference:
        raise ValueError("no questions to compare")
    agree, dp = 0, 0.0
    for key, ref_values in reference.items():
        ours, ref = _row(key, probabilities[key]), _row(key, ref_values)
        if len(ours) != len(ref):
            raise ValueError(f"question {key!r}: {len(ours)} probabilities for {len(ref)} codes")
        agree += _top1(ours) == _top1(ref)
        dp += max(abs(a - b) for a, b in zip(ours, ref, strict=True))
    n = len(reference)
    return Agreement(questions=n, top1=agree / n, mean_abs_dp=dp / n)


def read_golden_vectors(path: str | os.PathLike[str]) -> dict[tuple[int, str], list[float]]:
    """The reference probabilities of a `golden_vectors.jsonl` (docs/spec.md, Golden vectors
    and conformance), keyed by (line number from 0, question name), for `agreement`."""
    out: dict[tuple[int, str], list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            for name, question in record["questions"].items():
                out[(line_number, name)] = _row((line_number, name), question["probabilities"])
    return out
