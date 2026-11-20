"""How a runtime's parity tests measure its answers against the reference's.

A runtime conforms when, on at least MIN_QUESTIONS questions, its top-1 agreement is at least
TOP1 and its mean |dp| (the largest difference over a question's codes, averaged) at most
MEAN_ABS_DP.
"""

from __future__ import annotations

import os
from collections.abc import Hashable
from dataclasses import dataclass

from pydantic import BaseModel

# With fewer questions a 98% bound says little: a runtime at 98.8% fails a 200-question sample
# about one time in seven.
MIN_QUESTIONS = 1000
TOP1 = 0.98
MEAN_ABS_DP = 0.02


@dataclass(frozen=True)
class Agreement:
    questions: int
    top1: float
    mean_abs_dp: float

    def conforms(self, *, min_questions: int = MIN_QUESTIONS) -> bool:
        return (
            self.questions >= min_questions
            and self.top1 >= TOP1
            and self.mean_abs_dp <= MEAN_ABS_DP
        )


def _top1(row: list[float]) -> int:
    return max(range(len(row)), key=lambda i: (row[i], -i))  # the first on a tie


def agreement(
    probabilities: dict[Hashable, list[float]], reference: dict[Hashable, list[float]]
) -> Agreement:
    """Top-1 agreement and mean |dp| of a runtime's code probabilities against the reference's,
    both keyed by question."""
    if set(probabilities) != set(reference) or not reference:
        raise ValueError("the runtime and the reference must answer the same questions")
    agree, dp = 0, 0.0
    for key, ref in reference.items():
        ours = [float(p) for p in probabilities[key]]
        agree += _top1(ours) == _top1(ref)
        dp += max(abs(a - b) for a, b in zip(ours, ref, strict=True))
    n = len(reference)
    return Agreement(questions=n, top1=agree / n, mean_abs_dp=dp / n)


class _GoldenQuestion(BaseModel):
    probabilities: list[float]


class _GoldenLine(BaseModel):
    questions: dict[str, _GoldenQuestion]


def read_golden_vectors(path: str | os.PathLike[str]) -> dict[tuple[int, str], list[float]]:
    """A checkpoint's golden_vectors.jsonl as reference probabilities, keyed by (line number,
    question name) for `agreement`."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f):
            if line.strip():
                for name, q in _GoldenLine.model_validate_json(line).questions.items():
                    out[(number, name)] = q.probabilities
    return out
