"""Conformance on answers (docs/spec.md, section 10): the two measures and
their bounds, computed one way for every runtime.

A runtime's code probabilities are compared with the reference's, question by question:

    top-1 agreement   the fraction of questions whose most probable code is the reference's
                      (the first in code order on a tie, for both)
    mean |dp|         the mean over questions of the largest absolute difference between the
                      two probabilities of any one of the question's codes

A runtime conforms when, on at least MIN_QUESTIONS questions, its top-1 agreement is at least
the bound and its mean |dp| at most the bound recorded for its weight format (BOUNDS): bf16's
for bf16 weights, whatever the activations' precision, and for a quantized format the bound
recorded for that format at that size, where there is one. Where there is none, there is no
claim to make, and `Agreement.conforms` raises NoRecordedBound.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

# The fewest golden questions a conformance claim rests on: with fewer, a 98% bound says
# little (a runtime at 98.8% fails a 200-question sample about one time in seven).
MIN_QUESTIONS = 1000
TOP1 = 0.98
MEAN_ABS_DP = 0.02
BF16 = "bf16"
# The bounds docs/spec.md records, (top-1 agreement at least, mean |dp| at most), by weight
# format and size. bf16's hold at every size, and are keyed (BF16, None). A quantized format's
# hold only at the sizes it was measured at: MLX 8-bit (group size 64) and GGUF Q8_0 at E2B.
BOUNDS: dict[tuple[str, str | None], tuple[float, float]] = {
    (BF16, None): (TOP1, MEAN_ABS_DP),
    ("mlx-8bit", "E2B"): (TOP1, MEAN_ABS_DP),
    ("gguf-q8_0", "E2B"): (TOP1, MEAN_ABS_DP),
}

K = TypeVar("K", bound=Hashable)


class NoRecordedBound(ValueError):
    """A weight format, at a size, for which docs/spec.md records no bound: a runtime with it
    makes no conformance claim."""


def bounds(weight_format: str = BF16, size: str | None = None) -> tuple[float, float]:
    """The (top-1, mean |dp|) bounds of `weight_format`: bf16's whatever the size, and for a
    quantized format those recorded at `size`, which it needs. NoRecordedBound otherwise."""
    if weight_format == BF16:
        return BOUNDS[(BF16, None)]
    if size is None:
        raise NoRecordedBound(
            f"{weight_format!r} weights are bounded only at the sizes they were measured at; "
            "give the size"
        )
    try:
        return BOUNDS[(weight_format, size)]
    except KeyError:
        recorded = sorted(f"{f} at {z}" for f, z in BOUNDS if z is not None)
        raise NoRecordedBound(
            f"docs/spec.md records no bound for {weight_format!r} at {size!r}, so it makes no "
            f"conformance claim; recorded: {recorded}"
        ) from None


@dataclass(frozen=True)
class Agreement:
    """The two measures over `questions` questions."""

    questions: int
    top1: float
    mean_abs_dp: float

    def conforms(
        self,
        weight_format: str = BF16,
        size: str | None = None,
        *,
        min_questions: int = MIN_QUESTIONS,
    ) -> bool:
        """Whether these measures meet the bounds of `weight_format` at `size` (`bounds`), on
        at least `min_questions` questions. With no arguments, bf16's. NoRecordedBound for a
        format and size with no recorded bound, rather than a pass."""
        top1, mean_abs_dp = bounds(weight_format, size)
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
    probabilities: Mapping[K, Sequence[float]],
    reference: Mapping[K, Sequence[float]],
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
    """The reference probabilities of a `golden_vectors.jsonl` (docs/spec.md, section 10),
    keyed by (line number from 0, question name), for `agreement`."""
    out: dict[tuple[int, str], list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            for name, question in record["questions"].items():
                out[(line_number, name)] = _row((line_number, name), question["probabilities"])
    return out


def agreement_with_goldens(
    probabilities: Mapping[tuple[int, str], Sequence[float]], path: str | os.PathLike[str]
) -> Agreement:
    """`agreement` of a runtime's probabilities, keyed as `read_golden_vectors` keys the
    reference's, against the `golden_vectors.jsonl` at `path`."""
    return agreement(probabilities, read_golden_vectors(path))
