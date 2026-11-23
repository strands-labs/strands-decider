"""The prefix_bidir split of one step: which of the step's token rows attend to which keys.

A prefix_bidir request computes its prompt's tokens [c, L) in one step, c being those its prefix
cache holds. Its state queries [c, S) attend to keys [0, S) and its question queries [max(c, S), L)
to keys [0, L), both non-causally. Every other row stays causal: requests without a boundary, and a
prefix_bidir row that computes only the prompt's last token (an exact repeat), whose mask causal
attention over [0, L) already is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

Segment = tuple[int, int, int, int]  # (batch row, first token row, end token row, keys)


def plan_segments(
    bounds: Mapping[int, int],
    query_start_loc: Sequence[int],
    seq_lens: Sequence[int],
    num_reqs: int,
) -> tuple[list[Segment], list[Segment]] | None:
    """``(bidir, causal)`` for the batch rows ``bounds`` maps to their S, or None if no row needs the
    split. A bidir segment attends to keys [0, keys); a causal one (keys -1) is its row's own."""
    bidir: list[Segment] = []
    causal: list[Segment] = []
    for row in range(num_reqs):
        q0, q1 = int(query_start_loc[row]), int(query_start_loc[row + 1])
        s = bounds.get(row)
        if s is None or q1 - q0 == 1:
            causal.append((row, q0, q1, -1))
            continue
        length = int(seq_lens[row])
        question = q0 + max(s - (length - (q1 - q0)), 0)  # the row of the first question token
        if question > q0:
            bidir.append((row, q0, question, s))
        bidir.append((row, question, q1, length))
    return (bidir, causal) if bidir else None
