"""The prefix_bidir split of one step: which of the step's token rows attend to which keys.

A prefix_bidir request computes its prompt's tokens [c, e) in one step (c: those already computed or
cached; e = L once the prompt is whole). Its state queries [c, S) attend to keys [0, S), and its
question queries [max(c, S), e) to keys [0, e), both non-causally. Every other row stays causal:
requests without a boundary, decodes, and a prefix_bidir row that computes only the prompt's last
token (an exact repeat), whose mask causal attention over [0, L) already is. A prefix_bidir request
never has output tokens to compute (max_tokens is 1).
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
    split. A bidir segment attends to keys [0, keys); a causal one (keys -1) ends at its row's end."""
    bidir: list[Segment] = []
    causal: list[Segment] = []
    for row in range(num_reqs):
        q0, q1 = int(query_start_loc[row]), int(query_start_loc[row + 1])
        if q1 == q0:
            continue
        s = bounds.get(row)
        if s is None or q1 - q0 == 1:
            causal.append((row, q0, q1, -1))
            continue
        end = int(seq_lens[row])
        start = end - (q1 - q0)
        if start < s:
            keys = min(end, s)  # below S only for a prompt scheduled in parts
            bidir.append((row, q0, q0 + keys - start, keys))
        question = max(start, s)
        if end > question:
            bidir.append((row, q1 - (end - question), q1, end))
    return (bidir, causal) if bidir else None
