"""The per-step plan of the prefix_bidir split: which scheduled rows attend to which keys. Needs numpy only.

A request's scheduled tokens ``[c, e)`` (c being what is already computed or cached, e the step's end) become up to
two non-causal segments over the prompt and one causal tail:

- state queries ``[c, S)`` over keys ``[0, S)`` (absent when c reaches S);
- question queries ``[max(c, S), L)`` over keys ``[0, L)``;
- rows from L on, which only a request resumed after a preemption with output tokens has, causal as in a decode.

The plugin's startup checks make a prompt run to its end in one step (e >= L). Should a prompt nevertheless be
scheduled in parts (e < L), its rows attend only to keys already computed (``[0, min(S, e))`` for the state,
``[0, e)`` for the question) instead of raising; the model state logs that step as an error.
"""

from __future__ import annotations

import numpy as np

from . import PREFIX_END_KEY


def prefix_end_of(extra_args: dict | None, prompt_len: int) -> int | None:
    """The request's boundary S from ``vllm_xargs``, or None if it sent none or one with S outside [1, L).

    The engine core refuses such requests before they are scheduled; the worker serves anything that still arrives
    without a valid boundary causally rather than raise.
    """
    if not extra_args or PREFIX_END_KEY not in extra_args:
        return None
    s = extra_args[PREFIX_END_KEY]
    if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s < prompt_len:
        return None
    return s


def plan_segments(
    bounds: dict[int, tuple[int, int]],
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    num_reqs: int,
) -> tuple[list, list] | None:
    """Segments of a step. ``bounds`` maps batch rows that are prefilling prefix_bidir requests to (S, L).

    Returns ``(bidir, rest)``, each a list of ``(row, q_start, q_end, n_keys)`` in token rows of the step. A ``bidir``
    segment attends non-causally to keys ``[0, n_keys)``. A ``rest`` segment (n_keys -1) is causal over its request's
    keys and always ends at its row's end, which is what a causal call with the row's own sequence length computes.
    None when no row computes prefix_bidir tokens.
    """
    bidir: list[tuple[int, int, int, int]] = []
    rest: list[tuple[int, int, int, int]] = []
    for row in range(num_reqs):
        q0, q1 = int(query_start_loc[row]), int(query_start_loc[row + 1])
        if q1 == q0:
            continue
        if row not in bounds:
            rest.append((row, q0, q1, -1))
            continue
        s, length = bounds[row]
        end = int(seq_lens[row])
        start = end - (q1 - q0)
        prompt_end = min(end, length)  # < L only for a prompt scheduled in parts
        if start < s:
            # State queries over the state, or over as much of it as is computed.
            bidir.append((row, q0, q0 + (min(end, s) - start), min(end, s)))
        qa = max(start, s)
        if prompt_end - qa == 1:
            # The question's only scheduled token is the prompt's last computed one: causal attention over
            # [0, prompt_end) is its mask, so it joins the causal tail. An exact repeat served from the cache computes
            # only this token; such a step needs no split and can run as a FULL CUDA graph.
            tail = qa
        else:
            if prompt_end - qa > 1:
                bidir.append((row, q0 + (qa - start), q0 + (prompt_end - start), prompt_end))
            tail = max(start, prompt_end)
        if tail < end:
            rest.append((row, q0 + (tail - start), q1, -1))
    if not bidir:
        return None
    return bidir, rest
