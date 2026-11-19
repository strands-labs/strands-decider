"""The step's split against the mask it must reproduce."""

from __future__ import annotations

import pytest

from strands_decider_vllm.planning import plan_segments


def planned_keys(s: int, computed: int, length: int) -> dict[int, range]:
    """Prompt position -> the keys the plan gives it, for one request computing [computed, length)."""
    plan = plan_segments({0: s}, [0, length - computed], [length], 1)
    bidir, causal = plan if plan is not None else ([], [(0, 0, length - computed, -1)])
    keys: dict[int, range] = {}
    for _, first, end, n in bidir + causal:
        for row in range(first, end):
            p = computed + row
            assert p not in keys, "a token in two segments"
            keys[p] = range(n) if n >= 0 else range(p + 1)
    assert sorted(keys) == list(range(computed, length))
    return keys


@pytest.mark.parametrize("length", range(2, 11))
def test_every_boundary_and_cache_hit_gets_the_mask(length):
    for s in range(1, length):
        for computed in range(length):  # the prefix cache's hit
            for p, keys in planned_keys(s, computed, length).items():
                # A state token sees the state [0, S); a question token, the whole prompt.
                assert keys == (range(s) if p < s else range(length)), (s, computed, p)


def test_a_mixed_batch():
    # Row 0 a decode; row 1 a cold prefix_bidir prompt (S 5, L 12); row 2 causal; row 3 a
    # prefix_bidir prompt with 8 tokens cached; row 4 an exact repeat.
    bidir, causal = plan_segments(
        {1: 5, 3: 5, 4: 5}, [0, 1, 13, 20, 24, 25], [40, 12, 7, 12, 12], 5
    )
    assert bidir == [(1, 1, 6, 5), (1, 6, 13, 12), (3, 20, 24, 12)]
    assert causal == [(0, 0, 1, -1), (2, 13, 20, -1), (4, 24, 25, -1)]
    assert plan_segments({0: 5}, [0, 1], [12], 1) is None, "an exact repeat alone needs no split"
