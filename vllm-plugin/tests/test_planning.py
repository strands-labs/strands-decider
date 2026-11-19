"""The step's split against the mask it must reproduce."""

from __future__ import annotations

import pytest

from strands_decider_vllm.planning import plan_segments


def mask(p: int, s: int, length: int) -> range:
    """The keys prompt position p sees: the state [0, S) for a state token, else [0, L)."""
    return range(s) if p < s else range(length)


def planned(s: int, computed: int, end: int) -> dict[int, range]:
    """Position -> the keys the plan gives it, for one request scheduled over [computed, end)."""
    plan = plan_segments({0: s}, [0, end - computed], [end], 1)
    bidir, causal = plan if plan is not None else ([], [(0, 0, end - computed, -1)])
    keys: dict[int, range] = {}
    for _, a, b, n in bidir + causal:
        for t in range(a, b):
            assert computed + t not in keys, "a token in two segments"
            keys[computed + t] = range(n) if n >= 0 else range(computed + t + 1)
    assert sorted(keys) == list(range(computed, end))
    return keys


@pytest.mark.parametrize("length", range(2, 11))
def test_every_boundary_and_cache_hit_gets_the_mask(length):
    for s in range(1, length):
        for computed in range(length):  # the prefix cache's hit
            for p, keys in planned(s, computed, length).items():
                assert keys == mask(p, s, length), (s, computed, p)


@pytest.mark.parametrize("length", range(3, 10))
def test_a_prompt_scheduled_in_parts_reads_only_computed_keys(length):
    # The scheduler wrapper and the startup checks prevent it; the plan must still not read ahead.
    for s in range(1, length):
        for computed in range(length - 1):
            for end in range(computed + 1, length):
                for p, keys in planned(s, computed, end).items():
                    want = range(min(s, end)) if p < s else range(end)
                    assert keys == want, (s, end, p)


def test_a_mixed_batch():
    # Row 0 a decode; row 1 a cold prefix_bidir prompt (S 5, L 12); row 2 causal; row 3 nothing
    # scheduled; row 4 a prefix_bidir prompt with 8 tokens cached; row 5 an exact repeat.
    bidir, causal = plan_segments(
        {1: 5, 4: 5, 5: 5}, [0, 1, 13, 20, 20, 24, 25], [40, 12, 7, 30, 12, 12], 6
    )
    assert bidir == [(1, 1, 6, 5), (1, 6, 13, 12), (4, 20, 24, 12)]
    assert causal == [(0, 0, 1, -1), (2, 13, 20, -1), (5, 24, 25, -1)]
    assert plan_segments({0: 5}, [0, 1], [12], 1) is None, "an exact repeat alone needs no split"
