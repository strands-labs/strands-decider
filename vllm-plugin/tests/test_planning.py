"""The worker's boundary parsing and the per-step segment plan, checked against the mask. Needs numpy only."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
from strands_decider_vllm.planning import plan_segments, prefix_end_of  # noqa: E402

# --- prefix_end_of: the worker never raises -------------------------------------------------------------------------


@pytest.mark.parametrize("extra_args", [None, {}, {"other": 3}])
def test_no_boundary_is_none(extra_args):
    assert prefix_end_of(extra_args, 10) is None


@pytest.mark.parametrize("value", [1, 5, 9])
def test_valid_boundary(value):
    s = prefix_end_of({"prefix_end": value}, 10)
    assert s == value and type(s) is int


@pytest.mark.parametrize("value", [0, 10, 11, -1, 5.0, 5.5, math.nan, True, False, "5", None, [5]])
def test_invalid_boundary_is_none_not_an_exception(value):
    # The engine core refuses these with HTTP 400; should one reach a worker anyway, it is served causally.
    assert prefix_end_of({"prefix_end": value}, 10) is None


# --- plan_segments against the mask ---------------------------------------------------------------------------------


def spec_keys(p: int, s: int, length: int) -> range:
    """Keys the mask lets prompt position p see: the state [0, S) for a state token, the prompt [0, L) otherwise.
    A position at or after L is an output token, causal: [0, p]."""
    if p >= length:
        return range(p + 1)
    return range(s) if p < s else range(length)


def planned_keys(plan, scheduled: int, start: int) -> dict[int, range]:
    """Query position -> keys it attends to under a one-request plan (bidir: [0, n_keys); causal: [0, p])."""
    bidir, rest = plan if plan is not None else ([], [(0, 0, scheduled, -1)])
    keys: dict[int, range] = {}
    for _, a, b, n_keys in bidir:
        for t in range(a, b):
            assert start + t not in keys, "a token is in two segments"
            keys[start + t] = range(n_keys)
    for _, a, b, n_keys in rest:
        assert n_keys == -1
        for t in range(a, b):
            assert start + t not in keys, "a token is in two segments"
            keys[start + t] = range(start + t + 1)
    return keys


def check_tails(plan, query_start_loc):
    """Every causal segment is its row's tail, so the causal call's sequence length (the row's end) is right."""
    if plan is None:
        return
    rows = [r for r, *_ in plan[1]]
    assert len(rows) == len(set(rows)), "at most one causal segment per row"
    for row, _, b, _ in plan[1]:
        assert b == int(query_start_loc[row + 1])


@pytest.mark.parametrize("length", range(2, 11))
def test_plan_reproduces_the_mask_for_every_boundary_and_cache_hit(length):
    for s in range(1, length):
        for cached in range(length):  # prefix-cache hit c: tokens [c, L) are scheduled
            scheduled = length - cached
            qsl = np.array([0, scheduled])
            plan = plan_segments({0: (s, length)}, qsl, np.array([length]), 1)
            check_tails(plan, qsl)
            keys = planned_keys(plan, scheduled, cached)
            assert sorted(keys) == list(range(cached, length)), (s, cached)
            for p in range(cached, length):
                assert keys[p] == spec_keys(p, s, length), (s, cached, p)


@pytest.mark.parametrize("length", range(2, 9))
def test_resumed_request_masks_the_prompt_and_runs_its_outputs_causally(length):
    # After a preemption, Model Runner V2 re-sends prompt + k outputs as prefill: rows [c, L + k) are scheduled.
    for s in range(1, length):
        for k in range(1, 4):
            end = length + k
            for computed in range(end):
                scheduled = end - computed
                qsl = np.array([0, scheduled])
                plan = plan_segments({0: (s, length)}, qsl, np.array([end]), 1)
                check_tails(plan, qsl)
                keys = planned_keys(plan, scheduled, computed)
                assert sorted(keys) == list(range(computed, end))
                for p in range(computed, end):
                    assert keys[p] == spec_keys(p, s, length), (s, k, computed, p)


@pytest.mark.parametrize("length", range(3, 10))
def test_a_prompt_scheduled_in_parts_does_not_raise_and_reads_only_computed_keys(length):
    # Unreachable with the startup checks; if it happens, no row may attend to a key the step has not computed.
    for s in range(1, length):
        for computed in range(length - 1):
            for end in range(computed + 1, length):
                scheduled = end - computed
                qsl = np.array([0, scheduled])
                plan = plan_segments({0: (s, length)}, qsl, np.array([end]), 1)
                check_tails(plan, qsl)
                keys = planned_keys(plan, scheduled, computed)
                assert sorted(keys) == list(range(computed, end))
                for p, ks in keys.items():
                    assert ks.stop <= end, (s, computed, end, p)
                    if p < s:
                        assert ks == range(min(s, end)) or ks == range(p + 1)


def test_cold_prefill_is_state_then_question():
    plan = plan_segments({0: (10, 25)}, np.array([0, 25]), np.array([25]), 1)
    assert plan == ([(0, 0, 10, 10), (0, 10, 25, 25)], [])


def test_cache_hit_inside_the_state():
    # 4 state tokens cached: state queries [4, 10) over keys [0, 10), question queries [10, 25) over [0, 25)
    plan = plan_segments({0: (10, 25)}, np.array([0, 21]), np.array([25]), 1)
    assert plan == ([(0, 0, 6, 10), (0, 6, 21, 25)], [])


def test_cache_hit_reaching_the_question():
    plan = plan_segments({0: (10, 25)}, np.array([0, 15]), np.array([25]), 1)
    assert plan == ([(0, 0, 15, 25)], [])


def test_lone_last_token_stays_causal():
    # An exact repeat: everything but the last token cached. Causal over [0, L) is that token's mask, so no split.
    assert plan_segments({0: (10, 25)}, np.array([0, 1]), np.array([25]), 1) is None


def test_one_token_question_runs_causally_after_the_state():
    plan = plan_segments({0: (24, 25)}, np.array([0, 25]), np.array([25]), 1)
    assert plan == ([(0, 0, 24, 24)], [(0, 24, 25, -1)])


def test_resumed_request():
    # The review's case: an 828-token prompt with 2 outputs, resumed and scheduled to 830.
    plan = plan_segments({0: (42, 828)}, np.array([0, 830]), np.array([830]), 1)
    assert plan == ([(0, 0, 42, 42), (0, 42, 828, 828)], [(0, 828, 830, -1)])


def test_resumed_request_with_a_one_token_question():
    plan = plan_segments({0: (11, 12)}, np.array([0, 14]), np.array([14]), 1)
    assert plan == ([(0, 0, 11, 11)], [(0, 11, 14, -1)])


def test_resumed_request_with_its_prompt_cached_is_causal():
    assert plan_segments({0: (5, 12)}, np.array([0, 3]), np.array([15]), 1) is None


def test_mixed_batch():
    bounds = {1: (5, 12), 4: (5, 12)}
    query_start_loc = np.array([0, 1, 13, 20, 20, 24])
    seq_lens = np.array([40, 12, 7, 30, 12])
    # row 0 decode; row 1 prefix_bidir cold; row 2 causal prefill (no boundary); row 3 nothing scheduled;
    # row 4 prefix_bidir with 8 tokens cached (past S)
    bidir, rest = plan_segments(bounds, query_start_loc, seq_lens, 5)
    assert bidir == [(1, 1, 6, 5), (1, 6, 13, 12), (4, 20, 24, 12)]
    assert rest == [(0, 0, 1, -1), (2, 13, 20, -1)]


def test_no_prefix_bidir_tokens_is_none():
    assert plan_segments({}, np.array([0, 3, 4]), np.array([3, 9]), 2) is None


def test_partial_prompt_is_planned_not_refused():
    # S = 5, L = 12, only [0, 8) scheduled: state over [0, 5), question rows [5, 8) over the computed [0, 8).
    plan = plan_segments({0: (5, 12)}, np.array([0, 8]), np.array([8]), 1)
    assert plan == ([(0, 0, 5, 5), (0, 5, 8, 8)], [])
