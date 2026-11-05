"""The request boundary and the per-step segment plan (Model Runner V2 state), checked against the v2 mask on CPU."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
model_state = pytest.importorskip("strands_decider_v2_vllm.model_state")
from strands_decider_v2_vllm.attention import SPLIT_ATTR  # noqa: E402

prefix_end_of = model_state.prefix_end_of
plan_segments = model_state.plan_segments


# --- prefix_end_of -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra_args", [None, {}, {"other": 3}])
def test_no_boundary_is_none(extra_args):
    assert prefix_end_of(extra_args, 10) is None


@pytest.mark.parametrize("value, expected", [(1, 1), (5, 5), (9, 9), (5.0, 5)])
def test_valid_boundary(value, expected):
    s = prefix_end_of({"prefix_end": value}, 10)
    assert s == expected and type(s) is int


@pytest.mark.parametrize("value", [0, 10, 11, -1, 5.5, True, False, "5", None, [5]])
def test_invalid_boundary_is_refused(value):
    # S must leave at least one state token and one question token: 0 < S < L.
    with pytest.raises(ValueError, match="prefix_end must be an integer"):
        prefix_end_of({"prefix_end": value}, 10)


# --- plan_segments against the mask ---------------------------------------------------------------------------------


def spec_keys(p: int, s: int, length: int) -> range:
    """Keys the v2 mask lets query position p see: the state [0, S) for a state token, the prompt [0, L) otherwise."""
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


@pytest.mark.parametrize("length", range(2, 11))
def test_plan_reproduces_the_mask_for_every_boundary_and_cache_hit(length):
    for s in range(1, length):
        for cached in range(length):  # prefix-cache hit c: tokens [c, L) are scheduled
            scheduled = length - cached
            plan = plan_segments({0: (s, length)}, np.array([0, scheduled]), np.array([length]), 1)
            keys = planned_keys(plan, scheduled, cached)
            assert sorted(keys) == list(range(cached, length)), (s, cached)
            for p in range(cached, length):
                assert keys[p] == spec_keys(p, s, length), (s, cached, p)


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


def test_partial_prompt_is_refused():
    with pytest.raises(RuntimeError, match="computed to its end in one step"):
        plan_segments({0: (5, 12)}, np.array([0, 8]), np.array([8]), 1)


# --- metadata attached to a step -------------------------------------------------------------------------------------


@dataclass
class FakeMetadata:
    """The FlashAttentionMetadata fields the model state reads or replaces."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    causal: bool = True
    scheduler_metadata: object = "aot-schedule"


def test_rows_are_a_slice_when_contiguous():
    rows, _ = model_state._rows([(0, 0, 4, 9), (1, 4, 7, 9)], torch.device("cpu"))
    assert rows == slice(0, 7)
    rows, idx = model_state._rows([(0, 0, 2, 9), (2, 5, 7, 9)], torch.device("cpu"))
    assert rows.tolist() == [0, 1, 5, 6] and idx.tolist() == [0, 1, 5, 6]


def test_attach_builds_one_split_per_metadata_object():
    query_start_loc = np.array([0, 1, 13, 20, 20, 24])
    seq_lens = np.array([40, 12, 7, 30, 12])
    plan = plan_segments({1: (5, 12), 4: (5, 12)}, query_start_loc, seq_lens, 5)
    md = FakeMetadata(
        num_actual_tokens=24,
        max_query_len=12,
        query_start_loc=torch.tensor(query_start_loc),
        max_seq_len=40,
        seq_lens=torch.tensor(seq_lens),
        block_table=torch.arange(5).view(5, 1) * 10,
    )
    metadata = {"layers.0.self_attn.attn": md, "layers.1.self_attn.attn": md, "other": None}
    model_state.PrefixBidirModelState._attach(None, metadata, plan)  # uses no instance state

    split = getattr(md, SPLIT_ATTR)
    assert split.bidir_rows.tolist() == list(range(1, 13)) + [20, 21, 22, 23]
    assert split.rest_rows.tolist() == [0] + list(range(13, 20))
    b = split.bidir
    assert not b.causal and b.scheduler_metadata is None and not hasattr(b, SPLIT_ATTR)
    assert b.query_start_loc.tolist() == [0, 5, 12, 16]
    assert b.seq_lens.tolist() == [
        5,
        12,
        12,
    ]  # keys: [0, S) for the state, [0, L) for the questions
    assert b.block_table.flatten().tolist() == [10, 10, 40]
    assert (b.num_actual_tokens, b.max_query_len, b.max_seq_len) == (16, 7, 12)
    r = split.rest
    assert r.causal and r.scheduler_metadata is None
    assert r.query_start_loc.tolist() == [0, 1, 8]
    assert r.seq_lens.tolist() == [40, 7] and r.block_table.flatten().tolist() == [0, 20]
    assert (r.num_actual_tokens, r.max_query_len, r.max_seq_len) == (8, 7, 40)


def test_attach_without_causal_rows():
    plan = plan_segments({0: (3, 8)}, np.array([0, 8]), np.array([8]), 1)
    md = FakeMetadata(
        num_actual_tokens=8,
        max_query_len=8,
        query_start_loc=torch.tensor([0, 8]),
        max_seq_len=8,
        seq_lens=torch.tensor([8]),
        block_table=torch.tensor([[7]]),
    )
    model_state.PrefixBidirModelState._attach(None, {"a": md}, plan)
    split = getattr(md, SPLIT_ATTR)
    assert split.bidir_rows == slice(0, 8) and split.rest is None and split.rest_rows is None
    assert split.bidir.seq_lens.tolist() == [3, 8]
