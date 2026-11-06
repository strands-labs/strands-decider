"""The Model Runner V2 state: boundaries per request, the gating of prepare_attn, and the split metadata it attaches.
Needs torch and vLLM; the plan itself is tested in test_planning.py."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
model_state = pytest.importorskip("strands_decider_v2_vllm.model_state")
from vllm.config.compilation import CUDAGraphMode  # noqa: E402

from strands_decider_v2_vllm.attention import SPLIT_ATTR  # noqa: E402

plan_segments = model_state.plan_segments
State = model_state.PrefixBidirModelState


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


def metadata_for(query_start_loc, seq_lens, block_base=10):
    return FakeMetadata(
        num_actual_tokens=int(query_start_loc[-1]),
        max_query_len=int(np.diff(query_start_loc).max()),
        query_start_loc=torch.tensor(query_start_loc),
        max_seq_len=int(max(seq_lens)),
        seq_lens=torch.tensor(seq_lens),
        block_table=torch.arange(len(seq_lens)).view(-1, 1) * block_base,
    )


def test_rows_are_a_slice_when_contiguous():
    rows, _ = model_state._rows([(0, 0, 4, 9), (1, 4, 7, 9)], torch.device("cpu"))
    assert rows == slice(0, 7)
    rows, idx = model_state._rows([(0, 0, 2, 9), (2, 5, 7, 9)], torch.device("cpu"))
    assert rows.tolist() == [0, 1, 5, 6] and idx.tolist() == [0, 1, 5, 6]


def test_attach_builds_one_split_per_metadata_object():
    query_start_loc = np.array([0, 1, 13, 20, 20, 24])
    seq_lens = np.array([40, 12, 7, 30, 12])
    plan = plan_segments({1: (5, 12), 4: (5, 12)}, query_start_loc, seq_lens, 5)
    md = metadata_for(query_start_loc, seq_lens)
    metadata = {"layers.0.self_attn.attn": md, "layers.1.self_attn.attn": md, "other": None}
    State._attach(None, metadata, plan)  # uses no instance state

    split = getattr(md, SPLIT_ATTR)
    assert split.bidir_rows.tolist() == list(range(1, 13)) + [20, 21, 22, 23]
    assert split.rest_rows.tolist() == [0] + list(range(13, 20))
    b = split.bidir
    assert not b.causal and b.scheduler_metadata is None and not hasattr(b, SPLIT_ATTR)
    assert b.query_start_loc.tolist() == [0, 5, 12, 16]
    # keys: [0, S) for the state, [0, L) for the questions
    assert b.seq_lens.tolist() == [5, 12, 12]
    assert b.block_table.flatten().tolist() == [10, 10, 40]
    assert (b.num_actual_tokens, b.max_query_len, b.max_seq_len) == (16, 7, 12)
    r = split.rest
    assert r.causal and r.scheduler_metadata is None
    assert r.query_start_loc.tolist() == [0, 1, 8]
    assert r.seq_lens.tolist() == [40, 7] and r.block_table.flatten().tolist() == [0, 20]
    assert (r.num_actual_tokens, r.max_query_len, r.max_seq_len) == (8, 7, 40)


def test_attach_without_causal_rows():
    plan = plan_segments({0: (3, 8)}, np.array([0, 8]), np.array([8]), 1)
    md = metadata_for(np.array([0, 8]), np.array([8]), block_base=7)
    State._attach(None, {"a": md}, plan)
    split = getattr(md, SPLIT_ATTR)
    assert split.bidir_rows == slice(0, 8) and split.rest is None and split.rest_rows is None
    assert split.bidir.seq_lens.tolist() == [3, 8]


def test_attach_shares_the_plan_across_kv_cache_groups_but_not_their_block_tables():
    query_start_loc, seq_lens = np.array([0, 1, 13]), np.array([40, 12])
    plan = plan_segments({1: (5, 12)}, query_start_loc, seq_lens, 2)
    full, sliding = (
        metadata_for(query_start_loc, seq_lens, 10),
        metadata_for(query_start_loc, seq_lens, 100),
    )
    State._attach(None, {"full.0": full, "sliding.0": sliding, "full.1": full}, plan)
    a, b = getattr(full, SPLIT_ATTR), getattr(sliding, SPLIT_ATTR)
    assert a.bidir.query_start_loc is b.bidir.query_start_loc, "built once per device"
    assert a.bidir.seq_lens is b.bidir.seq_lens and a.rest.query_start_loc is b.rest.query_start_loc
    assert a.bidir.block_table.flatten().tolist() == [10, 10]
    assert b.bidir.block_table.flatten().tolist() == [100, 100]
    assert a.rest.block_table.flatten().tolist() == [
        0
    ] and b.rest.block_table.flatten().tolist() == [0]


def test_attach_for_a_resumed_request():
    # 828-token prompt, S = 42, resumed with 2 outputs: the outputs run causally with the row's own length.
    plan = plan_segments({0: (42, 828)}, np.array([0, 830]), np.array([830]), 1)
    md = metadata_for(np.array([0, 830]), np.array([830]))
    State._attach(None, {"a": md}, plan)
    split = getattr(md, SPLIT_ATTR)
    assert split.bidir_rows == slice(0, 828) and split.rest_rows == slice(828, 830)
    assert split.bidir.seq_lens.tolist() == [42, 828] and split.bidir.query_start_loc.tolist() == [
        0,
        42,
        828,
    ]
    assert split.rest.seq_lens.tolist() == [830] and split.rest.query_start_loc.tolist() == [0, 2]
    assert split.rest.causal


# --- add_request and prepare_attn (vLLM's own DefaultModelState methods stubbed) -------------------------------------


@pytest.fixture
def state(monkeypatch):
    default = model_state.DefaultModelState
    monkeypatch.setattr(default, "add_request", lambda self, i, d: None)
    monkeypatch.setattr(default, "remove_request", lambda self, rid: None)
    metadata = {}
    monkeypatch.setattr(default, "prepare_attn", lambda self, *a, **k: metadata)
    s = object.__new__(State)
    s._bounds, s._warned_causal, s._warned_partial = {}, False, False
    s.metadata = metadata
    return s


def new_request(rid, prompt_len, extra_args, outputs=0):
    return SimpleNamespace(
        req_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        prefill_token_ids=list(range(prompt_len + outputs)),
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )


def test_add_request_keeps_the_boundary_and_the_prompt_length(state):
    state.add_request(0, new_request("a", 20, {"prefix_end": 9}))
    state.add_request(
        1, new_request("b", 20, {"prefix_end": 9}, outputs=3)
    )  # resumed: L is still the prompt's
    assert state._bounds == {"a": (9, 20), "b": (9, 20)}
    state.remove_request("a")
    assert state._bounds == {"b": (9, 20)}


@pytest.mark.parametrize("value", ["9", False, True, 0, 20, 9.5, [9], None])
def test_add_request_never_raises_for_request_data(state, caplog, value):
    with caplog.at_level(logging.ERROR, logger="vllm.strands_decider_v2"):
        state.add_request(0, new_request("a", 20, {"prefix_end": value}))
    assert state._bounds == {} and "should have refused it" in caplog.text


def test_add_request_without_a_boundary_is_causal(state):
    state.add_request(0, new_request("a", 20, None))
    state.add_request(1, SimpleNamespace(req_id="b", prompt_token_ids=None, sampling_params=None))
    assert state._bounds == {}


def batch(req_ids, prefilling, query_start_loc, seq_lens):
    return SimpleNamespace(
        num_reqs=len(req_ids),
        req_ids=list(req_ids),
        is_prefilling_np=np.array(prefilling),
        query_start_loc_np=np.array(query_start_loc),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens),
    )


def prepare(state, input_batch, mode=CUDAGraphMode.PIECEWISE, for_capture=False):
    md = metadata_for(input_batch.query_start_loc_np, input_batch.seq_lens_cpu_upper_bound.numpy())
    state.metadata.clear()
    state.metadata["layer"] = md
    state.prepare_attn(input_batch, mode, (), None, [], None, for_capture=for_capture)
    return getattr(md, SPLIT_ATTR, None)


def test_prepare_attn_splits_a_v2_prefill(state):
    state._bounds = {"a": (5, 12)}
    assert prepare(state, batch(["a"], [True], [0, 12], [12])) is not None


def test_prepare_attn_leaves_captures_decodes_and_lone_last_tokens_alone(state):
    state._bounds = {"a": (5, 12)}
    assert prepare(state, batch(["a"], [True], [0, 12], [12]), for_capture=True) is None
    assert prepare(state, batch(["a"], [False], [0, 1], [13])) is None, "a decode row"
    # An exact repeat computes only the last prompt token, even under a FULL graph: no split, no error.
    assert prepare(state, batch(["a"], [True], [0, 1], [12]), mode=CUDAGraphMode.FULL) is None
    state._bounds = {}
    assert prepare(state, batch(["x"], [True], [0, 12], [12])) is None, "no v2 request"


def test_prepare_attn_refuses_a_split_under_a_full_graph(state):
    # check_config refuses cudagraph_mode FULL, so this cannot happen; a graph would ignore the split.
    state._bounds = {"a": (5, 12)}
    with pytest.raises(RuntimeError, match="FULL"):
        prepare(state, batch(["a"], [True], [0, 12], [12]), mode=CUDAGraphMode.FULL)


def test_prepare_attn_logs_a_prompt_scheduled_in_parts_once(state, caplog):
    state._bounds = {"a": (5, 12)}
    with caplog.at_level(logging.ERROR, logger="vllm.strands_decider_v2"):
        split = prepare(state, batch(["a"], [True], [0, 8], [8]))
        prepare(state, batch(["a"], [True], [0, 8], [8]))
    assert split is not None and caplog.text.count("scheduled in parts") == 1
