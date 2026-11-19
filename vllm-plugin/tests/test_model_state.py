"""The model state: which steps get a split, and the restricted metadata it attaches. Needs vLLM, no
GPU; the plan itself is in test_planning.py."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
model_state = pytest.importorskip("strands_decider_vllm.model_state")
from strands_decider_vllm.attention import SPLIT_ATTR  # noqa: E402
from strands_decider_vllm.planning import plan_segments  # noqa: E402


@dataclass
class Metadata:
    """The FlashAttentionMetadata fields the model state reads or replaces."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    causal: bool = True
    scheduler_metadata: object = "FA3's schedule"


def metadata(query_start_loc, seq_lens, block_base=10):
    return Metadata(
        num_actual_tokens=int(query_start_loc[-1]),
        max_query_len=int(np.diff(query_start_loc).max()),
        query_start_loc=torch.tensor(query_start_loc),
        max_seq_len=int(max(seq_lens)),
        seq_lens=torch.tensor(seq_lens),
        block_table=torch.arange(len(seq_lens)).view(-1, 1) * block_base,
    )


def test_each_kv_cache_group_gets_the_split_with_its_own_block_table():
    # Row 0 a decode; row 1 a cold prefix_bidir prompt (S 5, L 12); row 2 causal; row 3 nothing
    # scheduled; row 4 a prefix_bidir prompt with 8 tokens cached.
    qsl, seq = [0, 1, 13, 20, 20, 24], [40, 12, 7, 30, 12]
    full, sliding = metadata(qsl, seq, 10), metadata(qsl, seq, 100)
    model_state._attach(
        {"a": full, "b": sliding, "c": full, "d": None}, *plan_segments({1: 5, 4: 5}, qsl, seq, 5)
    )
    for md, base in ((full, 10), (sliding, 100)):
        (bidir_rows, bidir), (causal_rows, causal) = getattr(md, SPLIT_ATTR)
        assert bidir_rows.tolist() == [*range(1, 13), 20, 21, 22, 23]
        assert not bidir.causal and bidir.scheduler_metadata is None
        assert bidir.query_start_loc.tolist() == [0, 5, 12, 16]
        assert bidir.seq_lens.tolist() == [5, 12, 12], "keys: [0, S) for the state, [0, L) after"
        assert bidir.block_table.flatten().tolist() == [base, base, 4 * base]
        assert (bidir.num_actual_tokens, bidir.max_query_len, bidir.max_seq_len) == (16, 7, 12)
        assert causal_rows.tolist() == [0, *range(13, 20)] and causal.causal
        assert causal.query_start_loc.tolist() == [0, 1, 8]
        assert causal.seq_lens.tolist() == [40, 7] and causal.block_table.flatten().tolist() == [
            0,
            2 * base,
        ]
    assert (
        getattr(full, SPLIT_ATTR)[0][1].query_start_loc
        is getattr(sliding, SPLIT_ATTR)[0][1].query_start_loc
    ), "the index tensors are built once per device"


@pytest.fixture
def state(monkeypatch):
    default = model_state.DefaultModelState
    monkeypatch.setattr(default, "add_request", lambda self, i, d: None)
    monkeypatch.setattr(default, "remove_request", lambda self, rid: None)
    s = object.__new__(model_state.PrefixBidirModelState)
    s._bounds, s._warned_partial = {}, False
    s.metadata = {}
    monkeypatch.setattr(default, "prepare_attn", lambda self, *a, **k: s.metadata)
    return s


def add(state, rid, prompt_len, extra_args):
    data = SimpleNamespace(
        req_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )
    state.add_request(0, data)


def split(state, req_ids, qsl, seq, for_capture=False):
    md = metadata(qsl, seq)
    state.metadata.clear()
    state.metadata["layer"] = md
    batch = SimpleNamespace(
        num_reqs=len(req_ids),
        req_ids=list(req_ids),
        query_start_loc_np=np.array(qsl),
        seq_lens_cpu_upper_bound=torch.tensor(seq),
    )
    state.prepare_attn(batch, None, (), None, [], None, for_capture=for_capture)
    return getattr(md, SPLIT_ATTR, None)


def test_only_a_prefix_bidir_prompt_step_is_split(state):
    add(state, "a", 12, {"prefix_end": 5})
    add(state, "warm-up", 12, None)
    assert state._bounds == {"a": (5, 12)}
    assert split(state, ["a"], [0, 12], [12]) is not None
    assert split(state, ["a"], [0, 12], [12], for_capture=True) is None
    assert split(state, ["a"], [0, 1], [12]) is None, "an exact repeat computes one token"
    assert split(state, ["warm-up"], [0, 12], [12]) is None
    state.remove_request("a")
    assert state._bounds == {}


def test_a_prompt_scheduled_in_parts_is_logged_once(state, caplog):
    add(state, "a", 12, {"prefix_end": 5})
    with caplog.at_level(logging.ERROR, logger="vllm.strands_decider"):
        assert split(state, ["a"], [0, 8], [8]) is not None
        split(state, ["a"], [0, 8], [8])
    assert caplog.text.count("scheduled in parts") == 1
