"""The prefix_bidir split on vLLM's FlashAttention: which calls each step's rows get, through the model
state's prepare_attn and the backend's forward, with vLLM's own parts stubbed."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from strands_decider_vllm.attention import FlashBackend, PrefixBidirModelState

#: A step: row 0 decodes r0; row 1 computes a cold prefix_bidir prompt (S 5, L 12); row 2 a causal
#: prefill of 7 tokens; row 3 a prefix_bidir prompt with 8 of its 12 tokens cached.
REQUESTS = ["r0", "cold", "r2", "cached"]
QUERY_START_LOC, SEQ_LENS = [0, 1, 13, 20, 24], [40, 12, 7, 12]


@pytest.fixture
def stock_calls(monkeypatch):
    """vLLM's FlashAttention forward, recording each call's metadata and writing query * 2 for a
    causal call and query * 3 for a non-causal one."""
    calls = []

    def forward(self, layer, query, key, value, kv_cache, md, output, *scales):
        calls.append(md)
        output.copy_(query * (2 if md.causal else 3))
        return output

    monkeypatch.setattr(FlashAttentionImpl, "forward", forward)
    return calls


@pytest.fixture
def model_state(monkeypatch):
    """The model state over vLLM's DefaultModelState, whose prepare_attn returns ``.metadata``."""
    for name in ("__init__", "add_request", "remove_request"):
        monkeypatch.setattr(DefaultModelState, name, lambda self, *a, **k: None)
    monkeypatch.setattr(DefaultModelState, "prepare_attn", lambda self, *a, **k: self.metadata)
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    state = PrefixBidirModelState(config, None, None, torch.device("cpu"))
    for rid, s in [("r0", None), ("cold", 5), ("r2", None), ("cached", 5)]:
        state.add_request(0, request(rid, s))
    return state


def request(rid, s):
    extra_args = None if s is None else {"prefix_end": s}
    return SimpleNamespace(req_id=rid, sampling_params=SimpleNamespace(extra_args=extra_args))


def metadata(block_base):
    return FlashAttentionMetadata(
        num_actual_tokens=QUERY_START_LOC[-1],
        max_query_len=12,
        query_start_loc=torch.tensor(QUERY_START_LOC, dtype=torch.int32),
        max_seq_len=max(SEQ_LENS),
        seq_lens=torch.tensor(SEQ_LENS, dtype=torch.int32),
        block_table=torch.arange(len(SEQ_LENS)).view(-1, 1) * block_base,
        slot_mapping=torch.zeros(QUERY_START_LOC[-1], dtype=torch.int64),
        scheduler_metadata=torch.zeros(1),  # FA3's schedule for the whole batch
    )


def step(state, req_ids, query_start_loc, seq_lens, for_capture=False):
    batch = SimpleNamespace(
        num_reqs=len(req_ids),
        req_ids=req_ids,
        query_start_loc_np=np.array(query_start_loc),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens),
    )
    return state.prepare_attn(batch, None, (), None, [], None, for_capture=for_capture)


def attend(md):
    query = torch.randn(md.num_actual_tokens, 2, 8)
    output = torch.full_like(query, float("nan"))
    impl = object.__new__(FlashBackend.get_impl_cls())  # no layer config needed to route the calls
    impl.forward(None, query, None, None, None, md, output)
    return query, output


def test_a_step_with_prefix_bidir_rows_gets_one_non_causal_and_one_causal_call_per_group(
    model_state, stock_calls
):
    full, sliding = metadata(10), metadata(100)
    model_state.metadata = {"layer.0": full, "layer.1": sliding, "layer.2": full}
    step(model_state, REQUESTS, QUERY_START_LOC, SEQ_LENS)
    for md, base in ((full, 10), (sliding, 100)):
        stock_calls.clear()
        query, output = attend(md)
        bidir, causal = stock_calls
        assert not bidir.causal and causal.causal
        assert bidir.query_start_loc.tolist() == [0, 5, 12, 16]
        assert bidir.seq_lens.tolist() == [5, 12, 12], "keys: [0, S) for the state, [0, L) after"
        assert bidir.block_table.flatten().tolist() == [base, base, 3 * base]
        assert (bidir.num_actual_tokens, bidir.max_query_len, bidir.max_seq_len) == (16, 7, 12)
        assert bidir.scheduler_metadata is None
        assert causal.query_start_loc.tolist() == [0, 1, 8]
        assert causal.seq_lens.tolist() == [40, 7] and causal.block_table.flatten().tolist() == [
            0,
            2 * base,
        ]
        bidir_rows = [*range(1, 13), *range(20, 24)]
        assert torch.equal(output[bidir_rows], query[bidir_rows] * 3)
        causal_rows = [0, *range(13, 20)]
        assert torch.equal(output[causal_rows], query[causal_rows] * 2), "every head in its row"


def test_a_step_without_prefix_bidir_rows_takes_vllms_call(model_state, stock_calls):
    for req_ids, query_start_loc, seq_lens, for_capture in [
        (["r0", "r2"], [0, 1, 8], [40, 7], False),  # no request with S, as in vLLM's warm-up
        (["cold"], [0, 1], [12], False),  # an exact repeat computes only the last token
        (["cold"], [0, 12], [12], True),  # capturing a CUDA graph
    ]:
        md = metadata(10)
        model_state.metadata = {"layer.0": md}
        step(model_state, req_ids, query_start_loc, seq_lens, for_capture)
        stock_calls.clear()
        attend(md)
        assert stock_calls == [md]
    model_state.remove_request("cold")
    model_state.metadata = {"layer.0": (md := metadata(10))}
    step(model_state, ["cold"], [0, 12], [12])
    stock_calls.clear()
    attend(md)
    assert stock_calls == [md], "a finished request's S is forgotten"


def test_attention_layers_off_flash_attention_are_refused(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "__init__", lambda self, *a, **k: None)
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.attn_type, layer.impl = "decoder", object()  # Triton's, say, as vLLM picks on Ampere
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"l": layer})
    )
    with pytest.raises(ValueError, match="not on FLASH_ATTN"):
        PrefixBidirModelState(config, None, None, torch.device("cpu"))


def test_vllm_writes_the_kv_cache_before_attention():
    # Two calls per layer on one paged KV cache are right only because forward only reads it.
    assert FlashBackend.forward_includes_kv_cache_update is False
    assert "reshape_and_cache" not in inspect.getsource(FlashAttentionImpl.forward)
