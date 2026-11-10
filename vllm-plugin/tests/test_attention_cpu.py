"""SplitCPUAttentionImpl: a split step's segments go to vLLM's own CPU attention call (stubbed here), each with a
scheduler plan built for its own rows, once per step. Needs vLLM; vLLM's CUDA build imports the CPU backend too."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
attention_cpu = pytest.importorskip("strands_decider_vllm.attention_cpu")
from strands_decider_vllm.attention import SPLIT_ATTR, Split  # noqa: E402


@dataclass
class Meta:
    tag: float
    causal: bool = True
    seq_lens: torch.Tensor = field(default_factory=lambda: torch.tensor([5]))
    query_start_loc: torch.Tensor = field(default_factory=lambda: torch.tensor([0, 5]))
    scheduler_metadata: object = None


@pytest.fixture
def calls(monkeypatch):
    recorded, plans = [], []

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        recorded.append(dict(query=query.clone(), md=attn_metadata))
        output.copy_(query + (attn_metadata.tag if attn_metadata is not None else 0.0))
        return output

    def plan(**kw):
        plans.append(kw)
        return f"plan-{len(plans)}"

    monkeypatch.setattr(attention_cpu.CPUAttentionBackendImpl, "forward", forward)
    monkeypatch.setattr(attention_cpu.ops, "cpu_attn_get_scheduler_metadata", plan)
    return recorded, plans


def impl(sliding_window=-1):
    i = object.__new__(attention_cpu.SplitCPUAttentionImpl)
    i.sliding_window, i.num_heads, i.num_kv_heads, i.head_size, i.kv_cache_dtype = (
        sliding_window,
        4,
        1,
        8,
        "auto",
    )
    return i


def run(i, query, md):
    output = torch.full_like(query, float("nan"))
    layer = SimpleNamespace(isa="vec")
    return i.forward(layer, query, None, None, None, md, output), output


def test_metadata_without_split_takes_vllms_path(calls):
    recorded, plans = calls
    md = Meta(1.0, scheduler_metadata="vllm's own plan")
    query = torch.randn(5, 4, 8)
    _, output = run(impl(), query, md)
    assert [c["md"] for c in recorded] == [md] and plans == [] and torch.equal(output, query + 1.0)


def test_a_split_runs_two_calls_each_with_its_own_plan(calls):
    recorded, plans = calls
    query = torch.randn(8, 4, 8)
    bidir = Meta(
        10.0, causal=False, seq_lens=torch.tensor([3, 8]), query_start_loc=torch.tensor([0, 3, 6])
    )
    rest = Meta(100.0, seq_lens=torch.tensor([40]), query_start_loc=torch.tensor([0, 2]))
    md = Meta(0.0)
    setattr(
        md, SPLIT_ATTR, Split(bidir_rows=slice(0, 6), rest_rows=slice(6, 8), bidir=bidir, rest=rest)
    )
    _, output = run(impl(), query, md)
    assert [c["md"] for c in recorded] == [bidir, rest]
    assert torch.equal(output[:6], query[:6] + 10.0) and torch.equal(output[6:], query[6:] + 100.0)
    assert [p["causal"] for p in plans] == [False, True]
    assert [p["num_reqs"] for p in plans] == [2, 1] and all(
        p["sliding_window_size"] == -1 for p in plans
    )
    assert (
        plans[0]["seq_lens"] is bidir.seq_lens
        and plans[0]["query_start_loc"] is bidir.query_start_loc
    )
    # The next layer of the same KV-cache group reuses the plans attached to the restricted metadata.
    run(impl(), query, md)
    assert (
        len(plans) == 2
        and bidir.scheduler_metadata == "plan-1"
        and rest.scheduler_metadata == "plan-2"
    )


def test_a_windowed_layer_refuses_a_split(calls):
    md = Meta(0.0)
    setattr(
        md,
        SPLIT_ATTR,
        Split(bidir_rows=slice(0, 2), rest_rows=None, bidir=Meta(1.0, causal=False), rest=None),
    )
    with pytest.raises(RuntimeError, match="windowed layer"):
        run(impl(sliding_window=511), torch.randn(2, 4, 8), md)


def test_the_cpu_backend_writes_the_kv_cache_outside_forward():
    import inspect

    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend, CPUAttentionBackendImpl

    assert CPUAttentionBackend.forward_includes_kv_cache_update is False
    assert issubclass(attention_cpu.SplitCPUAttentionBackend, CPUAttentionBackend)
    assert (
        attention_cpu.SplitCPUAttentionBackend.get_impl_cls() is attention_cpu.SplitCPUAttentionImpl
    )
    # forward writes keys and values for encoder attention only; decoder layers' cache is written in its own op.
    src = inspect.getsource(CPUAttentionBackendImpl.forward)
    assert src.count("cpu_attn_reshape_and_cache") == 1 and "if is_encoder_attention:" in src


def test_the_model_state_accepts_both_implementations():
    model_state = pytest.importorskip("strands_decider_vllm.model_state")
    from strands_decider_vllm.attention import SplitFlashAttentionImpl

    assert model_state._split_impls() == (
        SplitFlashAttentionImpl,
        attention_cpu.SplitCPUAttentionImpl,
    )
    assert (
        SplitFlashAttentionImpl.no_window == (-1, -1)
        and attention_cpu.SplitCPUAttentionImpl.no_window == -1
    )
