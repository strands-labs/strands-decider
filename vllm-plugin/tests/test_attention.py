"""The split on vLLM's FlashAttention and CPU attention: each call gets its rows and metadata, with
vLLM's own call stubbed. Needs vLLM, no GPU."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
attention = pytest.importorskip("strands_decider_vllm.attention")


@dataclass
class Meta:
    tag: float
    causal: bool = True
    seq_lens: torch.Tensor = field(default_factory=lambda: torch.tensor([5]))
    query_start_loc: torch.Tensor = field(default_factory=lambda: torch.tensor([0, 5]))
    scheduler_metadata: object = None


@pytest.fixture
def calls(monkeypatch):
    """vLLM's forward of both backends, recording each call and writing query + tag."""
    recorded, plans = [], []

    def forward(self, layer, query, key, value, kv_cache, md, output, *scales):
        recorded.append((query.clone(), md))
        output.copy_(query + md.tag)
        return output

    for base in (attention.FlashAttentionImpl, attention.CPUAttentionBackendImpl):
        monkeypatch.setattr(base, "forward", forward)
    monkeypatch.setattr(
        attention.ops, "cpu_attn_get_scheduler_metadata", lambda **kw: plans.append(kw) or kw
    )
    return recorded, plans


def impl(cls):
    i = object.__new__(cls)  # no layer config needed to route the calls
    i.num_heads, i.num_kv_heads, i.head_size, i.kv_cache_dtype = 4, 1, 8, "auto"
    return i


def run(i, query, md):
    output = torch.full_like(query, float("nan"))
    return i.forward(SimpleNamespace(isa="vec"), query, None, None, None, md, output), output


@pytest.mark.parametrize("cls", [attention.FlashImpl, attention.CPUImpl])
def test_metadata_without_a_split_takes_vllms_call(calls, cls):
    recorded, plans = calls
    md, query = Meta(1.0), torch.randn(5, 4, 8)
    result, output = run(impl(cls), query, md)
    assert [m for _, m in recorded] == [md] and plans == [] and result is output
    assert torch.equal(output, query + 1.0)


@pytest.mark.parametrize("cls", [attention.FlashImpl, attention.CPUImpl])
def test_a_split_runs_one_call_per_part_on_its_rows(calls, cls):
    recorded, plans = calls
    query = torch.randn(8, 4, 8)
    bidir_rows, causal_rows = torch.tensor([1, 2, 3, 6, 7]), slice(4, 6)
    bidir, causal = Meta(10.0, causal=False), Meta(100.0)
    md = Meta(0.0)
    setattr(
        md,
        attention.SPLIT_ATTR,
        [(bidir_rows, bidir), (causal_rows, causal), (slice(0, 1), Meta(5.0))],
    )
    _, output = run(impl(cls), query, md)
    assert [m for _, m in recorded][:2] == [bidir, causal]
    assert torch.equal(recorded[0][0], query[bidir_rows]), "every head of a token stays in its row"
    assert torch.equal(output[bidir_rows], query[bidir_rows] + 10.0)
    assert torch.equal(output[4:6], query[4:6] + 100.0) and torch.equal(output[:1], query[:1] + 5.0)
    if cls is attention.CPUImpl:
        # One plan per restricted metadata, built from its own rows, which later layers reuse.
        assert [p["causal"] for p in plans] == [False, True, True]
        assert plans[0]["seq_lens"] is bidir.seq_lens and plans[0]["sliding_window_size"] == -1
        run(impl(cls), query, md)
        assert len(plans) == 3


@pytest.mark.parametrize(
    "backend, base",
    [
        (attention.FlashBackend, attention.FlashAttentionImpl),
        (attention.CPUBackend, attention.CPUAttentionBackendImpl),
    ],
)
def test_vllm_writes_the_kv_cache_before_attention(backend, base):
    # Two calls per layer on one paged KV cache are right only because forward only reads it.
    assert backend.forward_includes_kv_cache_update is False
    assert issubclass(backend.get_impl_cls(), base)
    if base is attention.FlashAttentionImpl:
        assert "reshape_and_cache" not in inspect.getsource(base.forward)
    else:  # it writes keys and values in forward for encoder attention only
        source = inspect.getsource(base.forward)
        assert (
            source.count("cpu_attn_reshape_and_cache") == 1 and "if is_encoder_attention:" in source
        )
