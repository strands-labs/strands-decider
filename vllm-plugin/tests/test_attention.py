"""SplitFlashAttentionImpl: a split step's segments go, unchanged, to vLLM's own FlashAttention call (stubbed here)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
attention = pytest.importorskip("strands_decider_vllm.attention")
SPLIT_ATTR, Split = attention.SPLIT_ATTR, attention.Split


@dataclass
class Meta:
    tag: float
    causal: bool = True


@pytest.fixture
def calls(monkeypatch):
    """Replace vLLM's FlashAttentionImpl.forward: record each call and write query + metadata tag to the output."""
    recorded = []

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
        recorded.append(
            dict(
                query=query.clone(),
                md=attn_metadata,
                key=key,
                value=value,
                kv_cache=kv_cache,
                output_scale=output_scale,
            )
        )
        output.copy_(query + (attn_metadata.tag if attn_metadata is not None else 0.0))
        return output

    monkeypatch.setattr(attention.FlashAttentionImpl, "forward", forward)
    return recorded


def make_impl(sliding_window=(-1, -1)):
    impl = object.__new__(
        attention.SplitFlashAttentionImpl
    )  # no vLLM layer config needed for routing
    impl.sliding_window = sliding_window
    return impl


def run(impl, query, md, **kw):
    key, value, kv_cache = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    output = torch.full_like(query, float("nan"))
    result = impl.forward(None, query, key, value, kv_cache, md, output, **kw)
    return result, output, (key, value, kv_cache)


@pytest.mark.parametrize("md", [Meta(1.0), None])
def test_metadata_without_split_takes_vllm_path(calls, md):
    query = torch.randn(6, 4, 8)
    result, output, _ = run(make_impl(), query, md)
    assert len(calls) == 1 and calls[0]["md"] is md and torch.equal(calls[0]["query"], query)
    assert result is output and torch.equal(output, query + (md.tag if md else 0.0))


def test_split_runs_one_non_causal_and_one_causal_call(calls):
    query = torch.randn(8, 4, 8)  # 8 tokens, 4 query heads per token
    bidir_rows, rest_rows = torch.tensor([1, 2, 3, 6, 7]), torch.tensor([0, 4, 5])
    md = Meta(0.0)
    setattr(
        md,
        SPLIT_ATTR,
        Split(
            bidir_rows=bidir_rows,
            rest_rows=rest_rows,
            bidir=Meta(10.0, causal=False),
            rest=Meta(100.0),
        ),
    )
    split = getattr(md, SPLIT_ATTR)
    scale = torch.ones(1)
    result, output, (key, value, kv_cache) = run(make_impl(), query, md, output_scale=scale)
    assert [c["md"] for c in calls] == [split.bidir, split.rest]
    assert torch.equal(calls[0]["query"], query[bidir_rows]), (
        "every query head of a token stays in its row"
    )
    assert torch.equal(calls[1]["query"], query[rest_rows])
    for c in calls:
        assert (
            c["key"] is key
            and c["value"] is value
            and c["kv_cache"] is kv_cache
            and c["output_scale"] is scale
        )
    assert result is output
    assert torch.equal(output[bidir_rows], query[bidir_rows] + 10.0)
    assert torch.equal(output[rest_rows], query[rest_rows] + 100.0)


def test_split_without_causal_rows(calls):
    query = torch.randn(5, 2, 4)
    md = Meta(0.0)
    setattr(
        md,
        SPLIT_ATTR,
        Split(bidir_rows=slice(0, 5), rest_rows=None, bidir=Meta(3.0, causal=False), rest=None),
    )
    _, output, _ = run(make_impl(), query, md)
    assert len(calls) == 1 and torch.equal(output, query + 3.0)


def test_windowed_layer_refuses_a_split(calls):
    md = Meta(0.0)
    setattr(
        md, SPLIT_ATTR, Split(bidir_rows=slice(0, 2), rest_rows=None, bidir=Meta(1.0), rest=None)
    )
    with pytest.raises(RuntimeError, match="windowed layer"):
        run(make_impl(sliding_window=(511, 0)), torch.randn(2, 1, 4), md)
    assert not calls


def test_flash_attention_writes_the_kv_cache_outside_forward():
    # The split calls FlashAttentionImpl.forward twice per layer on the same paged KV cache. That is only correct
    # because vLLM writes a step's keys and values in a separate op before attention (do_kv_cache_update), so forward
    # only reads the cache; Model Runner V2 reads this flag to decide that.
    import inspect

    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    assert FlashAttentionBackend.forward_includes_kv_cache_update is False
    assert attention.SplitFlashAttentionBackend.forward_includes_kv_cache_update is False
    assert issubclass(attention.SplitFlashAttentionBackend, FlashAttentionBackend)
    assert attention.SplitFlashAttentionBackend.get_impl_cls() is attention.SplitFlashAttentionImpl
    assert "reshape_and_cache" not in inspect.getsource(attention.FlashAttentionImpl.forward)
    assert "reshape_and_cache" in inspect.getsource(attention.FlashAttentionImpl.do_kv_cache_update)
