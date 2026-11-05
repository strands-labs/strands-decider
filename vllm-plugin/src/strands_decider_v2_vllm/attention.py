"""FlashAttention for prefix_bidir: a layer's step as one non-causal call over each request's two segments.

In a step that computes prefix_bidir tokens, :class:`~strands_decider_v2_vllm.model_state.PrefixBidirModelState`
attaches a :class:`Split` to the step's attention metadata. Every attention layer then runs:

- the scheduled tokens of each prefix_bidir request as up to two non-causal FlashAttention sequences on the paged KV
  cache: its state queries over keys ``[0, S)`` and its question queries over keys ``[0, L)``;
- every other token (requests without a boundary, decode rows), causal over its own request's keys, as vLLM would.

That is the v2 prefix_bidir mask, with no mask function: state tokens attend bidirectionally to the state [0, S) and
never to the question; question tokens attend bidirectionally to the whole prompt [0, L). Metadata without a split
takes vLLM's path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

#: Attribute under which the model state attaches a :class:`Split` to a FlashAttentionMetadata.
SPLIT_ATTR = "strands_decider_v2_split"


@dataclass
class Split:
    """The calls of a split step. ``bidir`` is the stock metadata restricted to the non-causal segments; ``rest`` the
    other segments (causal), or None. ``*_rows`` are the segments' token rows in the step."""

    bidir_rows: torch.Tensor | slice
    rest_rows: torch.Tensor | slice | None
    bidir: Any
    rest: Any | None


def _take(x: torch.Tensor, rows: torch.Tensor | slice) -> torch.Tensor:
    return x[rows] if isinstance(rows, slice) else x.index_select(0, rows)


def _put(dst: torch.Tensor, rows: torch.Tensor | slice, src: torch.Tensor) -> None:
    if isinstance(rows, slice):
        dst[rows] = src
    else:
        dst.index_copy_(0, rows, src)


class SplitFlashAttentionImpl(FlashAttentionImpl):
    """:class:`FlashAttentionImpl` that runs a :class:`Split` as one non-causal and one causal stock call."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        split = getattr(attn_metadata, SPLIT_ATTR, None) if attn_metadata is not None else None
        if split is None:
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        if self.sliding_window not in (None, (-1, -1)):
            raise RuntimeError(
                "strands-decider-v2: a windowed layer received prefix_bidir metadata; the model class builds "
                "every layer without one"
            )
        q = _take(query, split.bidir_rows)
        out = output.new_empty(q.shape)
        super().forward(
            layer, q, key, value, kv_cache, split.bidir, out, output_scale, output_block_scale
        )
        _put(output, split.bidir_rows, out)
        if split.rest is not None:
            q = _take(query, split.rest_rows)
            out = output.new_empty(q.shape)
            super().forward(
                layer, q, key, value, kv_cache, split.rest, out, output_scale, output_block_scale
            )
            _put(output, split.rest_rows, out)
        return output


class SplitFlashAttentionBackend(FlashAttentionBackend):
    """vLLM's FlashAttention backend with :class:`SplitFlashAttentionImpl`; registered in place of FLASH_ATTN."""

    @staticmethod
    def get_impl_cls() -> type[SplitFlashAttentionImpl]:
        return SplitFlashAttentionImpl
