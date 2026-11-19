"""vLLM's FlashAttention and CPU attention, which run a step's prefix_bidir split when its metadata
carries one (``model_state``): one stock call per list of (token rows, restricted metadata).

Two calls per layer on the same paged KV cache are correct because both backends write a step's
keys and values in their own op before attention (``do_kv_cache_update``); ``forward`` only reads.
Metadata without a split takes vLLM's path unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm import envs
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend, CPUAttentionBackendImpl
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

try:  # vLLM 0.34 moved the CPU ops to vllm._cpu_ops
    from vllm import _cpu_ops as ops
except ImportError:
    from vllm import _custom_ops as ops

#: The attribute under which the model state attaches a step's split to a metadata object.
SPLIT_ATTR = "strands_decider_split"
Split = list[tuple[torch.Tensor | slice, Any]]


class _SplitImpl:
    no_window: Any  # the implementation's sliding_window for "no window"

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
        split: Split | None = getattr(attn_metadata, SPLIT_ATTR, None)
        stock = super().forward
        if split is None:
            return stock(
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
        for rows, md in split:
            q = query[rows]
            out = output.new_empty(q.shape)
            md = self._prepared(layer, md, query.dtype)
            stock(layer, q, key, value, kv_cache, md, out, output_scale, output_block_scale)
            output[rows] = out
        return output

    def _prepared(self, layer: Any, md: Any, dtype: torch.dtype) -> Any:
        return md


class FlashImpl(_SplitImpl, FlashAttentionImpl):
    no_window = (-1, -1)


class CPUImpl(_SplitImpl, CPUAttentionBackendImpl):
    no_window = -1

    def _prepared(self, layer: Any, md: Any, dtype: torch.dtype) -> Any:
        # The CPU kernel takes a plan built from the batch's lengths: one per restricted metadata,
        # which every layer of its KV-cache group shares.
        if md.scheduler_metadata is None:
            md.scheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
                num_reqs=int(md.seq_lens.shape[0]),
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                seq_lens=md.seq_lens,
                dtype=dtype,
                query_start_loc=md.query_start_loc,
                causal=md.causal,
                sliding_window_size=self.no_window,
                isa=layer.isa,
                enable_kv_split=envs.VLLM_CPU_ATTN_SPLIT_KV,
                dynamic_causal=None,
                kv_cache_dtype=self.kv_cache_dtype,
            )
        return md


class FlashBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[FlashImpl]:
        return FlashImpl


class CPUBackend(CPUAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[CPUImpl]:
        return CPUImpl
