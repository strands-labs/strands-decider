"""CPU attention for prefix_bidir: the split of :mod:`strands_decider_vllm.attention` on vLLM's CPU_ATTN backend.

vLLM's CPU attention runs paged KV for a batch of sequences, each with its own query rows, key count and block table,
causal or not, and like FlashAttention it writes a step's keys and values in a separate op before attention
(``do_kv_cache_update``). The model state's :class:`~strands_decider_vllm.attention.Split` therefore runs here as one
non-causal and one causal call of vLLM's own ``forward``. The one difference from FlashAttention: the CPU kernel takes a
scheduler plan (``ops.cpu_attn_get_scheduler_metadata``) built from the batch's query and key lengths, so each restricted
metadata gets its own plan, built once per step and KV-cache group (the restricted metadata objects are). Metadata
without a split takes vLLM's path unchanged. A CPU build of vLLM is what lets the CPU tier of the tests run the plugin
end to end on machines without a GPU.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm import envs
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend, CPUAttentionBackendImpl

from .attention import SPLIT_ATTR, _put, _take

try:  # vLLM 0.34 moved the CPU ops wrappers to vllm._cpu_ops
    from vllm import _cpu_ops as ops
except ImportError:
    from vllm import _custom_ops as ops

#: The CPU impl's sliding_window for "no window".
NO_WINDOW = -1


class SplitCPUAttentionImpl(CPUAttentionBackendImpl):
    """:class:`CPUAttentionBackendImpl` that runs a split as one non-causal and one causal stock call."""

    no_window = NO_WINDOW

    def _planned(self, layer: Any, md: Any, dtype: torch.dtype) -> Any:
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
                sliding_window_size=NO_WINDOW,
                isa=layer.isa,
                enable_kv_split=envs.VLLM_CPU_ATTN_SPLIT_KV,
                dynamic_causal=None,
                kv_cache_dtype=self.kv_cache_dtype,
            )
        return md

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
        if self.sliding_window != NO_WINDOW:
            raise RuntimeError(
                "strands-decider: a windowed layer received prefix_bidir metadata; the model class builds "
                "every layer without one"
            )
        for rows, md in ((split.bidir_rows, split.bidir), (split.rest_rows, split.rest)):
            if md is None:
                continue
            q = _take(query, rows)
            out = output.new_empty(q.shape)
            super().forward(
                layer,
                q,
                key,
                value,
                kv_cache,
                self._planned(layer, md, query.dtype),
                out,
                output_scale,
                output_block_scale,
            )
            _put(output, rows, out)
        return output


class SplitCPUAttentionBackend(CPUAttentionBackend):
    """vLLM's CPU attention backend with :class:`SplitCPUAttentionImpl`; registered in place of CPU_ATTN."""

    @staticmethod
    def get_impl_cls() -> type[SplitCPUAttentionImpl]:
        return SplitCPUAttentionImpl
