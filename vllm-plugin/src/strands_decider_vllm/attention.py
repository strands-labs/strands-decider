"""The prefix_bidir mask on vLLM's FlashAttention, under Model Runner V2.

In each step that computes prefix_bidir tokens, the model state splits the step's token rows
(``planning``) and attaches to every KV-cache group's metadata restricted copies for one non-causal
and one causal call, which the backend runs as vLLM's own call on their rows. Two calls per layer
on one paged KV cache are correct because FlashAttention writes a step's keys and values in its own
op before attention (``do_kv_cache_update``), and ``forward`` only reads them. Requests without S
(vLLM's warm-up) and metadata without a split take vLLM's path unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from . import PREFIX_END_KEY
from .planning import Segment, plan_segments

#: The attribute under which a metadata object carries its step's split.
_SPLIT = "strands_decider_split"
#: What a restricted copy clears: FA3's ahead-of-time schedule, made for the whole batch, and the
#: masks of features the plugin refuses.
_CLEARED = {
    "scheduler_metadata": None,
    "mm_prefix_query_range_tensor": None,
    "rswa_prefix_lens": None,
}


class PrefixBidirModelState(DefaultModelState):
    def __init__(self, vllm_config: Any, model: Any, encoder_cache: Any, device: Any) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._boundaries: dict[str, int] = {}  # request id -> S
        layers = vllm_config.compilation_config.static_forward_context.items()
        other = sorted(
            name
            for name, layer in layers
            if isinstance(layer, Attention)
            and layer.attn_type == AttentionType.DECODER
            and not isinstance(layer.impl, _FlashImpl)
        )
        if other:  # vLLM picks Triton attention on Ampere and Ada, and --attention-backend can too
            raise ValueError(
                f"strands-decider: attention layers {other[:3]}... are not on FLASH_ATTN, which "
                "runs the prefix_bidir mask"
            )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        s = (new_req_data.sampling_params.extra_args or {}).get(PREFIX_END_KEY)
        if s is not None:
            self._boundaries[new_req_data.req_id] = s

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._boundaries.pop(req_id, None)

    def prepare_attn(
        self, input_batch: Any, *args: Any, for_capture: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        metadata = super().prepare_attn(input_batch, *args, for_capture=for_capture, **kwargs)
        if for_capture:
            return metadata
        n = input_batch.num_reqs
        bounds = {
            i: self._boundaries[r]
            for i, r in enumerate(input_batch.req_ids[:n])
            if r in self._boundaries
        }
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:n].numpy()
        plan = plan_segments(bounds, input_batch.query_start_loc_np, seq_lens, n)
        # FULL CUDA graphs replay only uniform one-token batches, and a one-token row is never
        # split, so no graph replays attention that should have been split.
        if plan is not None:
            _attach(metadata, *plan)
        return metadata


def _attach(metadata: dict[str, Any], bidir: list[Segment], causal: list[Segment]) -> None:
    """Gives every distinct metadata object of the step (one per KV-cache group) its split."""
    groups = {id(m): m for m in metadata.values() if isinstance(m, FlashAttentionMetadata)}
    device = next(iter(groups.values())).query_start_loc.device
    calls = [_Call(bidir, device, causal=False)]
    if causal:
        calls.append(_Call(causal, device, causal=True))
    for md in groups.values():
        setattr(md, _SPLIT, [(call.rows, call.restrict(md)) for call in calls])


class _Call:
    """One FlashAttention call of a split step: its segments' token rows (a slice when contiguous),
    and the index tensors that restrict a metadata object to them, copied to the device once."""

    def __init__(self, segments: list[Segment], device: torch.device, *, causal: bool) -> None:
        lengths = [b - a for _, a, b, _ in segments]
        tokens = np.concatenate([np.arange(a, b) for _, a, b, _ in segments])
        host = [
            tokens,
            np.cumsum([0, *lengths]),
            np.array([row for row, *_ in segments]),
            np.array([k for *_, k in segments]),
        ]
        flat = torch.from_numpy(np.concatenate(host).astype(np.int64))
        token_rows, query_start_loc, self.batch_rows, keys = flat.to(
            device, non_blocking=True
        ).split([len(h) for h in host])
        # FlashAttention takes these two in int32.
        self.query_start_loc, self.keys = query_start_loc.int(), keys.int()
        contiguous = tokens[-1] - tokens[0] + 1 == len(tokens)
        self.rows = slice(int(tokens[0]), int(tokens[-1]) + 1) if contiguous else token_rows
        self.causal = causal
        self.num_tokens, self.max_query_len = sum(lengths), max(lengths)
        self.max_keys = max(k for *_, k in segments)

    def restrict(self, md: FlashAttentionMetadata) -> FlashAttentionMetadata:
        fields: dict[str, Any] = dict(
            num_actual_tokens=self.num_tokens,
            max_query_len=self.max_query_len,
            query_start_loc=self.query_start_loc,
            block_table=md.block_table.index_select(0, self.batch_rows),
            causal=self.causal,
            **_CLEARED,
        )
        if self.causal:  # each causal segment ends its row: the row's own length, on the device
            fields.update(seq_lens=md.seq_lens.index_select(0, self.batch_rows))
        else:
            fields.update(seq_lens=self.keys, max_seq_len=self.max_keys)
        return dataclasses.replace(md, **fields)


class _FlashImpl(FlashAttentionImpl):
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
        scales = (output_scale, output_block_scale)
        split = getattr(attn_metadata, _SPLIT, None)
        if split is None:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, *scales
            )
        for rows, md in split:
            q = query[rows]
            out = torch.empty_like(q)
            super().forward(layer, q, key, value, kv_cache, md, out, *scales)
            output[rows] = out
        return output


class FlashBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[FlashAttentionImpl]:
        return _FlashImpl
