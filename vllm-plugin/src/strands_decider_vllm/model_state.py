"""Model Runner V2 state for prefix_bidir: each step's split, attached to its attention metadata.

The engine core has checked every request's boundary S (``vllm_xargs.prefix_end``; ``engine``). In
a step that computes prefix_bidir tokens, ``planning.plan_segments`` splits them, and every
KV-cache group's metadata gets restricted copies for one non-causal and one causal call, which the
attention backends run (``attention``). Requests without S (vLLM's warm-up) stay causal.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from . import PREFIX_END_KEY, logger
from .attention import SPLIT_ATTR, CPUImpl, FlashImpl
from .planning import Segment, plan_segments

#: Fields of vLLM's metadata a restricted copy drops: FA3's ahead-of-time schedule (CPU attention
#: plans its own), and masks for features the plugin refuses.
_DROPPED = (
    "scheduler_metadata",
    "mm_prefix_query_range_tensor",
    "rswa_prefix_lens",
    "dynamic_causal",
)


class PrefixBidirModelState(DefaultModelState):
    def __init__(
        self, vllm_config: Any, model: torch.nn.Module, encoder_cache: Any, device: torch.device
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._bounds: dict[str, tuple[int, int]] = {}  # request id -> (S, L)
        self._warned_partial = False
        layers = vllm_config.compilation_config.static_forward_context.items()
        other = sorted(
            name
            for name, layer in layers
            if getattr(layer, "attn_type", None) == "decoder"
            and hasattr(layer, "get_attn_backend")
            and not isinstance(getattr(layer, "impl", None), (FlashImpl, CPUImpl))
        )
        if other:  # vLLM picks Triton attention on Ampere and Ada, and --attention-backend can too
            raise ValueError(
                f"strands-decider: attention layers {other[:3]}... are not on FLASH_ATTN or "
                "CPU_ATTN, which run the prefix_bidir mask"
            )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        sp = new_req_data.sampling_params
        s = (sp.extra_args or {}).get(PREFIX_END_KEY) if sp is not None else None
        if s is not None:
            self._bounds[new_req_data.req_id] = (s, len(new_req_data.prompt_token_ids))

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._bounds.pop(req_id, None)

    def prepare_attn(
        self,
        input_batch: Any,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[Any]],
        kv_cache_config: Any,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: Any = None,
    ) -> dict[str, Any]:
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        if for_capture or not self._bounds:
            return metadata
        n = input_batch.num_reqs
        bounds = {
            i: self._bounds[r] for i, r in enumerate(input_batch.req_ids[:n]) if r in self._bounds
        }
        if not bounds:
            return metadata
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:n].numpy()
        if not self._warned_partial and any(seq_lens[i] < L for i, (_, L) in bounds.items()):
            # The startup checks and the scheduler wrapper exist so that this never happens.
            self._warned_partial = True
            logger.error(
                "strands-decider: a prefix_bidir prompt was scheduled in parts, so its answer is "
                "wrong; please report this configuration (logged once)"
            )
        plan = plan_segments(
            {i: s for i, (s, _) in bounds.items()}, input_batch.query_start_loc_np, seq_lens, n
        )
        # Only uniform one-token batches run as FULL CUDA graphs, and a one-token row is never
        # split (planning), so a graph never replays attention that should have been split.
        if plan is not None:
            _attach(metadata, *plan)
        return metadata


def _attach(metadata: dict[str, Any], bidir: list[Segment], causal: list[Segment]) -> None:
    """Attaches the split to every distinct metadata object of the step (one per KV-cache group)."""
    parts = [(bidir, False)] + ([(causal, True)] if causal else [])
    on_device: dict[torch.device, list[tuple[Any, dict[str, torch.Tensor]]]] = {}
    for md in {id(m): m for m in metadata.values() if hasattr(m, "block_table")}.values():
        device = md.query_start_loc.device
        if device not in on_device:
            on_device[device] = [_on(segments, device) for segments, _ in parts]
        split = []
        for (segments, is_causal), (rows, t) in zip(parts, on_device[device], strict=True):
            lengths = [b - a for _, a, b, _ in segments]
            fields = dict(
                num_actual_tokens=sum(lengths),
                max_query_len=max(lengths),
                query_start_loc=t["query_start_loc"],
                block_table=md.block_table.index_select(0, t["batch_rows"]),
                causal=is_causal,
                **{k: None for k in _DROPPED if hasattr(md, k)},
            )
            if is_causal:  # each causal segment ends its row: the row's own length is right
                fields.update(seq_lens=md.seq_lens.index_select(0, t["batch_rows"]))
            else:
                fields.update(seq_lens=t["keys"], max_seq_len=max(k for *_, k in segments))
            split.append((rows, dataclasses.replace(md, **fields)))
        setattr(md, SPLIT_ATTR, split)


def _on(segments: list[Segment], device: torch.device) -> tuple[Any, dict[str, torch.Tensor]]:
    """The segments' token rows (a slice when contiguous), and their per-call index tensors, on
    ``device`` in one host-to-device copy."""
    tokens = np.concatenate([np.arange(a, b) for _, a, b, _ in segments])
    host = {
        "tokens": tokens,
        "query_start_loc": np.cumsum([0] + [b - a for _, a, b, _ in segments]),
        "batch_rows": np.array([r for r, *_ in segments]),
        "keys": np.array([k for *_, k in segments]),
    }
    flat = torch.from_numpy(np.concatenate(list(host.values())).astype(np.int64))
    parts = flat.to(device, non_blocking=True).split([len(v) for v in host.values()])
    t = dict(zip(host, parts, strict=True))
    t["query_start_loc"], t["keys"] = t["query_start_loc"].int(), t["keys"].int()  # FA's int32
    contiguous = tokens[-1] - tokens[0] + 1 == len(tokens)
    return (slice(int(tokens[0]), int(tokens[-1]) + 1) if contiguous else t["tokens"]), t
