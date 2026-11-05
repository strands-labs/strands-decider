"""Model Runner V2 state for prefix_bidir: each request's state boundary, turned into split attention metadata.

A request sends its boundary S (the index of the first token of the question block in the v2 prompt format) in
``vllm_xargs.prefix_end``. In a step that computes any of its prompt tokens, its scheduled tokens ``[c, L)`` (c being
its prefix-cache hit, L its prompt length; chunked prefill is off, so a prompt is always scheduled to its end) become
up to two non-causal segments:

- state queries ``[c, S)`` over keys ``[0, S)`` (absent when the cache hit reaches S);
- question queries ``[max(c, S), L)`` over keys ``[0, L)``.

Every attention layer of the model gets this split (the model class builds every layer without a sliding window), through
:class:`~strands_decider_v2_vllm.attention.SplitFlashAttentionImpl`. Requests without a boundary and decode rows stay
causal. Cached keys are valid because the block hashes of a prefix_bidir request carry digests of its state and
question (:mod:`strands_decider_v2_vllm.hashing`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from . import PREFIX_END_KEY
from .attention import SPLIT_ATTR, Split, SplitFlashAttentionImpl

logger = logging.getLogger("strands_decider_v2_vllm")


def prefix_end_of(extra_args: dict | None, prompt_len: int) -> int | None:
    """The request's boundary from ``vllm_xargs``, validated; None if it sent none."""
    if not extra_args or PREFIX_END_KEY not in extra_args:
        return None
    s = extra_args[PREFIX_END_KEY]
    if isinstance(s, float) and s.is_integer():
        s = int(s)
    if not isinstance(s, int) or isinstance(s, bool) or not 0 < s < prompt_len:
        raise ValueError(
            f"strands-decider-v2: prefix_end must be an integer in (0, {prompt_len}), got {s!r}"
        )
    return s


def plan_segments(
    bounds: dict[int, tuple[int, int]],
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    num_reqs: int,
) -> tuple[list, list] | None:
    """Segments of a step. ``bounds`` maps batch rows that are prefilling prefix_bidir requests to (S, L).

    Returns ``(bidir, rest)``, each a list of ``(row, q_start, q_end, n_keys)`` in token rows of the step; n_keys -1
    means the row's own sequence length (causal). None when no row computes prefix_bidir tokens.
    """
    bidir: list[tuple[int, int, int, int]] = []
    rest: list[tuple[int, int, int, int]] = []
    for row in range(num_reqs):
        q0, q1 = int(query_start_loc[row]), int(query_start_loc[row + 1])
        if q1 == q0:
            continue
        if row not in bounds:
            rest.append((row, q0, q1, -1))
            continue
        s, length = bounds[row]
        end = int(seq_lens[row])
        start = end - (q1 - q0)
        if end != length:
            raise RuntimeError(
                f"strands-decider-v2: a prefix_bidir prompt of {length} tokens was scheduled to {end}; "
                "it must be computed to its end in one step (chunked prefill off)"
            )
        if start < s:
            bidir.append((row, q0, q0 + (s - start), s))
        qs = max(start, s)
        if q1 - (q0 + (qs - start)) == 1:
            # Only the prompt's last token (a full cache hit, as on an exact repeat): it already attends to every key
            # [0, L) under the causal mask, so it needs no split. Such steps can run as FULL CUDA graphs.
            rest.append((row, q1 - 1, q1, -1))
        else:
            bidir.append((row, q0 + (qs - start), q1, length))
    if not bidir:
        return None
    return bidir, rest


def _rows(segments: list, device: torch.device) -> tuple[torch.Tensor | slice, np.ndarray]:
    idx = (
        np.concatenate([np.arange(a, b) for _, a, b, _ in segments])
        if segments
        else np.zeros(0, np.int64)
    )
    if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
        return slice(int(idx[0]), int(idx[-1]) + 1), idx
    return torch.from_numpy(idx.astype(np.int64)).to(device, non_blocking=True), idx


class PrefixBidirModelState(DefaultModelState):
    """:class:`DefaultModelState` plus prefix_bidir on every attention layer."""

    def __init__(
        self, vllm_config: Any, model: torch.nn.Module, encoder_cache: Any, device: torch.device
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._bounds: dict[str, tuple[int, int]] = {}
        self._warned_causal = False
        layers = {
            n: layer
            for n, layer in vllm_config.compilation_config.static_forward_context.items()
            if getattr(layer, "attn_type", None) == "decoder" and hasattr(layer, "get_attn_backend")
        }
        bad = sorted(
            n
            for n, layer in layers.items()
            if not isinstance(getattr(layer, "impl", None), SplitFlashAttentionImpl)
        )
        if bad:
            raise ValueError(
                f"strands-decider-v2: attention layers not on the plugin's FlashAttention: {bad[:4]}... "
                "(use the default or FLASH_ATTN backend)"
            )
        windowed = sorted(
            n
            for n, layer in layers.items()
            if getattr(layer, "sliding_window", None) is not None
            and (
                layer.sliding_window < self.max_model_len
                or tuple(getattr(layer.impl, "sliding_window", ())) != (-1, -1)
            )
        )
        if windowed:
            raise ValueError(
                f"strands-decider-v2: {len(windowed)} layers keep a sliding window ({windowed[0]}, ...); "
                "prefix_bidir needs every window off (StrandsDeciderV2ForCausalLM builds its layers without one)"
            )
        logger.info(
            "strands-decider-v2: prefix_bidir on %d attention layers (no sliding windows)",
            len(layers),
        )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        rid = new_req_data.req_id
        self._bounds.pop(rid, None)
        sp = new_req_data.sampling_params
        length = len(new_req_data.prompt_token_ids or ())
        s = prefix_end_of(sp.extra_args if sp is not None else None, length)
        if s is None:
            if not self._warned_causal:
                self._warned_causal = True
                logger.warning(
                    "strands-decider-v2: a request without vllm_xargs.prefix_end is served causally; v2 "
                    "decisions need the boundary"
                )
            return
        self._bounds[rid] = (s, length)

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
        prefilling = input_batch.is_prefilling_np
        bounds = {
            i: self._bounds[rid]
            for i, rid in enumerate(input_batch.req_ids[:n])
            if prefilling[i] and rid in self._bounds
        }
        if not bounds:
            return metadata
        qsl = input_batch.query_start_loc_np[: n + 1]
        seq = input_batch.seq_lens_cpu_upper_bound[:n].numpy()
        plan = plan_segments(bounds, qsl, seq, n)
        if plan is None:
            return metadata
        if cudagraph_mode == CUDAGraphMode.FULL:
            raise RuntimeError(
                "strands-decider-v2: a step with prefix_bidir prompt tokens was scheduled as a FULL "
                "CUDA graph"
            )
        self._attach(metadata, plan)
        return metadata

    def _attach(self, metadata: dict[str, Any], plan: tuple[list, list]) -> None:
        bidir, rest = plan
        done: set[int] = set()
        for md in metadata.values():
            if md is None or id(md) in done or not hasattr(md, "block_table"):
                continue
            done.add(id(md))
            dev = md.query_start_loc.device
            b_rows, _ = _rows(bidir, dev)
            b_len = np.array([b - a for _, a, b, _ in bidir], np.int64)
            host = np.concatenate(
                [
                    np.concatenate([[0], np.cumsum(b_len)]),  # cu_bidir
                    [k for *_, k in bidir],  # keys
                    [r for r, *_ in bidir],
                ]
            ).astype(np.int64)  # src rows
            g = torch.from_numpy(host).to(dev, non_blocking=True)
            nb = len(bidir)
            cu_b = g[: nb + 1].to(torch.int32)
            keys_b = g[nb + 1 : 2 * nb + 1].to(torch.int32)
            src_b = g[2 * nb + 1 :]
            bidir_md = _restrict(
                md,
                num_actual_tokens=int(b_len.sum()),
                max_query_len=int(b_len.max()),
                query_start_loc=cu_b,
                max_seq_len=int(max(k for *_, k in bidir)),
                seq_lens=keys_b,
                block_table=md.block_table.index_select(0, src_b),
                causal=False,
            )
            rest_md, r_rows = None, None
            if rest:
                r_rows, _ = _rows(rest, dev)
                r_len = np.array([b - a for _, a, b, _ in rest], np.int64)
                host = np.concatenate(
                    [np.concatenate([[0], np.cumsum(r_len)]), [r for r, *_ in rest]]
                ).astype(np.int64)
                g = torch.from_numpy(host).to(dev, non_blocking=True)
                nr = len(rest)
                src_r = g[nr + 1 :]
                rest_md = _restrict(
                    md,
                    num_actual_tokens=int(r_len.sum()),
                    max_query_len=int(r_len.max()),
                    query_start_loc=g[: nr + 1].to(torch.int32),
                    max_seq_len=md.max_seq_len,
                    seq_lens=md.seq_lens.index_select(0, src_r),
                    block_table=md.block_table.index_select(0, src_r),
                    causal=True,
                )
            setattr(
                md,
                SPLIT_ATTR,
                Split(bidir_rows=b_rows, rest_rows=r_rows, bidir=bidir_md, rest=rest_md),
            )


def _restrict(md: Any, **fields: Any) -> Any:
    """``md`` restricted to some segments: no AOT schedule, no mm_prefix ranges, no R-SWA mask."""
    from dataclasses import replace

    extra = {
        k: None
        for k in ("scheduler_metadata", "mm_prefix_query_range_tensor", "rswa_prefix_lens")
        if hasattr(md, k)
    }
    out = replace(md, **fields, **extra)
    if hasattr(out, SPLIT_ATTR):
        delattr(out, SPLIT_ATTR)
    return out
