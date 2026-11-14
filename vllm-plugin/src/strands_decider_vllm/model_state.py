"""Model Runner V2 state for prefix_bidir: each request's state boundary, turned into split attention metadata.

A request sends its boundary S (the number of tokens in the prompt's state part) in ``vllm_xargs.prefix_end``; the
engine core has validated it (:mod:`strands_decider_vllm.engine`). In a step that computes any of its prompt tokens,
its scheduled tokens ``[c, e)`` (c being what is already computed or cached, e the step's end) become up to two
non-causal segments over the prompt and one causal tail:

- state queries ``[c, S)`` over keys ``[0, S)`` (absent when c reaches S);
- question queries ``[max(c, S), L)`` over keys ``[0, L)``;
- rows from L on, which only a request resumed after a preemption with output tokens has, causal as in a decode.

The plugin's startup checks make a prompt run to its end in one step (e >= L): chunked prefill is off, and the settings
that can still split a prompt are refused. Should a prompt nevertheless be scheduled in parts, its rows attend only to
keys already computed and the step is logged as an error; a worker never raises because of a request.

Every attention layer of the model gets this split (the model class builds every layer without a sliding window), through
:class:`~strands_decider_vllm.attention.SplitFlashAttentionImpl`. Decode rows and requests of other kinds stay
causal. Cached keys are valid because the block hashes of a prefix_bidir request carry digests of its state and question
(:mod:`strands_decider_vllm.hashing`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from . import PREFIX_END_KEY, logger
from .attention import SPLIT_ATTR, Split, SplitFlashAttentionImpl
from .planning import plan_segments, prefix_end_of

__all__ = ["PrefixBidirModelState", "plan_segments", "prefix_end_of"]


def _split_impls() -> tuple[type, ...]:
    """The plugin's attention implementations: FlashAttention, and CPU attention where vLLM has it."""
    impls: tuple[type, ...] = (SplitFlashAttentionImpl,)
    try:
        from .attention_cpu import SplitCPUAttentionImpl
    except ImportError:  # a vLLM build without CPU attention
        return impls
    return (*impls, SplitCPUAttentionImpl)


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
        self._warned_partial = False
        layers = {
            n: layer
            for n, layer in vllm_config.compilation_config.static_forward_context.items()
            if getattr(layer, "attn_type", None) == "decoder" and hasattr(layer, "get_attn_backend")
        }
        bad = sorted(
            n
            for n, layer in layers.items()
            if not isinstance(getattr(layer, "impl", None), _split_impls())
        )
        if bad:
            raise ValueError(
                f"strands-decider: attention layers not on the plugin's attention: {bad[:4]}... "
                "(use the default backend: FLASH_ATTN on GPUs, CPU_ATTN on CPUs)"
            )
        windowed = sorted(
            n
            for n, layer in layers.items()
            if getattr(layer, "sliding_window", None) is not None
            and (
                layer.sliding_window < self.max_model_len
                or getattr(layer.impl, "sliding_window", None)
                != getattr(layer.impl, "no_window", (-1, -1))
            )
        )
        if windowed:
            raise ValueError(
                f"strands-decider: {len(windowed)} layers keep a sliding window ({windowed[0]}, ...); "
                "prefix_bidir needs every window off (StrandsDeciderForCausalLM builds its layers without one)"
            )
        logger.info(
            "strands-decider: prefix_bidir on %d attention layers (no sliding windows)",
            len(layers),
        )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        rid = new_req_data.req_id
        self._bounds.pop(rid, None)
        sp = new_req_data.sampling_params
        # The original prompt, also for a request resumed after a preemption (whose prefill_token_ids add its outputs).
        length = len(new_req_data.prompt_token_ids or ())
        extra = sp.extra_args if sp is not None else None
        s = prefix_end_of(extra, length)
        if s is None:
            if extra and PREFIX_END_KEY in extra:
                logger.error(
                    "strands-decider: request %s reached the worker with an invalid prefix_end %r for a "
                    "%d-token prompt; serving it causally (the engine core should have refused it)",
                    rid,
                    extra[PREFIX_END_KEY],
                    length,
                )
            elif not self._warned_causal:
                # vLLM's own startup warm-up adds dummy requests without a boundary in the worker; client requests
                # without one are refused in the engine core. Either way such a request is served causally.
                self._warned_causal = True
                logger.debug(
                    "strands-decider: a request without vllm_xargs.prefix_end (vLLM's warm-up) is served "
                    "causally"
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
        if not self._warned_partial and any(
            int(seq[i]) < length for i, (_, length) in bounds.items()
        ):
            self._warned_partial = True
            logger.error(
                "strands-decider: a prefix_bidir prompt was scheduled in parts, so its rows attend only to the "
                "keys computed so far and its answer is wrong. The plugin refuses the settings known "
                "to split a prompt; please report this configuration. (Logged once.)"
            )
        plan = plan_segments(bounds, qsl, seq, n)
        if plan is None:
            return metadata
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Unreachable: check_config refuses cudagraph_mode FULL, and under the other modes only uniform
            # one-token batches run as FULL graphs, which plan_segments never splits. A graph would replay its
            # captured attention and ignore the split, so refusing to run is the only safe answer.
            raise RuntimeError(
                "strands-decider: a step with prefix_bidir prompt tokens was scheduled as a FULL "
                "CUDA graph"
            )
        self._attach(metadata, plan)
        return metadata

    def _attach(self, metadata: dict[str, Any], plan: tuple[list, list]) -> None:
        """Attach the plan's :class:`Split` to every distinct FlashAttention metadata object of the step (one per
        KV-cache group). The rows and the plan's index tensors are built once per device; only the block tables and
        sequence lengths, which are per group, are gathered per metadata object."""
        bidir, rest = plan
        nb, nr = len(bidir), len(rest)
        b_len = np.array([b - a for _, a, b, _ in bidir], np.int64)
        r_len = np.array([b - a for _, a, b, _ in rest], np.int64)
        host = np.concatenate(
            [
                np.concatenate([[0], np.cumsum(b_len)]),  # cu_bidir
                [k for *_, k in bidir],  # keys per bidir segment
                [r for r, *_ in bidir],  # source rows of the bidir segments
                np.concatenate([[0], np.cumsum(r_len)]) if nr else np.zeros(0, np.int64),  # cu_rest
                [r for r, *_ in rest],  # source rows of the causal segments
            ]
        ).astype(np.int64)
        max_keys = int(max(k for *_, k in bidir))
        per_device: dict[torch.device, tuple] = {}
        done: set[int] = set()
        for md in metadata.values():
            if md is None or id(md) in done or not hasattr(md, "block_table"):
                continue
            done.add(id(md))
            dev = md.query_start_loc.device
            if dev not in per_device:
                g = torch.from_numpy(host).to(dev, non_blocking=True)
                o = 2 * nb + 1
                per_device[dev] = (
                    _rows(bidir, dev)[0],
                    _rows(rest, dev)[0] if nr else None,
                    g[: nb + 1].to(torch.int32),  # cu_bidir
                    g[nb + 1 : o].to(torch.int32),  # keys
                    g[o : o + nb],  # bidir source rows
                    g[o + nb : o + nb + nr + 1].to(torch.int32) if nr else None,  # cu_rest
                    g[o + nb + nr + 1 :] if nr else None,  # rest source rows
                )
            b_rows, r_rows, cu_b, keys_b, src_b, cu_r, src_r = per_device[dev]
            bidir_md = _restrict(
                md,
                num_actual_tokens=int(b_len.sum()),
                max_query_len=int(b_len.max()),
                query_start_loc=cu_b,
                max_seq_len=max_keys,
                seq_lens=keys_b,
                block_table=md.block_table.index_select(0, src_b),
                causal=False,
            )
            rest_md = None
            if nr:
                rest_md = _restrict(
                    md,
                    num_actual_tokens=int(r_len.sum()),
                    max_query_len=int(r_len.max()),
                    query_start_loc=cu_r,
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
    """``md`` restricted to some segments: no AOT schedule (CPU attention plans its own), no mm_prefix ranges, no R-SWA
    mask, no per-sequence causal flags."""
    from dataclasses import replace

    extra = {
        k: None
        for k in (
            "scheduler_metadata",
            "mm_prefix_query_range_tensor",
            "rswa_prefix_lens",
            "dynamic_causal",
        )
        if hasattr(md, k)
    }
    out = replace(md, **fields, **extra)
    if hasattr(out, SPLIT_ATTR):
        delattr(out, SPLIT_ATTR)
    return out
