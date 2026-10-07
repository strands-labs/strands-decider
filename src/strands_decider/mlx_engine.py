"""Serving on Apple silicon through MLX: the torso runs on Metal via mlx-lm, the rest as on torch.

On MPS the Qwen3.5 torso has no fused Gated DeltaNet kernel (`flash-linear-attention` needs
Triton, `causal_conv1d` needs CUDA) and every op is a separate dispatch. mlx-lm runs the same
architecture on Metal with fused kernels, including a recurrent Gated DeltaNet kernel. Measured
with v19 on an M4 Pro (`evaluation/bench_local.py`, median): one question at 222 / 1,118 / 4,094
input tokens takes 162 / 682 / 2,685 ms on MPS and 113 / 486 / 1,764 ms here.

Only the torso forward moves. `MLXEngine` subclasses `SystemOneEngine`, so rendering,
tokenisation, the question-first truncation, option positions, batching, temperatures and the
masked softmax are the torch engine's own code, and the head is the checkpoint's own fp32
module, run by torch on the CPU. Both of `evaluate`'s paths are kept: one question forwards the
whole prompt; several encode the state once into an mlx-lm prompt cache, which is copied
across the question suffixes.

The LoRA adapter is folded into the base weights at load, W + (alpha / r) B A, formed in fp32
on the CPU and rounded once to the torso dtype; the torch path keeps it unmerged. Against v19 in
fp32 on the CPU, over `evaluation/device_parity.py`'s 54 answers, no answer changes and the
largest probability difference is 0.0138 here and 0.0051 on MPS. Most of the gap is the merge's
rounding: MPS with the adapter merged the same way differs by 0.0105, and this engine in fp32 by
0.0036.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
from mlx.utils import tree_flatten
from mlx_lm.models.cache import BatchKVCache, KVCache, make_prompt_cache
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

from .infer import EngineConfig, SystemOneEngine
from .modeling import (
    StrandsDeciderConfig,
    apply_temperature,
    base_revision,
    build_head,
    checkpoint_dir,
    config_path,
    load_head_state,
    masked_log_softmax,
)
from .prompting import RenderedQuestion
from .schema import SystemOneRequest, SystemOneResponse

# MLX keeps freed buffers for reuse, by default up to its memory limit (about 95% of RAM on
# a 48 GB M4 Pro), and a new request shape allocates new ones. 1 GiB holds a 2B torso's
# working set; it cost no latency on the M4 Pro and kept the process near its weights' size.
DEFAULT_CACHE_LIMIT = 1 << 30

_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
# mlx-lm names the text-only Qwen3.5 checkpoint's type after the multimodal one, whose module
# loads either layout (the multimodal wrapper takes a bare text config as its text_config).
_MODEL_TYPES = {"qwen3_5_text": "qwen3_5"}


class _Readout:
    """What the engine reads from `StrandsDeciderModel` besides the torso."""

    def __init__(self, config: StrandsDeciderConfig, tokenizer: Any, head: torch.nn.Module) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.head = head


def merge_lora(lm: Any, adapter_dir: str, prefix: str) -> int:
    """Fold a PEFT LoRA adapter into `lm`'s weights in place. Returns the number merged.

    PEFT names a target `base_model.model.<path>`, where <path> is relative to the torch
    torso; mlx-lm names the same weight `<prefix><path>.weight`. Only plain linear LoRA is
    folded: any other adapter tensor (an embedding LoRA, DoRA magnitudes, a LoRA bias, saved
    modules, trainable tokens) is refused rather than skipped, and so are the options that
    change the arithmetic without adding tensors. Each weight is swapped in as soon as it is
    formed, so the old and new copies of only one weight are alive at a time.
    """
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    unsupported = [key for key in ("fan_in_fan_out", "rank_pattern", "alpha_pattern") if cfg.get(key)]
    if unsupported:
        raise ValueError(f"{adapter_dir}: the MLX backend cannot merge an adapter with {unsupported}")
    adapter = mx.load(os.path.join(adapter_dir, "adapter_model.safetensors"))
    stems = {name.removesuffix(".lora_A.weight") for name in adapter if name.endswith(".lora_A.weight")}
    expected = {stem + suffix for stem in stems for suffix in (".lora_A.weight", ".lora_B.weight")}
    if set(adapter) != expected:
        unknown = sorted(set(adapter) - expected) + sorted(expected - set(adapter))
        raise ValueError(
            f"{adapter_dir}: the MLX backend merges only lora_A/lora_B weight pairs; "
            f"cannot merge {unknown[:3]}{' ...' if len(unknown) > 3 else ''}"
        )
    rank = cfg["r"]
    scale = cfg["lora_alpha"] / (rank**0.5 if cfg.get("use_rslora") else rank)
    params = dict(tree_flatten(lm.parameters()))
    merged = 0
    # Metal's fp32 matmul is a reduced-precision fast path; the merge runs once and on the CPU.
    with mx.stream(mx.cpu):
        for stem in sorted(stems):
            target = prefix + stem.removeprefix("base_model.model.") + ".weight"
            base = params.pop(target, None)
            if base is None:
                raise ValueError(f"adapter tensor {stem} has no weight {target} in the MLX torso")
            if not mx.issubdtype(base.dtype, mx.floating):
                raise ValueError(f"{target} is {base.dtype}; the MLX backend cannot merge into quantised weights")
            lora_a, lora_b = adapter.pop(stem + ".lora_A.weight"), adapter.pop(stem + ".lora_B.weight")
            delta = lora_b.astype(mx.float32) @ lora_a.astype(mx.float32)
            weight = (base.astype(mx.float32) + scale * delta).astype(base.dtype)
            mx.eval(weight)
            lm.load_weights([(target, weight)], strict=False)
            del base, lora_a, lora_b, delta, weight
            merged += 1
    return merged


class MLXEngine(SystemOneEngine):
    """`SystemOneEngine` with the torso forward on MLX. Build one with `load_mlx_engine`."""

    def __init__(self, decoder: Any, cache_owner: Any, readout: _Readout, config: EngineConfig,
                 state_cache_entries: int = 8) -> None:
        # Not super().__init__(): that moves a torch torso to a torch device. These are the
        # attributes the inherited helpers read; `device` is where their small index and
        # temperature tensors live, next to the head.
        self.cfg = config
        self.model = readout
        self.tok = readout.tokenizer
        self.device = "cpu"
        self._decoder = decoder
        self._cache_owner = cache_owner
        # Cross-request state cache: state token ids -> a batch-1 snapshot of the prompt
        # cache after encoding that state. Only `_slot_probs_shared_prefix` populates it
        # (it already builds a state-only prefix), and both paths read it, so no request
        # ever pays an extra forward to fill an entry. `evaluate` holds `_lock` throughout,
        # which also guards this dict.
        self.state_cache_entries = state_cache_entries
        self._state_cache: OrderedDict[tuple[int, ...], list[Any]] = OrderedDict()
        self.state_encodes = 0  # state-only forwards served, cache or not
        # One evaluation at a time: `_fit` leaves the request's option offsets on the engine
        # (`_last_offsets`) for `_option_idx` to read, so two requests in the server's thread
        # pool would read each other's. The torch engine has the same race (#9).
        self._lock = threading.Lock()

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        with self._lock:
            return super().evaluate(request)

    def _hidden(self, rows: list[list[int]], cache: list[Any] | None = None) -> Any:
        """Last hidden states [N, L, d] of right-padded rows. Both layer kinds are causal, so
        no real position reads a pad and no attention mask is needed."""
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        width = max(len(row) for row in rows)
        ids = mx.array([row + [pad_id] * (width - len(row)) for row in rows], dtype=mx.int32)
        hidden = self._decoder(ids, cache=cache)
        mx.eval(hidden)
        return hidden

    def _probs(self, hidden: Any, last: list[int], opt_idx: torch.Tensor | None,
               n_slots: list[int], kinds: list[str]) -> torch.Tensor:
        """Read the head's inputs out of `hidden` on Metal, then score them on torch.

        Row i is pooled at `last[i]`; a pointer head also reads `opt_idx[i]`, where padding
        (-1) reads position 0 as `gather_options` does and is masked out by the softmax.
        """
        rows, width, dim = hidden.shape
        positions = [row * width + position for row, position in enumerate(last)]
        options = opt_idx.clamp_min(0).tolist() if opt_idx is not None else []
        for row, row_options in enumerate(options):
            positions += [row * width + position for position in row_options]
        picked = hidden.reshape(rows * width, dim)[mx.array(positions, dtype=mx.int32)]
        states = _to_torch(picked.astype(mx.float32)).reshape(len(positions), dim)
        pooled = states[:rows]
        with torch.inference_mode():
            if options:
                raw = self.model.head(pooled, states[rows:].reshape(rows, len(options[0]), dim))
            else:
                raw = self.model.head(pooled)
            logits = apply_temperature(raw, self._temperatures(kinds))
            return masked_log_softmax(logits, torch.tensor(n_slots)).exp()

    def _option_positions(self, rendered: list[RenderedQuestion] | None, base: int) -> torch.Tensor | None:
        if self.model.config.head_type != "pointer":
            return None
        assert rendered is not None
        return self._option_idx(rendered, base)

    def _state_snapshot(self, state: list[int]) -> list[Any] | None:
        """The cached batch-1 prompt-cache layers for this state, or None. LRU-touched."""
        key = tuple(state)
        snap = self._state_cache.get(key)
        if snap is not None:
            self._state_cache.move_to_end(key)
        return snap

    def _slot_probs_batched(
        self, state_text: str, question_texts: list[str], n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        state, questions = self._fit(state_text, question_texts)
        snap = self._state_snapshot(state) if self.state_cache_entries else None
        if snap is not None:
            # A cached state serves even one question through the suffix-only path: the
            # miss path's single whole-prompt forward is cheaper than state + suffix, but
            # a hit skips the state's forward entirely.
            batch = [type(layer).merge([layer] * len(questions)) for layer in snap]
            hidden = self._hidden(questions, batch)
            probs = self._probs(hidden, [len(question) - 1 for question in questions],
                                self._option_positions(rendered, 0), n_slots, kinds)
            return probs, len(state) + sum(len(question) for question in questions)
        rows = [state + question for question in questions]
        hidden = self._hidden(rows)
        probs = self._probs(hidden, [len(row) - 1 for row in rows],
                            self._option_positions(rendered, len(state)), n_slots, kinds)
        return probs, sum(len(row) for row in rows)

    def _slot_probs_shared_prefix(
        self, state_text: str, question_texts: list[str], n_slots: list[int], kinds: list[str],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        state, questions = self._fit(state_text, question_texts)
        snap = self._state_snapshot(state) if self.state_cache_entries else None
        if snap is None:
            prefix = make_prompt_cache(self._cache_owner)
            self._hidden([state], prefix)
            self.state_encodes += 1
            if self.state_cache_entries:
                snap = [_snapshot(layer) for layer in prefix]
                key = tuple(state)
                self._state_cache[key] = snap
                while len(self._state_cache) > self.state_cache_entries:
                    self._state_cache.popitem(last=False)
            else:
                snap = list(prefix)
        # `merge` copies each layer's state into a batch of len(questions); the snapshot is untouched.
        batch = [type(layer).merge([layer] * len(questions)) for layer in snap]
        hidden = self._hidden(questions, batch)
        probs = self._probs(hidden, [len(question) - 1 for question in questions],
                            self._option_positions(rendered, 0), n_slots, kinds)
        return probs, len(state) + sum(len(question) for question in questions)


def _snapshot(layer: Any) -> Any:
    """A batch-1 copy of a prompt-cache layer that `merge` can consume again.

    `KVCache.merge` returns a `BatchKVCache` whose `offset` is a per-row mx array; merging
    that a second time slices with the array and raises ("Slice indices must be integers").
    mlx-lm only ever merges a fresh cache, so the conversion back is on us: a plain KVCache
    carrying the same keys/values merges as often as needed. `ArraysCache.merge` (the
    Gated DeltaNet states) already returns a re-mergeable cache and passes through.
    """
    merged: Any = type(layer).merge([layer])  # Any: mlx-lm's cache stubs type keys as None
    if type(merged) is BatchKVCache and type(layer) is not BatchKVCache:
        plain = KVCache()
        plain.keys, plain.values = merged.keys, merged.values
        keys: Any = merged.keys  # Any: the type() narrowing above re-types merged from mlx-lm's stubs
        plain.offset = int(keys.shape[2])  # the batch-1 seq length: the cached token count
        return plain
    return merged


def _to_torch(array: Any) -> torch.Tensor:
    """A float32 MLX array as a CPU torch tensor (a copy, through the buffer protocol)."""
    return torch.frombuffer(bytearray(memoryview(array)), dtype=torch.float32)


def _load_torso(base: Path, dtype: str) -> tuple[Any, Any, Any, str]:
    """mlx-lm's model for `base`: (the model, its decoder, which returns the last hidden
    states, the module that makes its prompt cache, and the decoder's weight-name prefix)."""
    with open(base / "config.json", encoding="utf-8") as fh:
        model_type = json.load(fh)["model_type"]
    overrides = {"model_type": _MODEL_TYPES[model_type]} if model_type in _MODEL_TYPES else None
    lm, _ = load_model(base, model_config=overrides)
    owner = getattr(lm, "language_model", lm)  # multimodal checkpoints keep the text model here
    prefix = "language_model.model." if owner is not lm else "model."
    if dtype not in _DTYPES:
        raise ValueError(f"the MLX backend has no dtype {dtype!r}; use one of {sorted(_DTYPES)}")
    # The checkpoint's dtype, as torch loads it; mlx-lm's predicate keeps e.g. A_log in fp32.
    keep = getattr(owner, "cast_predicate", None) or (lambda _path: True)
    target = _DTYPES[dtype]
    cast = [(path, value.astype(target)) for path, value in tree_flatten(lm.parameters())
            if mx.issubdtype(value.dtype, mx.floating) and value.dtype != target and keep(path)]
    if cast:
        lm.load_weights(cast, strict=False)
    # Evaluate the casts now: a lazy array belongs to this thread's stream, and the server
    # evaluates requests on its thread pool.
    mx.eval(lm.parameters())
    return lm, owner.model, owner, prefix


def load_mlx_engine(
    checkpoint: str,
    config: EngineConfig | None = None,
    *,
    cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT,
    state_cache_entries: int = 8,
) -> MLXEngine:
    """The MLX counterpart of `infer.load_engine`: a local checkpoint directory or a Hub repo id.

    `config` is the torch engine's `EngineConfig`, with its device set to "mlx".
    `cache_limit_bytes` sets MLX's process-wide buffer cache limit; None leaves it as it is.
    `state_cache_entries` caps the cross-request state KV cache (LRU); 0 disables it.
    """
    path = checkpoint_dir(checkpoint)
    decider_config = StrandsDeciderConfig.from_json(config_path(path))
    decider_config.base_revision = base_revision(path, decider_config)
    lora_dir = os.path.join(path, "lora")
    if decider_config.use_lora and not os.path.isdir(lora_dir):
        raise FileNotFoundError(
            f"{lora_dir}: missing, but {os.path.basename(config_path(path))} sets use_lora"
        )
    head_state = load_head_state(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cache_limit_bytes is not None:
        mx.set_cache_limit(cache_limit_bytes)
    lm, decoder, owner, prefix = _load_torso(
        Path(checkpoint_dir(decider_config.base_model, decider_config.base_revision)),
        decider_config.torch_dtype,
    )
    if decider_config.use_lora:
        merge_lora(lm, lora_dir, prefix)
    mx.clear_cache()  # the merge's fp32 transients
    head = build_head(decider_config, int(decoder.embed_tokens.weight.shape[1]))
    head.load_state_dict(head_state)
    head.to(torch.float32).eval()
    engine_config = replace(config or EngineConfig(), device="mlx")
    return MLXEngine(decoder, owner, _Readout(decider_config, tokenizer, head), engine_config,
                     state_cache_entries=state_cache_entries)
