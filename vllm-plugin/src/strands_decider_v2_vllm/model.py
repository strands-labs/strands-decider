"""``StrandsDeciderV2ForCausalLM``: vLLM's Gemma 4 text decoder with the v2 readout and prefix_bidir.

The checkpoint is a bare ``Gemma4TextModel`` (keys ``embed_tokens.*``, ``layers.*``, ``norm.*``, and the per-layer
embedding tensors of E2B and E4B), ``readout.safetensors`` with one tensor ``weight`` [255, hidden] and
``decision_config.json``. Serve it with ``--hf-overrides '{"architectures": ["StrandsDeciderV2ForCausalLM"],
"use_bidirectional_attention": null}'`` (the second key turns off vLLM's image PrefixLM mode, which the 12B, 26B-A4B
and 31B configs would otherwise switch on; v2 is text-only);
every attention layer is built without a sliding window (see ``__init__``).

``compute_logits`` returns, for each sampled position (the last prompt token), the readout's 255 values at the codes'
token ids and -inf at every other id, as the v2 decision specification defines them: a bf16 linear on the final-norm
hidden state, then in float32 ``cap * tanh(z / cap)`` with Gemma's ``final_logit_softcapping``, divided by the
checkpoint temperature. A completion with ``logprobs: 255`` therefore returns the log-softmax over the 255 codes;
renormalizing over a question's first ``n`` codes gives the specification's probabilities, a softmax in which the codes
past the question's options take no part.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

from .model_state import PrefixBidirModelState

logger = logging.getLogger("strands_decider_v2_vllm")
N_CODES = 255


def check_config(vllm_config: VllmConfig) -> None:
    """Refuse engine settings under which prefix_bidir would be served incorrectly."""
    mc, sc, cc = vllm_config.model_config, vllm_config.scheduler_config, vllm_config.cache_config
    pc = vllm_config.parallel_config
    problems = []
    if getattr(mc, "is_mm_prefix_lm", False):
        problems.append(
            "vLLM's multimodal PrefixLM mode is on (the 12B, 26B-A4B and 31B configs carry "
            'use_bidirectional_attention: "vision"), which selects the TRITON_FLASH_ATTN composite backend; '
            'add "use_bidirectional_attention": null to --hf-overrides (it only concerns image tokens)'
        )
    if not getattr(vllm_config, "use_v2_model_runner", False):
        problems.append(
            "Model Runner V1 is in use; prefix_bidir needs Model Runner V2 (VLLM_USE_V2_MODEL_RUNNER=1)"
        )
    if sc.enable_chunked_prefill:
        problems.append(
            "chunked prefill is on; pass --no-enable-chunked-prefill (a prompt must be computed whole)"
        )
    if sc.max_num_batched_tokens < mc.max_model_len:
        problems.append(
            f"--max-num-batched-tokens {sc.max_num_batched_tokens} < --max-model-len {mc.max_model_len}"
        )
    if cc.enable_prefix_caching and not str(
        getattr(cc, "prefix_caching_hash_algo", "sha256")
    ).startswith("sha256"):
        problems.append("--prefix-caching-hash-algo must be sha256 or sha256_cbor")
    if vllm_config.speculative_config is not None:
        problems.append("speculative decoding is not supported")
    if vllm_config.kv_transfer_config is not None:
        problems.append("KV connectors are not supported")
    if getattr(cc, "kv_sharing_fast_prefill", False):
        problems.append("--kv-sharing-fast-prefill is not supported")
    for name in (
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(pc, name, 1) not in (None, 1):
            problems.append(f"{name} > 1 is not supported")
    if problems:
        raise ValueError(
            "strands-decider-v2 cannot serve with this configuration:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


class StrandsDeciderV2ForCausalLM(Gemma4ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        check_config(vllm_config)
        # v2 applies the prefix_bidir mask on the sliding-window layers too, with no window cut (the v2 reference
        # implementation gives both layer types the same mask). Each sliding layer is built with a window of
        # max_model_len tokens, which cuts nothing: its KV cache group keeps every token of a request (a
        # sliding-window group whose window covers the whole context), and vLLM's hybrid KV-cache manager keeps one
        # group per layer shape, as for stock Gemma 4.
        # FlashAttention then gets window (-1, -1) on every layer. (Building the layers without a window instead puts
        # both head sizes in one UniformTypeKVCacheSpecs group, which vLLM 0.33 cannot combine with E2B/E4B's KV-shared
        # layers; and the transformers 5.20 config refuses sliding_window None.)
        import vllm.model_executor.models.gemma4 as gemma4

        attention_cls = gemma4.Attention
        no_cut = vllm_config.model_config.max_model_len

        def attention_without_window_cut(*args, **kwargs):
            if kwargs.get("per_layer_sliding_window") is not None:
                kwargs["per_layer_sliding_window"] = no_cut
            return attention_cls(*args, **kwargs)

        gemma4.Attention = attention_without_window_cut
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            gemma4.Attention = attention_cls
        n = 0
        for module in self.modules():
            impl = getattr(module, "impl", None)
            if impl is not None and getattr(module, "sliding_window", None) is not None:
                if module.sliding_window < no_cut:
                    raise ValueError(
                        f"strands-decider-v2: {module.layer_name} kept window {module.sliding_window}"
                    )
                impl.sliding_window = (-1, -1)
                n += 1
        logger.info(
            "strands-decider-v2: %d sliding-window layers built with a %d-token window (no cut)",
            n,
            no_cut,
        )
        self._init_readout(vllm_config)

    def _init_readout(self, vllm_config: VllmConfig) -> None:
        self.model_dir = vllm_config.model_config.model
        dc = json.loads(open(os.path.join(self.model_dir, "decision_config.json")).read())
        problems = {
            k: dc.get(k)
            for k, v in (
                ("format_version", 1),
                ("attention_mode", "prefix_bidir"),
                ("pooling", "last"),
            )
            if dc.get(k) != v
        }
        if problems or len(dc.get("token_ids") or ()) != N_CODES:
            raise ValueError(
                f"strands-decider-v2: unsupported decision_config {problems}, "
                f"{len(dc.get('token_ids') or ())} codes"
            )
        self.temperature = float(dc["temperature"])
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(f"strands-decider-v2: bad temperature {self.temperature}")
        self.softcap = getattr(self.config, "final_logit_softcapping", None)
        self.register_buffer(
            "code_ids", torch.tensor(dc["token_ids"], dtype=torch.long), persistent=False
        )
        self.readout = nn.Linear(
            self.config.hidden_size, N_CODES, bias=False, dtype=vllm_config.model_config.dtype
        )
        logger.info(
            "strands-decider-v2: readout over %d codes, softcap %s, temperature %s",
            N_CODES,
            self.softcap,
            self.temperature,
        )

    @classmethod
    def get_model_state_cls(cls) -> type[PrefixBidirModelState]:
        return PrefixBidirModelState

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        codes = self.readout(hidden_states.to(self.readout.weight.dtype)).float()
        if self.softcap:
            codes = torch.tanh(codes / self.softcap) * self.softcap
        if self.temperature != 1.0:
            codes = codes / self.temperature
        out = torch.full(
            (hidden_states.shape[0], self.config.vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        out[:, self.code_ids] = codes
        return out

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def renamed():
            for name, w in weights:
                if name == "weight":  # readout.safetensors, if a loader ever passes it here
                    continue
                yield (name if name.startswith("model.") else "model." + name), w

        loaded = super().load_weights(renamed())
        from safetensors.torch import load_file

        w = load_file(os.path.join(self.model_dir, "readout.safetensors"))["weight"]
        if tuple(w.shape) != tuple(self.readout.weight.shape):
            raise ValueError(
                f"strands-decider-v2: readout {tuple(w.shape)} != {tuple(self.readout.weight.shape)}"
            )
        with torch.no_grad():
            self.readout.weight.copy_(
                w.to(device=self.readout.weight.device, dtype=self.readout.weight.dtype)
            )
        loaded.add("readout.weight")
        return loaded
