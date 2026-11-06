"""``StrandsDeciderV2ForCausalLM``: vLLM's Gemma 4 text decoder with the v2 readout and prefix_bidir.

The checkpoint is a bare ``Gemma4TextModel`` (keys ``embed_tokens.*``, ``layers.*``, ``norm.*``, and the per-layer
embedding tensors of E2B and E4B), ``readout.safetensors`` with one tensor ``weight`` [255, hidden] and
``decision_config.json``. The model may be a local directory or a Hugging Face Hub repo id: ``decision_config.json`` is
read the way vLLM reads a model's other config files, and the readout is a second weight source of vLLM's default
loader. Serve it with ``--hf-overrides '{"architectures": ["StrandsDeciderV2ForCausalLM"],
"use_bidirectional_attention": null}'`` (the second key turns off vLLM's image PrefixLM mode, which the 12B, 26B-A4B
and 31B configs would otherwise switch on; v2 is text-only); every attention layer is built without a sliding window
(see ``__init__``).

``compute_logits`` returns, for each sampled position (the last prompt token), the readout's 255 values at the codes'
token ids and -inf at every other id, as the v2 decision specification defines them: a bf16 linear on the final-norm
hidden state, then in float32 ``cap * tanh(z / cap)`` with the checkpoint's ``final_logit_softcapping``, divided by the
checkpoint temperature. A completion with ``logprobs: 255`` therefore returns the log-softmax over the 255 codes;
renormalizing over a question's first ``n`` codes gives the specification's probabilities, a softmax in which the codes
past the question's options take no part.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

from . import check_vllm_version, logger
from .decision_config import DecisionConfigError, read_decision_config
from .model_state import PrefixBidirModelState
from .validation import N_CODES

READOUT_FILE = "readout.safetensors"
DECISION_CONFIG_FILE = "decision_config.json"
#: The name under which the readout's tensor reaches ``load_weights`` from its weight source.
READOUT_WEIGHT = "readout.weight"


def check_config(vllm_config: VllmConfig) -> None:
    """Refuse engine settings under which prefix_bidir would be served incorrectly, or under which one request could
    stop the engine. Every attribute is read explicitly, so a renamed vLLM attribute fails here rather than pass."""
    mc, sc, cc = vllm_config.model_config, vllm_config.scheduler_config, vllm_config.cache_config
    pc, comp = vllm_config.parallel_config, vllm_config.compilation_config
    problems = []
    if mc.is_mm_prefix_lm:
        problems.append(
            "vLLM's multimodal PrefixLM mode is on (the 12B, 26B-A4B and 31B configs carry "
            'use_bidirectional_attention: "vision"), which selects the TRITON_FLASH_ATTN composite backend; '
            'add "use_bidirectional_attention": null to --hf-overrides (it only concerns image tokens)'
        )
    if not vllm_config.use_v2_model_runner:
        problems.append(
            "Model Runner V1 is in use; prefix_bidir needs Model Runner V2 (VLLM_USE_V2_MODEL_RUNNER=1)"
        )
    # A prompt must be computed in one step: its state and question attend to keys that a later chunk would compute.
    if sc.enable_chunked_prefill:
        problems.append(
            "chunked prefill is on; pass --no-enable-chunked-prefill (a prompt must be computed whole)"
        )
    if sc.max_num_batched_tokens < mc.max_model_len:
        problems.append(
            f"--max-num-batched-tokens {sc.max_num_batched_tokens} < --max-model-len {mc.max_model_len}"
        )
    if sc.max_num_scheduled_tokens is not None and sc.max_num_scheduled_tokens < mc.max_model_len:
        problems.append(
            f"--max-num-scheduled-tokens {sc.max_num_scheduled_tokens} < --max-model-len {mc.max_model_len}: with "
            "chunked prefill off, a longer prompt can never be scheduled and blocks the queue behind it"
        )
    if 0 < sc.long_prefill_token_threshold < mc.max_model_len:
        problems.append(
            f"--long-prefill-token-threshold {sc.long_prefill_token_threshold} splits prompts longer than that "
            "into chunks even with chunked prefill off; leave it at 0"
        )
    if pc.data_parallel_size > 1 and (pc.enable_dbo or pc.ubatch_size > 1):
        problems.append(
            "data parallelism with micro-batching (--enable-dbo or --ubatch-size > 1) can split one request's "
            "prompt across micro-batches"
        )
    if comp.cudagraph_mode == CUDAGraphMode.FULL:
        problems.append(
            "cudagraph_mode FULL captures prefill steps as full CUDA graphs, which replay their captured "
            "attention and cannot run the prefix_bidir split; use the default (FULL_AND_PIECEWISE), "
            "FULL_DECODE_ONLY, PIECEWISE or NONE"
        )
    if mc.logprobs_mode not in PROCESSED_LOGPROBS_MODES:
        problems.append(
            f"--logprobs-mode {mc.logprobs_mode}: the codes' probabilities are the processed logprobs, taken "
            "after allowed_token_ids; pass --logprobs-mode processed_logprobs"
        )
    # The v2 readout is defined in bf16 (specification, section 7.1); the checkpoints are bf16.
    if mc.dtype != torch.bfloat16:
        problems.append(f"--dtype {mc.dtype}: v2 is served in bfloat16; pass --dtype bfloat16")
    if mc.quantization is not None:
        problems.append(f"--quantization {mc.quantization} is not supported (bf16 weights only)")
    if cc.cache_dtype not in ("auto", "bfloat16"):
        problems.append(
            f"--kv-cache-dtype {cc.cache_dtype}: only a bf16 KV cache is supported (auto or bfloat16)"
        )
    if mc.enable_prompt_embeds:
        problems.append("--enable-prompt-embeds is not supported; v2 requests are token ids")
    if cc.enable_prefix_caching and not str(cc.prefix_caching_hash_algo).startswith("sha256"):
        problems.append("--prefix-caching-hash-algo must be sha256 or sha256_cbor")
    if vllm_config.speculative_config is not None:
        problems.append("speculative decoding is not supported")
    if vllm_config.kv_transfer_config is not None:
        problems.append("KV connectors are not supported")
    if cc.kv_sharing_fast_prefill:
        problems.append("--kv-sharing-fast-prefill is not supported")
    for name, value in (
        ("pipeline_parallel_size", pc.pipeline_parallel_size),
        ("decode_context_parallel_size", pc.decode_context_parallel_size),
        ("prefill_context_parallel_size", pc.prefill_context_parallel_size),
    ):
        if value != 1:
            problems.append(f"{name} > 1 is not supported")
    if problems:
        raise ValueError(
            "strands-decider-v2 cannot serve with this configuration:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def load_decision_config(model: str, revision: str | None) -> dict:
    """``decision_config.json`` of a local checkpoint directory or a Hub repo id (through the Hugging Face cache, or a
    download when the Hub is reachable), as vLLM reads a model's other config files."""
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    dc = get_hf_file_to_dict(DECISION_CONFIG_FILE, model, revision)
    if dc is None:
        raise DecisionConfigError(
            f"not found for model {model!r} (revision {revision!r}); the model must be a Strands Decider v2 "
            "checkpoint: a local directory or a Hugging Face Hub repo id that holds decision_config.json and "
            "readout.safetensors"
        )
    return dc


class StrandsDeciderV2ForCausalLM(Gemma4ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        import vllm

        check_vllm_version(vllm.__version__)
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
        mc = vllm_config.model_config
        spec = read_decision_config(
            load_decision_config(mc.model, mc.revision),
            text_softcap=self.config.final_logit_softcapping,
            vocab_size=self.config.vocab_size,
        )
        for w in spec.warnings:
            logger.warning("strands-decider-v2: %s", w)
        self.temperature = spec.temperature
        self.softcap = spec.softcap
        self.register_buffer(
            "code_ids", torch.tensor(spec.token_ids, dtype=torch.long), persistent=False
        )
        self.readout = nn.Linear(self.config.hidden_size, N_CODES, bias=False, dtype=mc.dtype)
        # vLLM's default loader reads this source after the decoder's own weights, from the same directory or Hub
        # repo, and passes its one tensor ``weight`` to load_weights as ``readout.weight``.
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=mc.model,
                revision=mc.revision,
                prefix="readout.",
                fall_back_to_pt=False,
                allow_patterns_overrides=[READOUT_FILE],
            )
        ]
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
        # The bf16 GEMM, rounded to bf16, then the soft-cap and the temperature in float32 (specification, 7.1).
        codes = self.readout(hidden_states.to(self.readout.weight.dtype)).float()
        codes = torch.tanh(codes / self.softcap) * self.softcap
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
        readout: list[torch.Tensor] = []

        def decoder_weights():
            for name, w in weights:
                if name == READOUT_WEIGHT:  # the readout's own weight source
                    readout.append(w)
                elif name == "weight":
                    # readout.safetensors read as a decoder file, as happens for a checkpoint without a
                    # model.safetensors.index.json; the readout's own source delivers it again.
                    continue
                else:
                    yield (name if name.startswith("model.") else "model." + name), w

        loaded = super().load_weights(decoder_weights())
        if len(readout) != 1:
            raise ValueError(
                f"strands-decider-v2: {READOUT_FILE} was not loaded ({len(readout)} readout tensors received); "
                "serve with vLLM's default loader (--load-format auto or safetensors), which reads the readout "
                "as a second weight source"
            )
        w = readout[0]
        if tuple(w.shape) != tuple(self.readout.weight.shape):
            raise ValueError(
                f"strands-decider-v2: readout {tuple(w.shape)} != {tuple(self.readout.weight.shape)}"
            )
        with torch.no_grad():
            # A float32 or float16 readout is rounded to bf16 here (specification, 7.1).
            self.readout.weight.copy_(
                w.to(device=self.readout.weight.device, dtype=self.readout.weight.dtype)
            )
        loaded.add(READOUT_WEIGHT)
        return loaded
