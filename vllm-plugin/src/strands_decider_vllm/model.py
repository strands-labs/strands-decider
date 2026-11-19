"""The plugin's architectures: vLLM's Gemma 4 classes with the trained readout in place of the LM
head, and every attention layer under the prefix_bidir mask (``model_state``), with no window.

- ``StrandsDeciderForConditionalGeneration`` (E2B, E4B, 26B-A4B, 31B) and
  ``StrandsDeciderUnifiedForConditionalGeneration`` (12B): the base model's multimodal layout, with
  its vision encoder and image processor (``processing``), the fine-tuned language model, no LM head.
- ``StrandsDeciderForCausalLM``: a bare ``Gemma4TextModel`` decoder, text only (intermediate
  checkpoints).

``compute_logits`` gives the core's readout at the codes' token ids, -inf at every other id: a bf16
GEMM of the final-norm hidden state, then in float32 ``c * tanh(z / c)`` divided by the checkpoint's
temperature. A request's processed logprobs over its ``allowed_token_ids`` are then the
log-softmax over its codes.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration

from . import logger
from .attention import CPUImpl, FlashImpl
from .model_state import PrefixBidirModelState
from .processing import register_multimodal, register_unified

READOUT_FILE = "readout.safetensors"
#: The readout's tensor ``weight``, as its own weight source hands it to ``load_weights``.
READOUT_WEIGHT = "readout.weight"


def check_config(vllm_config: VllmConfig) -> None:
    """Refuses the engine settings under which the mask or the answers would be served wrongly."""
    from vllm.platforms import current_platform

    mc, sc, cc = vllm_config.model_config, vllm_config.scheduler_config, vllm_config.cache_config
    pc = vllm_config.parallel_config
    # bf16, as the readout is defined and parity was measured; float32 for the CPU tier's tests.
    dtypes = (torch.bfloat16, torch.float32) if current_platform.is_cpu() else (torch.bfloat16,)
    checks = [
        (
            mc.is_mm_prefix_lm,
            "vLLM's multimodal PrefixLM mode is on (the 12B, 26B-A4B and 31B bases set "
            'use_bidirectional_attention "vision"), which prefix_bidir replaces: add '
            '"use_bidirectional_attention": null to --hf-overrides, under "text_config" for the '
            "multimodal layout",
        ),
        # vLLM 0.34 removed Model Runner V1, and this attribute with it.
        (
            not getattr(vllm_config, "use_v2_model_runner", True),
            "prefix_bidir is a Model Runner V2 model state: set VLLM_USE_V2_MODEL_RUNNER=1",
        ),
        # A prompt must be computed whole in one step: its tokens attend to keys a later step computes.
        (sc.enable_chunked_prefill, "chunked prefill is on: pass --no-enable-chunked-prefill"),
        (
            (sc.max_num_scheduled_tokens or mc.max_model_len) < mc.max_model_len,
            f"--max-num-scheduled-tokens {sc.max_num_scheduled_tokens} < --max-model-len {mc.max_model_len}: "
            "a longer prompt would never be scheduled",
        ),
        (
            0 < sc.long_prefill_token_threshold < mc.max_model_len,
            f"--long-prefill-token-threshold {sc.long_prefill_token_threshold} cuts longer prompts into chunks",
        ),
        (
            pc.data_parallel_size > 1 and (pc.enable_dbo or pc.ubatch_size > 1),
            "micro-batching under data parallelism (--enable-dbo, --ubatch-size) can split a prompt",
        ),
        (
            vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL,
            "cudagraph_mode FULL replays prefill steps' captured attention, without the split",
        ),
        (
            mc.logprobs_mode not in PROCESSED_LOGPROBS_MODES,
            f"--logprobs-mode {mc.logprobs_mode}: the codes' probabilities are the processed logprobs",
        ),
        (mc.dtype not in dtypes, f"--dtype {mc.dtype}: only bfloat16 is served"),
        (
            mc.quantization is not None,
            f"--quantization {mc.quantization}: only bf16 weights are served",
        ),
        (
            cc.cache_dtype not in ("auto", "bfloat16"),
            f"--kv-cache-dtype {cc.cache_dtype}: only a bf16 KV cache is served",
        ),
        (mc.enable_prompt_embeds, "--enable-prompt-embeds: decision requests are token ids"),
        (
            cc.enable_prefix_caching and not str(cc.prefix_caching_hash_algo).startswith("sha256"),
            "--prefix-caching-hash-algo must be sha256 or sha256_cbor",
        ),
        (vllm_config.speculative_config is not None, "speculative decoding is not supported"),
        (vllm_config.kv_transfer_config is not None, "KV connectors are not supported"),
        (
            vllm_config.ec_transfer_config is not None,
            "encoder-cache connectors are not supported: an image's encodings must come from the image",
        ),
        (cc.kv_sharing_fast_prefill, "--kv-sharing-fast-prefill is not supported"),
        (pc.pipeline_parallel_size > 1, "pipeline parallelism is not supported"),
        (
            pc.decode_context_parallel_size > 1 or pc.prefill_context_parallel_size > 1,
            "context parallelism is not supported",
        ),
    ]
    problems = [message for wrong, message in checks if wrong]
    if problems:
        raise ValueError(
            "strands-decider cannot serve with this configuration:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


@contextmanager
def _sliding_windows_cut_nothing(max_model_len: int):
    """Builds Gemma 4's sliding-window layers with a window of max_model_len tokens, which cuts
    nothing, since the mask applies on them too. Their KV-cache groups then keep every token, one
    per layer shape as for stock Gemma 4. (Built without a window, both head sizes share one
    group, which vLLM 0.33 can't combine with E2B and E4B's KV-shared layers; vLLM #64095.)"""
    import vllm.model_executor.models.gemma4 as gemma4

    attention = gemma4.Attention

    def attention_with_full_window(*args, **kwargs):
        if kwargs.get("per_layer_sliding_window") is not None:
            kwargs["per_layer_sliding_window"] = max_model_len
        return attention(*args, **kwargs)

    gemma4.Attention = attention_with_full_window
    try:
        yield
    finally:
        gemma4.Attention = attention


class _Decider:
    """What the three architectures share: the readout, compute_logits and the model state."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        check_config(vllm_config)
        mc = vllm_config.model_config
        with _sliding_windows_cut_nothing(mc.max_model_len):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        for module in self.modules():
            impl = getattr(module, "impl", None)
            if getattr(module, "sliding_window", None) is not None and isinstance(
                impl, (FlashImpl, CPUImpl)
            ):
                # The kernel's own window: a non-causal call with one would still cut keys.
                impl.sliding_window = impl.no_window
        # decision_config.json is read as vLLM reads a model's other config files: from the local
        # directory, the Hugging Face cache or the Hub.
        dc = get_hf_file_to_dict("decision_config.json", mc.model, mc.revision)
        if dc is None:
            raise ValueError(f"strands-decider: {mc.model!r} has no decision_config.json")
        text = mc.hf_text_config
        # Intermediate checkpoints leave the soft-cap to their config.json.
        self.softcap = dc.get("final_logit_softcapping") or text.final_logit_softcapping
        self.temperature = dc["temperature"]
        self.vocab_size = text.vocab_size
        self.register_buffer("code_ids", torch.tensor(dc["token_ids"]), persistent=False)
        # A bf16 GEMM whatever the decoder runs in (a float32 model on the CPU tier too), as the
        # core's readout is defined.
        self.readout = nn.Linear(
            text.hidden_size, len(dc["token_ids"]), bias=False, dtype=torch.bfloat16
        )
        # vLLM's default loader reads this source after the model's weights, from the same place.
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
            "strands-decider: readout over %d codes, soft-cap %s, temperature %s",
            len(dc["token_ids"]),
            self.softcap,
            self.temperature,
        )

    @classmethod
    def get_model_state_cls(cls) -> type[PrefixBidirModelState]:
        return PrefixBidirModelState

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        z = self.readout(hidden_states.to(torch.bfloat16)).float()
        z = torch.tanh(z / self.softcap) * self.softcap / self.temperature
        logits = torch.full(
            (hidden_states.shape[0], self.vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        logits[:, self.code_ids] = z
        return logits

    @staticmethod
    def checkpoint_name(name: str) -> str:
        """A checkpoint weight's name for the parent class's loader."""
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        readout = []

        def model_weights():
            for name, w in weights:
                if name == READOUT_WEIGHT:
                    readout.append(w)
                # Without an index file, readout.safetensors is read as a model file too.
                elif name != "weight":
                    yield self.checkpoint_name(name), w

        loaded = super().load_weights(model_weights())
        if len(readout) != 1:
            raise ValueError(
                f"strands-decider: {READOUT_FILE} was not loaded; serve with vLLM's default loader "
                "(--load-format auto or safetensors), which reads it as a second weight source"
            )
        with torch.no_grad():
            self.readout.weight.copy_(readout[0])
        return loaded | {READOUT_WEIGHT}


class StrandsDeciderForCausalLM(_Decider, Gemma4ForCausalLM):
    @staticmethod
    def checkpoint_name(name: str) -> str:
        # A bare decoder's keys (embed_tokens.*, layers.*, norm.*) under vLLM's model.*.
        return name if name.startswith("model.") else "model." + name


@register_multimodal
class StrandsDeciderForConditionalGeneration(_Decider, Gemma4ForConditionalGeneration):
    pass


@register_unified
class StrandsDeciderUnifiedForConditionalGeneration(
    _Decider, Gemma4UnifiedForConditionalGeneration
):
    pass
