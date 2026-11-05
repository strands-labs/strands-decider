"""Diagnostic only, not part of strands-decider-v2-vllm: the v2 readout on Gemma 4's own causal, sliding-window
attention, which is what stock vLLM would compute from a repackaged v2 checkpoint. It measures how much of v2's
accuracy depends on the prefix_bidir mask.

Install it next to strands-decider-v2-vllm (``pip install --no-deps ./tools/causal_control``), serve a checkpoint with
``"architectures": ["StrandsDeciderV2CausalControlForCausalLM"]`` in ``--hf-overrides`` and the plugin's other flags,
and send requests without ``vllm_xargs.prefix_end`` (a prefix_end is ignored). It is a vLLM general plugin only so
that every vLLM process registers the architecture; it changes nothing for ``StrandsDeciderV2ForCausalLM``.
"""

from __future__ import annotations

ARCH = "StrandsDeciderV2CausalControlForCausalLM"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            ARCH, "strands_decider_v2_causal_control.model:StrandsDeciderV2CausalControlForCausalLM"
        )
    # The same Gemma 4 config hook as the plugin's architecture (FlashAttention 4 for both head sizes on Hopper).
    MODELS_CONFIG_MAP.setdefault(ARCH, MODELS_CONFIG_MAP["Gemma4ForCausalLM"])
