"""strands-decider-vllm: Strands Decider checkpoints on stock vLLM 0.33 (README).

``register`` is the ``vllm.general_plugins`` entry point, which vLLM calls in every process.
"""

from __future__ import annotations

import logging
import re

__version__ = "0.3.0"
PREFIX_END_KEY = "prefix_end"
#: The plugin's server marks its engines with this key in vLLM's ``additional_config``.
SERVER_CONFIG_KEY = "strands_decider_server"
#: A checkpoint's base architecture (config.json) -> the plugin's architecture for that layout, and
#: the vLLM class it subclasses.
ARCHITECTURES = {
    "Gemma4TextModel": ("StrandsDeciderForCausalLM", "Gemma4ForCausalLM"),
    "Gemma4ForConditionalGeneration": (
        "StrandsDeciderForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
    ),
    "Gemma4UnifiedForConditionalGeneration": (
        "StrandsDeciderUnifiedForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    ),
}
PLUGIN_ARCHITECTURES = frozenset(arch for arch, _ in ARCHITECTURES.values())

#: A child of vLLM's logger, so the plugin's lines get vLLM's handler in every process.
logger = logging.getLogger("vllm.strands_decider")


def register() -> None:
    import vllm

    # The plugin uses vLLM 0.33 internals that any minor release may change.
    if not re.fullmatch(r"0\.33\.\d+(\+[\w.]+)?", vllm.__version__):
        # vLLM doesn't catch a plugin's exception: raising would stop every model it serves.
        logger.warning(
            "strands-decider-vllm %s needs vLLM 0.33.x, not %s, and registered nothing",
            __version__,
            vllm.__version__,
        )
        return
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from . import engine, hashing, scheduling

    if PLUGIN_ARCHITECTURES <= set(ModelRegistry.get_supported_archs()):
        return  # registered already: the wrappers below must not wrap twice
    for arch, parent in ARCHITECTURES.values():
        ModelRegistry.register_model(arch, f"strands_decider_vllm.model:{arch}")
        # Gemma 4's config hook, found by architecture name, picks FA4 for its two head sizes.
        MODELS_CONFIG_MAP[arch] = MODELS_CONFIG_MAP[parent]
    # vLLM's own backends, which run the prefix_bidir split when a step's metadata carries one.
    register_backend(AttentionBackendEnum.FLASH_ATTN, "strands_decider_vllm.attention.FlashBackend")
    register_backend(AttentionBackendEnum.CPU_ATTN, "strands_decider_vllm.attention.CPUBackend")
    engine.install()
    hashing.install()
    scheduling.install()
