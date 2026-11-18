"""strands-decider-vllm: serve Strands Decider checkpoints on stock vLLM 0.33.

The plugin adds, for its own architectures only: the trained readout in place of the LM head
(:mod:`strands_decider_vllm.model`), the prefix_bidir mask on every layer with no sliding window
(:mod:`strands_decider_vllm.model_state`), prefix-cache keys that stay correct under the mask
(:mod:`strands_decider_vllm.hashing`), per-request checks in the engine core (:mod:`strands_decider_vllm.engine`),
scheduling that keeps a prompt with images whole (:mod:`strands_decider_vllm.scheduling`), and a System One server
(:mod:`strands_decider_vllm.server`).

``register`` is the ``vllm.general_plugins`` entry point; vLLM calls it in every process. On a vLLM release other than
0.33.x it logs a warning and registers nothing, so the environment's other models are unaffected.
"""

from __future__ import annotations

import logging
import re

__version__ = "0.3.0"
#: The bare-decoder layout (intermediate checkpoints, text only).
ARCH = "StrandsDeciderForCausalLM"
#: The multimodal layout over Gemma4ForConditionalGeneration (E2B, E4B, 26B-A4B, 31B).
ARCH_MULTIMODAL = "StrandsDeciderForConditionalGeneration"
#: The multimodal layout over Gemma4UnifiedForConditionalGeneration (12B).
ARCH_UNIFIED = "StrandsDeciderUnifiedForConditionalGeneration"
#: Each architecture, its class, and the vLLM architecture whose config hook (FA4 for both head sizes) it takes.
ARCHITECTURES = {
    ARCH: ("strands_decider_vllm.model:StrandsDeciderForCausalLM", "Gemma4ForCausalLM"),
    ARCH_MULTIMODAL: (
        "strands_decider_vllm.model:StrandsDeciderForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
    ),
    ARCH_UNIFIED: (
        "strands_decider_vllm.model:StrandsDeciderUnifiedForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    ),
}
ARCHS = tuple(ARCHITECTURES)
#: The base model's architecture (config.json) -> the plugin's architecture that serves that layout.
ARCH_FOR_BASE = {
    "Gemma4TextModel": ARCH,
    "Gemma4ForConditionalGeneration": ARCH_MULTIMODAL,
    "Gemma4UnifiedForConditionalGeneration": ARCH_UNIFIED,
}
PREFIX_END_KEY = "prefix_end"
#: The key under which the plugin's server marks its engines in vLLM's ``additional_config``.
SERVER_CONFIG_KEY = "strands_decider_server"
SUPPORTED_VLLM = ">=0.33.0,<0.34"
#: vLLM 0.33.x only: the plugin uses vLLM internals that are not a public API and may change in any minor release.
_SUPPORTED_MINORS = ((0, 33),)

#: A child of vLLM's logger, so the plugin's lines get vLLM's handler and format in every process.
logger = logging.getLogger("vllm.strands_decider")


def vllm_version_supported(version: str) -> bool:
    """True for vLLM 0.33.<patch> releases, optionally with a local suffix such as ``+cu130``. Pre-, post- and dev
    releases and every other minor release are not supported."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(\+[\w.]+)?", version.strip())
    return m is not None and (int(m[1]), int(m[2])) in _SUPPORTED_MINORS


def unsupported_vllm_message(version: str) -> str:
    return (
        f"strands-decider-vllm {__version__} requires vLLM 0.33.x ({SUPPORTED_VLLM}); the installed vLLM is "
        f"{version!r}. Other releases, 0.32.x included, are not supported, because the plugin relies on vLLM "
        "0.33 internals. Install a supported vLLM with: pip install 'vllm>=0.33.0,<0.34'"
    )


def check_vllm_version(version: str) -> None:
    """Raise ``RuntimeError`` unless ``version`` is a vLLM 0.33.x release (see :func:`vllm_version_supported`)."""
    if not vllm_version_supported(version):
        raise RuntimeError(unsupported_vllm_message(version))


def register() -> None:
    import vllm

    version = vllm.__version__
    if not vllm_version_supported(version):
        # vLLM does not catch a plugin's exception, so raising here would stop every model in the environment.
        logger.warning(
            "%s The plugin registered nothing: other models are unaffected, and %s cannot be served.",
            unsupported_vllm_message(version),
            ", ".join(ARCHS),
        )
        return
    from vllm import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for arch, (cls, _) in ARCHITECTURES.items():
        if arch not in supported:
            ModelRegistry.register_model(arch, cls)
    # FLASH_ATTN with an implementation that also runs the prefix_bidir split; it is vLLM's own for any metadata
    # without a split, which only this package's model state attaches.
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "strands_decider_vllm.attention.SplitFlashAttentionBackend",
    )
    # The same split on vLLM's CPU attention, for CPU builds of vLLM (the tests' CPU tier).
    register_backend(
        AttentionBackendEnum.CPU_ATTN,
        "strands_decider_vllm.attention_cpu.SplitCPUAttentionBackend",
    )
    # Gemma 4's config hooks (FA4 for both head sizes on Hopper) are looked up by architecture name.
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    for arch, (_, base) in ARCHITECTURES.items():
        MODELS_CONFIG_MAP.setdefault(arch, MODELS_CONFIG_MAP[base])
    from .engine import install_request_validation
    from .hashing import install_block_hash_keys
    from .scheduling import install_whole_prompt_scheduling

    install_block_hash_keys()
    install_request_validation()
    install_whole_prompt_scheduling()
