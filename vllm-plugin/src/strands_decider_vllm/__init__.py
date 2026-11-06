"""strands-decider-vllm: serve Strands Decider checkpoints on stock vLLM 0.33.

A checkpoint is a merged bf16 Gemma 4 text decoder (``Gemma4TextModel``, bare ``layers.*`` keys), a trained
readout ``readout.safetensors`` ([255, hidden]) and ``decision_config.json`` (``attention_mode: prefix_bidir``,
``pooling: last``, the 255 code token ids, the temperature). The decision specification (``docs/spec.md``) reads a
decision at the last prompt token: the readout applied to the final-norm hidden state, soft-capped, divided by the
temperature, then a softmax over the question's own codes, with the prefix_bidir attention mask described below on
every layer.

What this plugin adds to vLLM, and nothing else:

- ``StrandsDeciderForCausalLM``: vLLM's Gemma 4 text decoder whose logits are the readout's 255 values (soft-capped
  as the specification defines, divided by the temperature) at the codes' token ids and -inf elsewhere, so a
  completion's logprobs are exactly the codes' log-softmax. The client renormalizes over a question's own codes.
- ``prefix_bidir`` attention on every layer: a request sends its state boundary S as ``vllm_xargs.prefix_end``; state
  tokens [0, S) attend to each other in both directions and never to the question, and question tokens [S, L) attend
  to the whole prompt in both directions. Sliding-window layers get the same mask with no window
  (the model class builds every layer without a window). Each layer runs one non-causal FlashAttention
  call over the request's two segments (state queries over keys [0, S), question queries over keys [0, L)).
- Prefix-cache keys that stay correct under the mask: block 0 carries a digest of the state x[0:S), and every block
  that ends past S carries a digest of the question x[S:L). A different question on the same state reuses the state's
  full blocks; nothing else is shared.
- Per-request checks in the engine core: a request to a Strands Decider model that the workers could not serve as a
  decision gets HTTP 400 before it is scheduled (:mod:`strands_decider_vllm.validation`).

``register`` is the ``vllm.general_plugins`` entry point; vLLM calls it in every process. On a vLLM release other than
0.33.x it logs a warning and registers nothing, so the environment's other models are unaffected.
"""

from __future__ import annotations

import logging
import re

__version__ = "0.2.1"
ARCH = "StrandsDeciderForCausalLM"
PREFIX_END_KEY = "prefix_end"
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
            ARCH,
        )
        return
    from vllm import ModelRegistry

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "strands_decider_vllm.model:StrandsDeciderForCausalLM")
    # FLASH_ATTN with an implementation that also runs the prefix_bidir split; it is vLLM's own for any metadata
    # without a split, which only this package's model state attaches.
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        "strands_decider_vllm.attention.SplitFlashAttentionBackend",
    )
    # Gemma 4's config hook (FA4 for both head sizes on Hopper) is looked up by architecture name.
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    MODELS_CONFIG_MAP.setdefault(ARCH, MODELS_CONFIG_MAP["Gemma4ForCausalLM"])
    from .engine import install_request_validation
    from .hashing import install_block_hash_keys

    install_block_hash_keys()
    install_request_validation()
