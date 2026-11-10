"""Engine-core request checks: a request that a Strands Decider model's workers could not serve as a decision is refused
before it is scheduled.

vLLM's engine core turns a ``VLLMValidationError`` raised in ``EngineCore.preprocess_add_request`` into an error output
for that request alone (HTTP 400 from the OpenAI server and from the plugin's server) and keeps serving, whereas an
exception a worker raises during a step kills the engine and fails every request in flight. The plugin therefore wraps
``preprocess_add_request``: for an engine that serves one of the plugin's architectures it runs
:func:`strands_decider_vllm.validation.check_request` and :func:`~strands_decider_vllm.validation.check_images` on
every request, with prefix caching on or off, and registers the validated request's cache digests
(:mod:`strands_decider_vllm.hashing`). Engines serving any other model call vLLM's function unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import ARCHS, SERVER_CONFIG_KEY, hashing
from .validation import RequestRefused, check_images, check_request

_MARK = "_strands_decider"
_SERVES_DECIDER = "_strands_decider_serves"
_POLICY = "_strands_decider_policy"


def serves_decider(engine: Any) -> bool:
    """Whether this engine core (or scheduler: anything with ``vllm_config``) serves a Strands Decider model (its
    architectures, after ``--hf-overrides``, name one of the plugin's)."""
    cached = engine.__dict__.get(_SERVES_DECIDER)
    if cached is None:
        cached = any(a in ARCHS for a in engine.vllm_config.model_config.architectures)
        engine.__dict__[_SERVES_DECIDER] = cached
    return cached


def started_by_server(vllm_config: Any) -> bool:
    """Whether the plugin's own server (``strands-decider-vllm serve``) built this engine: it marks its engines in
    vLLM's ``additional_config``."""
    extra = getattr(vllm_config, "additional_config", None) or {}
    return bool(isinstance(extra, dict) and extra.get(SERVER_CONFIG_KEY))


@dataclass(frozen=True)
class Policy:
    """What an engine accepts beyond text: images when the checkpoint takes them and the plugin's server started it."""

    takes_images: bool
    accepts_images: bool


def policy(engine_core: Any) -> Policy:
    cached = engine_core.__dict__.get(_POLICY)
    if cached is None:
        vc = engine_core.vllm_config
        takes = False
        if vc.model_config.is_multimodal_model:
            # The multimodal layout only; a decoder checkpoint never receives images (vLLM refuses them for a
            # text-only architecture before the engine core). The workers validated the file at startup.
            from .model import checkpoint_spec

            takes = checkpoint_spec(vc.model_config).takes_images
        cached = Policy(takes_images=takes, accepts_images=started_by_server(vc))
        engine_core.__dict__[_POLICY] = cached
    return cached


def wrap_preprocess_add_request(original: Any, validation_error: type[Exception]) -> Any:
    def preprocess_add_request(self, request):
        if not serves_decider(self):
            return original(self, request)
        try:
            s = check_request(
                prompt_token_ids=request.prompt_token_ids,
                has_prompt_embeds=request.prompt_embeds is not None
                or request.prompt_is_token_ids is not None,
                sampling_params=request.sampling_params,
                resumable=request.resumable,
            )
            features = getattr(request, "mm_features", None)
            images = ()
            if features:
                p = policy(self)
                images = check_images(
                    features, s, takes_images=p.takes_images, accepts_images=p.accepts_images
                )
        except RequestRefused as e:
            raise validation_error(str(e), parameter=e.parameter, value=e.value) from None
        if self.request_block_hasher is None:  # prefix caching off: no block hashes
            return original(self, request)
        hashing.expect(request.request_id, request.prompt_token_ids, s, images)
        try:
            req, wave = original(self, request)
        except BaseException:
            hashing.forget(request.request_id)
            raise
        hashing.settle(req)
        return req, wave

    preprocess_add_request._strands_decider = True
    preprocess_add_request.__wrapped__ = original
    return preprocess_add_request


def install_request_validation() -> None:
    """Wrap ``vllm.v1.engine.core.EngineCore.preprocess_add_request`` (idempotent)."""
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.preprocess_add_request, _MARK, False):
        return
    EngineCore.preprocess_add_request = wrap_preprocess_add_request(
        EngineCore.preprocess_add_request, VLLMValidationError
    )
