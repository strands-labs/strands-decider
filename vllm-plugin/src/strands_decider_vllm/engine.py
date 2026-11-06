"""Engine-core request checks: a request that a Strands Decider model's workers could not serve as a decision is refused
before it is scheduled.

vLLM's engine core turns a ``VLLMValidationError`` raised in ``EngineCore.preprocess_add_request`` into an error output
for that request alone (HTTP 400 from the OpenAI server) and keeps serving, whereas an exception a worker raises during
a step kills the engine and fails every request in flight. The plugin therefore wraps ``preprocess_add_request``: for an
engine that serves ``StrandsDeciderForCausalLM`` it runs :func:`strands_decider_vllm.validation.check_request` on
every request, with prefix caching on or off, and registers the validated request's cache digests
(:mod:`strands_decider_vllm.hashing`). Engines serving any other model call vLLM's function unchanged.
"""

from __future__ import annotations

from typing import Any

from . import ARCH, hashing
from .validation import RequestRefused, check_request

_MARK = "_strands_decider"
_SERVES_DECIDER = "_strands_decider_serves"


def serves_decider(engine_core: Any) -> bool:
    """Whether this engine core serves a Strands Decider model (its architectures, after ``--hf-overrides``, name
    the plugin's)."""
    cached = engine_core.__dict__.get(_SERVES_DECIDER)
    if cached is None:
        cached = ARCH in engine_core.vllm_config.model_config.architectures
        engine_core.__dict__[_SERVES_DECIDER] = cached
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
        except RequestRefused as e:
            raise validation_error(str(e), parameter=e.parameter, value=e.value) from None
        if self.request_block_hasher is None:  # prefix caching off: no block hashes
            return original(self, request)
        hashing.expect(request.request_id, request.prompt_token_ids, s)
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
