"""Engine-core request checks: what a Strands Decider engine refuses before scheduling.

vLLM's engine core answers a ``VLLMValidationError`` raised in ``preprocess_add_request`` with an
error for that request alone (HTTP 400 from vLLM's server, 422 from the plugin's), whereas an
exception in a worker stops the engine and every request in flight. So for engines that serve the
plugin's architectures, the plugin wraps ``preprocess_add_request``: it refuses every request that a
worker could not serve as one decision, or whose parameters would change the answer, and attaches
the cache digests (``hashing``) to the rest.
"""

from __future__ import annotations

from typing import Any

from . import PLUGIN_ARCHITECTURES, PREFIX_END_KEY, SERVER_CONFIG_KEY, hashing

N_CODES = 255  # every checkpoint's answer codes, the readout's rows
#: Sampling parameters that would move or drop codes, and the values that leave them alone.
NEUTRAL = {
    "temperature": 1.0,  # the model's logits are already divided by the checkpoint's temperature
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


class RequestRefused(ValueError):
    def __init__(self, message: str, parameter: str) -> None:
        super().__init__(f"strands-decider: {message}")
        self.parameter = parameter


def check_request(prompt_token_ids: list[int], sp: Any) -> int:
    """The request's boundary S, or RequestRefused."""
    s = (sp.extra_args or {}).get(PREFIX_END_KEY)
    # S = 0 leaves no state and S = L no question. vLLM's request model makes JSON true and false
    # 1 and 0; a Python bool can only come from the offline API.
    if type(s) is not int or not 1 <= s < len(prompt_token_ids):
        raise RequestRefused(
            f"vllm_xargs.{PREFIX_END_KEY} must be S, the number of tokens in the prompt's state, "
            f"an integer with 1 <= S < L = {len(prompt_token_ids)}; got {s!r}",
            f"vllm_xargs.{PREFIX_END_KEY}",
        )
    if sp.max_tokens != 1:
        raise RequestRefused(
            f"max_tokens must be 1, one decision, not {sp.max_tokens}", "max_tokens"
        )
    for name, neutral in NEUTRAL.items():
        if getattr(sp, name) != neutral:
            raise RequestRefused(f"{name} must be {neutral}, not {getattr(sp, name)}", name)
    if 0 < sp.top_k < N_CODES:
        raise RequestRefused(f"top_k {sp.top_k} can drop answer codes; send -1 or 0", "top_k")
    for name in ("logit_bias", "bad_words", "structured_outputs"):
        if getattr(sp, name):
            raise RequestRefused(f"{name} would change or drop the answer codes", name)
    # Without allowed_token_ids the top logprobs rank all 255 codes: a question's own codes are all
    # there only if every code is.
    if not sp.allowed_token_ids and sp.logprobs != -1 and (sp.logprobs or 0) < N_CODES:
        raise RequestRefused(
            f"send the question's code token ids as allowed_token_ids, or logprobs {N_CODES}",
            "allowed_token_ids",
        )
    return s


def image_spans(features: list[Any] | None, from_server: bool) -> tuple[hashing.Image, ...]:
    """Each image's content hash and placeholder span, or RequestRefused for images that didn't
    come through the plugin's server: vLLM's endpoints let a client choose an image's id, which
    vLLM uses as its cache key, so two different images could share cached state."""
    if features and not from_server:
        raise RequestRefused(
            "image requests are served only by `strands-decider-vllm serve`", "multi_modal_data"
        )
    return tuple((f.identifier, f.mm_position.offset, f.mm_position.length) for f in features or ())


def serves_decider(vllm_config: Any) -> bool:
    return any(a in PLUGIN_ARCHITECTURES for a in vllm_config.model_config.architectures)


def started_by_server(vllm_config: Any) -> bool:
    """Whether the plugin's server built this engine. Its mark is trust at the operator's level: an
    operator who sets it for ``vllm serve`` lets that server's clients choose images' cache keys."""
    return bool((vllm_config.additional_config or {}).get(SERVER_CONFIG_KEY))


def install() -> None:
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.core import EngineCore

    original = EngineCore.preprocess_add_request

    def preprocess_add_request(self, request):
        if serves_decider(self.vllm_config):
            sp = request.sampling_params
            try:
                s = check_request(request.prompt_token_ids, sp)
                images = image_spans(request.mm_features, started_by_server(self.vllm_config))
            except RequestRefused as e:
                raise VLLMValidationError(str(e), parameter=e.parameter) from None
            setattr(sp, hashing.DIGESTS_ATTR, hashing.digests(request.prompt_token_ids, s, images))
        return original(self, request)

    EngineCore.preprocess_add_request = preprocess_add_request
