"""What a Strands Decider engine does with each request before vLLM schedules it: refuse what a
worker couldn't serve as one decision, and key its prefix-cache blocks by what the mask makes them
depend on.

vLLM's engine core answers a VLLMValidationError from ``preprocess_add_request`` with an error for
that request alone (HTTP 400 from ``vllm serve``, 422 from the plugin's server), where an exception
in a worker would stop the engine and every request in flight.

Under the mask a state token's keys and values depend on the whole state, images included, and a
question token's on the whole prompt. So block 0 of a decision gets a digest of the state's token
ids and of each image's content hash and placeholder span (two images of one size have the same
placeholder ids, and vLLM's own image key covers only the blocks that hold the image), and every
block that ends past S a digest of the state's digest and the question's ids. A further question on
a cached state reuses the state's full blocks, and nothing else is shared.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, NoReturn

import numpy as np
from vllm.exceptions import VLLMValidationError

from . import PLUGIN_ARCHITECTURES, PREFIX_END_KEY, SERVER_CONFIG_KEY

_N_CODES = 255  # every checkpoint's answer codes, the readout's rows
#: Sampling parameters that would move or drop codes, and the values that leave them alone.
_NEUTRAL = {
    "temperature": 1.0,  # the model's logits are already divided by the checkpoint's temperature
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
#: Where a checked request's SamplingParams hold its cache keys: no client can set an attribute.
_KEYS = "_strands_decider_cache_keys"


def serves_decider(vllm_config: Any) -> bool:
    return any(a in PLUGIN_ARCHITECTURES for a in vllm_config.model_config.architectures)


def install() -> None:
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine.core import EngineCore

    preprocess = EngineCore.preprocess_add_request
    extra_keys = kv_cache_utils.generate_block_hash_extra_keys

    def preprocess_add_request(self, request):
        if serves_decider(self.vllm_config):
            # The server's mark is trust at the operator's level: an operator who sets it for
            # `vllm serve` lets that server's clients choose images' cache keys.
            from_server = bool((self.vllm_config.additional_config or {}).get(SERVER_CONFIG_KEY))
            s = _check(request, from_server)
            setattr(request.sampling_params, _KEYS, _cache_keys(request, s))
            request.block_hashes = None  # a frontend's hashes would lack these keys
        return preprocess(self, request)

    # vLLM's block hasher looks this function up in its module at call time.
    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        keys, next_mm_idx = extra_keys(request, start_token_idx, end_token_idx, start_mm_idx)
        decision = getattr(request.sampling_params, _KEYS, None)
        if decision is None:
            return keys, next_mm_idx
        s, state, question = decision
        extra = (("decider_state", state),) if start_token_idx == 0 else ()
        if end_token_idx > s:
            extra += (("decider_question", question),)
        return (tuple(keys or ()) + extra if extra else keys), next_mm_idx

    EngineCore.preprocess_add_request = preprocess_add_request
    kv_cache_utils.generate_block_hash_extra_keys = generate_block_hash_extra_keys


def _refuse(message: str, parameter: str) -> NoReturn:
    raise VLLMValidationError(f"strands-decider: {message}", parameter=parameter)


def _check(request: Any, from_server: bool) -> int:
    """The request's boundary S, after refusing a request a worker couldn't serve as one decision.
    vLLM's endpoints let a client choose an image's id, its cache key, so images come only through
    the plugin's server."""
    sp, length = request.sampling_params, len(request.prompt_token_ids)
    s = (sp.extra_args or {}).get(PREFIX_END_KEY)
    # S = 0 leaves no state and S = L no question. vLLM's request model reads JSON true and false
    # as 1 and 0; a bool can only come from the offline API.
    if type(s) is not int or not 1 <= s < length:
        _refuse(
            f"vllm_xargs.{PREFIX_END_KEY} must be S, the number of tokens in the prompt's state, "
            f"an integer with 1 <= S < L = {length}; got {s!r}",
            f"vllm_xargs.{PREFIX_END_KEY}",
        )
    if sp.max_tokens != 1:
        _refuse(f"max_tokens must be 1, one decision, not {sp.max_tokens}", "max_tokens")
    for name, neutral in _NEUTRAL.items():
        if getattr(sp, name) != neutral:
            _refuse(f"{name} must be {neutral}, not {getattr(sp, name)}", name)
    if 0 < sp.top_k < _N_CODES:
        _refuse(f"top_k {sp.top_k} can drop answer codes; send -1 or 0", "top_k")
    for name in ("logit_bias", "bad_words", "structured_outputs"):
        if getattr(sp, name):
            _refuse(f"{name} would change or drop the answer codes", name)
    # Without allowed_token_ids the top logprobs rank all 255 codes: a question's own codes are all
    # there only if every code is.
    if not sp.allowed_token_ids and sp.logprobs != -1 and (sp.logprobs or 0) < _N_CODES:
        _refuse(
            f"send the question's code token ids as allowed_token_ids, or logprobs {_N_CODES}",
            "allowed_token_ids",
        )
    if request.mm_features and not from_server:
        _refuse(
            "image requests are served only by `strands-decider-vllm serve`", "multi_modal_data"
        )
    return s


def _cache_keys(request: Any, s: int) -> tuple[int, bytes, bytes]:
    """(S, the state's digest, the question's digest)."""
    prompt = request.prompt_token_ids
    state = hashlib.sha256(b"strands-decider/state\0")
    state.update(np.asarray(prompt[:s], dtype=np.uint64).tobytes())
    for f in request.mm_features or ():
        key = f.identifier.encode()
        span = struct.pack("<QQQ", len(key), f.mm_position.offset, f.mm_position.length)
        state.update(b"strands-decider/image\0" + span + key)
    question = hashlib.sha256(b"strands-decider/question\0" + state.digest())
    question.update(np.asarray(prompt[s:], dtype=np.uint64).tobytes())
    return s, state.digest(), question.digest()
