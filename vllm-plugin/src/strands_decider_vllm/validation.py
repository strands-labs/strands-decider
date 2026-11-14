"""Per-request checks for Strands Decider, run in the engine core before a request is scheduled.

A decision request is one decision: a prompt of L token ids, its state boundary S in ``vllm_xargs.prefix_end`` and one
sampled token whose processed logprobs are the codes' log-softmax. :func:`check_request` refuses every request that the
workers could not serve as that decision, or that would silently change its answer. The engine core turns a refusal into
vLLM's ``VLLMValidationError``, which the OpenAI server returns as HTTP 400 for that request alone
(:mod:`strands_decider_vllm.engine`). Nothing here imports vLLM or torch.
"""

from __future__ import annotations

import math
from typing import Any

from . import PREFIX_END_KEY

#: The number of answer codes of every checkpoint: the readout's rows.
N_CODES = 255
#: At most this many images in one request, as v1's image engine and the core library allow.
MAX_IMAGES = 4


class RequestRefused(ValueError):
    """A request the plugin will not serve. ``parameter`` and ``value`` name what was refused."""

    def __init__(self, message: str, *, parameter: str | None = None, value: Any = None) -> None:
        super().__init__(f"strands-decider: {message}")
        self.parameter = parameter
        self.value = value


def _refuse(message: str, parameter: str, value: Any = None) -> RequestRefused:
    return RequestRefused(message, parameter=parameter, value=value)


def check_prefix_end(extra_args: dict[str, Any] | None, prompt_len: int) -> int:
    """S from ``vllm_xargs``, or :class:`RequestRefused` unless it is an integer with 1 <= S < L.

    S = 0 would leave no state and S = L no question (the answer is read at the question's last token, and the
    plugin's cache keys and attention split both need a non-empty question), so both are refused. vLLM's OpenAI
    request model coerces JSON ``true`` and ``false`` to 1 and 0 before the engine sees them, so over HTTP ``false``
    is refused as S = 0 and ``true`` cannot be told apart from S = 1. A Python ``bool`` (offline ``LLM`` API) is refused.
    """
    parameter = f"vllm_xargs.{PREFIX_END_KEY}"
    if not extra_args or PREFIX_END_KEY not in extra_args:
        raise _refuse(
            f"{parameter} is required: every request to a Strands Decider model must send its state boundary S "
            "(the number of tokens in the prompt's state part)",
            parameter,
        )
    s = extra_args[PREFIX_END_KEY]
    if isinstance(s, bool) or not isinstance(s, int):
        shown = "NaN" if isinstance(s, float) and math.isnan(s) else repr(s)
        raise _refuse(
            f"{parameter} must be an integer, got {type(s).__name__} {shown}", parameter, shown
        )
    if not 1 <= s < prompt_len:
        raise _refuse(
            f"{parameter} must satisfy 1 <= S < L, where L = {prompt_len} is the prompt's length in tokens; got {s}",
            parameter,
            s,
        )
    return s


def check_sampling(sp: Any) -> None:
    """Refuse sampling parameters under which the returned logprobs are not the decision's.

    The model's logits already are the readout divided by the checkpoint temperature, so the request's temperature
    must be 1.0; truncation (top-k, top-p, min-p), penalties, logit biases, bad words and structured outputs would drop
    or move codes. One decision is one sampled token, so ``max_tokens`` must be 1; this also means a request is never
    resumed with output tokens after a preemption.
    """
    if sp.max_tokens != 1:
        raise _refuse(
            f"max_tokens must be 1 (one decision per request), got {sp.max_tokens}",
            "max_tokens",
            sp.max_tokens,
        )
    if sp.temperature != 1.0:
        raise _refuse(
            f"temperature must be 1.0, got {sp.temperature}: the model's logits are already divided by the "
            "checkpoint temperature, and vLLM would divide them again",
            "temperature",
            sp.temperature,
        )
    if 0 < sp.top_k < N_CODES:
        raise _refuse(
            f"top_k {sp.top_k} can drop answer codes; send -1 or 0 (off)", "top_k", sp.top_k
        )
    if sp.top_p != 1.0:
        raise _refuse(f"top_p {sp.top_p} can drop answer codes; send 1.0", "top_p", sp.top_p)
    if sp.min_p != 0.0:
        raise _refuse(f"min_p {sp.min_p} can drop answer codes; send 0.0", "min_p", sp.min_p)
    for name, neutral in (
        ("repetition_penalty", 1.0),
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
    ):
        value = getattr(sp, name)
        if value != neutral:
            raise _refuse(f"{name} {value} changes the codes' logits; send {neutral}", name, value)
    if sp.logit_bias:
        raise _refuse("logit_bias changes the codes' logits", "logit_bias")
    if sp.bad_words:
        raise _refuse("bad_words can remove answer codes", "bad_words")
    if sp.structured_outputs is not None:
        raise _refuse("structured outputs can remove answer codes", "structured_outputs")
    # Without allowed_token_ids the top-n logprobs rank all 255 codes, so a question's own codes can be missing from
    # them unless every code is returned.
    if not sp.allowed_token_ids and sp.logprobs != -1 and (sp.logprobs or 0) < N_CODES:
        raise _refuse(
            f"without allowed_token_ids (the question's code token ids), logprobs must be {N_CODES} so that "
            f"every code is returned; got logprobs {sp.logprobs}",
            "allowed_token_ids",
        )


#: (identifier, offset, length) of one image in the prompt: vLLM's hash of the image (its encoder-cache key) and the
#: placeholder range, ``<|image>`` through ``<image|>``, that vLLM expanded it to.
ImageSpan = tuple[str, int, int]


def check_images(
    features: Any,
    s: int,
    *,
    takes_images: bool,
    accepts_images: bool,
) -> tuple[ImageSpan, ...]:
    """The request's images as :data:`ImageSpan`, or :class:`RequestRefused`.

    ``features`` are the request's multimodal items as vLLM's engine core receives them (each with ``modality``,
    ``identifier`` and ``mm_position``), after vLLM has expanded each image's placeholder. Refused: images for a
    checkpoint whose ``inputs`` lack "image" (``takes_images``); images on an engine that the plugin's server did not
    start (``accepts_images``), because vLLM's own endpoints let a client name an image by an id of its choosing, which
    vLLM then uses as the image's cache key; any modality other than an image; more than :data:`MAX_IMAGES` images;
    and an image whose tokens do not lie inside the state ``[1, S)``. Image tokens are state tokens: the mask makes
    them bidirectional with the rest of the state, and every question sees them.
    """
    if not features:
        return ()
    if not takes_images:
        raise _refuse(
            'this checkpoint does not take images: its decision_config.json "inputs" lacks "image"',
            "images",
        )
    if not accepts_images:
        raise _refuse(
            "image requests are served only by `strands-decider-vllm serve`, which keys the cache on each image's "
            "content; vLLM's own endpoints let a client name an image by an id of its choosing",
            "images",
        )
    other = sorted({str(f.modality) for f in features if f.modality != "image"})
    if other:
        raise _refuse(f"only image input is supported, not {', '.join(other)}", "multi_modal_data")
    if len(features) > MAX_IMAGES:
        raise _refuse(
            f"{len(features)} images; a request takes at most {MAX_IMAGES}", "images", len(features)
        )
    spans = []
    for i, f in enumerate(features):
        offset, length = int(f.mm_position.offset), int(f.mm_position.length)
        if not (1 <= offset and offset + length <= s):
            raise _refuse(
                f"image {i}'s tokens [{offset}, {offset + length}) do not lie inside the state [1, S = {s}): "
                "images go first in the state",
                "images",
            )
        spans.append((str(f.identifier), offset, length))
    return tuple(spans)


def check_request(
    *,
    prompt_token_ids: list[int] | None,
    has_prompt_embeds: bool,
    sampling_params: Any,
    resumable: bool,
) -> int:
    """The request's boundary S, or :class:`RequestRefused`. The checks run in this order: the request's kind
    (token ids, not embeddings or a streaming-input session), its boundary, then its sampling parameters. The
    images are checked after these (:func:`check_images`)."""
    if sampling_params is None:
        raise _refuse(
            "only completion requests are served (no sampling parameters)", "sampling_params"
        )
    if has_prompt_embeds or prompt_token_ids is None:
        raise _refuse(
            "prompt embeddings are not supported; send the prompt as token ids", "prompt_embeds"
        )
    if resumable:
        raise _refuse(
            "streaming-input (resumable) requests are not supported: a later chunk would extend the prompt "
            "after the state boundary and the cache keys were fixed",
            "resumable",
        )
    s = check_prefix_end(sampling_params.extra_args, len(prompt_token_ids))
    check_sampling(sampling_params)
    return s
