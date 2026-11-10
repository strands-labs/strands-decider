"""Prefix-cache keys that stay correct under prefix_bidir.

vLLM keys a cached KV block by a hash chain over the request's tokens up to the end of that block, plus per-block extra
keys. Under prefix_bidir a state token's keys and values depend on the whole state x[0:S), images included, and a
question token's on the whole prompt x[0:L). This module adds two extra keys to every block hash of a prefix_bidir
request:

- block 0: ``("decider_state", d_state)``, where d_state is a SHA-256 of the state's token ids x[0:S) and of each of
  its images: vLLM's hash of the image's content (its multimodal identifier) and the image's placeholder span. Every
  block of the chain is therefore keyed on the whole state, pixels included;
- every block that ends past S: ``("decider_question", d_question)``, a SHA-256 of d_state and the question's token ids
  x[S:L), so a block holding question tokens is keyed on the whole question and, through d_state, on every image too.

An image's placeholder ids are the same for any two images of the same size, so the token ids alone cannot tell two
such images apart; vLLM's own multimodal extra key covers only the blocks that hold the image's tokens, not the state
tokens before it, which attend to it under the mask. With d_state, a request reuses a cached block only if it has the
same state (same S, same tokens, the same images at the same places) and, for blocks that reach into the question, the
same question: exactly when the block's keys and values are the same. A further question on a cached state reuses the
state up to its last full block; requests without the digests never share blocks with prefix_bidir requests, whose
block 0 differs.

The digests are computed in the engine core, from the request's own token ids, the boundary S and the images that
:mod:`strands_decider_vllm.engine` has validated, just before vLLM builds the request and hashes its blocks
(:func:`expect`). Only requests to a Strands Decider model get them: the wrapper adds nothing to any other request, so
other models served from the same environment keep vLLM's own keys.
"""

from __future__ import annotations

import hashlib
import struct
from array import array
from typing import Any

_ATTR = "_strands_decider_digests"
#: request id -> (S, state digest, question digest), from validation until vLLM has built the request.
_PENDING: dict[str, tuple[int, bytes, bytes]] = {}


STATE_DOMAIN = b"strands-decider/state\0"
QUESTION_DOMAIN = b"strands-decider/question\0"
IMAGE_DOMAIN = b"strands-decider/image\0"


def digests(
    prompt: list[int], s: int, images: tuple[tuple[str, int, int], ...] = ()
) -> tuple[int, bytes, bytes]:
    """(S, d_state, d_question). d_state hashes the state's token ids x[0:S), then each image of the state in order:
    its identifier (vLLM's hash of the image) and its placeholder span (offset, length). A text request's d_state is
    the hash of its tokens alone. d_question hashes d_state and the question's token ids x[S:L)."""
    h = hashlib.sha256(STATE_DOMAIN)
    h.update(array("Q", prompt[:s]).tobytes())
    for identifier, offset, length in images:
        key = identifier.encode()
        h.update(IMAGE_DOMAIN + struct.pack("<QQQ", len(key), offset, length) + key)
    state = h.digest()
    q = hashlib.sha256(QUESTION_DOMAIN + state)
    q.update(array("Q", prompt[s:]).tobytes())
    return s, state, q.digest()


def expect(
    request_id: str, prompt: list[int], s: int, images: tuple[tuple[str, int, int], ...] = ()
) -> None:
    """Register a validated request's digests before vLLM builds its ``Request`` (which hashes its blocks)."""
    _PENDING[request_id] = digests(prompt, s, images)


def settle(request: Any) -> None:
    """After vLLM built the request: drop the pending entry, and attach the digests if no block was hashed yet (a
    prompt shorter than one block), so that later block hashes see them too."""
    d = _PENDING.pop(request.request_id, None)
    if getattr(request, _ATTR, None) is None:
        setattr(request, _ATTR, d or ())


def forget(request_id: str) -> None:
    _PENDING.pop(request_id, None)


def _digests(request: Any) -> tuple[int, bytes, bytes] | None:
    cached = getattr(request, _ATTR, None)
    if cached is None:
        cached = _PENDING.get(request.request_id) or ()
        setattr(request, _ATTR, cached)
    return cached or None


def install_block_hash_keys() -> None:
    """Wrap ``vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys`` (idempotent). The request block hasher looks
    the function up in its module at call time, so the wrapper applies to every hash the engine core computes."""
    import vllm.v1.core.kv_cache_utils as kcu

    if getattr(kcu.generate_block_hash_extra_keys, "_strands_decider", False):
        return
    original = kcu.generate_block_hash_extra_keys

    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        keys, next_mm_idx = original(request, start_token_idx, end_token_idx, start_mm_idx)
        d = _digests(request)
        if d is None:
            return keys, next_mm_idx
        s, state, question = d
        extra = []
        if start_token_idx == 0:
            extra.append(("decider_state", state))
        if end_token_idx > s:
            extra.append(("decider_question", question))
        if not extra:
            return keys, next_mm_idx
        return (tuple(keys) if keys else ()) + tuple(extra), next_mm_idx

    generate_block_hash_extra_keys._strands_decider = True
    kcu.generate_block_hash_extra_keys = generate_block_hash_extra_keys
