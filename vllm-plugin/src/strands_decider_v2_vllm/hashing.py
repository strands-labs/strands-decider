"""Prefix-cache keys that stay correct under prefix_bidir.

vLLM keys a cached KV block by a hash chain over the request's tokens up to the end of that block, plus per-block extra
keys. Under prefix_bidir a state token's keys and values depend on the whole state x[0:S), and a question token's on the
whole prompt x[0:L). This module adds two extra keys to every block hash of a v2 request:

- block 0: ``("sdv2_state", sha256(x[0:S)))``, so every block of the chain is keyed on the whole state;
- every block that ends past S: ``("sdv2_question", sha256(x[S:L)))``, so a block holding question tokens is keyed on
  the whole question as well.

A request therefore reuses a cached block only if it has the same state (same S, same tokens) and, for blocks that reach
into the question, the same question, which is exactly when the block's keys and values are the same. A further
question on a cached state reuses the state up to its last full block; requests without the digests never share
blocks with v2 requests, whose block 0 differs.

The digests are computed in the engine core, from the request's own token ids and the boundary S that
:mod:`strands_decider_v2_vllm.engine` has validated, just before vLLM builds the request and hashes its blocks
(:func:`expect`). Only requests to a v2 model get them: the wrapper adds nothing to any other request, so other models
served from the same environment keep vLLM's own keys.
"""

from __future__ import annotations

import hashlib
from array import array
from typing import Any

_ATTR = "_strands_decider_v2_digests"
#: request id -> (S, state digest, question digest), from validation until vLLM has built the request.
_PENDING: dict[str, tuple[int, bytes, bytes]] = {}


def _digest(token_ids: Any, a: int, b: int, domain: bytes) -> bytes:
    h = hashlib.sha256(domain)
    h.update(array("Q", token_ids[a:b]).tobytes())
    return h.digest()


def digests(prompt: list[int], s: int) -> tuple[int, bytes, bytes]:
    """(S, sha256 of the state x[0:S), sha256 of the question x[S:L)), each in its own domain."""
    return (
        s,
        _digest(prompt, 0, s, b"strands-decider-v2/state\0"),
        _digest(prompt, s, len(prompt), b"strands-decider-v2/question\0"),
    )


def expect(request_id: str, prompt: list[int], s: int) -> None:
    """Register a validated v2 request's digests before vLLM builds its ``Request`` (which hashes its blocks)."""
    _PENDING[request_id] = digests(prompt, s)


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

    if getattr(kcu.generate_block_hash_extra_keys, "_strands_decider_v2", False):
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
            extra.append(("sdv2_state", state))
        if end_token_idx > s:
            extra.append(("sdv2_question", question))
        if not extra:
            return keys, next_mm_idx
        return (tuple(keys) if keys else ()) + tuple(extra), next_mm_idx

    generate_block_hash_extra_keys._strands_decider_v2 = True
    kcu.generate_block_hash_extra_keys = generate_block_hash_extra_keys
