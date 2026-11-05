"""Prefix-cache keys that stay correct under prefix_bidir.

vLLM keys a cached KV block by a hash chain over the request's tokens up to the end of that block, plus per-block extra
keys. Under prefix_bidir a state token's keys and values depend on the whole state x[0:S), and a question token's on the
whole prompt x[0:L). This module adds two extra keys to every block hash of a request that sends ``prefix_end``:

- block 0: ``("sdv2_state", sha256(x[0:S)))``, so every block of the chain is keyed on the whole state;
- every block that ends past S: ``("sdv2_question", sha256(x[S:L)))``, so a block holding question tokens is keyed on
  the whole question as well.

A request therefore reuses a cached block only if it has the same state (same S, same tokens) and, for blocks that reach
into the question, the same question, which is exactly when the block's keys and values are the same. A further
question on a cached state reuses the state up to its last full block; requests without ``prefix_end`` (causal) never
share blocks with prefix_bidir requests, whose block 0 differs.

The digests are computed in the engine core from the request's own token ids, so a client can choose only S, and any S
gives keys consistent with the attention the workers apply for that S.
"""

from __future__ import annotations

import hashlib
from array import array
from typing import Any

from . import PREFIX_END_KEY

_ATTR = "_strands_decider_v2_digests"


def _digest(token_ids: Any, a: int, b: int, domain: bytes) -> bytes:
    h = hashlib.sha256(domain)
    h.update(array("Q", token_ids[a:b]).tobytes())
    return h.digest()


def _boundary(request: Any) -> int | None:
    sp = getattr(request, "sampling_params", None)
    args = getattr(sp, "extra_args", None) if sp is not None else None
    if not args or PREFIX_END_KEY not in args:
        return None
    s = args[PREFIX_END_KEY]
    if isinstance(s, float) and s.is_integer():
        s = int(s)
    if not isinstance(s, int) or isinstance(s, bool):
        return None  # the worker rejects it
    return s


def _digests(request: Any) -> tuple[int, bytes, bytes] | None:
    cached = getattr(request, _ATTR, None)
    if cached is not None:
        return cached or None
    s = _boundary(request)
    prompt = request.prompt_token_ids
    if s is None or prompt is None or not 0 < s < len(prompt):
        setattr(request, _ATTR, ())
        return None
    out = (
        s,
        _digest(prompt, 0, s, b"strands-decider-v2/state\0"),
        _digest(prompt, s, len(prompt), b"strands-decider-v2/question\0"),
    )
    setattr(request, _ATTR, out)
    return out


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
