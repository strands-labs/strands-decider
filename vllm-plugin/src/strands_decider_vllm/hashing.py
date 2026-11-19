"""Prefix-cache keys that stay correct under prefix_bidir.

vLLM keys a cached KV block by a hash chain over the tokens up to its end, plus per-block extra
keys. Under the mask a state token's keys and values depend on the whole state, images included,
and a question token's on the whole prompt, so every block hash of a prefix_bidir request gets more:

- block 0: a digest of the state's token ids and of each image's content hash and placeholder span
  (two images of one size have the same placeholder ids, and vLLM's own image key covers only the
  blocks that hold the image, not the state tokens before it, which attend to it);
- every block that ends past S: a digest of the state's digest and the question's token ids.

A further question on a cached state reuses the state's full blocks, and nothing else is shared.
The engine core attaches the digests to the requests it has checked (``engine``); every other
request keeps vLLM's keys.
"""

from __future__ import annotations

import hashlib
import struct
from array import array
from collections.abc import Sequence

#: The attribute of a checked request's SamplingParams that holds its digests, where no client can
#: set it (a key in extra_args would come from vllm_xargs).
DIGESTS_ATTR = "_strands_decider_digests"
#: An image: vLLM's hash of its content, and its placeholders' offset and length.
Image = tuple[str, int, int]


def digests(
    prompt: Sequence[int], s: int, images: Sequence[Image] = ()
) -> tuple[int, bytes, bytes]:
    """(S, the state's digest, the question's digest)."""
    state = hashlib.sha256(b"strands-decider/state\0")
    state.update(array("Q", prompt[:s]).tobytes())
    for identifier, offset, length in images:
        key = identifier.encode()
        state.update(
            b"strands-decider/image\0" + struct.pack("<QQQ", len(key), offset, length) + key
        )
    question = hashlib.sha256(b"strands-decider/question\0" + state.digest())
    question.update(array("Q", prompt[s:]).tobytes())
    return s, state.digest(), question.digest()


def install() -> None:
    """Wraps ``kv_cache_utils.generate_block_hash_extra_keys``, which vLLM's block hasher looks up
    in its module at call time."""
    import vllm.v1.core.kv_cache_utils as kcu

    original = kcu.generate_block_hash_extra_keys

    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        keys, next_mm_idx = original(request, start_token_idx, end_token_idx, start_mm_idx)
        d = getattr(request.sampling_params, DIGESTS_ATTR, None)
        if d is None:
            return keys, next_mm_idx
        s, state, question = d
        extra = (("decider_state", state),) if start_token_idx == 0 else ()
        if end_token_idx > s:
            extra += (("decider_question", question),)
        return (tuple(keys or ()) + extra if extra else keys), next_mm_idx

    kcu.generate_block_hash_extra_keys = generate_block_hash_extra_keys
