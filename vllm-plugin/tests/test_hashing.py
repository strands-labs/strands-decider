"""Which cached blocks a request may reuse under the mask, on vLLM's own block hasher."""

from __future__ import annotations

import itertools

import pytest

from strands_decider_vllm import hashing

_ids = itertools.count()


def test_the_state_digest_covers_each_image_and_its_place():
    tokens = [2, 255999, *[258880] * 6, 258882, 107, 9, 9]
    red, blue = ("red", 1, 8), ("blue", 1, 8)
    states = {hashing.digests(tokens, 10, images)[1] for images in [(), (red,), (blue,)]}
    assert len(states) == 3, "the same tokens, another image or none: another state"
    assert (
        hashing.digests(tokens, 10, (("red", 2, 8),))[1] != hashing.digests(tokens, 10, (red,))[1]
    )
    assert hashing.digests(tokens, 10, (red, blue)) != hashing.digests(tokens, 10, (blue, red))


@pytest.fixture
def build():
    """vLLM's Request, built as the engine core builds it for a checked request (``s``) or another."""
    pytest.importorskip("vllm.v1.core.kv_cache_utils")
    from vllm import SamplingParams
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    import strands_decider_vllm

    strands_decider_vllm.register()
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)

    def make(tokens, s=None, images=(), cache_salt=None):
        sp = SamplingParams(max_tokens=1)
        if s is not None:
            setattr(sp, hashing.DIGESTS_ATTR, hashing.digests(tokens, s, images))
        features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=i,
                mm_position=PlaceholderRange(offset=o, length=n),
                mm_hash=i,
            )
            for i, o, n in images
        ]
        return Request(
            request_id=f"r{next(_ids)}",
            prompt_token_ids=list(tokens),
            sampling_params=sp,
            pooling_params=None,
            cache_salt=cache_salt,
            block_hasher=hasher,
            mm_features=features or None,
        ).block_hashes

    return make


def shared(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


def test_a_further_question_reuses_the_states_full_blocks_and_nothing_else(build):
    # 16-token blocks: [0, 16) and [16, 32) lie in the state (S = 40); later ones reach the question.
    state, qa, qb = list(range(1000, 1040)), list(range(2000, 2030)), list(range(3000, 3030))
    a = build(state + qa, s=40)
    assert build(state + qa, s=40) == a, "an exact repeat reuses every block"
    assert shared(a, build(state + qb, s=40)) == 2
    assert shared(a, build([*state[:39], 7, *qa], s=40)) == 0, "every state token sees the last"
    assert shared(a, build(state + qa, s=39)) == 0, "another S"
    assert shared(a, build(state + qa)) == 0, "a request without the mask"
    assert shared(a, build(state + qa, s=40, cache_salt="x")) == 0, "vLLM's salt still applies"


def test_an_image_of_the_same_size_reuses_nothing(build):
    # A 300-token image at [40, 340) inside a 400-token state: two images of one size have the
    # same token ids, and vLLM's own key covers only the blocks that hold the image.
    image = [255999, *[258880] * 298, 258882]
    state = list(range(1000, 1040)) + image + list(range(2000, 2060))
    qa, qb = list(range(3000, 3060)), list(range(4000, 4060))
    red, blue = ("e3b0c442" * 8, 40, 300), ("2c26b46b" * 8, 40, 300)
    a = build(state + qa, s=400, images=[red])
    assert shared(a, build(state + qa, s=400, images=[blue])) == 0, "block 0 included"
    assert build(state + qa, s=400, images=[red]) == a
    assert shared(a, build(state + qb, s=400, images=[red])) == 400 // 16
    assert shared(a, build(state + qa, s=400)) == 0, "the same tokens without the image"
