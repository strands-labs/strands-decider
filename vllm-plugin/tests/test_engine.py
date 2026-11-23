"""What a Strands Decider engine refuses, and which cached blocks a decision may reuse, on vLLM's own
EngineCore.preprocess_add_request and block hasher."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils
from vllm.v1.engine import EngineCoreRequest, PrefixHashConfig
from vllm.v1.engine.core import EngineCore

from strands_decider_vllm import SERVER_CONFIG_KEY, register

BLOCK = 16
PROMPT = list(range(100, 140))  # L = 40
BOUNDARY = "vllm_xargs.prefix_end"
_ids = itertools.count()


@pytest.fixture(scope="module")
def preprocess():
    """The block hashes of a request after preprocess_add_request on an engine of ``architecture``."""
    register()
    kv_cache_utils.init_none_hash(sha256)
    hasher = kv_cache_utils.get_request_block_hasher(BLOCK, sha256)
    frontend = PrefixHashConfig("sha256", BLOCK, kv_cache_utils.NONE_HASH)

    def run(
        prompt=PROMPT,
        s=24,
        *,
        architecture="StrandsDeciderForCausalLM",
        from_server=False,
        images=(),
        cache_salt=None,
        block_hashes=None,
        **sampling,
    ):
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=[architecture]),
            additional_config={SERVER_CONFIG_KEY: True} if from_server else {},
            watermark_config=None,
        )
        engine = SimpleNamespace(
            vllm_config=vllm_config,
            mm_receiver_cache=None,
            prefix_hash_config=frontend,
            request_block_hasher=hasher,
        )
        sampling = {"max_tokens": 1, "logprobs": 2, "allowed_token_ids": [7, 8], **sampling}
        request = EngineCoreRequest(
            request_id=f"r{next(_ids)}",
            prompt_token_ids=list(prompt),
            mm_features=[
                MultiModalFeatureSpec(
                    data=None,
                    modality="image",
                    identifier=image,
                    mm_position=PlaceholderRange(offset=offset, length=length),
                    mm_hash=image,
                )
                for image, offset, length in images
            ]
            or None,
            sampling_params=SamplingParams(extra_args={"prefix_end": s}, **sampling),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=cache_salt,
            data_parallel_rank=None,
            block_hashes=block_hashes,
        )
        return EngineCore.preprocess_add_request(engine, request)[0].block_hashes

    return run


@pytest.mark.parametrize(
    "s, sampling",
    [
        (24, {}),
        (1, {}),
        (39, {}),
        (24, {"top_k": -1}),
        (24, {"top_k": 255}),
        (
            24,
            {"allowed_token_ids": None, "logprobs": 255},
        ),  # every code, renormalized by the client
    ],
)
def test_a_decision_passes(preprocess, s, sampling):
    assert preprocess(s=s, **sampling)


@pytest.mark.parametrize(
    "s, sampling, parameter",
    [
        # S = 0 leaves no state and S = L no question; JSON false arrives as 0.
        *[(s, {}, BOUNDARY) for s in (None, 0, 40, -1, "9", 9.0, True)],
        (24, {"max_tokens": 16}, "max_tokens"),  # what a client that omits it gets
        (24, {"temperature": 0.7}, "temperature"),
        (24, {"top_p": 0.9}, "top_p"),
        (24, {"min_p": 0.1}, "min_p"),
        (24, {"repetition_penalty": 1.1}, "repetition_penalty"),
        (24, {"presence_penalty": 0.5}, "presence_penalty"),
        (24, {"frequency_penalty": -0.5}, "frequency_penalty"),
        (24, {"top_k": 64}, "top_k"),  # Gemma's generation default
        (24, {"logit_bias": {7: 5.0}}, "logit_bias"),
        (24, {"bad_words": ["A"]}, "bad_words"),
        (24, {"structured_outputs": StructuredOutputsParams(regex="A")}, "structured_outputs"),
        (24, {"allowed_token_ids": None, "logprobs": 2}, "allowed_token_ids"),
    ],
)
def test_anything_else_is_refused(preprocess, s, sampling, parameter):
    with pytest.raises(VLLMValidationError, match=r"^strands-decider: ") as e:
        preprocess(s=s, **sampling)
    assert e.value.parameter == parameter


def test_images_come_only_through_the_plugins_server(preprocess):
    # vLLM's own endpoints let a client choose an image's id, which is its cache key.
    image = [("ab", 4, 6)]
    assert preprocess(images=image, from_server=True)
    with pytest.raises(VLLMValidationError, match="strands-decider-vllm serve"):
        preprocess(images=image)


def test_another_models_engine_is_left_alone(preprocess):
    plain = preprocess(architecture="Gemma4ForCausalLM", max_tokens=16)
    assert plain[0] != preprocess()[0], "a decision's blocks get the mask's keys"


def shared(a, b):
    """How many leading blocks two requests share."""
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


def test_a_further_question_reuses_the_states_full_blocks_and_nothing_else(preprocess):
    # 16-token blocks: [0, 16) and [16, 32) lie in the state (S = 40); later ones reach the question.
    state, qa, qb = list(range(1000, 1040)), list(range(2000, 2030)), list(range(3000, 3030))
    a = preprocess(state + qa, 40)
    assert preprocess(state + qa, 40) == a, "an exact repeat reuses every block"
    assert shared(a, preprocess(state + qb, 40)) == 2
    assert shared(a, preprocess([*state[:39], 7, *qa], 40)) == 0, "every state token sees the last"
    assert shared(a, preprocess(state + qa, 39)) == 0, "another S"
    plain = preprocess(state + qa, 40, architecture="Gemma4ForCausalLM")
    assert shared(a, plain) == 0, "a request without the mask"
    assert shared(a, preprocess(state + qa, 40, cache_salt="x")) == 0, "vLLM's salt still applies"


def test_an_image_of_the_same_size_reuses_nothing(preprocess):
    # A 300-token image at [40, 340) of a 400-token state: two images of one size have the same token
    # ids, and vLLM's own key covers only the blocks that hold the image.
    image = [255999, *[258880] * 298, 258882]
    state = list(range(1000, 1040)) + image + list(range(2000, 2060))
    qa, qb = list(range(3000, 3060)), list(range(4000, 4060))
    red, blue = ("e3b0c442" * 8, 40, 300), ("2c26b46b" * 8, 40, 300)
    a = preprocess(state + qa, 400, images=[red], from_server=True)
    assert shared(a, preprocess(state + qa, 400, images=[blue], from_server=True)) == 0
    assert shared(a, preprocess(state + qb, 400, images=[red], from_server=True)) == 400 // BLOCK
    assert shared(a, preprocess(state + qa, 400)) == 0, "the same tokens without the image"


def test_a_frontends_block_hashes_give_way_to_the_engines(preprocess):
    # vLLM 0.34's Rust frontend may send prompt block hashes, which would lack the mask's keys.
    frontend = [b"frontend"] * (len(PROMPT) // BLOCK)
    assert preprocess(block_hashes=frontend) == preprocess()
