"""The engine core's refusals, and the wrapper that runs them before vLLM builds a request."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strands_decider_vllm import SERVER_CONFIG_KEY, hashing
from strands_decider_vllm.engine import RequestRefused, check_request, image_spans

PROMPT = list(range(100, 120))  # L = 20


def sampling(**over):
    """A decision request's sampling parameters, as vLLM's SamplingParams holds them."""
    values = dict(
        extra_args={"prefix_end": 9},
        max_tokens=1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_bias=None,
        bad_words=None,
        structured_outputs=None,
        allowed_token_ids=[7, 8],
        logprobs=2,
    )
    return SimpleNamespace(**{**values, **over})


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"extra_args": {"prefix_end": 1}},
        {"extra_args": {"prefix_end": 19}},
        {"top_k": -1},
        {"top_k": 255},
        {"allowed_token_ids": None, "logprobs": 255},  # every code, renormalized by the client
    ],
)
def test_a_decision_passes(over):
    sp = sampling(**over)
    assert check_request(PROMPT, sp) == sp.extra_args["prefix_end"]


BOUNDARY = "vllm_xargs.prefix_end"


@pytest.mark.parametrize(
    "over, parameter",
    [
        ({"extra_args": None}, BOUNDARY),
        ({"extra_args": {"other": 1}}, BOUNDARY),
        # S = 0 leaves no state and S = L no question; JSON false arrives as 0.
        *[({"extra_args": {"prefix_end": v}}, BOUNDARY) for v in (0, 20, -1, "9", 9.0, True)],
        ({"max_tokens": 16}, "max_tokens"),  # what a client that omits it gets
        ({"temperature": 0.7}, "temperature"),
        ({"top_p": 0.9}, "top_p"),
        ({"min_p": 0.1}, "min_p"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"frequency_penalty": -0.5}, "frequency_penalty"),
        ({"top_k": 64}, "top_k"),  # Gemma's generation default
        ({"logit_bias": {7: 5.0}}, "logit_bias"),
        ({"bad_words": ["A"]}, "bad_words"),
        ({"structured_outputs": object()}, "structured_outputs"),
        ({"allowed_token_ids": None, "logprobs": 2}, "allowed_token_ids"),
    ],
)
def test_anything_else_is_refused(over, parameter):
    with pytest.raises(RequestRefused) as e:
        check_request(PROMPT, sampling(**over))
    assert e.value.parameter == parameter and str(e.value).startswith("strands-decider: ")


def test_images_come_only_through_the_plugins_server():
    # vLLM's own endpoints let a client choose an image's id, which is its cache key.
    image = SimpleNamespace(identifier="ab", mm_position=SimpleNamespace(offset=4, length=6))
    assert image_spans([image], from_server=True) == (("ab", 4, 6),)
    assert image_spans(None, from_server=False) == ()
    with pytest.raises(RequestRefused, match="strands-decider-vllm serve"):
        image_spans([image], from_server=False)


# --- on vLLM's own EngineCore.preprocess_add_request ----------------------------------------------


@pytest.fixture
def preprocess():
    core = pytest.importorskip("vllm.v1.engine.core")
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.engine import EngineCoreRequest

    import strands_decider_vllm

    strands_decider_vllm.register()
    init_none_hash(sha256)

    def run(architecture, extra_args, server=False, **sp):
        engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(architectures=[architecture]),
                additional_config={SERVER_CONFIG_KEY: "x"} if server else {},
            ),
            mm_receiver_cache=None,
            request_block_hasher=get_request_block_hasher(16, sha256),
        )
        params = SamplingParams(
            **{"max_tokens": 1, "logprobs": 2, "allowed_token_ids": [7, 8], **sp},
            extra_args=extra_args,
        )
        request = EngineCoreRequest("r", list(range(40)), None, params, None, 0.0, None, None, None)
        return core.EngineCore.preprocess_add_request(engine, request)[0]

    return run


def test_a_decider_engine_refuses_before_vllm_builds_the_request(preprocess):
    from vllm.exceptions import VLLMValidationError

    for extra_args, sp, parameter in [
        (None, {}, BOUNDARY),
        ({"prefix_end": 9}, {"max_tokens": 16}, "max_tokens"),
    ]:
        with pytest.raises(VLLMValidationError) as e:
            preprocess("StrandsDeciderForConditionalGeneration", extra_args, **sp)
        assert e.value.parameter == parameter


def test_a_decision_gets_its_digests_and_another_model_keeps_vllms_keys(preprocess):
    decision = preprocess("StrandsDeciderForCausalLM", {"prefix_end": 24})
    assert getattr(decision.sampling_params, hashing.DIGESTS_ATTR) == hashing.digests(
        list(range(40)), 24
    )
    # Another model's engine passes even a request the plugin would refuse, and adds no keys.
    plain = preprocess("Gemma4ForCausalLM", {"prefix_end": 24}, max_tokens=16)
    assert not hasattr(plain.sampling_params, hashing.DIGESTS_ATTR)
    assert decision.block_hashes[0] != plain.block_hashes[0]
