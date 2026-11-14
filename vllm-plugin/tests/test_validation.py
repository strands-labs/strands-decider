"""Per-request checks (engine core, before scheduling): every request a worker could not serve as a decision, or
whose answer its parameters would change, is refused, and so is every image the checkpoint or the entry point cannot
take or that lies outside the state. No vLLM or torch needed."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from strands_decider_vllm.validation import (
    MAX_IMAGES,
    N_CODES,
    RequestRefused,
    check_images,
    check_prefix_end,
    check_request,
    check_sampling,
)

PROMPT = list(range(100, 120))  # L = 20


def sampling(**overrides):
    """The README request's sampling parameters, as vLLM's SamplingParams holds them."""
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
        allowed_token_ids=[236776, 236799],
        logprobs=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def check(sp=None, *, prompt=PROMPT, embeds=False, resumable=False):
    return check_request(
        prompt_token_ids=prompt,
        has_prompt_embeds=embeds,
        sampling_params=sampling() if sp is None else sp,
        resumable=resumable,
    )


def test_the_readme_request_passes():
    assert check() == 9


@pytest.mark.parametrize("s", [1, 2, 19])
def test_every_boundary_with_a_state_and_a_question_passes(s):
    assert check(sampling(extra_args={"prefix_end": s})) == s


@pytest.mark.parametrize("top_k", [0, -1, N_CODES, 1000])
def test_top_k_that_cannot_drop_a_code_passes(top_k):
    check_sampling(sampling(top_k=top_k))


@pytest.mark.parametrize("logprobs", [N_CODES, -1])
def test_every_code_without_allowed_token_ids_passes(logprobs):
    # The other request form docs/architecture.md describes: all 255 codes' logprobs, renormalized over the question's.
    check_sampling(sampling(allowed_token_ids=None, logprobs=logprobs))


# --- the boundary ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extra_args", [None, {}, {"other": 1}])
def test_a_missing_boundary_is_refused(extra_args):
    with pytest.raises(RequestRefused, match="prefix_end is required") as err:
        check_prefix_end(extra_args, 20)
    assert err.value.parameter == "vllm_xargs.prefix_end"


@pytest.mark.parametrize(
    "value, shown",
    [
        (True, "True"),
        (False, "False"),  # a Python bool (offline API); over HTTP vLLM has already made it 1 or 0
        ("9", "'9'"),
        (9.0, "9.0"),
        (9.5, "9.5"),
        (math.nan, "NaN"),
        ([9], "[9]"),
        (None, "None"),
    ],
)
def test_a_boundary_that_is_not_an_integer_is_refused(value, shown):
    with pytest.raises(RequestRefused, match="must be an integer") as err:
        check_prefix_end({"prefix_end": value}, 20)
    assert err.value.value == shown and shown in str(err.value)


@pytest.mark.parametrize("s", [0, -1, 20, 21, 10**9])
def test_a_boundary_outside_1_to_L_is_refused(s):
    # S = 0 leaves no state; S = L leaves no question (and S = L would be served as a fully bidirectional prompt read
    # at a state token, with no cache digests).
    with pytest.raises(RequestRefused, match=r"1 <= S < L, where L = 20") as err:
        check_prefix_end({"prefix_end": s}, 20)
    assert err.value.value == s


def test_json_false_arrives_as_zero_and_is_refused():
    # vLLM's vllm_xargs field (str | int | float | list) turns JSON false into 0 before the engine sees it.
    with pytest.raises(RequestRefused, match="got 0"):
        check(sampling(extra_args={"prefix_end": 0}))


# --- the request's kind ---------------------------------------------------------------------------------------------


def test_prompt_embeds_are_refused_even_with_a_valid_boundary():
    with pytest.raises(RequestRefused, match="prompt embeddings") as err:
        check(embeds=True)
    assert err.value.parameter == "prompt_embeds"


def test_a_request_without_token_ids_is_refused():
    with pytest.raises(RequestRefused, match="prompt embeddings"):
        check(prompt=None)


def test_streaming_input_is_refused():
    with pytest.raises(RequestRefused, match="streaming-input") as err:
        check(resumable=True)
    assert err.value.parameter == "resumable"


def test_a_request_without_sampling_parameters_is_refused():
    with pytest.raises(RequestRefused, match="sampling parameters"):
        check_request(
            prompt_token_ids=PROMPT, has_prompt_embeds=False, sampling_params=None, resumable=False
        )


# --- sampling parameters --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, parameter",
    [
        ({"max_tokens": 16}, "max_tokens"),  # the completions default when max_tokens is omitted
        ({"max_tokens": 2}, "max_tokens"),
        ({"max_tokens": None}, "max_tokens"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": 0.7}, "temperature"),
        ({"temperature": 2.0}, "temperature"),
        ({"top_k": 1}, "top_k"),
        ({"top_k": 64}, "top_k"),  # Gemma's generation default
        ({"top_k": N_CODES - 1}, "top_k"),
        ({"top_p": 0.95}, "top_p"),
        ({"min_p": 0.05}, "min_p"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"frequency_penalty": -0.5}, "frequency_penalty"),
        ({"logit_bias": {236776: 5.0}}, "logit_bias"),
        ({"bad_words": ["A"]}, "bad_words"),
        ({"structured_outputs": object()}, "structured_outputs"),
        ({"allowed_token_ids": None, "logprobs": 2}, "allowed_token_ids"),
        ({"allowed_token_ids": [], "logprobs": 254}, "allowed_token_ids"),
        ({"allowed_token_ids": None, "logprobs": None}, "allowed_token_ids"),
    ],
)
def test_sampling_that_changes_the_answer_is_refused(overrides, parameter):
    with pytest.raises(RequestRefused) as err:
        check(sampling(**overrides))
    assert err.value.parameter == parameter
    assert str(err.value).startswith("strands-decider: ")


def test_the_boundary_is_checked_before_sampling():
    with pytest.raises(RequestRefused) as err:
        check(sampling(extra_args={"prefix_end": "9"}, max_tokens=16))
    assert err.value.parameter == "vllm_xargs.prefix_end"


# --- images ---------------------------------------------------------------------------------------------------------


def feature(offset, length=258, identifier="a" * 64, modality="image"):
    """A multimodal item as vLLM's engine core receives it, after the image's placeholder was expanded."""
    return SimpleNamespace(
        modality=modality,
        identifier=identifier,
        mm_position=SimpleNamespace(offset=offset, length=length),
    )


def images(features, s=600, takes=True, accepts=True):
    return check_images(features, s, takes_images=takes, accepts_images=accepts)


def test_a_text_request_has_no_images():
    assert images(None) == () and images([]) == ()
    # Text needs neither the checkpoint's image input nor the server.
    assert images([], takes=False, accepts=False) == ()


def test_images_inside_the_state_pass_as_identifier_offset_length():
    fs = [feature(4, identifier="x"), feature(263, identifier="y", length=250)]
    assert images(fs, s=520) == (("x", 4, 258), ("y", 263, 250))


def test_an_image_ending_at_s_is_inside_the_state():
    assert images([feature(10, 258)], s=268) == (("a" * 64, 10, 258),)


@pytest.mark.parametrize(
    "offset, length, s",
    [
        (10, 258, 267),  # ends one token past S
        (300, 258, 200),  # starts in the question
        (0, 258, 600),  # before <bos>: never where the template puts an image
    ],
)
def test_an_image_outside_the_state_is_refused(offset, length, s):
    with pytest.raises(RequestRefused, match=r"do not lie inside the state") as err:
        images([feature(offset, length)], s=s)
    assert err.value.parameter == "images"


def test_images_for_a_text_only_checkpoint_are_refused():
    with pytest.raises(RequestRefused, match='"inputs" lacks "image"') as err:
        images([feature(4)], takes=False)
    assert err.value.parameter == "images"


def test_images_outside_the_plugins_server_are_refused():
    with pytest.raises(RequestRefused, match="strands-decider-vllm serve"):
        images([feature(4)], accepts=False)


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_other_media_are_refused(modality):
    with pytest.raises(RequestRefused, match=f"not {modality}"):
        images([feature(4), feature(300, modality=modality)], s=900)


def test_more_than_four_images_are_refused():
    fs = [feature(4 + 260 * i) for i in range(MAX_IMAGES + 1)]
    with pytest.raises(RequestRefused, match="5 images; a request takes at most 4") as err:
        images(fs, s=2000)
    assert err.value.value == 5
    assert len(images(fs[:MAX_IMAGES], s=2000)) == MAX_IMAGES
