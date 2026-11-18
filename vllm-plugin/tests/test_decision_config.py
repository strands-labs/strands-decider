"""What the engine reads from decision_config.json, for the multimodal layout and for intermediate decoder
checkpoints. No vLLM or torch needed."""

from __future__ import annotations

import math

import pytest

from strands_decider_vllm.decision_config import (
    DECODER,
    MULTIMODAL,
    DecisionConfigError,
    read_decision_config,
)

VOCAB = 262144
IDS = list(range(236000, 236255))


def dc(**overrides):
    """An intermediate checkpoint's decision_config.json (no interface, inputs or soft-cap), plus overrides; a value
    of ... drops the field."""
    values = dict(
        format_version=1,
        token_ids=list(IDS),
        temperature=1.0,
        attention_mode="prefix_bidir",
        pooling="last",
    )
    values.update(overrides)
    return {k: v for k, v in values.items() if v is not ...}


def released(**overrides):
    """A multimodal checkpoint's decision_config.json, plus overrides."""
    fields = {
        "interface": "strands-decider",
        "inputs": ["text", "image"],
        "final_logit_softcapping": 30.0,
    }
    return dc(**{**fields, **overrides})


def read(config, text_softcap=30.0, vocab_size=VOCAB, layout=DECODER):
    return read_decision_config(
        config, text_softcap=text_softcap, vocab_size=vocab_size, layout=layout
    )


def test_an_intermediate_checkpoint_loads_as_text_only():
    spec = read(dc())
    assert spec.token_ids == tuple(IDS) and spec.temperature == 1.0 and spec.softcap == 30.0
    assert spec.inputs == ("text",) and not spec.takes_images


def test_a_released_checkpoint_loads():
    spec = read(released(), text_softcap=None, layout=MULTIMODAL)
    assert spec.softcap == 30.0 and spec.inputs == ("text", "image") and spec.takes_images


@pytest.mark.parametrize("inputs", [["image"], [], ["text", "audio"], "text", ["text", 1], None])
def test_bad_inputs_are_refused(inputs):
    with pytest.raises(DecisionConfigError, match="inputs must be"):
        read(released(inputs=inputs), layout=MULTIMODAL)


@pytest.mark.parametrize(
    "missing, match",
    [
        ("interface", '"interface": "strands-decider" is required'),
        ("inputs", "inputs must be"),
        ("final_logit_softcapping", "final_logit_softcapping is required"),
    ],
)
def test_the_multimodal_layout_requires_its_fields(missing, match):
    with pytest.raises(DecisionConfigError, match=match):
        read(released(**{missing: ...}), layout=MULTIMODAL)


def test_a_decoder_checkpoint_cannot_take_images():
    with pytest.raises(DecisionConfigError, match="no vision encoder"):
        read(dc(inputs=["text", "image"]))


@pytest.mark.parametrize("interface", ["pplx-decider", "", None, 2])
def test_another_interface_is_refused(interface):
    with pytest.raises(DecisionConfigError, match="interface"):
        read(dc(interface=interface))


@pytest.mark.parametrize(
    "key, value",
    [
        ("format_version", 2),
        ("format_version", True),
        ("attention_mode", "causal"),
        ("pooling", "mean"),
    ],
)
def test_other_formats_are_refused(key, value):
    with pytest.raises(DecisionConfigError, match=key):
        read(dc(**{key: value}))


@pytest.mark.parametrize(
    "ids", [IDS[:254], IDS[:254] + [IDS[0]], IDS[:254] + [VOCAB], IDS[:254] + [True], None]
)
def test_token_ids_must_be_255_distinct_vocabulary_ids(ids):
    with pytest.raises(DecisionConfigError, match="token_ids"):
        read(dc(token_ids=ids))


@pytest.mark.parametrize("t", [0, -1.0, math.inf, math.nan, True, "1.0", None])
def test_a_bad_temperature_is_refused(t):
    with pytest.raises(DecisionConfigError, match="temperature"):
        read(dc(temperature=t))


def test_the_soft_cap_is_decision_configs_or_the_decoder_configs_and_never_two_different_ones():
    assert read(dc(final_logit_softcapping=25.0), text_softcap=None).softcap == 25.0
    assert read(dc(), text_softcap=30.0).softcap == 30.0
    with pytest.raises(DecisionConfigError, match="final_logit_softcapping is required"):
        read(dc(), text_softcap=None)
    with pytest.raises(DecisionConfigError, match="differs"):
        read(dc(final_logit_softcapping=25.0), text_softcap=30.0)


@pytest.mark.parametrize("cap", [0, -30.0, math.inf, math.nan, True, "30"])
def test_a_bad_soft_cap_is_refused(cap):
    with pytest.raises(DecisionConfigError, match="soft"):
        read(dc(final_logit_softcapping=cap), text_softcap=None)
    with pytest.raises(DecisionConfigError, match="soft-cap"):
        read(dc(), text_softcap=cap)


def test_not_an_object_is_refused():
    with pytest.raises(DecisionConfigError, match="JSON object"):
        read([1, 2])
