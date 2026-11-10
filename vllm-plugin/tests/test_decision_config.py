"""decision_config.json at startup: the interface, the inputs, the provenance, the code token ids, the temperature and
the soft-cap, for the multimodal layout and for intermediate decoder checkpoints (specification, sections 8 and 9). No
vLLM or torch needed."""

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
    """An intermediate checkpoint's decision_config.json (no interface, no soft-cap), plus overrides."""
    values = dict(
        format_version=1,
        codes=[f"C{i}" for i in range(255)],
        token_ids=list(IDS),
        temperature=1.0,
        attention_mode="prefix_bidir",
        pooling="last",
    )
    values.update(overrides)
    return {k: v for k, v in values.items() if v is not ...}


def read(config, text_softcap=30.0, vocab_size=VOCAB, layout=DECODER):
    return read_decision_config(
        config, text_softcap=text_softcap, vocab_size=vocab_size, layout=layout
    )


PROVENANCE = {
    "base_model": "google/gemma-4-E2B-it",
    "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}


def released(**overrides):
    """A multimodal-layout checkpoint's decision_config.json (the specification's schema), plus overrides."""
    return dc(
        **{
            "interface": "strands-decider",
            "inputs": ["text", "image"],
            "final_logit_softcapping": 30.0,
            "provenance": dict(PROVENANCE),
            **overrides,
        }
    )


def test_an_intermediate_checkpoint_loads_as_text_only_with_one_warning_per_missing_field():
    spec = read(dc())
    assert spec.token_ids == tuple(IDS) and spec.temperature == 1.0 and spec.softcap == 30.0
    assert spec.inputs == ("text",) and not spec.takes_images
    assert [w.split('"')[1] for w in spec.warnings] == ["interface", "inputs", "provenance"]
    assert all("\n" not in w for w in spec.warnings)


def test_a_released_checkpoint_loads_without_warnings():
    spec = read(released(), layout=MULTIMODAL)
    assert spec.warnings == () and spec.softcap == 30.0
    assert spec.inputs == ("text", "image") and spec.takes_images
    assert (spec.base_model, spec.base_revision) == tuple(PROVENANCE.values())


@pytest.mark.parametrize("inputs", [["text"], ["image", "text"]])
def test_inputs_may_be_text_or_text_and_image_in_any_order(inputs):
    assert set(read(released(inputs=inputs), layout=MULTIMODAL).inputs) == set(inputs)


@pytest.mark.parametrize(
    "inputs",
    [["image"], [], ["text", "text"], ["text", "audio"], "text", ["text", 1], None],
)
def test_bad_inputs_are_refused(inputs):
    with pytest.raises(DecisionConfigError, match="inputs must be"):
        read(released(inputs=inputs), layout=MULTIMODAL)


@pytest.mark.parametrize(
    "missing, match",
    [
        ("interface", '"interface": "strands-decider" is required'),
        ("inputs", "inputs is required"),
        ("provenance", "provenance is required"),
        ("final_logit_softcapping", "final_logit_softcapping is required"),
    ],
)
def test_the_multimodal_layout_requires_every_field(missing, match):
    # Even with the decoder config's soft-cap at hand, the multimodal layout takes it from decision_config.json.
    with pytest.raises(DecisionConfigError, match=match):
        read(released(**{missing: ...}), layout=MULTIMODAL)


def test_a_decoder_checkpoint_cannot_take_images():
    with pytest.raises(DecisionConfigError, match="no vision encoder"):
        read(dc(inputs=["text", "image"]))
    assert read(dc(inputs=["text"])).inputs == ("text",)


@pytest.mark.parametrize(
    "provenance, match",
    [
        (None, "leave the field out instead of null"),
        ([], "must be an object"),
        ({"base_revision": PROVENANCE["base_revision"]}, "base_model must be a Hub id"),
        ({**PROVENANCE, "base_model": "gemma-4-E2B-it"}, "base_model must be a Hub id"),
        ({"base_model": PROVENANCE["base_model"]}, "base_revision must be a 40-character"),
        ({**PROVENANCE, "base_revision": "main"}, "base_revision must be a 40-character"),
        (
            {**PROVENANCE, "base_revision": "3E22461F65E89153144F8ADB70E3B8C2CC9845A7"},
            "40-character",
        ),
    ],
)
def test_bad_provenance_is_refused(provenance, match):
    with pytest.raises(DecisionConfigError, match=match):
        read(released(provenance=provenance), layout=MULTIMODAL)


def test_a_decoder_checkpoint_with_provenance_has_it_checked():
    with pytest.raises(DecisionConfigError, match="base_revision"):
        read(dc(provenance={"base_model": "google/gemma-4-E2B-it", "base_revision": "x"}))
    spec = read(dc(provenance=dict(PROVENANCE)))
    assert spec.base_model == "google/gemma-4-E2B-it"


def test_the_interface_is_checked_before_any_other_field():
    with pytest.raises(DecisionConfigError, match="interface must be"):
        read({"interface": "pplx-decider"}, layout=MULTIMODAL)


def test_an_unknown_layout_is_a_programming_error():
    with pytest.raises(ValueError, match="layout must be"):
        read(dc(), layout="text")


@pytest.mark.parametrize("interface", ["pplx-decider", "strands-decider-v1", "", None, 2])
def test_another_interface_is_refused(interface):
    with pytest.raises(DecisionConfigError, match="interface must be"):
        read(dc(interface=interface))


@pytest.mark.parametrize(
    "key, value",
    [
        ("format_version", 2),
        ("format_version", True),
        ("format_version", ...),
        ("attention_mode", "causal"),
        ("pooling", "mean"),
    ],
)
def test_other_layouts_are_refused(key, value):
    with pytest.raises(DecisionConfigError, match=key):
        read(dc(**{key: value}))


# --- token ids ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ids", [IDS[:254], IDS + [7], [], None, "A"])
def test_token_ids_must_be_255(ids):
    with pytest.raises(DecisionConfigError, match="list of 255"):
        read(dc(token_ids=ids))


@pytest.mark.parametrize("bad", [VOCAB, VOCAB + 5, -1, True, 1.0, "236776", None])
def test_token_ids_must_be_vocabulary_ids(bad):
    ids = list(IDS)
    ids[17] = bad
    with pytest.raises(DecisionConfigError, match=r"integers in \[0, 262144\)"):
        read(dc(token_ids=ids))


def test_token_ids_must_be_distinct():
    ids = list(IDS)
    ids[200] = ids[3]
    with pytest.raises(DecisionConfigError, match=f"repeated: \\[{IDS[3]}\\]"):
        read(dc(token_ids=ids))


def test_the_vocabulary_bound_comes_from_the_decoder():
    with pytest.raises(DecisionConfigError, match=r"\[0, 236100\)"):
        read(dc(), vocab_size=236100)


# --- temperature and soft-cap ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0, -1.0, math.inf, math.nan, 1e-39, 1e39, True, "1.0", None, ...])
def test_a_bad_temperature_is_refused(t):
    with pytest.raises(DecisionConfigError, match="temperature"):
        read(dc(temperature=t))


def test_the_soft_cap_comes_from_decision_config_when_present():
    assert read(dc(final_logit_softcapping=25.0), text_softcap=None).softcap == 25.0


def test_the_soft_cap_falls_back_to_the_decoder_config():
    assert read(dc(), text_softcap=30.0).softcap == 30.0


def test_a_missing_soft_cap_is_refused_never_skipped():
    with pytest.raises(DecisionConfigError, match="final_logit_softcapping is missing"):
        read(dc(), text_softcap=None)


def test_two_different_soft_caps_are_refused():
    with pytest.raises(DecisionConfigError, match="differs"):
        read(dc(final_logit_softcapping=25.0), text_softcap=30.0)
    with pytest.raises(DecisionConfigError, match="differs"):
        read(released(final_logit_softcapping=25.0), text_softcap=30.0, layout=MULTIMODAL)


def test_the_multimodal_soft_cap_is_decision_configs_even_without_the_decoder_configs():
    # In the multimodal layout config.json keeps the soft-cap under text_config; decision_config.json governs.
    assert read(released(), text_softcap=None, layout=MULTIMODAL).softcap == 30.0


@pytest.mark.parametrize("cap", [0, 0.0, -30.0, math.inf, math.nan, True, "30"])
def test_a_bad_soft_cap_is_refused(cap):
    with pytest.raises(DecisionConfigError, match="final_logit_softcapping"):
        read(dc(final_logit_softcapping=cap), text_softcap=None)
    with pytest.raises(DecisionConfigError, match="final_logit_softcapping"):
        read(dc(), text_softcap=cap)


def test_not_an_object_is_refused():
    with pytest.raises(DecisionConfigError, match="JSON object"):
        read([1, 2])
