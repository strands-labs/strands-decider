"""decision_config.json at startup: the interface, the code token ids, the temperature and the soft-cap (v2
specification, sections 8 and 9). No vLLM or torch needed."""

from __future__ import annotations

import math

import pytest

from strands_decider_v2_vllm.decision_config import (
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


def read(config, text_softcap=30.0, vocab_size=VOCAB):
    return read_decision_config(config, text_softcap=text_softcap, vocab_size=vocab_size)


def test_an_intermediate_checkpoint_loads_with_one_warning():
    spec = read(dc())
    assert spec.token_ids == tuple(IDS) and spec.temperature == 1.0 and spec.softcap == 30.0
    assert len(spec.warnings) == 1 and '"interface"' in spec.warnings[0]
    assert "\n" not in spec.warnings[0]


def test_a_released_checkpoint_loads_without_warnings():
    spec = read(dc(interface="strands-decider-v2", final_logit_softcapping=30.0))
    assert spec.warnings == () and spec.softcap == 30.0


@pytest.mark.parametrize("interface", ["pplx-decider", "strands-decider-v1", "", None, 2])
def test_another_interface_is_refused(interface):
    with pytest.raises(DecisionConfigError, match="interface must be"):
        read(dc(interface=interface))


def test_the_interface_can_be_required():
    with pytest.raises(DecisionConfigError, match="is required"):
        read_decision_config(dc(), text_softcap=30.0, vocab_size=VOCAB, interface_optional=False)


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


@pytest.mark.parametrize("cap", [0, 0.0, -30.0, math.inf, math.nan, True, "30"])
def test_a_bad_soft_cap_is_refused(cap):
    with pytest.raises(DecisionConfigError, match="final_logit_softcapping"):
        read(dc(final_logit_softcapping=cap), text_softcap=None)
    with pytest.raises(DecisionConfigError, match="final_logit_softcapping"):
        read(dc(), text_softcap=cap)


def test_not_an_object_is_refused():
    with pytest.raises(DecisionConfigError, match="JSON object"):
        read([1, 2])
