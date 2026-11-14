"""`decision_config.json` (docs/spec.md, section 8): strict, with
an `interface` discriminator, the inputs a checkpoint takes, and a required provenance in
public formats: a Hub id, a commit, a model name and an https URL. Keeping private paths out
of a released checkpoint is the release export's job, not the loader's. Torch-free; the
fixture is the E2B checkpoint's file reduced to the schema's fields."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from core_support import FIXTURES, decision_config_fixture

from strands_decider.core import (
    CODES,
    FORMAT_VERSION,
    INPUTS,
    INTERFACE,
    VARIANTS,
    DecisionConfig,
    Provenance,
)
from strands_decider.core.config import INT32_MAX, REQUIRED, check_float32_scale

BASE = {
    "base_model": "google/gemma-4-E2B-it",
    "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}


def _with(**kw) -> dict:
    return {**copy.deepcopy(decision_config_fixture()), **kw}


def test_fixture_loads_from_file_and_directory():
    a = DecisionConfig.load(FIXTURES / "decision_config.json")
    b = DecisionConfig.load(FIXTURES)
    assert a == b
    assert a.interface == INTERFACE == "strands-decider"
    assert a.format_version == FORMAT_VERSION and a.prompt in VARIANTS
    assert a.final_logit_softcapping == 30.0
    assert a.codes == CODES and len(a.token_ids) == 255
    assert a.attention_mode == "prefix_bidir" and a.pooling == "last"
    assert a.inputs == ("text", "image") and INPUTS == ("text", "image")
    assert a.provenance == Provenance(**BASE)
    assert DecisionConfig.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    assert a.to_dict() == decision_config_fixture()


def test_provenance_and_its_base_are_required():
    """The image processor, the tokenizer and any part not fine-tuned are the base's at
    `base_revision`, so a checkpoint must name them."""
    d = _with()
    del d["provenance"]
    with pytest.raises(ValueError, match=r"missing required fields \['provenance'\]"):
        DecisionConfig.from_dict(d)
    with pytest.raises(ValueError, match="provenance must be an object"):
        DecisionConfig.from_dict(_with(provenance=None))
    for field in BASE:
        partial = {k: v for k, v in BASE.items() if k != field}
        with pytest.raises(
            ValueError, match=f"provenance is missing required fields \\['{field}'\\]"
        ):
            DecisionConfig.from_dict(_with(provenance=partial))
        with pytest.raises(ValueError, match=f"provenance.{field}"):
            DecisionConfig.from_dict(_with(provenance={**BASE, field: None}))
    with pytest.raises(ValueError, match="missing required fields"):
        DecisionConfig.from_dict(_with(provenance={}))
    with pytest.raises(TypeError):
        Provenance(base_model=BASE["base_model"])  # type: ignore[call-arg]
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError, match="provenance must be a Provenance"):
        DecisionConfig(**{**good.__dict__, "provenance": None})


@pytest.mark.parametrize("inputs", [["text"], ["text", "image"], ["image", "text"]])
def test_inputs_name_what_the_checkpoint_takes(inputs):
    cfg = DecisionConfig.from_dict(_with(inputs=inputs))
    assert cfg.inputs == tuple(inputs)
    assert cfg.to_dict()["inputs"] == inputs


@pytest.mark.parametrize(
    "inputs",
    [
        [],
        ["image"],
        ["text", "text"],
        ["text", "image", "image"],
        ["text", "audio"],
        ["Text"],
        ["text", None],
        ["text", 1],
        "text",
        None,
        {"text": True},
    ],
)
def test_inputs_must_be_distinct_known_and_hold_text(inputs):
    with pytest.raises(ValueError, match="inputs must be an array"):
        DecisionConfig.from_dict(_with(inputs=inputs))


@pytest.mark.parametrize("field", [f for f in REQUIRED if f != "interface"])
def test_missing_required_fields_are_errors(field):
    d = _with()
    del d[field]
    with pytest.raises(ValueError, match=f"missing required fields \\['{field}'\\]"):
        DecisionConfig.from_dict(d)


@pytest.mark.parametrize(
    "interface",
    ["<missing>", None, "", "strands-decider-v1", "Strands-Decider", "other-decider", 2],
)
def test_the_interface_field_is_required_and_exact(interface):
    d = _with(interface=interface)
    if interface == "<missing>":
        del d["interface"]
    with pytest.raises(ValueError, match="not a Strands Decider decision config"):
        DecisionConfig.from_dict(d)
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError, match="not a Strands Decider decision config"):
        DecisionConfig(**{**good.__dict__, "interface": interface})


def test_another_formats_file_is_refused_by_its_interface():
    """A decision_config.json of another format, here shaped like the training side's
    (synthetic values), is refused by its interface before any field is checked."""
    other = {
        "format_version": 1,
        "base_model": "google/gemma-4-E2B-it",
        "codes": 255,
        "token_ids": list(range(255)),
        "temperature": 1.0,
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "finetune": {"lr": 1e-4},
    }
    with pytest.raises(ValueError, match="not a Strands Decider decision config"):
        DecisionConfig.from_dict(other)


@pytest.mark.parametrize(
    "extra",
    [
        # What a training run writes beside the public fields. Synthetic values.
        {"finetune": {"trainer": "/fsx/user/run/train.py", "lr": 1e-4}},
        {"soup": {"packages": ["/fsx/user/a", "/fsx/user/b"]}},
        {"base_model": "google/gemma-4-E2B-it"},
        {"revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7"},
        {"temperature_by_kind": {"score": 1.2}},
        {"Prompt": "strands-plain"},
    ],
    ids=lambda e: next(iter(e)),
)
def test_unknown_fields_are_errors(extra):
    with pytest.raises(ValueError, match="unknown fields"):
        DecisionConfig.from_dict(_with(**extra))


@pytest.mark.parametrize("prompt", ["strands-xml", "strands-v1", "", None, 1])
def test_unimplemented_prompts_are_refused(prompt):
    with pytest.raises(ValueError, match="not implemented"):
        DecisionConfig.from_dict(_with(prompt=prompt))


@pytest.mark.parametrize("prompt", VARIANTS)
def test_both_variants_load(prompt):
    assert DecisionConfig.from_dict(_with(prompt=prompt)).prompt == prompt


@pytest.mark.parametrize("value", [1e-3, 0.7, 1, 2.5, 1e3, 3.4e38, 1.2e-38])
def test_scales_within_float32_load(value):
    cfg = DecisionConfig.from_dict(_with(temperature=value, final_logit_softcapping=value))
    assert cfg.temperature == value and cfg.final_logit_softcapping == value


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("format_version", 2, "format_version"),
        ("format_version", "1", "format_version"),
        ("format_version", True, "format_version"),
        ("codes", list(CODES[:-1]), "255 answer codes"),
        ("codes", list(reversed(CODES)), "255 answer codes"),
        ("codes", [*CODES[:-1], "GZ"], "255 answer codes"),
        ("codes", "ABC", "must be a list"),
        ("token_ids", list(range(254)), "255 non-negative"),
        ("token_ids", [-1, *range(1, 255)], "255 non-negative"),
        ("token_ids", [1.0, *range(1, 255)], "255 non-negative"),
        ("token_ids", [True, *range(1, 255)], "255 non-negative"),
        ("token_ids", [0, 0, *range(2, 255)], "distinct"),
        ("token_ids", [INT32_MAX + 1, *range(1, 255)], "at most"),
        ("token_ids", [10**400, *range(1, 255)], "at most"),
        ("temperature", 0, "temperature"),
        ("temperature", -1.0, "temperature"),
        ("temperature", float("nan"), "temperature"),
        ("temperature", float("inf"), "temperature"),
        ("temperature", "1.0", "temperature"),
        ("temperature", True, "temperature"),
        ("temperature", None, "temperature"),
        # Past float32's normal range: the readout divides by it in float32.
        ("temperature", 1e39, "temperature"),
        ("temperature", 1e300, "temperature"),
        ("temperature", 1e-39, "temperature"),
        ("temperature", 5e-324, "temperature"),
        # An integer too large for a double: a ValueError, never an OverflowError.
        ("temperature", 10**400, "temperature"),
        ("temperature", -(10**400), "temperature"),
        # c / T, the largest final logit, past float32's range.
        ("temperature", 1.2e-38, "overflows float32"),
        ("final_logit_softcapping", 0, "final_logit_softcapping"),
        ("final_logit_softcapping", -30.0, "final_logit_softcapping"),
        ("final_logit_softcapping", None, "final_logit_softcapping"),
        ("final_logit_softcapping", "30.0", "final_logit_softcapping"),
        ("final_logit_softcapping", True, "final_logit_softcapping"),
        ("final_logit_softcapping", float("nan"), "final_logit_softcapping"),
        ("final_logit_softcapping", float("inf"), "final_logit_softcapping"),
        ("final_logit_softcapping", 1e300, "final_logit_softcapping"),
        ("final_logit_softcapping", 10**400, "final_logit_softcapping"),
        ("attention_mode", "causal", "attention_mode"),
        ("pooling", "mean", "pooling"),
        ("max_length_validated", 0, "max_length_validated"),
        ("max_length_validated", 8192.0, "max_length_validated"),
        ("max_length_validated", None, "max_length_validated"),
        ("max_length_validated", INT32_MAX + 1, "max_length_validated"),
        ("max_length_validated", 10**400, "max_length_validated"),
        ("format_version", 10**400, "format_version"),
    ],
)
def test_bad_values_are_errors(field, value, match):
    with pytest.raises(ValueError, match=match):
        DecisionConfig.from_dict(_with(**{field: value}))
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError):  # constructed directly too: a ValueError, whatever the type
        DecisionConfig(**{**good.__dict__, field: value})


@pytest.mark.parametrize("field", ["prompt", "attention_mode", "pooling", "inputs", "interface"])
def test_a_huge_integer_anywhere_is_a_plain_valueerror(field):
    """A value too large to print (Python refuses int-to-str past 4,300 digits) is
    described by its size, so the error is the schema's, not the interpreter's."""
    value = [10**5000] if field == "inputs" else 10**5000
    with pytest.raises(ValueError) as e:
        DecisionConfig.from_dict(_with(**{field: value}))
    assert "bits" in str(e.value) and "Exceeds the limit" not in str(e.value)


@pytest.mark.parametrize("field", ["codes", "token_ids"])
@pytest.mark.parametrize("value", [None, 3, "ABC"])
def test_codes_and_ids_constructed_directly_must_be_arrays(field, value):
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError, match=f"{field} must be"):
        DecisionConfig(**{**good.__dict__, field: value})


def test_integers_hold_to_their_bounds():
    ids = [INT32_MAX, *range(1, 255)]
    cfg = DecisionConfig.from_dict(_with(token_ids=ids, max_length_validated=INT32_MAX))
    assert cfg.token_ids[0] == INT32_MAX and cfg.max_length_validated == INT32_MAX


def test_the_version_is_read_before_the_fields():
    """A file of a later layout is refused as one, not for the fields it adds or drops."""
    d = _with(format_version=2, attention="prefix_bidir")
    del d["pooling"]
    with pytest.raises(ValueError, match="format_version 2 is not supported"):
        DecisionConfig.from_dict(d)


def test_scales_take_numpy_scalars_and_refuse_bools():
    assert check_float32_scale("t", np.float32(30.0)) == 30.0
    assert (
        check_float32_scale("t", np.float64(0.5)) == 0.5
        and check_float32_scale("t", np.int64(2)) == 2.0
    )
    cfg = DecisionConfig.from_dict(
        _with(temperature=np.float32(0.5), final_logit_softcapping=np.int64(30))
    )
    assert cfg.temperature == 0.5 and type(cfg.temperature) is float
    assert type(cfg.final_logit_softcapping) is float
    # Held as Python floats, the config goes through JSON and back unchanged.
    again = DecisionConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.to_dict() == cfg.to_dict()
    for bad in (True, np.bool_(True), "1", None, [1.0], 10**400, np.float32("nan")):
        with pytest.raises(ValueError, match="must be a finite number"):
            check_float32_scale("t", bad)


@pytest.mark.parametrize(
    ("provenance", "match"),
    [
        ({"base_model": "/fsx/user/models/gemma-4-E2B-it"}, "base_model must be a Hub id"),
        ({"source": "/fsx/user/run/train.py"}, "source must be an https URL"),
        ({"release": "fsx/user/run"}, "release must be a model name"),
        ({"source": "s3://bucket/checkpoints/run-1"}, "source must be an https URL"),
        ({"source": "file:///home/user/run"}, "source must be an https URL"),
        ({"source": "http://example.com/recipe"}, "source must be an https URL"),
        ({"release": "~/checkpoints/e2b"}, "release must be a model name"),
        ({"release": "../checkpoints/e2b"}, "release must be a model name"),
        ({"release": "C:\\runs\\e2b"}, "release must be a model name"),
        ({"base_model": "gemma-4-E2B-it"}, "base_model must be a Hub id"),
        ({"base_revision": "main"}, "base_revision must be a 40-character"),
        ({"base_revision": "3e22461"}, "base_revision must be a 40-character"),
        ({"release": ""}, "release must be a non-empty string"),
        ({"release": 3}, "release must be a non-empty string"),
        # Null is not absent: an optional field that does not apply is left out.
        ({"release": None}, "provenance.release is null"),
        ({"source": None}, "provenance.source is null"),
        # Whole strings, in ASCII.
        ({"base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7\n"}, "base_revision must be"),
        ({"base_model": "google/gemma-4-E\uff12B-it"}, "base_model must be a Hub id"),
        ({"base_model": "google/gemma-4-E2B-it\n"}, "base_model must be a Hub id"),
        ({"base_model": "\uff47oogle/gemma-4-E2B-it"}, "base_model must be a Hub id"),
        ({"release": "strands-decider-E2B\n"}, "release must be a model name"),
        ({"release": "strands-decider-é"}, "release must be a model name"),
        ({"source": "https://"}, "source must be an https URL"),
        ({"source": "https://example.com/a b"}, "source must be an https URL"),
        ({"source": "https://example.com/recipe\n"}, "source must be an https URL"),
        ({"trainer": "train.py"}, "unknown fields"),
        ({"data": "corpus-v1"}, "unknown fields"),
    ],
)
def test_internal_or_unknown_provenance_is_refused(provenance, match):
    """Each value beside a valid base, so that the check that refuses it is the one named."""
    with pytest.raises(ValueError, match=match):
        DecisionConfig.from_dict(_with(provenance={**BASE, **provenance}))


def test_public_provenance_is_accepted():
    prov = {
        "base_model": "google/gemma-4-E2B-it",
        "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "release": "StrandsAgents/strands-decider-E2B-gemma4-v2-2611",
        "source": "https://github.com/strands-labs/strands-decider/tree/v0.4.0/training",
    }
    assert DecisionConfig.from_dict(_with(provenance=prov)).provenance.to_dict() == prov
    with pytest.raises(ValueError, match="provenance must be an object"):
        DecisionConfig.from_dict(_with(provenance="google/gemma-4-E2B-it"))


def test_the_fixture_holds_no_internal_value():
    text = (FIXTURES / "decision_config.json").read_text(encoding="utf-8")
    assert "/fsx" not in text and "s3://" not in text
    assert set(json.loads(text)) == set(REQUIRED)


def test_constructing_directly_validates_too():
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError, match="not implemented"):
        DecisionConfig(**{**good.__dict__, "prompt": "strands-xml"})
