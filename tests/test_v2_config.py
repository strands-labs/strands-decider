"""`decision_config.json` (docs/v2-spec.md, The decision_config.json schema): strict, with
an `interface` discriminator, and a provenance check that catches common leaks of internal
locations (not a guarantee). Torch-free; the fixture is the E2B checkpoint's file reduced to
the schema's fields."""

from __future__ import annotations

import copy
import json

import pytest
from v2_support import FIXTURES, decision_config_fixture

from strands_decider.v2 import (
    CODES,
    FORMAT_VERSION,
    INTERFACE,
    VARIANTS,
    DecisionConfig,
    Provenance,
)
from strands_decider.v2.config import REQUIRED


def _with(**kw) -> dict:
    return {**copy.deepcopy(decision_config_fixture()), **kw}


def test_fixture_loads_from_file_and_directory():
    a = DecisionConfig.load(FIXTURES / "decision_config.json")
    b = DecisionConfig.load(FIXTURES)
    assert a == b
    assert a.interface == INTERFACE == "strands-decider-v2"
    assert a.format_version == FORMAT_VERSION and a.prompt in VARIANTS
    assert a.final_logit_softcapping == 30.0
    assert a.codes == CODES and len(a.token_ids) == 255
    assert a.attention_mode == "prefix_bidir" and a.pooling == "last"
    assert a.provenance == Provenance(
        base_model="google/gemma-4-E2B-it",
        base_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    )
    assert DecisionConfig.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    assert a.to_dict() == decision_config_fixture()


def test_provenance_is_optional_but_not_null():
    d = _with()
    del d["provenance"]
    assert DecisionConfig.from_dict(d).provenance is None
    assert "provenance" not in DecisionConfig.from_dict(d).to_dict()
    with pytest.raises(ValueError, match="leave the field out"):
        DecisionConfig.from_dict(_with(provenance=None))


@pytest.mark.parametrize("field", [f for f in REQUIRED if f != "interface"])
def test_missing_required_fields_are_errors(field):
    d = _with()
    del d[field]
    with pytest.raises(ValueError, match=f"missing required fields \\['{field}'\\]"):
        DecisionConfig.from_dict(d)


@pytest.mark.parametrize(
    "interface",
    ["<missing>", None, "", "strands-decider-v1", "Strands-Decider-v2", "pplx-decider", 2],
)
def test_the_interface_field_is_required_and_exact(interface):
    d = _with(interface=interface)
    if interface == "<missing>":
        del d["interface"]
    with pytest.raises(ValueError, match="not a Strands Decider v2 decision config"):
        DecisionConfig.from_dict(d)
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError, match="not a Strands Decider v2 decision config"):
        DecisionConfig(**{**good.__dict__, "interface": interface})


def test_another_formats_file_is_refused_by_its_interface():
    """A decision_config.json of another format, here shaped like the training side's
    (synthetic values), is named as not v2 before any field is checked."""
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
    with pytest.raises(ValueError, match="not a Strands Decider v2 decision config"):
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
        {"Prompt": "strands-v2-plain"},
    ],
    ids=lambda e: next(iter(e)),
)
def test_unknown_fields_are_errors(extra):
    with pytest.raises(ValueError, match="unknown fields"):
        DecisionConfig.from_dict(_with(**extra))


@pytest.mark.parametrize("prompt", ["strands-v2-xml", "strands-v1", "", None, 1])
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
        ("codes", list(CODES[:-1]), "255 v2 codes"),
        ("codes", list(reversed(CODES)), "255 v2 codes"),
        ("codes", [*CODES[:-1], "GZ"], "255 v2 codes"),
        ("codes", "ABC", "must be a list"),
        ("token_ids", list(range(254)), "255 non-negative"),
        ("token_ids", [-1, *range(1, 255)], "255 non-negative"),
        ("token_ids", [1.0, *range(1, 255)], "255 non-negative"),
        ("token_ids", [True, *range(1, 255)], "255 non-negative"),
        ("token_ids", [0, 0, *range(2, 255)], "distinct"),
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
        ("final_logit_softcapping", 0, "final_logit_softcapping"),
        ("final_logit_softcapping", -30.0, "final_logit_softcapping"),
        ("final_logit_softcapping", None, "final_logit_softcapping"),
        ("final_logit_softcapping", "30.0", "final_logit_softcapping"),
        ("final_logit_softcapping", True, "final_logit_softcapping"),
        ("final_logit_softcapping", float("nan"), "final_logit_softcapping"),
        ("final_logit_softcapping", float("inf"), "final_logit_softcapping"),
        ("final_logit_softcapping", 1e300, "final_logit_softcapping"),
        ("attention_mode", "causal", "attention_mode"),
        ("pooling", "mean", "pooling"),
        ("max_length_validated", 0, "max_length_validated"),
        ("max_length_validated", 8192.0, "max_length_validated"),
        ("max_length_validated", None, "max_length_validated"),
    ],
)
def test_bad_values_are_errors(field, value, match):
    with pytest.raises(ValueError, match=match):
        DecisionConfig.from_dict(_with(**{field: value}))


@pytest.mark.parametrize(
    "provenance",
    [
        {"base_model": "/fsx/user/models/gemma-4-E2B-it"},
        {"source": "/fsx/user/run/train.py"},
        {"release": "fsx/user/run"},
        {"source": "s3://bucket/checkpoints/run-1"},
        {"source": "file:///home/user/run"},
        {"source": "http://example.com/recipe"},
        {"release": "~/checkpoints/e2b"},
        {"release": "../checkpoints/e2b"},
        {"release": "C:\\runs\\e2b"},
        {"base_model": "gemma-4-E2B-it"},
        {"base_revision": "main"},
        {"base_revision": "3e22461"},
        {"release": ""},
        {"release": 3},
        {"trainer": "train.py"},
        {"base_model": "google/gemma-4-E2B-it", "data": "corpus-v1"},
    ],
)
def test_internal_or_unknown_provenance_is_refused(provenance):
    with pytest.raises(ValueError, match="provenance"):
        DecisionConfig.from_dict(_with(provenance=provenance))


# Synthetic values in the shapes internal locations take.
@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("source", "https://code.amazon.com/packages/StrandsDecider/trees/mainline", "host"),
        ("source", "https://w.amazon.com/bin/view/StrandsDecider", "host"),
        ("source", "https://drive.corp.amazon.com/documents/run-1", "host"),
        ("source", "https://user@code.amazon.com/pkg", "host"),
        ("source", "https://docs.hub.amazon.dev/recipe", "host"),
        ("source", "https://tools.a2z.com/run/1", "host"),
        ("source", "https://dashboard.aws.dev/run/1", "host"),
        ("source", "https://CODE.Amazon.COM/pkg", "host"),
        ("release", "code.amazon.com/StrandsDecider", "host"),
        ("source", "https://my-bucket.s3.amazonaws.com/checkpoints/run-1", "S3"),
        ("source", "https://s3.amazonaws.com/my-bucket/run-1", "S3"),
        ("source", "https://s3.us-west-2.amazonaws.com/my-bucket/run-1", "S3"),
        ("source", "https://my-bucket.s3.us-east-1.amazonaws.com/run-1", "S3"),
        ("source", "https://s3-us-west-2.amazonaws.com/my-bucket/run-1", "S3"),
        ("source", "https://ap.s3-accesspoint.us-east-1.amazonaws.com/run-1", "S3"),
        ("source", "https://github.com/org/repo/blob/main/fsx/user/train.py", "user path"),
        ("source", "https://example.org/home/user/run", "user path"),
        ("base_model", "Users/someone", "user path"),
        ("release", "workplace/user", "user path"),
        ("release", "fsx/user", "user path"),
        ("release", "efs/run-1", "user path"),
        ("source", "https://example.org/local/home/user/x", "user path"),
        ("source", "https://example.org/apollo/env/Trainer/bin", "user path"),
    ],
)
def test_common_internal_values_are_refused(field, value, why):
    with pytest.raises(ValueError, match=f"provenance.{field} is .*{why}"):
        DecisionConfig.from_dict(_with(provenance={field: value}))


@pytest.mark.parametrize(
    "provenance",
    [
        {"source": "https://github.com/strands-labs/strands-decider/tree/v0.4.0/training"},
        {"source": "https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611"},
        {"source": "https://notamazon.com/recipe"},
        {"source": "https://example.com/s3/guide"},
        {"source": "https://github.com/awslabs/s3-tools"},
        {"base_model": "amazon/chronos-t5-small"},  # a Hub organisation, not a host
        {"release": "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"},
    ],
)
def test_public_values_are_not_mistaken_for_internal_ones(provenance):
    assert DecisionConfig.from_dict(_with(provenance=provenance)).provenance.to_dict() == provenance


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
    assert set(json.loads(text)) <= {*REQUIRED, "provenance"}


def test_constructing_directly_validates_too():
    good = DecisionConfig.load(FIXTURES)
    with pytest.raises(ValueError, match="not implemented"):
        DecisionConfig(**{**good.__dict__, "prompt": "strands-v2-xml"})
