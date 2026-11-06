"""`decision_config.json` (docs/v2-spec.md, The decision_config.json schema): strict, and
public by construction. Torch-free; the fixture is the E2B checkpoint's file reduced to the
schema's fields."""

from __future__ import annotations

import copy
import json

import pytest
from v2_support import FIXTURES, decision_config_fixture

from strands_decider.v2 import CODES, FORMAT_VERSION, VARIANTS, DecisionConfig, Provenance
from strands_decider.v2.config import REQUIRED


def _with(**kw) -> dict:
    return {**copy.deepcopy(decision_config_fixture()), **kw}


def test_fixture_loads_from_file_and_directory():
    a = DecisionConfig.load(FIXTURES / "decision_config.json")
    b = DecisionConfig.load(FIXTURES)
    assert a == b
    assert a.format_version == FORMAT_VERSION and a.prompt in VARIANTS
    assert a.codes == CODES and len(a.token_ids) == 255
    assert a.attention_mode == "prefix_bidir" and a.pooling == "last"
    assert a.provenance == Provenance(
        base_model="google/gemma-4-E2B-it",
        base_revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    )
    assert DecisionConfig.from_dict(json.loads(json.dumps(a.to_dict()))) == a
    assert a.to_dict() == decision_config_fixture()


def test_provenance_is_optional():
    d = _with()
    del d["provenance"]
    assert DecisionConfig.from_dict(d).provenance is None
    assert "provenance" not in DecisionConfig.from_dict(d).to_dict()


@pytest.mark.parametrize("field", REQUIRED)
def test_missing_required_fields_are_errors(field):
    d = _with()
    del d[field]
    with pytest.raises(ValueError, match=f"missing required fields \\['{field}'\\]"):
        DecisionConfig.from_dict(d)


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
