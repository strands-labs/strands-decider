"""`decision_config.json`: the fixture is the E2B checkpoint's file; unknown fields, other formats'
files and values that aren't this interface's are refused."""

from __future__ import annotations

import copy
import json

import pytest
from core_support import FIXTURES, decision_config_fixture

from strands_decider.core import CODE_TOKEN_IDS, CODES, DecisionConfig


def _load(**kw) -> DecisionConfig:
    return DecisionConfig.model_validate_json(
        json.dumps({**copy.deepcopy(decision_config_fixture()), **kw})
    )


def test_fixture_loads_from_file_and_directory():
    a = DecisionConfig.load(FIXTURES / "decision_config.json")
    assert a == DecisionConfig.load(FIXTURES)
    assert a.prompt == "strands-chat" and a.final_logit_softcapping == 30.0
    assert a.codes == CODES and a.token_ids == CODE_TOKEN_IDS
    assert a.inputs == ("text", "image")
    assert a.provenance.base_model == "google/gemma-4-E2B-it"
    assert a.model_dump(mode="json") == decision_config_fixture()


@pytest.mark.parametrize("field", list(decision_config_fixture()))
def test_every_field_is_required(field):
    d = decision_config_fixture()
    del d[field]
    with pytest.raises(ValueError, match=field):
        DecisionConfig.model_validate_json(json.dumps(d))


@pytest.mark.parametrize(
    "change",
    [
        {"interface": "vllm-d3"},  # another decision model's file of the same name
        {"format_version": 2},
        {"prompt": "strands-plain"},
        {"attention_mode": "causal"},
        {"pooling": "mean"},
        {"inputs": ["image"]},
        {"inputs": ["text", "audio"]},
        {"temperature": 0},
        {"temperature": True},
        {"temperature": "1.0"},
        {"final_logit_softcapping": -1.0},
        {"max_length_validated": 0},
        {"codes": list(CODES[:-1])},
        {"token_ids": [*CODE_TOKEN_IDS[:-1], 5]},
        {"provenance": {"base_model": "google/gemma-4-E2B-it", "base_revision": "main"}},
        {"provenance": {**decision_config_fixture()["provenance"], "source": "s3://bucket"}},
        {"extra": 1},
    ],
)
def test_values_that_are_not_this_interfaces_are_refused(change):
    with pytest.raises(ValueError):
        _load(**change)


def test_the_fixture_holds_no_internal_value():
    text = (FIXTURES / "decision_config.json").read_text(encoding="utf-8")
    assert "/fsx" not in text and "s3://" not in text
