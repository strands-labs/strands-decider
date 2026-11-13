import copy
import random

import pytest
import torch

from training import targets


def test_targets_match_previous(previous):
    case = previous["target_matrix"]
    assert targets.target_matrix(case["rows"], torch.device("cpu")).tolist() == case["out"]


@pytest.mark.parametrize(
    "row",
    [
        {
            "id": "sum",
            "question": {"type": "choice", "criteria": {"a": None, "b": None}},
            "target": [0.6, 0.6],
        },
        {
            "id": "len",
            "question": {"type": "choice", "criteria": {"a": None, "b": None}},
            "target": [1.0],
        },
        {"id": "neg", "question": {"type": "noul"}, "target": 1.5},
        {
            "id": "score",
            "question": {"type": "score", "criteria": ["x", "y"]},
            "target": [0.5, 0.5],
        },
    ],
)
def test_targets_refuse_invalid_rows(row):
    with pytest.raises(ValueError):
        targets.target_matrix([row], torch.device("cpu"))


def test_shuffle_options_matches_previous(previous):
    case = previous["shuffle_options"]
    for c in case["cases"]:
        assert (
            targets.shuffle_options(copy.deepcopy(case["rows"]), random.Random(c["seed"]))
            == c["out"]
        )


def test_shuffle_options_keeps_fixed_order_and_noul_rows_but_draws_for_them():
    rows = [
        {
            "id": "f",
            "question": {"type": "choice", "criteria": {"low": None, "mid": None, "high": None}},
            "target": [0.1, 0.2, 0.7],
            "source": {"fixed_order": True},
        },
        {"id": "n", "question": {"type": "noul"}, "target": 0.3, "source": {}},
    ]
    rng = random.Random(1)
    out = targets.shuffle_options(rows, rng)
    assert out == rows and out[0] is not rows[0]
    probe = random.Random(1)
    probe.shuffle(list(range(3)))  # one shuffle was drawn, for the fixed-order row
    assert rng.getstate() == probe.getstate()


def test_shuffle_options_moves_the_target_with_its_option():
    row = {
        "id": "c",
        "question": {"type": "choice", "criteria": {f"k{i}": f"d{i}" for i in range(6)}},
        "target": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "source": {},
    }
    for seed in range(20):
        out = targets.shuffle_options([row], random.Random(seed))[0]
        names = list(out["question"]["criteria"])
        assert out["target"][names.index("k2")] == 1.0
        assert out["question"]["criteria"]["k2"] == "d2"


def test_options_and_labels():
    assert targets.option_labels({"type": "noul"}) == [False, True]
    assert targets.option_index([False, True], True) == 1
    with pytest.raises(ValueError):
        targets.option_index([0, 1], True)  # True is not 1
    assert targets.answer_label({"id": "x", "target": "b", "label": "a"}) == "a"
