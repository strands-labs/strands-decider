"""The conformance tier's reading of a golden line and its measure (tests/gpu/conformance.py), without a server."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "conformance", Path(__file__).parent / "gpu" / "conformance.py"
)
conformance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conformance)

REQUEST = {
    "state": "x",
    "images": ["iVBOR"],
    "questions": {"q": {"type": "noul", "instructions": "?"}},
}
LINE = {
    "request": REQUEST,
    "state_ids": [2, 5, 6],
    "images": [{"sha256": "ab" * 32, "span": [1, 2]}],
    "questions": {
        "q": {"question_ids": [7, 8], "codes": ["A", "B"], "probabilities": [0.25, 0.75]}
    },
}


def test_a_golden_line_and_the_fields_rendering_must_reproduce():
    g = conformance.read_golden(LINE)
    assert g["request"] == REQUEST and g["questions"]["q"]["probabilities"] == [0.25, 0.75]
    assert conformance.rendering(g) == {
        "state_ids": [2, 5, 6],
        "images": [{"sha256": "ab" * 32, "span": [1, 2]}],
        "questions": {"q": {"question_ids": [7, 8], "codes": ["A", "B"]}},
    }


@pytest.mark.parametrize(
    "line",
    [
        {},
        {k: v for k, v in LINE.items() if k != "state_ids"},
        {**LINE, "questions": {}},
        {**LINE, "request": "x"},
        {**LINE, "images": [{"span": [1, 2]}]},
        {
            **LINE,
            "questions": {"q": {"question_ids": [7], "codes": ["A"], "probabilities": [0.5, 0.5]}},
        },
    ],
)
def test_any_other_form_is_unreadable(line):
    with pytest.raises(ValueError, match="unreadable golden line"):
        conformance.read_golden(line)


def test_top1_agreement_and_mean_abs_dp():
    top1, dp = conformance.agreement(
        {0: [0.26, 0.74], 1: [0.6, 0.4]}, {0: [0.25, 0.75], 1: [0.4, 0.6]}
    )
    assert top1 == 0.5 and dp == pytest.approx((0.01 + 0.2) / 2)
