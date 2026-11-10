"""The conformance tier's reading of a golden line (tests/gpu/conformance.py), without a server: section 10's fields,
a refusal of any other form, and the core's measures and bounds."""

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
    "variant": "strands-chat",
    "state_ids": [2, 5, 6],
    "images": [{"sha256": "ab" * 32, "span": [1, 2]}],
    "questions": {
        "q": {"question_ids": [7, 8], "codes": ["A", "B"], "probabilities": [0.25, 0.75]}
    },
}


def test_a_line_with_section_10s_fields():
    g = conformance.read_golden(LINE)
    assert g["request"] == REQUEST and g["state_ids"] == [2, 5, 6]
    assert g["images"] == [{"sha256": "ab" * 32, "span": [1, 2]}]
    assert g["questions"]["q"]["probabilities"] == [0.25, 0.75]
    assert conformance.rendering(g) == {
        "variant": "strands-chat",
        "state_ids": [2, 5, 6],
        "images": [{"sha256": "ab" * 32, "span": [1, 2]}],
        "questions": {"q": {"question_ids": [7, 8], "codes": ["A", "B"]}},
    }


def drop(key):
    return {k: v for k, v in LINE.items() if k != key}


@pytest.mark.parametrize(
    "line",
    [
        {},
        drop("variant"),
        drop("state_ids"),
        drop("images"),
        drop("questions"),
        {**LINE, "questions": {}},
        {**LINE, "request": "x"},
        {**LINE, "images": [{"span": [1, 2]}]},  # no sha256
        {**LINE, "questions": {"q": {"question_ids": [7], "probabilities": [1.0]}}},  # no codes
        {
            **LINE,
            "questions": {"q": {"question_ids": [7], "codes": ["A"], "probabilities": [0.5, 0.5]}},
        },
    ],
)
def test_any_other_form_is_unreadable(line):
    with pytest.raises(ValueError, match="unreadable golden line"):
        conformance.read_golden(line)


def test_the_measures_and_bounds_are_the_cores():
    cc = pytest.importorskip("strands_decider.core.conformance")
    assert cc.bounds() == (0.98, 0.02) and cc.MIN_QUESTIONS == 1000
    a = cc.agreement({(0, "q"): [0.26, 0.74]}, {(0, "q"): [0.25, 0.75]})
    assert (a.questions, a.top1, a.mean_abs_dp) == (1, 1.0, pytest.approx(0.01))
    assert a.conforms(min_questions=1) and not a.conforms()
