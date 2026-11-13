import math

import pytest
from trainer_support import SAME_PYTHON_SUM

from training import metrics


def close(a, b) -> bool:
    """Equal, but for floats summed in another order (trainer_support.SAME_PYTHON_SUM)."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def test_metrics_match_previous(previous):
    case = previous["metrics"]
    for temp, expected in case["results"].items():
        got = metrics.score_predictions(
            metrics.predictions_from_logits(case["rows"], case["logits"], float(temp))
        )
        if SAME_PYTHON_SUM:
            assert got == expected["full"]
            assert metrics.summary(got) == expected["summary"]
        else:
            assert close(got, expected["full"])
            assert close(metrics.summary(got), expected["summary"])


def test_calibrate_temperature_matches_previous(previous):
    case = previous["metrics"]
    rows = case["rows"]
    idx = [
        metrics.option_index(metrics.option_labels(r["question"]), metrics.answer_label(r))
        for r in rows
    ]
    for c in case["calibrate_temperature"]:
        lg = [
            [v * c["scale"] for v in row[: len(metrics.option_labels(r["question"]))]]
            for row, r in zip(case["logits"], rows, strict=True)
        ]
        assert repr(metrics.calibrate_temperature(lg, idx)) == c["out"]


def test_duplicate_ids_are_refused():
    row = {
        "id": "a",
        "suite": "s",
        "family": "f",
        "source": {"dataset": "d"},
        "question": {"type": "noul"},
        "target": 1.0,
        "label": True,
    }
    with pytest.raises(ValueError):
        metrics.predictions_from_logits([row, dict(row)], [[0.0, 1.0], [0.0, 1.0]])


def test_perfect_and_uniform_predictions():
    rows = [
        {
            "id": str(i),
            "suite": "s",
            "family": str(i),
            "source": {"dataset": "d"},
            "question": {"type": "choice", "criteria": {"a": None, "b": None}},
            "target": "a",
            "label": "a",
        }
        for i in range(10)
    ]
    sure = metrics.score_predictions(metrics.predictions_from_logits(rows, [[50.0, 0.0]] * 10))
    assert sure["accuracy"] == 1.0 and sure["nll"] < 1e-12 and sure["ece"] < 1e-12
    even = metrics.score_predictions(metrics.predictions_from_logits(rows, [[0.0, 0.0]] * 10))
    assert (
        even["accuracy"] == 1.0 and abs(even["nll"] - 0.6931471805599453) < 1e-12
    )  # ties go to the first option
