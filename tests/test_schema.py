import math

import pytest

from strands_decider.schema import derive_confidence


@pytest.mark.parametrize(
    "probabilities, expected",
    [
        ([], 1.0),
        ([0.7], 1.0),
        ([0.5, 0.5], 0.0),
        ([0.8, 0.2], 0.6),
        ([1.0, 0.0], 1.0),
        ([0.2, 0.3, 0.5], 0.25),
        ([2.0, 0.0], 1.0),
        ([-1.0, -2.0], 0.0),
        ([float("nan"), 0.8], 1.0),
        ([0.8, float("nan")], 0.6),
        ([0.8, float("nan")], 0.6),
    ],
)
def test_derive_confidence(probabilities, expected):
    assert derive_confidence(probabilities) == pytest.approx(expected)


def test_derive_confidence_returns_positive_zero():
    result = derive_confidence([-0.0, -0.0])

    assert result == 0.0
    assert math.copysign(1.0, result) == 1.0
