"""The confidence formulas of the System One response, without dependencies.

`derive_confidence` gives a choice's confidence and `derive_score_confidence` a score's. The v1
engines (through `prompting.py`) and the torch-free core (`strands_decider.core`) both use them.
"""

from __future__ import annotations


def derive_score_confidence(probabilities: list[float], *, ordinal_smoothing: float = 0.0) -> float:
    """Confidence for an ordinal answer: how tightly the mass clusters on the scale.

    The max-probability formula used for `choice` is wrong for `score`, because it
    reads *spread* as *doubt*. On an ordered scale, mass on adjacent levels is not
    confusion -- a distribution split between levels 2 and 3 is confidently saying
    "about 2.5". Normalised standard deviation captures that, and correctly ranks a
    bimodal distribution (mass at both ends, expected value meaningless) as far worse
    than a uniform one, which max-probability cannot do.

        sigma     = sqrt( sum_i p_i * (i - mean)^2 )
        sigma_max = (L - 1) / 2                        # mass split at the extremes

    `ordinal_smoothing` corrects the floor. Training deliberately puts `eps` of each
    target's mass on neighbouring levels, so a perfectly fitted model still reports
    sigma ~ sqrt(eps) and could never reach 1.0. Without this correction a 3-level
    score would cap at 0.68 confidence, making the documented ">= 0.9 means act
    automatically" threshold unreachable by construction.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    p = [x / total for x in probabilities]

    mean = sum(i * pi for i, pi in enumerate(p))
    var = sum(pi * (i - mean) ** 2 for i, pi in enumerate(p))
    sigma = var**0.5

    sigma_max = (n - 1) / 2.0
    sigma_floor = ordinal_smoothing**0.5 if ordinal_smoothing > 0 else 0.0
    # Guard against a floor that meets or exceeds the ceiling (tiny L, large eps).
    if sigma_max <= sigma_floor:
        return 1.0 if sigma <= sigma_floor else 0.0

    conf = (sigma_max - sigma) / (sigma_max - sigma_floor)
    return float(max(0.0, min(1.0, conf)))


def derive_confidence(probabilities: list[float]) -> float:
    """Normalised max-probability: how concentrated the distribution is.

        (N * p_max - 1) / (N - 1)

    Uniform -> 0.0, one-hot -> 1.0, independent of N so thresholds mean the same
    thing for a 3-option and a 10-option question. For N == 1 the answer is forced,
    so confidence is 1.0 by definition.
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    p_max = max(probabilities)
    return float(max(0.0, min(1.0, (n * p_max - 1.0) / (n - 1.0))))
