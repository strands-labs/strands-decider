"""Wire types for the System One API.

Follows the request and response shape of the public Jev API documentation. JevBench's
typesafe adapter runs against it unchanged. Compatibility with the Jev API itself is not
verified.

    POST /v1/systemone
    {"state": ..., "model": "strands-decider-latest", "questions": {"<name>": {...}}}

Three primitives, one mechanism. Every question is rendered into N options that the
prompt describes. The readout gives one logit per option, and a softmax over them gives
the distribution.
What differs is only how many slots there are and how the distribution is read back:

    noul   -> 2 slots (false, true)   -> P(true)
    choice -> N slots (one per option) -> argmax + per-option probabilities
    score  -> L slots (ordered levels) -> expected value over level indices
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# A `state` or `instructions` may be a bare string, or structured data that we
# render deterministically (see strands_decider.prompting.render_content).
Content = str | dict[str, Any] | list[Any]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class NoulQuestion(BaseModel):
    """Yes/no. Returns the probability the statement is true.

    No confidence field: with two outcomes the probability *is* the uncertainty,
    and the derived-confidence formula would just be |2p - 1|, telling you nothing new.
    """

    type: Literal["noul"] = "noul"
    instructions: Content
    # Optional {"true": "...", "false": "..."} descriptions that sharpen the boundary.
    criteria: dict[str, Content | None] | None = None

    @field_validator("criteria")
    @classmethod
    def _check_keys(cls, v: dict[str, Content | None] | None) -> dict[str, Content | None] | None:
        if v is not None and not set(v).issubset({"true", "false"}):
            raise ValueError("noul criteria keys must be a subset of {'true', 'false'}")
        return v


class ChoiceQuestion(BaseModel):
    """Pick one of N named options. `criteria` maps option name -> description.

    A description may be structured data (a chess move, a palette), rendered like `state`,
    or null for a bare label.
    """

    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: dict[str, Content | None]

    @field_validator("criteria")
    @classmethod
    def _check_size(cls, v: dict[str, Content | None]) -> dict[str, Content | None]:
        if len(v) < 2:
            raise ValueError("choice requires at least 2 options")
        if len(v) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice allows at most {MAX_CHOICE_OPTIONS} options")
        return v


class ScoreQuestion(BaseModel):
    """Rate against an ordered rubric. `criteria` is ascending: index 0 is the low end."""

    type: Literal["score"] = "score"
    instructions: Content
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def _check_levels(cls, v: list[str]) -> list[str]:
        if not (MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS):
            raise ValueError(
                f"score requires between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels"
            )
        return v


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion


class SystemOneRequest(BaseModel):
    # May be empty when the question carries the whole task (a quiz item, a pair to compare).
    state: Content
    questions: dict[str, Question] = Field(..., min_length=1)
    model: str = "strands-decider-latest"
    # Base64 images (or data: URIs), part of the state. Needs a vision engine
    # (`serve --vision`); a text-only engine refuses them rather than ignore them.
    images: list[str] = Field(default_factory=list)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    # {"0": "Calm", "1": "Frustrated", ...} -- what each level index meant.
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def derive_score_confidence(
    probabilities: list[float], *, ordinal_smoothing: float = 0.0
) -> float:
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
    automatically" threshold unreachable by construction -- which is exactly the bug
    this replaced.
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
    sigma = var ** 0.5

    sigma_max = (n - 1) / 2.0
    sigma_floor = ordinal_smoothing ** 0.5 if ordinal_smoothing > 0 else 0.0
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
