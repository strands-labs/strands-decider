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

# Re-exported: prompting.py and callers import them from here.
from .confidence import derive_confidence as derive_confidence
from .confidence import derive_score_confidence as derive_score_confidence

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
