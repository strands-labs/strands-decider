"""The System One request: a state, optional base64 images, and named noul, choice or score questions.

Servers build it with `Request.model_validate_json(body)`. pydantic's JSON parser refuses nesting
deeper than 200 levels and lone surrogates, so rendered content is always UTF-8 and bounded in depth.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

NoulKey = Literal["false", "true"]
NOUL_KEYS: tuple[NoulKey, NoulKey] = ("false", "true")
# What a noul's options say when its criteria don't describe them: v1's defaults.
NOUL_DEFAULT_CRITERIA: dict[NoulKey, str] = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}
Kind = Literal["noul", "choice", "score"]

# A state, an instruction or a description: a string, or structured data rendered as JSON.
Content = str | dict[str, Any] | list[Any]


class Question(BaseModel):
    """A question, normalised: its options in request order (noul: false, true; score: "0".."L-1")
    and their descriptions, None for a bare label."""

    model_config = ConfigDict(frozen=True)

    kind: Kind
    instructions: Content
    options: tuple[str, ...]
    criteria: tuple[Content | None, ...]


class _Model(BaseModel):
    # Values as JSON gives them, never coerced. Undeclared fields are ignored.
    model_config = ConfigDict(strict=True, frozen=True)


class NoulQuestion(_Model):
    type: Literal["noul"] = "noul"
    instructions: Content
    criteria: dict[NoulKey, Content | None] | None = None

    def normalised(self) -> Question:
        given = self.criteria or {}
        return Question(
            kind="noul",
            instructions=self.instructions,
            options=NOUL_KEYS,
            criteria=tuple(given.get(k, NOUL_DEFAULT_CRITERIA[k]) for k in NOUL_KEYS),
        )


class ChoiceQuestion(_Model):
    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: dict[str, Content | None] = Field(min_length=2, max_length=255)  # v1's limits

    def normalised(self) -> Question:
        return Question(
            kind="choice",
            instructions=self.instructions,
            options=tuple(self.criteria),
            criteria=tuple(self.criteria.values()),
        )


class ScoreQuestion(_Model):
    """2 to 10 ordered levels (v1's limits), lowest first, each given by its rubric."""

    type: Literal["score"] = "score"
    instructions: Content
    criteria: list[str] = Field(min_length=2, max_length=10)

    def normalised(self) -> Question:
        return Question(
            kind="score",
            instructions=self.instructions,
            options=tuple(str(i) for i in range(len(self.criteria))),
            criteria=tuple(self.criteria),
        )


def _with_type(question: Any) -> Any:
    """A question may omit `type`, as in v1; it is then inferred from its criteria: noul when they
    are absent or keyed by false/true, choice for any other object, score for a list."""
    if not isinstance(question, dict) or "type" in question:
        return question
    criteria = question.get("criteria")
    if criteria is None or (isinstance(criteria, dict) and set(criteria) <= set(NOUL_KEYS)):
        kind = "noul"
    elif isinstance(criteria, dict):
        kind = "choice"
    elif isinstance(criteria, list):
        kind = "score"
    else:
        raise ValueError("cannot tell the question's type from its criteria; give `type`")
    return {**question, "type": kind}


AnyQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
    BeforeValidator(_with_type),
]


class Request(_Model):
    """A System One request. `images` stay base64 until `render` checks them (`images.decode_image`)."""

    state: Content = ""
    questions: dict[str, AnyQuestion] = Field(min_length=1)
    model: str = "strands-decider-latest"
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _no_audio(cls, data: Any) -> Any:
        # Refused rather than ignored, so a client never believes its audio was heard.
        if isinstance(data, dict) and data.get("audio") is not None:
            raise ValueError("audio input is not supported")
        return data
