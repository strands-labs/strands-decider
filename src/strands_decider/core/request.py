"""The System One request, as pydantic models.

The request is v1's (`strands_decider.schema.SystemOneRequest`): a state, optional images,
and named questions of type noul, choice or score. `parse_request` checks one, given as the
JSON object of `POST /v1/systemone` or as a v1 `SystemOneRequest`, and raises ValueError
saying what is wrong. Its images stay base64 until `images.decode_images` checks them, so a
runtime can refuse a request's images before it decodes any.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainValidator,
    ValidationError,
    model_validator,
)

# v1's limits (schema.py).
MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
NoulKey = Literal["false", "true"]
NOUL_KEYS: tuple[NoulKey, NoulKey] = ("false", "true")
# What a noul's options mean when its criteria do not say (v1's prompting.py). A key given
# as null renders as a bare label instead.
NOUL_DEFAULT_CRITERIA: dict[NoulKey, str] = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}
Kind = Literal["noul", "choice", "score"]
# How deep a request's content may nest: a scalar is level 0, and an object or an array is one
# level deeper than the deepest value in it. Indented JSON grows with the square of its depth,
# so deeper content would be a cheap way to make a server write megabytes.
MAX_DEPTH = 100


def _nests_within(value: Any, limit: int) -> bool:
    """Whether `value` nests at most `limit` levels deep, walked without recursion."""
    stack = [(value, 1)]
    while stack:
        v, depth = stack.pop()
        if isinstance(v, (dict, list)):
            if depth > limit:
                return False
            stack.extend((x, depth + 1) for x in (v.values() if isinstance(v, dict) else v))
    return True


def _check_json(value: Any) -> Any:
    """`value`, if it is JSON that renders alike everywhere: it nests at most MAX_DEPTH levels,
    and the text `prompt.render_content` writes for it reads back as `value`, encodes as
    UTF-8 and holds no NaN or Infinity. So numbers that are not finite, integers of more than
    4,300 digits (Python's limit), lone surrogates, object keys that are not strings and values
    that are not JSON are refused too. Every failure is a ValueError. A process that lifts
    Python's digit limit (PYTHONINTMAXSTRDIGITS=0) lifts that refusal."""
    if not _nests_within(value, MAX_DEPTH):
        raise ValueError(f"nested more than {MAX_DEPTH} levels deep")
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        same = json.loads(text) == value
        text.encode("utf-8")
    except (ValueError, TypeError, RecursionError) as e:  # UnicodeEncodeError is a ValueError
        raise ValueError(f"not JSON that renders alike everywhere: {e}") from None
    if not same:
        raise ValueError(
            "not JSON that renders alike everywhere: it reads back changed, as an object key "
            "that is not a string, or a tuple, does"
        )
    return value


def _content(value: Any) -> Any:
    if not isinstance(value, (str, dict, list)):
        raise ValueError("must be a string, object or array")
    return _check_json(value)


def _description(value: Any) -> Any:
    return None if value is None else _content(value)


def _text(value: Any) -> Any:
    if not isinstance(value, str):
        raise ValueError("must be a string")
    return _check_json(value)


# A state, an instruction or a description: a string, or structured data rendered as JSON.
Content = Annotated[str | dict[str, Any] | list[Any], PlainValidator(_content)]
# A description may also be null: the option renders as a bare label.
Description = Annotated[str | dict[str, Any] | list[Any] | None, PlainValidator(_description)]
# A choice's option name, or a score level's rubric text.
Text = Annotated[str, PlainValidator(_text)]


class Question(BaseModel):
    """One question, normalised: `options` are the answer values in request order, and
    `criteria` their descriptions, None for a bare label.

    noul: ("false", "true"), described as given or by the defaults.
    choice: the option names, with their descriptions.
    score: the level indices "0".."L-1", with the rubric text of each level.
    """

    model_config = ConfigDict(frozen=True)

    kind: Kind
    instructions: str | dict[str, Any] | list[Any]
    options: tuple[str, ...]
    criteria: tuple[str | dict[str, Any] | list[Any] | None, ...]


class _Model(BaseModel):
    # Values as JSON reads them, never coerced: a tuple is not an array, nor bytes a string.
    # Fields not declared here are ignored, as v1 ignores them.
    model_config = ConfigDict(strict=True, frozen=True)


class NoulQuestion(_Model):
    """Whether a statement holds of the state: the options `false` and `true`. `criteria`
    replaces either one's default description; a description given as null is a bare label."""

    type: Literal["noul"] = "noul"
    instructions: Content
    criteria: dict[NoulKey, Description] | None = None

    def normalised(self) -> Question:
        given = self.criteria or {}
        return Question(
            kind="noul",
            instructions=self.instructions,
            options=NOUL_KEYS,
            criteria=tuple(given.get(k, NOUL_DEFAULT_CRITERIA[k]) for k in NOUL_KEYS),
        )


class ChoiceQuestion(_Model):
    """One of 2 to 255 named options, in the order of `criteria`: option name to description."""

    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: dict[Text, Description] = Field(
        min_length=MIN_CHOICE_OPTIONS, max_length=MAX_CHOICE_OPTIONS
    )

    def normalised(self) -> Question:
        return Question(
            kind="choice",
            instructions=self.instructions,
            options=tuple(self.criteria),
            criteria=tuple(self.criteria.values()),
        )


class ScoreQuestion(_Model):
    """A rating on 2 to 10 ordered levels, `criteria` giving each one's rubric, lowest first."""

    type: Literal["score"] = "score"
    instructions: Content
    criteria: list[Text] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)

    def normalised(self) -> Question:
        return Question(
            kind="score",
            instructions=self.instructions,
            options=tuple(str(i) for i in range(len(self.criteria))),
            criteria=tuple(self.criteria),
        )


def _with_type(question: Any) -> Any:
    """v1 lets a question omit `type`, and its union then takes the first type whose criteria
    fit: noul when they are absent, null, or an object whose keys are among `false` and
    `true`; choice when they are any other object; score when they are an array."""
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
    """A System One request, checked. `images` are still base64: `images.decode_images` checks
    them. A non-null `audio` is refused, rather than ignored as other unknown fields are, so
    that a client never believes its audio was heard."""

    state: Content = ""
    questions: dict[str, AnyQuestion] = Field(min_length=1)
    model: str = "strands-decider-latest"
    images: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _no_audio(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("audio") is not None:
            raise ValueError("audio input is not supported")
        return data


def _message(e: ValidationError, most: int = 5) -> str:
    """The first `most` errors as `where: what`, `where` being the error's place in the
    request (`questions.q.noul.instructions`) with each part cut to 40 characters, as an
    option's name may be long, and `what` cut to 200, as it may quote the request; then how
    many more there are."""
    errors = e.errors(include_url=False, include_context=False, include_input=False)
    out = []
    for err in errors[:most]:
        what = err["msg"].removeprefix("Value error, ")[:200]
        where = ".".join(str(part)[:40] for part in err["loc"])
        out.append(f"{where}: {what}" if where else what)
    if len(errors) > most:
        out.append(f"and {len(errors) - most} more")
    return "; ".join(out)


def parse_request(request: dict[str, Any] | BaseModel) -> Request:
    """`request`, checked: the JSON object of `POST /v1/systemone`, as `json.loads` reads it,
    or a v1 `SystemOneRequest` (any pydantic model whose `model_dump()` is that object).
    ValueError, saying what is wrong, otherwise. The images are not decoded
    (`images.decode_images` checks them). A `Request` is returned as it is: it was checked
    when it was made, and its state, questions and images are not copies, so a change made
    to them since goes unchecked."""
    if isinstance(request, Request):
        return request
    data: object = request.model_dump() if isinstance(request, BaseModel) else request
    if not isinstance(data, dict):
        raise ValueError("a request must be a JSON object")
    try:
        return Request.model_validate(data)
    except ValidationError as e:
        raise ValueError(_message(e)) from None
