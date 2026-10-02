"""Shared plumbing for the tools: which decider answers, input tolerance, result shape.

Every tool returns ``{"status", "content": [{"json": ...}, {"text": one line}]}``: JSON for
code to branch on, one line for the model to read. Inputs are tolerated when the intent is
unambiguous (a JSON string where a list was meant, one item per line) and refused before any
forward pass when it is not (a threshold outside 0..1, a boolean where a number was meant).
Every error names the fix.

Which decider answers, in order:

1. ``invocation_state["decider"]``: per call, so one agent can carry several or a fake.
2. ``configure(decider)``: once, at startup.
3. The environment: ``STRANDS_DECIDER_URL`` (a server), else ``STRANDS_DECIDER_CHECKPOINT``
   (loaded once in this process on the first call).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from strands.types.tools import ToolContext

from ..schema import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Content,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from ._client import DeciderEndpoint, DeciderUnavailable, from_env

DECIDER_STATE_KEY = "decider"
"""Key in ``invocation_state`` under which a caller hands the tools an endpoint."""

MAX_QUESTIONS_PER_REQUEST = 128
"""Ceiling on one request. The context window is the real bound; this stops a runaway loop."""

MAX_ITEMS_PER_BATCH = 500
"""Most items a many-item tool judges in one call: one forward pass each."""

DEFAULT_MAX_STATE_CHARS = 12_000
"""Characters of state sent before truncating with a marker, about 3k tokens."""

DEFAULT_CONFIDENCE_FLOOR = 0.6
"""Below this a choice is a leaning, not a decision."""

DEFAULT_NOUL_THRESHOLD = 0.5
"""A noul at or above this reads as yes."""

_DEFAULT: DeciderEndpoint | None = None


def configure(decider: DeciderEndpoint | None) -> None:
    """Set the decider used when a call carries none. ``None`` clears it."""
    global _DEFAULT
    _DEFAULT = decider


def resolve(tool_context: ToolContext | None) -> DeciderEndpoint:
    """Find the decider for this call. See the module docstring for the order.

    Raises:
        TypeError: ``invocation_state["decider"]`` has no ``ask``.
        DeciderUnavailable: Nothing configured and nothing in the environment.
    """
    global _DEFAULT
    if tool_context is not None:
        supplied = (tool_context.invocation_state or {}).get(DECIDER_STATE_KEY)
        if supplied is not None:
            if not callable(getattr(supplied, "ask", None)):
                raise TypeError(
                    f"invocation_state[{DECIDER_STATE_KEY!r}] must be an HttpDecider, a LocalDecider "
                    f"or anything with an ask(state, questions) method, got {type(supplied).__name__}"
                )
            return supplied  # type: ignore[no-any-return]
    if _DEFAULT is None:
        _DEFAULT = from_env()
    return _DEFAULT


# ---------------------------------------------------------------------------- results


def ok(payload: Mapping[str, Any], summary: str) -> dict[str, Any]:
    """A successful result: JSON to branch on, one line to read."""
    return {"status": "success", "content": [{"json": dict(payload)}, {"text": summary}]}


def error(text: str) -> dict[str, Any]:
    """A failed result. The text names the fix."""
    return {"status": "error", "content": [{"text": text}]}


def endpoint_error(exc: Exception) -> dict[str, Any]:
    """Turn a client failure into a result the agent can act on."""
    if isinstance(exc, DeciderUnavailable):
        return error(f"The decider is not reachable: {exc}")
    if isinstance(exc, ValueError):
        return error(f"The decider refused the request: {exc}")
    return error(
        f"The decider did not answer ({type(exc).__name__}: {exc}). Decide without it or try once more."
    )


# ----------------------------------------------------------------------------- inputs


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters with a visible marker the model reads too."""
    if limit <= 0 or len(text) <= limit:
        return text
    marker = f" [truncated {len(text) - limit} characters]"
    keep = max(0, limit - len(marker))
    return text[:keep] + marker


def render_state(
    state: Any, context: str | None = None, max_chars: int = DEFAULT_MAX_STATE_CHARS
) -> Content:
    """Shape what the model reads: a string stays a string, anything else stays JSON.

    ``context`` is folded in under its own key so the caller can hand over the user's
    request or a policy once, next to the thing being judged.
    """
    if context:
        state = {"context": context, "subject": state}
    if isinstance(state, str):
        return truncate(state, max_chars)
    if not isinstance(state, (dict, list)):
        return truncate(str(state), max_chars)
    rendered = json.dumps(state, default=str, ensure_ascii=False)
    if max_chars > 0 and len(rendered) > max_chars:
        return truncate(rendered, max_chars)
    return state


def as_list(value: Any, what: str, *, ceiling: int = MAX_ITEMS_PER_BATCH) -> list[Any]:
    """Read a list the caller may have sent in a less convenient shape.

    A JSON string holding a list is decoded; a multi-line string becomes one item per
    non-empty line; any other string is a single item.

    Raises:
        ValueError: Nothing usable, or more than ``ceiling`` items.
    """
    items: list[Any]
    if isinstance(value, str):
        text = value.strip()
        decoded: Any = None
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except ValueError:
                decoded = None
        if isinstance(decoded, list):
            items = decoded
        elif "\n" in text:
            items = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            items = [text] if text else []
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]
    if not items:
        raise ValueError(f"{what} is empty; supply at least one")
    if len(items) > ceiling:
        raise ValueError(
            f"{what} has {len(items)} items, over the ceiling of {ceiling}; send it in batches"
        )
    return items


def as_mapping(value: Any, what: str) -> dict[str, Content | None]:
    """Read named options: a mapping, a list of strings, or a JSON string of either.

    A list makes each string its own name and description: the label is the meaning. A
    mapping's descriptions pass through as given (text, structured, or none).

    Raises:
        ValueError: Nothing usable, JSON that does not parse, or an empty name.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"{what} looks like JSON but does not parse: {exc}") from exc
    result: dict[str, Content | None]
    if isinstance(value, Mapping):
        result = {str(key): description for key, description in value.items()}
    else:
        result = {str(item): str(item) for item in as_list(value, what)}
    if not result:
        raise ValueError(f"{what} is empty; supply at least one")
    for key in result:
        if not key.strip():
            raise ValueError(f"{what} has an empty name")
    return result


def fraction(value: Any, name: str, default: float) -> float:
    """A number in 0..1, or the default when omitted. Refused, never clamped.

    Raises:
        ValueError: A boolean, not a number, or outside 0..1.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a number between 0 and 1, got a boolean; write {name}=0.5"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1, got {value!r}") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {number}")
    return number


def whole(value: Any, name: str, default: int, low: int, high: int) -> int:
    """A whole number in ``low..high``, or the default when omitted. Refused, never clamped."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number, got a boolean")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from exc
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {number}")
    return number


def preview(item: Any, limit: int = 120) -> str:
    """A short handle on an item for a result table."""
    text = item if isinstance(item, str) else json.dumps(item, default=str, ensure_ascii=False)
    return truncate(text, limit)


# --------------------------------------------------------------------------- questions

QUESTION_TYPES = {"noul", "choice", "score"}


def build_question(spec: Any, name: str) -> Question:
    """Turn a caller's question spec into one of the package's question models.

    Accepts a question model as-is, or a dict with ``type`` (``noul``, ``choice``,
    ``score``), ``instructions`` (string or structured; ``question`` is accepted as an alias)
    and ``criteria`` (noul: optional ``{"true": ..., "false": ...}``; choice: option map or
    list, ``options`` accepted; score: ordered level list, ``levels`` accepted). A plain string
    is a noul whose instructions are the string.

    Raises:
        ValueError: Missing type, bad criteria shape, or over a documented limit.
    """
    if isinstance(spec, (NoulQuestion, ChoiceQuestion, ScoreQuestion)):
        return spec
    if isinstance(spec, str):
        return NoulQuestion(instructions=spec)
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"question {name!r} must be a string, a question model or a dict, got {type(spec).__name__}"
        )
    kind = str(spec.get("type", "")).lower()
    instructions = spec.get("instructions", spec.get("question"))
    criteria = spec.get("criteria", spec.get("options", spec.get("levels")))
    if kind not in QUESTION_TYPES:
        raise ValueError(
            f"question {name!r} needs type noul, choice or score, got {spec.get('type')!r}"
        )
    if instructions is None or instructions == "":
        raise ValueError(f"question {name!r} has no instructions")
    try:
        if kind == "noul":
            if criteria is None:
                return NoulQuestion(instructions=instructions)
            if not isinstance(criteria, Mapping):
                raise ValueError("noul criteria must be a dict with only true and false keys")
            return NoulQuestion(
                instructions=instructions, criteria={str(k): str(v) for k, v in criteria.items()}
            )
        if kind == "choice":
            if criteria is None:
                raise ValueError("choice needs options as a dict or a list")
            return ChoiceQuestion(
                instructions=instructions, criteria=as_mapping(criteria, "options")
            )
        if not isinstance(criteria, Sequence) or isinstance(criteria, str):
            raise ValueError("score needs an ordered list of level descriptions")
        return ScoreQuestion(instructions=instructions, criteria=[str(level) for level in criteria])
    except ValueError as exc:
        raise ValueError(f"question {name!r}: {_first_line(exc)}") from exc


def _first_line(exc: Exception) -> str:
    """Pydantic's message, without its header and URL lines."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Value error, "):
            return line[len("Value error, ") :].split(" [type=")[0]
    return lines[-1] if lines else str(exc)


def build_questions(specs: Any) -> dict[str, Question]:
    """A named question map from a dict, a list (named q1..N) or a JSON string of either.

    Raises:
        ValueError: Nothing usable, or more than ``MAX_QUESTIONS_PER_REQUEST``.
    """
    if isinstance(specs, str):
        text = specs.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                specs = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"questions looks like JSON but does not parse: {exc}") from exc
        else:
            specs = [text]
    named: dict[str, Any]
    if isinstance(specs, Mapping):
        named = dict(specs)
    else:
        named = {
            f"q{index}": spec for index, spec in enumerate(as_list(specs, "questions"), start=1)
        }
    if not named:
        raise ValueError("questions is empty; supply at least one")
    if len(named) > MAX_QUESTIONS_PER_REQUEST:
        raise ValueError(
            f"{len(named)} questions in one request, over the ceiling of {MAX_QUESTIONS_PER_REQUEST}; split them"
        )
    return {str(name): build_question(spec, str(name)) for name, spec in named.items()}


def choice_question(instructions: Content, options: Any) -> ChoiceQuestion:
    """A choice over ``options`` (mapping, list or JSON string), limits checked by the schema."""
    named = as_mapping(options, "options")
    if len(named) > MAX_CHOICE_OPTIONS:
        raise ValueError(f"{len(named)} options, over the limit of {MAX_CHOICE_OPTIONS}")
    return ChoiceQuestion(instructions=instructions, criteria=named)


def score_question(instructions: Content, levels: Any) -> ScoreQuestion:
    """A score over ``levels`` (ascending list or JSON string), limits checked by the schema."""
    listed = [str(level) for level in as_list(levels, "levels", ceiling=MAX_SCORE_LEVELS + 1)]
    if not MIN_SCORE_LEVELS <= len(listed) <= MAX_SCORE_LEVELS:
        raise ValueError(
            f"{len(listed)} levels, needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS}"
        )
    return ScoreQuestion(instructions=instructions, criteria=listed)


def noul_question(instructions: Content, criteria: Any = None) -> NoulQuestion:
    """A noul with optional ``{"true": ..., "false": ...}`` criteria."""
    if criteria in (None, "", {}):
        return NoulQuestion(instructions=instructions)
    if isinstance(criteria, str):
        try:
            criteria = json.loads(criteria)
        except ValueError as exc:
            raise ValueError(f"criteria looks like JSON but does not parse: {exc}") from exc
    if not isinstance(criteria, Mapping):
        raise ValueError("criteria must be a dict with true and/or false keys")
    return NoulQuestion(
        instructions=instructions, criteria={str(k): str(v) for k, v in criteria.items()}
    )


# ----------------------------------------------------------------------------- answers


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    """A JSON-ready view of any answer."""
    row = answer.model_dump()
    if isinstance(answer, ScoreAnswer):
        row["legend"] = {int(k): v for k, v in answer.legend.items()}
        row["probabilities"] = {int(k): v for k, v in answer.probabilities.items()}
    return row


def expect_noul(answers: Mapping[str, Answer], key: str) -> NoulAnswer:
    answer = answers.get(key)
    if not isinstance(answer, NoulAnswer):
        raise ValueError(f"expected a noul answer for {key!r}, got {type(answer).__name__}")
    return answer


def expect_choice(answers: Mapping[str, Answer], key: str) -> ChoiceAnswer:
    answer = answers.get(key)
    if not isinstance(answer, ChoiceAnswer):
        raise ValueError(f"expected a choice answer for {key!r}, got {type(answer).__name__}")
    return answer


def expect_score(answers: Mapping[str, Answer], key: str) -> ScoreAnswer:
    answer = answers.get(key)
    if not isinstance(answer, ScoreAnswer):
        raise ValueError(f"expected a score answer for {key!r}, got {type(answer).__name__}")
    return answer


def ask_each(
    decider: DeciderEndpoint, states: Sequence[Content], questions: Mapping[str, Question]
) -> list[Mapping[str, Answer] | Exception]:
    """One request per state, in order. Errors are kept per item so one bad row cannot sink the batch."""
    results: list[Mapping[str, Answer] | Exception] = []
    for state in states:
        try:
            results.append(decider.ask(state, questions).answers)
        except Exception as exc:
            results.append(exc)
    return results
