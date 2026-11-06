"""The System One request, parsed without pydantic.

The v2 request is v1's (`strands_decider.schema.SystemOneRequest`): a state and named
questions of type noul, choice or score. This module accepts and refuses what v1's pydantic
models accept and refuse (tests/test_v2_prompt.py checks it against them when pydantic is
installed), so that the torch-free core needs only the standard library to read a request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# v1's limits (schema.py). Repeated, not imported: schema.py needs pydantic.
MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
NOUL_KEYS = ("false", "true")
# What a noul's options mean when its criteria do not say (v1's prompting.py). A key given
# as null renders as a bare label instead.
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}

KINDS = ("noul", "choice", "score")


@dataclass(frozen=True)
class Question:
    """One question, normalised: `options` are the answer values in request order.

    noul: ("false", "true"), with `criteria` their descriptions: as given, or the defaults.
    choice: the option names, with `criteria` their descriptions (None for a bare label).
    score: the level indices "0".."L-1", with `criteria` the rubric text of each level.
    """

    kind: str
    instructions: Any
    options: tuple[str, ...]
    criteria: tuple[Any, ...]


@dataclass(frozen=True)
class Request:
    state: Any
    questions: dict[str, Question]
    model: str = "strands-decider-latest"


def _is_content(v: Any) -> bool:
    return isinstance(v, (str, dict, list))


def _infer_kind(q: Mapping[str, Any]) -> str:
    """v1 lets a question omit `type`; its union then picks the first model that fits."""
    crit = q.get("criteria")
    if crit is None or (isinstance(crit, dict) and set(crit) <= set(NOUL_KEYS)):
        return "noul"
    if isinstance(crit, dict):
        return "choice"
    if isinstance(crit, list):
        return "score"
    raise ValueError("cannot tell the question's type from its criteria; give `type`")


def parse_question(name: str, q: Any) -> Question:
    if not isinstance(q, Mapping):
        raise ValueError(f"question {name!r} must be an object")
    kind = q["type"] if "type" in q else _infer_kind(q)
    if kind not in KINDS:
        raise ValueError(f"question {name!r} has unknown type {kind!r}; expected one of {KINDS}")
    if "instructions" not in q:
        raise ValueError(f"question {name!r} has no instructions")
    instructions = q["instructions"]
    if not _is_content(instructions):
        raise ValueError(f"question {name!r}: instructions must be a string, object or array")
    crit = q.get("criteria")

    if kind == "noul":
        if crit is None:
            crit = {}
        if not isinstance(crit, dict) or not set(crit) <= set(NOUL_KEYS):
            raise ValueError(
                f"noul {name!r}: criteria keys must be a subset of {{'true', 'false'}}"
            )
        for v in crit.values():
            if v is not None and not _is_content(v):
                raise ValueError(f"noul {name!r}: a criterion must be text, data or null")
        merged = {**NOUL_DEFAULT_CRITERIA, **crit}
        return Question("noul", instructions, NOUL_KEYS, tuple(merged[k] for k in NOUL_KEYS))

    if kind == "choice":
        if not isinstance(crit, dict):
            raise ValueError(f"choice {name!r} needs criteria: an object of option -> description")
        if len(crit) < MIN_CHOICE_OPTIONS:
            raise ValueError(f"choice {name!r} requires at least {MIN_CHOICE_OPTIONS} options")
        if len(crit) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice {name!r} allows at most {MAX_CHOICE_OPTIONS} options")
        if not all(isinstance(k, str) for k in crit):
            raise ValueError(f"choice {name!r}: option names must be strings")
        for v in crit.values():
            if v is not None and not _is_content(v):
                raise ValueError(f"choice {name!r}: a description must be text, data or null")
        return Question("choice", instructions, tuple(crit), tuple(crit.values()))

    if not isinstance(crit, list) or not all(isinstance(v, str) for v in crit):
        raise ValueError(f"score {name!r} needs criteria: a list of level descriptions")
    if not MIN_SCORE_LEVELS <= len(crit) <= MAX_SCORE_LEVELS:
        raise ValueError(
            f"score {name!r} requires between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels"
        )
    return Question("score", instructions, tuple(str(i) for i in range(len(crit))), tuple(crit))


def parse_request(request: Any) -> Request:
    """A request from its JSON object, or from a v1 `SystemOneRequest` (anything with
    `model_dump`). Unknown fields are ignored, as v1 ignores them. Images are refused: the
    v2 core renders text only."""
    if hasattr(request, "model_dump"):
        request = request.model_dump()
    if not isinstance(request, Mapping):
        raise ValueError("a request must be an object")
    if "state" not in request:
        raise ValueError("a request needs a state (it may be empty)")
    state = request["state"]
    if not _is_content(state):
        raise ValueError("state must be a string, object or array")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("a request needs at least one question")
    if request.get("images"):
        raise ValueError("the v2 core renders text only; images are not supported")
    model = request.get("model", "strands-decider-latest")
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    parsed = {str(n): parse_question(str(n), q) for n, q in questions.items()}
    return Request(state=state, questions=parsed, model=model)
