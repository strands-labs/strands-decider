"""The System One request, parsed without pydantic.

The request is v1's (`strands_decider.schema.SystemOneRequest`): a state, optional images,
and named questions of type noul, choice or score. This module accepts and refuses what
v1's pydantic models accept and refuse (tests/test_core_prompt.py checks it against them
when pydantic is installed), and checks images as v1's image engine checks them
(`images.decode_image`), so that the torch-free core needs only the standard library to
read a request.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .images import MAX_IMAGES, EncodedImage, decode_image

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
# Request content is JSON, rendered by docs/spec.md's rules (Rendering content). Beyond v1's
# checks the core refuses what those rules cannot render the same way everywhere: nesting
# past MAX_DEPTH, non-finite numbers, integers past MAX_INT_DIGITS digits (Python's own
# limit for int-to-str, from 3.10.7), lone surrogates, and anything that is not JSON.
MAX_DEPTH = 100
MAX_INT_DIGITS = 4300
_INT_LIMIT = 10**MAX_INT_DIGITS
_SURROGATE = re.compile("[\ud800-\udfff]")
# v1's text-only engine refuses images with this message (hidden_engine.HiddenStateEngine);
# `runtime.evaluate` gives it for a backend without an image path.
NO_VISION = "this engine has no vision tower; images are not supported"
NO_AUDIO = "audio input is not supported"


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
    """A request, normalised. `images` are the request's images in order, checked and sized
    from their headers; they go first in the state block."""

    state: Any
    questions: dict[str, Question]
    model: str = "strands-decider-latest"
    images: tuple[EncodedImage, ...] = ()


def _is_content(v: Any) -> bool:
    return isinstance(v, (str, dict, list))


def check_json(value: Any, where: str) -> None:
    """That `value` is JSON the template renders alike everywhere (docs/spec.md, Rendering
    content): strings without lone surrogates, finite numbers, integers of at most 4,300
    digits, objects with string keys, and nesting at most 100 levels deep (an object or an
    array is one level deeper than the deepest value in it). ValueError, naming `where`,
    otherwise."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        v, depth = stack.pop()
        if isinstance(v, str):
            if _SURROGATE.search(v):
                raise ValueError(
                    f"{where}: a string holds a lone surrogate (U+D800 to U+DFFF), which has "
                    "no UTF-8 form"
                )
        elif v is None or isinstance(v, bool):
            pass
        elif isinstance(v, int):
            if abs(v) >= _INT_LIMIT:
                raise ValueError(f"{where}: an integer has more than {MAX_INT_DIGITS:,} digits")
        elif isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(f"{where}: {v!r} is not a JSON number; numbers must be finite")
        elif isinstance(v, (dict, list)):
            if depth > MAX_DEPTH:
                raise ValueError(f"{where}: JSON nested more than {MAX_DEPTH} levels deep")
            if isinstance(v, dict):
                for k in v:
                    if not isinstance(k, str):
                        raise ValueError(f"{where}: object keys must be strings, not {k!r}")
                    stack.append((k, depth))
                stack.extend((x, depth + 1) for x in v.values())
            else:
                stack.extend((x, depth + 1) for x in v)
        else:
            raise ValueError(f"{where}: a {type(v).__name__} is not a JSON value")


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
    question = _parse_question(name, kind, instructions, crit)
    check_json(question.instructions, f"question {name!r} instructions")
    for option, description in zip(question.options, question.criteria, strict=True):
        where = f"question {name!r} option {option[:40]!r}"  # a name may be long
        check_json(option, where)
        check_json(description, where)
    return question


def _parse_question(name: str, kind: str, instructions: Any, crit: Any) -> Question:
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


def _as_mapping(request: Any) -> Any:
    return request.model_dump() if hasattr(request, "model_dump") else request


def has_images(request: Any) -> bool:
    """Whether the request carries a non-empty `images` list, before anything is checked.
    A runtime that cannot take images checks the rest of the request first
    (`parse_request(request, images=False)`), then refuses one with images before it decodes
    them, as v1's text-only engine does after v1's validation."""
    request = _as_mapping(request)
    return (
        isinstance(request, Mapping)
        and isinstance(request.get("images"), list)
        and bool(request["images"])
    )


def parse_request(request: Any, *, images: bool = True) -> Request:
    """A request from its JSON object, or from a v1 `SystemOneRequest` (anything with
    `model_dump`). Unknown fields are ignored, as v1 ignores them.

    `images`, as in v1, is optional and must be a list of strings; null or any other type is
    refused. Then, as v1's image engine checks them: at most 4, each base64 (behind an
    optional `data:image/...;base64,` prefix) of a PNG, JPEG, WebP or GIF file of at most
    4096 x 4096 pixels, read from its header (`images.decode_image`), every check of one
    image before the next image's. Every other check comes first, as v1 validates the
    request before its engine looks at the images. With `images=False` the images are
    neither counted nor decoded, and the request returned has none: a runtime that cannot
    take images checks the rest so, and then refuses the images (`runtime.evaluate`)."""
    request = _as_mapping(request)
    if not isinstance(request, Mapping):
        raise ValueError("a request must be an object")
    if "state" not in request:
        raise ValueError("a request needs a state (it may be empty)")
    state = request["state"]
    if not _is_content(state):
        raise ValueError("state must be a string, object or array")
    check_json(state, "state")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("a request needs at least one question")
    raw_images = request.get("images", [])
    if not isinstance(raw_images, list) or not all(isinstance(i, str) for i in raw_images):
        raise ValueError("images must be a list of strings (base64 or data: URIs)")
    # Refused rather than ignored, so a client never believes its audio was heard.
    if request.get("audio"):
        raise ValueError(NO_AUDIO)
    model = request.get("model", "strands-decider-latest")
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    parsed = {str(n): parse_question(str(n), q) for n, q in questions.items()}
    if not images:
        return Request(state=state, questions=parsed, model=model)
    # After the schema's checks, as v1's image engine checks images after validation.
    if len(raw_images) > MAX_IMAGES:
        raise ValueError(f"{len(raw_images)} images; this server takes at most {MAX_IMAGES}")
    decoded = tuple(decode_image(i) for i in raw_images)
    return Request(state=state, questions=parsed, model=model, images=decoded)
