"""Patterns: questions composed in one request, policy in code. ``decider_route``, ``decider_fan_out``.

``decider_route`` is the intent-routing pattern: one choice over the handlers plus one score
for how hard the request is, read back under a confidence floor with a person as the default.
``decider_fan_out`` is the general shape every pattern is built from: ask everything at
once, then let code keep the answers whose premise held.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from ..schema import Answer, Question
from ._common import (
    DEFAULT_CONFIDENCE_FLOOR,
    answer_to_dict,
    build_questions,
    choice_question,
    endpoint_error,
    error,
    expect_choice,
    expect_score,
    fraction,
    ok,
    render_state,
    resolve,
    score_question,
)
from .primitives import summarise

COMPLEXITY_LEVELS = [
    "a single clear request that one handler can finish in one step",
    "a request with one complication: a missing detail, a condition, or two steps",
    "several intertwined requests, an exception to policy, or an upset writer",
]
"""The default rubric behind ``decider_route``'s complexity score; index 2 escalates."""

ESCALATE_FROM_LEVEL = 2
"""A complexity score at or above this level sends the request to the fallback handler."""


@tool(context=True)
def decider_route(
    state: str | dict[str, Any] | list[Any],
    handlers: dict[str, str] | list[str] | str,
    question: str = "Which handler should take this request?",
    fallback: str = "human",
    confidence_floor: float | None = None,
    complexity_levels: list[str] | str | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Route a request to one handler, with a confidence floor and a complexity check.

    Use it in front of a dispatcher: which agent, queue, tool or team takes this. Two
    questions go out in one pass: a choice over the handlers and a score for how hard the
    request is. The route is the chosen handler when its confidence reaches
    ``confidence_floor`` and the request is not complex; otherwise it is ``fallback`` and
    ``reason`` says which test failed. Describe each handler by the requests it owns.

    Args:
        state: The request: text, or a JSON object such as ``{"message": ..., "history": ...}``.
        handlers: ``{"name": "the requests it owns", ...}``, a list of names, or a JSON string
            of either. ``fallback`` need not be among them.
        question: The routing question shown to the model.
        fallback: Where everything uncertain or complex goes (default ``human``).
        confidence_floor: Confidence at or above which the pick stands (default 0.6).
        complexity_levels: Optional ascending rubric replacing the default three levels; a
            score rounding to the top level escalates.
        context: Optional text folded in next to the state, such as the routing policy.

    Returns:
        JSON with ``route``, ``reason`` (``confident``, ``low_confidence`` or ``complex``),
        ``handler`` (the model's pick), ``confidence``, ``probabilities``, ``complexity``
        (``score``, ``level``, ``label``, ``confidence``) and ``model``; plus a summary.
    """
    try:
        pick = choice_question(question, handlers)
        levels = complexity_levels or COMPLEXITY_LEVELS
        hardness = score_question("How complex is this request to handle?", levels)
        floor = fraction(confidence_floor, "confidence_floor", DEFAULT_CONFIDENCE_FLOOR)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    questions: dict[str, Question] = {"handler": pick, "complexity": hardness}
    try:
        response = decider.ask(render_state(state, context), questions)
        chosen = expect_choice(response.answers, "handler")
        complexity = expect_score(response.answers, "complexity")
    except Exception as exc:
        return endpoint_error(exc)
    top = len(hardness.criteria) - 1
    level = max(0, min(top, round(complexity.score)))
    escalate_at = top if complexity_levels else min(ESCALATE_FROM_LEVEL, top)
    if chosen.confidence < floor:
        route, reason = fallback, "low_confidence"
    elif level >= escalate_at:
        route, reason = fallback, "complex"
    else:
        route, reason = chosen.choice, "confident"
    payload = {
        "route": route,
        "reason": reason,
        "handler": chosen.choice,
        "confidence": chosen.confidence,
        "confidence_floor": floor,
        "probabilities": dict(
            sorted(chosen.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "complexity": {
            "score": complexity.score,
            "level": level,
            "label": hardness.criteria[level],
            "confidence": complexity.confidence,
        },
        "model": response.model,
    }
    return ok(
        payload,
        f"-> {route} ({reason}; {chosen.choice} at {chosen.confidence:.2f}, complexity {complexity.score:.2f}/{top})",
    )


def _premise_holds(premise: Mapping[str, Any], answers: Mapping[str, Answer]) -> tuple[bool, str]:
    """Read a premise ``{"question": key, "equals": option}`` or ``{"question": key, "min": p}`` off the answers."""
    key = str(premise.get("question", ""))
    answer = answers.get(key)
    if answer is None:
        return False, f"premise names unknown question {key!r}"
    row = answer_to_dict(answer)
    if "equals" in premise:
        held = row.get("choice") == premise["equals"]
        return held, f"{key} = {row.get('choice')!r}, premise wanted {premise['equals']!r}"
    value = row.get("noul", row.get("score"))
    if value is None:
        return False, f"premise on {key!r} needs a noul or score answer"
    if "min" in premise:
        held = float(value) >= float(premise["min"])
        return held, f"{key} = {float(value):.2f}, premise wanted >= {float(premise['min']):.2f}"
    if "max" in premise:
        held = float(value) <= float(premise["max"])
        return held, f"{key} = {float(value):.2f}, premise wanted <= {float(premise['max']):.2f}"
    return False, f"premise on {key!r} needs equals, min or max"


@tool(context=True)
def decider_fan_out(
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, Any] | str,
    premises: dict[str, dict[str, Any]] | str | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Ask every question at once, including speculative ones, and keep the answers whose premise held.

    The questions cannot see each other, so a follow-up ("which plan?") is asked alongside
    the gate it depends on ("is this about billing?") in the same pass, and code applies the
    dependency afterwards. Name the dependency in ``premises``: an answer whose premise fails
    is reported under ``skipped`` with the reason, not thrown away silently. One pass over
    the state however many questions there are.

    Args:
        state: What the model reads: text, or a JSON object whose field names carry meaning.
        questions: A map of keys to question specs, as for ``decider_ask``.
        premises: Optional ``{"dependent_key": {"question": "gate_key", "min": 0.5}}``, or
            ``{"question": ..., "equals": "option"}`` for a choice gate, or ``"max"`` for an
            upper bound. A dependent question is kept only when its premise holds.
        context: Optional text folded in next to the state.

    Returns:
        JSON with ``answers`` (the questions that apply), ``skipped`` (key to reason),
        ``model`` and ``input_tokens``; plus a one-line summary.
    """
    try:
        built = build_questions(questions)
        rules = _premise_rules(premises, built)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    try:
        response = decider.ask(render_state(state, context), built)
    except Exception as exc:
        return endpoint_error(exc)
    kept: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for key in built:
        if key not in response.answers:
            continue
        rule = rules.get(key)
        if rule is not None:
            held, why = _premise_holds(rule, response.answers)
            if not held:
                skipped[key] = why
                continue
        kept[key] = answer_to_dict(response.answers[key])
    payload = {
        "answers": kept,
        "skipped": skipped,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
    }
    tail = f"; skipped {', '.join(skipped)}" if skipped else ""
    return ok(payload, summarise(kept) + tail)


def _premise_rules(premises: Any, built: Mapping[str, Question]) -> dict[str, Mapping[str, Any]]:
    if premises in (None, "", {}):
        return {}
    if isinstance(premises, str):
        try:
            premises = json.loads(premises)
        except ValueError as exc:
            raise ValueError(f"premises looks like JSON but does not parse: {exc}") from exc
    if not isinstance(premises, Mapping):
        raise ValueError("premises must be a dict of dependent question key to premise")
    rules: dict[str, Mapping[str, Any]] = {}
    for key, rule in premises.items():
        if key not in built:
            raise ValueError(f"premises names {key!r}, which is not among the questions")
        if not isinstance(rule, Mapping) or "question" not in rule:
            raise ValueError(f"premise for {key!r} must be a dict with a 'question' key")
        if rule["question"] not in built:
            raise ValueError(f"premise for {key!r} points at unknown question {rule['question']!r}")
        if rule["question"] == key:
            raise ValueError(f"premise for {key!r} cannot depend on itself")
        rules[str(key)] = rule
    return rules


PATTERN_TOOLS = [decider_route, decider_fan_out]
