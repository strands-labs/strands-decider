"""Primitives: the raw API as three tools.

``decider_ask`` is the whole of ``POST /v1/systemone``: one state, a map of mixed typed
questions, one forward pass over the state, typed answers back. Every other tool in this
package is a specialisation of it with the questions written for you. ``decider_info``
reports what is loaded; ``decider_usage`` reads the tally the client keeps.
"""

from __future__ import annotations

from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from ._common import (
    answer_to_dict,
    build_questions,
    endpoint_error,
    error,
    ok,
    render_state,
    resolve,
)


def summarise(rows: dict[str, dict[str, Any]]) -> str:
    """One line over a map of answer rows: ``key=value (confidence)``."""
    parts = []
    for key, row in rows.items():
        if row["type"] == "noul":
            parts.append(f"{key}={row['noul']:.2f}")
        elif row["type"] == "choice":
            parts.append(f"{key}={row['choice']} ({row['confidence']:.2f})")
        else:
            parts.append(f"{key}={row['score']:.2f} ({row['confidence']:.2f})")
    return f"{len(rows)} answers: " + ", ".join(parts)


@tool(context=True)
def decider_ask(
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, Any] | list[Any] | str,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Ask the decision model any mix of typed questions about one piece of state, in one pass.

    This is the raw System One call. Use it when no specialised tool fits: write the
    questions yourself. Three question types: ``noul`` is a yes/no question that returns the
    probability of yes; ``choice`` picks one option from a set you define (up to 255) and
    returns the option, its confidence and every option's probability; ``score`` rates the
    state on an ordered scale of 2 to 10 levels you describe and returns a
    probability-weighted position with index 0 the first level. The state is encoded once
    and every question reads it in parallel without seeing the others, so ask everything at
    once. The model reads literally: it does not count, do arithmetic, compare dates or
    follow indirection, and it cannot generate text. Confidence at or above 0.9 is right
    about 95% of the time on short classification; below that, confirm or ask a person.

    Args:
        state: What the model reads. Text, or a JSON object whose field names carry meaning
            (``{"ticket": ..., "policy": ...}``).
        questions: A map of your own keys to question specs. A spec is
            ``{"type": "noul", "instructions": "...", "criteria": {"true": ..., "false": ...}}``
            (criteria optional), ``{"type": "choice", "instructions": "...", "criteria":
            {"option": "when it applies", ...}}`` or ``{"type": "score", "instructions":
            "...", "criteria": ["level 0", "level 1", ...]}``. A plain string is a noul. A list
            is named q1..qN. A JSON string of either shape is accepted. Keys are not shown to
            the model.
        context: Optional text folded in next to the state, such as the user's request.

    Returns:
        JSON with ``answers`` (per key: ``noul``, or ``choice`` + ``confidence`` +
        ``probabilities``, or ``score`` + ``confidence`` + ``legend`` + ``probabilities``),
        ``model``, ``latency_ms`` and ``input_tokens``; plus a one-line summary.
    """
    try:
        built = build_questions(questions)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    try:
        response = decider.ask(render_state(state, context), built)
    except Exception as exc:
        return endpoint_error(exc)
    rows = {key: answer_to_dict(response.answers[key]) for key in built if key in response.answers}
    usage = getattr(decider, "usage", None)
    payload = {
        "answers": rows,
        "model": response.model,
        "latency_ms": getattr(usage, "last_latency_ms", None),
        "input_tokens": response.usage.input_tokens,
    }
    return ok(payload, summarise(rows))


@tool(context=True)
def decider_info(tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Report which decision model is answering: checkpoint, base model, window and device.

    Call this once to learn the context window (``max_length``, in tokens) before sending
    long states, and to confirm a server or checkpoint is reachable at all.

    Returns:
        JSON with ``endpoint`` and the health rows (``model``, ``checkpoint``, ``base_model``,
        ``num_slots``, ``max_length``, ``temperature``, ``device``, ``prefix_cache``); plus a
        one-line summary.
    """
    try:
        decider = resolve(tool_context)
    except (TypeError, RuntimeError) as exc:
        return endpoint_error(exc)
    health = getattr(decider, "health", None)
    if not callable(health):
        return ok({"endpoint": repr(decider)}, f"decider: {decider!r} (no health report)")
    try:
        rows = dict(health())
    except Exception as exc:
        return endpoint_error(exc)
    rows["endpoint"] = repr(decider)
    return ok(
        rows,
        f"{rows.get('model')} on {rows.get('base_model')} [{rows.get('device')}, window {rows.get('max_length')}]",
    )


@tool(context=True)
def decider_usage(reset: bool = False, tool_context: ToolContext | None = None) -> dict[str, Any]:
    """Report what the decision-model calls in this process have used so far.

    Args:
        reset: Zero the counters after reading them.

    Returns:
        JSON with ``calls``, ``questions``, ``input_tokens``, ``output_tokens``, ``errors``,
        ``latency_ms_mean``, ``latency_ms_total``, ``last_model`` and ``last_latency_ms``;
        plus a one-line summary.
    """
    try:
        decider = resolve(tool_context)
    except (TypeError, RuntimeError) as exc:
        return endpoint_error(exc)
    usage = getattr(decider, "usage", None)
    if usage is None or not hasattr(usage, "as_dict"):
        return error("This endpoint keeps no usage tally; HttpDecider and LocalDecider do.")
    payload = usage.as_dict()
    summary = str(usage)
    if reset and hasattr(usage, "reset"):
        usage.reset()
    return ok(payload, summary)


PRIMITIVE_TOOLS = [decider_ask, decider_info, decider_usage]
