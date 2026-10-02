"""One state, one question: ``decider_check`` (noul), ``decider_choose`` (choice), ``decider_rate`` (score).

Each is ``decider_ask`` with the question type fixed and the policy knob exposed as an
argument: a threshold for yes, a confidence floor for a decision. The verdict the tool
reports (``yes``/``no``, ``decided``/``leaning``) is code over the model's numbers, so the
agent can branch on a word while the probabilities stay in the result for anyone who wants
them.
"""

from __future__ import annotations

from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from ._common import (
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_NOUL_THRESHOLD,
    choice_question,
    endpoint_error,
    error,
    expect_choice,
    expect_noul,
    expect_score,
    fraction,
    noul_question,
    ok,
    render_state,
    resolve,
    score_question,
)


@tool(context=True)
def decider_check(
    state: str | dict[str, Any] | list[Any],
    question: str,
    criteria: dict[str, str] | str | None = None,
    threshold: float | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Ask one yes/no question about a piece of state and get the probability of yes.

    Use it for a single gate: is this urgent, does this answer the question, is this
    argument grounded in what the user said, is this safe to run. The probability is the
    confidence; ``verdict`` is ``yes`` when it reaches ``threshold``. Write ``criteria`` when
    the boundary matters: a one-line description of what makes the answer true and what
    makes it false sharpens the decision considerably.

    Args:
        state: What the model reads: text, or a JSON object whose field names carry meaning.
        question: The yes/no question, phrased so that "yes" is the thing you want to detect.
        criteria: Optional ``{"true": "...", "false": "..."}`` describing each side.
        threshold: Probability at or above which the verdict is ``yes`` (default 0.5).
        context: Optional text folded in next to the state, such as the user's request.

    Returns:
        JSON with ``verdict`` (``yes``/``no``), ``probability``, ``threshold`` and ``model``;
        plus a one-line summary.
    """
    try:
        built = noul_question(question, criteria)
        cut = fraction(threshold, "threshold", DEFAULT_NOUL_THRESHOLD)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    try:
        response = decider.ask(render_state(state, context), {"q": built})
        answer = expect_noul(response.answers, "q")
    except Exception as exc:
        return endpoint_error(exc)
    verdict = "yes" if answer.noul >= cut else "no"
    payload = {
        "verdict": verdict,
        "probability": answer.noul,
        "threshold": cut,
        "model": response.model,
    }
    return ok(payload, f"{verdict} (p={answer.noul:.2f}, threshold {cut:.2f})")


@tool(context=True)
def decider_choose(
    state: str | dict[str, Any] | list[Any],
    question: str,
    options: dict[str, str] | list[str] | str,
    confidence_floor: float | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Pick one option from a set you define, with a calibrated confidence.

    Use it for routing, labelling and intent: which team, which category, which tool, which
    of these candidates. Options are read from the request, not baked into the model, so
    change them freely; describe each one in a few words rather than relying on the label.
    Add an explicit escape option such as ``none_of_these`` when the state may match
    nothing. The result is ``decided`` when confidence reaches ``confidence_floor`` and a
    ``leaning`` otherwise: act on a decision, confirm a leaning.

    Args:
        state: What the model reads: text, or a JSON object whose field names carry meaning.
        question: What is being chosen, e.g. "Which team should handle this?".
        options: ``{"name": "when this option applies", ...}``, or a list of names, or a JSON
            string of either. Between 2 and 255 options.
        confidence_floor: Confidence at or above which the pick is a decision (default 0.6).
        context: Optional text folded in next to the state, such as the user's request.

    Returns:
        JSON with ``choice``, ``confidence``, ``verdict`` (``decided``/``leaning``),
        ``probabilities`` (every option), ``runner_up``, ``confidence_floor`` and ``model``;
        plus a one-line summary.
    """
    try:
        built = choice_question(question, options)
        floor = fraction(confidence_floor, "confidence_floor", DEFAULT_CONFIDENCE_FLOOR)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    try:
        response = decider.ask(render_state(state, context), {"q": built})
        answer = expect_choice(response.answers, "q")
    except Exception as exc:
        return endpoint_error(exc)
    ranked = sorted(answer.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    verdict = "decided" if answer.confidence >= floor else "leaning"
    payload = {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "verdict": verdict,
        "probabilities": dict(ranked),
        "runner_up": ranked[1][0] if len(ranked) > 1 else None,
        "confidence_floor": floor,
        "model": response.model,
    }
    return ok(payload, f"{answer.choice} ({verdict}, confidence {answer.confidence:.2f})")


@tool(context=True)
def decider_rate(
    state: str | dict[str, Any] | list[Any],
    question: str,
    levels: list[str] | str,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Rate a piece of state against an ordered rubric and get a position on the scale.

    Use it for degree: how urgent, how frustrated, how complete, how risky. Levels are
    ascending, index 0 the low end; the score is the probability-weighted position, so 1.4
    means "mostly level 1, leaning to level 2". Confidence measures how tightly the mass
    clusters, not how big the top probability is: a split between two neighbouring levels is
    a confident "in between", not doubt.

    Args:
        state: What the model reads: text, or a JSON object whose field names carry meaning.
        question: What is being rated, e.g. "How frustrated is the writer?".
        levels: Ascending level descriptions, 2 to 10 of them, as a list or a JSON string.
            Describe each level ("calm", "annoyed", "furious") rather than numbering them.
        context: Optional text folded in next to the state, such as the user's request.

    Returns:
        JSON with ``score``, ``level`` (the nearest level's index), ``label`` (its text),
        ``confidence``, ``legend``, ``probabilities`` and ``model``; plus a one-line summary.
    """
    try:
        built = score_question(question, levels)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    try:
        response = decider.ask(render_state(state, context), {"q": built})
        answer = expect_score(response.answers, "q")
    except Exception as exc:
        return endpoint_error(exc)
    nearest = round(answer.score)
    nearest = max(0, min(len(built.criteria) - 1, nearest))
    payload = {
        "score": answer.score,
        "level": nearest,
        "label": built.criteria[nearest],
        "confidence": answer.confidence,
        "legend": {int(k): v for k, v in answer.legend.items()},
        "probabilities": {int(k): v for k, v in answer.probabilities.items()},
        "model": response.model,
    }
    return ok(
        payload,
        f"{answer.score:.2f} of {len(built.criteria) - 1} ({built.criteria[nearest]}, confidence {answer.confidence:.2f})",
    )


SINGLE_TOOLS = [decider_check, decider_choose, decider_rate]
