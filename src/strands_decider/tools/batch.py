"""Many items, one question each: ``decider_sift``, ``decider_classify``, ``decider_rank``.

The model answers one item at a time, one forward pass per item, and the code does the
part the model cannot: counting, sorting, keeping. A failed item is reported in place
rather than sinking the batch.
"""

from __future__ import annotations

from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from ._common import (
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_NOUL_THRESHOLD,
    as_list,
    ask_each,
    choice_question,
    endpoint_error,
    error,
    expect_choice,
    expect_noul,
    expect_score,
    fraction,
    noul_question,
    ok,
    preview,
    render_state,
    resolve,
    score_question,
)


def _failed(index: int, item: Any, exc: Exception) -> dict[str, Any]:
    return {"index": index, "item": preview(item), "error": f"{type(exc).__name__}: {exc}"}


@tool(context=True)
def decider_sift(
    items: list[Any] | str,
    question: str,
    criteria: dict[str, str] | str | None = None,
    threshold: float | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Keep the items for which a yes/no question is true; the count is done in code.

    Use it to filter: which of these log lines are errors, which messages need a reply,
    which candidates meet the brief. Each item is judged on its own, with ``context`` (the
    brief, the policy) folded in beside it. Up to 500 items per call.

    Args:
        items: The items to judge: a list (strings or JSON objects), a JSON string holding a
            list, or one item per line.
        question: The yes/no question asked of each item, phrased so "yes" means keep.
        criteria: Optional ``{"true": "...", "false": "..."}`` describing each side.
        threshold: Probability at or above which an item is kept (default 0.5).
        context: Optional text folded in next to every item, such as the user's request.

    Returns:
        JSON with ``kept`` and ``dropped`` (each a list of ``{index, item, probability}``,
        most probable first), ``count`` (kept), ``total``, ``failed`` and ``threshold``; plus a
        one-line summary.
    """
    try:
        listed = as_list(items, "items")
        built = noul_question(question, criteria)
        cut = fraction(threshold, "threshold", DEFAULT_NOUL_THRESHOLD)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    states = [render_state(item, context) for item in listed]
    results = ask_each(decider, states, {"q": built})
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for index, (item, result) in enumerate(zip(listed, results, strict=True)):
        if isinstance(result, Exception):
            failed.append(_failed(index, item, result))
            continue
        try:
            p = expect_noul(result, "q").noul
        except ValueError as exc:
            failed.append(_failed(index, item, exc))
            continue
        row = {"index": index, "item": item, "probability": p}
        (kept if p >= cut else dropped).append(row)
    kept.sort(key=lambda r: r["probability"], reverse=True)
    dropped.sort(key=lambda r: r["probability"], reverse=True)
    payload = {
        "kept": kept,
        "dropped": dropped,
        "count": len(kept),
        "total": len(listed),
        "failed": failed,
        "threshold": cut,
    }
    tail = f", {len(failed)} failed" if failed else ""
    return ok(payload, f"kept {len(kept)} of {len(listed)} (threshold {cut:.2f}{tail})")


@tool(context=True)
def decider_classify(
    items: list[Any] | str,
    question: str,
    options: dict[str, str] | list[str] | str,
    confidence_floor: float | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Label each item with one option from a set you define, grouped by label.

    Use it to bucket: tickets by team, feedback by theme, lines by kind. Items whose
    confidence falls below ``confidence_floor`` are listed under ``uncertain`` instead of a
    label, so the agent can confirm them rather than file them wrong. Add an escape option
    such as ``other`` when some items may fit nothing. Up to 500 items per call.

    Args:
        items: The items to label: a list, a JSON string holding a list, or one per line.
        question: What is being decided for each item, e.g. "Which team should handle this?".
        options: ``{"name": "when it applies", ...}``, a list of names, or a JSON string of
            either. Between 2 and 255 options.
        confidence_floor: Confidence at or above which a label is kept (default 0.6).
        context: Optional text folded in next to every item.

    Returns:
        JSON with ``groups`` (label to list of ``{index, item, confidence}``), ``counts``
        (label to count, every option present), ``uncertain`` (``{index, item, choice,
        confidence}``), ``failed``, ``total`` and ``confidence_floor``; plus a summary.
    """
    try:
        listed = as_list(items, "items")
        built = choice_question(question, options)
        floor = fraction(confidence_floor, "confidence_floor", DEFAULT_CONFIDENCE_FLOOR)
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    states = [render_state(item, context) for item in listed]
    results = ask_each(decider, states, {"q": built})
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in built.criteria}
    uncertain: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for index, (item, result) in enumerate(zip(listed, results, strict=True)):
        if isinstance(result, Exception):
            failed.append(_failed(index, item, result))
            continue
        try:
            answer = expect_choice(result, "q")
        except ValueError as exc:
            failed.append(_failed(index, item, exc))
            continue
        row = {"index": index, "item": item, "confidence": answer.confidence}
        if answer.confidence >= floor:
            groups.setdefault(answer.choice, []).append(row)
        else:
            uncertain.append({**row, "choice": answer.choice})
    counts = {name: len(rows) for name, rows in groups.items()}
    payload = {
        "groups": groups,
        "counts": counts,
        "uncertain": uncertain,
        "failed": failed,
        "total": len(listed),
        "confidence_floor": floor,
    }
    shown = ", ".join(f"{name}={n}" for name, n in counts.items() if n)
    tail = f"; {len(uncertain)} uncertain" if uncertain else ""
    tail += f"; {len(failed)} failed" if failed else ""
    return ok(payload, f"{len(listed)} items: {shown or 'none labelled'}{tail}")


@tool(context=True)
def decider_rank(
    items: list[Any] | str,
    question: str,
    levels: list[str] | str | None = None,
    top_k: int | None = None,
    context: str | None = None,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Order items by how well each answers a question, best first; the sort is done in code.

    Use it to rerank: search hits against a query, candidates against a brief, drafts
    against a rubric. Without ``levels`` each item gets a yes/no probability (does this
    item satisfy the question?); with ``levels`` each item gets a score on your ascending
    rubric, which separates a crowded field better. Put the query or brief in ``context``
    so every item is judged against the same text. Up to 500 items per call.

    Args:
        items: The items to order: a list, a JSON string holding a list, or one per line.
        question: What "better" means, e.g. "Does this passage answer the query?".
        levels: Optional ascending rubric of 2 to 10 levels for a finer ordering.
        top_k: Return only the best ``top_k`` rows (default all).
        context: The query, brief or rubric every item is judged against.

    Returns:
        JSON with ``ranked`` (``{rank, index, item, value, confidence}`` best first, where
        ``value`` is the probability or the score), ``failed``, ``total`` and ``scale``
        (``probability`` or ``score``); plus a one-line summary.
    """
    try:
        listed = as_list(items, "items")
        question_model: Any = (
            score_question(question, levels) if levels else noul_question(question)
        )
        k = whole_top_k(top_k, len(listed))
        decider = resolve(tool_context)
    except (ValueError, TypeError) as exc:
        return error(str(exc))
    except RuntimeError as exc:
        return endpoint_error(exc)
    states = [render_state(item, context) for item in listed]
    results = ask_each(decider, states, {"q": question_model})
    scored: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for index, (item, result) in enumerate(zip(listed, results, strict=True)):
        if isinstance(result, Exception):
            failed.append(_failed(index, item, result))
            continue
        try:
            value, confidence = _value_of(result, bool(levels))
        except ValueError as exc:
            failed.append(_failed(index, item, exc))
            continue
        scored.append({"index": index, "item": item, "value": value, "confidence": confidence})
    scored.sort(key=lambda r: (r["value"], r["confidence"]), reverse=True)
    ranked = [{"rank": rank, **row} for rank, row in enumerate(scored[:k], start=1)]
    payload = {
        "ranked": ranked,
        "failed": failed,
        "total": len(listed),
        "scale": "score" if levels else "probability",
    }
    best = f"best #{ranked[0]['index']} ({ranked[0]['value']:.2f})" if ranked else "nothing ranked"
    tail = f", {len(failed)} failed" if failed else ""
    return ok(payload, f"ranked {len(ranked)} of {len(listed)}: {best}{tail}")


def _value_of(answers: Any, scored: bool) -> tuple[float, float]:
    if scored:
        answer = expect_score(answers, "q")
        return answer.score, answer.confidence
    noul = expect_noul(answers, "q")
    return noul.noul, abs(2 * noul.noul - 1)


def whole_top_k(value: Any, total: int) -> int:
    """``top_k`` as a whole number in 1..total, or ``total`` when omitted."""
    if value is None or value == "":
        return total
    if isinstance(value, bool):
        raise ValueError("top_k must be a whole number, got a boolean")
    try:
        k = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"top_k must be a whole number, got {value!r}") from exc
    if k < 1:
        raise ValueError(f"top_k must be at least 1, got {k}")
    return min(k, total)


BATCH_TOOLS = [decider_sift, decider_classify, decider_rank]
