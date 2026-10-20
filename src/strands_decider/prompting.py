"""Turning a (state, question) pair into text, and slots into meaning.

The whole trick of the architecture lives here. The classification heads are
*generic*: head k has no fixed meaning. It scores "whatever option was listed at
position k in the prompt". Nothing in the weights knows that slot 0 is `billing`
in one request and `Calm` in the next -- the prompt alone establishes that binding,
which is what makes the model adapt to novel label sets without retraining.

Two consequences shape everything downstream:

1. The prompt must number the options explicitly and unambiguously, so the
   binding is learnable and stable.
2. Training must shuffle option order (see data/collate.py), or the heads quietly
   memorise "slot 0 tends to be the positive class" and genericity is lost.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Content,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    derive_confidence,
    derive_score_confidence,
)

# Rendered on both sides of a noul so the two slots read like any other option list.
NOUL_SLOT_LABELS = ("false", "true")
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}


def render_content(content: Content | None) -> str:
    """Flatten a state or instruction into text, stably.

    Dicts and lists are emitted as indented JSON rather than str() so that key
    order and unicode are deterministic -- the same state must always tokenise
    identically, otherwise the shared-prefix cache in infer.py would be unsound.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


@dataclass(frozen=True)
class RenderedQuestion:
    """A question flattened to text plus the slot->label map needed to read it back."""

    text: str
    # slot_labels[k] is the option name bound to head k for this request.
    slot_labels: tuple[str, ...]
    # slot_descriptions[k] is that option's rubric text, kept so a score answer can
    # report a legend of what each level meant rather than bare indices.
    slot_descriptions: tuple[str, ...]
    kind: str  # "noul" | "choice" | "score"
    # (start, end) character span of each option's line within `text`. A pointer head
    # scores option k from the hidden state at the LAST token of this span -- the token
    # that has just read the whole option under causal attention. Characters rather than
    # token indices so callers can map them with the tokeniser's offset mapping instead
    # of rebuilding the prompt segment by segment.
    option_spans: tuple[tuple[int, int], ...] = ()

    @property
    def n_slots(self) -> int:
        return len(self.slot_labels)


def _option_block(pairs: Sequence[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Number options from 1 so slot k <-> the line reading `k+1.`.

    Also returns each line's (start, end) character span within the block. Descriptions
    are collapsed to a single line so one option is always exactly one line, which is
    what makes the span unambiguous for a pointer readout.
    """
    lines: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, (name, desc) in enumerate(pairs):
        desc = " ".join((desc or "").split())
        line = f"{i + 1}. {name}" + (f" \u2014 {desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1  # the joining newline
    return "\n".join(lines), spans


def render_question(
    question: Question,
    *,
    option_order: Sequence[int] | None = None,
) -> RenderedQuestion:
    """Render one question. `option_order` permutes the options (training augmentation).

    The permutation is applied to the *rendered* list, and slot_labels records the
    resulting binding, so callers never have to track the permutation themselves.
    """
    instructions = render_content(question.instructions)

    if isinstance(question, NoulQuestion):
        crit = {**NOUL_DEFAULT_CRITERIA, **(question.criteria or {})}
        pairs = [(lbl, render_content(crit[lbl])) for lbl in NOUL_SLOT_LABELS]
        kind = "noul"
        header = "Decide whether the statement is true of the state."
    elif isinstance(question, ChoiceQuestion):
        pairs = [(name, render_content(desc)) for name, desc in question.criteria.items()]
        kind = "choice"
        header = "Select exactly one option."
    elif isinstance(question, ScoreQuestion):
        # Level index is the label; the rubric text is the description.
        pairs = [(str(i), desc) for i, desc in enumerate(question.criteria)]
        kind = "score"
        header = "Rate the state against the ordered levels below (lowest first)."
    else:  # pragma: no cover - guarded by pydantic unions
        raise TypeError(f"unknown question type: {type(question)!r}")

    if option_order is not None:
        if sorted(option_order) != list(range(len(pairs))):
            raise ValueError("option_order must be a permutation of the option indices")
        pairs = [pairs[i] for i in option_order]

    block, block_spans = _option_block(pairs)
    prefix = f'<question type="{kind}">\n{header}\n{instructions}\n<options>\n'
    text = prefix + block + "\n</options>\n</question>\n<answer>"
    base = len(prefix)
    return RenderedQuestion(
        text=text,
        slot_labels=tuple(n for n, _ in pairs),
        slot_descriptions=tuple((d or "").strip() for _, d in pairs),
        kind=kind,
        option_spans=tuple((base + s, base + e) for s, e in block_spans),
    )


def render_state(state: Content) -> str:
    """The shared prefix. Everything before this point is identical across the
    questions in one request, which is exactly what infer.py caches once."""
    return f"<state>\n{render_content(state)}\n</state>\n"


def build_prompt(state: Content, question: Question, **kw: Any) -> tuple[str, RenderedQuestion]:
    rq = render_question(question, **kw)
    return render_state(state) + rq.text, rq


def read_noul(probs: Sequence[float], rq: RenderedQuestion) -> float:
    """Recover P(true) regardless of which slot `true` landed in."""
    idx = rq.slot_labels.index("true")
    return float(probs[idx])


def read_choice(probs: Sequence[float], rq: RenderedQuestion) -> dict[str, float]:
    return {name: float(p) for name, p in zip(rq.slot_labels, probs, strict=True)}


def score_legend(rq: RenderedQuestion) -> dict[str, str]:
    """{level index -> rubric text} in canonical ascending order.

    Needed because the rendering may have reversed the rubric; the caller should
    always see levels the way the request declared them.
    """
    return {
        level: desc
        for level, desc in sorted(
            zip(rq.slot_labels, rq.slot_descriptions, strict=True),
            key=lambda kv: int(kv[0]),
        )
    }


def read_score(probs: Sequence[float], rq: RenderedQuestion) -> tuple[float, dict[str, float]]:
    """Expected level index, and the distribution keyed by level.

    Because slots may have been permuted, we re-key by the true level index before
    taking the expectation -- otherwise a shuffled rendering would score backwards.
    """
    by_level = {name: float(p) for name, p in zip(rq.slot_labels, probs, strict=True)}
    expected = sum(int(level) * p for level, p in by_level.items())
    ordered = {str(i): by_level[str(i)] for i in range(len(by_level))}
    return float(expected), ordered


def option_token_index(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Sequence[int]],
    base: int,
) -> list[int]:
    """Last token index of each option's line, for a pointer readout.

    The last token of the span is the one that has just read the whole option under
    causal attention. Raises when an option has no surviving token: truncation has
    removed it, and scoring it from a neighbour's representation would be silently
    wrong.
    """
    out: list[int] = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was "
                "truncated through its option list"
            )
        out.append(last)
    return out


def to_answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> Answer:
    if rq.kind == "noul":
        return NoulAnswer(noul=round(read_noul(probs, rq), 4))

    if rq.kind == "choice":
        by_name = read_choice(probs, rq)
        best = max(by_name, key=lambda k: by_name[k])
        return ChoiceAnswer(
            choice=best,
            probabilities={k: round(v, 4) for k, v in by_name.items()},
            confidence=round(derive_confidence(list(by_name.values())), 4),
        )

    score, ordered = read_score(probs, rq)
    return ScoreAnswer(
        score=round(score, 4),
        # Report the rubric in canonical ascending order, not the order rendered.
        legend=score_legend(rq),
        probabilities={k: round(v, 4) for k, v in ordered.items()},
        # Ordinal confidence, not max-probability: see derive_score_confidence.
        confidence=round(
            derive_score_confidence(list(ordered.values()), ordinal_smoothing=ordinal_smoothing),
            4,
        ),
    )
