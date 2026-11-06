"""What a v2 runtime provides, and the glue around it (docs/v2-spec.md, The runtime interface).

A runtime runs the decoder. This core does the rest: it renders the request, hands the
runtime one state and its questions as token ids, and reads the answers. The contract is
v1's `hidden_engine.Backend` narrowed to v2: one row per question, at its last token.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import DecisionConfig
from .prompt import render
from .readout import Readout, read

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class Backend(Protocol):
    def forward(self, state: list[int], questions: list[list[int]]) -> np.ndarray:
        """Run each question as a continuation of the state, under the prefix_bidir mask,
        seeing no other question: positions 0..S-1 for the state, S.. for the question.
        Return the final-norm hidden state at each question's last token, an array
        [len(questions), hidden], in the order of `questions`.

        A backend that computes the readout itself instead returns the raw readout logits
        `W h`, [len(questions), 255], and says so with `outputs = "logits"`."""
        ...


def attention_mask(state_len: int, total_len: int) -> np.ndarray:
    """The prefix_bidir mask of one prompt, as booleans [L, L]: `mask[i, j]` is True when
    token i attends to token j. State tokens (i < S) see the state, both ways, and never
    the question; question tokens see the whole prompt, both ways. Every layer uses it,
    sliding-window layers included, with no window."""
    if not 0 < state_len <= total_len:
        raise ValueError(f"need 0 < S <= L, not S={state_len}, L={total_len}")
    m = np.ones((total_len, total_len), dtype=bool)
    m[:state_len, state_len:] = False
    return m


def evaluate(
    request: Any,
    tokenizer: Tokenizer,
    readout: Readout,
    config: DecisionConfig,
    backend: Backend,
    *,
    model_name: str = "strands-decider",
    max_length: int | None = None,
    ordinal_smoothing: float = 0.0,
) -> dict[str, Any]:
    """One `POST /v1/systemone` request, answered: a v1 `SystemOneResponse` as JSON.

    The prompt is the checkpoint's (`config.prompt`). Prompts longer than
    `config.max_length_validated` are refused unless `max_length` raises the limit.
    """
    rendered = render(
        request, tokenizer, config.prompt, max_length=max_length or config.max_length_validated
    )
    questions: Sequence[list[int]] = [list(q.question_ids) for q in rendered.questions.values()]
    out = np.asarray(backend.forward(list(rendered.state_ids), list(questions)))
    inputs = getattr(backend, "outputs", "hidden")
    answers = read(
        out, readout, config, rendered, inputs=inputs, ordinal_smoothing=ordinal_smoothing
    )
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
    }
