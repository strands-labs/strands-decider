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
from .readout import STAGES, Readout, read

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class Backend(Protocol):
    """A decoder. `outputs` says which stage `forward` returns (`readout.STAGES`), and the
    core applies the rest, so the soft-cap and the temperature are each applied once:

        "hidden"         the final-norm hidden state at the last token, [hidden]
        "raw_logits"     z = W h, the readout applied: [255], or the question's [k]
        "capped_logits"  tanh(z / c) * c, the soft-cap applied, not the temperature
        "final_logits"   tanh(z / c) * c / T, both applied (the vLLM plugin's compute_logits)
        "probabilities"  the codes' probabilities: the question's k, or all 255
    """

    outputs: str

    def forward(
        self, state: list[int], questions: list[list[int]]
    ) -> np.ndarray | Sequence[np.ndarray]:
        """Run each question as a continuation of the state, under the prefix_bidir mask,
        seeing no other question: positions 0..S-1 for the state, S.. for the question.
        Return one row per question, in the order of `questions`, of the stage `outputs`
        names: an array [len(questions), width], or a sequence of rows (whose widths may
        differ, as each question's k does)."""
        ...


def attention_mask(state_len: int, total_len: int) -> np.ndarray:
    """The prefix_bidir mask of one prompt, as booleans [L, L]: `mask[i, j]` is True when
    token i attends to token j. State tokens (i < S) see the state, both ways, and never
    the question; question tokens see the whole prompt, both ways. Every layer uses it,
    sliding-window layers included, with no window.

    An illustration of the rule, for tests and small examples, not for serving: the array
    is dense, O(L^2) in memory (64 MB at L = 8,192). A runtime gives its attention kernel
    the rule in the form that kernel takes, such as S and each question's span."""
    if not 0 < state_len <= total_len:
        raise ValueError(f"need 0 < S <= L, not S={state_len}, L={total_len}")
    m = np.ones((total_len, total_len), dtype=bool)
    m[:state_len, state_len:] = False
    return m


def evaluate(
    request: Any,
    tokenizer: Tokenizer,
    readout: Readout | None,
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
    `readout` may be None for a backend that applies the readout itself.
    """
    inputs = getattr(backend, "outputs", None)
    if inputs not in STAGES:
        raise ValueError(
            f"the backend must say what forward returns: `outputs` is {inputs!r}, not one "
            f"of {STAGES}"
        )
    rendered = render(
        request, tokenizer, config.prompt, max_length=max_length or config.max_length_validated
    )
    questions = [list(q.question_ids) for q in rendered.questions.values()]
    out = backend.forward(list(rendered.state_ids), questions)
    answers = read(
        out, readout, config, rendered, inputs=inputs, ordinal_smoothing=ordinal_smoothing
    )
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
    }
