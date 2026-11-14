"""What a runtime provides, and the glue around it (docs/architecture.md, The core's runtime
interface).

A runtime runs the decoder. This core does the rest: it renders the request, hands the
runtime one state and its questions as token ids, with the state's images and where their
placeholders sit, and reads the answers. The contract is v1's `hidden_engine.Backend`,
narrowed to one row per question at its last token.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import IMAGE, DecisionConfig
from .prompt import RenderedImage, render
from .readout import STAGES, Readout, check_rendered, check_stage, read
from .request import parse_request

if TYPE_CHECKING:
    from pydantic import BaseModel
    from tokenizers import Tokenizer

# What a backend without an image path raises when `forward` is given images: v1's text-only
# engine's message (hidden_engine.HiddenStateEngine).
NO_VISION = "this engine has no vision tower; images are not supported"


class Backend(Protocol):
    """A decoder. `outputs` says which stage `forward` returns (`readout.STAGES`), and the
    core applies the rest, so the soft-cap and the temperature are each applied once:

        "hidden"         the final-norm hidden state at the last token, [hidden]
        "raw_logits"     z = W h, the readout applied: [255], or the question's [k]
        "capped_logits"  tanh(z / c) * c, the soft-cap applied, not the temperature
        "final_logits"   tanh(z / c) * c / T, both applied, as in strands-decider-vllm's model
        "probabilities"  the codes' probabilities: the question's k, or all 255

    Every backend takes images: `forward` gets the state's images, which are empty for a
    request without them. A backend that has no image path (no vision encoder loaded)
    raises ValueError(NO_VISION) when it is given images.
    """

    outputs: str

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray | Sequence[np.ndarray]:
        """Run each question as a continuation of the state, under the prefix_bidir mask,
        seeing no other question: positions 0..S-1 for the state, S.. for the question.
        Return one row per question, in the order of `questions`, of the stage `outputs`
        names: an array [len(questions), width], or a sequence of rows (whose widths may
        differ, as each question's k does).

        `images` are the state's, in order: each one's file (`data`), and the span
        `[start, end)` of its N placeholders in `state`, which the backend fills with its
        vision encoder's output for the image, preprocessed by the checkpoint's image
        processor."""
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
    request: dict[str, Any] | BaseModel,
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
    `readout` may be None for a backend that applies the readout itself. A request with
    images is refused, before its images are decoded, unless the checkpoint's `inputs` hold
    "image"; the backend gets the images, and one without an image path refuses them.
    Everything that can be checked without the decoder is checked before `forward` runs:
    the stage, the readout, and the rendered request's prompt and code token ids.
    """
    stage = getattr(backend, "outputs", None)
    if stage not in STAGES:
        raise ValueError(
            f"the backend must say what forward returns: `outputs` is {stage!r}, not one "
            f"of {STAGES}"
        )
    check_stage(stage, readout, config)
    # The request's own checks, then the refusal of images by a checkpoint without them,
    # then the images, each in turn (in `render`).
    req = parse_request(request)
    if req.images and IMAGE not in config.inputs:
        raise ValueError(
            "this checkpoint does not take images: its decision_config.json inputs are "
            + json.dumps(list(config.inputs))
        )
    rendered = render(
        req, tokenizer, config.prompt, max_length=max_length or config.max_length_validated
    )
    check_rendered(config, rendered)
    state = list(rendered.state_ids)
    questions = [list(q.question_ids) for q in rendered.questions.values()]
    out = backend.forward(state, questions, images=rendered.images)
    answers = read(out, readout, config, rendered, stage=stage, ordinal_smoothing=ordinal_smoothing)
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
    }
