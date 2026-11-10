"""What a runtime provides, and the glue around it (docs/spec.md, The runtime interface).

A runtime runs the decoder. This core does the rest: it renders the request, hands the
runtime one state and its questions as token ids, with the state's images and where their
placeholders sit, and reads the answers. The contract is v1's `hidden_engine.Backend`,
narrowed to one row per question at its last token.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from .config import IMAGE, TEXT, DecisionConfig
from .prompt import RenderedImage, render
from .readout import STAGES, Readout, check_rendered, check_stage, read
from .request import NO_VISION, has_images, parse_request

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class Backend(Protocol):
    """A decoder. `outputs` says which stage `forward` returns (`readout.STAGES`), and the
    core applies the rest, so the soft-cap and the temperature are each applied once:

        "hidden"         the final-norm hidden state at the last token, [hidden]
        "raw_logits"     z = W h, the readout applied: [255], or the question's [k]
        "capped_logits"  tanh(z / c) * c, the soft-cap applied, not the temperature
        "final_logits"   tanh(z / c) * c / T, both applied (as strands-decider-vllm returns them)
        "probabilities"  the codes' probabilities: the question's k, or all 255

    A text-only backend is this protocol. A backend with an image path is also an
    `ImageBackend`: it has `inputs`, and its `forward` takes `images`.
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


class ImageBackend(Backend, Protocol):
    """A backend with an image path. `inputs` says what `forward` takes, as
    decision_config.json's `inputs` names it: ("text", "image"). The core passes `images`
    only when the request has some, and only to a backend whose `inputs` hold "image"; a
    backend without `inputs` takes text only, and the core refuses image requests for it."""

    inputs: tuple[str, ...]

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray | Sequence[np.ndarray]:
        """As `Backend.forward`, with the state's images in order: each one's file (`data`),
        and the span `[start, end)` of its N placeholders in `state`, which the backend fills
        with its vision encoder's output for the image, preprocessed by the checkpoint's
        image processor."""
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
    `readout` may be None for a backend that applies the readout itself. A request with
    images is refused, before its images are decoded, unless the checkpoint's `inputs`
    and the backend's both hold "image". Everything that can be checked without the
    decoder is checked before `forward` runs: the stage, the readout, and the rendered
    request's prompt and code token ids.
    """
    stage = getattr(backend, "outputs", None)
    if stage not in STAGES:
        raise ValueError(
            f"the backend must say what forward returns: `outputs` is {stage!r}, not one "
            f"of {STAGES}"
        )
    check_stage(stage, readout, config)
    # v1's order: the request's own checks, then the refusal of images by an engine without
    # them, then the images, each in turn.
    parse_request(request, images=False)
    if has_images(request):
        if IMAGE not in config.inputs:
            raise ValueError(
                "this checkpoint does not take images: its decision_config.json inputs are "
                + json.dumps(list(config.inputs))
            )
        if IMAGE not in tuple(getattr(backend, "inputs", (TEXT,))):
            raise ValueError(NO_VISION)
    rendered = render(
        request, tokenizer, config.prompt, max_length=max_length or config.max_length_validated
    )
    check_rendered(config, rendered)
    state = list(rendered.state_ids)
    questions = [list(q.question_ids) for q in rendered.questions.values()]
    if rendered.images:
        out = cast("ImageBackend", backend).forward(state, questions, images=rendered.images)
    else:
        out = backend.forward(state, questions)
    answers = read(out, readout, config, rendered, stage=stage, ordinal_smoothing=ordinal_smoothing)
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
    }
