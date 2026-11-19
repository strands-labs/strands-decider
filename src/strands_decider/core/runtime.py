"""What a backend provides, and `evaluate`, which answers a request with it.

The backend runs the decoder (and the vision encoder for images). The core renders the request,
hands the backend the state and each question as token ids, and reads the answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .config import IMAGE, DecisionConfig
from .prompt import RenderedImage, render
from .readout import read
from .request import Request

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# What a backend with no vision encoder raises when `forward` is given images.
NO_VISION = "this engine has no vision tower; images are not supported"


class Backend(Protocol):
    """A decoder. `forward` returns the final-norm hidden state at each question's last token, or,
    for a backend that applies the readout itself (the vLLM plugin), the codes' probabilities."""

    def forward(
        self, state: list[int], questions: list[list[int]], images: tuple[RenderedImage, ...] = ()
    ) -> np.ndarray | list[np.ndarray]:
        """One row per question, each question run after the state under the prefix_bidir mask,
        seeing no other question. The backend fills each image's placeholder span with its
        vision encoder's output, from the image resized with `images.resize_image`."""
        ...


def attention_mask(state_len: int, total_len: int) -> np.ndarray:
    """The prefix_bidir mask of one prompt as dense booleans, for tests: the state attends to
    itself both ways and never to the question; the question attends to everything both ways.
    Every layer uses it, sliding-window layers included, with no window."""
    m = np.ones((total_len, total_len), dtype=bool)
    m[:state_len, state_len:] = False
    return m


def evaluate(
    request: Request,
    tokenizer: Tokenizer,
    weight: np.ndarray | None,
    config: DecisionConfig,
    backend: Backend,
    *,
    model_name: str = "strands-decider",
    max_length: int | None = None,
    ordinal_smoothing: float = 0.0,
) -> dict[str, Any]:
    """One `POST /v1/systemone` request, answered as its response JSON. Prompts longer than
    `config.max_length_validated` are refused unless `max_length` raises the limit. `weight` is the
    readout weight, or None for a backend whose rows are already probabilities."""
    if request.images and IMAGE not in config.inputs:
        raise ValueError("this checkpoint does not take images")
    rendered = render(request, tokenizer, max_length=max_length or config.max_length_validated)
    questions = [list(q.question_ids) for q in rendered.questions.values()]
    rows = backend.forward(list(rendered.state_ids), questions, images=rendered.images)
    answers = read(rows, config, rendered, weight=weight, ordinal_smoothing=ordinal_smoothing)
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
    }
