"""The Strands Decider core: what every runtime shares, without torch.

It renders a request as token ids (`render`), checks a checkpoint's files at load
(`DecisionConfig.load`, `load_tokenizer`, `load_readout`, `check_image_processor`), and reads the
answers from a backend's rows (`read`). The backend runs the model (`Backend`); `evaluate` joins them.
"""

from __future__ import annotations

from .codes import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS, MAX_CODES
from .config import FORMAT_VERSION, INTERFACE, DecisionConfig
from .images import (
    check_decoded_size,
    check_image_processor,
    decode_image,
    resize_image,
    resized_size,
    soft_tokens,
)
from .prompt import (
    CHAT,
    RenderedImage,
    RenderedQuestion,
    RenderedRequest,
    render,
    render_content,
    render_text,
)
from .readout import (
    load_readout,
    question_probabilities,
    read,
)
from .request import Request
from .runtime import NO_VISION, Backend, attention_mask, evaluate
from .tokenizer import load_tokenizer, prepare_tokenizer

__all__ = [
    "CHAT",
    "CODES",
    "CODE_TOKEN_IDS",
    "CONTROL_TOKEN_IDS",
    "FORMAT_VERSION",
    "INTERFACE",
    "MAX_CODES",
    "NO_VISION",
    "Backend",
    "DecisionConfig",
    "RenderedImage",
    "RenderedQuestion",
    "RenderedRequest",
    "Request",
    "attention_mask",
    "check_decoded_size",
    "check_image_processor",
    "decode_image",
    "evaluate",
    "load_readout",
    "load_tokenizer",
    "prepare_tokenizer",
    "question_probabilities",
    "read",
    "render",
    "render_content",
    "render_text",
    "resize_image",
    "resized_size",
    "soft_tokens",
]
