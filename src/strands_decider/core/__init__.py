"""The Strands Decider core: the decision interface without torch.

docs/spec.md defines the interface; this package implements its torch-free half with
`tokenizers`, numpy and the standard library:

    render(request, tokenizer, variant)    -> RenderedRequest: ids, S, codes per question,
                                              and each image's span, N and sha256
    read(rows, readout, config, rendered, inputs=stage)
                                           -> answers in v1's response schema, from the
                                              stage a runtime returns (readout.STAGES)
    DecisionConfig.load(checkpoint)        -> decision_config.json, validated
    Readout.load(checkpoint)               -> readout.safetensors, and the soft-cap from
                                              decision_config.json

A runtime supplies the decoder (`Backend`), and for images the vision encoder and the
pixel preprocessing; `evaluate` joins the pieces. Images are checked and counted from their
file headers (`decode_image`, `soft_tokens`), never decoded here. v1's engines are
untouched: the core sits beside them.
"""

from __future__ import annotations

from .codes import CODES, MAX_CODES, codes_for
from .config import FORMAT_VERSION, INPUTS, INTERFACE, DecisionConfig, Provenance
from .images import (
    IMAGE_FORMATS,
    MAX_IMAGE_PIXELS,
    MAX_IMAGES,
    EncodedImage,
    decode_image,
    image_size,
    soft_tokens,
)
from .prompt import (
    CHAT,
    PLAIN,
    VARIANTS,
    RenderedImage,
    RenderedQuestion,
    RenderedRequest,
    render,
    render_content,
    render_text,
)
from .readout import (
    CAPPED_LOGITS,
    FINAL_LOGITS,
    HIDDEN,
    PROBABILITIES,
    RAW_LOGITS,
    STAGES,
    Readout,
    answer,
    code_probabilities,
    load_safetensors,
    question_probabilities,
    read,
    readout_logits,
    round_to_bf16,
    softcap,
)
from .request import parse_request
from .runtime import Backend, attention_mask, evaluate

__all__ = [
    "CAPPED_LOGITS",
    "CHAT",
    "CODES",
    "FINAL_LOGITS",
    "FORMAT_VERSION",
    "HIDDEN",
    "IMAGE_FORMATS",
    "INPUTS",
    "INTERFACE",
    "MAX_CODES",
    "MAX_IMAGES",
    "MAX_IMAGE_PIXELS",
    "PLAIN",
    "PROBABILITIES",
    "RAW_LOGITS",
    "STAGES",
    "VARIANTS",
    "Backend",
    "DecisionConfig",
    "EncodedImage",
    "Provenance",
    "Readout",
    "RenderedImage",
    "RenderedQuestion",
    "RenderedRequest",
    "answer",
    "attention_mask",
    "code_probabilities",
    "codes_for",
    "decode_image",
    "evaluate",
    "image_size",
    "load_safetensors",
    "parse_request",
    "question_probabilities",
    "read",
    "readout_logits",
    "render",
    "render_content",
    "render_text",
    "round_to_bf16",
    "soft_tokens",
    "softcap",
]
