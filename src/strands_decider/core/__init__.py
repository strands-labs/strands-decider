"""The Strands Decider core: the decision interface without torch.

docs/spec.md defines the interface; this package implements its torch-free half with
`tokenizers`, numpy, pydantic and the standard library, and Pillow for images:

    render(request, tokenizer, variant)    -> RenderedRequest: ids, S, codes per question,
                                              and each image's span, N and sha256
    read(rows, readout, config, rendered, stage=stage)
                                           -> answers in v1's response schema, from the
                                              stage a runtime returns (readout.STAGES)
    DecisionConfig.load(checkpoint)        -> decision_config.json, validated
    Readout.load(checkpoint)               -> readout.safetensors, and the soft-cap from
                                              decision_config.json
    load_tokenizer(checkpoint)             -> tokenizer.json, checked against the
                                              interface's control and code ids

A runtime supplies the decoder (`Backend`, whose `forward` takes the state's images), and for
images the vision encoder and the pixel preprocessing; `evaluate` joins the pieces. Images are
checked and counted from their file headers, which Pillow reads (`decode_image`,
`soft_tokens`), never decoded here; `resize_image` is the spec's resize for a runtime's
decoded pixels. Pillow is imported only for an image, so the core imports without it.
`agreement` measures a runtime's probabilities against the reference's (docs/spec.md,
section 10). v1's engines are untouched: the core sits beside them.
"""

from __future__ import annotations

from .codes import CODES, MAX_CODES, codes_for
from .config import FORMAT_VERSION, INPUTS, INTERFACE, DecisionConfig, Provenance
from .conformance import (
    BOUNDS,
    Agreement,
    NoRecordedBound,
    agreement,
    agreement_with_goldens,
    read_golden_vectors,
)
from .images import (
    IMAGE_FORMATS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGES,
    EncodedImage,
    check_decoded_size,
    check_image_processor,
    decode_image,
    decode_images,
    resize_image,
    resized_size,
    soft_tokens,
)
from .prompt import (
    CHAT,
    PLAIN,
    VARIANTS,
    RenderedImage,
    RenderedQuestion,
    RenderedRequest,
    plain_text_tokenizer,
    release_plain_text_tokenizer,
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
from .request import Request, parse_request
from .runtime import NO_VISION, Backend, attention_mask, evaluate
from .tokenizer import CODE_TOKEN_IDS, CONTROL_TOKEN_IDS, check_tokenizer, load_tokenizer

__all__ = [
    "BOUNDS",
    "CAPPED_LOGITS",
    "CHAT",
    "CODES",
    "CODE_TOKEN_IDS",
    "CONTROL_TOKEN_IDS",
    "FINAL_LOGITS",
    "FORMAT_VERSION",
    "HIDDEN",
    "IMAGE_FORMATS",
    "INPUTS",
    "INTERFACE",
    "MAX_CODES",
    "MAX_IMAGES",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_PIXELS",
    "NO_VISION",
    "PLAIN",
    "PROBABILITIES",
    "RAW_LOGITS",
    "STAGES",
    "VARIANTS",
    "Agreement",
    "Backend",
    "DecisionConfig",
    "EncodedImage",
    "NoRecordedBound",
    "Provenance",
    "Readout",
    "RenderedImage",
    "RenderedQuestion",
    "RenderedRequest",
    "Request",
    "agreement",
    "agreement_with_goldens",
    "answer",
    "attention_mask",
    "check_decoded_size",
    "check_image_processor",
    "check_tokenizer",
    "code_probabilities",
    "codes_for",
    "decode_image",
    "decode_images",
    "evaluate",
    "load_safetensors",
    "load_tokenizer",
    "parse_request",
    "plain_text_tokenizer",
    "question_probabilities",
    "read",
    "read_golden_vectors",
    "readout_logits",
    "release_plain_text_tokenizer",
    "render",
    "render_content",
    "render_text",
    "resize_image",
    "resized_size",
    "round_to_bf16",
    "soft_tokens",
    "softcap",
]
