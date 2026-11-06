"""The Strands Decider v2 core: the decision interface without torch.

docs/v2-spec.md defines the interface; this package implements its torch-free half with
`tokenizers`, numpy and the standard library:

    render(request, tokenizer, variant)          -> RenderedRequest: ids, S, codes per question
    read(hidden_or_logits, readout, config, rendered) -> answers in v1's response schema
    DecisionConfig.load(checkpoint)              -> decision_config.json, validated
    Readout.load(checkpoint)                     -> readout.safetensors, and the soft-cap
                                                    from decision_config.json

A runtime supplies the decoder (`Backend`); `evaluate` joins the three. v1's engines are
untouched: v2 sits beside them.
"""

from __future__ import annotations

from .codes import CODES, MAX_CODES, codes_for
from .config import FORMAT_VERSION, INTERFACE, DecisionConfig, Provenance
from .prompt import (
    CHAT,
    PLAIN,
    VARIANTS,
    RenderedQuestion,
    RenderedRequest,
    render,
    render_content,
    render_text,
)
from .readout import Readout, answer, code_probabilities, load_safetensors, read, softcap
from .request import parse_request
from .runtime import Backend, attention_mask, evaluate

__all__ = [
    "CHAT",
    "CODES",
    "FORMAT_VERSION",
    "INTERFACE",
    "MAX_CODES",
    "PLAIN",
    "VARIANTS",
    "Backend",
    "DecisionConfig",
    "Provenance",
    "Readout",
    "RenderedQuestion",
    "RenderedRequest",
    "answer",
    "attention_mask",
    "code_probabilities",
    "codes_for",
    "evaluate",
    "load_safetensors",
    "parse_request",
    "read",
    "render",
    "render_content",
    "render_text",
    "softcap",
]
