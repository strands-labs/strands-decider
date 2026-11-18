"""What the engine reads from a checkpoint's ``decision_config.json``: the readout's code token ids, temperature and
soft-cap, and whether the checkpoint takes images. Nothing here imports vLLM or torch.

The plugin's server validates the whole file with the core's ``DecisionConfig``. The engine reads only what it needs,
so that ``vllm serve`` also serves an intermediate checkpoint: a bare ``Gemma4TextModel`` decoder whose
``decision_config.json`` predates ``interface`` and ``inputs``, served as text-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .validation import N_CODES

INTERFACE = "strands-decider"
TEXT, IMAGE = "text", "image"
#: The two checkpoint layouts: the base model's multimodal one, and an intermediate bare decoder.
MULTIMODAL, DECODER = "multimodal", "decoder"


class DecisionConfigError(ValueError):
    """A ``decision_config.json`` (or decoder config) the plugin refuses to serve."""

    def __init__(self, message: str) -> None:
        super().__init__(f"strands-decider: decision_config.json: {message}")


@dataclass(frozen=True)
class ReadoutSpec:
    token_ids: tuple[int, ...]
    temperature: float
    softcap: float
    inputs: tuple[str, ...] = (TEXT,)

    @property
    def takes_images(self) -> bool:
        return IMAGE in self.inputs


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
        raise DecisionConfigError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def read_decision_config(
    dc: Any, *, text_softcap: Any, vocab_size: int, layout: str = DECODER
) -> ReadoutSpec:
    """``dc``, the parsed ``decision_config.json``, checked for serving a checkpoint of ``layout``. The soft-cap is
    the file's ``final_logit_softcapping``; a decoder checkpoint without one takes its decoder config's
    (``text_softcap``), and a checkpoint whose two values differ is refused."""
    if not isinstance(dc, dict):
        raise DecisionConfigError(f"expected a JSON object, got {type(dc).__name__}")
    if (layout == MULTIMODAL or "interface" in dc) and dc.get("interface") != INTERFACE:
        raise DecisionConfigError(
            f'"interface": "{INTERFACE}" is required, got {dc.get("interface")!r}'
        )
    for key, expected in (
        ("format_version", 1),
        ("attention_mode", "prefix_bidir"),
        ("pooling", "last"),
    ):
        if dc.get(key) != expected or isinstance(dc.get(key), bool):
            raise DecisionConfigError(f"{key} must be {expected!r}, got {dc.get(key)!r}")

    inputs = dc.get("inputs", [TEXT] if layout == DECODER else None)
    if not (isinstance(inputs, list) and TEXT in inputs and set(inputs) <= {TEXT, IMAGE}):
        raise DecisionConfigError(f'inputs must be ["text"] or ["text", "image"], got {inputs!r}')
    if layout == DECODER and IMAGE in inputs:
        raise DecisionConfigError(
            'inputs names "image", but the checkpoint is a bare text decoder with no vision encoder'
        )

    ids = dc.get("token_ids")
    if not (
        isinstance(ids, list)
        and len(ids) == N_CODES
        and all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < vocab_size for t in ids)
        and len(set(ids)) == N_CODES
    ):
        raise DecisionConfigError(
            f"token_ids must be {N_CODES} distinct token ids in [0, {vocab_size})"
        )

    temperature = _positive("temperature", dc.get("temperature"))
    text = (
        None if text_softcap is None else _positive("the decoder config's soft-cap", text_softcap)
    )
    if "final_logit_softcapping" in dc:
        softcap = _positive("final_logit_softcapping", dc["final_logit_softcapping"])
        if text is not None and text != softcap:
            raise DecisionConfigError(
                f"final_logit_softcapping {softcap} differs from the decoder config's {text}"
            )
    elif layout == DECODER and text is not None:
        softcap = text
    else:
        raise DecisionConfigError("final_logit_softcapping is required: the readout is soft-capped")
    return ReadoutSpec(tuple(ids), temperature, softcap, tuple(dict.fromkeys(inputs)))
