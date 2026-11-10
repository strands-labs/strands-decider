"""What the plugin reads from a checkpoint's ``decision_config.json``, validated (specification, section 8).

:func:`read_decision_config` takes the parsed file, the checkpoint's layout and the decoder config's
``final_logit_softcapping``, and returns the readout's code token ids, temperature and soft-cap and the inputs the
checkpoint takes, or raises :class:`DecisionConfigError` at startup. Nothing here imports vLLM or torch.

Two layouts are served:

- ``multimodal``: the base Gemma 4 model's own layout (``Gemma4ForConditionalGeneration``, or
  ``Gemma4UnifiedForConditionalGeneration`` for the 12B), the released format. ``interface``, ``inputs``,
  ``final_logit_softcapping`` and ``provenance`` (``base_model``, ``base_revision``) are required.
- ``decoder``: a bare ``Gemma4TextModel``, the layout of the intermediate checkpoints made before the multimodal one.
  They predate ``interface``, ``inputs`` and ``provenance``: each missing field is a one-line warning, and the
  checkpoint is served as text-only. A decoder checkpoint that names ``"image"`` in ``inputs`` is refused, because it
  has no vision encoder.

The plugin checks what the engine needs. The core library (``strands_decider.core.DecisionConfig``) validates the
whole schema strictly; the plugin's server loads it that way.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .validation import N_CODES

INTERFACE = "strands-decider"
TEXT, IMAGE = "text", "image"
INPUTS = (TEXT, IMAGE)
#: The two checkpoint layouts (see the module docstring).
MULTIMODAL, DECODER = "multimodal", "decoder"
LAYOUTS = (MULTIMODAL, DECODER)
_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_TINY = 1.1754943508222875e-38
_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class DecisionConfigError(ValueError):
    """A ``decision_config.json`` (or decoder config) the plugin refuses to serve."""

    def __init__(self, message: str) -> None:
        super().__init__(f"strands-decider: decision_config.json: {message}")


@dataclass(frozen=True)
class ReadoutSpec:
    token_ids: tuple[int, ...]
    temperature: float
    softcap: float
    #: The inputs the checkpoint takes: ("text",) or ("text", "image").
    inputs: tuple[str, ...] = (TEXT,)
    #: provenance.base_model and provenance.base_revision, when given.
    base_model: str | None = None
    base_revision: str | None = None
    #: One-line notes for the startup log (an intermediate checkpoint's missing fields).
    warnings: tuple[str, ...] = ()

    @property
    def takes_images(self) -> bool:
        return IMAGE in self.inputs


def _positive_float32(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionConfigError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not (math.isfinite(v) and _FLOAT32_TINY <= v <= _FLOAT32_MAX):
        raise DecisionConfigError(
            f"{name} must be finite, above 0 and in float32's normal range, got {value!r}"
        )
    return v


def _inputs(dc: dict, layout: str, warnings: list[str]) -> tuple[str, ...]:
    if "inputs" not in dc:
        if layout == MULTIMODAL:
            raise DecisionConfigError(
                'inputs is required: ["text"] or ["text", "image"], the inputs the checkpoint was trained on'
            )
        warnings.append(
            'decision_config.json has no "inputs" field (an intermediate decoder checkpoint); serving it as '
            'text-only, inputs ["text"]'
        )
        return (TEXT,)
    value = dc["inputs"]
    if (
        not isinstance(value, list)
        or not all(isinstance(i, str) and i in INPUTS for i in value)
        or len(set(value)) != len(value)
        or TEXT not in value
    ):
        raise DecisionConfigError(
            f"inputs must be an array of distinct strings from {list(INPUTS)} that holds {TEXT!r}, got {value!r}"
        )
    if IMAGE in value and layout == DECODER:
        raise DecisionConfigError(
            'inputs names "image", but the checkpoint is a bare text decoder (Gemma4TextModel) with no vision '
            "encoder; an image checkpoint has the base model's multimodal layout"
        )
    return tuple(value)


def _provenance(dc: dict, layout: str, warnings: list[str]) -> tuple[str | None, str | None]:
    prov = dc.get("provenance")
    if prov is None:
        if "provenance" in dc:
            raise DecisionConfigError(
                "provenance must be an object; leave the field out instead of null"
            )
        if layout == MULTIMODAL:
            raise DecisionConfigError(
                "provenance is required, with base_model (the base's Hub id) and base_revision (its 40-character "
                "commit): the checkpoint's image processor and tokenizer are the base's at that revision"
            )
        warnings.append(
            'decision_config.json has no "provenance" field (an intermediate decoder checkpoint)'
        )
        return None, None
    if not isinstance(prov, dict):
        raise DecisionConfigError(f"provenance must be an object, got {type(prov).__name__}")
    base_model, base_revision = prov.get("base_model"), prov.get("base_revision")
    if layout == MULTIMODAL or base_model is not None or base_revision is not None:
        if not isinstance(base_model, str) or not _HUB_ID.match(base_model):
            raise DecisionConfigError(
                f"provenance.base_model must be a Hub id such as google/gemma-4-E2B-it, got {base_model!r}"
            )
        if not isinstance(base_revision, str) or not _COMMIT.match(base_revision):
            raise DecisionConfigError(
                f"provenance.base_revision must be a 40-character commit hash, got {base_revision!r}"
            )
    return base_model, base_revision


def read_decision_config(
    dc: Any,
    *,
    text_softcap: Any,
    vocab_size: int,
    layout: str = DECODER,
) -> ReadoutSpec:
    """Validate ``dc`` (the parsed ``decision_config.json``) for serving a checkpoint of ``layout``.

    - ``interface``: ``"strands-decider"``. Required for the multimodal layout; an intermediate decoder checkpoint
      without it loads with a warning. Checked before any other field.
    - ``format_version`` 1, ``attention_mode: prefix_bidir``, ``pooling: last``.
    - ``inputs``: see :func:`_inputs`.
    - ``provenance``: ``base_model`` (a Hub id) and ``base_revision`` (a commit), required for the multimodal layout.
    - ``token_ids``: 255 distinct integers in ``[0, vocab_size)``.
    - ``temperature``: finite and above 0.
    - the soft-cap: ``final_logit_softcapping``, finite and above 0. Required in this file for the multimodal layout;
      a decoder checkpoint without it takes the decoder config's (``text_softcap``). It is never skipped: a checkpoint
      with neither is refused, and one whose two values differ is refused (specification, section 9).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if not isinstance(dc, dict):
        raise DecisionConfigError(f"expected a JSON object, got {type(dc).__name__}")
    warnings: list[str] = []
    if "interface" in dc:
        if dc["interface"] != INTERFACE:
            raise DecisionConfigError(
                f'interface must be "{INTERFACE}", got {dc["interface"]!r}; this is not a Strands Decider '
                "checkpoint"
            )
    elif layout == DECODER:
        warnings.append(
            f'decision_config.json has no "interface" field (an intermediate checkpoint); serving it as '
            f'"{INTERFACE}"'
        )
    else:
        raise DecisionConfigError(f'"interface": "{INTERFACE}" is required')
    for key, expected in (
        ("format_version", 1),
        ("attention_mode", "prefix_bidir"),
        ("pooling", "last"),
    ):
        value = dc.get(key)
        if value != expected or isinstance(value, bool):
            raise DecisionConfigError(f"{key} must be {expected!r}, got {value!r}")
    inputs = _inputs(dc, layout, warnings)
    base_model, base_revision = _provenance(dc, layout, warnings)

    ids = dc.get("token_ids")
    if not isinstance(ids, list) or len(ids) != N_CODES:
        n = len(ids) if isinstance(ids, list) else type(ids).__name__
        raise DecisionConfigError(f"token_ids must be a list of {N_CODES} token ids, got {n}")
    bad = [
        t for t in ids if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < vocab_size
    ]
    if bad:
        raise DecisionConfigError(
            f"token_ids must be integers in [0, {vocab_size}) (the vocabulary), got {bad[:4]!r}"
        )
    if len(set(ids)) != N_CODES:
        dup = sorted({t for t in ids if ids.count(t) > 1})
        raise DecisionConfigError(f"token_ids must be distinct; repeated: {dup[:4]!r}")

    if "temperature" not in dc:
        raise DecisionConfigError("temperature is required")
    temperature = _positive_float32("temperature", dc["temperature"])

    text = (
        None
        if text_softcap is None
        else _positive_float32("the decoder config's final_logit_softcapping", text_softcap)
    )
    if "final_logit_softcapping" in dc:
        softcap = _positive_float32("final_logit_softcapping", dc["final_logit_softcapping"])
        if text is not None and text != softcap:
            raise DecisionConfigError(
                f"final_logit_softcapping {softcap} differs from the decoder config's {text}"
            )
    elif layout == MULTIMODAL:
        raise DecisionConfigError(
            "final_logit_softcapping is required: the readout is soft-capped, and the plugin does not serve it "
            "without the cap"
        )
    elif text is not None:
        softcap = text
    else:
        raise DecisionConfigError(
            "final_logit_softcapping is missing from both decision_config.json and the decoder config; the "
            "readout is soft-capped, and the plugin does not serve it without the cap"
        )
    return ReadoutSpec(
        tuple(ids), temperature, softcap, inputs, base_model, base_revision, tuple(warnings)
    )
