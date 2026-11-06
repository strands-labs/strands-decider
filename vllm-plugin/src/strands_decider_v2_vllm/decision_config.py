"""What the plugin reads from a checkpoint's ``decision_config.json``, validated (v2 specification, section 8).

:func:`read_decision_config` takes the parsed file and the decoder config's ``final_logit_softcapping`` and returns the
readout's code token ids, temperature and soft-cap, or raises :class:`DecisionConfigError` at startup. Nothing here
imports vLLM or torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .validation import N_CODES

INTERFACE = "strands-decider-v2"
_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_TINY = 1.1754943508222875e-38


class DecisionConfigError(ValueError):
    """A ``decision_config.json`` (or decoder config) the plugin refuses to serve."""

    def __init__(self, message: str) -> None:
        super().__init__(f"strands-decider-v2: decision_config.json: {message}")


@dataclass(frozen=True)
class ReadoutSpec:
    token_ids: tuple[int, ...]
    temperature: float
    softcap: float
    #: One-line notes for the startup log (an intermediate checkpoint without ``interface``).
    warnings: tuple[str, ...] = ()


def _positive_float32(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionConfigError(f"{name} must be a number, got {value!r}")
    v = float(value)
    if not (math.isfinite(v) and _FLOAT32_TINY <= v <= _FLOAT32_MAX):
        raise DecisionConfigError(
            f"{name} must be finite, above 0 and in float32's normal range, got {value!r}"
        )
    return v


def read_decision_config(
    dc: Any, *, text_softcap: Any, vocab_size: int, interface_optional: bool = True
) -> ReadoutSpec:
    """Validate ``dc`` (the parsed ``decision_config.json``) for serving.

    - ``interface``: when present it must be ``"strands-decider-v2"``. Intermediate checkpoints predate the field; they
      load with a warning (``interface_optional``).
    - ``format_version`` 1, ``attention_mode: prefix_bidir``, ``pooling: last``.
    - ``token_ids``: 255 distinct integers in ``[0, vocab_size)``.
    - ``temperature``: finite and above 0.
    - the soft-cap: ``final_logit_softcapping`` from this file when present, else the decoder config's (``text_softcap``);
      finite and above 0. It is never skipped: a checkpoint with neither is refused, and one whose two values differ
      is refused (specification, section 9).
    """
    if not isinstance(dc, dict):
        raise DecisionConfigError(f"expected a JSON object, got {type(dc).__name__}")
    warnings: list[str] = []
    if "interface" in dc:
        if dc["interface"] != INTERFACE:
            raise DecisionConfigError(
                f'interface must be "{INTERFACE}", got {dc["interface"]!r}; this is not a Strands Decider v2 '
                "checkpoint"
            )
    elif interface_optional:
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
    elif text is not None:
        softcap = text
    else:
        raise DecisionConfigError(
            "final_logit_softcapping is missing from both decision_config.json and the decoder config; the v2 "
            "readout is soft-capped, and the plugin does not serve it without the cap"
        )
    return ReadoutSpec(tuple(ids), temperature, softcap, tuple(warnings))
