"""`decision_config.json`: what a runtime needs, beyond the weights, to serve a checkpoint.

The schema is docs/spec.md's (section 8). Validation is strict: a file without `"interface":
"strands-decider"`, a missing required field, an unknown field, a value of the wrong type, and
a prompt this runtime does not implement are all errors. The file carries the soft-cap, so a
converted decoder (GGUF, MLX) needs no `config.json` beside it. It names the inputs the
checkpoint takes (`inputs`: text, and images where it serves them), and its provenance is
required: the image processor, the tokenizer and any part not fine-tuned come from the base
model at `base_revision`.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .codes import CODES
from .prompt import VARIANTS

FILENAME = "decision_config.json"
# Tells this interface's decision config from other decision-model formats that use the same
# file names (converters, such as llama.cpp's, claim any directory with decision_config.json
# and readout.safetensors).
INTERFACE = "strands-decider"
FORMAT_VERSION = 1
ATTENTION_MODES = ("prefix_bidir",)
POOLINGS = ("last",)
# The inputs a checkpoint may take (`inputs`); every checkpoint takes text.
TEXT = "text"
IMAGE = "image"
INPUTS = (TEXT, IMAGE)

REQUIRED = (
    "interface",
    "format_version",
    "prompt",
    "inputs",
    "codes",
    "token_ids",
    "temperature",
    "final_logit_softcapping",
    "attention_mode",
    "pooling",
    "max_length_validated",
    "provenance",
)
OPTIONAL: tuple[str, ...] = ()
# The provenance fields every checkpoint gives; `release` and `source` are optional.
PROVENANCE_REQUIRED = ("base_model", "base_revision")

# The readout's soft-cap and temperature act in float32 (docs/spec.md, section 7), so
# each must be a normal float32: a value past these bounds would become inf or lose precision.
F32_MAX = 3.4028234663852886e38
F32_MIN_NORMAL = 1.1754943508222875e-38
# Token ids and max_length_validated fit a signed 32-bit integer, so that every runtime can
# hold them (docs/spec.md, section 8).
INT32_MAX = 2**31 - 1


def _real(v: Any) -> float | None:
    """`v` as a float if it is a real number (Python's or numpy's, not a bool) that a
    double holds finite; otherwise None. An int past a double's range is not one."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


def check_float32_scale(name: str, v: Any) -> float:
    """`v` as a float, if it is a finite number above 0 that float32 holds as a normal
    number (about 1.2e-38 to 3.4e38); otherwise ValueError. For the soft-cap and the
    temperature, which divide the readout logits in float32. numpy scalars count as numbers;
    bools and strings do not, and neither does an integer too large for a double."""
    f = _real(v)
    if f is None or not F32_MIN_NORMAL <= f <= F32_MAX:
        raise ValueError(
            f"{name} must be a finite number above 0 within float32's normal range "
            f"({F32_MIN_NORMAL:.3g} to {F32_MAX:.3g}), not {_shown(v)}"
        )
    return f


def _shown(v: Any) -> str:
    """`v` for a message: an integer past a double's range by its size, since its digits may
    be too many to print (Python refuses to), a list or tuple item by item, and anything long
    cut short."""
    if isinstance(v, int) and not isinstance(v, bool) and abs(v).bit_length() > 1024:
        return f"an integer of {abs(v).bit_length()} bits"
    if isinstance(v, (list, tuple)):
        items = ", ".join(_shown(x) for x in v[:8]) + (", ..." if len(v) > 8 else "")
        return f"[{items}]" if isinstance(v, list) else f"({items})"
    text = repr(v)
    return text if len(text) <= 200 else text[:200] + "..."


# The provenance formats (docs/spec.md, section 8), ASCII and whole.
_HUB_NAME = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
_HUB_ID = re.compile(rf"{_HUB_NAME}/[A-Za-z0-9_.-]+", re.ASCII)
_RELEASE = re.compile(rf"({_HUB_NAME}/)?{_HUB_NAME}", re.ASCII)
_COMMIT = re.compile(r"[0-9a-f]{40}", re.ASCII)
_HTTPS_URL = re.compile(r"https://[!-~]+", re.ASCII)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _wrong_interface(interface: Any) -> str:
    return (
        f"not a Strands Decider decision config: {FILENAME} needs "
        f'"interface": "{INTERFACE}", and has {_shown(interface)}'
    )


def _unsupported_version(version: Any) -> str:
    return (
        f"format_version {_shown(version)} is not supported; this runtime reads "
        f"format_version {FORMAT_VERSION}"
    )


@dataclass(frozen=True)
class Provenance:
    """Where a checkpoint came from, in public terms. `base_model` and `base_revision` are
    required: the checkpoint's image processor, tokenizer and any part not fine-tuned are
    the base's at that revision.

    base_model: the Hub id of the base model, such as `google/gemma-4-E2B-it`.
    base_revision: the base model's 40-character commit on the Hub.
    release: optional, the released model's name or Hub id.
    source: optional, an https URL of the public code and recipe that trained it.
    """

    base_model: str
    base_revision: str
    release: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is None and name not in PROVENANCE_REQUIRED:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"provenance.{name} must be a non-empty string")
        if not _HUB_ID.fullmatch(self.base_model):
            raise ValueError(
                f"provenance.base_model must be a Hub id, not {_shown(self.base_model)}"
            )
        if not _COMMIT.fullmatch(self.base_revision):
            raise ValueError("provenance.base_revision must be a 40-character commit hash")
        if self.release is not None and not _RELEASE.fullmatch(self.release):
            raise ValueError(f"provenance.release must be a model name, not {_shown(self.release)}")
        if self.source is not None and not _HTTPS_URL.fullmatch(self.source):
            raise ValueError("provenance.source must be an https URL")

    @classmethod
    def from_dict(cls, d: Any) -> Provenance:
        """From the file's `provenance` object. A field given as null is an error: an
        optional field that does not apply is left out."""
        if not isinstance(d, dict):
            raise ValueError("provenance must be an object")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"provenance has unknown fields {unknown}; known: {sorted(known)}")
        missing = [k for k in PROVENANCE_REQUIRED if k not in d]
        if missing:
            raise ValueError(f"provenance is missing required fields {missing}")
        nulls = [k for k in d if d[k] is None]
        if nulls:
            raise ValueError(
                f"provenance.{nulls[0]} is null; a provenance field is a string, and an "
                "optional one that does not apply is left out"
            )
        return cls(**d)

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class DecisionConfig:
    interface: str
    format_version: int
    prompt: str
    inputs: tuple[str, ...]
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float
    final_logit_softcapping: float
    attention_mode: str
    pooling: str
    max_length_validated: int
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.interface != INTERFACE:
            raise ValueError(_wrong_interface(self.interface))
        if not _is_int(self.format_version) or self.format_version != FORMAT_VERSION:
            raise ValueError(_unsupported_version(self.format_version))
        if not isinstance(self.prompt, str) or self.prompt not in VARIANTS:
            raise ValueError(
                f"prompt {_shown(self.prompt)} is not implemented; this runtime implements {VARIANTS}"
            )
        inputs = self.inputs
        if (
            not isinstance(inputs, (list, tuple))
            or not all(isinstance(i, str) and i in INPUTS for i in inputs)
            or len(set(inputs)) != len(inputs)
            or TEXT not in inputs
        ):
            raise ValueError(
                f"inputs must be an array of distinct strings from {list(INPUTS)} that holds "
                f"{TEXT!r}, not {_shown(inputs)}"
            )
        object.__setattr__(self, "inputs", tuple(inputs))
        for name in ("codes", "token_ids"):
            if not isinstance(getattr(self, name), (list, tuple)):
                raise ValueError(f"{name} must be an array, not {_shown(getattr(self, name))}")
        if tuple(self.codes) != CODES:
            raise ValueError(
                "codes must be the 255 answer codes, A through IV without GZ, in order"
            )
        ids = tuple(self.token_ids)
        if len(ids) != len(CODES) or not all(_is_int(i) and 0 <= i <= INT32_MAX for i in ids):
            raise ValueError(
                f"token_ids must be {len(CODES)} non-negative integers, each at most {INT32_MAX}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("token_ids must be distinct: one token per code")
        t = check_float32_scale("temperature", self.temperature)
        c = check_float32_scale("final_logit_softcapping", self.final_logit_softcapping)
        # Held as Python floats, so a numpy scalar given here neither reaches to_dict, which
        # json.dumps must take, nor compares differently in numpy 1 and 2.
        object.__setattr__(self, "temperature", t)
        object.__setattr__(self, "final_logit_softcapping", c)
        # The readout's largest final logit is c / T in float32; it must be finite there.
        with np.errstate(over="ignore"):
            if not np.isfinite(np.float32(c) / np.float32(t)):
                raise ValueError(
                    f"final_logit_softcapping / temperature ({c!r} / {t!r}) overflows float32, "
                    "where the readout divides the capped logits by the temperature"
                )
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"attention_mode {_shown(self.attention_mode)} is not supported; expected {ATTENTION_MODES}"
            )
        if self.pooling not in POOLINGS:
            raise ValueError(
                f"pooling {_shown(self.pooling)} is not supported; expected {POOLINGS}"
            )
        if not _is_int(self.max_length_validated) or not 0 < self.max_length_validated <= INT32_MAX:
            raise ValueError(
                f"max_length_validated must be a positive integer, at most {INT32_MAX}"
            )
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")

    @classmethod
    def from_dict(cls, d: Any) -> DecisionConfig:
        if not isinstance(d, dict):
            raise ValueError(f"{FILENAME} must hold a JSON object")
        # First, so that another format's file gets this message, not a list of fields; then
        # the version, so that a later layout's file is named as one, not as unknown fields.
        if d.get("interface") != INTERFACE:
            raise ValueError(_wrong_interface(d.get("interface", "<missing>")))
        version = d.get("format_version")
        if "format_version" in d and (not _is_int(version) or version != FORMAT_VERSION):
            raise ValueError(_unsupported_version(version))
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise ValueError(f"{FILENAME} is missing required fields {missing}")
        unknown = sorted(set(d) - set(REQUIRED) - set(OPTIONAL))
        if unknown:
            raise ValueError(
                f"{FILENAME} has unknown fields {unknown}; the schema defines "
                f"{list(REQUIRED + OPTIONAL)}"
            )
        for k in ("codes", "token_ids"):
            if not isinstance(d[k], list):
                raise ValueError(f"{k} must be a list")
        if not isinstance(d["inputs"], list):
            raise ValueError(f"inputs must be an array, not {_shown(d['inputs'])}")
        return cls(
            interface=d["interface"],
            format_version=d["format_version"],
            prompt=d["prompt"],
            inputs=tuple(d["inputs"]),
            codes=tuple(d["codes"]),
            token_ids=tuple(d["token_ids"]),
            temperature=d["temperature"],
            final_logit_softcapping=d["final_logit_softcapping"],
            attention_mode=d["attention_mode"],
            pooling=d["pooling"],
            max_length_validated=d["max_length_validated"],
            provenance=Provenance.from_dict(d["provenance"]),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> DecisionConfig:
        """From a `decision_config.json`, or a checkpoint directory holding one."""
        p = os.fspath(path)
        if os.path.isdir(p):
            p = os.path.join(p, FILENAME)
        with open(p, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "interface": self.interface,
            "format_version": self.format_version,
            "prompt": self.prompt,
            "inputs": list(self.inputs),
            "codes": list(self.codes),
            "token_ids": list(self.token_ids),
            "temperature": self.temperature,
            "final_logit_softcapping": self.final_logit_softcapping,
            "attention_mode": self.attention_mode,
            "pooling": self.pooling,
            "max_length_validated": self.max_length_validated,
            "provenance": self.provenance.to_dict(),
        }
        return d
