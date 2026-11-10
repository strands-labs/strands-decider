"""`decision_config.json`: what a runtime needs, beyond the weights, to serve a checkpoint.

The schema is docs/spec.md's (The decision_config.json schema). Validation is strict: a
file without `"interface": "strands-decider"`, a missing required field, an unknown
field, a value of the wrong type, and a prompt this runtime does not implement are all
errors. The file carries the soft-cap, so a converted decoder (GGUF, MLX) needs no
`config.json` beside it. It names the inputs the checkpoint takes (`inputs`: text, and
images where it serves them), and its provenance is required: the image processor, the
tokenizer and any part not fine-tuned come from the base model at `base_revision`.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from .codes import CODES
from .prompt import VARIANTS

FILENAME = "decision_config.json"
# Tells this interface's decision config from other formats that use the same file names
# (llama.cpp's pplx-decider loader claims any directory with decision_config.json and
# readout.safetensors).
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

# The readout's soft-cap and temperature act in float32 (docs/spec.md, The readout), so
# each must be a normal float32: a value past these bounds would become inf or lose precision.
F32_MAX = 3.4028234663852886e38
F32_MIN_NORMAL = 1.1754943508222875e-38


def check_float32_scale(name: str, v: Any) -> float:
    """`v` as a float, if it is a finite number above 0 that float32 holds as a normal
    number (about 1.2e-38 to 3.4e38); otherwise ValueError. For the soft-cap and the
    temperature, which divide the readout logits in float32."""
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not ok or not F32_MIN_NORMAL <= v <= F32_MAX:
        raise ValueError(
            f"{name} must be a finite number above 0 within float32's normal range "
            f"({F32_MIN_NORMAL:.3g} to {F32_MAX:.3g}), not {v!r}"
        )
    return float(v)


_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_RELEASE = re.compile(r"^([A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _wrong_interface(interface: Any) -> str:
    return (
        f"not a Strands Decider decision config: {FILENAME} needs "
        f'"interface": "{INTERFACE}", and has {interface!r}'
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
        if not _HUB_ID.match(self.base_model):
            raise ValueError(f"provenance.base_model must be a Hub id, not {self.base_model!r}")
        if not _COMMIT.match(self.base_revision):
            raise ValueError("provenance.base_revision must be a 40-character commit hash")
        if self.release is not None and not _RELEASE.match(self.release):
            raise ValueError(f"provenance.release must be a model name, not {self.release!r}")
        if self.source is not None and not self.source.startswith("https://"):
            raise ValueError("provenance.source must be an https URL")

    @classmethod
    def from_dict(cls, d: Any) -> Provenance:
        if not isinstance(d, dict):
            raise ValueError("provenance must be an object")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"provenance has unknown fields {unknown}; known: {sorted(known)}")
        missing = [k for k in PROVENANCE_REQUIRED if k not in d]
        if missing:
            raise ValueError(f"provenance is missing required fields {missing}")
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
            raise ValueError(
                f"format_version {self.format_version!r} is not supported; this runtime "
                f"reads format_version {FORMAT_VERSION}"
            )
        if not isinstance(self.prompt, str) or self.prompt not in VARIANTS:
            raise ValueError(
                f"prompt {self.prompt!r} is not implemented; this runtime implements {VARIANTS}"
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
                f"{TEXT!r}, not {inputs!r}"
            )
        object.__setattr__(self, "inputs", tuple(inputs))
        if tuple(self.codes) != CODES:
            raise ValueError(
                "codes must be the 255 answer codes, A through IV without GZ, in order"
            )
        ids = tuple(self.token_ids)
        if len(ids) != len(CODES) or not all(_is_int(i) and i >= 0 for i in ids):
            raise ValueError(f"token_ids must be {len(CODES)} non-negative integers")
        if len(set(ids)) != len(ids):
            raise ValueError("token_ids must be distinct: one token per code")
        check_float32_scale("temperature", self.temperature)
        check_float32_scale("final_logit_softcapping", self.final_logit_softcapping)
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"attention_mode {self.attention_mode!r} is not supported; expected {ATTENTION_MODES}"
            )
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling {self.pooling!r} is not supported; expected {POOLINGS}")
        if not _is_int(self.max_length_validated) or self.max_length_validated <= 0:
            raise ValueError("max_length_validated must be a positive integer")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")

    @classmethod
    def from_dict(cls, d: Any) -> DecisionConfig:
        if not isinstance(d, dict):
            raise ValueError(f"{FILENAME} must hold a JSON object")
        # First, so that another format's file gets this message, not a list of fields.
        if d.get("interface") != INTERFACE:
            raise ValueError(_wrong_interface(d.get("interface", "<missing>")))
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
            raise ValueError(f"inputs must be an array, not {d['inputs']!r}")
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
