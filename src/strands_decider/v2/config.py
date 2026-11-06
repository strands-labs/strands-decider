"""`decision_config.json`: what a runtime needs, beyond the weights, to serve a checkpoint.

The schema is docs/v2-spec.md's (The decision_config.json schema). Validation is strict: a
missing required field, an unknown field, a value of the wrong type, and a prompt this
runtime does not implement are all errors. The optional `provenance` block is public by
construction: its fields are fixed, and none may hold a filesystem path or a non-https URI.
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
FORMAT_VERSION = 1
ATTENTION_MODES = ("prefix_bidir",)
POOLINGS = ("last",)

REQUIRED = (
    "format_version",
    "prompt",
    "codes",
    "token_ids",
    "temperature",
    "attention_mode",
    "pooling",
    "max_length_validated",
)
OPTIONAL = ("provenance",)

_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_RELEASE = re.compile(r"^([A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _internal(s: str) -> str | None:
    """Why `s` looks like a private location, or None. A public config names Hub ids and
    https URLs, never a path on a training host or a bucket."""
    if s.startswith(("/", "~", "./", "../")) or "\\" in s or _WINDOWS_PATH.match(s):
        return "a filesystem path"
    if "/fsx/" in s or s.startswith("fsx/"):
        return "a filesystem path"
    if _URI_SCHEME.match(s) and not s.startswith("https://"):
        return "a non-https URI"
    return None


@dataclass(frozen=True)
class Provenance:
    """Where a checkpoint came from, in public terms. Every field is optional.

    base_model: the Hub id of the base model, such as `google/gemma-4-E2B-it`.
    base_revision: the base model's 40-character commit on the Hub.
    release: the released model's name or Hub id.
    source: an https URL of the public code and recipe that trained it.
    """

    base_model: str | None = None
    base_revision: str | None = None
    release: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"provenance.{name} must be a non-empty string")
            why = _internal(value)
            if why:
                raise ValueError(f"provenance.{name} is {why} ({value!r}); provenance is public")
        if self.base_model is not None and not _HUB_ID.match(self.base_model):
            raise ValueError(f"provenance.base_model must be a Hub id, not {self.base_model!r}")
        if self.base_revision is not None and not _COMMIT.match(self.base_revision):
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
        return cls(**d)

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class DecisionConfig:
    format_version: int
    prompt: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: float
    attention_mode: str
    pooling: str
    max_length_validated: int
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.format_version) or self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"format_version {self.format_version!r} is not supported; this runtime "
                f"reads format_version {FORMAT_VERSION}"
            )
        if not isinstance(self.prompt, str) or self.prompt not in VARIANTS:
            raise ValueError(
                f"prompt {self.prompt!r} is not implemented; this runtime implements {VARIANTS}"
            )
        if tuple(self.codes) != CODES:
            raise ValueError("codes must be the 255 v2 codes, A through IV without GZ, in order")
        ids = tuple(self.token_ids)
        if len(ids) != len(CODES) or not all(_is_int(i) and i >= 0 for i in ids):
            raise ValueError(f"token_ids must be {len(CODES)} non-negative integers")
        if len(set(ids)) != len(ids):
            raise ValueError("token_ids must be distinct: one token per code")
        t = self.temperature
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t) or t <= 0:
            raise ValueError(f"temperature must be a finite number above 0, not {t!r}")
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"attention_mode {self.attention_mode!r} is not supported; expected {ATTENTION_MODES}"
            )
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling {self.pooling!r} is not supported; expected {POOLINGS}")
        if not _is_int(self.max_length_validated) or self.max_length_validated <= 0:
            raise ValueError("max_length_validated must be a positive integer")
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")

    @classmethod
    def from_dict(cls, d: Any) -> DecisionConfig:
        if not isinstance(d, dict):
            raise ValueError(f"{FILENAME} must hold a JSON object")
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
        prov = d.get("provenance")
        return cls(
            format_version=d["format_version"],
            prompt=d["prompt"],
            codes=tuple(d["codes"]),
            token_ids=tuple(d["token_ids"]),
            temperature=d["temperature"],
            attention_mode=d["attention_mode"],
            pooling=d["pooling"],
            max_length_validated=d["max_length_validated"],
            provenance=None if prov is None else Provenance.from_dict(prov),
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
            "format_version": self.format_version,
            "prompt": self.prompt,
            "codes": list(self.codes),
            "token_ids": list(self.token_ids),
            "temperature": self.temperature,
            "attention_mode": self.attention_mode,
            "pooling": self.pooling,
            "max_length_validated": self.max_length_validated,
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d
