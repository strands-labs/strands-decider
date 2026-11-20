"""`decision_config.json`: what a runtime needs, beyond the weights, to serve a checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .codes import CODE_TOKEN_IDS, CODES

FILENAME = "decision_config.json"
# Names this format, since other decision models ship files with the same names.
INTERFACE = "strands-decider"
FORMAT_VERSION = 1
TEXT = "text"
IMAGE = "image"

Positive = Annotated[float, Field(gt=0, strict=True)]


class Provenance(BaseModel):
    """The base the checkpoint was fine-tuned from; its tokenizer, image processor and every
    part not fine-tuned are that base's files at that commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str  # its Hub id, such as google/gemma-4-E2B-it
    base_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    # The release this checkpoint was part of, as published; v2 checkpoints ship it.
    release: str | None = None


class DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    interface: Literal["strands-decider"]
    format_version: Literal[1]
    prompt: Literal["strands-chat"]
    inputs: tuple[Literal["text", "image"], ...]
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    temperature: Positive
    final_logit_softcapping: Positive
    attention_mode: Literal["prefix_bidir"]
    pooling: Literal["last"]
    max_length_validated: Annotated[int, Field(gt=0, strict=True)]
    provenance: Provenance

    @model_validator(mode="after")
    def _gemma4(self) -> DecisionConfig:
        if TEXT not in self.inputs:
            raise ValueError("inputs must include text")
        if self.codes != CODES or self.token_ids != CODE_TOKEN_IDS:
            raise ValueError("codes and token_ids must be Gemma 4's 255 answer codes and their ids")
        return self

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> DecisionConfig:
        """From a `decision_config.json`, or a checkpoint directory holding one."""
        p = Path(path)
        return cls.model_validate_json((p / FILENAME if p.is_dir() else p).read_text("utf-8"))
