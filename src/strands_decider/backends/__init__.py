"""Runtimes for decision checkpoints (docs/architecture.md, The core's runtime interface): each
implements `strands_decider.core.Backend` for one framework."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.config import FILENAME as DECISION_CONFIG


def is_decision_checkpoint(checkpoint: str | os.PathLike[str]) -> bool:
    """Whether `checkpoint`, a local directory or a Hub repo id, holds a decision_config.json:
    a decision checkpoint (or a converted artifact of one) rather than a v1 checkpoint. The
    file itself is validated when the checkpoint loads, so another format's file is refused
    there, with its reason."""
    p = os.fspath(checkpoint)
    if os.path.isdir(p):
        return os.path.isfile(os.path.join(p, DECISION_CONFIG))
    if os.path.exists(p):
        return False
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(p, DECISION_CONFIG)
    except Exception:  # not a repo id, no such file, offline: nothing a backend can load
        return False
    return True


def checkpoint_path(checkpoint: str | os.PathLike[str]) -> Path:
    """`checkpoint` if it is a local directory, else the Hub repo it names, downloaded."""
    p = os.fspath(checkpoint)
    if os.path.isdir(p):
        return Path(p)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(p))
    except Exception as e:
        raise FileNotFoundError(
            f"{p}: not a local directory, and could not be downloaded from the Hugging Face "
            f"Hub: {type(e).__name__}: {e}"
        ) from e
