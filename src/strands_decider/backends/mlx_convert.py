"""MLX artifacts of a decision checkpoint: `strands-decider convert-mlx` (docs/spec.md section 9).

An MLX artifact carries the model in mlx-vlm's format, its decoder in bf16 or quantized, and
beside it the checkpoint's `readout.safetensors` and `decision_config.json` unchanged, its
`processor_config.json` and its tokenizer. `mlx_vlm.convert` alone does not make one: it
copies only `*.json` and `*.py` files, so the readout is lost; it quantizes the audio
projection, whose name is not on its skip list; and it rewrites `processor_config.json` and
`tokenizer.json`. Here the model is converted with mlx-vlm's own loader, quantizer and writer,
and everything else is copied byte for byte:

- Only the decoder (`language_model.`) is quantized, with mlx-vlm's own rule for it (a
  mixture-of-experts router stays at 8 bits). The vision encoder, its projection and, where
  the base has them, the audio encoder and its projection stay in the checkpoint's bf16. The
  readout is never part of the model.
- The weights are written with an index (`model.safetensors.index.json`), which mlx-vlm needs
  to tell the model's shards from the readout.
- The artifact is reloaded and checked: the same decision files, and no quantized module
  outside the decoder.

A quantized artifact makes no conformance claim until its bit width has been measured against
the reference (docs/spec.md section 10; `mlx.MEASURED_BITS`): 8-bit has, 4-bit has not, and
converting to it, or loading it, says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

from ..core import DecisionConfig, Readout
from ..core.config import FILENAME as DECISION_CONFIG
from ..core.readout import READOUT_FILE
from . import checkpoint_path
from .mlx import (
    AUDIO_MODULES,
    INDEX,
    PROCESSOR_CONFIG,
    TOKENIZER,
    VISION_MODULES,
    conformance_notice,
    load_mlx_model,
    model_type,
)

BITS = (4, 8)
DEFAULT_GROUP_SIZE = 64
DECODER = "language_model."
# Copied byte for byte when present; the first five must be.
REQUIRED = (DECISION_CONFIG, READOUT_FILE, PROCESSOR_CONFIG, TOKENIZER, "tokenizer_config.json")
OPTIONAL = ("golden_vectors.jsonl", "LICENSE", "NOTICE", "README.md")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def decoder_only(model: Any) -> Any:
    """A quantization predicate for mlx-vlm's quantizer: the decoder's layers, by the
    model's own rule for them, and nothing else."""
    own = getattr(model, "quant_predicate", None)

    def predicate(path: str, module: Any) -> Any:
        if not path.startswith(DECODER) or not hasattr(module, "to_quantized"):
            return False
        return own(path, module) if own is not None else True

    return predicate


def quantized_parts(model: Any) -> dict[str, int]:
    """How many quantized modules each top-level part of `model` holds."""
    out: dict[str, int] = {}
    for path, module in model.named_modules():
        if "scales" in module and "weight" in module:
            top = path.split(".")[0]
            out[top] = out.get(top, 0) + 1
    return out


def convert(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    bits: int | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, Any]:
    """Write the MLX artifact of the decision checkpoint `source` (a directory or a Hub repo
    id) to the new directory `dest`: bf16, or with the decoder quantized to `bits` (8 or 4)
    in groups of `group_size`. Returns a summary: the files written, their sizes, and which
    parts are quantized."""
    import mlx.core as mx
    from mlx_vlm.quant_utils import quantize_model
    from mlx_vlm.utils import save_config, save_weights

    src = checkpoint_path(source)
    dst = Path(dest).resolve()
    if bits is not None and bits not in BITS:
        raise ValueError(f"bits must be one of {BITS} (or none, for bf16), not {bits!r}")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty")
    missing = [f for f in REQUIRED if not (src / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{src} is not a decision checkpoint: it has no {missing}")
    decision = DecisionConfig.load(src)
    Readout.load(src, decision)  # the readout's shape and soft-cap agree with the config
    model_type(src)
    with open(src / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    if "quantization" in config:
        raise ValueError(f"{src} is already a quantized MLX artifact; convert the bf16 checkpoint")

    notice = conformance_notice({"quantization": {"bits": bits}} if bits else {})
    if notice:
        warnings.warn(notice, stacklevel=2)
    # Written beside `dest` and renamed into place once verified, so that a failed or rejected
    # conversion never leaves a directory that loads like an artifact.
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent))
    try:
        model = load_mlx_model(src, keep=AUDIO_MODULES + VISION_MODULES)
        if bits is not None:
            model, config = quantize_model(
                model, config, group_size, bits, quant_predicate=decoder_only(model)
            )
        save_weights(work, model, donate_weights=True)
        del model
        mx.clear_cache()
        save_config(config, config_path=work / "config.json")
        for name in REQUIRED + OPTIONAL:
            if (src / name).is_file():
                shutil.copyfile(src / name, work / name)
        summary = verify(src, work)
        if dst.exists():
            dst.rmdir()  # empty, as checked above
        work.rename(dst)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    summary["artifact"] = str(dst)
    summary["notice"] = notice
    return summary


def verify(source: Path, artifact: Path) -> dict[str, Any]:
    """Check an MLX artifact against its checkpoint: the copied files are the same bytes,
    the decision files load, the model loads strictly with its index, and nothing outside
    the decoder is quantized. Returns a summary."""
    copied = [n for n in REQUIRED + OPTIONAL if (source / n).is_file()]
    changed = [n for n in copied if _sha256(source / n) != _sha256(artifact / n)]
    if changed:
        raise AssertionError(f"{artifact}: {changed} differ from the checkpoint's")
    if not (artifact / INDEX).is_file():
        raise AssertionError(f"{artifact} has no {INDEX}")
    decision = DecisionConfig.load(artifact)
    Readout.load(artifact, decision)
    model = load_mlx_model(artifact, keep=AUDIO_MODULES + VISION_MODULES)
    parts = quantized_parts(model)
    outside = sorted(p for p in parts if p != DECODER.rstrip("."))
    if outside:
        raise AssertionError(f"{artifact}: {outside} are quantized; only the decoder may be")
    with open(artifact / "config.json", encoding="utf-8") as f:
        quantization = json.load(f).get("quantization")
    files = sorted(os.listdir(artifact))
    return {
        "artifact": str(artifact),
        "quantization": (
            {k: quantization[k] for k in ("bits", "group_size", "mode") if k in quantization}
            if quantization
            else None
        ),
        "quantized_modules": parts,
        "modules": sorted(name for name, child in model.items() if child is not None),
        "files": {n: (artifact / n).stat().st_size for n in files},
        "bytes": sum((artifact / n).stat().st_size for n in files),
    }
