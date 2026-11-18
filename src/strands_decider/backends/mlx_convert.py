"""MLX artifacts of a model checkpoint: `strands-decider convert-mlx`.

An MLX artifact holds the model in mlx-vlm's format, its decoder in bf16 or quantized, beside
the checkpoint's own decision files copied byte for byte. `mlx_vlm.convert` alone doesn't make
one: it copies only `*.json` and `*.py` files, so the readout is lost; it quantizes the audio
projection, whose name isn't on its skip list; and it rewrites `processor_config.json` and
`tokenizer.json`. So the model is converted here with mlx-vlm's own loader, quantizer and
writer, and only the decoder (`language_model.`) is quantized, by mlx-vlm's own rule for it.

The weights are written with an index, which mlx-vlm needs to tell the model's shards from the
readout, and config.json carries a marker (`mlx_format.ARTIFACT`), so the router refuses the
artifact on any device but mlx. The artifact is written beside the destination, reloaded, and
only then renamed into place, so a failed conversion leaves nothing that loads.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..core import DecisionConfig, load_readout, load_tokenizer
from ..core.config import FILENAME as DECISION_CONFIG
from ..core.readout import READOUT_FILE
from ..decision import checkpoint_dir
from .mlx import (
    AUDIO_MODULES,
    PROCESSOR_CONFIG,
    TOKENIZER,
    VISION_MODULES,
    load_mlx_model,
    model_type,
)
from .mlx_format import ARTIFACT, ARTIFACT_KEY, bases_form, is_mlx_artifact

BITS = (4, 8)
DEFAULT_GROUP_SIZE = 64
DECODER = "language_model."
# Copied byte for byte when present; the first five must be.
REQUIRED = (DECISION_CONFIG, READOUT_FILE, PROCESSOR_CONFIG, TOKENIZER, "tokenizer_config.json")
OPTIONAL = ("golden_vectors.jsonl", "LICENSE", "NOTICE", "README.md")
PARTIAL = ".partial-"  # <dest>.partial-<random>: a conversion in progress


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


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
    """Write the MLX artifact of the model checkpoint `source` (a directory or a Hub repo
    id) to the new directory `dest`: bf16, or with the decoder quantized to `bits` (8 or 4) in
    groups of `group_size`. Returns a summary of what was written."""
    if bits is not None and bits not in BITS:
        raise ValueError(f"bits must be one of {BITS} (or none, for bf16), not {bits!r}")
    dst = Path(dest).resolve()
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"{dst} exists and is not an empty directory")

    import mlx.core as mx
    from mlx_vlm.quant_utils import quantize_model
    from mlx_vlm.utils import save_config, save_weights

    found = checkpoint_dir(os.fspath(source))
    if found is None:
        raise FileNotFoundError(f"{source}: not a local directory or a Hugging Face Hub repo")
    src = Path(found).resolve()
    missing = [f for f in REQUIRED if not (src / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{src} is not a model checkpoint: it has no {missing}")
    DecisionConfig.load(src)
    load_readout(src)
    load_tokenizer(src)
    model_type(src)
    with open(src / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    if is_mlx_artifact(config):
        raise ValueError(f"{src} is already an MLX artifact; convert the bf16 checkpoint")
    config = bases_form(config) or config  # the form mlx-vlm reads, written into the artifact
    config[ARTIFACT_KEY] = dict(ARTIFACT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{dst.name}{PARTIAL}", dir=dst.parent))
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
        summary = verify(work)
        # mkdtemp makes the directory 0700; an artifact gets the permissions any new directory
        # would, so that the account that serves it can read it.
        work.chmod(0o777 & ~_umask())
        os.rename(work, dst)  # replaces an empty `dst`, in one step
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(work)
        raise
    summary["artifact"] = str(dst)
    return summary


def verify(artifact: Path) -> dict[str, Any]:
    """That an MLX artifact loads, strictly, with its decision files; returns a summary: its
    quantization, its quantized modules by part, and its files' sizes."""
    DecisionConfig.load(artifact)
    load_readout(artifact)
    model = load_mlx_model(artifact, keep=AUDIO_MODULES + VISION_MODULES)
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
        "quantized_modules": quantized_parts(model),
        "files": {n: (artifact / n).stat().st_size for n in files},
        "bytes": sum((artifact / n).stat().st_size for n in files),
    }
