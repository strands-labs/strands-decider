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
- config.json carries a marker (`mlx_format.ARTIFACT`), so that the router refuses the
  artifact on any device but mlx: only mlx-vlm reads its layout.
- The artifact is written into a directory beside the destination (`<dest>.partial-*`),
  reloaded and checked (the same decision files, the marker, the decoder quantized as asked
  and nothing else quantized), and only then renamed to the destination, with the usual
  permissions. A conversion that fails or is rejected leaves nothing that loads; one that was
  killed leaves its partial directory, which the next conversion to the same destination
  removes.

A quantized artifact makes no conformance claim unless docs/spec.md section 10 records bounds
for its format at that size (`mlx_format.RECORDED`: MLX 8-bit, group 64, at E2B), and
converting to it, or loading it, says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..core import DecisionConfig, Readout, load_tokenizer
from ..core.config import FILENAME as DECISION_CONFIG
from ..core.readout import READOUT_FILE
from .mlx import (
    AUDIO_MODULES,
    INDEX,
    PROCESSOR_CONFIG,
    TOKENIZER,
    VISION_MODULES,
    load_mlx_model,
    model_type,
)
from .mlx_format import ARTIFACT, ARTIFACT_KEY, bases_form, conformance, is_mlx_artifact

BITS = (4, 8)
DEFAULT_GROUP_SIZE = 64
DECODER = "language_model."
# Copied byte for byte when present; the first five must be.
REQUIRED = (DECISION_CONFIG, READOUT_FILE, PROCESSOR_CONFIG, TOKENIZER, "tokenizer_config.json")
OPTIONAL = ("golden_vectors.jsonl", "LICENSE", "NOTICE", "README.md")
PARTIAL = ".partial-"  # <dest>.partial-<random>: a conversion in progress
OWNER = ".converting-pid"  # in a partial directory: the converting process's id


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's live process
        return True
    return True


def sweep_partials(dst: Path) -> list[Path]:
    """Remove the partial directories that killed conversions to `dst` left: those whose
    converting process no longer runs. Returns what it removed."""
    removed = []
    for p in dst.parent.glob(f"{dst.name}{PARTIAL}*"):
        try:
            pid = int((p / OWNER).read_text())
        except (OSError, ValueError):
            pid = None
        if pid is None or not _alive(pid):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
    return removed


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


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


def _source_dir(checkpoint: str | os.PathLike[str]) -> Path:
    """`checkpoint` as an absolute directory, or the Hub repo it names, downloaded."""
    p = os.fspath(checkpoint)
    if os.path.isdir(p):
        return Path(p).resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(p))
    except Exception as e:
        raise FileNotFoundError(
            f"{p}: not a local directory, and could not be downloaded from the Hugging Face "
            f"Hub: {type(e).__name__}: {e}"
        ) from e


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
    if bits is not None and bits not in BITS:
        raise ValueError(f"bits must be one of {BITS} (or none, for bf16), not {bits!r}")
    dst = Path(dest).resolve()
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"{dst} exists and is not an empty directory")

    import mlx.core as mx
    from mlx_vlm.quant_utils import quantize_model
    from mlx_vlm.utils import save_config, save_weights

    src = _source_dir(source)
    missing = [f for f in REQUIRED if not (src / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{src} is not a decision checkpoint: it has no {missing}")
    decision = DecisionConfig.load(src)
    Readout.load(src, decision)  # the readout's shape and soft-cap agree with the config
    load_tokenizer(src, decision)  # its control and code ids are the interface's
    model_type(src)
    with open(src / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    if is_mlx_artifact(config):
        raise ValueError(f"{src} is already an MLX artifact; convert the bf16 checkpoint")
    config = bases_form(config) or config  # the form mlx-vlm reads, written into the artifact
    config[ARTIFACT_KEY] = dict(ARTIFACT)

    claim = conformance(  # as the artifact will be: marked, its decoder quantized as asked
        {
            ARTIFACT_KEY: ARTIFACT,
            "quantization": {"bits": bits, "group_size": group_size, "mode": "affine"}
            if bits
            else None,
        },
        decision.provenance.base_model,
    )
    if claim["notice"]:
        warnings.warn(claim["notice"], stacklevel=2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sweep_partials(dst)
    work = Path(tempfile.mkdtemp(prefix=f"{dst.name}{PARTIAL}", dir=dst.parent))
    try:
        (work / OWNER).write_text(str(os.getpid()))
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
        summary = verify(src, work, bits=bits)
        (work / OWNER).unlink()
        # mkdtemp makes the directory 0700; an artifact gets the permissions any new
        # directory would, so that the account that serves it can read it.
        work.chmod(0o777 & ~_umask())
        os.rename(work, dst)  # replaces an empty `dst`, in one step
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(work)
        raise
    summary["artifact"] = str(dst)
    summary["notice"] = claim["notice"]
    summary["conformance"] = {k: v for k, v in claim.items() if k != "notice"}
    return summary


def verify(source: Path, artifact: Path, *, bits: int | None = None) -> dict[str, Any]:
    """Check an MLX artifact against its checkpoint: the copied files are the same bytes,
    the decision files load, config.json carries the marker and, with `bits`, a quantization
    of that width, the model loads strictly with its index, the decoder is quantized when
    `bits` is given and not otherwise, and nothing outside the decoder is. Returns a
    summary."""
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
    decoder_parts = parts.get(DECODER.rstrip("."), 0)
    if bits is not None and not decoder_parts:
        raise AssertionError(
            f"{artifact}: asked for {bits} bits, and no decoder layer is quantized"
        )
    if bits is None and decoder_parts:
        raise AssertionError(f"{artifact}: a bf16 artifact with {decoder_parts} quantized layers")
    with open(artifact / "config.json", encoding="utf-8") as f:
        model_config = json.load(f)
    quantization = model_config.get("quantization")
    if model_config.get(ARTIFACT_KEY) != ARTIFACT:
        raise AssertionError(f"{artifact}: config.json has no {ARTIFACT_KEY} marker")
    if (quantization or {}).get("bits") != bits:
        raise AssertionError(
            f"{artifact}: config.json's quantization is {quantization}, not {bits} bits"
        )
    files = sorted(n for n in os.listdir(artifact) if n != OWNER)
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
