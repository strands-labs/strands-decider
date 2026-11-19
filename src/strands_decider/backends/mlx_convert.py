"""MLX artifacts of a model checkpoint: `strands-decider convert-mlx`.

An artifact is the model in mlx-vlm's format, its decoder bf16 or quantized, beside the
checkpoint's decision files copied byte for byte, and its config.json marked
(`mlx_format.ARTIFACT`) so that only `--device mlx` serves it. `mlx_vlm.convert` doesn't make
one: it copies only *.json and *.py files, losing the readout, it quantizes the audio projection,
and it rewrites processor_config.json and tokenizer.json.
"""

from __future__ import annotations

import os
import secrets
import shutil
from pathlib import Path
from typing import Any

from ..core import DecisionConfig, load_readout, load_tokenizer
from ..core.config import FILENAME as DECISION_CONFIG
from ..core.images import PROCESSOR_CONFIG_FILE
from ..core.readout import READOUT_FILE
from ..core.tokenizer import TOKENIZER_FILE
from ..modeling import checkpoint_dir
from .mlx import load_mlx_model, model_config
from .mlx_format import ARTIFACT, ARTIFACT_KEY

BITS = (4, 8)
# The checkpoint's files an artifact holds, copied byte for byte: those of them it has.
COPIED = (
    DECISION_CONFIG,
    READOUT_FILE,
    TOKENIZER_FILE,
    PROCESSOR_CONFIG_FILE,
    "tokenizer_config.json",
    "golden_vectors.jsonl",
    "LICENSE",
    "NOTICE",
    "README.md",
)


def decoder_only(model: Any) -> Any:
    """mlx-vlm's quantization rule for the decoder's layers (`language_model.`), and no other."""
    rule = model.quant_predicate
    return lambda path, module: path.startswith("language_model.") and rule(path, module)


def convert(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    bits: int | None = None,
    group_size: int = 64,
) -> Path:
    """Write the MLX artifact of the model checkpoint `source` (a directory or a Hub id) to the
    new or empty directory `dest`: bf16, or the decoder quantized to `bits` in groups of
    `group_size`. It is written beside `dest`, reloaded, and only then renamed to `dest`, so a
    failed conversion leaves nothing behind. Returns the artifact's path."""
    import mlx.core as mx
    from mlx_vlm.quant_utils import quantize_model
    from mlx_vlm.utils import save_config, save_weights

    if bits is not None and bits not in BITS:
        raise ValueError(f"bits must be one of {BITS} (or none, for bf16), not {bits!r}")
    dst = Path(dest).resolve()
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"{dst} exists and is not an empty directory")
    src = Path(checkpoint_dir(os.fspath(source))).resolve()
    DecisionConfig.load(src)  # the files the runtime reads, checked before the long part
    load_readout(src)
    load_tokenizer(src)
    config = model_config(src)
    if ARTIFACT_KEY in config:
        raise ValueError(f"{src} is already an MLX artifact; convert the checkpoint it came from")
    config[ARTIFACT_KEY] = ARTIFACT
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = dst.with_name(f"{dst.name}.partial-{secrets.token_hex(4)}")
    work.mkdir()  # the permissions any new directory gets, so the serving account can read it
    try:
        model = load_mlx_model(src)
        if bits is not None:
            model, config = quantize_model(
                model, config, group_size, bits, quant_predicate=decoder_only(model)
            )
        save_weights(work, model, donate_weights=True)
        del model
        mx.clear_cache()
        save_config(config, config_path=work / "config.json")
        for name in COPIED:
            if (src / name).is_file():
                shutil.copyfile(src / name, work / name)
        load_mlx_model(work)  # strictly: every tensor the model has, of its shape
        os.rename(work, dst)  # replaces an empty `dst`, in one step
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return dst
