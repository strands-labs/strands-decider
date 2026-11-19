"""Model checkpoints and their MLX artifacts as mlx-vlm 0.7.12 reads them: `load_mlx_model`
loads either, and `convert` (`strands-decider convert-mlx`) writes an artifact. MLX is imported
only to load or write one, so any platform can tell an artifact (`is_mlx_artifact`).

An artifact is the model in mlx-vlm's format, its decoder bf16 or quantized, beside the
checkpoint's decision files copied byte for byte, and its config.json marked so that only
`--device mlx` serves it. `mlx_vlm.convert` doesn't make one: it copies only *.json and *.py
files, losing the readout, it quantizes the audio projection, and it rewrites
processor_config.json and tokenizer.json.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..core import DecisionConfig
from ..core.config import FILENAME as DECISION_CONFIG
from ..core.images import PROCESSOR_CONFIG_FILE
from ..core.readout import READOUT_FILE
from ..core.tokenizer import TOKENIZER_FILE
from ..modeling import checkpoint_dir

MLX_EXTRA = (
    "model checkpoints on MLX need macOS on Apple silicon and the mlx extra: "
    "pip install 'strands-decider[mlx]'"
)
# The mlx-vlm release the MLX backend is tested on, and the only one it loads with: it relies on
# behaviour no release promises (a passed mask reaching every layer as it is, `logits_to_keep`
# through the KV-shared layers, its caches and image processors).
_MLX_VLM = "0.7.12"
_MARKER = "strands_decider_artifact"  # convert's field in an artifact's config.json
_MODEL_TYPES = ("gemma4", "gemma4_unified")  # the 12B is gemma4_unified
_SHARD = re.compile(r"model(-\d+-of-\d+)?\.safetensors")  # as transformers and mlx-vlm name them
_BITS = (4, 8)
# The checkpoint's files an artifact holds, copied byte for byte: those of them it has.
_COPIED = (
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


def is_mlx_artifact(path: str | os.PathLike[str]) -> bool:
    """Whether the checkpoint directory `path` is an MLX artifact."""
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        return _MARKER in json.load(f)


def model_config(path: Path) -> dict[str, Any]:
    """The checkpoint's config.json as mlx-vlm 0.7.12 reads it. transformers 5.2x gives each
    full-attention layer its head size and key/value heads in `text_config.per_layer_config`;
    mlx-vlm, like the Gemma 4 bases, reads one `global_head_dim` and one
    `num_global_key_value_heads` for them all, and without them builds those layers with its own
    defaults. ValueError unless the checkpoint is a Gemma 4 base's multimodal one."""
    config: dict[str, Any] = json.loads((path / "config.json").read_text("utf-8"))
    if config.get("model_type") not in _MODEL_TYPES:
        raise ValueError(
            f"{path}: config.json has model_type {config.get('model_type')!r}; a decision "
            f"checkpoint on MLX is a Gemma 4 base's multimodal one ({', '.join(_MODEL_TYPES)})"
        )
    text = config["text_config"]
    if per_layer := text.pop("per_layer_config", None):
        # transformers writes the same values for every full-attention layer.
        full = per_layer[str(text["layer_types"].index("full_attention"))]
        text["global_head_dim"] = full["head_dim"]
        text["num_global_key_value_heads"] = full.get(
            "num_key_value_heads", text["num_key_value_heads"]
        )
    return config


def _check_mlx_vlm() -> None:
    try:
        found = version("mlx-vlm")
    except PackageNotFoundError:
        raise ImportError(MLX_EXTRA) from None
    if found != _MLX_VLM:
        raise RuntimeError(
            f"the MLX backend is tested on mlx-vlm {_MLX_VLM}, and mlx-vlm {found} is installed; "
            "its internals may differ and change answers without an error. Install the tested "
            "release: pip install 'strands-decider[mlx]'"
        )


def load_mlx_model(path: str | os.PathLike[str]) -> Any:
    """mlx-vlm's model for the checkpoint or MLX artifact in directory `path`, loaded strictly and
    lazily. mlx-vlm reads config.json from the directory and, with no index, every *.safetensors
    in it, the readout too; so it reads a directory of links to the model's shards and the other
    files, with `model_config`'s config.json. An error names `path`, not that directory."""
    _check_mlx_vlm()
    from mlx_vlm.utils import load_model

    path = Path(path).resolve()  # absolute: a link's target is read from the link's directory
    config = model_config(path)
    try:
        # The shards stay open once loaded, so the links may go before the tensors are read.
        with tempfile.TemporaryDirectory(prefix="strands-decider-mlx-") as view:
            for entry in os.listdir(path):
                if entry != "config.json" and (
                    _SHARD.fullmatch(entry) or not entry.endswith(".safetensors")
                ):
                    os.symlink(path / entry, Path(view) / entry)
            (Path(view) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            return load_model(Path(view), lazy=True, strict=True)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise ValueError(f"{path}: mlx-vlm cannot load this checkpoint: {e}") from e


def convert(
    source: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    bits: int | None = None,
    group_size: int = 64,
) -> Path:
    """Writes the MLX artifact of the model checkpoint `source` (a directory or a Hub id) to
    the new or empty directory `dest`, and returns its path: bf16, or the decoder quantized to
    `bits` in groups of `group_size`. It is written beside `dest`, reloaded, and only then renamed
    to `dest`, so a failed conversion leaves nothing behind."""
    _check_mlx_vlm()
    import mlx.core as mx
    from mlx_vlm.quant_utils import quantize_model
    from mlx_vlm.utils import save_config, save_weights

    if bits is not None and bits not in _BITS:
        raise ValueError(f"bits must be one of {_BITS} (or none, for bf16), not {bits!r}")
    dst = Path(dest).resolve()
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"{dst} exists and is not an empty directory")
    src = Path(checkpoint_dir(os.fspath(source))).resolve()
    DecisionConfig.load(src)  # a model checkpoint, checked before the long part
    if is_mlx_artifact(src):
        raise ValueError(f"{src} is already an MLX artifact; convert the checkpoint it came from")
    config = model_config(src)
    config[_MARKER] = {"format": "mlx", "written_by": "strands-decider convert-mlx"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = dst.with_name(f"{dst.name}.partial-{secrets.token_hex(4)}")
    work.mkdir()  # the permissions any new directory gets, so the serving account can read it
    try:
        model = load_mlx_model(src)
        if bits is not None:
            # mlx-vlm's rule, for the decoder alone: the encoders and their projections stay bf16.
            rule = model.quant_predicate
            model, config = quantize_model(
                model,
                config,
                group_size,
                bits,
                quant_predicate=lambda p, m: p.startswith("language_model.") and rule(p, m),
            )
        save_weights(work, model, donate_weights=True)
        del model
        mx.clear_cache()
        save_config(config, config_path=work / "config.json")
        for name in _COPIED:
            if (src / name).is_file():
                shutil.copyfile(src / name, work / name)
        load_mlx_model(work)  # strictly: every tensor the model has, of its shape
        os.rename(work, dst)  # replaces an empty `dst`, in one step
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return dst
