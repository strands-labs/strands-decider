"""What the MLX backend and `convert-mlx` know about checkpoints and artifacts, without MLX, so
the router can tell an MLX artifact apart on any platform."""

from __future__ import annotations

import copy
from typing import Any

# The mlx-vlm release the backend is tested on. It relies on behaviour no release promises (a
# passed mask reaching every layer as it is, `logits_to_keep` through the KV-shared layers, its
# caches and image processors), so another release is refused at load.
TESTED_MLX_VLM = "0.7.12"

# Written by convert-mlx into the artifact's config.json, beside mlx-vlm's own fields.
ARTIFACT_KEY = "strands_decider_artifact"
ARTIFACT = {"format": "mlx", "written_by": "strands-decider convert-mlx"}

# Full-attention layers' head settings: transformers 5.2x writes them per layer, the bases (and
# mlx-vlm 0.7.12) as one field each.
_HEAD_FIELDS = {"head_dim": "global_head_dim", "num_key_value_heads": "num_global_key_value_heads"}


def check_mlx_vlm(found: str) -> None:
    """RuntimeError unless `found` is the mlx-vlm release the backend is tested on."""
    if found != TESTED_MLX_VLM:
        raise RuntimeError(
            f"the MLX backend is tested on mlx-vlm {TESTED_MLX_VLM}, and mlx-vlm {found} is "
            "installed; its internals may differ and change answers without an error. Install "
            "the tested release: pip install 'strands-decider[mlx]'"
        )


def is_mlx_artifact(model_config: dict[str, Any]) -> bool:
    """Whether a config.json is an MLX artifact's: convert-mlx's marker, or the `quantization`
    field mlx's quantizers write (transformers' quantized checkpoints use `quantization_config`)."""
    marker = model_config.get(ARTIFACT_KEY)
    if isinstance(marker, dict) and marker.get("format") == "mlx":
        return True
    q = model_config.get("quantization")
    return isinstance(q, dict) and "bits" in q and "quant_method" not in q


def bases_form(model_config: dict[str, Any]) -> dict[str, Any] | None:
    """config.json in the Gemma 4 bases' own form, which mlx-vlm 0.7.12 reads, or None if it's
    already in it. transformers 5.2x writes each full-attention layer's head size and key/value
    heads under `text_config.per_layer_config`; the bases give one `global_head_dim` and one
    `num_global_key_value_heads` for every full-attention layer, and without them mlx-vlm builds
    those layers with its own defaults. ValueError if the full-attention layers disagree."""
    text = model_config.get("text_config")
    if not isinstance(text, dict) or "per_layer_config" not in text:
        return None
    out = copy.deepcopy(model_config)
    per_layer = out["text_config"].pop("per_layer_config")
    if not per_layer:
        return out
    full = [i for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"]
    for field, base_field in _HEAD_FIELDS.items():
        values = {per_layer.get(str(i), {}).get(field, text.get(field)) for i in full}
        if len(values) != 1 or text.get(base_field, next(iter(values))) not in values:
            raise ValueError(f"config.json gives the full-attention layers different {field}s")
        out["text_config"][base_field] = values.pop()
    return out
