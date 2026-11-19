"""The MLX backend's checkpoint rules that need no MLX, so that any platform can tell an MLX
artifact from a checkpoint."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

# Written by convert-mlx into the artifact's config.json, beside mlx-vlm's own fields.
ARTIFACT_KEY = "strands_decider_artifact"
ARTIFACT = {"format": "mlx", "written_by": "strands-decider convert-mlx"}

# Full-attention layers' head settings: transformers 5.2x writes them per layer, the bases (and
# mlx-vlm 0.7.12) as one field each.
_HEAD_FIELDS = {"head_dim": "global_head_dim", "num_key_value_heads": "num_global_key_value_heads"}


def is_mlx_artifact(path: str | os.PathLike[str]) -> bool:
    """Whether the checkpoint directory `path` is an MLX artifact, in mlx-vlm's layout."""
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        return ARTIFACT_KEY in json.load(f)


def bases_form(config: dict[str, Any]) -> dict[str, Any]:
    """config.json in the Gemma 4 bases' own form, which mlx-vlm 0.7.12 reads. transformers 5.2x
    writes each full-attention layer's head size and key/value heads under
    `text_config.per_layer_config`; the bases give one `global_head_dim` and one
    `num_global_key_value_heads` for them all, and without those mlx-vlm builds the layers with
    its own defaults. ValueError if the full-attention layers disagree."""
    text = config.get("text_config")
    if not isinstance(text, dict) or "per_layer_config" not in text:
        return config
    out = copy.deepcopy(config)
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
