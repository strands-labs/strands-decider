"""What the MLX backend and `convert-mlx` know about checkpoints and artifacts, without MLX.

None of this imports mlx, so the router can tell an MLX artifact apart on any platform, and
the tests of these rules run everywhere:

- `ARTIFACT`: the marker `convert-mlx` writes into an artifact's config.json, and
  `is_mlx_artifact`, which the router checks before loading a checkpoint on a device other
  than mlx: an MLX artifact holds mlx-vlm's tensor layout, which no other backend reads.
- `bases_form`: config.json in the Gemma 4 bases' own form, which mlx-vlm 0.7.12 reads, from
  the form transformers 5.2x writes.
- `RECORDED` and `conformance`: the quantized formats docs/spec.md section 10 records bounds
  for, and the claim an artifact may make.
- `TESTED_MLX_VLM` and `check_mlx_vlm`: the mlx-vlm release the backend is tested on.
"""

from __future__ import annotations

import json
from typing import Any

# The mlx-vlm release the backend is tested on. It relies on behaviour no release promises (a
# passed mask reaching every layer as it is, `logits_to_keep` through the KV-shared layers,
# its caches and image processors), so another release is refused at load.
TESTED_MLX_VLM = "0.7.12"

# Written by convert-mlx into the artifact's config.json, beside mlx-vlm's own fields.
ARTIFACT_KEY = "strands_decider_artifact"
ARTIFACT = {"format": "mlx", "written_by": "strands-decider convert-mlx"}

# The conformance bounds docs/spec.md section 10 records (top-1 agreement at least, mean |dp| at
# most), for bf16 and by quantized format: (base model, bits, group size, mode). MLX 8-bit in
# groups of 64 has bf16's bounds at E2B, where it was measured (98.83% top-1, mean |dp| 0.0099,
# over 1,200 prompts); no other size and no other quantized format has bounds recorded, and an
# artifact in one makes no conformance claim.
BF16_BOUNDS = {"top1": 0.98, "mean_dp": 0.02}
RECORDED: dict[tuple[str, int, int, str], dict[str, float]] = {
    ("google/gemma-4-E2B-it", 8, 64, "affine"): dict(BF16_BOUNDS),
}


def check_mlx_vlm(found: str) -> None:
    """RuntimeError unless `found` is the mlx-vlm release the backend is tested on."""
    if found != TESTED_MLX_VLM:
        raise RuntimeError(
            f"the MLX backend is tested on mlx-vlm {TESTED_MLX_VLM}, and mlx-vlm {found} is "
            f"installed; its internals may differ and change answers without an error. "
            f"Install the tested release: pip install 'strands-decider[mlx]'"
        )


def is_mlx_artifact(model_config: dict[str, Any]) -> bool:
    """Whether a config.json is an MLX artifact's: the marker convert-mlx writes, or the
    `quantization` field mlx's quantizers write (transformers' own quantized checkpoints use
    `quantization_config` alone, with a `quant_method`)."""
    marker = model_config.get(ARTIFACT_KEY)
    if isinstance(marker, dict) and marker.get("format") == "mlx":
        return True
    q = model_config.get("quantization")
    return isinstance(q, dict) and "bits" in q and "quant_method" not in q


def refuse_off_mlx(checkpoint: str, device: str) -> ValueError:
    return ValueError(
        f"{checkpoint} is an MLX artifact (written by strands-decider convert-mlx, in "
        f"mlx-vlm's layout), which only the MLX backend reads: use --device mlx, not "
        f"{device!r}, or the bf16 checkpoint it was converted from"
    )


def quantization_format(quantization: dict[str, Any] | None) -> tuple[int, int, str] | None:
    """(bits, group size, mode) of config.json's `quantization`, or None for bf16."""
    if not quantization:
        return None
    return (
        int(quantization["bits"]),
        int(quantization["group_size"]),
        str(quantization.get("mode", "affine")),
    )


def conformance(base_model: str, quantization: dict[str, Any] | None) -> dict[str, Any]:
    """The conformance claim an artifact of `base_model` in this quantization may make
    (docs/spec.md section 10), for `/health` and for the notice at conversion and load:
    `claim` is "bf16" or "recorded" with the `bounds` it must meet, or "none" with the
    `notice` to give."""
    fmt = quantization_format(quantization)
    if fmt is None:
        return {"format": "bf16", "claim": "bf16", "bounds": dict(BF16_BOUNDS), "notice": None}
    bits, group, mode = fmt
    name = f"{bits}-bit, group {group}, {mode}"
    bounds = RECORDED.get((base_model, bits, group, mode))
    if bounds is not None:
        return {"format": name, "claim": "recorded", "bounds": dict(bounds), "notice": None}
    return {
        "format": name,
        "claim": "none",
        "bounds": None,
        "notice": (
            f"this MLX artifact ({name}, of {base_model}) makes no conformance claim: "
            f"docs/spec.md section 10 records no bounds for its format at this size"
        ),
    }


_HEAD_FIELDS = {"head_dim": "global_head_dim", "num_key_value_heads": "num_global_key_value_heads"}


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def bases_form(model_config: dict[str, Any]) -> dict[str, Any] | None:
    """config.json in the Gemma 4 bases' own form, or None if it is already in it.

    transformers 5.2x writes each full-attention layer's head size and KV heads under
    `text_config.per_layer_config`, a mapping from layer index to the fields that differ from
    `head_dim` and `num_key_value_heads`. The bases give one `global_head_dim` and one
    `num_global_key_value_heads` for every full-attention layer, which is all mlx-vlm reads:
    without them it builds those layers with its own defaults. Every full-attention layer's
    values, its entry's or else the text config's, must agree, and are written as the bases'
    fields; a field given both ways must agree too. Anything else is refused with ValueError:
    an entry for a sliding layer or for no layer, an unknown field, a value that is not a
    positive integer, a mapping that is not one."""
    text = model_config.get("text_config")
    if not isinstance(text, dict) or "per_layer_config" not in text:
        return None
    per_layer = text["per_layer_config"]
    if per_layer is None or per_layer == {}:
        out: dict[str, Any] = json.loads(json.dumps(model_config))
        out["text_config"].pop("per_layer_config")
        return out
    if not isinstance(per_layer, dict):
        raise ValueError(f"config.json's per_layer_config must be a mapping, not {per_layer!r}")
    types = text.get("layer_types")
    if not isinstance(types, list) or not types:
        raise ValueError("config.json's text_config has per_layer_config but no layer_types")
    entries: dict[int, dict[str, Any]] = {}
    for layer, settings in per_layer.items():
        if not (isinstance(layer, str) and layer.isdigit() and int(layer) < len(types)):
            raise ValueError(f"config.json's per_layer_config names no layer: {layer!r}")
        i = int(layer)
        if types[i] != "full_attention":
            raise ValueError(
                f"config.json's per_layer_config overrides layer {i}, a {types[i]} layer; "
                "mlx-vlm sets only the full-attention layers apart"
            )
        if not isinstance(settings, dict) or set(settings) - set(_HEAD_FIELDS):
            raise ValueError(
                f"config.json's per_layer_config[{layer!r}] is {settings!r}; mlx-vlm reads only "
                f"{sorted(_HEAD_FIELDS)} for a full-attention layer"
            )
        for k, v in settings.items():
            if not _int(v):
                raise ValueError(f"config.json's per_layer_config[{layer!r}][{k!r}] is {v!r}")
        entries[i] = settings
    out = json.loads(json.dumps(model_config))
    t = out["text_config"]
    t.pop("per_layer_config")
    for field, base_field in _HEAD_FIELDS.items():
        default = text.get(field)
        values = {
            i: entries.get(i, {}).get(field, default)
            for i, kind in enumerate(types)
            if kind == "full_attention"
        }
        distinct = {repr(v) for v in values.values()}
        if len(distinct) != 1:
            raise ValueError(
                f"config.json gives the full-attention layers different {field}s: "
                + ", ".join(f"layer {i}: {v!r}" for i, v in values.items())
            )
        value = next(iter(values.values()))
        if not _int(value):
            raise ValueError(f"config.json gives the full-attention layers {field} {value!r}")
        given = text.get(base_field)
        if given is not None and given != value:
            raise ValueError(
                f"config.json's {base_field} is {given!r}, and its per_layer_config gives the "
                f"full-attention layers {field} {value!r}"
            )
        t[base_field] = value
    return out
