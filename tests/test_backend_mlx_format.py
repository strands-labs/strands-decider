"""What strands_decider.backends.mlx_format decides without MLX, on any platform: config.json as
mlx-vlm reads it, and an MLX artifact refused off `--device mlx`. tests/test_backend_mlx.py runs
the rest on Apple silicon."""

from __future__ import annotations

import json

import pytest
from core_support import decision_config_fixture

from strands_decider.backends.mlx_format import model_config
from strands_decider.decision import load_decision_engine

FULL, SLIDING = "full_attention", "sliding_attention"


@pytest.mark.parametrize(
    ("per_layer", "head_dim", "kv_heads"),
    [
        ({"1": {"head_dim": 64}, "3": {"head_dim": 64}}, 64, 2),  # the KV heads: the decoder's
        ({str(i): {"head_dim": 64, "num_key_value_heads": 1} for i in (1, 3)}, 64, 1),  # 26B, 31B
        (None, None, None),  # the bases' form: nothing to set
    ],
)
def test_model_config_reads_transformers_5_2x_configs(tmp_path, per_layer, head_dim, kv_heads):
    text = {"layer_types": [SLIDING, FULL, SLIDING, FULL], "head_dim": 16, "num_key_value_heads": 2}
    if per_layer is not None:
        text["per_layer_config"] = per_layer
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": text}))
    got = model_config(tmp_path)["text_config"]
    assert "per_layer_config" not in got
    assert (got.get("global_head_dim"), got.get("num_global_key_value_heads")) == (
        head_dim,
        kv_heads,
    )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_an_mlx_artifact_is_refused_off_mlx_before_anything_loads(tmp_path, device):
    """Only decision_config.json and the config.json convert-mlx marks are here: the refusal
    comes before the readout, the tokenizer or a backend is read."""
    (tmp_path / "decision_config.json").write_text(json.dumps(decision_config_fixture()))
    marked = {"model_type": "gemma4", "strands_decider_artifact": {"format": "mlx"}}
    (tmp_path / "config.json").write_text(json.dumps(marked))
    with pytest.raises(ValueError, match="is an MLX artifact: serve it with --device mlx"):
        load_decision_engine(str(tmp_path), device=device)
