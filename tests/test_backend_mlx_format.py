"""The MLX backend's checkpoint rules that need no MLX (strands_decider.backends.mlx_format), on
any platform: config.json read in the bases' form, and an MLX artifact refused off `--device mlx`.
tests/test_backend_mlx.py runs the backend itself, on Apple silicon."""

from __future__ import annotations

import json

import pytest
from core_support import decision_config_fixture

from strands_decider import decision
from strands_decider.backends.mlx_format import ARTIFACT, ARTIFACT_KEY, bases_form

FULL, SLIDING = "full_attention", "sliding_attention"


def _config(per_layer, **text):
    return {
        "model_type": "gemma4",
        "text_config": {
            "layer_types": [SLIDING, FULL, SLIDING, FULL],
            "head_dim": 16,
            "num_key_value_heads": 2,
            **text,
            "per_layer_config": per_layer,
        },
    }


@pytest.mark.parametrize(
    ("per_layer", "head_dim", "kv_heads"),
    [
        ({"1": {"head_dim": 64}, "3": {"head_dim": 64}}, 64, 2),  # the KV heads: the decoder's
        ({str(i): {"head_dim": 64, "num_key_value_heads": 1} for i in (1, 3)}, 64, 1),
        ({}, None, None),  # nothing per layer: nothing to set
    ],
)
def test_bases_form_reads_transformers_5_2x_configs(per_layer, head_dim, kv_heads):
    text = bases_form(_config(per_layer))["text_config"]
    assert "per_layer_config" not in text
    assert (text.get("global_head_dim"), text.get("num_global_key_value_heads")) == (
        head_dim,
        kv_heads,
    )
    base = {"text_config": {"head_dim": 16, "global_head_dim": 64}}
    assert bases_form(base) is base


@pytest.mark.parametrize(
    ("per_layer", "text"),
    [
        ({"1": {"head_dim": 512}}, {}),  # layer 3 keeps 16
        ({"1": {"head_dim": 64}, "3": {"head_dim": 64}}, {"global_head_dim": 512}),
    ],
)
def test_bases_form_refuses_full_attention_layers_that_disagree(per_layer, text):
    with pytest.raises(ValueError, match="different head_dims"):
        bases_form(_config(per_layer, **text))


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_an_mlx_artifact_is_refused_off_mlx_before_anything_loads(tmp_path, device):
    """Only decision_config.json and the marked config.json are here: the refusal comes before
    the readout, the tokenizer or a backend is read."""
    (tmp_path / "decision_config.json").write_text(json.dumps(decision_config_fixture()))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "gemma4", ARTIFACT_KEY: ARTIFACT})
    )
    with pytest.raises(ValueError, match="is an MLX artifact: serve it with --device mlx"):
        decision.load_decision_engine(str(tmp_path), device=device)
