"""Quantization safety and end-to-end tests using a tiny locally generated checkpoint."""
import pytest
import torch

from strands_decider.infer import EngineConfig, load_engine


def test_reject_invalid_config():
    for kwargs in [{"device": "cpu", "quant_bits": 4}, {"device": "mlx", "quant_bits": 3},
                   {"device": "mlx", "quant_group_size": 17}]:
        with pytest.raises(ValueError):
            EngineConfig(**kwargs)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_engine(tmp_path, bits):
    pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from test_mlx_engine import REQUESTS, _checkpoint

    from strands_decider.mlx_engine import QUANT_PROJECTIONS
    path = _checkpoint(tmp_path, "pointer")
    engine = load_engine(str(path), device="mlx", quant_bits=bits)
    assert engine.quantized_paths
    assert all(p.rsplit(".", 1)[-1] in QUANT_PROJECTIONS for p in engine.quantized_paths)
    modules = dict(engine._cache_owner.named_modules())
    assert isinstance(modules["model.layers.0.linear_attn.in_proj_qkv"], nn.QuantizedLinear)
    assert isinstance(modules["model.layers.0.linear_attn.in_proj_a"], nn.Linear)
    assert isinstance(modules["model.embed_tokens"], nn.Embedding)
    assert all(p.dtype == torch.float32 for p in engine.model.head.parameters())
    for request in REQUESTS:
        response = engine.evaluate(request)
        assert set(response.answers) == set(request.questions)
        for answer in response.answers.values():
            if answer.type == "noul":
                assert 0 <= answer.noul <= 1
            else:
                assert sum(answer.probabilities.values()) == pytest.approx(1, abs=.001)
