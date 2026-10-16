"""A Gated DeltaNet torso served on CUDA without flash-linear-attention says so.

transformers runs its reference PyTorch code for the DeltaNet ops when `fla` is missing:
answers equal to float rounding, 1.5-3.6x slower on an L40S. The engine warns at load and
`/health` reports which kernels are installed. CPU only: package lookups are faked, and the
"cuda" engine never moves its tiny torso off the CPU.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch
from fastapi.testclient import TestClient

transformers = pytest.importorskip("transformers")

from strands_decider import server  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine, linear_attention_kernels  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["<state> <question> <options> <answer> choice"],
                            trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<eos>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


def _model(hybrid):
    torch.manual_seed(0)
    tok = _tokenizer()
    if hybrid:
        if not hasattr(transformers, "Qwen3_5TextConfig"):
            pytest.skip("this transformers has no Qwen3.5")
        torso = transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(
            vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=["linear_attention"] * 3 + ["full_attention"])).model
    else:
        torso = transformers.Qwen3Model(transformers.Qwen3Config(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8))
    cfg = StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16,
                               use_lora=False, torch_dtype="float32", max_length=512)
    return StrandsDeciderModel(cfg, torso, tok)


def _packages(monkeypatch, installed):
    """Make importlib.util.find_spec see exactly `installed` among fla, triton and causal_conv1d."""
    real = importlib.util.find_spec

    def find(name, *args, **kwargs):
        if name in ("fla", "triton", "causal_conv1d"):
            return real("torch") if name in installed else None  # any spec means installed
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find)


def _cuda_engine(monkeypatch, model):
    """An engine configured for "cuda" whose model stays on the CPU."""
    monkeypatch.setattr(StrandsDeciderModel, "to", lambda self, *a, **k: self)
    return SystemOneEngine(model, EngineConfig(device="cuda"))


def test_plain_torso_has_no_linear_attention_kernels():
    assert linear_attention_kernels(_model(hybrid=False).torso) is None


@pytest.mark.parametrize("installed, want", [
    ((), {"flash_linear_attention": False, "causal_conv1d": False}),
    # fla without Triton (macOS) imports but cannot run
    (("fla",), {"flash_linear_attention": False, "causal_conv1d": False}),
    (("fla", "triton"), {"flash_linear_attention": True, "causal_conv1d": False}),
    (("fla", "triton", "causal_conv1d"), {"flash_linear_attention": True, "causal_conv1d": True}),
])
def test_hybrid_torso_reports_installed_kernels(monkeypatch, installed, want):
    _packages(monkeypatch, installed)
    assert linear_attention_kernels(_model(hybrid=True).torso) == want


def test_cuda_without_fla_warns_at_load(monkeypatch, capsys):
    _packages(monkeypatch, ())
    eng = _cuda_engine(monkeypatch, _model(hybrid=True))
    err = capsys.readouterr().err
    assert "flash-linear-attention is not installed" in err
    assert 'pip install "strands-decider[cuda]"' in err
    assert eng.kernels == {"flash_linear_attention": False, "causal_conv1d": False}


@pytest.mark.parametrize("hybrid, installed, device", [
    (True, ("fla", "triton"), "cuda"),  # the fast path is installed
    (True, (), "cpu"),                  # no fla kernels on CPU in any case
    (False, (), "cuda"),                # no Gated DeltaNet layers
])
def test_no_warning_otherwise(monkeypatch, capsys, hybrid, installed, device):
    _packages(monkeypatch, installed)
    model = _model(hybrid)
    if device == "cuda":
        _cuda_engine(monkeypatch, model)
    else:
        SystemOneEngine(model, EngineConfig(device=device))
    assert "flash-linear-attention" not in capsys.readouterr().err


@pytest.mark.parametrize("hybrid", [True, False])
def test_health_reports_kernels(monkeypatch, hybrid):
    _packages(monkeypatch, ("fla", "triton"))
    model = _model(hybrid)
    monkeypatch.setattr(server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: model))
    client = TestClient(server.create_app("checkpoints/does-not-exist", device="cpu"))
    kernels = client.get("/health").json()["kernels"]
    assert kernels == ({"flash_linear_attention": True, "causal_conv1d": False} if hybrid else None)
