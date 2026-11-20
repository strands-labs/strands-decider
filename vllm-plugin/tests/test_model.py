"""The readout's arithmetic and the checkpoint's weights, on the model classes without their vLLM
layers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from strands_decider_vllm.model import (
    StrandsDeciderForCausalLM,
    StrandsDeciderForConditionalGeneration,
    StrandsDeciderUnifiedForConditionalGeneration,
)

CLASSES = [
    StrandsDeciderForCausalLM,
    StrandsDeciderForConditionalGeneration,
    StrandsDeciderUnifiedForConditionalGeneration,
]
IDS = list(range(1000, 1255))


def readout_only(cls, weight, temperature=1.0):
    """An instance with the attributes __init__ gives the readout, and no vLLM layers."""
    m = object.__new__(cls)
    nn.Module.__init__(m)
    m.softcap, m.temperature, m.vocab_size = 30.0, temperature, 1300
    m.register_buffer("code_ids", torch.tensor(IDS), persistent=False)
    m.readout = nn.Linear(weight.shape[1], 255, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        m.readout.weight.copy_(weight)
    return m


def test_the_logits_are_the_cores_readout_at_the_code_ids():
    torch.manual_seed(0)
    w = (torch.randn(255, 32) * 0.5).to(torch.bfloat16)
    h = (torch.randn(3, 32) * 4).to(torch.bfloat16)
    out = readout_only(CLASSES[0], w, temperature=2.0).compute_logits(h)
    z = nn.functional.linear(h, w).float()  # the bf16 GEMM, rounded to bf16
    assert out.dtype == torch.float32
    assert torch.equal(out[:, IDS], torch.tanh(z / 30.0) * 30.0 / 2.0)
    others = torch.ones(1300, dtype=torch.bool)
    others[IDS] = False
    assert torch.isneginf(out[:, others]).all()


def test_the_gemm_is_rounded_to_bf16_before_the_cap():
    # h . w = 16 + 1/16, halfway between the bf16 values 16 and 16.125: rounding to even gives 16.
    w = torch.zeros(255, 2, dtype=torch.bfloat16)
    w[0] = 1.0
    h = torch.tensor([[16.0, 0.0625]], dtype=torch.bfloat16)
    out = readout_only(CLASSES[0], w).compute_logits(h)
    assert out[0, IDS[0]] == torch.tanh(torch.tensor(16.0) / 30.0) * 30.0


@pytest.fixture
def parent_loads(monkeypatch):
    """The names each class's vLLM parent receives in load_weights."""
    received = []

    def load_weights(self, weights):
        names = [name for name, _ in weights]
        received.extend(names)
        return set(names)

    for cls in CLASSES:
        monkeypatch.setattr(cls.__mro__[2], "load_weights", load_weights)
    return received


@pytest.mark.parametrize("cls", CLASSES)
def test_the_readout_comes_from_its_own_source_and_the_rest_from_the_checkpoint(parent_loads, cls):
    t, readout = torch.zeros(1), torch.randn(255, 4)
    name = "embed_tokens.weight" if cls is StrandsDeciderForCausalLM else "model.language_model.x"
    m = readout_only(cls, torch.zeros(255, 4, dtype=torch.bfloat16))
    # "weight" is readout.safetensors read again as a model file, without an index file.
    loaded = m.load_weights([(name, t), ("weight", t), ("readout.weight", readout)])
    # A bare decoder's keys go under vLLM's model.*; the multimodal ones are vLLM's already.
    assert parent_loads == ["model." + name if cls is StrandsDeciderForCausalLM else name]
    assert "readout.weight" in loaded
    assert torch.equal(m.readout.weight, readout.to(torch.bfloat16))


def test_a_loader_without_the_readouts_source_is_refused(parent_loads):
    m = readout_only(CLASSES[1], torch.zeros(255, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match=r"readout\.safetensors was not loaded"):
        m.load_weights([("model.norm.weight", torch.zeros(1))])
