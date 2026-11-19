"""The readout's arithmetic and the checkpoint's weights, on the model classes without their vLLM
layers. Needs vLLM, no GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
model = pytest.importorskip("strands_decider_vllm.model")
from torch import nn  # noqa: E402

CLASSES = [
    model.StrandsDeciderForCausalLM,
    model.StrandsDeciderForConditionalGeneration,
    model.StrandsDeciderUnifiedForConditionalGeneration,
]
IDS = list(range(1000, 1255))


def bare(cls, weight=None, softcap=30.0, temperature=1.0):
    """An instance with only the readout (no vLLM layers)."""
    m = object.__new__(cls)
    nn.Module.__init__(m)
    m.softcap, m.temperature, m.vocab_size = softcap, temperature, 1300
    m.register_buffer("code_ids", torch.tensor(IDS), persistent=False)
    m.readout = nn.Linear(
        4 if weight is None else weight.shape[1], 255, bias=False, dtype=torch.bfloat16
    )
    if weight is not None:
        with torch.no_grad():
            m.readout.weight.copy_(weight)
    return m


def test_the_logits_are_the_cores_readout_at_the_code_ids():
    torch.manual_seed(0)
    w = (torch.randn(255, 32) * 0.5).to(torch.bfloat16)
    h = torch.randn(3, 32) * 4  # float32: the readout takes it in bf16
    out = bare(CLASSES[0], w, temperature=2.0).compute_logits(h)
    z = nn.functional.linear(h.to(torch.bfloat16), w).float()  # the bf16 GEMM, rounded to bf16
    assert out.dtype == torch.float32 and torch.equal(
        out[:, IDS], torch.tanh(z / 30.0) * 30.0 / 2.0
    )
    others = torch.ones(1300, dtype=torch.bool)
    others[IDS] = False
    assert torch.isneginf(out[:, others]).all()


def test_the_gemm_is_rounded_to_bf16_before_the_cap():
    # h . w = 16 + 1/16, halfway between the bf16 values 16 and 16.125: rounding to even gives 16.
    w = torch.zeros(255, 2, dtype=torch.bfloat16)
    w[0] = 1.0
    out = bare(CLASSES[0], w).compute_logits(torch.tensor([[16.0, 0.0625]]))
    capped = torch.tanh(torch.tensor([16.0]) / 30.0) * 30.0
    assert out[0, IDS[0]] == capped[0] != torch.tanh(torch.tensor([16.0625]) / 30.0) * 30.0


@pytest.fixture
def parent_loads(monkeypatch):
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
    names = ["embed_tokens.weight", "model.language_model.layers.0.mlp.fc1.weight"]
    # "weight" is readout.safetensors read again as a model file, without an index file.
    m = bare(cls)
    loaded = m.load_weights([(n, t) for n in names] + [("weight", t), ("readout.weight", readout)])
    # A bare decoder's keys go under vLLM's model.*; the multimodal ones are vLLM's already.
    renamed = "model.embed_tokens.weight" if cls is CLASSES[0] else names[0]
    assert parent_loads == [renamed, names[1]] and "readout.weight" in loaded
    assert torch.equal(m.readout.weight, readout.to(torch.bfloat16))


def test_a_loader_without_the_readouts_source_is_refused(parent_loads):
    with pytest.raises(ValueError, match=r"readout\.safetensors was not loaded"):
        bare(CLASSES[1]).load_weights([("model.norm.weight", torch.zeros(1))])
