"""The model class without a GPU: the readout's arithmetic (compute_logits), the checkpoint's key renaming and readout
loading (load_weights), and decision_config.json at startup (_init_readout), from a directory or a Hub repo id."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
model = pytest.importorskip("strands_decider_v2_vllm.model")
from torch import nn  # noqa: E402

from strands_decider_v2_vllm.decision_config import DecisionConfigError  # noqa: E402

Cls = model.StrandsDeciderV2ForCausalLM
IDS = list(range(1000, 1255))


def bare(**attrs):
    """An instance of the model class with only the attributes a method under test reads (no vLLM layers)."""
    m = object.__new__(Cls)
    nn.Module.__init__(m)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# --- compute_logits -------------------------------------------------------------------------------------------------


def readout_model(weight, softcap=30.0, temperature=1.0, vocab=1300, ids=IDS):
    m = bare(config=SimpleNamespace(vocab_size=vocab), softcap=softcap, temperature=temperature)
    m.register_buffer("code_ids", torch.tensor(ids), persistent=False)
    m.readout = nn.Linear(weight.shape[1], 255, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        m.readout.weight.copy_(weight)
    return m


def test_logits_are_the_capped_readout_at_the_code_ids_and_minus_inf_elsewhere():
    torch.manual_seed(0)
    w = (torch.randn(255, 32) * 0.5).to(torch.bfloat16)
    h = (torch.randn(3, 32) * 4).to(torch.bfloat16)
    out = readout_model(w, temperature=2.0).compute_logits(h)
    assert out.dtype == torch.float32 and out.shape == (3, 1300)
    z = torch.nn.functional.linear(h, w).float()  # the bf16 GEMM, rounded to bf16
    expected = torch.tanh(z / 30.0) * 30.0 / 2.0
    assert torch.equal(out[:, IDS], expected), "code i at token id IDS[i], in float32"
    others = torch.ones(1300, dtype=torch.bool)
    others[IDS] = False
    assert torch.isneginf(out[:, others]).all()


def test_the_gemm_is_rounded_to_bf16_before_the_cap():
    # h . w = 16 + 1/16, halfway between the bf16 values 16 and 16.125: rounding to even gives 16. A float32 GEMM
    # without the rounding would cap 16.0625 instead.
    w = torch.zeros(255, 2, dtype=torch.bfloat16)
    w[0] = torch.tensor([1.0, 1.0])
    h = torch.tensor([[16.0, 0.0625]], dtype=torch.bfloat16)
    out = readout_model(w).compute_logits(h)

    def capped(z):  # in float32, as the specification computes it
        z = torch.tensor([z], dtype=torch.float32)
        return (torch.tanh(z / 30.0) * 30.0).item()

    assert out[0, IDS[0]].item() == capped(16.0)
    assert capped(16.0) != capped(16.0625)


def test_the_cap_bounds_the_logits_in_float32():
    w = torch.zeros(255, 1, dtype=torch.bfloat16)
    w[:, 0] = 1.0
    out = readout_model(w, softcap=30.0, temperature=0.5).compute_logits(
        torch.tensor([[1e4]], dtype=torch.bfloat16)
    )
    assert (out[0, IDS] <= 30.0 / 0.5).all() and (out[0, IDS] > 59.0).all()


# --- load_weights ---------------------------------------------------------------------------------------------------


@pytest.fixture
def decoder_loads(monkeypatch):
    received = []

    def load_weights(self, weights):
        names = set()
        for name, _ in weights:
            received.append(name)
            names.add(name)
        return names

    monkeypatch.setattr(model.Gemma4ForCausalLM, "load_weights", load_weights)
    return received


def loadable():
    m = bare()
    m.readout = nn.Linear(4, 255, bias=False, dtype=torch.bfloat16)
    return m


def test_load_weights_renames_the_bare_decoder_and_loads_the_readout(decoder_loads):
    m = loadable()
    t = torch.zeros(1)
    readout = torch.randn(255, 4, dtype=torch.float32)
    loaded = m.load_weights(
        [
            ("embed_tokens.weight", t),
            ("layers.0.self_attn.q_proj.weight", t),
            ("model.norm.weight", t),
            (
                "weight",
                t,
            ),  # readout.safetensors also read as a decoder file (no index file): skipped
            ("readout.weight", readout),  # the readout's own weight source
        ]
    )
    assert decoder_loads == [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
    ]
    assert "readout.weight" in loaded
    assert torch.equal(m.readout.weight, readout.to(torch.bfloat16)), (
        "a float32 readout is rounded to bf16"
    )


@pytest.mark.parametrize("count", [0, 2])
def test_load_weights_refuses_a_missing_or_doubled_readout(decoder_loads, count):
    with pytest.raises(ValueError, match=r"readout\.safetensors was not loaded"):
        loadable().load_weights([("readout.weight", torch.zeros(255, 4))] * count)


def test_load_weights_refuses_a_readout_of_the_wrong_shape(decoder_loads):
    with pytest.raises(ValueError, match=r"readout \(255, 5\)"):
        loadable().load_weights([("readout.weight", torch.zeros(255, 5))])


# --- _init_readout --------------------------------------------------------------------------------------------------


def decision_config(**overrides):
    values = dict(
        format_version=1,
        token_ids=list(IDS),
        temperature=1.0,
        attention_mode="prefix_bidir",
        pooling="last",
    )
    values.update(overrides)
    return values


def init_readout(model_dir, softcap=30.0, revision=None):
    m = bare(
        config=SimpleNamespace(final_logit_softcapping=softcap, vocab_size=262144, hidden_size=8)
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(model=str(model_dir), revision=revision, dtype=torch.bfloat16)
    )
    Cls._init_readout(m, vc)
    return m


def test_init_readout_reads_a_checkpoint_directory(tmp_path, caplog):
    (tmp_path / "decision_config.json").write_text(json.dumps(decision_config(temperature=1.5)))
    with caplog.at_level(logging.WARNING, logger="vllm.strands_decider_v2"):
        m = init_readout(tmp_path)
    assert m.code_ids.tolist() == IDS and m.temperature == 1.5 and m.softcap == 30.0
    assert m.readout.weight.shape == (255, 8) and m.readout.weight.dtype == torch.bfloat16
    assert caplog.text.count('no "interface" field') == 1
    (source,) = m.secondary_weights
    assert (source.model_or_path, source.prefix, source.allow_patterns_overrides) == (
        str(tmp_path),
        "readout.",
        ["readout.safetensors"],
    )


@pytest.mark.parametrize(
    "overrides, softcap, match",
    [
        ({"interface": "pplx-decider"}, 30.0, "interface must be"),
        ({"token_ids": IDS[:-1] + [IDS[0]]}, 30.0, "distinct"),
        ({"token_ids": IDS[:-1] + [262144]}, 30.0, "integers in"),
        ({"temperature": 0}, 30.0, "temperature"),
        ({}, None, "final_logit_softcapping is missing"),
        ({"final_logit_softcapping": 20.0}, 30.0, "differs"),
    ],
)
def test_init_readout_refuses_a_bad_decision_config(tmp_path, overrides, softcap, match):
    (tmp_path / "decision_config.json").write_text(json.dumps(decision_config(**overrides)))
    with pytest.raises(DecisionConfigError, match=match):
        init_readout(tmp_path, softcap=softcap)


def test_init_readout_refuses_a_directory_without_decision_config(tmp_path):
    with pytest.raises(DecisionConfigError, match="not found for model"):
        init_readout(tmp_path)


def test_init_readout_resolves_a_hub_repo_id_through_the_hf_cache(tmp_path, monkeypatch):
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    commit = "0123456789abcdef0123456789abcdef01234567"
    repo = tmp_path / "hub" / "models--example--sdv2-test"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit)
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "decision_config.json").write_text(
        json.dumps(decision_config(interface="strands-decider-v2", final_logit_softcapping=30.0))
    )
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.chdir(tmp_path)  # so "example/sdv2-test" is not a local path
    m = init_readout("example/sdv2-test")
    assert (
        m.code_ids.tolist() == IDS and m.secondary_weights[0].model_or_path == "example/sdv2-test"
    )
