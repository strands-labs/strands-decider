"""Seed-variance tools: val_split_seed, the EMA of the trained weights (train.py) and the
checkpoint soup (soup.py).

CPU only: tiny random Qwen3 torsos, as test_checkpoint_load.py and test_ddp.py.
"""
from __future__ import annotations

import copy
import json

import pytest
import torch
from test_checkpoint_load import _tokenizer
from test_ddp import _examples, _tiny_model_factory, _tokenizer_file
from test_hf_export import _export, _jev

import strands_decider.train as T
from strands_decider import hf_export
from strands_decider.data.format import write_jsonl
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.soup import soup

REAL_CLIP = torch.nn.utils.clip_grad_norm_
REAL_SPLIT = T.split_examples

# ---------------------------------------------------------------- soup


@pytest.fixture
def ckpts(tmp_path, monkeypatch):
    """Three checkpoints of one recipe on one torso: their own adapters and heads."""
    from transformers import Qwen3Config, Qwen3Model

    base = Qwen3Model(Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=1, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=8))
    monkeypatch.setattr(StrandsDeciderModel, "_load_torso",
                        staticmethod(lambda *a: copy.deepcopy(base)))
    paths = []
    for seed in range(3):
        torch.manual_seed(seed)
        cfg = StrandsDeciderConfig(base_model=hf_export.BASE_MODEL, head_type="pointer",
                                   pointer_dim=16, lora_r=2, torch_dtype="float32",
                                   temperature=1.0 + seed)
        model = StrandsDeciderModel(cfg, copy.deepcopy(base), _tokenizer())
        model.attach_lora()
        with torch.no_grad():  # lora_B starts at zero, the norm at 1/0: make them matter
            for n, p in list(model.torso.named_parameters()) + list(model.head.named_parameters()):
                if "lora_" in n or "norm" in n:
                    p.normal_()
        model.save_pretrained(str(tmp_path / f"s{seed}"))
        paths.append(str(tmp_path / f"s{seed}"))
    return paths


def _inputs():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 6, (2, 7), generator=g)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), n_slots=torch.tensor([3, 2]),
                opt_idx=torch.tensor([[2, 4, 6], [3, 5, 0]]), temperature=1.0)


def _delta_w(model):
    """{module name: s * B @ A} over the adapter's layers."""
    return {n: m.lora_B["default"].weight @ m.lora_A["default"].weight * m.scaling["default"]
            for n, m in model.torso.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A}


def test_a_soup_of_one_checkpoint_is_that_checkpoint(ckpts, tmp_path):
    soup([ckpts[0]], str(tmp_path / "soup"))
    a = StrandsDeciderModel.load(ckpts[0]).eval()
    b = StrandsDeciderModel.load(str(tmp_path / "soup")).eval()
    da, db = _delta_w(a), _delta_w(b)
    assert set(da) == set(db) and all(torch.equal(da[k], db[k]) for k in da)
    with torch.no_grad():
        torch.testing.assert_close(b(**_inputs())["log_probs"], a(**_inputs())["log_probs"])
    assert b.config.temperature == 1.0 and b.config.temperature_by_kind == {}  # recalibrate


def test_the_soup_averages_delta_w_and_the_heads_logits(ckpts, tmp_path):
    acfg = json.load(open(f"{ckpts[1]}/lora/adapter_config.json"))  # PEFT's order is arbitrary
    acfg["target_modules"] = acfg["target_modules"][::-1]
    json.dump(acfg, open(f"{ckpts[1]}/lora/adapter_config.json", "w"))
    soup(ckpts, str(tmp_path / "soup"))
    models = [StrandsDeciderModel.load(c).eval() for c in ckpts]
    s = StrandsDeciderModel.load(str(tmp_path / "soup")).eval()
    assert s.config.lora_r == 6 and s.config.pointer_dim == 48
    ds, dws = _delta_w(s), [_delta_w(m) for m in models]
    for k in ds:
        torch.testing.assert_close(ds[k], sum(d[k] for d in dws) / 3)
    g = torch.Generator().manual_seed(2)
    decide, options = torch.randn(4, 32, generator=g), torch.randn(4, 5, 32, generator=g)
    with torch.no_grad():
        torch.testing.assert_close(s.head(decide, options),
                                   sum(m.head(decide, options) for m in models) / 3)
    # averaging the head tensors instead would not give this
    naive = copy.deepcopy(models[0].head)
    with torch.no_grad():
        for name, p in naive.named_parameters():
            p.copy_(sum(dict(m.head.named_parameters())[name] for m in models) / 3)
        assert not torch.allclose(naive(decide, options), s.head(decide, options), atol=1e-3)


@pytest.mark.parametrize("hidden", [0, 8])
def test_slot_heads_soup_to_their_mean_logit(hidden):
    from strands_decider.modeling import SlotHead
    from strands_decider.soup import soup_heads

    heads = [SlotHead(32, 5, hidden=hidden) for _ in range(3)]
    with torch.no_grad():
        for h in heads:
            h.norm.weight.normal_()
            h.norm.bias.normal_()
    s = SlotHead(32, 5, hidden=3 * hidden)
    s.load_state_dict(soup_heads([h.state_dict() for h in heads], "slot"))
    x = torch.randn(4, 32)
    with torch.no_grad():
        torch.testing.assert_close(s(x), sum(h(x) for h in heads) / 3)


def test_the_soup_exports(ckpts, tmp_path):
    out = tmp_path / "soup"
    soup(ckpts, str(out))
    jev = tmp_path / "jev"
    _jev(str(jev), hf_export.fingerprint(str(out)), 4096)
    assert _export(tmp_path, out, tmp_path / "hf", [jev]) == "written"
    hf_export.verify(str(tmp_path / "hf"))
    assert not (tmp_path / "hf" / "soup.json").exists()  # its host paths stay out of the export


def test_the_cli_writes_the_soup(ckpts, tmp_path):
    from typer.testing import CliRunner

    from strands_decider.cli import app

    res = CliRunner().invoke(app, ["soup", *ckpts, "--out", str(tmp_path / "cli")])
    assert res.exit_code == 0, res.output
    assert json.load(open(tmp_path / "cli" / "soup.json"))["checkpoints"] == ckpts


def test_checkpoints_of_different_recipes_are_refused(ckpts, tmp_path):
    cfg = json.load(open(f"{ckpts[1]}/strands_decider_config.json"))
    json.dump({**cfg, "kl_frozen_weight": 0.5}, open(f"{ckpts[1]}/strands_decider_config.json", "w"))
    with pytest.raises(ValueError, match="one recipe"):
        soup(ckpts, str(tmp_path / "soup"))


def _tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("where", ["same", "inside", "parent"])
def test_an_out_overlapping_an_input_is_refused_before_any_write(ckpts, tmp_path, where):
    from pathlib import Path

    first = Path(ckpts[0])
    out = {"same": first, "inside": first / "soup", "parent": tmp_path}[where]
    before = _tree(first)
    with pytest.raises(ValueError, match="overlaps"):
        soup(ckpts, str(out))
    assert _tree(first) == before


def test_expert_parameter_adapters_are_refused(ckpts, tmp_path):
    for c in ckpts:
        acfg = json.load(open(f"{c}/lora/adapter_config.json"))
        json.dump({**acfg, "target_parameters": ["experts.weight"]}, open(f"{c}/lora/adapter_config.json", "w"))
    with pytest.raises(ValueError, match="target_parameters"):
        soup(ckpts, str(tmp_path / "soup"))
    assert not (tmp_path / "soup").exists()


# ---------------------------------------------------------------- training


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("variance_corpus")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    return d


def _train(corpus, out, monkeypatch, **over):
    """3 optimizer steps; (each step's gradients, the validation rows, the saved head)."""
    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base",
                        staticmethod(_tiny_model_factory(str(corpus / "tokenizer.json"))))
    grads, vals = [], []

    def clip(params, max_norm, *a, **k):
        params = list(params)
        grads.append([p.grad.detach().clone() for p in params])
        return REAL_CLIP(params, max_norm, *a, **k)

    def split(*a, **k):
        tr, va = REAL_SPLIT(*a, **k)
        vals.append([ex.state for ex in va])
        return tr, va

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(T, "split_examples", split)
    base = dict(train_files=[str(corpus / "a.jsonl")], val_fraction=0.1, base_model="tiny",
                head_type="pointer", pointer_dim=16, head_dropout=0.0, max_length=2048,
                micro_batch_size=8, grad_accum=2, max_steps=3, log_every=1, eval_every=0,
                output_dir=str(out))
    T.train(T.TrainConfig(**{**base, **over}))
    return grads, vals[0], torch.load(out / "slot_head.pt", weights_only=True)


def _same(a, b):
    return all(torch.equal(x, y) for p, q in zip(a, b, strict=True) for x, y in zip(p, q, strict=True))


def test_val_split_seed_fixes_the_validation_rows_and_nothing_else(corpus, tmp_path, monkeypatch):
    g0, v0, _ = _train(corpus, tmp_path / "a", monkeypatch, seed=0)
    v1 = _train(corpus, tmp_path / "b", monkeypatch, seed=1)[1]
    assert v0 != v1  # by default each seed holds out its own rows
    f0 = _train(corpus, tmp_path / "c", monkeypatch, seed=0, val_split_seed=7)[1]
    f1 = _train(corpus, tmp_path / "d", monkeypatch, seed=1, val_split_seed=7)[1]
    assert f0 == f1 and f0 != v0
    g, v, h = _train(corpus, tmp_path / "e", monkeypatch, seed=0, val_split_seed=0)
    h0 = torch.load(tmp_path / "a" / "slot_head.pt", weights_only=True)
    assert v == v0 and _same(g, g0) and all(torch.equal(h[k], h0[k]) for k in h)


def test_ema_saves_the_average_and_leaves_training_alone(corpus, tmp_path, monkeypatch):
    g_off, _, h_off = _train(corpus, tmp_path / "off", monkeypatch)
    g_on, _, h_on = _train(corpus, tmp_path / "on", monkeypatch, ema_decay=0.99, save_every=1)
    assert _same(g_on, g_off)  # the mid-run saves restored the live weights
    assert not all(torch.equal(h_on[k], h_off[k]) for k in h_on)
    # a vanishing decay averages nothing in: the last weights
    _, _, h_last = _train(corpus, tmp_path / "last", monkeypatch, ema_decay=1e-12)
    for k in h_off:
        torch.testing.assert_close(h_last[k], h_off[k])


def test_ema_arithmetic():
    lin = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema = T._Ema(lin, 0.5)
    for step, w in ((1, 3.0), (2, 5.0)):
        with torch.no_grad():
            lin.weight.fill_(w)
        ema.update(step)
    # step 1: decay min(0.5, 2/11); step 2: min(0.5, 3/12)
    d1, d2 = 2 / 11, 3 / 12
    want = (1.0 * d1 + 3.0 * (1 - d1)) * d2 + 5.0 * (1 - d2)
    ema.apply()
    assert lin.weight.item() == pytest.approx(want)
    ema.restore()
    assert lin.weight.item() == 5.0
