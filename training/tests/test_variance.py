"""The checkpoint soup (soup.py), which averages a pointer-head model's seeds.

CPU only: tiny random Qwen3 torsos, as test_checkpoint_load.py does.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from test_checkpoint_load import _tokenizer
from test_hf_export import _export, _jev

from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from training import hf_export
from training.soup import soup

# ---------------------------------------------------------------- soup


@pytest.fixture
def ckpts(tmp_path, monkeypatch):
    """Three checkpoints of one recipe on one torso: their own adapters and heads."""
    from transformers import Qwen3Config, Qwen3Model

    base = Qwen3Model(
        Qwen3Config(
            vocab_size=6,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    monkeypatch.setattr(
        StrandsDeciderModel, "_load_torso", staticmethod(lambda *a: copy.deepcopy(base))
    )
    paths = []
    for seed in range(3):
        torch.manual_seed(seed)
        cfg = StrandsDeciderConfig(
            base_model=hf_export.BASE_MODEL,
            pointer_dim=16,
            lora_r=2,
            torch_dtype="float32",
            temperature=1.0 + seed,
        )
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
    return dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        n_slots=torch.tensor([3, 2]),
        opt_idx=torch.tensor([[2, 4, 6], [3, 5, 0]]),
        temperature=1.0,
    )


def _delta_w(model):
    """{module name: s * B @ A} over the adapter's layers."""
    return {
        n: m.lora_B["default"].weight @ m.lora_A["default"].weight * m.scaling["default"]
        for n, m in model.torso.named_modules()
        if hasattr(m, "lora_A") and "default" in m.lora_A
    }


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
        torch.testing.assert_close(
            s.head(decide, options), sum(m.head(decide, options) for m in models) / 3
        )
    # averaging the head tensors instead would not give this
    naive = copy.deepcopy(models[0].head)
    with torch.no_grad():
        for name, p in naive.named_parameters():
            p.copy_(sum(dict(m.head.named_parameters())[name] for m in models) / 3)
        assert not torch.allclose(naive(decide, options), s.head(decide, options), atol=1e-3)


def test_the_soup_exports(ckpts, tmp_path):
    out = tmp_path / "soup"
    soup(ckpts, str(out))
    jev = tmp_path / "jev"
    _jev(str(jev), hf_export.fingerprint(str(out)), 4096)
    assert _export(tmp_path, out, tmp_path / "hf", [jev]) == "written"
    hf_export.verify(str(tmp_path / "hf"))
    assert not (tmp_path / "hf" / "soup.json").exists()  # its host paths stay out of the export


def test_the_cli_writes_the_soup(ckpts, tmp_path):
    # `python -m training.soup` is a typer-wrapped command; replicate what it does.
    soup(ckpts, str(tmp_path / "cli"))
    assert json.load(open(tmp_path / "cli" / "soup.json"))["checkpoints"] == ckpts


def test_checkpoints_of_different_recipes_are_refused(ckpts, tmp_path):
    cfg = json.load(open(f"{ckpts[1]}/strands_decider_config.json"))
    json.dump({**cfg, "lora_r": 4}, open(f"{ckpts[1]}/strands_decider_config.json", "w"))
    with pytest.raises(ValueError, match="one recipe"):
        soup(ckpts, str(tmp_path / "soup"))


def test_soup_accepts_stale_slot_era_keys_in_the_input_configs(ckpts, tmp_path):
    """Pre-PR and v1-2610 configs still carry num_slots, head_type: 'pointer', head_hidden,
    head_init, kl_frozen_weight. Soup drops them cleanly through from_json and does not
    propagate them to the output config."""
    for c in ckpts:
        p = f"{c}/strands_decider_config.json"
        cfg = json.load(open(p))
        cfg.update(
            {"num_slots": 24, "head_hidden": 0, "head_init": "random", "kl_frozen_weight": 0.3}
        )
        json.dump(cfg, open(p, "w"))
    out = tmp_path / "soup"
    soup(ckpts, str(out))
    souped = json.load(open(out / "strands_decider_config.json"))
    for k in ("num_slots", "head_hidden", "head_init", "kl_frozen_weight"):
        assert k not in souped, k


def test_soup_rejects_a_slot_head_input_config(ckpts, tmp_path):
    """A slot-era checkpoint fails the config read before any adapter file is written."""
    p = f"{ckpts[0]}/strands_decider_config.json"
    cfg = json.load(open(p))
    json.dump({**cfg, "head_type": "slot"}, open(p, "w"))
    out = tmp_path / "soup"
    with pytest.raises(ValueError, match="pointer-head"):
        soup(ckpts, str(out))
    assert not out.exists()


def _tree(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


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
        json.dump(
            {**acfg, "target_parameters": ["experts.weight"]},
            open(f"{c}/lora/adapter_config.json", "w"),
        )
    with pytest.raises(ValueError, match="target_parameters"):
        soup(ckpts, str(tmp_path / "soup"))
    assert not (tmp_path / "soup").exists()


def test_a_nonempty_out_is_refused_unless_replaced(ckpts, tmp_path):
    out = tmp_path / "soup"
    out.mkdir()
    (out / "head.safetensors").write_bytes(b"stale")  # an earlier export's head
    (out / "provenance.json").write_text(
        json.dumps({"base_model": hf_export.BASE_MODEL, "base_model_revision": "unrelated"})
    )
    with pytest.raises(ValueError, match="not empty"):
        soup(ckpts, str(out))
    assert (out / "head.safetensors").read_bytes() == b"stale"
    soup(ckpts, str(out), replace=True)
    # the stale pre-existing files are gone; the fresh soup's head.safetensors replaces the
    # stale bytes but is a real safetensors file (loads cleanly).
    assert (out / "head.safetensors").read_bytes() != b"stale"
    assert not (out / "provenance.json").exists()
    assert StrandsDeciderModel.load(str(out)).config.base_revision is None


def test_a_regex_target_is_kept(ckpts, tmp_path):
    regex = r".*self_attn\.(q_proj|v_proj)"
    acfg = json.load(open(f"{ckpts[0]}/lora/adapter_config.json"))
    json.dump({**acfg, "target_modules": regex}, open(f"{ckpts[0]}/lora/adapter_config.json", "w"))
    soup([ckpts[0]], str(tmp_path / "soup"))
    assert (
        json.load(open(tmp_path / "soup" / "lora" / "adapter_config.json"))["target_modules"]
        == regex
    )
    da, db = (
        _delta_w(StrandsDeciderModel.load(ckpts[0])),
        _delta_w(StrandsDeciderModel.load(str(tmp_path / "soup"))),
    )
    assert set(da) == set(db) and all(torch.equal(da[k], db[k]) for k in da)


def test_base_revisions_resolve_as_the_loader_does(ckpts, tmp_path):
    def provenance(c, rev):
        with open(f"{c}/provenance.json", "w") as fh:
            json.dump({"base_model": hf_export.BASE_MODEL, "base_model_revision": rev}, fh)

    provenance(ckpts[0], "rev-a")
    provenance(ckpts[1], "rev-b")
    with pytest.raises(ValueError, match="base revisions"):
        soup(ckpts, str(tmp_path / "mixed"))
    # One pinned by its config, two by provenance: one base, and the soup pins it.
    provenance(ckpts[1], "rev-a")
    cfg = json.load(open(f"{ckpts[2]}/strands_decider_config.json"))
    json.dump(
        {**cfg, "base_revision": "rev-a"}, open(f"{ckpts[2]}/strands_decider_config.json", "w")
    )
    soup(ckpts, str(tmp_path / "soup"))
    assert (
        json.load(open(tmp_path / "soup" / "strands_decider_config.json"))["base_revision"]
        == "rev-a"
    )
