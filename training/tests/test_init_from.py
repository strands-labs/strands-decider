"""init_from: training continues from a trained checkpoint (a released model, or a soup).

CPU only: test_ddp.py's tiny Qwen3 torso, saved to disk as the checkpoints' base model,
and its test-trained tokenizer.
"""
from __future__ import annotations

import copy
import json

import pytest
import torch
from test_ddp import (
    _examples,
    _load_saved_torso,
    _tiny_checkpoint,
    _tiny_torso,
    _tokenizer_file,
)

import training.train as T
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import load_examples, write_jsonl
from training.soup import soup


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("init_from_corpus")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    tok = str(d / "tokenizer.json")
    torso = _tiny_torso(str(d / "torso"), tok)
    for seed in (0, 1):
        _tiny_checkpoint(str(d / f"s{seed}"), tok, torso, seed=seed)
    return d


@pytest.fixture(autouse=True)
def saved_torso(corpus, monkeypatch):
    monkeypatch.setattr(StrandsDeciderModel, "_load_torso",
                        staticmethod(_load_saved_torso(str(corpus / "torso"))))


def _batch(model, corpus):
    rows = load_examples([str(corpus / "val.jsonl")])[:8]
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=2048, head_type="pointer"),
                             train=False)
    b = coll(rows)
    return dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                n_slots=b["n_slots"], opt_idx=b["opt_idx"])


def _state(model):
    """The head and adapter tensors, by name."""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n.startswith("head.") or "lora_" in n}


def _train(corpus, out, monkeypatch, cfg=None, **over):
    """3 optimizer steps of train(); what the model was before the first one."""
    start = {}
    real = T._build_optimizer

    def build_optimizer(model, cfg):  # called once the model is ready, before any step
        b = _batch(model, corpus)
        model.eval()
        with torch.no_grad():
            start["out"] = model(**b)["log_probs"]
            start["frozen"] = _frozen(model, b)
        model.train()
        start["state"] = _state(model)
        start["trainable"] = {n for n, p in model.named_parameters() if p.requires_grad}
        start["checkpointing"] = model.torso.base_model.model.is_gradient_checkpointing
        start["config"] = copy.deepcopy(model.config)
        return real(model, cfg)

    monkeypatch.setattr(T, "_build_optimizer", build_optimizer)
    base = dict(train_files=[str(corpus / "a.jsonl")], val_files=[str(corpus / "val.jsonl")],
                init_from=str(corpus / "s0"), head_dropout=0.0, micro_batch_size=8,
                grad_accum=2, max_steps=3, log_every=1, eval_every=0, output_dir=str(out))
    T.train(cfg or T.TrainConfig(**{**base, **over}))
    with open(out / "history.json", encoding="utf-8") as fh:
        return start, json.load(fh)


def _frozen(model, b):
    """The frozen-KL readout: the torso's own option numbers, adapter disabled."""
    return model.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"])[0]


def _same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_training_continues_from_the_checkpoint(corpus, tmp_path, monkeypatch):
    src = StrandsDeciderModel.load(str(corpus / "s0")).eval()
    # head_type is set to the checkpoint's: no conflict
    start, _ = _train(corpus, tmp_path / "out", monkeypatch, head_type="pointer")
    # step 0 is the checkpoint: its head and adapter, its outputs at temperature 1
    assert _same(start["state"], _state(src))
    b = _batch(src, corpus)
    with torch.no_grad():
        want = src(**b, temperature=1.0)["log_probs"]
    torch.testing.assert_close(start["out"], want, rtol=0, atol=1e-6)
    assert start["config"].temperature == 1.0 and start["config"].temperature_by_kind == {}
    # the frozen readout is the base torso's own, without the checkpoint's adapter
    cfg = copy.deepcopy(src.config)
    cfg.use_lora = False
    plain = StrandsDeciderModel(cfg, _load_saved_torso(str(corpus / "torso"))(cfg), src.tokenizer)
    with torch.no_grad():
        torch.testing.assert_close(start["frozen"], _frozen(plain, b), rtol=0, atol=1e-6)
        hidden = src.encode(b["input_ids"], b["attention_mask"])  # the adapter matters here
        assert not torch.allclose(hidden, plain.encode(b["input_ids"], b["attention_mask"]))
    # the adapter and the head train, the base torso does not; checkpointing is on
    assert start["trainable"] == set(start["state"])
    assert any("lora_A" in n for n in start["trainable"]) and "head.q.weight" in start["trainable"]
    assert start["checkpointing"]
    # the result is a normal checkpoint, with the trained weights
    out = StrandsDeciderModel.load(str(tmp_path / "out")).eval()
    assert set(_state(out)) == set(_state(src)) and not _same(_state(out), _state(src))
    with torch.no_grad():
        assert not torch.allclose(out(**b)["log_probs"], want)
    assert (out.config.lora_r, out.config.pointer_dim, out.config.temperature) == (2, 16, 1.0)
    saved = json.load(open(tmp_path / "out" / "train_config.json"))
    assert (saved["base_model"], saved["lora_r"], saved["max_length"]) == ("tiny", 2, 2048)


def test_a_soup_trains_on_and_its_fine_tunes_soup(corpus, tmp_path, monkeypatch):
    soup([str(corpus / "s0"), str(corpus / "s1")], str(tmp_path / "soup"))
    src = StrandsDeciderModel.load(str(tmp_path / "soup")).eval()
    assert (src.config.lora_r, src.config.pointer_dim) == (4, 32)
    tunes = []
    for seed in (0, 1):
        out = tmp_path / f"tune{seed}"
        start, _ = _train(corpus, out, monkeypatch, init_from=str(tmp_path / "soup"), seed=seed)
        assert _same(start["state"], _state(src))
        with torch.no_grad():
            want = src(**_batch(src, corpus), temperature=1.0)["log_probs"]
        torch.testing.assert_close(start["out"], want, rtol=0, atol=1e-6)
        tuned = StrandsDeciderModel.load(str(out))
        assert (tuned.config.lora_r, tuned.config.pointer_dim) == (4, 32)
        assert not _same(_state(tuned), _state(src))
        tunes.append(str(out))
    # other seeds of one fine-tune are one recipe: they soup
    soup(tunes, str(tmp_path / "soup2"))
    s2 = StrandsDeciderModel.load(str(tmp_path / "soup2")).eval()
    assert (s2.config.lora_r, s2.config.pointer_dim) == (8, 64)
    with torch.no_grad():
        assert torch.isfinite(s2(**_batch(s2, corpus))["log_probs"][:, :2]).all()


def test_anchored_to_the_checkpoint_itself(corpus, tmp_path, monkeypatch):
    """kl_frozen_reference = init_from: the first step starts at KL 0 from its reference."""
    _, history = _train(corpus, tmp_path / "out", monkeypatch, kl_frozen_weight=0.3,
                        precompute_frozen_kl=True, group_by_length=True, length_group_mega=2,
                        kl_frozen_reference=str(corpus / "s0"))
    kl = [h["kl"] for h in history if "kl" in h]
    assert len(kl) == 3 and kl[0] < 1e-9 < kl[2]  # step 1 is at lr 0 (warmup)
    # without a reference, the anchor is the base torso's readout: not 0 at the start
    _, history = _train(corpus, tmp_path / "base", monkeypatch, kl_frozen_weight=0.3)
    assert history[0]["kl"] > 1e-3


@pytest.mark.parametrize("over", [dict(pointer_dim=32), dict(base_model="Qwen/Qwen3.5-2B-Base"),
                                  dict(head_hidden=8)])
def test_an_architecture_other_than_the_checkpoint_is_refused(corpus, tmp_path, monkeypatch, over):
    with pytest.raises(ValueError, match=f"sets {next(iter(over))}=.*remove {next(iter(over))}"):
        _train(corpus, tmp_path, monkeypatch, **over)


def test_every_conflict_is_named_in_one_error(corpus, tmp_path, monkeypatch):
    """A copied release recipe sets several of them: the error names all, not the first."""
    with pytest.raises(ValueError,
                       match=r"sets pointer_dim=32.*sets lora_r=4.*remove pointer_dim, lora_r from"):
        _train(corpus, tmp_path, monkeypatch, pointer_dim=32, lora_r=4)


def test_a_yaml_key_at_its_default_value_still_conflicts(corpus, tmp_path, monkeypatch):
    """A config copied from configs/train.yaml sets lora_r: 16, the default value."""
    path = tmp_path / "c.yaml"
    path.write_text(f"init_from: {corpus / 's0'}\nlora_r: 16\n"
                    f"train_files: [{corpus / 'a.jsonl'}]\n")
    with pytest.raises(ValueError, match="sets lora_r=16, the checkpoint has 2"):
        _train(corpus, tmp_path, monkeypatch, cfg=T.TrainConfig.from_yaml(str(path)))


def test_a_yaml_null_lora_targets_is_the_default_list(corpus, tmp_path):
    """lora_targets: null is the model's default list, the list the checkpoint has."""
    path = tmp_path / "c.yaml"
    path.write_text(f"init_from: {corpus / 's0'}\nlora_targets: null\n")
    cfg = T.TrainConfig.from_yaml(str(path))
    T._init_from(cfg)
    assert cfg.lora_targets == StrandsDeciderConfig().lora_targets


def test_head_init_cannot_replace_the_trained_head(corpus, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="head_init"):
        _train(corpus, tmp_path, monkeypatch, head_init="lm_head")
