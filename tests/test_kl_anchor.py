"""kl_frozen_reference: a trained checkpoint's own distribution as the frozen-KL reference.

CPU only, with test_ddp.py's tiny Qwen3 torso and test-trained tokenizer.
"""
from __future__ import annotations

import json
import math

import pytest
import torch
from test_ddp import _examples, _tiny_model_factory, _tokenizer_file

import strands_decider.train as T
from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import load_examples, write_jsonl
from strands_decider.modeling import StrandsDeciderModel

REAL_CLIP = torch.nn.utils.clip_grad_norm_


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("anchor_corpus")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    _tokenizer_file(str(d / "other_tokenizer.json"), _examples(60, seed=7), vocab_size=300)
    return d


def _train(corpus, out, monkeypatch, **over):
    """2 optimizer steps of train(); the gradient of each step, where train.py clips it."""
    factory = _tiny_model_factory(str(corpus / "tokenizer.json"))
    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", staticmethod(factory))
    grads = []

    def clip(params, max_norm, *a, **k):
        params = list(params)
        grads.append([p.grad.detach().clone() for p in params])
        return REAL_CLIP(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    base = dict(train_files=[str(corpus / "a.jsonl")], val_files=[str(corpus / "val.jsonl")],
                base_model="tiny", head_type="pointer", pointer_dim=16, kl_frozen_weight=0.3,
                head_dropout=0.0, max_length=2048, micro_batch_size=8, grad_accum=4,
                max_steps=2, group_by_length=True, length_group_mega=2, log_every=1,
                eval_every=0, output_dir=str(out))
    T.train(T.TrainConfig(**{**base, **over}))
    with open(out / "history.json", encoding="utf-8") as fh:
        return grads, json.load(fh)


def _tiny(corpus, tokenizer="tokenizer.json"):
    cfg = T.StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16)
    return _tiny_model_factory(str(corpus / tokenizer))(cfg)


def _batches():
    return [list(range(i, i + 8)) for i in range(0, 48, 8)]


@pytest.mark.parametrize("over", [dict(), dict(kl_frozen_weight=0.0, precompute_frozen_kl=True)])
def test_reference_needs_the_precomputed_pass_and_the_kl_term(corpus, tmp_path, monkeypatch, over):
    with pytest.raises(ValueError, match="kl_frozen_reference"):
        _train(corpus, tmp_path, monkeypatch, kl_frozen_reference=str(tmp_path), **over)


def test_reference_rows_are_the_checkpoint_own_distribution(corpus):
    torch.manual_seed(0)
    student, ref_model = _tiny(corpus), _tiny(corpus).eval()
    rows = load_examples([str(corpus / "a.jsonl")])
    coll_cfg = CollatorConfig(max_length=2048, head_type="pointer", seed=0)
    ref = T._frozen_reference(student, rows, _batches(), coll_cfg, "cpu", ref_model)
    coll = SystemOneCollator(student.tokenizer, coll_cfg, train=True)  # the same renders
    for m, idx in enumerate(_batches()):
        b = coll([rows[i] for i in idx])
        with torch.no_grad():
            lp = ref_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                           n_slots=b["n_slots"], opt_idx=b["opt_idx"], temperature=1.0)["log_probs"]
        for r, n in enumerate(b["n_slots"].tolist()):
            assert torch.equal(ref[m, r, :n], lp[r, :n])
            assert torch.isneginf(ref[m, r, n:]).all()
    base = T._frozen_reference(student, rows, _batches(), coll_cfg, "cpu")
    assert not torch.equal(base, ref)


def test_reference_with_its_own_tokenizer_sees_the_student_slot_order(corpus):
    """A Gemma student, a Qwen reference: the reference renders the rows with its own
    tokenizer, and every row has the same options in the same slots as in training."""
    torch.manual_seed(0)
    student, ref_model = _tiny(corpus), _tiny(corpus, "other_tokenizer.json").eval()
    assert ref_model.tokenizer.get_vocab() != student.tokenizer.get_vocab()
    rows = load_examples([str(corpus / "a.jsonl")])
    for i, ex in enumerate(rows):  # a teacher distribution makes each slot identifiable
        ex.teacher = [(j + 1) / 100 + i for j in range(ex.n_options)]
    coll_cfg = CollatorConfig(max_length=2048, head_type="pointer", seed=0)
    ref = T._frozen_reference(student, rows, _batches(), coll_cfg, "cpu", ref_model)
    stu_coll = SystemOneCollator(student.tokenizer, coll_cfg, train=True)  # training's renders
    ref_coll = SystemOneCollator(ref_model.tokenizer, coll_cfg, train=True)
    differ = 0
    for m, idx in enumerate(_batches()):
        s, b = stu_coll([rows[i] for i in idx]), ref_coll([rows[i] for i in idx])
        differ += s["input_ids"].shape != b["input_ids"].shape or not torch.equal(
            s["input_ids"], b["input_ids"])
        for k in ("labels", "n_slots", "teacher", "label_dist"):
            assert torch.equal(s[k], b[k]), (m, k)  # the same permutation of every row
        with torch.no_grad():
            lp = ref_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                           n_slots=b["n_slots"], opt_idx=b["opt_idx"], temperature=1.0)["log_probs"]
        for r, n in enumerate(b["n_slots"].tolist()):
            assert torch.equal(ref[m, r, :n], lp[r, :n])
            assert torch.isneginf(ref[m, r, n:]).all()
    assert differ == len(_batches())  # the two tokenizers really encode differently


@pytest.mark.parametrize("tokenizer", ["tokenizer.json", "other_tokenizer.json"])
def test_train_with_a_reference(corpus, tmp_path, monkeypatch, tokenizer):
    torch.manual_seed(1)  # seed 0 would build the student's own initial weights: KL 0
    ref_model = _tiny(corpus, tokenizer)
    monkeypatch.setattr(StrandsDeciderModel, "load", staticmethod(lambda path, **k: ref_model))
    plain, _ = _train(corpus, tmp_path / "plain", monkeypatch, precompute_frozen_kl=True)
    grads, history = _train(corpus, tmp_path / "ref", monkeypatch, precompute_frozen_kl=True,
                            kl_frozen_reference="ref")
    assert all(h["kl"] > 0 for h in history if "kl" in h)
    assert not all(torch.equal(x, y) for p, q in zip(plain, grads, strict=True)
                   for x, y in zip(p, q, strict=True))


def _same(a, b):
    return all(torch.equal(x, y) for p, q in zip(a, b, strict=True)
               for x, y in zip(p, q, strict=True))


def _dists(seed):
    g = torch.Generator().manual_seed(seed)
    stu = torch.log_softmax(torch.randn(6, 5, generator=g), -1)
    ref = torch.log_softmax(torch.randn(6, 5, generator=g), -1)
    stu[:, 4] = ref[:, 4] = float("-inf")  # a masked slot, as past a row's options
    tea = ref.exp()
    tea[0] = 0.0  # a row without a teacher
    return stu, ref, tea


def test_forward_kl_is_the_formula_every_run_so_far_used():
    stu, ref, tea = _dists(0)
    valid = torch.isfinite(ref) & torch.isfinite(stu)
    old = (ref.exp().masked_fill(~valid, 0.0) * (ref - stu).masked_fill(~valid, 0.0)).sum(-1)
    assert torch.equal(T._kl(ref, stu, valid, "forward"), old)
    tv = (tea > 0) & torch.isfinite(stu)
    old_t = (tea.masked_fill(~tv, 0.0)
             * (tea.clamp_min(1e-12).log() - stu).masked_fill(~tv, 0.0)).sum(-1)
    assert torch.equal(T._kl(tea.clamp_min(1e-12).log(), stu, tv, "forward", tea), old_t)


def test_reverse_kl_is_student_against_target():
    stu, ref, _ = _dists(1)
    valid = torch.isfinite(ref) & torch.isfinite(stu)
    want = (stu.exp()[:, :4] * (stu[:, :4] - ref[:, :4])).sum(-1)
    assert torch.allclose(T._kl(ref, stu, valid, "reverse"), want)
    assert not torch.allclose(T._kl(ref, stu, valid, "forward"), want)


def test_reverse_kl_changes_training_and_unknown_direction_is_refused(corpus, tmp_path, monkeypatch):
    plain, _ = _train(corpus, tmp_path / "plain", monkeypatch)
    fwd, _ = _train(corpus, tmp_path / "fwd", monkeypatch, kl_direction="forward")
    rev, _ = _train(corpus, tmp_path / "rev", monkeypatch, kl_direction="reverse")
    assert _same(plain, fwd) and not _same(fwd, rev)
    with pytest.raises(ValueError, match="kl_direction"):
        _train(corpus, tmp_path / "bad", monkeypatch, kl_direction="sideways")


def test_teacher_forward_kl_is_unchanged():
    stu, _, tea = _dists(2)
    tv = (tea > 0) & torch.isfinite(stu)
    old = T._kl(tea.clamp_min(1e-12).log(), stu, tv, "forward", tea)
    assert torch.equal(T._teacher_kl(tea, stu, "forward"), old)


def test_reverse_teacher_kl_charges_mass_on_a_zero_target():
    # A rounded teacher [1, 0] against a uniform student: the review's reproducer.
    tea = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    logits = torch.tensor([[0.0, 0.0, float("-inf")]], dtype=torch.float64, requires_grad=True)
    loss = T._teacher_kl(tea, logits.log_softmax(-1), "reverse").sum()
    loss.backward()
    assert loss.item() > 0
    after = (logits.detach() - 0.1 * logits.grad).softmax(-1)
    assert after[0, 0].item() > 0.5  # one step moves the student toward the teacher
    floor = T.REVERSE_KL_TARGET_FLOOR
    want = 0.5 * math.log(0.5 / (1 / (1 + floor))) + 0.5 * math.log(0.5 / (floor / (1 + floor)))
    assert loss.item() == pytest.approx(want)
