"""Seed-variance tools: val_split_seed and the EMA of the trained weights (train.py).

CPU only: tiny random Qwen3 torsos, as test_ddp.py.
"""
from __future__ import annotations

import pytest
import torch
from test_ddp import _examples, _tiny_model_factory, _tokenizer_file

import strands_decider.train as T
from strands_decider.data.format import write_jsonl
from strands_decider.modeling import StrandsDeciderModel

REAL_CLIP = torch.nn.utils.clip_grad_norm_
REAL_SPLIT = T.split_examples

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
