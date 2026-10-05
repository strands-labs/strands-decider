"""TrainConfig.continue_from and kl_frozen_skip_kinds, trained for a few steps on CPU on
the tiny Qwen3.5 of tests/tiny_qwen35.py (nothing is downloaded).

* `continue_from` trains the checkpoint's own adapter and head (both move, the head
  from where it was, not from a fresh initialisation) and resets the calibration;
* `kl_frozen_skip_kinds` drops the frozen-KL term on rows of the listed kinds only, with
  the reference computed in the step or ahead of training.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import load_file
from tiny_qwen35 import needs_tiny_qwen35, save_base, save_checkpoint

from strands_decider.data.format import Example, write_jsonl
from strands_decider.train import TrainConfig, train

pytestmark = needs_tiny_qwen35


def _rows(kinds: str, n: int) -> list[Example]:
    out = []
    for k in range(n):
        if kinds[k % len(kinds)] == "n":
            out.append(Example(kind="noul", state=f"Order {k} shipped late.", instructions="Was it late?",
                               options=[["false", "on time"], ["true", "late"]], label=1, task="late"))
        else:
            out.append(Example(kind="choice", state=f"Ticket {k}: refund please.", instructions="Which team?",
                               options=[["billing", ""], ["shipping", ""], ["sales", ""]], label=0,
                               task="route"))
    return out


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    base = save_base(str(tmp_path_factory.mktemp("tiny-qwen35")))
    return save_checkpoint(base, str(tmp_path_factory.mktemp("ckpt")))


def _train(ckpt: str, tmp_path, kinds: str, **over) -> tuple[str, list[dict]]:
    write_jsonl(str(tmp_path / "train.jsonl"), _rows(kinds, 8))
    write_jsonl(str(tmp_path / "val.jsonl"), _rows(kinds, 2))
    cfg = TrainConfig(continue_from=ckpt, train_files=[str(tmp_path / "train.jsonl")],
                      val_files=[str(tmp_path / "val.jsonl")], head_type="pointer", max_length=512,
                      kl_frozen_weight=0.3, micro_batch_size=2, grad_accum=1, max_steps=3, lr=1e-2,
                      head_lr=1e-2, log_every=1, eval_every=0, output_dir=str(tmp_path / "out"), **over)
    out = train(cfg)
    with open(os.path.join(out, "history.json"), encoding="utf-8") as fh:
        return out, json.load(fh)


def _kl(history: list[dict]) -> list[float]:
    return [h["kl"] for h in history if "kl" in h]


def _lora(path: str) -> dict[str, torch.Tensor]:
    return load_file(os.path.join(path, "lora", "adapter_model.safetensors"))


def test_continue_from_trains_the_checkpoints_adapter_and_head(ckpt, tmp_path):
    out, _ = _train(ckpt, tmp_path, "nc")
    before, after = _lora(ckpt), _lora(out)
    assert before.keys() == after.keys()
    assert any(not torch.equal(before[k], after[k]) for k in before)  # the adapter trained
    h0 = torch.load(os.path.join(ckpt, "slot_head.pt"))
    h1 = torch.load(os.path.join(out, "slot_head.pt"))
    moved = sum(float((h1[k] - h0[k]).norm()) for k in h0)
    assert moved > 0  # the head trained ...
    assert moved < 0.5 * sum(float(v.norm()) for v in h0.values())  # ... from the checkpoint's head
    with open(os.path.join(out, "strands_decider_config.json"), encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["temperature"] == 1.0 and saved["temperature_by_kind"] == {}


def test_continue_from_records_and_trains_with_this_runs_head_settings(ckpt, tmp_path, monkeypatch):
    """The saved config is what was trained: this run's smoothing, and its head dropout,
    applied to the continued head (the checkpoint's own is 0)."""
    from strands_decider import train as train_mod

    seen = {}
    build = train_mod._build_optimizer

    def spy(model, *args, **kw):
        seen["p"] = [m.p for m in model.head.modules() if isinstance(m, torch.nn.Dropout)]
        return build(model, *args, **kw)

    monkeypatch.setattr(train_mod, "_build_optimizer", spy)
    out, _ = _train(ckpt, tmp_path, "nc", head_dropout=0.2, ordinal_smoothing=0.05)
    with open(os.path.join(out, "strands_decider_config.json"), encoding="utf-8") as fh:
        saved = json.load(fh)
    assert (saved["head_dropout"], saved["ordinal_smoothing"]) == (0.2, 0.05)
    assert seen["p"] == [0.2]


def test_continue_from_and_init_from_are_exclusive(ckpt):
    with pytest.raises(ValueError, match="at most one"):
        train(TrainConfig(continue_from=ckpt, init_from=ckpt))


def test_unknown_skip_kind_is_refused(ckpt):
    with pytest.raises(ValueError, match="kl_frozen_skip_kinds"):
        train(TrainConfig(continue_from=ckpt, kl_frozen_skip_kinds=["yesno"]))


@pytest.mark.parametrize("precompute", [False, True])
def test_skipped_kinds_get_no_frozen_kl_and_the_others_keep_it(ckpt, tmp_path, precompute):
    over = dict(kl_frozen_skip_kinds=["noul"], precompute_frozen_kl=precompute, group_by_length=precompute)
    _, yes_no = _train(ckpt, tmp_path / "noul", "n", **over)
    _, choice = _train(ckpt, tmp_path / "choice", "c", **over)
    _, anchored = _train(ckpt, tmp_path / "anchored", "n", precompute_frozen_kl=precompute,
                         group_by_length=precompute)
    assert _kl(yes_no) == [0.0, 0.0, 0.0]
    assert len(_kl(choice)) == 3 and all(kl > 0 for kl in _kl(choice))
    assert len(_kl(anchored)) == 3 and all(kl > 0 for kl in _kl(anchored))  # without the skip
