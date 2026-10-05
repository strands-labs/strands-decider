"""TrainConfig.init_seed: it seeds the random initialisation (head, LoRA A) and nothing
else; `seed` keeps the data order and every draw made while training.

On CPU, on the tiny Qwen3.5 of tests/tiny_qwen35.py (nothing is downloaded). The weights a
run starts from are read where the optimiser is built (after initialisation, before the
first step), and the data order from the rows the training collator is handed.
"""

from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import load_file
from tiny_qwen35 import TARGETS, needs_tiny_qwen35, save_base, save_checkpoint

from strands_decider import train as train_mod
from strands_decider.data.collate import SystemOneCollator
from strands_decider.data.format import Example, write_jsonl
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.train import TrainConfig, train

pytestmark = needs_tiny_qwen35

BUILD_OPTIMIZER = train_mod._build_optimizer
COLLATE = SystemOneCollator.__call__


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return save_base(str(tmp_path_factory.mktemp("tiny-qwen35")))


@pytest.fixture(scope="module")
def ckpt(base, tmp_path_factory):
    return save_checkpoint(base, str(tmp_path_factory.mktemp("ckpt")))


def _rows(n: int) -> list[Example]:
    return [Example(kind="choice", state=f"Ticket {k}: refund please.", instructions="Which team?",
                    options=[["billing", ""], ["shipping", ""], ["sales", ""]], label=k % 3,
                    task="route") for k in range(n)]


def _start(tmp_path, monkeypatch, **over) -> dict:
    """Train one step; return the starting weights, the torch RNG state training began
    with, and the order the training collator saw the rows in."""
    from peft import get_peft_model_state_dict

    seen: dict = {"order": []}
    def snapshot(model, *args, **kw):
        seen["head"] = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
        seen["lora"] = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model.torso).items()}
        seen["rng"] = torch.get_rng_state()
        return BUILD_OPTIMIZER(model, *args, **kw)

    def record(self, rows):
        if self.train:
            seen["order"] += [ex.state for ex in rows]
        return COLLATE(self, rows)

    monkeypatch.setattr(train_mod, "_build_optimizer", snapshot)
    monkeypatch.setattr(SystemOneCollator, "__call__", record)
    write_jsonl(str(tmp_path / "train.jsonl"), _rows(16))
    write_jsonl(str(tmp_path / "val.jsonl"), _rows(2))
    cfg = TrainConfig(train_files=[str(tmp_path / "train.jsonl")], val_files=[str(tmp_path / "val.jsonl")],
                      head_type="pointer", pointer_dim=16, lora_targets=TARGETS, max_length=512,
                      micro_batch_size=2, grad_accum=1, max_steps=8, log_every=100, eval_every=0,
                      output_dir=str(tmp_path / "out"), **over)
    train(cfg)
    return seen


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_one_init_seed_gives_one_start_and_seed_still_orders_the_data(base, tmp_path, monkeypatch):
    s0 = _start(tmp_path / "s0", monkeypatch, base_model=base, init_seed=0, seed=0)
    s1 = _start(tmp_path / "s1", monkeypatch, base_model=base, init_seed=0, seed=1)
    assert _same(s0["head"], s1["head"]) and _same(s0["lora"], s1["lora"])
    assert any(".lora_A." in k and v.abs().sum() > 0 for k, v in s0["lora"].items())  # A is random
    assert s0["order"] != s1["order"] and sorted(s0["order"]) == sorted(s1["order"])
    # Training draws from `seed`: the torch RNG is reseeded with it once init is done.
    torch.manual_seed(1)
    assert torch.equal(s1["rng"], torch.get_rng_state())

    other = _start(tmp_path / "i1", monkeypatch, base_model=base, init_seed=1, seed=0)
    assert not _same(s0["head"], other["head"]) and not _same(s0["lora"], other["lora"])
    assert other["order"] == s0["order"]  # the data order is seed's alone


def test_without_init_seed_nothing_changes(base, tmp_path, monkeypatch):
    """init_seed=None is the code as it was: seed the torch RNG with `seed`, build, train
    on from there with no reseed."""
    run = _start(tmp_path / "run", monkeypatch, base_model=base, seed=3)
    torch.manual_seed(3)
    model = StrandsDeciderModel.from_pretrained_base(StrandsDeciderConfig(
        base_model=base, head_type="pointer", pointer_dim=16, lora_targets=TARGETS, max_length=512,
        head_dropout=0.05, ordinal_smoothing=0.1))
    from peft import get_peft_model_state_dict

    assert _same(run["head"], model.head.state_dict())
    assert _same(run["lora"], get_peft_model_state_dict(model.torso))
    assert torch.equal(run["rng"], torch.get_rng_state())  # no reseed after init


@pytest.mark.parametrize("init_seed", [None, 0, 7])
def test_continue_from_starts_from_the_checkpoint_whatever_the_init_seed(ckpt, tmp_path, monkeypatch,
                                                                         init_seed):
    run = _start(tmp_path, monkeypatch, continue_from=ckpt, init_seed=init_seed, seed=1)
    saved = load_file(os.path.join(ckpt, "lora", "adapter_model.safetensors"))
    assert _same(run["lora"], saved)
    assert _same(run["head"], torch.load(os.path.join(ckpt, "slot_head.pt")))
