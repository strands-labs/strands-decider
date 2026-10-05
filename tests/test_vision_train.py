"""Image training (src/strands_decider/vision_train.py), on the tiny Qwen3.5 of
tests/tiny_qwen35.py with synthetic images; nothing is downloaded (the builders' own tests
are in test_image_data.py).

* image-removed copies are made only for rows the frozen reading covers, carry no image,
  no label weight and their own KL weight, and never reach validation; no image's rows
  straddle the validation split;
* a training run moves the checkpoint's adapter and head, and saves a checkpoint the
  vision engine loads.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

import pytest

transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from tiny_qwen35 import save_base, save_checkpoint  # noqa: E402

from strands_decider.vision import VisionDeciderModel  # noqa: E402
from strands_decider.vision_train import (  # noqa: E402
    ImageCollator,
    VisionTrainConfig,
    load_rows,
    plan_batches,
    row_losses,
    split,
    train,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data", "image"))
import common  # noqa: E402


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    base = save_base(str(tmp_path_factory.mktemp("tiny-qwen35")))
    return save_checkpoint(base, str(tmp_path_factory.mktemp("ckpt")))


def _image_rows(root, n: int = 8) -> None:
    """`n` pairs of image rows (a yes/no question and a 3-way choice per image), and one
    10-option row the frozen reading cannot cover, as the builders write them."""
    rng = random.Random(0)
    os.makedirs(root / "img", exist_ok=True)
    rows = []
    for k in range(n):
        name = f"img/{k}.png"
        Image.new("RGB", (64 + 16 * k, 48), (40 * k % 255, 90, 160)).save(root / name)
        kw = dict(source="synthetic", images=[name], source_id=f"img-{k}")
        rows.append(common.yesno(f"Is image {k} blue?", k % 2 == 0, rng, task="blue", pair_id=f"pair-{k // 2}", **kw))
        rows.append(common.row("choice", "Which colour?", [["red", ""], ["green", ""], ["blue", ""]], 2,
                               task="colour", **kw))
    rows.append(common.row("choice", "Which number?", [[str(i), ""] for i in range(10)], 3, task="many",
                           source="synthetic", images=["img/0.png"], source_id="img-0"))
    common.write(str(root / "rows.jsonl"), rows)
    text = [{"kind": "noul", "state": f"Ticket {k}", "instructions": "Urgent?", "options": common.NOUL_DEFAULT,
             "label": k % 2, "task": "text"} for k in range(6)]
    common.write(str(root / "text.jsonl"), text)


def _cfg(root, **over) -> VisionTrainConfig:
    return VisionTrainConfig(data_root=str(root), train_files=["rows.jsonl"],
                             text_replay_files=[str(root / "text.jsonl")], text_replay_n=4,
                             image_long_side=0, image_max_pixels=4096, workers=0, **over)


# ---- rows, copies and the split ---------------------------------------------------------


def test_image_removed_copies(tmp_path):
    _image_rows(tmp_path)
    rows = load_rows(_cfg(tmp_path, ablation_fraction=1.0))
    originals = [r for r in rows if r["images"]]
    copies = [r for r in rows if r["ablation"]]
    assert len(copies) == len(originals) - 1  # all but the 10-option row
    for c in copies:
        assert c["images"] == [] and c["weight"] == 0.0 and c["task"].endswith("/ablation")
        assert len(c["options"]) <= 9
    assert sum(r["source"] == "text_replay" for r in rows) == 4
    assert not any(r["ablation"] for r in load_rows(_cfg(tmp_path)))


def test_dedupe_drops_rows_by_image(tmp_path):
    _image_rows(tmp_path)
    (tmp_path / "drop.txt").write_text("img/3.png\n")
    rows = load_rows(_cfg(tmp_path, dedupe_drop=str(tmp_path / "drop.txt")))
    assert rows and not any("img/3.png" in r["images"] for r in rows)


def test_split_keeps_each_source_and_its_copies_on_one_side(tmp_path):
    _image_rows(tmp_path)
    rows = load_rows(_cfg(tmp_path, ablation_fraction=1.0))
    train_rows, val_rows = split(rows, 0.3, seed=0)
    assert val_rows and train_rows and not any(r["ablation"] for r in val_rows)
    assert {r["source_id"] for r in val_rows}.isdisjoint(r["source_id"] for r in train_rows)
    held = {r["source_id"] for r in val_rows}
    assert all(r["source_id"] in held for r in rows if r["ablation"] and r not in train_rows)


# ---- batches and the loss -----------------------------------------------------------------


def test_micro_batches_cover_every_row_once_within_budget():
    rng = random.Random(0)
    lengths = [rng.choice([rng.randrange(100, 600)] * 9 + [rng.randrange(2000, 5000)]) for _ in range(5000)]
    batches = plan_batches(lengths, max_rows=16, max_tokens=12000, seed=3)
    assert sorted(i for b in batches for i in b) == list(range(5000))
    for b in batches:
        assert len(b) <= 16 and (len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 12000)
    assert batches == plan_batches(lengths, max_rows=16, max_tokens=12000, seed=3)


@pytest.fixture(scope="module")
def model(ckpt):
    return VisionDeciderModel.load(ckpt, trainable=True)


def _batch(model, tmp_path, **over):
    _image_rows(tmp_path)
    cfg = _cfg(tmp_path, ablation_fraction=1.0, **over)
    rows = load_rows(cfg)
    rows = [r for r in rows if r["task"] == "colour"][:2] + [r for r in rows if r["task"] == "colour/ablation"][:2]
    return rows, ImageCollator(model.tokenizer, model.image_prompt(), cfg, train=True)(rows, index=3)


def test_collated_copies_have_no_image_and_no_label_weight(model, tmp_path):
    rows, batch = _batch(model, tmp_path)
    pad = model.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for i, r in enumerate(rows):
        has_image = bool((batch["input_ids"][i] == pad).any())
        assert has_image == (not r["ablation"])
        assert float(batch["weights"][i]) == (0.0 if r["ablation"] else 1.0)
        assert bool(batch["ablation"][i]) == r["ablation"]
    assert batch["image_grid_thw"].shape[0] == sum(not r["ablation"] for r in rows)


def test_copies_take_the_kl_only_weight(model, tmp_path):
    rows, batch = _batch(model, tmp_path)
    copy = torch.tensor([r["ablation"] for r in rows])
    assert copy.any() and (~copy).any()
    with torch.no_grad():
        ce, kl, _ = row_losses(model, batch, 0.3, 1.0)
        _, kl2, _ = row_losses(model, batch, 0.3, 2.0)
    assert torch.all(ce[copy] == 0) and torch.all(ce[~copy] > 0)
    assert torch.all(kl[copy] > 0)
    assert torch.allclose(kl2[copy], 2 * kl[copy]) and torch.allclose(kl2[~copy], kl[~copy])



def test_skipped_kinds_get_no_frozen_kl(model, tmp_path):
    _, plain = _batch(model, tmp_path)
    _, skipped = _batch(model, tmp_path, kl_frozen_skip_kinds=["choice"])  # every row is a choice
    assert not plain["kl_skip"].any() and skipped["kl_skip"].all()
    with torch.no_grad():
        _, kl, _ = row_losses(model, plain, 0.3, 1.0)
        _, none, _ = row_losses(model, skipped, 0.3, 1.0)
    assert torch.all(kl > 0) and torch.all(none == 0)


def test_replay_rows_with_a_teacher_train_toward_it(model, tmp_path):
    row = {"kind": "noul", "state": "Ticket 1 is open.", "instructions": "Is the ticket open?",
           "options": [["false", ""], ["true", ""]], "label": 1, "task": "text", "images": [],
           "source": "text_replay", "source_id": "t1", "teacher": [0.3, 0.7]}
    cfg = _cfg(tmp_path, teacher_weight=1.0)
    batch = ImageCollator(model.tokenizer, model.image_prompt(), cfg, train=True)([row])
    gold = int(batch["labels"][0])
    assert batch["label_dist"][0, gold] == pytest.approx((1 + 0.7) / 2)
    assert batch["label_dist"][0, 1 - gold] == pytest.approx(0.3 / 2)
    held_out = ImageCollator(model.tokenizer, model.image_prompt(), cfg, train=False)([row])
    assert "label_dist" not in held_out  # validation keeps the plain label


# ---- a run ---------------------------------------------------------------------------------


def test_training_moves_the_adapter_and_head_and_saves_a_vision_checkpoint(ckpt, tmp_path):
    import shutil

    from strands_decider.modeling import StrandsDeciderConfig, config_path

    _image_rows(tmp_path)
    start = shutil.copytree(ckpt, tmp_path / "start")
    cfg = StrandsDeciderConfig.from_json(config_path(str(start)))
    cfg.image_temperature_by_kind = {"noul": 0.5}  # fitted on the starting model's answers
    with open(config_path(str(start)), "w", encoding="utf-8") as fh:
        fh.write(cfg.to_json())
    out = train(_cfg(tmp_path, init_from=str(start), init_revision=None, ablation_fraction=0.5, rows_per_step=4,
                     micro_rows=4, max_steps=3, lr=1e-2, head_lr=1e-2, log_every=1, eval_every=0,
                     val_fraction=0.2, output_dir=str(tmp_path / "out")))
    before = load_file(os.path.join(ckpt, "lora", "adapter_model.safetensors"))
    after = load_file(os.path.join(out, "lora", "adapter_model.safetensors"))
    assert any(not torch.equal(v, after[k.replace("base_model.model.", "base_model.model.language_model.", 1)])
               for k, v in before.items())
    h0, h1 = torch.load(os.path.join(ckpt, "slot_head.pt")), torch.load(os.path.join(out, "slot_head.pt"))
    assert any(not torch.equal(h0[k], h1[k]) for k in h0)
    with open(os.path.join(out, "history.json"), encoding="utf-8") as fh:
        history = json.load(fh)
    assert [h["step"] for h in history if "ce" in h] == [1, 2, 3] and history[-1]["final"]
    with open(os.path.join(out, "data_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["base_model"] == StrandsDeciderConfig.from_json(config_path(ckpt)).base_model
    assert "transformers" in manifest["versions"]
    assert VisionDeciderModel.load(out).config.image_temperature_by_kind == {}  # stale: reset


def test_teacher_target_follows_the_slot_order_and_mixes_with_gold():
    """A replay row's teacher distribution (canonical order) lands on the slots of this
    rendering, mixed with the one-hot gold at teacher_weight."""
    from strands_decider.vision_train import ImageCollator, VisionTrainConfig

    col = object.__new__(ImageCollator)
    col.cfg = VisionTrainConfig(teacher_weight=1.0)
    # slot 0 shows canonical option 1, slot 1 shows canonical option 0; gold is canonical 1 -> slot 0
    t = col.teacher_target([0.2, 0.8], label=0, order=[1, 0])
    assert torch.allclose(t[:2], torch.tensor([(1.0 + 0.8) / 2, 0.2 / 2]))
    assert torch.isclose(t.sum(), torch.tensor(1.0))
    assert torch.all(t[2:] == 0)
