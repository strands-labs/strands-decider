"""Training grafted eyes (src/strands_decider/graft_align.py, and vision_train.py with
`projector_from`), on the tiny SigLIP and Llama of tests/tiny_graft.py; nothing is
downloaded. The tests pin what training promises:

* stage 1 lowers the caption loss and changes only the projector;
* stage 2 lowers the decision loss, moves the adapter, head and projector, never the
  encoder, and saves a checkpoint the server and evaluation load;
* the caption builder and stage 1 refuse COCO val2014 images (POPE's).
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
from tiny_graft import (  # noqa: E402
    save_minicpm_like,
    save_siglip,
    save_stage1,
    save_text_checkpoint,
)

from strands_decider.graft import GraftedDeciderModel, load_projector_state  # noqa: E402
from strands_decider.vision import is_grafted  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data", "image"))
import captions  # noqa: E402
import common  # noqa: E402


@pytest.fixture(scope="module")
def enc(tmp_path_factory):
    return save_siglip(str(tmp_path_factory.mktemp("siglip")))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return save_minicpm_like(str(tmp_path_factory.mktemp("minicpm-like")))


@pytest.fixture(scope="module")
def text_ckpt(base, tmp_path_factory):
    return save_text_checkpoint(base, str(tmp_path_factory.mktemp("text-ckpt")))


@pytest.fixture(scope="module")
def stage1(enc, tmp_path_factory):
    return save_stage1(enc, str(tmp_path_factory.mktemp("stage1")))


# ---- stage 1 -------------------------------------------------------------------------------------


def _caption_rows(root, n: int = 8) -> None:
    os.makedirs(root / "coco", exist_ok=True)
    colours = {"red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220), "yellow": (230, 220, 30)}
    with open(root / "captions.jsonl", "w", encoding="utf-8") as fh:
        for k in range(n):
            name = list(colours)[k % 4]
            path = f"coco/COCO_train2014_{k:012d}.jpg"
            Image.new("RGB", (48 + 8 * k, 40), colours[name]).save(root / path, format="JPEG")
            fh.write(json.dumps({"images": [path], "caption": f"A {name} square.",
                                 "source": "coco_captions_train2014", "source_id": f"coco-{k}"}) + "\n")


def _align_cfg(root, base, enc, **over):
    from strands_decider.graft_align import AlignConfig

    kw = dict(base_model=base, base_revision=None, torch_dtype="float32", encoder=enc,
              encoder_revision=None, data_root=str(root), val_rows=0, batch_size=4, micro_batch=4,
              workers=0, max_steps=3, lr=1e-2, warmup_ratio=0.0, log_every=1, eval_every=0,
              output_dir=str(root / "align"))
    return AlignConfig(**{**kw, **over})


def test_stage1_lowers_caption_loss_and_moves_only_the_projector(base, enc, text_ckpt, tmp_path):
    from strands_decider.graft_align import (
        _loader,
        build_aligner,
        caption_loss,
        fit,
        load_rows,
        train,
    )

    _caption_rows(tmp_path)
    cfg = _align_cfg(tmp_path, base, enc)
    aligner = build_aligner(cfg)
    assert (aligner.pcfg.tokens_per_image, aligner.pcfg.text_hidden) == (16, 64)
    rows = load_rows(cfg)
    # the loss covers the caption and the end-of-text token, nothing of the prompt
    batch = next(iter(_loader(aligner, rows[:1], cfg, 0, shuffle=False)))
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    caption = aligner.tokenizer(rows[0]["caption"], add_special_tokens=False)["input_ids"]
    target = [*caption, aligner.tokenizer.eos_token_id]
    assert [y for y in labels if y != -100] == target == ids[-len(target):]
    assert ids.count(aligner.slot_id) == 16 and ids[0] == aligner.tokenizer.bos_token_id
    frozen = {f"lm.{k}": v.clone() for k, v in aligner.lm.state_dict().items()}
    frozen.update({f"enc.{k}": v.clone() for k, v in aligner.encoder.state_dict().items()})
    proj0 = {k: v.clone() for k, v in aligner.projector.state_dict().items()}
    before = caption_loss(aligner, _loader(aligner, rows, cfg, 0, shuffle=False), "cpu")
    history = fit(aligner, rows, [], cfg, "cpu")
    after = caption_loss(aligner, _loader(aligner, rows, cfg, 0, shuffle=False), "cpu")
    assert [h["step"] for h in history] == [1, 2, 3]
    assert after < before
    now = {f"lm.{k}": v for k, v in aligner.lm.state_dict().items()}
    now.update({f"enc.{k}": v for k, v in aligner.encoder.state_dict().items()})
    assert all(torch.equal(frozen[k], now[k]) for k in frozen)
    assert all(not torch.equal(proj0[k], v) for k, v in aligner.projector.state_dict().items())
    # the whole stage writes a projector directory stage 2 starts from
    out = train(_align_cfg(tmp_path, base, enc, max_steps=1))
    assert os.path.exists(os.path.join(out, "projector.safetensors"))
    with open(os.path.join(out, "data_manifest.json"), encoding="utf-8") as fh:
        assert "torch" in json.load(fh)["versions"]
    grafted = GraftedDeciderModel.load(text_ckpt, projector_from=out)
    assert grafted.pcfg == aligner.pcfg


def test_caption_rows_refuse_val2014(base, enc, tmp_path):
    good = {"images": [{"id": 1, "file_name": "COCO_train2014_000000000001.jpg"},
                       {"id": 2, "file_name": "COCO_train2014_000000000002.jpg"}],
            "annotations": [{"id": 10, "image_id": 1, "caption": " a dog  on a sofa"},
                            {"id": 11, "image_id": 1, "caption": "A brown dog."},
                            {"id": 12, "image_id": 2, "caption": "two cats"}]}
    rows = captions.build(good, n=10, seed=0)
    assert [r["images"] for r in rows] == [["coco/COCO_train2014_000000000001.jpg"],
                                           ["coco/COCO_train2014_000000000002.jpg"]]
    assert rows[0]["caption"] in {"A dog on a sofa.", "A brown dog."} and rows[1]["caption"] == "Two cats."
    assert len(captions.build(good, n=1, seed=0)) == 1
    bad = {**good, "images": [*good["images"], {"id": 3, "file_name": "COCO_val2014_000000000003.jpg"}]}
    with pytest.raises(ValueError, match="val2014"):
        captions.build(bad, n=10)
    # and stage 1 refuses a row file holding one, whoever wrote it
    from strands_decider.graft_align import load_rows

    _caption_rows(tmp_path, n=2)
    with open(tmp_path / "captions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"images": ["coco/COCO_val2014_000000000003.jpg"], "caption": "x",
                             "source": "s", "source_id": "v"}) + "\n")
    with pytest.raises(ValueError, match="val2014"):
        load_rows(_align_cfg(tmp_path, base, enc))


# ---- stage 2 -------------------------------------------------------------------------------------


def _image_rows(root, n: int = 6) -> None:
    rng = random.Random(0)
    os.makedirs(root / "img", exist_ok=True)
    rows = []
    for k in range(n):
        name = f"img/{k}.png"
        Image.new("RGB", (64 + 16 * k, 48), (200 if k % 2 else 20, 90, 160)).save(root / name)
        kw = dict(source="synthetic", images=[name], source_id=f"img-{k}")
        rows.append(common.yesno(f"Is image {k} red?", k % 2 == 1, rng, task="red", **kw))
        rows.append(common.row("choice", "Which colour?", [["red", ""], ["blue", ""]], 0 if k % 2 else 1,
                               task="colour", **kw))
    common.write(str(root / "rows.jsonl"), rows)
    text = [{"kind": "noul", "state": f"Ticket {k}", "instructions": "Urgent?", "options": common.NOUL_DEFAULT,
             "label": k % 2, "task": "text"} for k in range(4)]
    common.write(str(root / "text.jsonl"), text)


def test_stage2_lowers_the_loss_and_saves_a_servable_checkpoint(text_ckpt, stage1, tmp_path):
    from safetensors.torch import load_file

    from strands_decider.vision_train import (
        ImageCollator,
        VisionTrainConfig,
        load_model,
        row_losses,
        train,
    )

    _image_rows(tmp_path)
    cfg = VisionTrainConfig(init_from=text_ckpt, init_revision=None, projector_from=stage1,
                            data_root=str(tmp_path), train_files=["rows.jsonl"],
                            text_replay_files=[str(tmp_path / "text.jsonl")], text_replay_n=4,
                            ablation_fraction=0.0, image_long_side=0, image_max_pixels=4096, workers=0,
                            # every row in each of the 3 steps (3 epochs): the loss is the full-batch one
                            epochs=3, rows_per_step=16, micro_rows=8, lr=3e-3, head_lr=3e-3,
                            projector_lr=3e-3, warmup_ratio=0.0, log_every=1, eval_every=0,
                            val_fraction=0.0, est_image_tokens=16, output_dir=str(tmp_path / "out"))
    start = load_model(cfg)
    assert not any(p.requires_grad for p in start.encoder.parameters())
    trainable = {n.split(".")[0] for n, p in start.named_parameters() if p.requires_grad}
    assert trainable == {"torso", "head", "projector"}

    with open(tmp_path / "rows.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    def loss(model):
        model.eval()
        batch = ImageCollator(model.tokenizer, model.image_prompt(), cfg, train=False)(rows)
        assert "pixel_values" in batch and "image_grid_thw" not in batch
        with torch.no_grad():
            ce, _, _ = row_losses(model, batch, 0.0, 0.0)
        return float(ce.mean())

    before = loss(start)
    out = train(cfg)
    after_model = GraftedDeciderModel.load(out)
    assert loss(after_model) < before
    p0, p1 = load_projector_state(stage1), load_projector_state(out)
    assert any(not torch.equal(p0[k], p1[k]) for k in p0)
    l0 = load_file(os.path.join(text_ckpt, "lora", "adapter_model.safetensors"))
    l1 = load_file(os.path.join(out, "lora", "adapter_model.safetensors"))
    assert l0.keys() == l1.keys() and any(not torch.equal(l0[k], l1[k]) for k in l0)
    h0, h1 = torch.load(os.path.join(text_ckpt, "slot_head.pt")), torch.load(os.path.join(out, "slot_head.pt"))
    assert any(not torch.equal(h0[k], h1[k]) for k in h0)
    with open(os.path.join(out, "history.json"), encoding="utf-8") as fh:
        assert [h["step"] for h in json.load(fh) if "ce" in h] == [1, 2, 3]
    assert is_grafted(out)
    with open(os.path.join(out, "data_manifest.json"), encoding="utf-8") as fh:
        assert "torch" in json.load(fh)["versions"]
