"""Stage 1 of the graft (graft.py): align the projector to a frozen text-only base LM.

Only the projector trains. The base LM (with its own output head, which a decider
checkpoint never loads) and the SigLIP encoder are frozen and run as they will at
inference. Each row is one COCO train2014 image and one of its captions
(data/image/captions.py), laid out as a decider prompt lays out an image, then the
caption:

    <s><state>\\n<image>SLOT x 144</image>\\n</state>\\nA man rides a horse on a beach.</s>

and the loss is next-token cross-entropy on the caption and the end-of-text token only.
So the projector learns to write image slots the base LM reads as the image's content,
in the position stage 2 and the server put them.

    python -m strands_decider.graft_align configs/align/strands-decider-2.5B-minicpm-v21-vl.yaml [key=value ...]

Writes `projector.safetensors` and `projector_config.json` (what stage 2's
`projector_from` reads), plus history.json and train_config.json. One process, one GPU:
80,000 rows less the validation share (79,488 trained) at 128 a step is 621 steps.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .graft import (
    SIGLIP2_SO400M,
    SIGLIP2_SO400M_REV,
    Projector,
    ProjectorConfig,
    SiglipImages,
    encode_images,
    load_encoder,
    load_image_processor,
    place_images,
    save_projector,
    slot_token_id,
)
from .runinfo import library_versions
from .train import YamlConfig, _lr_lambda
from .vision import fit_image, read_image

Row = dict[str, Any]


@dataclass
class AlignConfig(YamlConfig):
    # the frozen base LM, loaded with its output head
    base_model: str = "openbmb/MiniCPM5-2B"
    base_revision: str | None = "f97400052a43d642bbc6e9975e2397e3ae6a6b52"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None

    # the encoder and projector (graft.ProjectorConfig)
    encoder: str = SIGLIP2_SO400M
    encoder_revision: str | None = SIGLIP2_SO400M_REV
    unshuffle: int = 2
    image_slot: str = "<unused_token_0>"

    # data: data/image/captions.py's rows, image paths relative to data_root
    data_root: str = "data/image/build"
    caption_files: list[str] = field(default_factory=lambda: ["captions.jsonl"])
    val_rows: int = 512
    image_max_pixels: int = 400_000  # decoded and resized as vision_train decodes
    max_caption_tokens: int = 64

    # optimisation
    epochs: int = 1
    max_steps: int = 0  # 0 = full epochs
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    batch_size: int = 128  # rows per optimizer step ...
    micro_batch: int = 64  # ... in micro-batches of this many
    gradient_checkpointing: bool = False
    workers: int = 8

    output_dir: str = "checkpoints/strands-decider-2.5B-minicpm-v21-vl-align"
    seed: int = 0
    log_every: int = 20
    eval_every: int = 200


# ---- data --------------------------------------------------------------------------


def refuse_val2014(path: str) -> None:
    """COCO val2014 holds POPE's images; captions must come from train2014 only."""
    if "val2014" in path:
        raise ValueError(f"{path}: a COCO val2014 image; POPE evaluates on val2014, train on train2014 only")


def load_rows(cfg: AlignConfig) -> list[Row]:
    rows: list[Row] = []
    for f in cfg.caption_files:
        with open(os.path.join(cfg.data_root, f), encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    for r in rows:
        for p in r["images"]:
            refuse_val2014(p)
    return rows


class CaptionCollator:
    """Rows -> one micro-batch: the image inside <state>, then the caption, labelled."""

    def __init__(self, tokenizer: Any, prompter: SiglipImages, cfg: AlignConfig):
        self.tok, self.prompter, self.cfg = tokenizer, prompter, cfg

    def image(self, path: str) -> Any:
        with open(os.path.join(self.cfg.data_root, path), "rb") as fh:
            return fit_image(read_image(fh.read()), 0, self.cfg.image_max_pixels)

    def __call__(self, rows: list[Row]) -> dict[str, torch.Tensor]:
        counts, mm = self.prompter.process([self.image(p) for r in rows for p in r["images"]])
        ids, labels, c = [], [], 0
        for r in rows:
            n = len(r["images"])
            prompt = self.tok(self.prompter.state("", counts[c : c + n]), add_special_tokens=True)["input_ids"]
            c += n
            caption = self.tok(r["caption"], add_special_tokens=False)["input_ids"][: self.cfg.max_caption_tokens]
            caption = [*caption, self.tok.eos_token_id]
            ids.append(prompt + caption)
            labels.append([-100] * len(prompt) + caption)
        pad, longest = self.tok.pad_token_id, max(map(len, ids))
        return {
            "input_ids": torch.tensor([x + [pad] * (longest - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (longest - len(x)) for x in ids]),
            "labels": torch.tensor([y + [-100] * (longest - len(y)) for y in labels]),
            **mm,
        }


class Batches(Dataset):
    def __init__(self, rows: list[Row], batches: list[list[int]], collate: CaptionCollator):
        self.rows, self.batches, self.collate = rows, batches, collate

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.collate([self.rows[j] for j in self.batches[i]])


# ---- model ---------------------------------------------------------------------------


class CaptionAligner(nn.Module):
    """The frozen base LM and encoder, and the projector between them (trainable)."""

    def __init__(self, lm: nn.Module, encoder: nn.Module, projector: Projector, tokenizer: Any,
                 pcfg: ProjectorConfig):
        super().__init__()
        self.lm, self.encoder, self.projector, self.tokenizer, self.pcfg = lm, encoder, projector, tokenizer, pcfg
        self.slot_id = slot_token_id(tokenizer, pcfg.image_slot)
        for p in list(lm.parameters()) + list(encoder.parameters()):
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> CaptionAligner:
        super().train(mode)
        self.lm.eval()  # frozen: no dropout, as served
        self.encoder.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed caption cross-entropy and the number of tokens it covers."""
        lm: Any = self.lm
        embeds = lm.get_input_embeddings()(input_ids)
        images = encode_images(self.encoder, self.projector, pixel_values)
        embeds = place_images(embeds, input_ids, self.slot_id, images)
        hidden = lm.model(inputs_embeds=embeds, attention_mask=attention_mask, return_dict=True).last_hidden_state
        target = labels[:, 1:]
        keep = target != -100
        # the output head on the labelled positions only: the vocabulary is 130k wide
        logits = lm.lm_head(hidden[:, :-1][keep]).float()
        return F.cross_entropy(logits, target[keep], reduction="sum"), keep.sum()


def build_aligner(cfg: AlignConfig) -> CaptionAligner:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = {"revision": cfg.base_revision} if cfg.base_revision else {}
    tok = AutoTokenizer.from_pretrained(cfg.base_model, **rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"dtype": getattr(torch, cfg.torch_dtype), **rev}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    import transformers

    vcfg = transformers.SiglipVisionConfig.from_pretrained(
        cfg.encoder, **({"revision": cfg.encoder_revision} if cfg.encoder_revision else {}))
    pcfg = ProjectorConfig(encoder=cfg.encoder, encoder_revision=cfg.encoder_revision,
                           encoder_dtype=cfg.torch_dtype, vision_hidden=vcfg.hidden_size,
                           text_hidden=lm.config.hidden_size, image_size=vcfg.image_size,
                           patch_size=vcfg.patch_size, unshuffle=cfg.unshuffle, image_slot=cfg.image_slot)
    torch.manual_seed(cfg.seed)
    aligner = CaptionAligner(lm, load_encoder(pcfg), Projector(pcfg).to(torch.float32), tok, pcfg)
    if cfg.gradient_checkpointing:
        lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return aligner


# ---- training ------------------------------------------------------------------------


def _loader(aligner: CaptionAligner, rows: list[Row], cfg: AlignConfig, seed: int, shuffle: bool) -> DataLoader:
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    batches = [idx[i : i + cfg.micro_batch] for i in range(0, len(idx), cfg.micro_batch)]
    prompter = SiglipImages(load_image_processor(aligner.pcfg), aligner.pcfg)
    return DataLoader(Batches(rows, batches, CaptionCollator(aligner.tokenizer, prompter, cfg)),
                      batch_size=None, num_workers=cfg.workers, prefetch_factor=4 if cfg.workers else None)


def caption_loss(aligner: CaptionAligner, loader: DataLoader, device: str) -> float:
    """Mean caption cross-entropy per token over `loader`."""
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, count = aligner(**batch)
            total, n = total + float(loss), n + int(count)
    return total / max(1, n)


def fit(aligner: CaptionAligner, train_rows: list[Row], val_rows: list[Row], cfg: AlignConfig,
        device: str) -> list[dict[str, Any]]:
    """The training loop: AdamW on the projector, cosine schedule, token-mean loss per step."""
    aligner.to(device).train()
    params = list(aligner.projector.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    accum = max(1, cfg.batch_size // cfg.micro_batch)
    per_epoch = -(-len(train_rows) // cfg.micro_batch)
    if not train_rows:
        raise ValueError("no training rows")
    # max_steps, when set, wins over epochs (the loop passes over the rows again as needed)
    total_steps = cfg.max_steps or max(1, cfg.epochs * per_epoch // accum)
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: _lr_lambda(s, warmup, total_steps))
    val = _loader(aligner, val_rows, cfg, 0, shuffle=False) if val_rows else None
    print(f"[align] {len(train_rows)} rows, {total_steps} steps of {accum} x {cfg.micro_batch}", flush=True)
    history: list[dict[str, Any]] = []
    step, micro, sum_loss, sum_tok, t0 = 0, 0, 0.0, 0, time.time()
    epoch = 0
    while step < total_steps:
        for batch in _loader(aligner, train_rows, cfg, cfg.seed * 1000 + epoch, shuffle=True):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, count = aligner(**batch)
            # summed per micro-batch; the step divides by its token count below
            loss.backward()
            sum_loss, sum_tok, micro = sum_loss + float(loss.detach()), sum_tok + int(count), micro + 1
            if micro < accum:
                continue
            for p in params:
                if p.grad is not None:
                    p.grad.div_(max(1, sum_tok))
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step, micro = step + 1, 0
            entry = {"step": step, "loss": sum_loss / max(1, sum_tok), "lr": sched.get_last_lr()[0],
                     "elapsed_s": round(time.time() - t0)}
            sum_loss, sum_tok = 0.0, 0
            if step % cfg.log_every == 0 or step == total_steps:
                history.append(entry)
                print(f"[align] {json.dumps(entry)} / {total_steps}", flush=True)
            if val is not None and cfg.eval_every and step % cfg.eval_every == 0:
                history.append({"step": step, "val_loss": caption_loss(aligner, val, device)})
                print(f"[align] eval {history[-1]}", flush=True)
                aligner.train()
            if step >= total_steps:
                break
        epoch += 1
    if val is not None:
        history.append({"step": step, "val_loss": caption_loss(aligner, val, device), "final": True})
        print(f"[align] final {history[-1]} in {time.time() - t0:.0f} s", flush=True)
    return history


def train(cfg: AlignConfig) -> str:
    """Train the projector and save it (with its config and history) to `cfg.output_dir`."""
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(cfg)
    random.Random(cfg.seed).shuffle(rows)
    val_rows, train_rows = rows[: cfg.val_rows], rows[cfg.val_rows :]
    aligner = build_aligner(cfg)
    history = fit(aligner, train_rows, val_rows, cfg, device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_projector(aligner.projector, aligner.pcfg, cfg.output_dir)
    manifest = {"n_train": len(train_rows), "n_val": len(val_rows), "caption_files": cfg.caption_files,
                "versions": library_versions()}
    for name, obj in (("history.json", history), ("train_config.json", asdict(cfg)),
                      ("data_manifest.json", manifest)):
        with open(os.path.join(cfg.output_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
    print(f"[align] saved to {cfg.output_dir}", flush=True)
    return cfg.output_dir


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit("usage: python -m strands_decider.graft_align CONFIG.yaml [key=value ...]")
    train(AlignConfig.from_yaml(args[0]).with_overrides(args[1:]))


if __name__ == "__main__":
    main()
