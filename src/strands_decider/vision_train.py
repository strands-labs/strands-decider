"""Continuing a Strands Decider checkpoint on images: LoRA and pointer head trainable,
the vision tower frozen.

Starts from a text checkpoint loaded with `VisionDeciderModel.load` (its adapter mapped
onto the multimodal torso), so step 0 is that checkpoint served with `--vision`. The
rows are the data/image/ builders' output (an `Example` plus `images` and provenance),
plus a sample of the checkpoint's own text training rows to protect text behaviour.

The loss per row, summed over a step's rows and divided by `rows_per_step`:

    CE(gold)                                    labelled rows (score rows ordinal-smoothed)
    + kl_frozen_weight * KL(frozen || student)  labelled rows: the untouched torso's
                                                option-number reading, image in the prompt
    + kl_only_weight * KL(frozen || student)    image-removed copies (`ablation`): weight 0,
                                                no label, the frozen reading of the
                                                text-only prompt as the only target

Both KL terms are skipped for the kinds in `kl_frozen_skip_kinds`, the copies included:
with `["noul"]`, only the choice and score copies train toward the text-only reading,
and the yes/no copies carry no loss at all.

The image-removed copies teach the model what v19 does not know on its own: with the
image gone, the question has no answer in the prompt, so it should not answer as if it
had seen one. Training forwards at temperature 1; per-kind image temperatures are fitted
afterwards on held-out items (evaluation/vision/temps.py). Rows are rendered as the
server renders them (one collator draw per row: option order and phrasing), and images
are decoded (`read_image`) and resized (`fit_image`) as the server does, then processed
by the PIL image processor the engine pins.

`key=value` arguments after the config override its fields (values read as YAML):

    python -m strands_decider.vision_train configs/vision/v19-images.yaml
    python -m strands_decider.vision_train configs/vision/v19-images.yaml \
        init_from=checkpoints/my-text-checkpoint max_steps=100

Single process, one GPU: about 46 minutes for the recorded 1,404 steps on one H100.

With `projector_from` set, the checkpoint is a text-only torso (MiniCPM5) given grafted
eyes (graft.py): the stage-1 projector in that directory is loaded beside the text
checkpoint, trains at `projector_lr`, and is saved with the adapter and head; the SigLIP
encoder stays frozen. Everything else (rows, copies, losses) is the same recipe.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset

from .data.collate import CollatorConfig, SystemOneCollator
from .data.format import Example
from .prompting import render_question
from .runinfo import library_versions
from .train import YamlConfig, _build_optimizer, _lr_lambda
from .vision import (
    ImagePrompt,
    VisionDeciderModel,
    fit_image,
    load_vision_model,
    mm_base,
    read_image,
)

# The image tensors a collated batch may carry (`ImagePrompt.process`).
MM_KEYS = ("pixel_values", "image_grid_thw")

Row = dict[str, Any]
MAX_KL_OPTIONS = 9  # the frozen reading scores the single-token option numbers 1-9


@dataclass
class VisionTrainConfig(YamlConfig):
    # the checkpoint to continue: a local directory, or a Hub repo at `init_revision`
    init_from: str = "StrandsAgents/strands-decider-2B-hobson-v19"
    init_revision: str | None = "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
    # A stage-1 projector directory (graft_align.py): grafts its encoder and projector onto
    # `init_from`, a text checkpoint on a text-only torso. None: Qwen3.5's own tower.
    projector_from: str | None = None
    projector_lr: float = 2e-5

    # data: the builders' JSONL files under `data_root`, image paths relative to it
    data_root: str = "data/image/build"
    train_files: list[str] = field(default_factory=list)
    # training image paths to leave out (data/checks/dedupe_images.py writes the list)
    dedupe_drop: str | None = None
    # the checkpoint's own text training rows, `text_replay_n` of them sampled
    text_replay_files: list[str] = field(default_factory=list)
    text_replay_n: int = 3000
    # Share of image rows with at most 9 options that also get an image-removed copy.
    ablation_fraction: float = 0.0
    kl_frozen_weight: float = 0.3
    kl_only_weight: float = 1.0
    # Row kinds ("noul", "choice", "score") the frozen-KL term is not applied to, as
    # TrainConfig.kl_frozen_skip_kinds: the frozen torso reads yes/no near chance. This also
    # drops the KL-only target of those kinds' image-removed copies, which then carry no loss.
    kl_frozen_skip_kinds: list[str] = field(default_factory=list)
    # Replay rows that carry a `teacher` distribution (canonical option order) train toward
    # (one-hot gold + teacher_weight * teacher) / (1 + teacher_weight).
    teacher_weight: float = 1.0
    val_fraction: float = 0.03

    # images, resized as `VisionEngineConfig` resizes them at inference
    image_long_side: int = 448
    image_max_pixels: int = 0
    est_image_tokens: int = 200  # for length-grouped batching only
    max_length: int = 4096
    shuffle_options: bool = True
    ordinal_smoothing: float = 0.1

    # optimisation
    epochs: int = 1
    max_steps: int = 0  # 0 = full epochs
    lr: float = 5e-5
    head_lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    micro_rows: int = 16  # a micro-batch holds at most this many rows ...
    micro_tokens: int = 12000  # ... and this many padded tokens
    rows_per_step: int = 32
    gradient_checkpointing: bool = True
    workers: int = 8

    # bookkeeping
    output_dir: str = "checkpoints/v19-images"
    seed: int = 0
    log_every: int = 20
    eval_every: int = 400
    val_batches: int = 60


# ---- data --------------------------------------------------------------------------


def _jsonl(path: str) -> list[Row]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_rows(cfg: VisionTrainConfig) -> list[Row]:
    """The image rows, their image-removed copies, then the sampled text replay rows."""
    rng = random.Random(cfg.seed)
    rows = [r for f in cfg.train_files for r in _jsonl(os.path.join(cfg.data_root, f))]
    if cfg.dedupe_drop:
        with open(cfg.dedupe_drop, encoding="utf-8") as fh:
            drop = {line.strip() for line in fh if line.strip()}
        rows = [r for r in rows if not drop.intersection(r.get("images", []))]
    text = []
    for f in cfg.text_replay_files:
        for r in _jsonl(f):
            key = hashlib.md5(json.dumps(r["state"])[:2000].encode()).hexdigest()[:12]
            text.append({**r, "images": [], "ablation": False, "source": "text_replay",
                         "source_id": f"text-{key}"})
    rng.shuffle(text)
    text = text[: cfg.text_replay_n]
    ablation = []
    if cfg.ablation_fraction > 0:
        for r in rows:
            if len(r["options"]) <= MAX_KL_OPTIONS and rng.random() < cfg.ablation_fraction:
                ablation.append({**r, "images": [], "ablation": True, "weight": 0.0, "pair_id": None,
                                 "task": r["task"] + "/ablation"})
    return rows + ablation + text


def split(rows: list[Row], frac: float, seed: int) -> tuple[list[Row], list[Row]]:
    """Train and validation rows, drawn by minimal pair (else by source). Every row of a
    source drawn for validation follows it, so no image (or text row) is on both sides,
    and no image-removed copy is validated: it has no label."""

    def key(r: Row) -> str:
        return str(r.get("pair_id") or r["source_id"])

    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    val_keys = set(keys[: int(len(keys) * frac)])
    val_sources = {r["source_id"] for k in val_keys for r in groups[k]}  # copies have no pair_id
    train: list[Row] = []
    val: list[Row] = []
    for r in rows:
        (val if key(r) in val_keys or r["source_id"] in val_sources else train).append(r)
    return train, [r for r in val if not r.get("ablation")]


def est_length(r: Row, image_tokens: int) -> int:
    """A row's token count, roughly: for grouping rows of similar length."""
    text = json.dumps(r["state"]) + r["instructions"] + json.dumps(r["options"])
    return len(text) // 3 + 60 + image_tokens * len(r.get("images", []))


def plan_batches(lengths: list[int], max_rows: int, max_tokens: int, seed: int) -> list[list[int]]:
    """Length-grouped micro-batches under a padded-token budget, in shuffled order."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    out = []
    for m in range(0, len(idx), 2000):  # sorted within chunks of 2,000: grouped, still mixed
        cur: list[int] = []
        for i in sorted(idx[m : m + 2000], key=lambda j: lengths[j]):
            longest = max([lengths[j] for j in cur] + [lengths[i]])
            if cur and (len(cur) >= max_rows or longest * (len(cur) + 1) > max_tokens):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


class ImageCollator:
    """Rows -> one micro-batch: the image(s) inside <state>, then the question, rendered
    and labelled as SystemOneCollator renders a text row."""

    def __init__(self, tokenizer: Any, prompter: ImagePrompt, cfg: VisionTrainConfig, *, train: bool):
        self.tok, self.prompter, self.cfg, self.train = tokenizer, prompter, cfg, train
        self.ccfg = CollatorConfig(max_length=cfg.max_length, num_slots=24, head_type="pointer",
                                   shuffle_options=cfg.shuffle_options,
                                   ordinal_smoothing=cfg.ordinal_smoothing, seed=0)

    def image(self, path: str) -> Any:
        with open(os.path.join(self.cfg.data_root, path), "rb") as fh:
            return fit_image(read_image(fh.read()), self.cfg.image_long_side, self.cfg.image_max_pixels)

    def teacher_target(self, teacher: list[float], label: int, order: Any) -> torch.Tensor:
        """(one-hot gold + w * teacher) / (1 + w), mapped onto the slots of this rendering."""
        slots = list(order) if order is not None else list(range(len(teacher)))
        dist = torch.zeros(24, dtype=torch.float32)
        w = self.cfg.teacher_weight
        for s, canonical in enumerate(slots):
            dist[s] = w * float(teacher[canonical])
        dist[label] += 1.0
        return dist / dist.sum()

    def __call__(self, rows: list[Row], index: int = 0) -> dict[str, torch.Tensor]:
        # Each micro-batch draws from its own stream, so workers render it the same way.
        base = SystemOneCollator(self.tok, self.ccfg, train=self.train)
        base.rng = random.Random(self.cfg.seed * 1_000_003 + index * 7)
        images = [self.image(p) for r in rows for p in r.get("images", [])]
        counts: list[int] = []
        out: dict[str, torch.Tensor] = {}
        if images:
            counts, out = self.prompter.process(images)
        ids, opts, labels, targets = [], [], [], []
        c = 0
        for r in rows:
            ex, n_img = Example.from_dict(r), len(r.get("images", []))
            order, question, label, target = base.draw(ex)
            if self.train and r.get("teacher"):
                target = self.teacher_target(r["teacher"], label, order)
            rq = render_question(question, option_order=order)
            prompt = self.prompter.state(ex.state, counts[c : c + n_img]) + rq.text
            c += n_img
            enc = self.tok(prompt, return_offsets_mapping=True)
            if len(enc["input_ids"]) > self.cfg.max_length:
                raise ValueError(f"a {r['task']} row is {len(enc['input_ids'])} tokens, over max_length")
            ids.append(enc["input_ids"])
            opts.append(base.option_token_index(enc["offset_mapping"], rq.option_spans, len(prompt) - len(rq.text)))
            labels.append(label)
            targets.append(target)
        pad, width, longest = self.tok.pad_token_id, max(map(len, opts)), max(map(len, ids))
        out.update({
            "input_ids": torch.tensor([x + [pad] * (longest - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (longest - len(x)) for x in ids]),
            "opt_idx": torch.tensor([o + [-1] * (width - len(o)) for o in opts]),
            "labels": torch.tensor(labels),
            "n_slots": torch.tensor([len(r["options"]) for r in rows]),
            "weights": torch.tensor([0.0 if r.get("ablation") else 1.0 for r in rows]),
            "ablation": torch.tensor([bool(r.get("ablation")) for r in rows]),
            "kl_skip": torch.tensor([r.get("kind") in self.cfg.kl_frozen_skip_kinds for r in rows]),
        })
        if any(t is not None for t in targets):
            out["label_dist"] = torch.stack([
                t if t is not None else torch.nn.functional.one_hot(torch.tensor(lab), 24).float()
                for t, lab in zip(targets, labels, strict=True)])[:, :width]
        return out


class MicroBatches(Dataset):
    """Item i is micro-batch i, collated (in a loader worker) with its own seed."""

    def __init__(self, rows: list[Row], batches: list[list[int]], collate: ImageCollator):
        self.rows, self.batches, self.collate = rows, batches, collate

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.collate([self.rows[j] for j in self.batches[i]], index=i)


# ---- training ------------------------------------------------------------------------


def row_losses(model: VisionDeciderModel, batch: dict[str, torch.Tensor], kl_weight: float,
               kl_only_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row label loss and frozen-KL term (see the module docstring), and log probs."""
    mm = {k: batch[k] for k in MM_KEYS if k in batch}
    out = model(batch["input_ids"], batch["attention_mask"], batch["n_slots"], opt_idx=batch["opt_idx"],
                temperature=1.0, **mm)
    lp = out["log_probs"]
    safe = torch.where(torch.isinf(lp), torch.zeros_like(lp), lp)
    if "label_dist" in batch:
        ce = -(batch["label_dist"] * safe).sum(-1)
    else:
        ce = -safe.gather(1, batch["labels"].view(-1, 1)).squeeze(1)
    ce = ce * batch["weights"]
    kl = torch.zeros_like(ce)
    if kl_weight > 0 or kl_only_weight > 0:
        model.reset_positions()  # positions of this batch, not the last
        ref, eligible = model.frozen_slot_log_probs(batch["input_ids"], batch["attention_mask"],
                                                    batch["n_slots"], **mm)
        if ref.numel():
            ref = ref[:, : lp.shape[-1]]
            valid = torch.isfinite(ref) & torch.isfinite(lp)
            p = ref.exp().masked_fill(~valid, 0.0)
            per_row = (p * (ref - lp).masked_fill(~valid, 0.0)).sum(-1) * eligible.float()
            kl = torch.where(batch["ablation"], kl_only_weight, kl_weight) * per_row
            if "kl_skip" in batch:
                kl = kl * (~batch["kl_skip"]).to(kl.dtype)
    model.reset_positions()
    return ce, kl, lp


def evaluate(model: VisionDeciderModel, loader: DataLoader, device: str) -> dict[str, float]:
    """Accuracy and label loss on the validation rows."""
    model.eval()
    n, correct, nll = 0, 0.0, 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            ce, _, lp = row_losses(model, batch, 0.0, 0.0)
        n += len(ce)
        correct += float((lp.argmax(-1) == batch["labels"]).sum())
        nll += float(ce.sum())
    model.train()
    return {"val_acc": correct / max(1, n), "val_nll": nll / max(1, n), "val_n": n}


def load_model(cfg: VisionTrainConfig) -> VisionDeciderModel:
    """The checkpoint to continue, its adapter and head (and a grafted projector)
    trainable, its vision tower or encoder frozen."""
    path = cfg.init_from
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(path, revision=cfg.init_revision)
    model: VisionDeciderModel
    if cfg.projector_from:
        from .graft import GraftedDeciderModel

        model = cast(VisionDeciderModel, GraftedDeciderModel.load(
            path, projector_from=cfg.projector_from, trainable=True))
    else:
        model = load_vision_model(path, trainable=True)
    # Fitted on the checkpoint's own answers to images; stale once it trains on them.
    model.config.image_temperature_by_kind = {}
    if cfg.gradient_checkpointing:
        mm_base(model.torso).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def train(cfg: VisionTrainConfig) -> str:
    """Train, save the checkpoint to `cfg.output_dir` with its history, and return the path."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)
    model = load_model(cfg).to(device)
    model.train()
    trainable, total = model.trainable_parameters()
    print(f"[vision-train] trainable {trainable:,} / {total:,}", flush=True)

    train_rows, val_rows = split(load_rows(cfg), cfg.val_fraction, cfg.seed)
    by_source: dict[str, int] = {}
    for r in train_rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    manifest = {"n_train": len(train_rows), "n_val": len(val_rows), "by_source_train": by_source,
                "ablation_train": sum(bool(r.get("ablation")) for r in train_rows),
                "base_model": model.config.base_model, "base_revision": model.config.base_revision,
                "versions": library_versions()}
    print(f"[vision-train] {json.dumps(manifest)}", flush=True)

    proc = model.image_prompt()  # as VisionEngine pins it
    lengths = [est_length(r, cfg.est_image_tokens) for r in train_rows]
    batches = [b for e in range(cfg.epochs)
               for b in plan_batches(lengths, cfg.micro_rows, cfg.micro_tokens, cfg.seed * 100 + e)]
    total_steps = max(1, sum(map(len, batches)) // cfg.rows_per_step)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    loader = DataLoader(MicroBatches(train_rows, batches, ImageCollator(model.tokenizer, proc, cfg, train=True)),
                        batch_size=None, num_workers=cfg.workers, prefetch_factor=4 if cfg.workers else None)
    val_lengths = [est_length(r, cfg.est_image_tokens) for r in val_rows]
    val_batches = plan_batches(val_lengths, cfg.micro_rows, cfg.micro_tokens, 1)[: cfg.val_batches]
    val_loader = DataLoader(MicroBatches(val_rows, val_batches, ImageCollator(model.tokenizer, proc, cfg, train=False)),
                            batch_size=None, num_workers=min(cfg.workers, 4))

    optim = _build_optimizer(model, lr=cfg.lr, head_lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    params = list(model.head.parameters()) + [p for p in model.torso.parameters() if p.requires_grad]
    projector = getattr(model, "projector", None)
    if projector is not None:  # a grafted model's projector: its own learning rate
        optim.add_param_group({"params": list(projector.parameters()), "lr": cfg.projector_lr,
                               "weight_decay": 0.0})
        params += list(projector.parameters())
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: _lr_lambda(s, warmup, total_steps))
    print(f"[vision-train] {len(batches)} micro-batches, {total_steps} steps", flush=True)

    history: list[dict[str, Any]] = []
    step, rows_in_step, log_rows, sum_ce, sum_kl, t0 = 0, 0, 0, 0.0, 0.0, time.time()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            ce, kl, _ = row_losses(model, batch, cfg.kl_frozen_weight, cfg.kl_only_weight)
        # Summed now, divided by the step's row count: every row weighs the same.
        (ce.sum() + kl.sum()).div(cfg.rows_per_step).backward()
        sum_ce, sum_kl = sum_ce + float(ce.detach().sum()), sum_kl + float(kl.detach().sum())
        rows_in_step += len(ce)
        log_rows += len(ce)
        if rows_in_step < cfg.rows_per_step:
            continue
        if rows_in_step != cfg.rows_per_step:  # micro-batches overshot: rescale to the true count
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(cfg.rows_per_step / rows_in_step)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        step, rows_in_step = step + 1, 0
        if step % cfg.log_every == 0:
            entry = {"step": step, "ce": sum_ce / log_rows, "kl": sum_kl / log_rows,
                     "lr": sched.get_last_lr()[-1], "elapsed_s": round(time.time() - t0)}
            history.append(entry)
            print(f"[vision-train] {json.dumps(entry)} / {total_steps}", flush=True)
            sum_ce, sum_kl, log_rows = 0.0, 0.0, 0
        if cfg.eval_every and step % cfg.eval_every == 0:
            history.append({"step": step, **evaluate(model, val_loader, device)})
            print(f"[vision-train] eval {history[-1]}", flush=True)
        if step >= total_steps:
            break
    history.append({"step": step, **evaluate(model, val_loader, device), "final": True})
    print(f"[vision-train] final {history[-1]} in {time.time() - t0:.0f} s", flush=True)
    model.save_pretrained(cfg.output_dir)
    for name, obj in (("history.json", history), ("data_manifest.json", manifest), ("train_config.json", asdict(cfg))):
        with open(os.path.join(cfg.output_dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
    print(f"[vision-train] saved to {cfg.output_dir}", flush=True)
    return cfg.output_dir


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit("usage: python -m strands_decider.vision_train CONFIG.yaml [key=value ...]")
    train(VisionTrainConfig.from_yaml(args[0]).with_overrides(args[1:]))


if __name__ == "__main__":
    main()
