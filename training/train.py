"""Training the pointer head (see modeling.py) and a LoRA adapter on the torso.

The objective is deliberately plain -- cross-entropy over the masked slots, plus an
optional teacher term -- because the interesting work is in the *distribution* the
examples are drawn from, not the loss. Each step shows the model a different task, a
different number of options, and a different option ordering. What has to be learned
to fit that is precisely the behaviour we want at serving time.

The TrainConfig defaults are sized for a single 24 GB card: a Qwen3-1.7B-Base torso in
bf16, LoRA rank 16, gradient checkpointing on, micro-batch 8 at 3072 tokens. The
reference recipe (configs/train.yaml) trains the pointer head on a Qwen3.5-2B-Base torso
at 4096 tokens; see training/hardware.md, "The Qwen3.5 torso" and "Larger torsos".
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import MISSING, asdict, dataclass, field
from typing import Any

import torch
import typer
from torch.utils.data import DataLoader, Dataset

from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    checkpoint_dir,
    config_path,
)

from . import distributed
from .data.collate import CollatorConfig, SystemOneCollator
from .data.format import Example, load_examples, split_examples
from .data.sampling import LengthGroupedBatchSampler, example_length, padding_fraction


@dataclass
class TrainConfig:
    # data
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    val_fraction: float = 0.03
    # Seed of that train/validation split alone. None: `seed`, so every seed holds out (and
    # so never trains on) its own val_fraction of the rows. Fixed, every seed trains on the
    # same rows; initialisation, data order and option order still follow `seed`.
    val_split_seed: int | None = None

    # model
    base_model: str = "Qwen/Qwen3-1.7B-Base"
    head_dropout: float = 0.05
    max_length: int = 3072
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    # None keeps StrandsDeciderConfig's default (attention and MLP projections). A hybrid torso
    # such as Qwen3.5 needs its recurrent layers' projections listed too, or LoRA reaches
    # only its few full-attention layers.
    lora_targets: list[str] | None = None
    # Start every prompt with BOS even if the tokenizer adds none (modeling.ensure_bos);
    # gemma-4-E2B-it's tokenizer leaves it to the chat template.
    force_bos: bool = False
    # Keep a per-layer embedding table (Gemma 4 E2B's, 4.4 GiB) in host memory
    # (modeling.HostEmbedding). For one GPU: under torchrun every rank would hold a copy.
    host_embeddings: bool = False
    freeze_torso: bool = False
    pointer_dim: int = 256
    # A frozen teacher's option distribution per training row (data/teacher.py): JSONL of
    # {"i": row index in the concatenated train_files, "probs": [...canonical order]}.
    # Rows it covers get teacher_weight * KL(teacher || student) on top of the label loss.
    teacher_file: str | None = None
    teacher_weight: float = 0.0
    # > 0 adds KL(reference || student) so training is anchored to a trained reference's
    # own distribution. The reference is a trained checkpoint (`kl_frozen_reference`), not
    # the base torso's option-number readout: the pointer head scores each option from its
    # own hidden state, with no fixed slot grid for the LM's digit rows to project onto.
    kl_frozen_weight: float = 0.0
    # Trained checkpoint whose option distribution (temperature 1, no calibration) is the
    # frozen-KL reference. It may use another tokenizer than the student (`_frozen_reference`
    # re-renders every row with the reference's tokenizer, keeping the collator's option
    # permutation). Required when `kl_frozen_weight > 0`.
    kl_frozen_reference: str | None = None
    # Rows trained ONLY by the KL term: no label loss, never in the validation split.
    # For prompts with no gold answer -- e.g. changed questions, where the reference's
    # own reading is the target (data/question_transforms.py).
    kl_only_files: list[str] = field(default_factory=list)
    # A KL-only row's KL weight. The label loss is weighted 1 per row, so 1.0 gives a
    # KL-only row the same total weight as a labelled one.
    kl_only_weight: float = 1.0
    # Speed only: compute the frozen-KL reference for every training row in one no-grad
    # pass before step 1 (`_frozen_reference`). Required when `kl_frozen_weight > 0` and
    # `group_by_length` must be on so the batch order is known. Same numbers up to bf16
    # batch-shape rounding as a per-step forward.
    precompute_frozen_kl: bool = False
    # Direction of both KL terms, the teacher's and the frozen anchor's: "forward" =
    # KL(target || student) (mass-covering, every run so far); "reverse" = KL(student ||
    # target) (mode-seeking).
    kl_direction: str = "forward"
    # Continue training a trained checkpoint (a local directory or a Hub id, e.g. a released
    # model) on this config's data: its adapter and its head, pointer or slot, souped or not,
    # train on from their trained values (`_init_from`). The checkpoint fixes the architecture
    # (INIT_FROM_ARCH): leave those fields out, as a config that sets one to another value
    # than the checkpoint's is refused. max_length is the checkpoint's unless set. Fitted
    # temperatures are reset to 1: calibrate the result. freeze_torso: true trains the head
    # alone on the checkpoint's frozen adapter.
    init_from: str | None = None

    # optimisation
    epochs: int = 1
    micro_batch_size: int = 8
    grad_accum: int = 4
    lr: float = 1e-4
    # The head is randomly initialised while LoRA starts as a no-op, so the head
    # needs a much larger step or the torso adapts to a head that is still noise.
    # With init_from both start trained, and no value is measured for that yet.
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    # > 0 keeps an exponential moving average of the trained weights (adapter and head), in
    # fp32, updated after every optimizer step with decay min(ema_decay, (1 + step) /
    # (10 + step)); the average is what is validated at the end and saved. 0 = off: the last
    # step's weights, as before.
    ema_decay: float = 0.0
    gradient_checkpointing: bool = True

    # data augmentation
    shuffle_options: bool = True
    # Batch examples of similar length together (see data/sampling.py). Off by default
    # so that every saved train_config.json from before it existed still reproduces the
    # run it describes; the shipped configs turn it on.
    group_by_length: bool = False
    length_group_mega: int = 50
    ordinal_smoothing: float = 0.1
    reverse_score_prob: float = 0.5

    # bookkeeping
    output_dir: str = "checkpoints/hobson-1.7b"
    seed: int = 0
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 0  # 0 = only at the end
    max_steps: int = 0  # 0 = full epochs; otherwise a hard cap (smoke tests)
    attn_implementation: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # Model-architecture fields the slot head owned, which the v1-2610 and earlier YAMLs
        # still carry. Drop them silently so a historical recipe retrains with the current
        # pointer-only architecture; a typo in any other key still raises.
        known = set(cls.__dataclass_fields__)
        for k, v in raw.items():
            if k == "head_type" and v != "pointer":
                raise ValueError(
                    f"{path}: head_type={v!r}; this version trains pointer-head checkpoints only"
                )
            if k not in known and k not in RETIRED_YAML_KEYS:
                raise ValueError(f"{path}: unknown config key {k!r}")
        raw = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**raw)
        cfg.yaml_keys = frozenset(raw)  # type: ignore[attr-defined]  # what _given reads
        return cfg


# Slot-era TrainConfig keys the current dataclass no longer carries. Dropped silently by
# from_yaml so a published recipe still parses; a typo in any other key raises.
RETIRED_YAML_KEYS = frozenset({"num_slots", "head_type", "head_hidden", "head_init"})


# The fields of a TrainConfig that a checkpoint fixes: with init_from they are the checkpoint's.
INIT_FROM_ARCH = (
    "base_model",
    "pointer_dim",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_targets",
    "force_bos",
)


def _given(cfg: TrainConfig, name: str) -> bool:
    """True if `cfg` sets field `name`: its YAML file has the key, or its value is not the
    default."""
    f = TrainConfig.__dataclass_fields__[name]
    default = f.default_factory() if f.default_factory is not MISSING else f.default
    return name in getattr(cfg, "yaml_keys", ()) or getattr(cfg, name) != default


def _init_from(cfg: TrainConfig) -> StrandsDeciderModel:
    """The checkpoint `cfg.init_from`, loaded to train on (TrainConfig.init_from).

    The adapter and the head keep their trained values and train; the architecture fields of
    `cfg` are set to the checkpoint's, so the collator and train_config.json describe the
    model that trains. A soup loads as any checkpoint: a rank-N*r adapter and a head with
    pointer_dim N*d whose LayerNorm has gain 1 and bias 0 (soup.py). Both train as any other.
    """
    path = checkpoint_dir(cfg.init_from or "")
    ck = StrandsDeciderConfig.from_json(config_path(path))
    conflicts = []
    for name in INIT_FROM_ARCH:
        mine, theirs = getattr(cfg, name), getattr(ck, name)
        if name == "lora_targets":  # an order-free list; None is the model's default list
            mine, theirs = sorted(mine or StrandsDeciderConfig().lora_targets), sorted(theirs)
        if _given(cfg, name) and mine != theirs:
            conflicts.append((name, mine, theirs))
        setattr(cfg, name, getattr(ck, name))
    if conflicts:
        sets = "; ".join(
            f"the config sets {n}={m!r}, the checkpoint has {t!r}" for n, m, t in conflicts
        )
        raise ValueError(
            f"init_from {cfg.init_from}: {sets}. The checkpoint fixes the "
            f"architecture: remove {', '.join(n for n, _, _ in conflicts)} "
            "from the config."
        )
    if not ck.use_lora and not cfg.freeze_torso:
        raise ValueError(
            f"init_from {cfg.init_from}: the checkpoint has no LoRA adapter to train "
            "on; set freeze_torso: true to train its head alone"
        )
    if not _given(cfg, "max_length"):
        cfg.max_length = ck.max_length
    print(f"[strands-decider] continuing from {cfg.init_from}")
    model = StrandsDeciderModel.load(
        path,
        attn_implementation=cfg.attn_implementation,
        trainable=True,
        host_embeddings=cfg.host_embeddings,
    )
    c = model.config
    # The fitted temperatures describe the checkpoint, not what trains from it: forward()
    # would apply them during training. Reset; calibrate after training.
    c.temperature, c.temperature_by_kind = 1.0, {}
    # This run's settings, saved with the result as a run from the base model saves them.
    c.max_length, c.ordinal_smoothing = cfg.max_length, cfg.ordinal_smoothing
    c.head_dropout = cfg.head_dropout
    model.head.dropout = (
        torch.nn.Dropout(cfg.head_dropout) if cfg.head_dropout > 0 else torch.nn.Identity()
    )
    if cfg.freeze_torso:
        model.freeze_torso()
    return model


def _random_batches(n: int, cfg: TrainConfig) -> list[list[int]]:
    """What shuffle=True would have produced, for reporting the padding saved."""
    order = list(range(n))
    random.Random(cfg.seed).shuffle(order)
    b = cfg.micro_batch_size
    return [order[i : i + b] for i in range(0, n - n % b, b)]


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def _build_optimizer(model: StrandsDeciderModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Two parameter groups: a fast head and a slow adapter.

    Also excludes norms and biases from weight decay, which otherwise shrinks the
    LayerNorm gain and quietly degrades calibration.
    """
    head_decay: list[torch.nn.Parameter] = []
    head_no_decay: list[torch.nn.Parameter] = []
    torso_params: list[torch.nn.Parameter] = []
    for _name, p in model.head.named_parameters():
        if not p.requires_grad:
            continue
        (head_no_decay if p.ndim <= 1 else head_decay).append(p)
    for _, p in model.torso.named_parameters():
        if p.requires_grad:
            torso_params.append(p)

    groups: list[dict[str, Any]] = [
        {"params": head_decay, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"params": head_no_decay, "lr": cfg.head_lr, "weight_decay": 0.0},
    ]
    if torso_params:
        groups.append({"params": torso_params, "lr": cfg.lr, "weight_decay": 0.0})
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


class _Ema:
    """An fp32 exponential moving average of `model`'s trainable parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.avg = [p.detach().float().clone() for p in self.params]
        self.live: list[torch.Tensor] = []

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def update(self, step: int) -> None:
        # The warm-up keeps the first steps' weights, far from the end ones, out of the average.
        d = min(self.decay, (1 + step) / (10 + step))
        for a, p in zip(self.avg, self.params, strict=True):
            a.mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def apply(self) -> None:
        """Put the average in the model, keeping the live weights for `restore`."""
        self.live = [p.detach().clone() for p in self.params]
        for a, p in zip(self.avg, self.params, strict=True):
            p.copy_(a)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def restore(self) -> None:
        for w, p in zip(self.live, self.params, strict=True):
            p.copy_(w)
        self.live = []


def _lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()  # type: ignore[untyped-decorator]
def evaluate_loss(
    model: StrandsDeciderModel, loader: DataLoader, device: str, max_batches: int = 50
) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_n, batches = 0.0, 0, 0, 0
    for batch in loader:
        if batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            n_slots=batch["n_slots"],
            opt_idx=batch.get("opt_idx"),
            labels=batch["labels"],
            label_dist=batch.get("label_dist"),
            weights=batch.get("weights"),
        )
        total_loss += float(out["loss"])
        pred = out["log_probs"].argmax(dim=-1)
        total_correct += int((pred == batch["labels"]).sum())
        total_n += batch["labels"].numel()
        batches += 1
    model.train()
    return {
        "val_loss": total_loss / max(1, batches),
        "val_acc": total_correct / max(1, total_n),
    }


def _frozen_reference(
    model: StrandsDeciderModel,
    examples: list[Example],
    batches: list[list[int]],
    coll_cfg: CollatorConfig,
    device: str,
    ref_model: StrandsDeciderModel,
) -> torch.Tensor:
    """`ref_model`'s log-probs for every row the run trains on: ref[micro-batch, row, slot].

    A fresh collator with the training collator's seed walks the run's micro-batches in
    training order, so each row is rendered exactly as training will render it. Each
    rendered micro-batch goes through `ref_model` as it is, at temperature 1; columns past
    a row's option count are -inf. The reference renders with its own tokenizer, so it may
    have another than the student's (a Gemma student, a Qwen reference): a fresh collator
    with the reference's tokenizer draws the same option permutation from its seed before
    any tokenisation, so both see each row with the same options in the same slots.

    Under torchrun rank m % world renders micro-batch m (the others only draw its
    augmentations) and a sum over ranks assembles the table.
    """
    rank, _, world = distributed.env()
    coll = SystemOneCollator(ref_model.tokenizer, coll_cfg, train=True)
    width = max(max(examples[i].n_options for i in b) for b in batches)
    ref = torch.zeros(len(batches), max(map(len, batches)), width, device=device)
    for m, idx in enumerate(batches):
        rows = [examples[i] for i in idx]
        if m % world != rank:
            coll.skip(rows)
            continue
        b = coll(rows)
        with torch.no_grad():
            lp = ref_model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                n_slots=b["n_slots"].to(device),
                opt_idx=b["opt_idx"].to(device),
                temperature=1.0,
            )["log_probs"]
        # -inf past each row's option count and past the batch's max K (padding slots
        # the ref_model would not score). all_reduce adds zeros from the other ranks,
        # which leaves both -inf and real values unchanged.
        w = min(lp.shape[-1], ref.shape[-1])
        ref[m, : len(rows)] = float("-inf")
        ref[m, : len(rows), :w] = lp[:, :w]
    if world > 1:
        torch.distributed.all_reduce(ref)
    return ref


def _kl(
    target: torch.Tensor,
    student: torch.Tensor,
    valid: torch.Tensor,
    direction: str,
    target_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row KL between two log-distributions over the `valid` entries: forward =
    KL(target || student), reverse = KL(student || target). `target_probs`, when the caller
    has them, weight the forward sum in place of exp(target)."""
    p, q = (target, student) if direction == "forward" else (student, target)
    diff = (p - q).masked_fill(~valid, 0.0)
    w = target_probs if direction == "forward" and target_probs is not None else p.exp()
    return (w.masked_fill(~valid, 0.0) * diff).sum(dim=-1)


# The teacher writer rounds to six decimals, so a real option can carry a target of exactly 0.
REVERSE_KL_TARGET_FLOOR = 1e-6


def _teacher_kl(tea: torch.Tensor, stu: torch.Tensor, direction: str) -> torch.Tensor:
    """Per-row KL between a teacher's option probabilities `tea` (0 past a row's options) and
    the student's log-probabilities `stu` (-inf past them). Forward skips the teacher's zeros,
    which carry no weight in KL(teacher || student). Reverse must charge student mass on them,
    so it floors the target on every real option at REVERSE_KL_TARGET_FLOOR and renormalises."""
    real = torch.isfinite(stu)
    if direction == "forward":
        return _kl(tea.clamp_min(1e-12).log(), stu, (tea > 0) & real, direction, tea)
    floored = tea.clamp_min(REVERSE_KL_TARGET_FLOOR).masked_fill(~real, 0.0)
    target = floored / floored.sum(dim=-1, keepdim=True)
    return _kl(target.clamp_min(1e-12).log(), stu, real, direction)


@distributed.entry_point  # under torchrun, this process is one rank of the run
def train(cfg: TrainConfig) -> str:
    rank, _, world = distributed.env()
    if cfg.kl_direction not in ("forward", "reverse"):
        raise ValueError(f"kl_direction must be forward or reverse, not {cfg.kl_direction!r}")
    if cfg.kl_frozen_weight > 0 and not cfg.kl_frozen_reference:
        raise ValueError("kl_frozen_weight > 0 needs kl_frozen_reference (a trained checkpoint)")
    if cfg.kl_frozen_reference and not (cfg.precompute_frozen_kl and cfg.kl_frozen_weight > 0):
        raise ValueError("kl_frozen_reference needs kl_frozen_weight > 0 and precompute_frozen_kl")
    if cfg.host_embeddings and world > 1:  # DDP refuses a module with parameters on the CPU
        raise ValueError("host_embeddings is for one GPU; under torchrun keep the table on the GPU")
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.init_from:
        model = _init_from(cfg)
    else:
        print(f"[strands-decider] loading base model {cfg.base_model}")
        model_cfg = StrandsDeciderConfig(
            base_model=cfg.base_model,
            head_dropout=cfg.head_dropout,
            max_length=cfg.max_length,
            use_lora=cfg.use_lora and not cfg.freeze_torso,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            force_bos=cfg.force_bos,
            host_embeddings=cfg.host_embeddings,
            # Serving needs this to correct the variance floor smoothing imposes on
            # score confidence; without it a score could never reach high confidence.
            ordinal_smoothing=cfg.ordinal_smoothing,
            pointer_dim=cfg.pointer_dim,
        )
        if cfg.lora_targets:
            model_cfg.lora_targets = list(cfg.lora_targets)
        model = StrandsDeciderModel.from_pretrained_base(
            model_cfg, attn_implementation=cfg.attn_implementation
        )
        if cfg.freeze_torso:
            model.freeze_torso()
    # Checkpointing needs a grad-requiring input; a frozen torso has none, and it
    # buys nothing anyway since no backward pass traverses it.
    if cfg.gradient_checkpointing and not cfg.freeze_torso:
        base = getattr(model.torso, "base_model", model.torso)
        inner = getattr(base, "model", base)
        if hasattr(inner, "gradient_checkpointing_enable"):
            # use_reentrant=False is required for checkpointing to coexist with LoRA:
            # the reentrant path loses the grad graph for inputs that do not require grad.
            inner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    model.to(device)
    fwd = distributed.wrap(model, cfg.seed)

    trainable, total = model.trainable_parameters()
    print(f"[strands-decider] trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ---- data ----
    train_examples = load_examples(cfg.train_files)
    if cfg.teacher_file:
        # Attached before the split, which shuffles; the collator reads `ex.teacher`.
        n_t = 0
        with open(cfg.teacher_file, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                ex = train_examples[d["i"]]
                if len(d["probs"]) != ex.n_options:
                    raise ValueError(
                        f"teacher row {d['i']} has {len(d['probs'])} probs for "
                        f"{ex.n_options} options -- wrong train_files?"
                    )
                ex.teacher = d["probs"]
                n_t += 1
        print(
            f"[strands-decider] teacher distributions on {n_t:,} of {len(train_examples):,} rows "
            f"(weight {cfg.teacher_weight})"
        )
    if cfg.val_files:
        val_examples = load_examples(cfg.val_files)
    else:
        train_examples, val_examples = split_examples(
            train_examples,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed if cfg.val_split_seed is None else cfg.val_split_seed,
        )
    if cfg.kl_only_files:
        if cfg.kl_frozen_weight <= 0:
            raise ValueError("kl_only_files needs kl_frozen_weight > 0 to enable the KL term")
        if any(ex.weight <= 0 for ex in train_examples):
            raise ValueError("labelled rows must have weight > 0: weight 0 marks KL-only rows")
        kl_rows = load_examples(cfg.kl_only_files)
        for ex in kl_rows:
            ex.weight = 0.0  # no label loss; the training loop reads 0 as "KL only"
        # Added after the split, so none reaches validation, whose labels would be fake.
        train_examples = train_examples + kl_rows
        random.Random(cfg.seed).shuffle(train_examples)
        print(f"[strands-decider] +{len(kl_rows):,} KL-only rows (weight {cfg.kl_only_weight})")
    print(f"[strands-decider] train={len(train_examples):,} val={len(val_examples):,}")
    if not train_examples:
        raise SystemExit("no training examples; run `python -m training.data build` first")

    coll_cfg = CollatorConfig(
        max_length=cfg.max_length,
        shuffle_options=cfg.shuffle_options,
        ordinal_smoothing=cfg.ordinal_smoothing,
        reverse_score_prob=cfg.reverse_score_prob,
        seed=cfg.seed,
    )
    train_collate = SystemOneCollator(model.tokenizer, coll_cfg, train=True)
    if cfg.group_by_length:
        lengths = [example_length(ex) for ex in train_examples]
        sampler = LengthGroupedBatchSampler(
            lengths,
            cfg.micro_batch_size,
            mega=cfg.length_group_mega,
            seed=cfg.seed,
            drop_last=True,
        )
        # Measured in prompt characters, the proxy the sampler sorts by -- which flatters
        # the grouped figure, since grouping is exact in that unit. Tables and prose
        # tokenise at different rates, so real token padding is higher (measured ~16%
        # grouped against ~58% ungrouped on the v10 corpus). The ungrouped figure agrees
        # in both units, so the comparison is still a fair signal of what was saved.
        print(
            f"[strands-decider] length-grouped batching: padding "
            f"{padding_fraction(lengths, sampler.batches(0)):.1%} of positions grouped vs "
            f"{padding_fraction(lengths, _random_batches(len(lengths), cfg)):.1%} "
            f"ungrouped (in prompt characters; token padding runs higher)"
        )
        train_loader = DataLoader(
            ExampleDataset(train_examples),
            batch_sampler=sampler,
            collate_fn=train_collate,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            ExampleDataset(train_examples),
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            collate_fn=train_collate,
            drop_last=True,
            num_workers=0,  # tokenisation is fast here; worker processes would cost more than they save
        )
    val_loader = DataLoader(
        ExampleDataset(val_examples),
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        collate_fn=SystemOneCollator(model.tokenizer, coll_cfg, train=False),
    )

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    optim = _build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: _lr_lambda(s, warmup, total_steps))

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(cfg), fh, indent=2)
    if world > 1:  # each rank iterates its share of every step's rows instead
        # 0 for kl_slots: all rows are eligible under a pointer-head reference, so no
        # eligibility counter is needed in StepSlices.
        train_loader = distributed.StepSlices(train_loader, 0, cfg.grad_accum, total_steps)

    refs = None
    if cfg.precompute_frozen_kl and cfg.kl_frozen_weight > 0:
        if not cfg.group_by_length:
            raise ValueError("precompute_frozen_kl needs group_by_length (a known batch order)")
        run = [b for e in range(cfg.epochs) for b in sampler.batches(e)]
        t_ref = time.time()
        model.train()
        ref_model = StrandsDeciderModel.load(
            cfg.kl_frozen_reference, attn_implementation=cfg.attn_implementation
        )
        ref_model.to(device).eval()
        own = ref_model.tokenizer.get_vocab() == model.tokenizer.get_vocab()
        print(
            f"[strands-decider] frozen-KL reference: {cfg.kl_frozen_reference}"
            + ("" if own else " (its own tokenizer, not the student's)")
        )
        refs = _frozen_reference(
            model, train_examples, run[: total_steps * cfg.grad_accum], coll_cfg, device, ref_model
        )
        del ref_model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[strands-decider] frozen-KL reference for {refs.shape[0]:,} micro-batches "
            f"in {time.time() - t_ref:.0f} s"
        )

    ema = _Ema(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    print(f"[strands-decider] {total_steps} optimizer steps (warmup {warmup})")
    model.train()
    step, micro, running, t0 = 0, 0, 0.0, time.time()
    running_kl = 0.0
    running_tkl = 0.0
    history: list[dict[str, float]] = []
    done = False

    for epoch in range(cfg.epochs):
        if done:
            break
        for batch in train_loader:
            # Under torchrun a forward is this rank's slice of a micro-batch (distributed.py).
            part = batch.pop("part", distributed.WHOLE)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if world > 1:  # what DDP's no_sync() sets: all-reduce on the step's last forward
                fwd.require_backward_grad_sync = part.last
            out = fwd(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                n_slots=batch["n_slots"],
                opt_idx=batch.get("opt_idx"),
                labels=batch["labels"],
                label_dist=batch.get("label_dist"),
                weights=batch.get("weights"),
            )
            step_loss = out["loss"] * part.weights
            if cfg.kl_frozen_weight > 0:
                m, a = (part.micro, part.start) if world > 1 else (micro, 0)
                ref_lp = refs[m, a : a + batch["labels"].numel()]
                stu = out["log_probs"]
                # The reference is sized to the widest option count the run saw; a batch
                # whose rows are narrower than that gets the padded columns trimmed off.
                # Trimmed columns are past some row's option count and already -inf in
                # both the reference and the student, so no probability mass is dropped.
                ref_lp = ref_lp[:, : stu.shape[-1]]
                # Masked slots are -inf in both; -inf - -inf is NaN, so restrict the sum
                # to entries finite on both sides rather than patching the NaN afterwards.
                valid = torch.isfinite(ref_lp) & torch.isfinite(stu)
                per_row = _kl(ref_lp, stu, valid, cfg.kl_direction)
                kl = per_row.mean() * part.kl
                if cfg.kl_only_files:
                    # KL-only rows (weight 0) carry their own KL weight; with none in the
                    # batch this equals kl_frozen_weight * kl exactly.
                    only = batch["weights"] == 0
                    coef = torch.where(only, cfg.kl_only_weight, cfg.kl_frozen_weight)
                    step_loss = step_loss + (coef * per_row).mean() * part.kl
                else:
                    step_loss = step_loss + cfg.kl_frozen_weight * kl
                running_kl += float(kl)
            if (
                cfg.teacher_weight > 0
                and "has_teacher" in batch
                and bool(batch["has_teacher"].any())
            ):
                has = batch["has_teacher"]
                stu = out["log_probs"][has]
                tea = batch["teacher"][has][:, : stu.shape[-1]]
                tkl = _teacher_kl(tea, stu, cfg.kl_direction).mean() * part.teacher
                step_loss = step_loss + cfg.teacher_weight * tkl
                running_tkl += float(tkl)
            # DDP averages the ranks' gradients; `* world` makes that the 1-GPU sum.
            loss = step_loss * world / cfg.grad_accum
            loss.backward()
            running += float(step_loss)
            micro += 1

            ends_step = part.last if world > 1 else micro % cfg.grad_accum == 0
            if not ends_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
            )
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                ema.update(step)

            if step % cfg.log_every == 0:
                running, running_kl, running_tkl = distributed.all_reduce(
                    [running, running_kl, running_tkl]
                )  # each rank holds its slices' part
                avg = running / (cfg.log_every * cfg.grad_accum)
                rate = step / max(1e-6, time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                mem = distributed.all_reduce([mem], "max")[0]
                print(
                    f"[strands-decider] epoch {epoch} step {step}/{total_steps} "
                    f"loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{rate:.2f} step/s peak_mem {mem:.1f}GiB"
                )
                entry = {"step": step, "loss": avg}
                if cfg.kl_frozen_weight > 0:
                    entry["kl"] = running_kl / (cfg.log_every * cfg.grad_accum)
                    print(f"[strands-decider]   kl {entry['kl']:.4f}")
                if cfg.teacher_weight > 0:
                    entry["teacher_kl"] = running_tkl / (cfg.log_every * cfg.grad_accum)
                    print(f"[strands-decider]   teacher kl {entry['teacher_kl']:.4f}")
                history.append(entry)
                running = running_kl = running_tkl = 0.0

            # Validation and saving run on rank 0 alone; every rank holds the same weights.
            if cfg.eval_every and step % cfg.eval_every == 0 and rank == 0:
                metrics = evaluate_loss(model, val_loader, device)
                print(f"[strands-decider] eval @ {step}: {metrics}")
                history.append({"step": step, **metrics})

            if cfg.save_every and step % cfg.save_every == 0 and rank == 0:
                if ema is not None:
                    ema.apply()  # save the average, then train on from the live weights
                model.save_pretrained(cfg.output_dir)
                if ema is not None:
                    ema.restore()

            if step >= total_steps:
                done = True
                break

    if rank != 0:
        return cfg.output_dir
    if ema is not None:
        ema.apply()
    metrics = evaluate_loss(model, val_loader, device)
    print(f"[strands-decider] final eval: {metrics}")
    history.append({"step": step, **metrics})

    model.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"[strands-decider] saved to {cfg.output_dir}")
    return cfg.output_dir


# ---------------------------------------------------------------- CLI


def _cli(
    config: str | None = typer.Option(None, "--config", "-c", help="YAML training config."),
    train_file: list[str] | None = typer.Option(None, "--train-file", help="Override data."),
    base_model: str | None = typer.Option(None, help="Override base model id."),
    output_dir: str | None = typer.Option(None, help="Override checkpoint directory."),
    epochs: int | None = typer.Option(None),
    micro_batch_size: int | None = typer.Option(None),
    max_steps: int | None = typer.Option(None, help="Hard step cap (smoke tests)."),
) -> None:
    """Train the pointer head and a LoRA adapter on the torso."""
    cfg = TrainConfig.from_yaml(config) if config else TrainConfig()
    if train_file:
        cfg.train_files = list(train_file)
    for key, val in (
        ("base_model", base_model),
        ("output_dir", output_dir),
        ("epochs", epochs),
        ("micro_batch_size", micro_batch_size),
        ("max_steps", max_steps),
    ):
        if val is not None:
            setattr(cfg, key, val)
    if not cfg.train_files:
        raise typer.BadParameter("no training files; pass --train-file or set train_files")
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    typer.run(_cli)
