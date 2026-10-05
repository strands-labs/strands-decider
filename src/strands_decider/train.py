"""Training the readout head (slot or pointer; see modeling.py) and a LoRA adapter on the torso.

The objective is deliberately plain -- cross-entropy over the masked slots, plus
optional teacher and frozen-KL terms (the reference recipe uses both) -- because the
interesting work is in the *distribution* the examples are drawn from, not the loss.
Each step shows the model a different task, a different number of options, and a
different option ordering. What has to be learned to fit that is precisely the
behaviour we want at serving time.

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
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar, cast

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from . import distributed
from .data.collate import KIND_IDS, CollatorConfig, SystemOneCollator
from .data.format import Example, load_examples, split_examples
from .data.sampling import LengthGroupedBatchSampler, example_length, padding_fraction
from .modeling import StrandsDeciderConfig, StrandsDeciderModel

C = TypeVar("C", bound="YamlConfig")


class YamlConfig:
    """A dataclass config read from YAML, refusing keys it does not define."""

    @classmethod
    def from_yaml(cls: type[C], path: str) -> C:
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = set(getattr(cls, "__dataclass_fields__", {}))
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def with_overrides(self: C, pairs: list[str]) -> C:
        """A copy with `key=value` overrides applied (values read as YAML), refusing keys
        the config does not define."""
        import dataclasses

        import yaml

        known = set(getattr(self, "__dataclass_fields__", {}))
        over: dict[str, Any] = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep or key not in known:
                raise ValueError(f"not a key=value override of a config field: {pair!r}")
            over[key] = yaml.safe_load(value)
        return cast(C, dataclasses.replace(cast(Any, self), **over))


@dataclass
class TrainConfig(YamlConfig):
    # data
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    val_fraction: float = 0.03

    # model
    base_model: str = "Qwen/Qwen3-1.7B-Base"
    base_revision: str | None = None  # a Hub commit to pin the base to
    num_slots: int = 24
    head_hidden: int = 0
    head_dropout: float = 0.05
    max_length: int = 3072
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    # None keeps StrandsDeciderConfig's default (attention and MLP projections). A hybrid torso
    # such as Qwen3.5 needs its recurrent layers' projections listed too, or LoRA reaches
    # only its few full-attention layers.
    lora_targets: list[str] | None = None
    freeze_torso: bool = False
    # "lm_head" seeds the slot head from the LM's own option-number readout rather
    # than at random. Measured teacher quality for that readout on JevBench is
    # 0.550 overall but uneven -- 0.474 long_policy against the trained model's
    # 0.368, 0.167 ordinal against 0.833 -- so it is a starting point, not a target.
    head_type: str = "slot"
    pointer_dim: int = 256
    head_init: str = "random"
    # >0 adds KL(frozen || student) so training cannot drift away from that readout.
    kl_frozen_weight: float = 0.0
    # Rows trained ONLY by that KL term: no label loss, never in the validation split.
    # For prompts with no gold answer -- e.g. changed questions, where the frozen
    # torso's own reading is the target (data/question_transforms.py). Each needs <= 9
    # options, since the frozen readout reads single-token option numbers.
    kl_only_files: list[str] = field(default_factory=list)
    # A KL-only row's KL weight. The label loss is weighted 1 per row, so 1.0 gives a
    # KL-only row the same total weight as a labelled one.
    kl_only_weight: float = 1.0
    # A frozen teacher's option distribution per training row (data/teacher.py): JSONL of
    # {"i": row index in the concatenated train_files, "probs": [...canonical order]}.
    # Rows it covers get teacher_weight * KL(teacher || student) on top of the label loss.
    teacher_file: str | None = None
    teacher_weight: float = 0.0
    # Continue from an existing checkpoint, keeping its trained LoRA adapter and
    # attaching a freshly-initialised head. `freeze_torso` alone cannot do this: it
    # drops the adapter and freezes the *base* torso, which would train the new head
    # against unadapted features and silently measure the wrong thing.
    init_from: str | None = None
    # Continue training an existing checkpoint as it is: its adapter stays trainable and
    # its trained head is kept (`init_from` instead freezes the torso under a fresh head).
    # The stored calibration is reset to 1.0, since forward() applies it; recalibrate after.
    continue_from: str | None = None
    # Row kinds ("noul", "choice", "score") that get no frozen-KL term. The frozen torso's
    # option-number reading is near chance on yes/no questions, so anchoring yes/no rows
    # to it pulls their answers toward 0.5.
    kl_frozen_skip_kinds: list[str] = field(default_factory=list)

    # optimisation
    epochs: int = 1
    micro_batch_size: int = 8
    grad_accum: int = 4
    lr: float = 1e-4
    # The head is randomly initialised while LoRA starts as a no-op, so the head
    # needs a much larger step or the torso adapts to a head that is still noise.
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
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
    # `seed` drives everything drawn while training: the validation split, the data order,
    # option shuffles, score reversals and dropout. `init_seed`, when set, drives only the
    # random initialisation -- the fresh head and the LoRA A matrices -- after which the
    # torch RNG is reseeded with `seed`. Runs that share an init_seed and differ in seed
    # start from identical weights and see different data, which is what averaging their
    # weights (strands_decider.soup) needs. None (the default) seeds the initialisation
    # with `seed` and reseeds nothing, exactly as before the option existed. A
    # continue_from run initialises nothing at random, so init_seed does not touch it.
    seed: int = 0
    init_seed: int | None = None
    log_every: int = 20
    eval_every: int = 500
    save_every: int = 0  # 0 = only at the end
    max_steps: int = 0  # 0 = full epochs; otherwise a hard cap (smoke tests)
    attn_implementation: str | None = None
    # Speed only: compute the frozen-KL reference for every training row in one no-grad
    # pass before step 1 (`_frozen_reference`) instead of a second forward in every step.
    # Same numbers up to bf16 batch-shape rounding. Needs group_by_length.
    precompute_frozen_kl: bool = False


def _random_batches(n: int, cfg: TrainConfig) -> list[list[int]]:
    """What shuffle=True would have produced, for reporting the padding saved."""
    order = list(range(n))
    random.Random(cfg.seed).shuffle(order)
    b = cfg.micro_batch_size
    return [order[i:i + b] for i in range(0, n - n % b, b)]


class ExampleDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def _build_optimizer(model: StrandsDeciderModel, *, lr: float, head_lr: float,
                     weight_decay: float) -> torch.optim.Optimizer:
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
        {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]
    if torso_params:
        groups.append({"params": torso_params, "lr": lr, "weight_decay": 0.0})
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


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


# Padded tokens per forward of the frozen-reference pass: it has no backward, and it is
# launch-bound, so fewer, larger forwards win.
FROZEN_PASS_TOKENS = 32768


def _frozen_reference(model: StrandsDeciderModel, examples: list[Example], batches: list[list[int]],
                      coll_cfg: CollatorConfig, device: str) -> torch.Tensor:
    """`frozen_slot_log_probs` for every row the run trains on: ref[micro-batch, row].

    A fresh collator with the training collator's seed walks the run's micro-batches in
    training order, so each row is rendered exactly as training will render it. Under
    torchrun rank m % world renders micro-batch m (the others only draw its augmentations)
    and a sum over ranks assembles the table. Rows are forwarded sorted by length and
    packed; right padding never reaches a row's real tokens (causal attention and
    recurrence), the reference is the adapter-disabled torso whose weights training never
    changes, and it draws no random numbers -- so only the batch shapes differ from
    computing it inside each step.
    """
    rank, _, world = distributed.env()
    coll = SystemOneCollator(model.tokenizer, coll_cfg, train=True)
    ref = torch.zeros(len(batches), max(map(len, batches)), model.config.num_slots,
                      device=device)
    todo: list[Any] = []  # (micro-batch, row, token ids, option count)
    for m, idx in enumerate(batches):
        rows = [examples[i] for i in idx]
        if m % world != rank:
            coll.skip(rows)
            continue
        b = coll(rows)
        for r in range(len(rows)):
            n = int(b["attention_mask"][r].sum())
            todo.append((m, r, b["input_ids"][r, :n], int(b["n_slots"][r])))
    todo.sort(key=lambda t: -t[2].numel())
    i = 0
    while i < len(todo):
        longest = todo[i][2].numel()
        chunk = todo[i:i + max(1, FROZEN_PASS_TOKENS // longest)]
        i += len(chunk)
        ts = [t for _, _, t, _ in chunk]  # right padding, as the collator pads
        ids = pad_sequence(ts, batch_first=True, padding_value=model.tokenizer.pad_token_id)
        mask = pad_sequence([torch.ones_like(t) for t in ts], batch_first=True)
        n_slots = torch.tensor([t[3] for t in chunk], device=device)
        lp, _ = model.frozen_slot_log_probs(ids.to(device), mask.to(device), n_slots)
        if lp.numel():
            ref[[t[0] for t in chunk], [t[1] for t in chunk]] = lp
    if world > 1:
        torch.distributed.all_reduce(ref)  # others add zeros: -inf + 0 stays -inf
    return ref


@distributed.entry_point  # under torchrun, this process is one rank of the run
def train(cfg: TrainConfig) -> str:
    rank, _, world = distributed.env()
    torch.manual_seed(cfg.seed if cfg.init_seed is None else cfg.init_seed)
    random.seed(cfg.seed)  # model construction draws nothing from Python's RNG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[strands-decider] loading base model {cfg.base_model}")
    model_cfg = StrandsDeciderConfig(
        base_model=cfg.base_model,
        base_revision=cfg.base_revision,
        num_slots=cfg.num_slots,
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
        max_length=cfg.max_length,
        use_lora=cfg.use_lora and not cfg.freeze_torso,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        # Serving needs this to correct the variance floor smoothing imposes on
        # score confidence; without it a score could never reach high confidence.
        ordinal_smoothing=cfg.ordinal_smoothing,
        head_init=cfg.head_init,
        kl_frozen_weight=cfg.kl_frozen_weight,
        head_type=cfg.head_type,
        pointer_dim=cfg.pointer_dim,
    )
    if cfg.lora_targets:
        model_cfg.lora_targets = list(cfg.lora_targets)
    if cfg.continue_from and cfg.init_from:
        raise ValueError("set at most one of continue_from and init_from")
    unknown = set(cfg.kl_frozen_skip_kinds) - set(KIND_IDS)
    if unknown:
        raise ValueError(f"unknown kl_frozen_skip_kinds: {sorted(unknown)}")
    if cfg.continue_from:
        print(f"[strands-decider] continuing {cfg.continue_from} (adapter and head trainable)")
        model = StrandsDeciderModel.load(cfg.continue_from, attn_implementation=cfg.attn_implementation,
                                         trainable=True)
        if model.config.head_type != cfg.head_type:
            raise ValueError(f"continue_from has head_type {model.config.head_type!r}, "
                             f"the config asks for {cfg.head_type!r}")
        model.config.temperature = 1.0
        model.config.temperature_by_kind = {}
        # The checkpoint records this run's settings, as a fresh model's config would, and
        # the head trains with this run's dropout (a dropout layer holds no weights).
        model.config.kl_frozen_weight = cfg.kl_frozen_weight
        model.config.max_length = cfg.max_length
        model.config.ordinal_smoothing = cfg.ordinal_smoothing
        model.config.head_dropout = cfg.head_dropout
        for module in model.head.modules():
            if isinstance(getattr(module, "dropout", None), (torch.nn.Dropout, torch.nn.Identity)):
                module.dropout = (torch.nn.Dropout(cfg.head_dropout) if cfg.head_dropout > 0
                                  else torch.nn.Identity())
    elif cfg.init_from:
        from .modeling import SlotHead

        print(f"[strands-decider] loading torso + adapter from {cfg.init_from}")
        model = StrandsDeciderModel.load(cfg.init_from, attn_implementation=cfg.attn_implementation)
        # Swap in a fresh head of the requested shape; the loaded one is discarded so
        # that head capacity is the only thing differing between arms.
        model.config.head_hidden = cfg.head_hidden
        model.config.head_dropout = cfg.head_dropout
        model.config.ordinal_smoothing = cfg.ordinal_smoothing
        # Any calibration stored on the source checkpoint describes its old head and
        # is meaningless for a fresh one. Left in place it would be applied during
        # training by forward(), which the new head would simply absorb into its own
        # scale -- harmless but misleading. Reset, and recalibrate after training.
        model.config.temperature = 1.0
        model.config.temperature_by_kind = {}
        model.head = SlotHead(
            StrandsDeciderModel.hidden_size(model.torso),
            model.config.num_slots,
            hidden=cfg.head_hidden,
            dropout=cfg.head_dropout,
        ).to(torch.float32)
        model.freeze_torso()
    else:
        model = StrandsDeciderModel.from_pretrained_base(
            model_cfg, attn_implementation=cfg.attn_implementation
        )
        if cfg.freeze_torso:
            model.freeze_torso()
    if cfg.head_init == "lm_head":
        seeded = model.init_head_from_lm_head()
        print(f"[strands-decider] head seeded from lm_head rows for {seeded} slots")
    elif cfg.head_init != "random":
        raise ValueError(f"unknown head_init {cfg.head_init!r}")
    if cfg.init_seed is not None:
        # Initialisation is over: from here on the torch RNG (the unsorted DataLoader's
        # order, dropout) follows `seed`, whatever init_seed was.
        torch.manual_seed(cfg.seed)
    # Checkpointing needs a grad-requiring input; a frozen torso has none, and it
    # buys nothing anyway since no backward pass traverses it.
    if cfg.gradient_checkpointing and not (cfg.freeze_torso or cfg.init_from):
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
                    raise ValueError(f"teacher row {d['i']} has {len(d['probs'])} probs for "
                                     f"{ex.n_options} options -- wrong train_files?")
                ex.teacher = d["probs"]
                n_t += 1
        print(f"[strands-decider] teacher distributions on {n_t:,} of {len(train_examples):,} rows "
              f"(weight {cfg.teacher_weight})")
    if cfg.val_files:
        val_examples = load_examples(cfg.val_files)
    else:
        train_examples, val_examples = split_examples(
            train_examples, val_fraction=cfg.val_fraction, seed=cfg.seed
        )
    if cfg.kl_only_files:
        if cfg.kl_frozen_weight <= 0:
            raise ValueError("kl_only_files needs kl_frozen_weight > 0 to enable the KL term")
        if any(ex.weight <= 0 for ex in train_examples):
            raise ValueError("labelled rows must have weight > 0: weight 0 marks KL-only rows")
        kl_rows = load_examples(cfg.kl_only_files)
        for ex in kl_rows:
            if ex.n_options > 9:
                raise ValueError(f"KL-only row from {ex.task!r} has {ex.n_options} options")
            ex.weight = 0.0  # no label loss; the training loop reads 0 as "KL only"
        # Added after the split, so none reaches validation, whose labels would be fake.
        train_examples = train_examples + kl_rows
        random.Random(cfg.seed).shuffle(train_examples)
        print(f"[strands-decider] +{len(kl_rows):,} KL-only rows (weight {cfg.kl_only_weight})")
    print(f"[strands-decider] train={len(train_examples):,} val={len(val_examples):,}")
    if not train_examples:
        raise SystemExit("no training examples; run `strands-decider data build` first")

    coll_cfg = CollatorConfig(
        max_length=cfg.max_length,
        num_slots=cfg.num_slots,
        shuffle_options=cfg.shuffle_options,
        ordinal_smoothing=cfg.ordinal_smoothing,
        reverse_score_prob=cfg.reverse_score_prob,
        seed=cfg.seed,
        head_type=cfg.head_type,
    )
    train_collate = SystemOneCollator(model.tokenizer, coll_cfg, train=True)
    if cfg.group_by_length:
        lengths = [example_length(ex) for ex in train_examples]
        sampler = LengthGroupedBatchSampler(
            lengths, cfg.micro_batch_size, mega=cfg.length_group_mega,
            seed=cfg.seed, drop_last=True,
        )
        # Measured in prompt characters, the proxy the sampler sorts by -- which flatters
        # the grouped figure, since grouping is exact in that unit. Tables and prose
        # tokenise at different rates, so real token padding is higher (measured ~16%
        # grouped against ~58% ungrouped on the v10 corpus). The ungrouped figure agrees
        # in both units, so the comparison is still a fair signal of what was saved.
        print(f"[strands-decider] length-grouped batching: padding "
              f"{padding_fraction(lengths, sampler.batches(0)):.1%} of positions grouped vs "
              f"{padding_fraction(lengths, _random_batches(len(lengths), cfg)):.1%} "
              f"ungrouped (in prompt characters; token padding runs higher)")
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

    optim = _build_optimizer(model, lr=cfg.lr, head_lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: _lr_lambda(s, warmup, total_steps)
    )

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(cfg), fh, indent=2)
    slots = model.slot_token_ids()  # the rows frozen_slot_log_probs covers
    kl_slots = max(slots) + 1 if slots else 0
    if world > 1:  # each rank iterates its share of every step's rows instead
        train_loader = distributed.StepSlices(train_loader, kl_slots, cfg.grad_accum, total_steps,
                                              frozenset(cfg.kl_frozen_skip_kinds))

    refs = None
    if cfg.precompute_frozen_kl and cfg.kl_frozen_weight > 0:
        if not cfg.group_by_length:
            raise ValueError("precompute_frozen_kl needs group_by_length (a known batch order)")
        run = [b for e in range(cfg.epochs) for b in sampler.batches(e)]
        t_ref = time.time()
        model.train()  # the mode the in-step reference forward runs in
        refs = _frozen_reference(model, train_examples, run[: total_steps * cfg.grad_accum],
                                 coll_cfg, device)
        print(f"[strands-decider] frozen-KL reference for {refs.shape[0]:,} micro-batches "
              f"in {time.time() - t_ref:.0f} s")

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
                if refs is None:
                    ref_lp, eligible = model.frozen_slot_log_probs(
                        batch["input_ids"], batch["attention_mask"], batch["n_slots"]
                    )
                else:  # the same numbers, computed before step 1 (_frozen_reference)
                    m, a = (part.micro, part.start) if world > 1 else (micro, 0)
                    ref_lp = refs[m, a:a + batch["labels"].numel()]
                    eligible = batch["n_slots"] <= kl_slots
                for kind in cfg.kl_frozen_skip_kinds:
                    eligible = eligible & (batch["kind_id"] != KIND_IDS[kind])
                if ref_lp.numel() and bool(eligible.any()):
                    ref = ref_lp[eligible]
                    stu = out["log_probs"][eligible]
                    # The frozen reference is always num_slots wide; a pointer student
                    # is only as wide as the batch's widest question. Every column the
                    # student drops is past some row's option count and already -inf in
                    # the reference, so trimming changes no probability mass.
                    ref = ref[:, : stu.shape[-1]]
                    # Masked slots are -inf in both; -inf - -inf is NaN, so restrict
                    # the sum to entries finite on both sides rather than patching
                    # the NaN afterwards.
                    valid = torch.isfinite(ref) & torch.isfinite(stu)
                    p = ref.exp().masked_fill(~valid, 0.0)
                    diff = (ref - stu).masked_fill(~valid, 0.0)
                    per_row = (p * diff).sum(dim=-1)
                    kl = per_row.mean() * part.kl
                    if cfg.kl_only_files:
                        # KL-only rows (weight 0) carry their own KL weight; with none
                        # in the batch this equals kl_frozen_weight * kl exactly.
                        only = batch["weights"][eligible] == 0
                        coef = torch.where(only, cfg.kl_only_weight, cfg.kl_frozen_weight)
                        step_loss = step_loss + (coef * per_row).mean() * part.kl
                    else:
                        step_loss = step_loss + cfg.kl_frozen_weight * kl
                    running_kl += float(kl.detach())
            if cfg.teacher_weight > 0 and "has_teacher" in batch and bool(batch["has_teacher"].any()):
                has = batch["has_teacher"]
                stu = out["log_probs"][has]
                tea = batch["teacher"][has][:, : stu.shape[-1]]
                valid = (tea > 0) & torch.isfinite(stu)
                diff = (tea.clamp_min(1e-12).log() - stu).masked_fill(~valid, 0.0)
                tkl = (tea.masked_fill(~valid, 0.0) * diff).sum(dim=-1).mean() * part.teacher
                step_loss = step_loss + cfg.teacher_weight * tkl
                running_tkl += float(tkl.detach())
            # DDP averages the ranks' gradients; `* world` makes that the 1-GPU sum.
            loss = step_loss * world / cfg.grad_accum
            loss.backward()
            running += float(step_loss.detach())
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

            if step % cfg.log_every == 0:
                running, running_kl, running_tkl = distributed.all_reduce(
                    [running, running_kl, running_tkl])  # each rank holds its slices' part
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
                model.save_pretrained(cfg.output_dir)

            if step >= total_steps:
                done = True
                break

    if rank != 0:
        return cfg.output_dir
    metrics = evaluate_loss(model, val_loader, device)
    print(f"[strands-decider] final eval: {metrics}")
    history.append({"step": step, **metrics})

    model.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"[strands-decider] saved to {cfg.output_dir}")
    return cfg.output_dir
