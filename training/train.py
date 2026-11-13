"""The trainer: LoRA adapters on a frozen decoder plus a trained readout, in one process or on the GPUs of one machine.

    python -m training.train --config training/configs/e2b.yaml --base <base checkpoint dir> \\
        --train <conversion>/train.jsonl --dev <conversion>/dev.jsonl --output <run dir> [--seed 20260920]

A config (training/configs/<name>.yaml) holds a recipe's settings; any flag given on the command line overrides it.
`torchrun --standalone --nproc_per_node N -m training.train ...` trains on N GPUs of one machine
(training/distributed.py). The head is one of two:

  token_readout  (the default) Gemma 4's text decoder under the prefix_bidir mask, prompts rendered by
                 strands_decider.core (strands-chat or strands-plain), read at the last prompt token through a
                 [255, hidden] readout (training/model.py). Its runs merge into the base and export to the spec's
                 checkpoint (training/merge.py, average.py, export.py).
  pointer        v1's pointer head on its numbered render, on a Qwen3, Qwen3.5 or Gemma 4 torso (training/pointer.py).
                 Its saves are v1 checkpoints, which strands-decider's v1 engines serve.

What a step is, for both:
  * the plan (training/batching.py) cuts the rows into microbatches of at most --batch-size rows and --token-budget
    padded tokens; one optimizer step is --rows-per-step / --batch-size microbatches, spread over the GPUs, each GPU
    running accum = microbatches / GPUs of them. So the step plan is the same on any number of GPUs that divides the
    step's microbatches;
  * the loss of a microbatch is weighted by rows * world / rows_in_step before backward, so after the gradient
    all-reduce's mean the gradient is the per-row mean over the whole step; zero-weight fillers pad a short last step;
  * the decoder is frozen (bf16 on a GPU, float32 on the CPU); the adapters and the readout are float32 masters, and
    only they are reduced across GPUs; gradient checkpointing is on;
  * augmentation is seeded by (seed, step, microbatch index), and dropout by (seed, step, rank), so a resumed run on
    the same GPU count is the uninterrupted run. The token readout shuffles each choice's options
    (training/targets.py); the pointer head draws what v1's collator draws (training/pointer.py).

--attn-split N (token readout) runs the prefix_bidir attention of microbatches at least N tokens wide as two unmasked
attentions; it changes no step's rows or targets, and matches the masked attention up to bf16 rounding.

Writes, under --output: `run_config.json` (the arguments, the plan's statistics, the data and code hashes),
`train_log.jsonl` (one line per step), `eval_log.jsonl` (the dev metrics every --eval-every steps), and every
--save-every steps and at the end `step-NNNNN/` with `trainer_state.pt`, with `last` -> the newest. The token readout
saves `step-NNNNN/adapter/{adapter.safetensors, adapter_config.json}` (adapter_config.json is the training record the
merge carries into the merged checkpoint); unless --no-merge, the primary process merges the run's last checkpoint
(and every --merge-every-th) into `merged/step-NNNNN` on the CPU in the background (training/merge.py) and waits for
the merges before it exits; a resumed run that has nothing left to train merges its last checkpoint if that merge is
missing. The pointer head saves `step-NNNNN/checkpoint/`, a v1 checkpoint with its record as training.json.

Booleans in a config can be turned off on the command line: --no-expert-lora, --no-allow-family-overlap,
--no-force-bos, --augment (after a config's no_augment), --merge (after a config's no_merge).

--plan-only prints the plan's statistics for --world-size GPUs and exits (single process, no GPU).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import torch
import torch.distributed as dist

from . import dev_eval, distributed, expert_lora, lora, pointer
from .batching import Placement, build_plan, plan_statistics
from .folds import TrainIndex, check_conversion, family_key, read_fold, sha256_file
from .loss import training_loss
from .merge import base_digest
from .prompt import TEMPLATES
from .schedule import lr_multiplier
from .targets import shuffle_options, target_matrix

HERE = Path(__file__).resolve().parent
HEADS = ("token_readout", "pointer")
PROMPTS = (*TEMPLATES, pointer.PROMPT)

# The recipe's settings, which the configs refine.
DEFAULTS = {
    "head": "token_readout",
    "prompt": "strands-chat",
    "epochs": 1.0,
    "rows_per_step": 3072,
    "accum": 0,
    "batch_size": 32,
    "token_budget": 32768,
    "max_length": 16384,
    "eval_batch_size": None,
    "eval_token_budget": None,
    "sort_window": 0,
    "lr": 1e-4,
    "warmup_ratio": 0.05,
    "lr_floor": 0.1,
    "weight_decay": 0.0,
    "brier_weight": 0.0,
    "max_grad_norm": 1.0,
    "rank": 64,
    "alpha": 128.0,
    "dropout": 0.05,
    "targets": ",".join(lora.DEFAULT_TARGETS),
    "no_augment": False,
    "seed": 20260920,
    "save_every": 100,
    "eval_every": 100,
    "merge_every": 0,
    "no_merge": False,
    "log_every": 1,
    "keep_optimizer": 2,
    "bucket_mb": 100,
    "attn_split": 0,
    "expert_lora": False,
    "expert_rank": 0,
    "expert_alpha": None,
    "cpu_threads": 8,
    "train_limit": 0,
    "dev_limit": 0,
    "allow_family_overlap": False,
    # the pointer head's
    "pointer_dim": 256,
    "head_dropout": 0.05,
    "head_lr": 1e-3,
    "force_bos": False,
    "reverse_score_prob": 0.5,
}
# Config keys that describe the recipe but are not trainer flags.
CONFIG_ONLY = {"seeds", "tokenizer_sha256", "base_sha256", "size"}
# The arguments a resumed run must share with the saved one.
RESUME_KEYS = (
    "head",
    "base_model",
    "revision",
    "base_sha256",
    "prompt",
    "lr",
    "warmup_ratio",
    "lr_floor",
    "weight_decay",
    "brier_weight",
    "max_grad_norm",
    "rank",
    "alpha",
    "dropout",
    "targets",
    "seed",
    "accum",
    "batch_size",
    "token_budget",
    "max_length",
    "no_augment",
    "attn_split",
    "expert_lora",
    "expert_rank",
    "expert_alpha",
    "pointer_dim",
    "head_dropout",
    "head_lr",
    "force_bos",
    "reverse_score_prob",
)


def log(payload: dict) -> None:
    print(json.dumps(payload, default=str), flush=True)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, default=str) + "\n")


def plain_number(value: float) -> float | int:
    return int(value) if float(value).is_integer() else float(value)


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(directory.rglob("*.py")):
        digest.update(f.relative_to(directory).as_posix().encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


def code_hashes(head: str = "token_readout") -> dict:
    """sha256 of this package's modules and of the strands_decider code that renders the prompts (the core; for the
    pointer head also v1's model and render)."""
    from strands_decider import core

    code = {f"training/{p.name}": sha256_file(p) for p in sorted(HERE.glob("*.py"))}
    code["strands_decider.core"] = _tree_digest(Path(core.__file__).resolve().parent)
    if head == "pointer":
        import strands_decider

        package = Path(strands_decider.__file__).resolve().parent
        for name in ("modeling.py", "prompting.py", "schema.py"):
            code[f"strands_decider/{name}"] = sha256_file(package / name)
        code["training/data/format.py"] = sha256_file(HERE / "data" / "format.py")
    return code


# ----------------------------------------------------------------------------------------------
# model


def build_model(args: argparse.Namespace, device: torch.device, ordinal_smoothing: float) -> Any:
    """The frozen decision model on `device`, gradient checkpointing on. The pointer head's adapter and head are
    attached here, as v1's loader attaches them; the token readout's in wrap_lora."""
    if args.head == "pointer":
        torch.set_num_threads(args.cpu_threads)
        model = pointer.PointerModel(
            args.base,
            base_model=args.base_model,
            revision=args.revision,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            lora_targets=[t for t in args.targets.split(",") if t],
            pointer_dim=args.pointer_dim,
            head_dropout=args.head_dropout,
            force_bos=args.force_bos,
            max_length=args.max_length,
            ordinal_smoothing=ordinal_smoothing,
            device=str(device),
        )
        model.gradient_checkpointing_enable()
        return model
    from .model import DeciderModel

    readout = DeciderModel(
        args.base, prompt_template=args.prompt, device=str(device), cpu_threads=args.cpu_threads
    )
    readout.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return readout


def wrap_lora(model: Any, args: argparse.Namespace) -> tuple[dict, list[torch.nn.Parameter]]:
    """Attach the adapters (B = 0; A drawn from the seed, in module-tree order, dense adapters first, then the experts')
    and re-hold the readout as a float32 master. Returns ({name: module}, the trainable parameters in a fixed order).
    The pointer head's are already attached (build_model)."""
    if args.head == "pointer":
        parameters = model.trainable_parameters()
        attached = {n: m for n, m in model.decider.torso.named_modules() if hasattr(m, "lora_A")}
    else:
        torch.manual_seed(args.seed)
        targets_ = tuple(t for t in args.targets.split(",") if t)
        attached = lora.attach(
            model.backbone, rank=args.rank, alpha=args.alpha, dropout=args.dropout, targets=targets_
        )
        extra = []
        if args.expert_lora:
            experts = expert_lora.attach(
                model.backbone, rank=args.expert_rank, alpha=args.expert_alpha, dropout=args.dropout
            )
            attached = {**attached, **experts}
            extra = expert_lora.parameters(model.backbone)
        model.readout = lora.promote(model.readout)
        parameters = lora.parameters(model.backbone) + extra + [model.readout.weight]
    trainable = {id(p) for p in parameters}
    stray = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in trainable]
    if stray:
        raise RuntimeError(f"unexpected trainable parameters: {stray[:5]}")
    return attached, parameters


def adapter_record(args: argparse.Namespace, extra: dict) -> dict:
    """The training record saved beside each adapter, carried into the merged checkpoint's training.json (the
    pointer head's: its checkpoint's training.json)."""
    record = {
        "trainer": "training.train",
        "head": args.head,
        # What the run trained on, which the export declares: text only (folds.TrainIndex drops rows with images).
        "inputs": ["text"],
        "base_model": args.base_model,
        "revision": args.revision,
        "base_sha256": args.base_sha256,
        "train": os.path.abspath(args.train),
        "dev": os.path.abspath(args.dev),
        "temperature_data": os.path.abspath(args.temperature_data)
        if args.temperature_data
        else None,
        "epochs": float(args.epochs),
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "schedule": f"linear warmup, then cosine to {plain_number(args.lr_floor)}x",
        "rows_per_step": args.rows_per_step,
        "accum": args.accum,
        "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "max_length": args.max_length,
        "seed": args.seed,
        "brier_weight": args.brier_weight,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "augment": not args.no_augment,
        "lora": {
            "rank": args.rank,
            "alpha": plain_number(args.alpha),
            "dropout": args.dropout,
            "targets": ",".join(t for t in args.targets.split(",") if t),
        },
        "expert_lora": (
            {
                "rank": args.expert_rank,
                "alpha": plain_number(args.expert_alpha),
                "dropout": args.dropout,
                "scale": args.expert_alpha / args.expert_rank,
                "targets": "experts.gate_up_proj,experts.down_proj",
                "layout": "<layer>.experts.{gate_up_a [E,r,H], gate_up_b [E,2I,r], down_a [E,r,I], "
                "down_b [E,H,r]}",
                "router": "frozen",
            }
            if args.expert_lora
            else None
        ),
        "backbone": "frozen, bfloat16",
        "readout": "trained, float32 master",
        "attention_mode": "prefix_bidir",
        "prompt": args.prompt,
        "pooling": "last",
        "calibration": "uncalibrated",
        **extra,
    }
    if args.head == "pointer":
        record.update(
            {
                "backbone": "frozen, bfloat16 on a GPU, float32 on the CPU",
                "readout": "the pointer head, float32",
                "attention_mode": "causal",
                "pooling": "the last token, and each option's last token",
                "pointer": {
                    "pointer_dim": args.pointer_dim,
                    "head_dropout": args.head_dropout,
                    "head_lr": args.head_lr,
                    "force_bos": args.force_bos,
                    "reverse_score_prob": args.reverse_score_prob,
                },
            }
        )
    return record


# ----------------------------------------------------------------------------------------------
# saving and resuming


def fsync_tree(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_adapter(model: Any, directory: Path, record: dict) -> None:
    """lora.save()'s layout; with --expert-lora the expert adapters go in the same adapter.safetensors."""
    if not expert_lora.modules(model.backbone):
        lora.save(model.backbone, str(directory), record, readout=model.readout)
        return
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        **lora.state_dict(model.backbone),
        **expert_lora.state_dict(model.backbone),
        "readout.weight": model.readout.weight.detach().float().cpu().contiguous(),
    }
    save_file(tensors, str(directory / "adapter.safetensors"))
    config = {
        **record,
        "tensors": len(tensors),
        "parameters": int(sum(t.numel() for t in tensors.values())),
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2))


def load_adapter(model: Any, path: Path) -> None:
    lora.load(model.backbone, str(path), model.readout)
    if expert_lora.modules(model.backbone):
        from safetensors.torch import load_file

        expert_lora.load(model.backbone, load_file(str(path)))


def _fsync_all(directory: Path) -> None:
    for sub in sorted(p for p in directory.rglob("*") if p.is_dir()):
        fsync_tree(sub)
    fsync_tree(directory)


def _publish(output: Path, pending: Path, step: int) -> Path:
    """Move a finished save into place and point `last` at it."""
    final = output / f"step-{step:05d}"
    _fsync_all(pending)
    if final.exists():
        shutil.rmtree(final)
    os.replace(pending, final)
    link = output / ".last.pending"
    link.unlink(missing_ok=True)
    link.symlink_to(final.name, target_is_directory=True)
    os.replace(link, output / "last")
    return final


def save_step(
    output: Path,
    step: int,
    model: Any,
    optimizer: torch.optim.Optimizer,
    record: dict,
    state: dict,
    keep_optimizer: int,
) -> Path:
    pending = output / f".step-{step:05d}.pending"
    shutil.rmtree(pending, ignore_errors=True)
    if isinstance(model, pointer.PointerModel):
        model.save(pending / "checkpoint", record)
    else:
        save_adapter(model, pending / "adapter", record)
    torch.save({**state, "optimizer": optimizer.state_dict()}, pending / "trainer_state.pt")
    final = _publish(output, pending, step)
    states = sorted(output.glob("step-*/trainer_state.pt"))
    for old in states[: max(0, len(states) - keep_optimizer)]:
        old.unlink()
    return final


def adapter_dir(path: str | Path) -> Path:
    """A step directory (step-NNNNN/adapter/) or an adapter directory itself."""
    path = Path(path).resolve()
    return path / "adapter" if (path / "adapter" / "adapter_config.json").exists() else path


def resume_source(args: argparse.Namespace) -> Path | None:
    if args.resume_from:
        return Path(args.resume_from).resolve()
    if args.resume:
        last = Path(args.output) / "last"
        return last.resolve() if last.exists() else None
    return None


class Merges:
    """Background CPU merges of saved checkpoints (the primary process): training/merge.py into
    <output>/merged/step-NNNNN."""

    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        self.args, self.output = args, output
        self.running: list[tuple[int, subprocess.Popen, IO[str]]] = []

    def start(self, step: int) -> None:
        target = self.output / "merged" / f"step-{step:05d}"
        if (target / "training.json").exists():
            return
        shutil.rmtree(target, ignore_errors=True)  # an interrupted merge's leftovers
        target.parent.mkdir(parents=True, exist_ok=True)
        logfile = open(target.parent / f"step-{step:05d}.log", "w", encoding="utf-8")
        command = [
            "nice",
            "-n",
            "10",
            sys.executable,
            "-m",
            "training.merge",
            "--base",
            str(self.args.base),
            "--adapter",
            str(self.output / f"step-{step:05d}" / "adapter"),
            "--out",
            str(target),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(key, None)
        process = subprocess.Popen(
            command, cwd=str(HERE.parent), env=env, stdout=logfile, stderr=subprocess.STDOUT
        )
        self.running.append((step, process, logfile))
        log({"event": "merge_started", "step": step, "out": str(target)})

    def wait(self) -> list[int]:
        """Wait for every merge started; returns the steps whose merge failed."""
        failed = []
        for step, process, logfile in self.running:
            code = process.wait()
            logfile.close()
            log(
                {
                    "event": "merged" if code == 0 else "merge_failed",
                    "step": step,
                    "returncode": code,
                }
            )
            if code:
                failed.append(step)
        self.running = []
        return failed


# ----------------------------------------------------------------------------------------------
# modes


def held_out(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    dev_rows = read_fold(args.dev, args.dev_limit)
    temperature_rows = read_fold(args.temperature_data, 0)
    return dev_rows, temperature_rows


def train_index(args: argparse.Namespace, protected_rows: list[dict]) -> TrainIndex:
    return TrainIndex(
        args.train,
        args.train_limit,
        args.max_length,
        {r["id"] for r in protected_rows},
        {family_key(r) for r in protected_rows},
        args.allow_family_overlap,
    )


def plan_only(args: argparse.Namespace) -> None:
    if args.base:
        check_conversion(args.train, Path(args.base) / "tokenizer.json", args.prompt)
    dev_rows, temperature_rows = held_out(args)
    index = train_index(args, dev_rows + temperature_rows)
    world = args.world_size
    accum = resolve_accum(args, world)
    window = args.sort_window or world * accum * args.batch_size
    plan = build_plan(
        index.lengths,
        args.epochs,
        args.seed,
        world,
        accum,
        args.batch_size,
        args.token_budget,
        window,
    )
    placement = Placement(world, accum)
    log(
        {
            "mode": "plan_only",
            "world_size": world,
            "batch_size": args.batch_size,
            "accum": accum,
            "rows": len(index),
            "dropped": index.dropped,
            "family_overlap_rows": index.family_overlap,
            "fixed_order_rows": index.fixed_order_rows,
            **plan_statistics(plan, index.lengths, placement),
        }
    )


def resolve_accum(args: argparse.Namespace, world: int) -> int:
    """--accum, or the step's microbatches (--rows-per-step / --batch-size) over the world size, which must divide."""
    if args.accum:
        accum: int = args.accum
        return accum
    per_step, remainder = divmod(args.rows_per_step, args.batch_size)
    accum, uneven = divmod(per_step, world)
    if remainder or uneven or accum < 1:
        raise SystemExit(
            f"{args.rows_per_step} rows a step in microbatches of {args.batch_size} rows do not divide "
            f"over {world} GPUs; choose a GPU count that divides {per_step}, or pass --accum"
        )
    return accum


def check_pointer_conversion(conversion: dict, args: argparse.Namespace) -> float:
    """The pointer head's checks of the conversion manifest; returns the ordinal smoothing its score targets carry
    (written into the checkpoint's config, which serving reads)."""
    if bool(conversion.get("force_bos", False)) != bool(args.force_bos):
        raise ValueError(
            f"{args.train} was measured with force_bos {conversion.get('force_bos', False)}, the run has "
            f"{args.force_bos}; convert it again"
        )
    smoothing: float = float(conversion.get("ordinal_smoothing", 0.0))
    return smoothing


def train(args: argparse.Namespace) -> None:
    run = distributed.Process()
    output = Path(args.output).resolve()
    args.output = str(output)
    if run.primary:
        output.mkdir(parents=True, exist_ok=True)
    run.barrier()
    attempt = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    started = time.monotonic()
    args.accum = resolve_accum(args, run.world)
    readout = args.head == "token_readout"

    # Folds. Every rank indexes the train file (offsets and lengths only) and keeps just its own rows in memory.
    conversion = check_conversion(args.train, Path(args.base) / "tokenizer.json", args.prompt)
    ordinal_smoothing = 0.0 if readout else check_pointer_conversion(conversion, args)
    dev_rows, temperature_rows = held_out(args)
    index = train_index(args, dev_rows + temperature_rows)
    if readout:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(args.base) / "tokenizer.json"))
        # Scored rows only (every held-out row's id and family is protected above), measured under the template.
        dev_rows, dev_dropped = dev_eval.measure(dev_rows, tokenizer, args.prompt, args.max_length)
        temperature_rows, temperature_dropped = dev_eval.measure(
            temperature_rows, tokenizer, args.prompt, args.max_length
        )
    else:
        tokenizer = pointer.load_tokenizer(args.base, args.force_bos)
        dev_rows, dev_dropped = pointer.measure(dev_rows, tokenizer, args.max_length)
        temperature_rows, temperature_dropped = pointer.measure(
            temperature_rows, tokenizer, args.max_length
        )
    window = args.sort_window or run.world * args.accum * args.batch_size
    plan = build_plan(
        index.lengths,
        args.epochs,
        args.seed,
        run.world,
        args.accum,
        args.batch_size,
        args.token_budget,
        window,
    )
    placement = Placement(run.world, args.accum)
    stats = plan_statistics(plan, index.lengths, placement)
    reference = run.broadcast_object(stats["sha256"])
    if reference != stats["sha256"]:
        raise RuntimeError(f"rank {run.rank} built a different step plan than rank 0")
    total_steps = len(plan)
    stop = min(total_steps, args.stop_after) if args.stop_after else total_steps

    hashes = None
    if run.primary:
        hashes = {
            "train_file": sha256_file(args.train),
            "train_rows_used": index.prefix_sha256,
            "dev": sha256_file(args.dev),
        }
        if args.temperature_data:
            hashes["temperature"] = sha256_file(args.temperature_data)
    hashes = run.broadcast_object(hashes)
    recorded = (conversion.get("train") or {}).get("sha256")
    if recorded and recorded != hashes["train_file"]:
        raise ValueError(
            f"{args.train} is not the file its conversion manifest describes (sha256 differs); convert it "
            "again"
        )
    code = code_hashes(args.head)
    data_record = {
        "train_rows": len(index),
        "train_lines_read": index.lines_read,
        "train_limit": args.train_limit,
        "dropped_rows": index.dropped,
        "family_overlap_rows": index.family_overlap,
        "dev_rows": len(dev_rows),
        "temperature_rows": len(temperature_rows),
        "held_out_rows_not_scored": dev_dropped + temperature_dropped,
    }
    run_config = {
        "args": vars(args),
        "world_size": run.world,
        "sort_window": window,
        "data_sha256": hashes,
        "code_sha256": code,
        **data_record,
        "plan": stats,
        "optimizer_steps": total_steps,
    }

    # Resume state, checked before the model loads.
    source = resume_source(args)
    saved = None
    if source is not None:
        saved = torch.load(source / "trainer_state.pt", map_location="cpu", weights_only=True)
        expected = {
            "world_size": run.world,
            "plan_sha256": stats["sha256"],
            "data_sha256": hashes,
            "optimizer_steps": total_steps,
        }
        mismatched = [k for k in expected if saved[k] != expected[k]]
        mismatched += [f"args.{k}" for k in RESUME_KEYS if saved["args"].get(k) != vars(args)[k]]
        if mismatched:
            raise ValueError(f"cannot resume from {source}: differs in {mismatched}")
    elif args.resume and run.primary:
        log({"event": "resume_requested_but_no_save", "output": str(output)})
    if saved is None and (output / "last").exists():
        raise ValueError(f"{output} already holds saves; pass --resume or choose a new --output")
    start_step = saved["step"] if saved else 0
    merging = readout and not args.no_merge
    if start_step >= stop:
        failed = []
        if run.primary:
            log({"event": "nothing_to_do", "step": start_step, "stop": stop})
            if merging and (output / f"step-{start_step:05d}" / "adapter").is_dir():
                final_merge = Merges(args, output)
                final_merge.start(start_step)
                failed = final_merge.wait()
        run.barrier()
        run.close()
        if failed:
            raise RuntimeError(
                f"the merge of step {start_step} failed; see {output}/merged/step-{start_step:05d}.log"
            )
        return

    if run.primary:
        if saved is None:
            (output / "run_config.json").write_text(
                json.dumps(run_config, indent=2, default=str) + "\n"
            )
        else:
            append_jsonl(
                output / "attempts.jsonl",
                {"attempt": attempt, "resumed_from": str(source), "step": start_step},
            )
        log(
            {
                "event": "plan",
                "attempt": attempt,
                "world_size": run.world,
                "accum": args.accum,
                **data_record,
                **{k: v for k, v in stats.items() if k != "sha256"},
                "stop": stop,
                "start_step": start_step,
            }
        )

    # Model: every rank loads the frozen decoder, then attaches the adapters.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.monotonic()
    model = build_model(args, run.device, ordinal_smoothing)
    if args.attn_split:
        model.install_split_attention(args.attn_split)
    attached, parameters = wrap_lora(model, args)
    if saved is not None:
        assert source is not None  # a save is read only from a source
        if readout:
            load_adapter(model, adapter_dir(source) / "adapter.safetensors")
        else:
            model.load_trained(source / "checkpoint")
    load_seconds = time.monotonic() - t0
    trainable_ids = {id(p) for p in parameters}
    trainable_names = {n for n, p in model.named_parameters() if id(p) in trainable_ids}
    ddp = (
        distributed.wrap(model, trainable_names, run.device, args.bucket_mb)
        if run.world > 1
        else model
    )
    if readout:
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay, fused=run.device.type == "cuda"
        )
    else:  # v1's optimizer for the pointer head and its adapter (pointer.PointerModel.parameter_groups)
        optimizer = torch.optim.AdamW(
            model.parameter_groups(args.lr, args.head_lr, args.weight_decay),
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    peaks = [group["lr"] for group in optimizer.param_groups]
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
    del saved
    memory_after_load = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    if run.primary:
        log(
            {
                "event": "model_ready",
                "load_seconds": round(load_seconds),
                "adapted_modules": len(attached),
                "trainable_parameters": sum(p.numel() for p in parameters),
                "gpu_allocated_gb": round(memory_after_load, 2),
                **(
                    {
                        "expert_adapters": len(expert_lora.modules(model.backbone)),
                        "expert_rank": args.expert_rank,
                        "expert_parameters": sum(
                            p.numel() for p in expert_lora.parameters(model.backbone)
                        ),
                    }
                    if args.expert_lora
                    else {}
                ),
            }
        )

    settings = SimpleNamespace(
        batch_size=args.eval_batch_size,
        token_budget=args.eval_token_budget,
        max_length=args.max_length,
    )
    merges = Merges(args, output) if run.primary and merging else None
    augment: Callable[[list[dict], random.Random], list[dict]] = (
        shuffle_options
        if readout
        else functools.partial(pointer.augment, reverse_score_prob=args.reverse_score_prob)
    )
    targets_of = target_matrix if readout else pointer.target_matrix

    def run_eval(step: int) -> None:
        if run.primary:
            result = {
                "step": step,
                "attempt": attempt,
                **dev_eval.evaluate(model, dev_rows, temperature_rows, settings),
            }
            append_jsonl(output / "eval_log.jsonl", result)
            log({"event": "eval", **result})
        run.barrier()

    def records(step: int, examples_seen: int) -> tuple[dict, dict]:
        record = adapter_record(
            args,
            {
                "world_size": run.world,
                "train_rows": len(index),
                "optimizer_steps": total_steps,
                "step": step,
                "examples_seen": examples_seen,
                "data_sha256": hashes,
                "code_sha256": code,
                "plan_sha256": stats["sha256"],
                "dropped_rows": index.dropped,
                "train_limit": args.train_limit,
            },
        )
        state = {
            "step": step,
            "examples_seen": examples_seen,
            "world_size": run.world,
            "plan_sha256": stats["sha256"],
            "data_sha256": hashes,
            "optimizer_steps": total_steps,
            "args": vars(args),
            "attempt": attempt,
        }
        return record, state

    def run_save(step: int, examples_seen: int) -> None:
        began = time.monotonic()
        if run.primary:
            save_step(
                output, step, model, optimizer, *records(step, examples_seen), args.keep_optimizer
            )
            log(
                {
                    "event": "saved",
                    "step": step,
                    "path": str(output / f"step-{step:05d}"),
                    "seconds": round(time.monotonic() - began, 1),
                }
            )
            if merges is not None and (
                step == stop or (args.merge_every and step % args.merge_every == 0)
            ):
                merges.start(step)
        run.barrier()

    if start_step == 0 and not args.no_eval_at_start:
        run_eval(0)

    examples_seen = sum(sum(len(mb) for mb in micro) for _e, micro in plan[:start_step])
    cache_epoch: int | None = None
    cache: dict[int, dict] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step = start_step
    while step < stop:
        epoch = plan[step][0]
        if epoch != cache_epoch:  # this rank's rows for the epoch
            needed = [
                i
                for s in plan
                if s[0] == epoch
                for mb, _real, _j in placement.local_work(s, run.rank)
                for i in mb
            ]
            cache, cache_epoch = index.read(needed), epoch
        began = time.monotonic()
        model.train()
        work = placement.local_work(plan[step], run.rank)
        rows_in_step = sum(len(mb) for mb in plan[step][1])
        torch.manual_seed(args.seed + 100003 * (step + 1) + run.rank)
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(8, dtype=torch.float64, device=run.device)
        cpu_prepare = 0.0
        for slot, (indices, real, j) in enumerate(work):
            rows = [cache[i] for i in indices]
            if not args.no_augment:
                # seeded by the microbatch's plan index j, whatever rank it runs on
                rows = augment(rows, random.Random(args.seed + 100003 * (step + 1) + 1009 * j))
            t_prep = time.monotonic()
            prepared = model.encode(rows, max_length=args.max_length)
            cpu_prepare += time.monotonic() - t_prep
            count = len(rows) if real else 0
            last = slot == len(work) - 1
            sync = contextlib.nullcontext() if last or run.world == 1 else ddp.no_sync()
            with sync:
                logits = ddp(prepared)
                target = targets_of(rows, logits.device)
                parts = training_loss(logits, target, args.brier_weight)
                (parts.total * (count * run.world / rows_in_step)).backward()
            finite = torch.isfinite(parts.total.detach())
            sums += torch.stack(
                [
                    parts.total.detach().double() * count,
                    parts.cross_entropy.detach().double() * count,
                    parts.brier.detach().double() * count,
                    torch.tensor(float(count), device=run.device, dtype=torch.float64),
                    torch.tensor(
                        float(prepared.input_tokens if real else 0),
                        device=run.device,
                        dtype=torch.float64,
                    ),
                    torch.tensor(
                        float(prepared.inputs["input_ids"].numel() if real else 0),
                        device=run.device,
                        dtype=torch.float64,
                    ),
                    (~finite).double(),
                    torch.tensor(1.0 if real else 0.0, device=run.device, dtype=torch.float64),
                ]
            ).nan_to_num(nan=0.0)
            del logits, target, parts, prepared
        run.all_reduce(sums)
        if sums[6] > 0:
            raise RuntimeError(f"step {step + 1}: the loss is NaN or infinite")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"step {step + 1}: the gradient norm is NaN or infinite")
        step += 1
        factor = lr_multiplier(step, total_steps, args.warmup_ratio, args.lr_floor)
        for group, peak in zip(optimizer.param_groups, peaks, strict=True):
            group["lr"] = peak * factor
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.monotonic() - began
        examples_seen += rows_in_step
        peaks_gb = torch.tensor(
            [torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9]
            if torch.cuda.is_available()
            else [0.0, 0.0],
            device=run.device,
        )
        run.all_reduce(peaks_gb, dist.ReduceOp.MAX if run.world > 1 else None)
        if run.primary and (step % args.log_every == 0 or step == stop or step == start_step + 1):
            row_count = float(sums[3])
            value = {
                "step": step,
                "epoch": epoch,
                "loss": float(sums[0]) / row_count,
                "cross_entropy": float(sums[1]) / row_count,
                "brier": float(sums[2]) / row_count,
                "lr": args.lr * factor,
                "grad_norm": grad_norm,
                "rows": int(row_count),
                "microbatches": int(sums[7]),
                "tokens": int(sums[4]),
                "padded_tokens": int(sums[5]),
                "examples_seen": examples_seen,
                "step_seconds": round(seconds, 3),
                "rows_per_s": round(row_count / seconds, 2),
                "tokens_per_s": round(float(sums[4]) / seconds),
                "gpu_peak_allocated_gb": round(float(peaks_gb[0]), 2),
                "gpu_peak_reserved_gb": round(float(peaks_gb[1]), 2),
                "rank0_prepare_s": round(cpu_prepare, 3),
                "elapsed_s": round(time.monotonic() - started),
                "attempt": attempt,
            }
            append_jsonl(output / "train_log.jsonl", value)
            log(value)
        if step % args.eval_every == 0 or step == stop:
            run_eval(step)
        if step % args.save_every == 0 or step == stop:
            run_save(step, examples_seen)
    failed = merges.wait() if merges is not None else []
    if run.primary:
        log(
            {
                "event": "done",
                "step": step,
                "total_steps": total_steps,
                "elapsed_s": round(time.monotonic() - started),
                "last": str((output / "last").resolve()),
            }
        )
    run.barrier()
    run.close()
    if failed:
        raise RuntimeError(f"the merges of steps {failed} failed; see {output}/merged/step-*.log")


# ----------------------------------------------------------------------------------------------
# arguments


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    config = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(config) - set(DEFAULTS) - CONFIG_ONLY - {"base_model", "revision"}
    if unknown:
        raise SystemExit(f"{path}: unknown settings {sorted(unknown)}")
    if isinstance(config.get("targets"), list):
        config["targets"] = ",".join(config["targets"])
    for key, value in config.items():
        default = DEFAULTS.get(key)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise SystemExit(f"{path}: {key} must be true or false, not {value!r}")
        elif isinstance(default, float) and not isinstance(value, bool):
            try:
                config[key] = float(value)
            except (TypeError, ValueError):
                raise SystemExit(f"{path}: {key} must be a number, not {value!r}") from None
        elif isinstance(default, int) and not isinstance(value, bool):
            if not isinstance(value, int):
                raise SystemExit(f"{path}: {key} must be an integer, not {value!r}")
    return config


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", help="a config (training/configs/<name>.yaml); flags override it")
    ap.add_argument(
        "--base",
        help="local directory of the full base checkpoint (the config's base at --revision)",
    )
    ap.add_argument("--base-model", help="the base's Hub id, recorded (from the config)")
    ap.add_argument("--revision", help="the base's 40-character commit, recorded (from the config)")
    ap.add_argument(
        "--train", required=True, help="training rows (training/convert.py's train.jsonl)"
    )
    ap.add_argument("--dev", required=True, help="the dev fold every evaluation scores")
    ap.add_argument(
        "--temperature-data",
        help="optional calibration fold: also report temperature-fitted dev metrics",
    )
    ap.add_argument("--output", help="the run directory")
    ap.add_argument("--head", choices=HEADS, help="the readout: token_readout (default) or pointer")
    ap.add_argument(
        "--prompt",
        choices=PROMPTS,
        help="strands-chat or strands-plain for the token readout, numbered for the pointer head",
    )
    ap.add_argument(
        "--train-limit", type=int, help="use only the first N lines of --train (smoke runs)"
    )
    ap.add_argument("--dev-limit", type=int, help="evaluate only the first N dev rows")
    ap.add_argument(
        "--allow-family-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow training rows that share a (dataset, family) with a held-out row",
    )
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--rows-per-step", type=int, help="rows per optimizer step (default 3,072)")
    ap.add_argument(
        "--accum", type=int, help="microbatches per GPU per step (default: from --rows-per-step)"
    )
    ap.add_argument("--batch-size", type=int, help="max rows per microbatch")
    ap.add_argument("--token-budget", type=int, help="max padded tokens per microbatch")
    ap.add_argument(
        "--max-length", type=int, help="rows longer than this (prompt + 16) are dropped"
    )
    ap.add_argument("--eval-batch-size", type=int)
    ap.add_argument("--eval-token-budget", type=int)
    ap.add_argument(
        "--sort-window", type=int, help="rows sorted together (default world * accum * batch)"
    )
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup-ratio", type=float)
    ap.add_argument(
        "--lr-floor",
        type=float,
        help="where the cosine ends, as a fraction of the peak (default 0.1)",
    )
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--brier-weight", type=float)
    ap.add_argument("--max-grad-norm", type=float)
    ap.add_argument("--rank", type=int, help="LoRA rank")
    ap.add_argument("--alpha", type=float, help="LoRA alpha")
    ap.add_argument("--dropout", type=float, help="LoRA dropout")
    ap.add_argument("--targets", help="comma-separated linear names to adapt")
    ap.add_argument(
        "--no-augment",
        dest="no_augment",
        action="store_true",
        default=None,
        help="no augmentation (option order, phrasing)",
    )
    ap.add_argument(
        "--augment",
        dest="no_augment",
        action="store_false",
        default=None,
        help="augmentation (the default)",
    )
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--no-eval-at-start", action="store_true")
    ap.add_argument(
        "--merge-every",
        type=int,
        help="also merge every N-th saved step (the last is always merged)",
    )
    ap.add_argument(
        "--no-merge",
        dest="no_merge",
        action="store_true",
        default=None,
        help="do not merge any checkpoint",
    )
    ap.add_argument(
        "--merge",
        dest="no_merge",
        action="store_false",
        default=None,
        help="merge the last checkpoint (the default for the token readout)",
    )
    ap.add_argument("--log-every", type=int)
    ap.add_argument(
        "--keep-optimizer", type=int, help="optimizer states kept (adapters are all kept)"
    )
    ap.add_argument(
        "--resume", action="store_true", help="continue from --output/last if it exists"
    )
    ap.add_argument("--resume-from", help="continue from this step directory")
    ap.add_argument(
        "--stop-after",
        type=int,
        help="stop (and save) at this step; the schedule keeps its length",
    )
    ap.add_argument("--bucket-mb", type=int, help="DDP's gradient bucket size, on several GPUs")
    ap.add_argument(
        "--attn-split",
        type=int,
        help="token readout: microbatches at least this wide run the prefix_bidir attention as "
        "two unmasked attentions per row; 0 = off",
    )
    ap.add_argument(
        "--expert-lora",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="token readout: also a LoRA on every routed expert of a mixture-of-experts decoder (the "
        "26B-A4B)",
    )
    ap.add_argument(
        "--expert-rank", type=int, help="per-expert rank (default max(1, --rank // top_k))"
    )
    ap.add_argument("--expert-alpha", type=float, help="per-expert alpha (default --alpha)")
    ap.add_argument("--pointer-dim", type=int, help="pointer head: its query and key width")
    ap.add_argument("--head-dropout", type=float, help="pointer head: dropout before the head")
    ap.add_argument("--head-lr", type=float, help="pointer head: the head's learning rate")
    ap.add_argument(
        "--force-bos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pointer head: start every prompt with the tokenizer's BOS (Gemma 4 -it bases)",
    )
    ap.add_argument(
        "--reverse-score-prob",
        type=float,
        help="pointer head: the chance a score question's levels are listed in reverse",
    )
    ap.add_argument("--cpu-threads", type=int)
    ap.add_argument(
        "--plan-only", action="store_true", help="print the plan's statistics for --world-size GPUs"
    )
    ap.add_argument("--world-size", type=int, default=8, help="for --plan-only")
    return ap


def resolve(argv: list[str] | None = None) -> argparse.Namespace:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    for key, default in DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, config.get(key, default))
    for key in ("base_model", "revision"):
        if getattr(args, key) is None:
            setattr(args, key, config.get(key))
    args.tokenizer_sha256 = config.get("tokenizer_sha256")
    args.expected_base_sha256 = config.get("base_sha256")
    args.no_eval_at_start = bool(args.no_eval_at_start)
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.eval_token_budget = args.eval_token_budget or args.token_budget
    if (
        args.token_budget < 1
        or args.max_length < 1
        or args.batch_size < 1
        or args.epochs <= 0
        or args.accum < 0
        or args.rows_per_step < 1
        or args.save_every < 1
        or args.eval_every < 1
        or args.log_every < 1
    ):
        raise SystemExit(
            "batch size, token budget, max length, rows per step, epochs and the save, eval and log "
            "intervals must be positive, accum >= 0"
        )
    if args.head not in HEADS or args.prompt not in PROMPTS:
        raise SystemExit(
            f"head {args.head!r} or prompt {args.prompt!r} is not one the trainer knows"
        )
    if (args.head == "pointer") != (args.prompt == pointer.PROMPT):
        raise SystemExit(
            f"the {args.head} head takes the prompt "
            f"{pointer.PROMPT if args.head == 'pointer' else ' or '.join(TEMPLATES)}, not {args.prompt!r}"
        )
    if not 0 < args.warmup_ratio < 1:
        raise SystemExit("--warmup-ratio must be in (0, 1)")
    if not 0 <= args.lr_floor <= 1:
        raise SystemExit("--lr-floor must be in [0, 1]")
    if args.head == "pointer":
        if args.expert_lora or args.attn_split:
            raise SystemExit("--expert-lora and --attn-split are token-readout options")
        if not 0 <= args.reverse_score_prob <= 1:
            raise SystemExit("--reverse-score-prob must be in [0, 1]")
    if args.expert_lora:
        if args.base:
            from .model import text_config

            args.expert_rank = args.expert_rank or expert_lora.default_rank(
                args.rank, text_config(args.base)
            )
        args.expert_alpha = args.alpha if args.expert_alpha is None else args.expert_alpha
    else:
        args.expert_rank, args.expert_alpha = 0, None
    if args.base:
        args.base = str(Path(args.base).resolve())
    args.base_sha256 = base_digest(args.base) if args.base and not args.plan_only else None
    return args


def main(argv: list[str] | None = None) -> None:
    args = resolve(argv)
    if args.plan_only:
        return plan_only(args)
    if not (args.base and args.base_model and args.revision and args.output):
        raise SystemExit(
            "--base, --base-model, --revision (or a --config naming them) and --output are required"
        )
    from .model import check_revision

    check_revision(args.revision)
    if args.expected_base_sha256 and args.base_sha256 != args.expected_base_sha256:
        raise SystemExit(
            f"{args.base} is not {args.base_model} at {args.revision}: its digest differs from the "
            "config's base_sha256 (another size, revision or a modified copy)"
        )
    tokenizer = Path(args.base) / "tokenizer.json"
    if args.tokenizer_sha256 and sha256_file(tokenizer) != args.tokenizer_sha256:
        raise SystemExit(
            f"{tokenizer} is not the tokenizer of {args.base_model} at {args.revision} (sha256 "
            f"{args.tokenizer_sha256[:12]}...): download the base at the config's revision"
        )
    train(args)


if __name__ == "__main__":
    main()
