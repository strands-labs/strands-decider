"""Export a token-readout checkpoint (or a soup of them) as a decision checkpoint.

    python -m training.export_v2 <checkpoint> [<checkpoint> ...] <out> [--base-model ...]

Several checkpoints of one recipe (other seeds) give their soup: each one merged into the
base, then every merged tensor, the readout included, averaged with equal weights in float32
and written in bf16. A soup of one is the single export. (A soup checkpoint written by
training.soup, whose stacked adapters average the seeds' updates, exports as one checkpoint.)

The output is docs/spec.md's checkpoint layout (section 9): the base model's own multimodal
checkpoint with the fine-tuning merged in, and this interface's files beside it.

- The base's safetensors shards, under the base's names and file names. Each decoder tensor
  (`model.language_model.*`) that the LoRA adapter or the expert adapters (expert_lora.py)
  change gets `W + s * B @ A`, summed in float32 and written in bf16. Every other tensor,
  the encoders' included, is the base's, byte for byte.
- The base's `config.json` (its soft-cap set to the checkpoint's `readout_softcap`),
  `processor_config.json`, `tokenizer.json` and `tokenizer_config.json`, without any chat
  template.
- `readout.safetensors` (`weight`, [255, hidden], bf16) and `decision_config.json`, with the
  spec's keys only (section 8).
- `training.json`: the training record (recipe, LoRA settings, steps, soup size). It holds
  no local paths.

Before the merge, the base on disk must be the one the run trained on: the digest of its
config, tokenizer and tensor headers is compared with the `base.json` that training wrote.
After the write, the output is read back with the core's loaders and scanned for local
paths. If any check fails, the output directory is deleted and the export raises.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
from collections.abc import Callable
from functools import partial, reduce
from typing import Any

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from strands_decider import core, expert_lora
from strands_decider.modeling import StrandsDeciderConfig, config_path, load_head_state
from training.soup import check_one_recipe

DECODER = "model.language_model."
PEFT_PREFIX = "base_model.model."
BASE_RECORD = "base.json"  # written by training beside train_config.json
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
BASE_FILES = ("config.json", "processor_config.json", *TOKENIZER_FILES)
_HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+")


# ---- the base model ---------------------------------------------------------------------


def base_dir(base_model: str, revision: str | None) -> str:
    """The base's files on disk: a local directory, or its Hub snapshot in the local cache."""
    if os.path.isdir(base_model):
        return base_model
    return snapshot_download(base_model, revision=revision, local_files_only=True)


def _shards(d: str) -> list[str]:
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(index):
        return sorted(set(_json(index)["weight_map"].values()))
    return ["model.safetensors"]


def _header(path: str) -> bytes:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)


def base_digest(d: str) -> str:
    """sha256 of the base's config, tokenizer files and tensor headers (names, dtypes,
    shapes and offsets): what a merge into it depends on, without reading the weights."""
    h = hashlib.sha256()
    for name in ("config.json", *TOKENIZER_FILES, *_shards(d)):
        p = os.path.join(d, name)
        h.update(name.encode() + b"\0")
        h.update(_header(p) if name.endswith(".safetensors") else open(p, "rb").read())
    return h.hexdigest()


def write_base_record(out: str, base_model: str, revision: str | None) -> bool:
    """Write `base.json` (the base's digest) into `out`, if the base is on disk."""
    try:
        d = base_dir(base_model, revision)
        digest = base_digest(d)
    except (OSError, ValueError):  # not on disk (a Hub id not cached, a test stand-in)
        return False
    with open(os.path.join(out, BASE_RECORD), "w", encoding="utf-8") as fh:
        json.dump({"digest": digest}, fh, indent=2)
    return True


# ---- the merge --------------------------------------------------------------------------


def _json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _lora_delta(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    return scale * (b.double() @ a.double()).float()


def adapter_deltas(
    checkpoint: str, cfg: StrandsDeciderConfig
) -> dict[str, Callable[[], torch.Tensor]]:
    """Base tensor name -> the float32 update s * B @ A, from the checkpoint's adapters.
    Each update is computed when called, so that only one full-size update is held at a time."""
    out: dict[str, Callable[[], torch.Tensor]] = {}
    if not cfg.use_lora:
        return out
    lora = os.path.join(checkpoint, "lora")
    acfg = _json(os.path.join(lora, "adapter_config.json"))
    if (
        any(
            acfg.get(k)
            for k in (
                "use_dora",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
                "target_parameters",
            )
        )
        or acfg.get("bias", "none") != "none"
    ):
        raise ValueError("the export merges plain LoRA only")
    r = acfg["r"]
    scale = acfg["lora_alpha"] / (math.sqrt(r) if acfg.get("use_rslora") else r)
    tensors = load_file(os.path.join(lora, "adapter_model.safetensors"))
    for key, a in tensors.items():
        if ".lora_B." in key:
            continue
        if not key.startswith(PEFT_PREFIX) or not key.endswith(".lora_A.weight"):
            raise ValueError(f"{key}: not a LoRA A matrix of a linear layer")
        module = key[len(PEFT_PREFIX) : -len(".lora_A.weight")]
        b = tensors[f"{PEFT_PREFIX}{module}.lora_B.weight"]
        out[f"{DECODER}{module}.weight"] = partial(_lora_delta, a, b, scale)
    if cfg.expert_lora:
        e = load_file(os.path.join(lora, expert_lora.FILE))
        alpha = cfg.lora_alpha if cfg.expert_lora_alpha is None else cfg.expert_lora_alpha
        s = alpha / cfg.expert_lora_r
        for key in e:
            if key.endswith(".gate_up_a") or key.endswith(".down_a"):
                part = key.rsplit(".", 1)[1][: -len("_a")]
                module = key[len(PEFT_PREFIX) : -len(f".{part}_a")]
                delta = partial(expert_lora.delta, e[key], e[f"{key[:-2]}_b"], s)
                out[f"{DECODER}{module}.{part}_proj"] = delta
    return out


def merge_shards(base: str, out: str, deltas: list[dict[str, Callable[[], torch.Tensor]]]) -> None:
    """Write the base's shards to `out`, each decoder tensor that `deltas` (one dict per seed)
    change being the float32 mean of the seeds' merged tensors W + delta_i.

    That mean is W + mean(delta_i); it is summed here as merged tensors, one tensor and one
    seed's update at a time, so that it is exactly the float32 average of the merged models
    (and one seed's export is W + delta, as before)."""
    left = {name for d in deltas for name in d}
    for shard in _shards(base):
        tensors = {}
        with safe_open(os.path.join(base, shard), "pt") as f:
            metadata = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if name in left:
                    left.discard(name)
                    w = t.float()
                    acc = torch.zeros_like(w)  # 0 + x is x exactly: the sum is the seeds' own
                    for seed in deltas:
                        d = seed[name]() if name in seed else torch.zeros_like(w)
                        if d.shape != t.shape:
                            raise ValueError(f"{name}: adapter update {d.shape}, base {t.shape}")
                        acc = acc + (w + d)
                    t = acc / len(deltas)
                if name.startswith(DECODER):
                    t = t.to(torch.bfloat16)
                tensors[name] = t.contiguous()
        save_file(tensors, os.path.join(out, shard), metadata=metadata)
    if left:
        raise ValueError(f"adapter updates for tensors the base does not have: {sorted(left)[:3]}")
    index = os.path.join(base, "model.safetensors.index.json")
    if os.path.exists(index):
        shutil.copy2(index, out)


# ---- the files beside the weights -------------------------------------------------------


def _no_chat_template(src: str, dst: str) -> None:
    d = _json(src)
    d.pop("chat_template", None)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)


def _model_config(src: str, dst: str, softcap: float) -> None:
    d = _json(src)
    for c in (d.get("text_config"), d):
        if isinstance(c, dict) and "final_logit_softcapping" in c:
            c["final_logit_softcapping"] = softcap
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


def _no_paths(v: Any) -> Any:
    """`v` with every string that is an absolute path cut to its last component."""
    if isinstance(v, dict):
        return {k: _no_paths(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_no_paths(x) for x in v]
    if isinstance(v, str) and os.path.isabs(v):
        return os.path.basename(v.rstrip("/"))
    return v


def training_record(checkpoint: str, cfg: StrandsDeciderConfig, soup_of: int = 1) -> dict[str, Any]:
    """The run's recipe; a soup's (whose adapters are the runs' stacked, or `soup_of` seeds
    averaged by the export) gives one run's."""
    history = _optional_json(os.path.join(checkpoint, "history.json")) or []
    soup = _optional_json(os.path.join(checkpoint, "soup.json"))
    run = _optional_json(os.path.join(checkpoint, "train_config.json")) or {}

    n = soup_of * (len(soup["checkpoints"]) if soup else 1)

    def per_run(key: str) -> Any:
        return run.get(key, getattr(cfg, key))

    record: dict[str, Any] = _no_paths(
        {
            "trainer": "strands-decider training.train",
            "lora": {
                "rank": per_run("lora_r"),
                "alpha": per_run("lora_alpha"),
                "dropout": cfg.lora_dropout,
                "targets": sorted(cfg.lora_targets),
            }
            if cfg.use_lora
            else None,
            "expert_lora": {
                "rank": per_run("expert_lora_r"),
                "alpha": per_run("expert_lora_alpha"),
                "targets": ["experts.gate_up_proj", "experts.down_proj"],
                "router": "frozen",
            }
            if cfg.expert_lora
            else None,
            "merge": "float32 sum, saved bfloat16",
            "soup_of": n if soup or n > 1 else None,
            "optimizer_steps": max((h["step"] for h in history if "step" in h), default=None),
            "train_config": run or None,
        }
    )
    return record


def _optional_json(path: str) -> Any:
    return _json(path) if os.path.exists(path) else None


def check_no_local_paths(out: str, paths: list[str]) -> None:
    """No file in `out` holds any of `paths` (text files, and safetensors headers), and
    `out` holds no code."""
    needles = sorted({os.path.abspath(p).encode() for p in paths if p}, key=len, reverse=True)
    for name in os.listdir(out):
        p = os.path.join(out, name)
        if name.endswith(".py"):
            raise ValueError(f"{name}: a checkpoint carries no code")
        data = _header(p) if name.endswith(".safetensors") else open(p, "rb").read()
        for n in needles:
            if n in data:
                raise ValueError(f"{name} holds the local path {n.decode()}")


# ---- the export -------------------------------------------------------------------------


def export_v2(
    checkpoint: str | list[str],
    out: str,
    *,
    base_model: str | None = None,
    max_length_validated: int | None = None,
    unrecorded_base: bool = False,
) -> str:
    """Write `checkpoint`'s decision checkpoint to `out` (absent or empty); returns `out`.

    A list of checkpoints of one recipe (other seeds) writes their soup: each merged, every
    tensor and the readout averaged with equal weights. Its temperature is 1: calibrate it.

    `base_model` is the base's Hub id for `provenance` (default: the checkpoint's base_model,
    which must then be a Hub id). `max_length_validated` defaults to the run's max_length.
    `unrecorded_base` skips the base digest check for a run that wrote no base.json."""
    paths = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    if not paths:
        raise ValueError("no checkpoints")
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"{out}: not empty")
    if len(paths) > 1:  # same config beyond temperatures, base revision and files, adapters
        check_one_recipe(paths)
    checkpoint = paths[0]
    cfg = StrandsDeciderConfig.from_json(config_path(checkpoint))
    if (cfg.head_type, cfg.attention_mode) != ("token_readout", "prefix_bidir"):
        raise ValueError(
            "a decision checkpoint is a token_readout checkpoint under prefix_bidir, "
            f"not {cfg.head_type} / {cfg.attention_mode}"
        )
    if len(paths) == 1 and cfg.temperature_by_kind:
        raise ValueError(
            "decision_config.json holds one temperature; this checkpoint has per-kind ones"
        )
    hub_id = base_model or cfg.base_model
    if not _HUB_ID.fullmatch(hub_id) or os.path.isdir(hub_id):
        raise ValueError(f"{hub_id}: provenance needs the base's Hub id; pass --base-model")
    base = base_dir(cfg.base_model, cfg.base_revision)
    revision = cfg.base_revision or os.path.basename(os.path.normpath(base))
    record = _optional_json(os.path.join(checkpoint, BASE_RECORD))
    if record is not None:
        if record["digest"] != base_digest(base):
            raise ValueError(
                f"{cfg.base_model}@{revision} is not the base this checkpoint was trained on "
                "(its config, tokenizer or tensor headers differ)"
            )
    elif not unrecorded_base:
        raise ValueError(
            f"{checkpoint}: no {BASE_RECORD}; pass --unrecorded-base to skip the check"
        )

    decision = {
        "interface": core.INTERFACE,
        "format_version": core.FORMAT_VERSION,
        "prompt": core.CHAT,
        "inputs": ["text"],
        "codes": list(core.CODES),
        "token_ids": list(core.CODE_TOKEN_IDS),
        "temperature": cfg.temperature if len(paths) == 1 else 1.0,
        "final_logit_softcapping": cfg.readout_softcap,
        "attention_mode": cfg.attention_mode,
        "pooling": "last",
        "max_length_validated": max_length_validated or cfg.max_length,
        "provenance": {"base_model": hub_id, "base_revision": revision},
    }
    softcap = core.DecisionConfig.from_dict(decision).final_logit_softcapping  # before any write
    heads = [load_head_state(p)["proj.weight"].float() for p in paths]
    weight = reduce(torch.add, heads) / len(paths)
    deltas = [adapter_deltas(p, cfg) for p in paths]  # one recipe: the first's settings

    os.makedirs(out, exist_ok=True)
    try:
        merge_shards(base, out, deltas)
        save_file(
            {"weight": weight.to(torch.bfloat16).contiguous()},
            os.path.join(out, core.readout.READOUT_FILE),
            metadata={"format": "pt"},
        )
        _model_config(os.path.join(base, "config.json"), os.path.join(out, "config.json"), softcap)
        for name in BASE_FILES[1:]:
            if os.path.exists(os.path.join(base, name)):
                _no_chat_template(os.path.join(base, name), os.path.join(out, name))
        with open(os.path.join(out, core.config.FILENAME), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(decision, indent=2) + "\n")
        with open(os.path.join(out, "training.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(training_record(checkpoint, cfg, len(paths)), indent=2) + "\n")

        loaded = core.DecisionConfig.load(out)
        core.Readout.load(out, loaded)  # also: config.json's soft-cap equals decision_config's
        core.load_tokenizer(out, loaded)
        check_no_local_paths(out, [*paths, base, out, os.path.expanduser("~")])
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return out


def _cli(
    checkpoints: list[str] = typer.Argument(
        ..., help="A token_readout checkpoint or soup; several of one recipe: their soup."
    ),
    out: str = typer.Argument(..., help="Directory for the decision checkpoint: absent or empty."),
    base_model: str = typer.Option(None, help="The base's Hub id, for provenance."),
    max_length_validated: int = typer.Option(None, help="Default: the run's max_length."),
    unrecorded_base: bool = typer.Option(False, help="Skip the base digest check (no base.json)."),
) -> None:
    """Merge the adapters into the base (or average several merged seeds) and write the
    decision checkpoint layout."""
    path = export_v2(
        checkpoints,
        out,
        base_model=base_model,
        max_length_validated=max_length_validated,
        unrecorded_base=unrecorded_base,
    )
    print(f"decision checkpoint -> {path}")


if __name__ == "__main__":  # pragma: no cover
    typer.run(_cli)
