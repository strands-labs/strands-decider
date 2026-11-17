"""Export a token-readout checkpoint (or a soup of them) as a decision checkpoint.

    python -m training.export_v2 <checkpoint> [<checkpoint> ...] <out> [--base-model ...]

Several checkpoints of one recipe (runs that may differ in seed, data, learning rate or
steps) give their soup: each one merged into the base, then every merged tensor, the readout
included, averaged with equal weights in float32 and written in bf16. A soup of one is the
single export. (A soup checkpoint written by training.soup, whose stacked adapters average
the runs' updates, exports as one checkpoint.)

The output is docs/spec.md's checkpoint layout (section 9): the base model's own multimodal
checkpoint with the fine-tuning merged in, and this interface's files beside it.

- The base's safetensors shards, under the base's names and file names. Each decoder tensor
  (`model.language_model.*`) that the LoRA adapter or the expert adapters (expert_lora.py)
  change gets `W + s * B @ A`, summed in float32 and written in bf16. Every other tensor,
  the encoders' included, is the base's, byte for byte.
- The base's `config.json` and `tokenizer.json`, byte for byte, and its
  `processor_config.json` and `tokenizer_config.json`, without any chat template. The export
  refuses a base whose `config.json` caps its logits at another value than the checkpoint's
  `readout_softcap`.
- `readout.safetensors` (`weight`, [255, hidden], bf16) and `decision_config.json`, with the
  spec's keys only (section 8). `inputs` is text, and also image when the base has a vision
  encoder and its projection. The export then refuses a base whose `processor_config.json`
  does not size images as the core counts them (`core.check_image_processor`).
- `LICENSE` (the Apache License 2.0), a `NOTICE` that names the base and its revision, what
  changed and which parts are the base's, and `README.md`, the model card: the file given
  as `--model-card`, else one written from the runs' records (LoRA settings, and each run's
  seed, data file names and steps).

The output holds these files and no others. `golden_vectors.jsonl` (section 10) is not
written here.

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
import textwrap
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
from training.soup import check_one_recipe, member_records

DECODER = "model.language_model."
PEFT_PREFIX = "base_model.model."
BASE_RECORD = "base.json"  # written by training beside train_config.json
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
BASE_FILES = ("config.json", "processor_config.json", *TOKENIZER_FILES)
LICENSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "LICENSE")
# The parts of a base beside its decoder (docs/spec.md, section 9), by tensor name prefix.
ENCODERS = {
    "model.vision_tower.": "vision encoder",
    "model.vision_embedder.": "vision encoder",
    "model.embed_vision.": "vision projection",
    "model.audio_tower.": "audio encoder",
    "model.embed_audio.": "audio projection",
}
_HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+")


# ---- the base model ---------------------------------------------------------------------


def base_dir(base_model: str, revision: str | None) -> str:
    """The base's files on disk: a local directory, or its Hub snapshot in the local cache."""
    if os.path.isdir(base_model):
        return base_model
    snapshot: str = snapshot_download(base_model, revision=revision, local_files_only=True)
    return snapshot


def _shards(d: str) -> list[str]:
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(index):
        return sorted(set(_json(index)["weight_map"].values()))
    return ["model.safetensors"]


def _header(path: str) -> bytes:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n)


def tensor_names(d: str) -> set[str]:
    """The names of a base's tensors, from its shards' headers."""
    names: set[str] = set()
    for shard in _shards(d):
        with safe_open(os.path.join(d, shard), "pt") as f:
            names.update(f.keys())
    return names


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


def drop_kv_shared_updates(
    base_config: dict[str, Any], deltas: list[dict[str, Callable[[], torch.Tensor]]]
) -> None:
    """Take the updates of the KV-shared layers' own k_proj, v_proj and k_norm out of `deltas`,
    so that the export writes the base's tensors for them; refuse one that is not zero.

    The last `num_kv_shared_layers` layers of a Gemma 4 E2B or E4B decoder attend with an
    earlier layer's keys and values, so no runtime reads their own. transformers builds no
    such modules for those layers, and an adapter has nothing to train there. An update that
    is not zero means the run trained weights that a runtime never uses."""
    text = base_config.get("text_config")
    text = text if isinstance(text, dict) else base_config
    layers, shared = text.get("num_hidden_layers") or 0, text.get("num_kv_shared_layers") or 0
    for i in range(layers - shared, layers):
        for part in ("k_proj", "v_proj", "k_norm"):
            name = f"{DECODER}layers.{i}.self_attn.{part}.weight"
            for run in deltas:
                update = run.pop(name, None)
                if update is not None and bool(update().any()):
                    raise ValueError(
                        f"{name}: layer {i} shares an earlier layer's keys and values, so a "
                        "runtime never reads this tensor, but the adapter changes it"
                    )


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


def _without_chat_template(src: str, dst: str) -> None:
    """Copy a JSON file of the base: byte for byte, or rewritten without its `chat_template`."""
    d = _json(src)
    if "chat_template" not in d:
        shutil.copyfile(src, dst)
        return
    del d["chat_template"]
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)


def _base_softcaps(config: dict[str, Any]) -> set[float]:
    """Every `final_logit_softcapping` a base's config.json gives (text_config, top level)."""
    return {
        float(c["final_logit_softcapping"])
        for c in (config.get("text_config"), config)
        if isinstance(c, dict) and c.get("final_logit_softcapping") is not None
    }


def _optional_json(path: str) -> Any:
    return _json(path) if os.path.exists(path) else None


def training_record(paths: list[str], cfg: StrandsDeciderConfig) -> dict[str, Any]:
    """What trained the checkpoints, for the model card: the LoRA settings of one run and
    every run's record (training.soup.member_records). `runs` counts the averaged runs: those
    of a soup checkpoint, whose adapters are its runs' stacked, times the checkpoints here."""
    run = _optional_json(os.path.join(paths[0], "train_config.json")) or {}

    def per_run(key: str) -> Any:  # a stacked soup's config holds N times the rank and alpha
        return run.get(key, getattr(cfg, key))

    soups = [_optional_json(os.path.join(p, "soup.json")) for p in paths]
    return {
        "lora": {
            "rank": per_run("lora_r"),
            "alpha": per_run("lora_alpha"),
            "dropout": cfg.lora_dropout,
            "targets": sorted(cfg.lora_targets),
        }
        if cfg.use_lora
        else None,
        "expert_lora": {"rank": per_run("expert_lora_r"), "alpha": per_run("expert_lora_alpha")}
        if cfg.expert_lora
        else None,
        "runs": sum(len(s["checkpoints"]) if s else 1 for s in soups),
        "members": [m for p in paths for m in member_records(p)],
    }


def notice(hub_id: str, revision: str, runs: int, names: set[str]) -> str:
    """The NOTICE: the base and its revision, what changed, and which parts are the base's."""
    parts = list(dict.fromkeys(part for prefix, part in ENCODERS.items() if _any(names, prefix)))
    kept = (
        f"The {', the '.join(parts)} and every other tensor outside the text decoder are "
        "those of the base model, unchanged."
        if parts
        else "Every tensor outside the text decoder is that of the base model, unchanged."
    )
    kept = textwrap.fill(kept, 92)
    averaged = f"\n  The weights are the average of {runs} such merged models." if runs > 1 else ""
    return f"""Strands Decider decision checkpoint

This checkpoint is a modified copy of {hub_id}, revision
{revision}.

What changed:
- Low-rank adapters (LoRA), trained by supervised fine-tuning, are merged into the weights
  of the text decoder ({DECODER}*).{averaged}
- A decision readout of 255 codes (readout.safetensors) is in the place of the output layer.
- The decoder attends prefix-bidirectionally on every layer, with no window.

{kept}
"""


def _any(names: set[str], prefix: str) -> bool:
    return any(n.startswith(prefix) for n in names)


def model_card(hub_id: str, revision: str, temperature: float, record: dict[str, Any]) -> str:
    """The default README.md: what was trained, from the runs' records."""
    lines = [
        "# Strands Decider decision checkpoint",
        "",
        f"Base model: `{hub_id}`, revision `{revision}`.",
        "",
        "A decision model: it reads a state and a question, and gives one probability for each "
        "option. Written by `python -m training.export_v2` of strands-decider, in the "
        "checkpoint layout of its specification (`docs/spec.md`, section 9).",
        "",
        "## What was trained",
        "",
        "- A token readout of 255 codes, read at the last prompt token, with the `strands-chat` "
        "prompt and prefix-bidirectional attention.",
    ]
    lora, experts = record["lora"], record["expert_lora"]
    if lora:
        lines.append(
            f"- LoRA adapters of rank {lora['rank']}, alpha {lora['alpha']} and dropout "
            f"{lora['dropout']} on {', '.join(f'`{t}`' for t in lora['targets'])}."
        )
    if experts:
        lines.append(
            f"- LoRA adapters of rank {experts['rank']}, alpha {experts['alpha']} on the "
            "gate, up and down projections of every expert. The router was not trained."
        )
    lines.append(
        "- The adapters are added to the base weights in float32 and saved in bfloat16."
        + (
            f" The checkpoint is the average, with equal weights, of {record['runs']} runs."
            if record["runs"] > 1
            else ""
        )
    )
    lines.append(f"- Temperature: {temperature}.")
    lines += [
        "",
        "| run | seed | training files | optimizer steps | learning rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, m in enumerate(record["members"], 1):
        files = ", ".join(m.get("train_files") or []) or "-"
        cells = [m.get("seed"), files, m.get("optimizer_steps"), m.get("lr")]
        lines.append(f"| {i} | " + " | ".join("-" if c is None else str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


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
    temperature: float | None = None,
    model_card_path: str | None = None,
) -> str:
    """Write `checkpoint`'s decision checkpoint to `out` (absent or empty); returns `out`.

    A list of checkpoints of one recipe writes their soup: each merged, every tensor and the
    readout averaged with equal weights.

    `base_model` is the base's Hub id for `provenance` (default: the checkpoint's base_model,
    which must then be a Hub id). `max_length_validated` defaults to the run's max_length.
    `unrecorded_base` skips the base digest check for a run that wrote no base.json.
    `temperature` is the one that decision_config.json holds, for a value fitted on the
    exported model. The default is the checkpoint's global temperature (its per-type ones are
    not carried over), and 1 for several checkpoints. `model_card_path` is the README.md to
    write; the default is `model_card`."""
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
    if temperature is None:
        temperature = cfg.temperature if len(paths) == 1 else 1.0
        if len(paths) == 1 and cfg.temperature_by_kind:
            print(
                f"[strands-decider] export: decision_config.json holds one temperature, the "
                f"checkpoint's global one ({cfg.temperature}). Its per-type temperatures "
                f"{cfg.temperature_by_kind} are not carried over; pass --temperature for a "
                "value fitted on the exported model."
            )
    hub_id = base_model or cfg.base_model
    if not _HUB_ID.fullmatch(hub_id) or os.path.isdir(hub_id):
        raise ValueError(f"{hub_id}: provenance needs the base's Hub id; pass --base-model")
    base = base_dir(cfg.base_model, cfg.base_revision)
    missing = [n for n in BASE_FILES if not os.path.exists(os.path.join(base, n))]
    if missing:
        raise ValueError(
            f"{base} has no {', '.join(missing)}: a decision checkpoint carries the base's; "
            "download all of the base's files at this revision"
        )
    revision = cfg.base_revision or os.path.basename(os.path.normpath(base))
    # A base with a vision encoder and its projection takes images through them: they are the
    # base's, unchanged. Its image processor must then size images as the core counts them.
    names = tensor_names(base)
    inputs = ["text"]
    if (_any(names, "model.vision_tower.") or _any(names, "model.vision_embedder.")) and _any(
        names, "model.embed_vision."
    ):
        core.check_image_processor(os.path.join(base, "processor_config.json"))
        inputs.append("image")
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
        "inputs": inputs,
        "codes": list(core.CODES),
        "token_ids": list(core.CODE_TOKEN_IDS),
        "temperature": temperature,
        "final_logit_softcapping": cfg.readout_softcap,
        "attention_mode": cfg.attention_mode,
        "pooling": "last",
        "max_length_validated": max_length_validated or cfg.max_length,
        "provenance": {"base_model": hub_id, "base_revision": revision},
    }
    softcap = core.DecisionConfig.from_dict(decision).final_logit_softcapping  # before any write
    base_config = _json(os.path.join(base, "config.json"))
    if _base_softcaps(base_config) - {softcap}:  # config.json is copied as it is
        raise ValueError(
            f"{cfg.base_model}'s config.json caps its logits at "
            f"{sorted(_base_softcaps(base_config))}; the checkpoint's readout_softcap is {softcap}"
        )
    heads = [load_head_state(p)["proj.weight"].float() for p in paths]
    weight = reduce(torch.add, heads) / len(paths)
    deltas = [adapter_deltas(p, cfg) for p in paths]  # one recipe: the first's settings
    drop_kv_shared_updates(base_config, deltas)
    trained = training_record(paths, cfg)

    os.makedirs(out, exist_ok=True)
    try:
        merge_shards(base, out, deltas)
        save_file(
            {"weight": weight.to(torch.bfloat16).contiguous()},
            os.path.join(out, core.readout.READOUT_FILE),
            metadata={"format": "pt"},
        )
        for name in ("config.json", "tokenizer.json"):  # byte for byte: their sha256 is pinned
            shutil.copyfile(os.path.join(base, name), os.path.join(out, name))
        for name in ("processor_config.json", "tokenizer_config.json"):
            _without_chat_template(os.path.join(base, name), os.path.join(out, name))
        with open(os.path.join(out, core.config.FILENAME), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(decision, indent=2) + "\n")
        shutil.copyfile(LICENSE_FILE, os.path.join(out, "LICENSE"))
        with open(os.path.join(out, "NOTICE"), "w", encoding="utf-8") as fh:
            fh.write(notice(hub_id, revision, trained["runs"], names))
        if model_card_path:
            shutil.copyfile(model_card_path, os.path.join(out, "README.md"))
        else:
            with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
                fh.write(model_card(hub_id, revision, temperature, trained))

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
    temperature: float = typer.Option(
        None, help="Default: the checkpoint's global temperature; 1 for several checkpoints."
    ),
    model_card: str = typer.Option(
        None, help="The README.md to write. Default: a card written from the runs' records."
    ),
) -> None:
    """Merge the adapters into the base (or average several merged seeds) and write the
    decision checkpoint layout."""
    path = export_v2(
        checkpoints,
        out,
        base_model=base_model,
        max_length_validated=max_length_validated,
        unrecorded_base=unrecorded_base,
        temperature=temperature,
        model_card_path=model_card,
    )
    print(f"decision checkpoint -> {path}")


if __name__ == "__main__":  # pragma: no cover
    typer.run(_cli)
