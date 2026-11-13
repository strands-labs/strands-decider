"""A trained adapter folded into the base checkpoint: a merged checkpoint.

    python -m training.merge --base <full base checkpoint dir> --adapter <run>/step-NNNNN/adapter --out <dir>

The merged checkpoint is the base's own checkpoint (its config, tokenizer and processor files, its shards under their
names) with every adapted language-model weight replaced by the merged one, plus `readout.safetensors` (the trained
readout, `weight` [255, hidden], bf16) and `training.json` (the adapter's training record and the decision fields the
export needs). transformers loads it as the base, and training/model.py loads it with its readout.

The arithmetic is float32 on the CPU, whatever device is present, and the result is stored in bf16:

    W'  = bf16(f32(W) + (B @ A) * alpha / rank)                     each adapted linear (B, A float32)
    W'e = bf16(f32(We) + alpha_e / rank_e * B_e @ A_e)              each routed expert's fused weights (--expert-lora)
    readout = bf16(trained float32 readout)

Every other tensor is the base's, byte for byte, under the base's name, dtype and shape: the bases are bf16
throughout, and a base tensor in another dtype is refused rather than cast. It reads one base shard at a time and
needs about that shard's size in memory. The adapter's record names the rank, alpha and targets, so the merge cannot
run with another geometry than the one trained; an adapter file with expert tensors but no expert_lora record is
refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import torch

from . import expert_lora, lora
from .model import READOUT_FILE, text_config

RECORD_FILE = "training.json"
# The base's files the merged checkpoint carries as they are (the weights and their index are rewritten); nothing else
# is copied, so another weight format or a stray file in the base never reaches the release.
CARRIED = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def weight_files(directory: Path) -> list[str]:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    if (directory / "model.safetensors").exists():
        return ["model.safetensors"]
    raise SystemExit(f"{directory}: no model.safetensors or model.safetensors.index.json")


def base_digest(base: str | Path) -> str:
    """The base checkpoint's identity, cheap to compute: sha256 over its config.json, tokenizer.json and each weight
    file's safetensors header (every tensor's name, dtype, shape and offsets)."""
    base = Path(base)
    h = hashlib.sha256()
    for name in ("config.json", "tokenizer.json"):
        h.update(name.encode() + b"\0" + (base / name).read_bytes())
    for name in weight_files(base):
        with open(base / name, "rb") as stream:
            (size,) = struct.unpack("<Q", stream.read(8))
            h.update(name.encode() + b"\0" + stream.read(size))
    return h.hexdigest()


def read_adapter(adapter: Path) -> tuple[dict, dict, dict, torch.Tensor]:
    """(record, {module: (A, B)}, {experts module: (gate_up_a, gate_up_b, down_a, down_b)}, readout) of an adapter."""
    from safetensors.torch import load_file

    record = json.loads((adapter / "adapter_config.json").read_text())
    tensors = load_file(str(adapter / "adapter.safetensors"))
    if "readout.weight" not in tensors:
        raise SystemExit(f"{adapter}: no readout.weight")
    readout = tensors.pop("readout.weight")
    dense, experts = {}, {}
    for key in tensors:
        module, part = key.rsplit(".", 1)
        if part == "lora_a":
            dense[module] = (tensors[key], tensors[f"{module}.lora_b"])
        elif part == "gate_up_a":
            experts[module] = tuple(tensors[f"{module}.{p}"] for p in expert_lora.PARTS)
        elif part not in ("lora_b", *expert_lora.PARTS):
            raise SystemExit(f"{adapter}: unexpected tensor {key}")
    if experts and not record.get("expert_lora"):
        raise SystemExit(
            f"{adapter}: routed-expert adapters but adapter_config.json has no expert_lora record"
        )
    if record.get("expert_lora") and not experts:
        raise SystemExit(
            f"{adapter}: the record has expert_lora but the file holds no expert adapters"
        )
    targets = {t for t in record["lora"]["targets"].split(",") if t}
    stray = sorted(m for m in dense if m.rsplit(".", 1)[-1] not in targets)
    if stray:
        raise SystemExit(
            f"{adapter}: adapters on modules outside the recorded targets: {stray[:3]}"
        )
    return record, dense, experts, readout


def decision_fields(base: Path, prompt_template: str) -> dict:
    """The decision interface's fields a checkpoint of this base needs: codes, token ids, soft-cap."""
    from tokenizers import Tokenizer

    from strands_decider import core

    from .prompt import code_token_ids

    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    return {
        "codes": list(core.CODES),
        "token_ids": code_token_ids(tokenizer, prompt_template),
        "final_logit_softcapping": text_config(base).final_logit_softcapping,
    }


def merge(base: Path, adapter: Path, out: Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    record, dense, experts, readout = read_adapter(adapter)
    if record.get("base_sha256") and record["base_sha256"] != base_digest(base):
        raise SystemExit(
            f"{base} is not the base the adapter was trained on (its base_sha256 differs)"
        )
    scale = record["lora"]["alpha"] / record["lora"]["rank"]
    # (read only when the adapter has expert adapters)
    expert_scale = (
        record["expert_lora"]["alpha"] / record["expert_lora"]["rank"] if experts else 0.0
    )
    replace = {f"model.{name}.weight": ("dense", name) for name in dense}
    for name in experts:
        replace[f"model.{name}.gate_up_proj"] = ("gate_up", name)
        replace[f"model.{name}.down_proj"] = ("down", name)
    found: set[str] = set()
    out.mkdir(parents=True, exist_ok=False)
    total = 0
    try:
        for shard in weight_files(base):
            tensors = {}
            with safe_open(str(base / shard), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                for key in keys:
                    weight = handle.get_tensor(key)
                    if weight.dtype != torch.bfloat16:
                        raise SystemExit(
                            f"{shard}: {key} is {weight.dtype}; the merge expects a bf16 base"
                        )
                    kind, name = replace.get(key, (None, None))
                    if kind == "dense":
                        a, b = dense[name]
                        if tuple(weight.shape) != (b.shape[0], a.shape[1]):
                            raise SystemExit(
                                f"{key}: base {tuple(weight.shape)}, adapter {(b.shape[0], a.shape[1])}"
                            )
                        weight = lora.merged_weight(weight, a, b, scale)
                    elif kind is not None:
                        gate_up_a, gate_up_b, down_a, down_b = experts[name]
                        a, b = (gate_up_a, gate_up_b) if kind == "gate_up" else (down_a, down_b)
                        if tuple(weight.shape) != (b.shape[0], b.shape[1], a.shape[2]):
                            raise SystemExit(
                                f"{key}: base {tuple(weight.shape)}, adapter {(b.shape[0], b.shape[1], a.shape[2])}"
                            )
                        weight = expert_lora.merged_weight(weight, a, b, expert_scale)
                    tensors[key] = weight
                found.update(k for k in keys if k in replace)
            for key, value in tensors.items():
                # bf16: the base's, or merged_weight's bf16 result
                tensors[key] = value.contiguous()
                total += tensors[key].numel() * 2
            save_file(tensors, str(out / shard), metadata={"format": "pt"})
            del tensors
        missing = sorted(set(replace) - found)
        if missing:
            raise SystemExit(
                f"the base has no tensor for the adapter's {missing[:3]} (not this adapter's base?)"
            )
        save_file({"weight": readout.to(torch.bfloat16).contiguous()}, str(out / READOUT_FILE))
        index = base / "model.safetensors.index.json"
        if index.exists():
            data = json.loads(index.read_text())
            data["metadata"] = {"total_size": total}
            (out / index.name).write_text(json.dumps(data, indent=2) + "\n")
        for name in CARRIED:
            if (base / name).is_file():
                shutil.copy2(base / name, out / name)
        fields = decision_fields(base, record["prompt"])
        info = {
            "base_model": record["base_model"],
            "base_revision": record["revision"],
            "prompt": record["prompt"],
            "attention_mode": record["attention_mode"],
            "pooling": record["pooling"],
            "temperature": 1.0,
            **fields,
            "finetune": record,
        }
        (out / RECORD_FILE).write_text(json.dumps(info, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"merged_modules": len(dense), "merged_expert_layers": len(experts), "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--base",
        required=True,
        help="the full base checkpoint directory the adapter was trained on",
    )
    ap.add_argument(
        "--adapter", required=True, help="an adapter directory (<run>/step-NNNNN/adapter)"
    )
    ap.add_argument("--out", required=True, help="the merged checkpoint directory (must not exist)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads for the float32 arithmetic")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    began = time.time()
    result = merge(Path(a.base).resolve(), Path(a.adapter).resolve(), Path(a.out))
    print(json.dumps({**result, "seconds": round(time.time() - began)}))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
