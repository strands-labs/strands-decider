"""Export a token-readout checkpoint to the Strands Decider v2 package layout.

    python -m training.export_v2 <checkpoint> <out>

The package is what the v2 reference packages hold and their `source/` code
(`autojev.model.DecisionModel(<out>)`) loads: the torso with its LoRA adapter merged in,
saved in bfloat16 as a text-only Gemma 4 decoder (`config.json`, `model-*.safetensors`,
no `model.` prefix), `readout.safetensors` (one bfloat16 tensor `weight`, [255, hidden]),
`decision_config.json` (prompt, codes, token ids, temperature, attention, pooling and a
`finetune` record), and the checkpoint's tokenizer files, copied byte for byte. The
soft-cap is the torso config's `final_logit_softcapping`, set to the checkpoint's
`readout_softcap`. The package's `source/` directory is not written here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from strands_decider.core import CHAT, CODES
from strands_decider.modeling import StrandsDeciderModel

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _optional_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def export_v2(checkpoint: str, out: str) -> str:
    """Write checkpoint `checkpoint`'s v2 package to `out` (absent or empty); returns `out`."""
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"{out}: not empty")
    model = StrandsDeciderModel.load(checkpoint, host_embeddings=False)
    cfg = model.config
    if (cfg.head_type, cfg.render, cfg.attention_mode) != ("token_readout", CHAT, "prefix_bidir"):
        raise ValueError(
            "a v2 package is a token_readout checkpoint rendered strands-chat under prefix_bidir, "
            f"not {cfg.head_type} / {cfg.render} / {cfg.attention_mode}"
        )
    if cfg.temperature_by_kind:
        raise ValueError(
            "decision_config.json holds one temperature; this checkpoint has per-kind ones"
        )
    revision = cfg.base_revision
    if not revision and not os.path.isdir(cfg.base_model):
        # None trained on the Hub's default branch: the snapshot the local cache resolves it to.
        revision = os.path.basename(snapshot_download(cfg.base_model, local_files_only=True))
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError(f"{cfg.base_model}: no 40-character base revision ({revision!r})")

    torso: Any = model.torso
    if cfg.use_lora:
        torso = torso.merge_and_unload()
    torso.config.final_logit_softcapping = cfg.readout_softcap or None
    torso.to(torch.bfloat16).save_pretrained(out, max_shard_size="5GB")  # run-2's shards
    weight = model.head.state_dict()["proj.weight"]
    save_file(
        {"weight": weight.to(torch.bfloat16).contiguous()},
        os.path.join(out, "readout.safetensors"),
        metadata={"format": "pt"},
    )
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(checkpoint, name)):
            shutil.copy2(os.path.join(checkpoint, name), os.path.join(out, name))

    history = _optional_json(os.path.join(checkpoint, "history.json")) or []
    steps = [h["step"] for h in history if "step" in h]
    finetune = {
        "trainer": "strands-decider training.train",
        "checkpoint": os.path.abspath(checkpoint),
        "base_model": cfg.base_model,
        "revision": revision,
        "lora": {
            "rank": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "targets": ",".join(cfg.lora_targets),
        }
        if cfg.use_lora
        else None,
        "backbone": "LoRA merged, bfloat16" if cfg.use_lora else "as released, bfloat16",
        "readout": "trained, F32 master, saved bfloat16",
        "attention_mode": cfg.attention_mode,
        "prompt": cfg.render,
        "readout_mode": "trained",
        "pooling": "last",
        "optimizer_steps": max(steps, default=None),
        "train_config": _optional_json(os.path.join(checkpoint, "train_config.json")),
    }
    decision = {
        "finetune": finetune,
        "prompt": cfg.render,
        "format_version": 1,
        "base_model": cfg.base_model,
        "revision": revision,
        "codes": list(CODES),
        # Row k of the readout is code k's: the ids training seeded it from.
        "token_ids": [model.tokenizer.encode(c, add_special_tokens=False)[0] for c in CODES],
        "temperature": cfg.temperature,
        "attention_mode": cfg.attention_mode,
        "pooling": "last",
    }
    with open(os.path.join(out, "decision_config.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(decision, indent=2) + "\n")
    return out


def _cli(
    checkpoint: str = typer.Argument(..., help="A token_readout strands-chat checkpoint."),
    out: str = typer.Argument(..., help="Directory for the package: absent or empty."),
) -> None:
    """Merge the adapter and write the Strands Decider v2 package layout."""
    print(f"v2 package -> {export_v2(checkpoint, out)}")


if __name__ == "__main__":  # pragma: no cover
    typer.run(_cli)
