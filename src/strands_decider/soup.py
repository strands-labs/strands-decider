"""A soup of N same-recipe checkpoints: one checkpoint whose weights average theirs.

Averaging A and B separately would average nothing meaningful: each seed's adapter is
only defined up to an invertible r x r mixing (B M^-1, M A), and the seeds start from
different random A. What is comparable across seeds is each layer's update
delta W = s * B @ A (s = lora_alpha / r). The soup's update is exactly their mean, as a
rank-N*r adapter: A' = [A_1; ...; A_N], B' = [B_1 ... B_N] / N, with lora_alpha scaled so
that s is unchanged. The existing loader and the HF export read it as any adapter.

The head is averaged the same way, by what it computes rather than by its tensors. A
pointer head's logit is a dot product q . k, invariant to any rotation of the q/k space,
so independently initialised heads have unrelated bases. After folding each head's
LayerNorm gain and bias into its projections, the heads are stacked: the soup's head has
pointer_dim N * dim and its logit is exactly the mean of the N heads' logits on the same
features. A linear slot head is averaged directly (its logit is linear in its weights);
an MLP slot head stacks its hidden units.

The temperatures fitted on each checkpoint do not describe the soup: they are reset to
1, so run `strands-decider calibrate` on the soup before evaluating or serving it.
"""

from __future__ import annotations

import json
import math
import os
import shutil

import torch

from .modeling import CONFIG_NAME, config_path, load_head_state

ADAPTER = "lora/adapter_model.safetensors"
ADAPTER_CONFIG = "lora/adapter_config.json"
FITTED = {"temperature", "temperature_by_kind"}  # calibration: may differ between the inputs


def _json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        out: dict = json.load(fh)
    return out


def _fold_norm(w: torch.Tensor, b: torch.Tensor, g: torch.Tensor,
               beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear(LayerNorm(x)) with the norm's gain and bias moved into the linear map."""
    return w * g[None, :], w @ beta + b


def soup_heads(states: list[dict[str, torch.Tensor]], head_type: str) -> dict[str, torch.Tensor]:
    """The head whose logit is the mean of `states`' logits (see the module docstring)."""
    n = len(states)
    folded = []
    for s in states:
        g, beta = s["norm.weight"].double(), s["norm.bias"].double()
        f = {}
        for name in ("q", "k", "proj", "proj.0"):
            if f"{name}.weight" in s:
                f[name] = _fold_norm(s[f"{name}.weight"].double(), s[f"{name}.bias"].double(), g, beta)
        if "proj.2.weight" in s:
            f["proj.2"] = (s["proj.2.weight"].double(), s["proj.2.bias"].double())
        folded.append(f)
    out: dict[str, torch.Tensor] = {}
    if head_type == "pointer":
        # cat over heads; 1/sqrt(n) on k turns the new (n * dim) ** -0.5 scale into the old / n
        out["q.weight"] = torch.cat([f["q"][0] for f in folded])
        out["q.bias"] = torch.cat([f["q"][1] for f in folded])
        out["k.weight"] = torch.cat([f["k"][0] for f in folded]) / math.sqrt(n)
        out["k.bias"] = torch.cat([f["k"][1] for f in folded]) / math.sqrt(n)
    elif "proj.weight" in states[0]:  # linear slot head
        out["proj.weight"] = torch.stack([f["proj"][0] for f in folded]).mean(0)
        out["proj.bias"] = torch.stack([f["proj"][1] for f in folded]).mean(0)
    else:  # MLP slot head: stack the hidden units
        out["proj.0.weight"] = torch.cat([f["proj.0"][0] for f in folded])
        out["proj.0.bias"] = torch.cat([f["proj.0"][1] for f in folded])
        out["proj.2.weight"] = torch.cat([f["proj.2"][0] for f in folded], dim=1) / n
        out["proj.2.bias"] = torch.stack([f["proj.2"][1] for f in folded]).mean(0)
    hidden = states[0]["norm.weight"].numel()
    out["norm.weight"] = torch.ones(hidden, dtype=torch.float64)
    out["norm.bias"] = torch.zeros(hidden, dtype=torch.float64)
    return {k: v.float().contiguous() for k, v in out.items()}


def soup_adapters(adapters: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The rank-N*r adapter whose every delta W is the mean of the adapters' (same scale s)."""
    n = len(adapters)
    if any(set(a) != set(adapters[0]) for a in adapters):
        raise ValueError("the adapters cover different modules")
    out = {}
    for key in adapters[0]:
        if ".lora_A." in key:
            out[key] = torch.cat([a[key] for a in adapters], dim=0)
        elif ".lora_B." in key:
            out[key] = (torch.cat([a[key].double() for a in adapters], dim=1) / n).to(adapters[0][key].dtype)
        else:
            raise ValueError(f"{key}: only lora_A / lora_B tensors can be souped")
    return out


def _check_paths(checkpoints: list[str], out: str) -> None:
    """Refuse an `out` that overlaps an input."""
    dst = os.path.realpath(out)
    for c in checkpoints:
        src = os.path.realpath(c)
        if os.path.commonpath([src, dst]) in (src, dst):
            raise ValueError(f"--out {out} overlaps the input {c}")


def soup(checkpoints: list[str], out: str) -> str:
    """Write the soup of `checkpoints` (local directories of one recipe) to `out`."""
    from safetensors.torch import load_file, save_file

    if not checkpoints:
        raise ValueError("no checkpoints")
    _check_paths(checkpoints, out)
    n = len(checkpoints)
    cfgs = [_json(config_path(c)) for c in checkpoints]
    recipe = [{k: v for k, v in c.items() if k not in FITTED} for c in cfgs]
    if any(r != recipe[0] for r in recipe):
        raise ValueError("the checkpoints' configs differ beyond their temperatures: not one recipe")
    cfg = cfgs[0]
    if cfg.get("full_weight_targets"):  # a research build's torso-matrix checkpoint
        raise ValueError("soup reads LoRA checkpoints; full-weight checkpoints are not supported")
    os.makedirs(os.path.join(out, "lora"), exist_ok=True)

    if cfg.get("use_lora"):
        acfgs = [_json(os.path.join(c, ADAPTER_CONFIG)) for c in checkpoints]
        for a in acfgs:  # PEFT writes this set in no fixed order
            a["target_modules"] = sorted(a["target_modules"] or [])
        acfg = acfgs[0]
        if any(a != acfg for a in acfgs):
            raise ValueError("the adapter configs differ")
        if acfg.get("use_dora") or acfg.get("rank_pattern") or acfg.get("alpha_pattern") \
                or acfg.get("modules_to_save"):
            raise ValueError("soup supports plain LoRA (no DoRA, rank/alpha patterns or modules_to_save)")
        r = acfg["r"]
        # PEFT scales by alpha / r, or alpha / sqrt(r) with rsLoRA: keep that scale at rank n * r
        alpha = acfg["lora_alpha"] * (math.sqrt(n) if acfg.get("use_rslora") else n)
        save_file(soup_adapters([load_file(os.path.join(c, ADAPTER)) for c in checkpoints]),
                  os.path.join(out, ADAPTER))
        with open(os.path.join(out, ADAPTER_CONFIG), "w", encoding="utf-8") as fh:
            json.dump({**acfg, "r": n * r, "lora_alpha": alpha}, fh, indent=2)
        cfg = {**cfg, "lora_r": n * r, "lora_alpha": alpha}

    head = soup_heads([load_head_state(c) for c in checkpoints], cfg["head_type"])
    torch.save(head, os.path.join(out, "slot_head.pt"))
    if cfg["head_type"] == "pointer":
        cfg = {**cfg, "pointer_dim": n * cfg["pointer_dim"]}
    elif cfg.get("head_hidden"):
        cfg = {**cfg, "head_hidden": n * cfg["head_hidden"]}
    cfg = {**cfg, "temperature": 1.0, "temperature_by_kind": {}}  # calibrate the soup
    with open(os.path.join(out, CONFIG_NAME), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    # tokenizer, chat template and the first checkpoint's train_config.json (the recipe)
    for f in os.listdir(checkpoints[0]):
        src = os.path.join(checkpoints[0], f)
        if os.path.isfile(src) and f not in {"slot_head.pt", "head.safetensors", "history.json",
                                             os.path.basename(config_path(checkpoints[0]))}:
            shutil.copy2(src, os.path.join(out, f))
    with open(os.path.join(out, "soup.json"), "w", encoding="utf-8") as fh:
        json.dump({"checkpoints": [os.path.abspath(c) for c in checkpoints]}, fh, indent=2)
    return out
