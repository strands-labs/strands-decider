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
1, so run `python -m training.evaluate calibrate` on the soup before evaluating or serving it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict

import torch
import typer
from rich.console import Console

from strands_decider.modeling import (
    CONFIG_NAME,
    StrandsDeciderConfig,
    base_revision,
    config_path,
    load_head_state,
)

ADAPTER = "lora/adapter_model.safetensors"
ADAPTER_CONFIG = "lora/adapter_config.json"
FITTED = {"temperature", "temperature_by_kind"}  # calibration: may differ between the inputs


def _json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        out: dict = json.load(fh)
    return out


def _fold_norm(
    w: torch.Tensor, b: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear(LayerNorm(x)) with the norm's gain and bias moved into the linear map."""
    return w * g[None, :], w @ beta + b


def soup_heads(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The pointer head whose logit is the mean of `states`' logits (see the module docstring)."""
    n = len(states)
    folded = []
    for s in states:
        g, beta = s["norm.weight"].double(), s["norm.bias"].double()
        f = {}
        for name in ("q", "k"):
            f[name] = _fold_norm(s[f"{name}.weight"].double(), s[f"{name}.bias"].double(), g, beta)
        folded.append(f)
    out: dict[str, torch.Tensor] = {}
    # cat over heads; 1/sqrt(n) on k turns the new (n * dim) ** -0.5 scale into the old / n
    out["q.weight"] = torch.cat([f["q"][0] for f in folded])
    out["q.bias"] = torch.cat([f["q"][1] for f in folded])
    out["k.weight"] = torch.cat([f["k"][0] for f in folded]) / math.sqrt(n)
    out["k.bias"] = torch.cat([f["k"][1] for f in folded]) / math.sqrt(n)
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
            out[key] = (torch.cat([a[key].double() for a in adapters], dim=1) / n).to(
                adapters[0][key].dtype
            )
        else:
            raise ValueError(f"{key}: only lora_A / lora_B tensors can be souped")
    return out


def _check_paths(checkpoints: list[str], out: str, replace: bool) -> None:
    """Refuse an `out` that overlaps an input, and a nonempty `out` unless `replace`."""
    dst = os.path.realpath(out)
    for c in checkpoints:
        src = os.path.realpath(c)
        if os.path.commonpath([src, dst]) in (src, dst):
            raise ValueError(f"--out {out} overlaps the input {c}")
    if os.path.isdir(dst) and os.listdir(dst) and not replace:
        raise ValueError(f"--out {out} is not empty; pass --replace to replace its contents")


def soup(checkpoints: list[str], out: str, replace: bool = False) -> str:
    """Write the soup of `checkpoints` (local directories of one recipe) to `out`, which must
    be empty or absent; `replace` deletes what it holds first."""
    from safetensors.torch import load_file, save_file

    if not checkpoints:
        raise ValueError("no checkpoints")
    _check_paths(checkpoints, out, replace)
    n = len(checkpoints)
    # from_json drops stale slot-era keys (num_slots, head_type="pointer", head_hidden,
    # head_init, kl_frozen_weight) that pre-PR and v1-2610 pointer checkpoints still carry,
    # and rejects head_type="slot" cleanly so a slot-era input stops here rather than
    # halfway through, after soup has written an adapter file.
    typed = [StrandsDeciderConfig.from_json(config_path(c)) for c in checkpoints]
    cfgs = [asdict(c) for c in typed]
    recipe = [{k: v for k, v in c.items() if k not in FITTED | {"base_revision"}} for c in cfgs]
    if any(r != recipe[0] for r in recipe):
        raise ValueError(
            "the checkpoints' configs differ beyond their temperatures: not one recipe"
        )
    # The base revision is compared as the loader resolves it (config, else provenance.json).
    revisions = {base_revision(c, t) for c, t in zip(checkpoints, typed, strict=True)}
    if len(revisions) > 1:
        raise ValueError(
            f"the checkpoints were trained on different base revisions: {sorted(map(str, revisions))}"
        )
    cfg = {**cfgs[0], "base_revision": revisions.pop()}  # pinned: no provenance fallback needed
    if cfg.get("full_weight_targets"):  # a research build's torso-matrix checkpoint
        raise ValueError("soup reads LoRA checkpoints; full-weight checkpoints are not supported")

    if cfg.get("use_lora"):
        acfgs = [_json(os.path.join(c, ADAPTER_CONFIG)) for c in checkpoints]
        for a in acfgs:  # PEFT writes a list in no fixed order; a string is a regex, kept as is
            if isinstance(a.get("target_modules"), list):
                a["target_modules"] = sorted(a["target_modules"])
        acfg = acfgs[0]
        if any(a != acfg for a in acfgs):
            raise ValueError("the adapter configs differ")
        if (
            acfg.get("use_dora")
            or acfg.get("rank_pattern")
            or acfg.get("alpha_pattern")
            or acfg.get("modules_to_save")
            or acfg.get("target_parameters")
        ):
            raise ValueError(
                "soup supports plain LoRA (no DoRA, rank/alpha patterns, modules_to_save "
                "or target_parameters)"
            )
    if replace and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "lora"), exist_ok=True)

    if cfg.get("use_lora"):
        r = acfg["r"]
        # PEFT scales by alpha / r, or alpha / sqrt(r) with rsLoRA: keep that scale at rank n * r
        alpha = acfg["lora_alpha"] * (math.sqrt(n) if acfg.get("use_rslora") else n)
        save_file(
            soup_adapters([load_file(os.path.join(c, ADAPTER)) for c in checkpoints]),
            os.path.join(out, ADAPTER),
        )
        with open(os.path.join(out, ADAPTER_CONFIG), "w", encoding="utf-8") as fh:
            json.dump({**acfg, "r": n * r, "lora_alpha": alpha}, fh, indent=2)
        cfg = {**cfg, "lora_r": n * r, "lora_alpha": alpha}

    head = soup_heads([load_head_state(c) for c in checkpoints])
    save_file(
        {k: v.contiguous() for k, v in head.items()},
        os.path.join(out, "head.safetensors"),
        metadata={"format": "pt"},
    )
    cfg = {**cfg, "pointer_dim": n * cfg["pointer_dim"]}
    cfg = {**cfg, "temperature": 1.0, "temperature_by_kind": {}}  # calibrate the soup
    with open(os.path.join(out, CONFIG_NAME), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    # tokenizer, chat template and the first checkpoint's train_config.json (the recipe)
    for f in os.listdir(checkpoints[0]):
        src = os.path.join(checkpoints[0], f)
        if os.path.isfile(src) and f not in {
            "head.safetensors",
            "history.json",
            os.path.basename(config_path(checkpoints[0])),
        }:
            shutil.copy2(src, os.path.join(out, f))
    with open(os.path.join(out, "soup.json"), "w", encoding="utf-8") as fh:
        json.dump({"checkpoints": [os.path.abspath(c) for c in checkpoints]}, fh, indent=2)
    return out


# ---------------------------------------------------------------- CLI


console = Console()


def _cli(
    checkpoints: list[str] = typer.Argument(..., help="Checkpoints of one recipe (other seeds)."),
    out: str = typer.Option(
        ..., "--out", help="Directory for the souped checkpoint: absent or empty."
    ),
    replace: bool = typer.Option(False, "--replace", help="Delete what --out holds first."),
) -> None:
    """Average same-recipe checkpoints by their LoRA updates and head logits.

    The soup's temperatures are reset: run `python -m training.evaluate calibrate` next.
    """
    soup(checkpoints, out, replace=replace)
    console.print(f"[green]soup of {len(checkpoints)} checkpoints -> {out}[/] (calibrate it next)")


if __name__ == "__main__":  # pragma: no cover
    typer.run(_cli)
