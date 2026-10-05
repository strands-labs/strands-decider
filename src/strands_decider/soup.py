"""A weight-averaged checkpoint (a "model soup") from runs that share one initialisation.

    python -m strands_decider.soup --out checkpoints/v21-soup CKPT CKPT CKPT

The head (and any other trained tensor that is not a LoRA factor) is averaged element by
element. LoRA factors are not: the mean of the A matrices times the mean of the B matrices
is not the mean of the updates B_i A_i, since each run's adapter is only determined up to
an invertible r x r change of basis. What is averaged is the merged update of each target
module,

    mean_i  s * B_i A_i  =  (s / n) * [B_1 ... B_n] [A_1; ...; A_n],

so the soup's adapter is the n adapters' factors concatenated along the rank axis, rank
n * r, unchanged, with the scaling divided by n. PEFT's scaling is lora_alpha / r, so
dividing by n at rank n * r keeps lora_alpha as it is. (An adapter trained with use_rslora,
which no Strands Decider config sets, is refused: its scaling is lora_alpha / sqrt(r).) The
soup's update is exactly the mean update, with no SVD and no rounding of the factors; for three rank-16 runs it is a rank-48
adapter of the same structure, which every load path (StrandsDeciderModel.load, the MLX
engine, hf_export) reads from adapter_config.json as it reads any other.

Averaging is only meaningful between runs that started from the same weights: the same
base at the same revision, the same head shape, and the same initialisation -- the same
`init_seed` (train.py) for runs from a raw base, or the same `continue_from` checkpoint.
The inputs' train_config.json files say which; inputs that disagree are refused. The
soup's stored calibration is reset to 1.0, as `continue_from` resets it: the average of
calibrated models is not calibrated, so `strands-decider calibrate` it before serving.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from .modeling import (
    CONFIG_NAME,
    StrandsDeciderConfig,
    checkpoint_dir,
    config_path,
    load_head_state,
)
from .runinfo import library_versions

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
# StrandsDeciderConfig fields every input must agree on: what the soup's weights mean.
MATCH_FIELDS = (
    "base_model", "base_revision", "head_type", "num_slots", "head_hidden",
    "pointer_dim", "torch_dtype", "use_lora", "lora_r", "lora_alpha", "lora_targets",
)
# adapter_config.json fields every input must agree on.
ADAPTER_MATCH = (
    "r", "lora_alpha", "use_rslora", "use_dora", "target_modules", "rank_pattern",
    "alpha_pattern", "bias", "fan_in_fan_out", "modules_to_save", "peft_type",
)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def init_identity(path: str) -> dict[str, Any]:
    """Where checkpoint `path`'s trained weights started, from its train_config.json.

    A continue_from run starts from that checkpoint (its init_seed draws nothing that
    survives, see train.py); any other run starts from its random initialisation, seeded
    by init_seed or, when that is unset, by seed.
    """
    tc = os.path.join(path, "train_config.json")
    if not os.path.exists(tc):
        raise ValueError(f"{path}: no train_config.json, so nothing says how it was initialised")
    cfg = _read_json(tc)
    if cfg.get("continue_from"):
        return {"continue_from": os.path.normpath(cfg["continue_from"])}
    seed = cfg.get("init_seed")
    return {
        "init_from": cfg.get("init_from"),
        "head_init": cfg.get("head_init", "random"),
        "init_seed": cfg.get("seed", 0) if seed is None else seed,
    }


def _check(paths: list[str], configs: list[StrandsDeciderConfig],
           adapters: list[dict[str, Any] | None]) -> None:
    if len(paths) < 2:
        raise ValueError("a soup needs at least two checkpoints")
    ref = configs[0]
    for p, c in zip(paths[1:], configs[1:], strict=True):
        for f in MATCH_FIELDS:
            if getattr(c, f) != getattr(ref, f):
                raise ValueError(f"{p}: {f} is {getattr(c, f)!r}, {paths[0]} has {getattr(ref, f)!r}")
    idents = [init_identity(p) for p in paths]
    for p, ident in zip(paths[1:], idents[1:], strict=True):
        if ident != idents[0]:
            raise ValueError(f"{p}: initialised from {ident}, {paths[0]} from {idents[0]}; "
                             "averaging weights is only meaningful from one initialisation")
    if ref.use_lora:
        a0 = adapters[0] or {}
        if a0.get("use_dora"):
            raise ValueError("DoRA adapters are not supported")
        if a0.get("use_rslora"):
            raise ValueError("use_rslora adapters are not supported")
        for p, a in zip(paths[1:], adapters[1:], strict=True):
            for f in ADAPTER_MATCH:
                av, rv = (a or {}).get(f), a0.get(f)
                if isinstance(av, list) and isinstance(rv, list):
                    av, rv = sorted(av), sorted(rv)
                if av != rv:
                    raise ValueError(f"{p}: adapter {f} is {av!r}, {paths[0]} has {rv!r}")
        if a0.get("rank_pattern") or a0.get("alpha_pattern"):
            raise ValueError("per-module rank_pattern / alpha_pattern adapters are not supported")


def _mean(tensors: list[torch.Tensor]) -> torch.Tensor:
    out = torch.stack([t.to(torch.float64) for t in tensors]).mean(dim=0)
    return out.to(tensors[0].dtype)


def soup_head(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The element-wise mean of each head tensor."""
    keys = set(states[0])
    for s in states[1:]:
        if set(s) != keys:
            raise ValueError("the heads have different parameters")
    out = {}
    for k in states[0]:
        ts = [s[k] for s in states]
        if any(t.shape != ts[0].shape for t in ts):
            raise ValueError(f"head tensor {k} differs in shape")
        out[k] = _mean(ts)
    return out


def soup_lora(weights: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The concatenated factors (A along rows, B along columns) of each LoRA module; any
    other tensor in the adapter averaged element-wise. Scaling is the caller's: the
    returned adapter's update is n times the mean update at the input scaling."""
    keys = set(weights[0])
    for w in weights[1:]:
        if set(w) != keys:
            raise ValueError("the adapters cover different modules")
    out = {}
    for k in sorted(keys):
        ts = [w[k] for w in weights]
        if any(t.shape != ts[0].shape for t in ts):
            raise ValueError(f"adapter tensor {k} differs in shape")
        if ".lora_A." in k:
            out[k] = torch.cat(ts, dim=0)  # [r, in] -> [n r, in]
        elif ".lora_B." in k:
            out[k] = torch.cat(ts, dim=1)  # [out, r] -> [out, n r]
        elif "lora_" in k:
            raise ValueError(f"adapter tensor {k}: not a plain LoRA factor, cannot be souped")
        else:
            out[k] = _mean(ts)
    return out


def soup(paths: list[str], out: str) -> dict[str, Any]:
    """Write the soup of checkpoints `paths` to `out`; returns what soup.json records."""
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    paths = [checkpoint_dir(p) for p in paths]
    configs = [StrandsDeciderConfig.from_json(config_path(p)) for p in paths]
    use_lora = configs[0].use_lora
    adapters = [_read_json(os.path.join(p, "lora", ADAPTER_CONFIG)) if use_lora else None
                for p in paths]
    _check(paths, configs, adapters)
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError(f"{out} exists and is not empty")
    n = len(paths)

    os.makedirs(out, exist_ok=True)
    torch.save(soup_head([load_head_state(p) for p in paths]), os.path.join(out, "slot_head.pt"))

    cfg = configs[0]
    record: dict[str, Any] = {"inputs": [os.path.abspath(p) for p in paths],
                              "init": init_identity(paths[0]), "method": "mean",
                              "versions": library_versions()}
    if use_lora:
        a0 = dict(adapters[0] or {})
        r, new_r = int(a0["r"]), n * int(a0["r"])
        merged = soup_lora([load_file(os.path.join(p, "lora", ADAPTER_WEIGHTS)) for p in paths])
        os.makedirs(os.path.join(out, "lora"))
        save_file(merged, os.path.join(out, "lora", ADAPTER_WEIGHTS))
        # lora_alpha / (n r) is each input's lora_alpha / r divided by n: the mean update.
        a0.update(r=new_r)
        with open(os.path.join(out, "lora", ADAPTER_CONFIG), "w", encoding="utf-8") as fh:
            json.dump(a0, fh, indent=2)
        cfg.lora_r = new_r
        scale = float(a0["lora_alpha"]) / r
        record["method"] = (f"head: element-wise mean; LoRA: factors concatenated, rank {r} -> "
                            f"{new_r}, scaling {scale:g} -> {scale / n:g} (exactly the mean update)")
    # The mean of calibrated models is not calibrated; forward() would apply these.
    cfg.temperature = 1.0
    cfg.temperature_by_kind = {}
    cfg.image_temperature_by_kind = {}
    with open(os.path.join(out, CONFIG_NAME), "w", encoding="utf-8") as fh:
        fh.write(cfg.to_json())
    AutoTokenizer.from_pretrained(paths[0]).save_pretrained(out)
    # Read by a later soup of soups (init_identity), and by people.
    tc = _read_json(os.path.join(paths[0], "train_config.json"))
    tc.update(output_dir=out, soup_of=record["inputs"])
    with open(os.path.join(out, "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump(tc, fh, indent=2)
    with open(os.path.join(out, "soup.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m strands_decider.soup", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="new checkpoint directory (must be empty)")
    ap.add_argument("checkpoints", nargs="+", help="two or more checkpoints from one initialisation")
    args = ap.parse_args(argv)
    record = soup(args.checkpoints, args.out)
    print(f"[strands-decider] soup of {len(args.checkpoints)} -> {args.out}: {record['method']}")
    print("[strands-decider] calibration reset to 1.0: run `strands-decider calibrate` on it")


if __name__ == "__main__":
    main()
