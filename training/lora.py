"""Low-rank adapters over the frozen bf16 decoder, and the readout's float32 master weight.

Small and explicit rather than `peft`, so the module tree keeps the decoder's own parameter names and the adapter is a
flat file of `<module>.lora_a` / `<module>.lora_b` tensors that `training/merge.py` folds into the base's weights.

Adapter weights are float32, computed in the decoder's bf16: the master keeps the precision an Adam update at lr 1e-4
needs (a bf16 mantissa resolves about 0.4% relative, and these updates are smaller), while the matmuls stay bf16.
`MasterLinear` gives the readout, which is trained outright, the same treatment.

The recipe's targets are every linear of the Gemma 4 decoder layers (`q_proj, k_proj, v_proj, o_proj, gate_proj,
up_proj, down_proj`, plus `per_layer_input_gate, per_layer_projection` on E2B and E4B); the configs list them.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Collection
from typing import Any

import torch

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PREFIX = "language_model.layers."


class LoRALinear(torch.nn.Module):
    """y = base(x) + (alpha / rank) * B @ A @ dropout(x), with `base` frozen. B starts at zero, so the adapted model is
    exactly the base at step 0."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        device = base.weight.device
        a = torch.empty(rank, base.in_features, dtype=torch.float32, device=device)
        torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_a = torch.nn.Parameter(a)
        self.lora_b = torch.nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=torch.float32, device=device)
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # float32 masters, bf16 math: the casts are two small tensors per call ([r, in] and [out, r]), and autograd
        # routes the gradient back through them into the float32 parameters.
        hidden = torch.nn.functional.linear(self.dropout(x), self.lora_a.to(x.dtype))
        delta = torch.nn.functional.linear(hidden, self.lora_b.to(x.dtype))
        return out + self.scale * delta.to(out.dtype)


class MasterLinear(torch.nn.Module):
    """A bias-free nn.Linear held as a float32 master weight and computed in the input's dtype. `.weight` stays its
    only parameter, under the same name."""

    def __init__(self, base: torch.nn.Linear) -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("MasterLinear expects a bias-free Linear")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = torch.nn.Parameter(base.weight.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


def promote(base: torch.nn.Module) -> MasterLinear:
    """A MasterLinear holding `base`'s weight in float32 (a no-op if it already is one)."""
    return base if isinstance(base, MasterLinear) else MasterLinear(base)


def _parent(root: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def attach(
    backbone: torch.nn.Module,
    rank: int = 64,
    alpha: float = 128,
    dropout: float = 0.05,
    targets: Collection[str] = DEFAULT_TARGETS,
    prefix: str = PREFIX,
) -> dict[str, LoRALinear]:
    """Wrap every targeted nn.Linear under `prefix` in a LoRALinear, in module-tree order (the order in which their A
    matrices draw from the RNG); returns {name: LoRALinear}."""
    chosen = [
        (name, module)
        for name, module in backbone.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith(prefix)
        and name.split(".")[-1] in targets
    ]
    if not chosen:
        raise ValueError(f"no nn.Linear under {prefix!r} matched {list(targets)}")
    wrapped: dict[str, LoRALinear] = {}
    for name, module in chosen:
        parent, attribute = _parent(backbone, name)
        layer = LoRALinear(module, rank, alpha, dropout)
        setattr(parent, attribute, layer)
        wrapped[name] = layer
    return wrapped


def adapter_modules(backbone: torch.nn.Module) -> dict[str, LoRALinear]:
    return {
        name: module for name, module in backbone.named_modules() if isinstance(module, LoRALinear)
    }


def parameters(backbone: torch.nn.Module) -> list[torch.nn.Parameter]:
    """The adapter parameters in a fixed order (sorted by module name), so every rank reduces them alike."""
    out: list[torch.nn.Parameter] = []
    for _name, module in sorted(adapter_modules(backbone).items()):
        out += [module.lora_a, module.lora_b]
    return out


def state_dict(
    backbone: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, module in sorted(adapter_modules(backbone).items()):
        out[f"{name}.lora_a"] = module.lora_a.detach().to(dtype).cpu().contiguous()
        out[f"{name}.lora_b"] = module.lora_b.detach().to(dtype).cpu().contiguous()
    return out


def save(
    backbone: torch.nn.Module,
    directory: str,
    config: dict[str, Any],
    readout: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Write `adapter.safetensors` (and the trained readout as `readout.weight`) and `adapter_config.json`."""
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    tensors = state_dict(backbone)
    if readout is not None:
        tensors["readout.weight"] = readout.weight.detach().float().cpu().contiguous()
    save_file(tensors, os.path.join(directory, "adapter.safetensors"))
    config = {
        **config,
        "tensors": len(tensors),
        "parameters": int(sum(t.numel() for t in tensors.values())),
    }
    with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def load(backbone: torch.nn.Module, path: str, readout: torch.nn.Module | None = None) -> int:
    """Load an adapter written by `save()` into an attached backbone (and its readout)."""
    from safetensors.torch import load_file

    tensors = load_file(path)
    modules = adapter_modules(backbone)
    missing = {name for name in modules} - {
        key.rsplit(".", 1)[0] for key in tensors if key != "readout.weight"
    }
    if missing:
        raise ValueError(f"adapter is missing {len(missing)} of {len(modules)} wrapped modules")
    with torch.no_grad():
        for name, module in modules.items():
            module.lora_a.copy_(tensors[f"{name}.lora_a"].to(module.lora_a.dtype))
            module.lora_b.copy_(tensors[f"{name}.lora_b"].to(module.lora_b.dtype))
        if readout is not None and "readout.weight" in tensors:
            readout.weight.copy_(tensors["readout.weight"].to(readout.weight.dtype))
    return len(tensors)


def merged_weight(
    weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float
) -> torch.Tensor:
    """A base weight with its adapter folded in: computed in float32, returned in the weight's dtype."""
    update = (lora_b.float() @ lora_a.float()) * scale
    return (weight.float() + update).to(weight.dtype)
