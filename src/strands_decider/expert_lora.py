"""LoRA on the routed experts of a Gemma 4 mixture-of-experts decoder (the 26B-A4B).

transformers keeps the experts of a layer fused in `layers.<i>.experts`, two 3-D parameters:

    gate_up_proj  [E, 2 * I, H]   (gate rows first, then up rows)
    down_proj     [E, H, I]

PEFT's `target_modules` wraps nn.Linear modules only, so it never reaches them. ExpertLoRA
replaces `experts` by a module that holds the frozen original as `.base` and adds one LoRA
per expert to each fused matrix:

    gate_up[e] + s * gate_up_b[e] @ gate_up_a[e]    gate_up_a [E, r, H], gate_up_b [E, 2I, r]
    down[e]    + s * down_b[e] @ down_a[e]          down_a    [E, r, I], down_b    [E, H, r]

with s = alpha / r and B = 0 at the start, so step 0 computes the base model. The router
(`layers.<i>.router`) is not touched and stays frozen. The masters are float32; the math
runs in the activations' dtype. The forward never builds a full delta: each token's low-rank
terms go through the same expert-sorted matmuls as the base weights.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

PARTS = ("gate_up_a", "gate_up_b", "down_a", "down_b")
FILE = "experts.safetensors"  # beside PEFT's adapter, in a checkpoint's lora/ directory


def _expert_mm(x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """x [S, K] (rows sorted by expert, counts[e] rows for expert e) @ weight [E, K, N]."""
    if x.is_cuda:
        offsets = torch.cumsum(counts, 0, dtype=torch.int32)
        return torch.nn.functional.grouped_mm(x, weight.to(x.dtype), offs=offsets)
    # On the CPU (tests), loop over the experts: torch 2.15's CPU grouped_mm has no backward.
    # Every expert is in the graph, empty or not, so each one gets a gradient (DDP needs that).
    segments = x.split(counts.tolist())
    return torch.cat([seg @ weight[e].to(x.dtype) for e, seg in enumerate(segments)])


class ExpertLoRA(nn.Module):
    def __init__(self, base: Any, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1 or rank % 8:
            # grouped_mm needs 16-byte strides: the [S, r] bf16 intermediate needs r % 8 == 0
            raise ValueError(f"expert_lora_r must be a positive multiple of 8, not {rank}")
        self.base: Any = base  # transformers' fused experts (Gemma4TextExperts)
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        for p in base.parameters():
            p.requires_grad_(False)
        e, two_i, hidden = base.gate_up_proj.shape
        inter = base.down_proj.shape[-1]
        device = base.gate_up_proj.device

        def a_init(n_in: int) -> nn.Parameter:
            a = torch.empty(e, rank, n_in, dtype=torch.float32, device=device)
            for k in range(e):  # PEFT's LoRA A init, per expert
                nn.init.kaiming_uniform_(a[k], a=math.sqrt(5))
            return nn.Parameter(a)

        self.gate_up_a = a_init(hidden)
        self.gate_up_b = nn.Parameter(torch.zeros(e, two_i, rank, device=device))
        self.down_a = a_init(inter)
        self.down_b = nn.Parameter(torch.zeros(e, hidden, rank, device=device))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _lora(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """s * x A_e^T B_e^T for the rows of each expert e."""
        h = _expert_mm(self.dropout(x), a.transpose(-2, -1), counts)
        return self.scale * _expert_mm(h, b.transpose(-2, -1), counts)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        base = self.base
        n_experts = base.gate_up_proj.shape[0]
        tokens, hidden = hidden_states.shape
        top_k = top_k_index.size(-1)
        expert_ids, perm = torch.sort(top_k_index.reshape(-1))
        x = hidden_states[perm // top_k]
        counts = torch.bincount(expert_ids, minlength=n_experts)

        proj = _expert_mm(x, base.gate_up_proj.transpose(-2, -1), counts)
        proj = proj + self._lora(x, self.gate_up_a, self.gate_up_b, counts).to(proj.dtype)
        gate, up = proj.chunk(2, dim=-1)
        inner = base.act_fn(gate) * up
        out = _expert_mm(inner, base.down_proj.transpose(-2, -1), counts)
        out = out + self._lora(inner, self.down_a, self.down_b, counts).to(out.dtype)

        out = out * top_k_weights.reshape(-1)[perm].unsqueeze(-1).to(out.dtype)
        final = torch.zeros(tokens, hidden, dtype=out.dtype, device=out.device)
        final.index_add_(0, perm // top_k, out)
        return final.to(hidden_states.dtype)

    def merged(self) -> Any:
        """The base experts with s * B @ A added per expert (float32 sum, weights' dtype)."""
        with torch.no_grad():
            for w, a, b in (
                (self.base.gate_up_proj, self.gate_up_a, self.gate_up_b),
                (self.base.down_proj, self.down_a, self.down_b),
            ):
                w.copy_((w.float() + delta(a.to(w.device), b.to(w.device), self.scale)).to(w.dtype))
        return self.base


def delta(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """The per-expert update scale * B[e] @ A[e], [E, out, in], in float32."""
    return scale * torch.bmm(b.float(), a.float())


def attach(torso: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    """Wrap every `*.experts` module with fused gate_up_proj / down_proj; returns the count."""
    chosen = [
        name
        for name, m in torso.named_modules()
        if name.endswith(".experts") and hasattr(m, "gate_up_proj")
    ]
    if not chosen:
        raise ValueError("expert_lora: this torso has no fused experts (not a MoE model)")
    for name in chosen:
        parent = torso.get_submodule(name.rsplit(".", 1)[0])
        parent.experts = ExpertLoRA(parent.experts, rank, alpha, dropout)
    return len(chosen)


def modules(torso: nn.Module) -> dict[str, ExpertLoRA]:
    return {n: m for n, m in torso.named_modules() if isinstance(m, ExpertLoRA)}


def state_dict(torso: nn.Module) -> dict[str, torch.Tensor]:
    """`<layer>.experts.<part>` -> float32 tensor, for every attached ExpertLoRA."""
    return {
        f"{name}.{p}": getattr(m, p).detach().float().cpu().contiguous()
        for name, m in modules(torso).items()
        for p in PARTS
    }


def load_state(torso: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    found = modules(torso)
    keys = {f"{n}.{p}" for n in found for p in PARTS}
    if keys != set(tensors):
        raise ValueError(
            f"expert adapter: {len(keys - set(tensors))} tensors missing, "
            f"{len(set(tensors) - keys)} unexpected"
        )
    with torch.no_grad():
        for name, m in found.items():
            for p in PARTS:
                getattr(m, p).copy_(tensors[f"{name}.{p}"])


def merge_and_unwrap(torso: nn.Module) -> int:
    """Fold every ExpertLoRA into its base experts and put the base back; returns the count."""
    found = modules(torso)
    for name, m in found.items():
        torso.get_submodule(name.rsplit(".", 1)[0]).experts = m.merged()
    return len(found)
