"""LoRA on the routed experts of a Gemma 4 mixture-of-experts decoder (the 26B-A4B), for `--expert-lora`.

transformers stores the 128 experts of a layer fused, as two 3-D parameters of Gemma4TextExperts
(`language_model.layers.<i>.experts`), in the same layout as the checkpoint's tensors:

    gate_up_proj  [E, 2 * moe_intermediate, hidden]   (gate rows first, then up rows)
    down_proj     [E, hidden, moe_intermediate]

and routes with `language_model.layers.<i>.router`, which stays frozen. `lora.attach()` wraps only nn.Linear modules, so
it touches neither.

ExpertLoRA replaces `experts` by a module that holds the frozen original as `.base` and adds, per expert e,

    gate_up:  W_gu[e] + s * B_gu[e] @ A_gu[e]     A_gu [E, r, hidden],  B_gu [E, 2I, r]
    down:     W_d[e]  + s * B_d[e]  @ A_d[e]      A_d  [E, r, I],       B_d  [E, hidden, r]

(one LoRA per expert per fused matrix), s = alpha / r, B = 0 at the start so step 0 is the base. The forward is
transformers' grouped-GEMM experts forward with the two low-rank terms added, so with B = 0 it is the base computation
exactly. Masters float32, math in bf16, as lora.LoRALinear.

Saved beside lora.save()'s tensors as `<module>.gate_up_a / gate_up_b / down_a / down_b`, with an `expert_lora` record in
adapter_config.json; training/merge.py folds them into gate_up_proj / down_proj per expert (merged_weights below).
"""

from __future__ import annotations

import math
from typing import Any

import torch

PARTS = ("gate_up_a", "gate_up_b", "down_a", "down_b")


def _grouped_mm(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """x [S, K] @ grouped weight [E, K, N] -> [S, N] (rows of expert e are offsets[e-1]:offsets[e])."""
    if hasattr(torch.nn.functional, "grouped_mm"):
        return torch.nn.functional.grouped_mm(x.to(weight.dtype), weight, offs=offsets)
    return torch._grouped_mm(x.to(weight.dtype), weight, offs=offsets)  # noqa: SLF001 (torch before 2.9)


class ExpertLoRA(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1 or rank % 8:
            # grouped_mm wants 16-byte strides: the [S, r] bf16 intermediate needs r a multiple of 8
            raise ValueError(f"expert rank must be a positive multiple of 8 (got {rank})")
        impl = getattr(base.config, "_experts_implementation", None)
        if impl != "grouped_mm":
            raise ValueError(
                f"ExpertLoRA mirrors the grouped_mm experts forward; this model uses {impl!r}"
            )
        if (
            getattr(base, "has_bias", False)
            or not getattr(base, "has_gate", True)
            or getattr(base, "is_transposed", False)
        ):
            raise ValueError(
                "ExpertLoRA expects gated, bias-free, untransposed fused experts (Gemma4TextExperts)"
            )
        self.base = base
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        e, two_i, hidden = base.gate_up_proj.shape
        inter = base.down_proj.shape[-1]
        device = base.gate_up_proj.device

        def a_init(n_in: int) -> torch.nn.Parameter:
            a = torch.empty(e, rank, n_in, dtype=torch.float32, device=device)
            for k in range(e):  # LoRALinear's init, per expert
                torch.nn.init.kaiming_uniform_(a[k], a=math.sqrt(5))
            return torch.nn.Parameter(a)

        self.gate_up_a = a_init(hidden)
        self.gate_up_b = torch.nn.Parameter(
            torch.zeros(e, two_i, rank, dtype=torch.float32, device=device)
        )
        self.down_a = a_init(inter)
        self.down_b = torch.nn.Parameter(
            torch.zeros(e, hidden, rank, dtype=torch.float32, device=device)
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()

    @property
    def num_experts(self) -> int:
        count: int = self.base.num_experts
        return count

    def _lora(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        """s * (x A_e^T) B_e^T for the rows of each expert e."""
        hidden = _grouped_mm(self.dropout(x), a.to(x.dtype).transpose(-2, -1), offsets)  # [S, r]
        # [S, out]
        return self.scale * _grouped_mm(hidden, b.to(x.dtype).transpose(-2, -1), offsets)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        base = self.base
        num_top_k = top_k_index.size(-1)
        num_tokens, hidden_dim = hidden_states.shape
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)
        expert_ids_g, perm = torch.sort(expert_ids)
        x = hidden_states[perm // num_top_k]
        sample_weights_g = sample_weights[perm]
        histc_input = (
            expert_ids_g.float()
            if expert_ids_g.device.type in ("cpu", "mps")
            else expert_ids_g.int()
        )
        tokens_per_expert = torch.histc(
            histc_input, bins=base.num_experts, min=0, max=base.num_experts - 1
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        sentinel = (expert_ids_g >= base.num_experts).unsqueeze(-1)
        x.masked_fill_(sentinel, 0.0)

        proj = _grouped_mm(x, base.gate_up_proj.transpose(-2, -1), offsets)
        proj = proj + self._lora(x, self.gate_up_a, self.gate_up_b, offsets).to(proj.dtype)
        proj = proj.masked_fill(sentinel, 0.0)
        proj = base._apply_gate(proj)  # noqa: SLF001 (transformers' own gate)
        out = _grouped_mm(proj, base.down_proj.transpose(-2, -1), offsets)
        out = out + self._lora(proj, self.down_a, self.down_b, offsets).to(out.dtype)
        out = out.masked_fill(sentinel, 0.0)

        weighted = out * sample_weights_g.unsqueeze(-1)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=perm.device)
        weighted = weighted[inv_perm]
        return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def default_rank(rank: int, config: Any) -> int:
    """'LoRA Without Regret': one LoRA per expert at rank = total rank / active experts."""
    return max(1, rank // int(config.top_k_experts))


def attach(
    backbone: torch.nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "language_model.layers.",
) -> dict[str, ExpertLoRA]:
    """Wrap every `<prefix><i>.experts` (Gemma4TextExperts) -> {name: ExpertLoRA}."""
    chosen = [
        (name, module)
        for name, module in backbone.named_modules()
        if name.startswith(prefix)
        and name.endswith(".experts")
        and hasattr(module, "gate_up_proj")
        and not isinstance(module, ExpertLoRA)
    ]
    if not chosen:
        raise ValueError(f"no fused experts under {prefix!r}: --expert-lora needs a MoE backbone")
    wrapped: dict[str, ExpertLoRA] = {}
    for name, module in chosen:
        parent = backbone.get_submodule(name.rsplit(".", 1)[0])
        layer = ExpertLoRA(module, rank, alpha, dropout)
        parent.experts = layer
        wrapped[name] = layer
    return wrapped


def modules(backbone: torch.nn.Module) -> dict[str, ExpertLoRA]:
    return {name: m for name, m in backbone.named_modules() if isinstance(m, ExpertLoRA)}


def parameters(backbone: torch.nn.Module) -> list[torch.nn.Parameter]:
    out: list[torch.nn.Parameter] = []
    for _name, m in sorted(modules(backbone).items()):
        out += [getattr(m, p) for p in PARTS]
    return out


def state_dict(
    backbone: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, m in sorted(modules(backbone).items()):
        for p in PARTS:
            out[f"{name}.{p}"] = getattr(m, p).detach().to(dtype).cpu().contiguous()
    return out


def load(backbone: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> int:
    """Copy `<name>.gate_up_a ...` from a loaded safetensors dict into the attached ExpertLoRA modules."""
    found = modules(backbone)
    missing = [n for n in found if f"{n}.gate_up_a" not in tensors]
    if missing:
        raise ValueError(f"adapter is missing {len(missing)} of {len(found)} expert adapters")
    with torch.no_grad():
        for name, m in found.items():
            for p in PARTS:
                getattr(m, p).copy_(tensors[f"{name}.{p}"].to(getattr(m, p).dtype))
    return len(found) * len(PARTS)


def merged_weight(
    weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float
) -> torch.Tensor:
    """A fused expert weight with its adapters folded in, per expert e: W[e] + scale * B[e] @ A[e], computed in float32
    and returned in the weight's dtype (gate_up_proj with gate_up_a / gate_up_b, down_proj with down_a / down_b)."""
    return (weight.float() + scale * torch.bmm(b.float(), a.float())).to(weight.dtype)
