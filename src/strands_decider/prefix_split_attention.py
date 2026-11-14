"""Prefix-bidirectional attention without its dense mask, for speed on long micro-batches.

`prefix_bidir_masks` hands every attention layer a dense `[B, 1, L, L]` boolean mask. SDPA can
only serve a mask from its memory-efficient kernel, which is slow at Gemma 4's head sizes (256
and 512). Row by row the same attention is two unmasked calls:

  - the state's queries over the state's keys,
  - the question's queries over the state's and the question's keys,

which SDPA runs on its flash kernel. The values are the same up to floating-point rounding: the
same keys, the same scale, the same softmax over the same set. The padding positions, which
nothing reads (their keys are masked for every real query, and pooling takes the last real
token), come out zero here instead of an average over the state's values.

Rows are right-padded, as `training.data.collate` pads them: row i is state `[0, prefix_end)`,
question `[prefix_end, valid)`, padding `[valid, L)`.

    install(torso, min_length=1024)          # once, at model build
    set_rows(prefix_end, attention_mask)     # before every forward

Micro-batches under `min_length` padded tokens keep the masked call: for short
rows one batched call beats 2B small ones. This is the same scheme, and the same default, as the
reference v2 trainer's `--attn-split 1024`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

NAME = "prefix_split"
_STATE: dict[str, Any] = {"rows": None, "min_length": 1024}


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


def prefix_split_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """transformers' attention interface: query [B, Hq, L, D] -> ([B, L, Hq, D], None)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    rows = _STATE["rows"]
    b, heads, length, dim = query.shape
    if rows is None or len(rows) != b or length < _STATE["min_length"] or dropout:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    groups = heads // key.shape[1]
    key, value = _repeat_kv(key, groups), _repeat_kv(value, groups)
    out = []
    for i, (end, valid) in enumerate(rows):
        q, k, v = query[i : i + 1], key[i : i + 1], value[i : i + 1]
        state = F.scaled_dot_product_attention(
            q[:, :, :end], k[:, :, :end], v[:, :, :end], scale=scaling
        )
        question = F.scaled_dot_product_attention(
            q[:, :, end:valid], k[:, :, :valid], v[:, :, :valid], scale=scaling
        )
        parts = [state, question]
        if valid < length:  # right padding: nothing reads these positions
            parts.append(query.new_zeros(1, heads, length - valid, dim))
        out.append(torch.cat(parts, dim=2))
    return torch.cat(out, dim=0).transpose(1, 2).contiguous(), None


def install(torso: Any, min_length: int = 1024) -> None:
    """Point `torso`'s attention layers at `prefix_split_forward`, for good.

    It must stay in place through backward: with gradient checkpointing, backward runs each
    layer's forward again, and a recompute that took another attention path would save other
    tensors (torch.utils.checkpoint refuses). Idempotent. `set_rows` gives it each forward's rows.
    """
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, prefix_split_forward)
    _STATE["min_length"] = int(min_length)
    inner = getattr(torso, "model", torso)
    configs = [inner.config] + [
        layer.self_attn.config
        for layer in getattr(inner, "layers", [])
        if hasattr(getattr(layer, "self_attn", None), "config")
    ]
    for config in configs:
        config._attn_implementation = NAME  # noqa: SLF001 -- transformers' own switch


def set_rows(prefix_end: torch.Tensor, attention_mask: torch.Tensor) -> None:
    """Each row's (state end, real length) for the forward about to run and its backward.

    A micro-batch's backward (and its checkpoint recompute) runs before the next forward
    calls this again, so the recompute reads the same rows as the forward it repeats.
    """
    valid = attention_mask.long().sum(dim=1)
    _STATE["rows"] = list(
        zip(
            (int(x) for x in prefix_end.flatten().tolist()),
            (int(x) for x in valid.tolist()),
            strict=True,
        )
    )
