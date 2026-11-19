"""prefix_bidir in transformers' Gemma 4 decoder, without a dense mask.

`use_prefix_bidir` gives every decoder layer an attention function whose mask is a `StateSplit`
(a whole prompt, split at its state's length), None (every query sees every key), or a boolean
mask over the keys of each row. `StateCache` serves one state's keys and values to any number of
question passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import transformers

# The name the attention function is registered under in transformers; the vision encoder keeps
# transformers' scaled dot-product attention (SDPA).
ATTENTION = "strands_decider_prefix_bidir"


@dataclass(frozen=True)
class StateSplit:
    """The prefix_bidir mask of a whole prompt: its first `length` tokens (the state) attend to
    the state, and the others (the question) to every token. A dense L x L mask would cost L^2
    bytes, and more once SDPA turns it into a float mask."""

    length: int

    def __getitem__(self, index: Any) -> StateSplit:
        # transformers 5.23 slices each layer's mask ([:, :, 0:, :]) ahead of Gemma 4's
        # KV-shared layers; a slice that dropped tokens would shift the split, so it's refused.
        whole = slice(None).indices(1 << 62)
        if isinstance(index, tuple) and all(
            isinstance(i, slice) and i.indices(1 << 62) == whole for i in index
        ):
            return self
        raise TypeError(f"a StateSplit keeps every token; it cannot be indexed by {index!r}")


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float | None,
) -> torch.Tensor:
    """torch's scaled dot-product attention, never causal, for queries [rows, heads, n, head_dim]
    over keys and values [rows, kv heads, keys, head_dim]. Grouped keys go to SDPA as they are
    only with no mask and head_dim at most 256, on CPU or on CUDA in 16 bits: in float32, torch
    2.7's CUDA SDPA takes them only in its math kernel, with an n x keys matrix per head.
    Otherwise each is repeated per query head, as transformers' SDPA attention does; MPS always
    repeats them."""
    groups = query.shape[1] // key.shape[1]
    gqa = (
        groups > 1
        and mask is None
        and query.shape[-1] <= 256
        and (query.device.type == "cpu" or (query.is_cuda and query.dtype != torch.float32))
    )
    if groups > 1 and not gqa:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    out: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return out


def _attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | StateSplit | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """A decoder layer's attention, in the signature of transformers' attention functions. A
    StateSplit runs as two unmasked SDPA calls: the state's queries over the state's keys, and the
    question's over every key. Returns [rows, tokens, heads, head_dim] and no attention weights."""
    scale = kwargs.get("scaling")
    if not isinstance(attention_mask, StateSplit):
        out = _scaled_dot_product_attention(query, key, value, attention_mask, scale)
        return out.transpose(1, 2).contiguous(), None
    s = attention_mask.length
    rows, heads, n, dim = query.shape
    out = query.new_empty(rows, n, heads, dim)
    out[:, :s] = _scaled_dot_product_attention(
        query[:, :, :s], key[:, :, :s], value[:, :, :s], None, scale
    ).transpose(1, 2)
    # torch 2.7's CUDA SDPA gets 64k+1 queries over a longer run of keys wrong with one key/value
    # head (E2B): such a question also takes the state's last query, whose row is dropped.
    first = s - 1 if (n - s) % 64 == 1 else s
    rest = _scaled_dot_product_attention(query[:, :, first:], key, value, None, scale)
    out[:, s:] = rest[:, :, s - first :].transpose(1, 2)
    return out, None


def use_prefix_bidir(model: Any) -> None:
    """Sets the decoder's attention to prefix_bidir's. transformers only warns when a layer can't
    switch, and its SDPA with no mask is causal: a layer left on it would keep the state from
    seeing its own later tokens, with no error, so a model where any layer stays is refused."""
    transformers.AttentionInterface.register(ATTENTION, _attention)
    model.set_attn_implementation({"text_config": ATTENTION})
    # The only record of the attention a layer runs is its config's private field.
    layers = [m for m in model.modules() if type(m).__name__.endswith("TextAttention")]
    got = {m.config._attn_implementation for m in layers}  # noqa: SLF001
    if not layers or got != {ATTENTION}:
        raise RuntimeError(
            f"transformers {transformers.__version__} did not set the decoder's attention to "
            f"{ATTENTION!r} (its layers run {sorted(map(str, got))})"
        )


class StateCache:
    """A state's keys and values, layer by layer, behind the two methods of a transformers cache
    that Gemma 4's decoders call (`update` and `get_seq_length`).

    In the state's pass it records each layer's keys and values. Frozen, it serves question
    passes: `update` returns the state's keys and values, repeated across the batch, followed by
    the questions', and stores nothing, so one state serves any number of question batches.
    E2B's and E4B's KV-shared layers call nothing: they reuse what the last layer of their type
    got from `update`."""

    def __init__(self) -> None:
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.length = 0
        self.frozen = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *_: Any,
        **__: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.frozen:
            self.layers[layer_idx] = (key_states, value_states)
            return key_states, value_states
        k, v = self.layers[layer_idx]
        n = key_states.shape[0]
        k, v = k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)
        return torch.cat([k, key_states], dim=-2), torch.cat([v, value_states], dim=-2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def freeze(self, length: int) -> None:
        self.length, self.frozen = length, True
