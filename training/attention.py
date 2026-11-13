"""`prefix_bidir`, the attention pattern the decision interface serves with (docs/spec.md section 6), applied to
Gemma 4's text decoder.

A prompt has two parts: the state, then the question. Tokens of the state see every state token, earlier or later, and
no question token. Tokens of the question see every token of the prompt. One mask serves every layer; a layer that
Gemma 4 would restrict to a local window sees under it exactly what a full-attention layer sees. Because nothing in
the state looks at the question, a state's hidden states do not change from one question to the next, and a server
can compute them once.

`install_mask` hooks the decoder's forward. From the batch's [B, L] padding mask and the index where each row's
question begins it builds one boolean [B, 1, L, L] mask and hands it to both kinds of layer. It also derives the
position ids from the padding mask, counting from each row's first real token, so padding shifts no position. The left
padding counts as part of the state: padding queries attend to the real state tokens, and no real query attends to
padding.

`install_split` is the trainer's `--attn-split N`. The hook still builds the mask, but for a microbatch of N tokens or
more the attention function leaves it unused: per row it makes two attention calls without a mask, state queries
against state keys and question queries against all keys, which SDPA can run on its fused kernels. The result matches
the masked attention up to bf16 rounding. At padding positions, whose outputs no later computation uses, it writes
zeros where the masked call writes a weighted average of state values. Narrower microbatches, and any call with
attention dropout, go through the mask.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def prefix_bidir_mask(padding: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    """A boolean [B, 1, L, L] mask; entry (i, j) is True when query i may look at key j. That needs key j to be a real
    token and, in addition, key j to lie before `boundary` or query i to lie at or after it. `padding` is [B, L], 1 for
    real tokens; `boundary` holds, per row, the index of the question's first token, left padding included."""
    positions = torch.arange(padding.shape[1], device=padding.device)
    in_prefix = positions[None, :] < boundary.to(padding.device)[:, None]
    valid = padding.bool()
    return (valid[:, None, :] & (in_prefix[:, None, :] | ~in_prefix[:, :, None]))[:, None, :, :]


def install_mask(text_model: torch.nn.Module, owner: Any) -> None:
    """Makes every forward of `text_model` (a Gemma 4 text decoder) use the prefix_bidir mask. The boundaries come from
    `owner._prefix_end`, which DeciderModel.forward sets for the length of one call and clears afterwards."""
    if text_model.config._attn_implementation != "sdpa":  # noqa: SLF001 (transformers keeps it there)
        raise ValueError("install_mask needs the decoder loaded with attn_implementation='sdpa'")

    def add_prefix_bidir_mask(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if args:
            raise ValueError(
                "call the decoder with keyword arguments (input_ids=..., attention_mask=...)"
            )
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError(
                "the decoder runs without a KV cache here: pass use_cache=False and no past_key_values"
            )
        embeddings = kwargs.get("inputs_embeds")
        if embeddings is None:
            embeddings = module.embed_tokens(kwargs["input_ids"])
        padding = kwargs.get("attention_mask")
        if padding is None:
            padding = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if not isinstance(padding, torch.Tensor) or padding.ndim != 2:
            raise ValueError("attention_mask must be the [batch, length] padding mask")
        if padding.shape != embeddings.shape[:2] or not padding.bool().any(dim=-1).all():
            raise ValueError(
                "the padding mask must have the inputs' shape and at least one real token per row"
            )
        if kwargs.get("position_ids") is None:
            kwargs["position_ids"] = (padding.long().cumsum(-1) - 1).clamp(min=0)
        boundary = owner._prefix_end  # noqa: SLF001 (the owner sets it for one forward)
        if boundary is None or boundary.shape != padding.shape[:1]:
            raise ValueError(
                "no question boundaries for this batch: run the decoder through DeciderModel.forward"
            )
        split = getattr(owner, "_split", None)
        if split is not None:
            width = padding.shape[1]
            # each row's count of left-padding tokens
            starts = (width - padding.long().sum(dim=1)).tolist()
            split["rows"] = list(
                zip((int(s) for s in starts), (int(e) for e in boundary.tolist()), strict=True)
            )
        allowed = prefix_bidir_mask(padding, boundary)
        kwargs["attention_mask"] = {"full_attention": allowed, "sliding_attention": allowed}
        return args, kwargs

    text_model.register_forward_pre_hook(add_prefix_bidir_mask, with_kwargs=True)


def _repeat(x: torch.Tensor, n: int) -> torch.Tensor:
    b, h, s, d = x.shape
    return x if n == 1 else x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def install_split(owner: Any, text_model: torch.nn.Module, min_length: int = 1024) -> None:
    """Adds the split attention to transformers' registry of attention functions and switches the decoder and each of
    its layers to it."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    state: dict[str, Any] = {
        "rows": None,
        "min_length": int(min_length),
        "calls": 0,
        "split_calls": 0,
    }
    owner._split = state  # noqa: SLF001 (read back by the mask hook)

    def prefix_split_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Takes query [B, Hq, L, D] and key, value [B, Hkv, L, D]; returns ([B, L, Hq, D], None), the shape
        transformers expects back from an attention function."""
        state["calls"] += 1
        rows = state["rows"]
        b, hq, length, d = query.shape
        if rows is None or length < state["min_length"] or len(rows) != b or dropout:
            unsplit: tuple[torch.Tensor, None] = sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )
            return unsplit
        state["split_calls"] += 1
        groups = hq // key.shape[1]
        key, value = _repeat(key, groups), _repeat(value, groups)
        out = []
        for i, (start, end) in enumerate(rows):
            q, k, v = (
                query[i : i + 1, :, start:],
                key[i : i + 1, :, start:],
                value[i : i + 1, :, start:],
            )
            n = end - start
            a = F.scaled_dot_product_attention(q[:, :, :n], k[:, :, :n], v[:, :, :n], scale=scaling)
            c = F.scaled_dot_product_attention(q[:, :, n:], k, v, scale=scaling)
            parts = [a, c] if start == 0 else [query.new_zeros(1, hq, start, d), a, c]
            out.append(torch.cat(parts, dim=2))
        return torch.cat(out, dim=0).transpose(1, 2).contiguous(), None

    # The name carries the model's id: two models in one process then never share the state dict above.
    name = f"prefix_bidir_split_{id(owner)}"
    AttentionInterface.register(name, prefix_split_forward)
    text_model.config._attn_implementation = name  # noqa: SLF001
    for layer in text_model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "config"):
            attention.config._attn_implementation = name  # noqa: SLF001
