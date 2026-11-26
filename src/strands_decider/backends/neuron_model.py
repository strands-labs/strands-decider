"""Gemma 4's text decoder on Trainium2 through TorchNeuron, one tensor-parallel rank per process.

`neuron.py` (the backend) starts one worker process per rank. Each loads its shard of the
checkpoint's text decoder straight from the safetensors files and runs every request as one
packed pass: the state's tokens, then each question's, under prefix_bidir. The state's queries
attend to the state's keys; each question's queries attend to the state's keys and to its own
question's keys, never another question's. The state never attends to a question, so packing
every question after one copy of the state gives each question the result it would get alone.

Every pass is padded into a bucket, so the graphs compile once per bucket: one bucket of tokens
for the state and the questions together (joint packing), or for a long state a state bucket and
a question bucket (the split pass). A joint bucket's decoder runs as graphs of `layers_per_graph`
layers, a split one's as one graph per layer type, called once per layer with its weights.

The arithmetic follows transformers' Gemma 4 (modeling_gemma4.py) in bf16, as the reference
runs on CUDA: RMSNorm in float32, attention probabilities in float32, the residual stream in
bf16. The attention scores are computed from float32 copies of q and k.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

PREFIX = "model.language_model."
STATE = -2  # joint packing's qid of a state token
NEG = -1.0e9  # an additive mask value: finite, so a fully masked padding row can't make a NaN


@dataclass(frozen=True)
class LayerShape:
    """One layer type's attention, as one rank holds it."""

    head_dim: int
    heads: int  # query heads, all ranks
    kv_heads: int  # key/value heads, all ranks
    k_eq_v: bool
    q_lo: int  # this rank's query heads [q_lo, q_lo + q_local)
    q_local: int
    kv_lo: int  # this rank's key/value heads [kv_lo, kv_lo + kv_local)
    kv_local: int

    @property
    def group(self) -> int:  # query heads per key/value head on this rank
        return self.q_local // self.kv_local


def layer_shape(text: Any, layer_type: str, world: int, rank: int) -> LayerShape:
    full = layer_type == "full_attention"
    # transformers 5.2x resolves a layer's head_dim and kv heads through per_layer_config.
    layer = text.per_layer_config[list(text.layer_types).index(layer_type)]
    head_dim = int(layer.head_dim)
    kv_heads = int(layer.num_key_value_heads)
    heads = text.num_attention_heads
    if heads % world:
        raise ValueError(f"{heads} query heads don't split over {world} ranks")
    q_local = heads // world
    group = heads // kv_heads
    q_lo = rank * q_local
    kv_lo = q_lo // group
    kv_hi = (q_lo + q_local - 1) // group + 1
    shape = LayerShape(
        head_dim=head_dim,
        heads=heads,
        kv_heads=kv_heads,
        k_eq_v=bool(full and text.attention_k_eq_v),
        q_lo=q_lo,
        q_local=q_local,
        kv_lo=kv_lo,
        kv_local=kv_hi - kv_lo,
    )
    if shape.q_local % shape.kv_local:
        raise ValueError(f"rank {rank}: {q_local} query heads over {shape.kv_local} kv heads")
    return shape


# ------------------------------------------------------------------------------- the math


def _rms(x: Any, w: Any | None, eps: float) -> Any:
    """Gemma4RMSNorm: in float32, times the weight (no 1 + w), back to x's dtype."""
    import torch

    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def _rope(x: Any, cos: Any, sin: Any) -> Any:
    """x [T, heads, hd] with cos, sin [T, hd], in x's dtype as transformers applies them."""
    import torch

    cos, sin = cos[:, None, :], sin[:, None, :]
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def _attend(q: Any, k: Any, v: Any, bias: Any) -> Any:
    """softmax(q k^T + bias) v for q [kv, n, hd] over k, v [kv, m, hd] and bias [n or 1, m]:
    scores and probabilities in float32, the probabilities cast to v's dtype for the product."""
    import torch

    s = torch.matmul(q.float(), k.float().transpose(-1, -2))
    p = torch.softmax(s + bias, dim=-1).to(v.dtype)
    return torch.matmul(p, v)


def _identity(x: Any) -> Any:
    return x


_NKI_ATTN: list[Any] = []


def _nki_attention() -> Any:
    """nkilib's attention_cte wrapped for torch (as torch-neuronx's own SDPA decomposition
    wraps it), launched over the 2 physical cores of an LNC=2 NeuronCore."""
    if not _NKI_ATTN:
        from nkilib.core.attention.attention_cte import attention_cte
        from torch_neuronx.nki_hop import wrap_nki

        _NKI_ATTN.append(wrap_nki(attention_cte))
    return _NKI_ATTN[0]


def decoder_layer(
    h: Any,
    w_qkv: Any,
    w_o: Any,
    w_gu: Any,
    w_down: Any,
    ln_in: Any,
    ln_post_attn: Any,
    ln_pre_ff: Any,
    ln_post_ff: Any,
    q_norm: Any,
    k_norm: Any,
    scalar: Any,
    cos: Any,
    sin: Any,
    state_valid: Any,
    qid: Any,
    *,
    shape: LayerShape,
    state_len: int,
    eps: float,
    reduce: Any,
    q_chunk: int = 0,
    gather: Any = None,
    nki_state: bool = False,
) -> Any:
    """One Gemma4TextDecoderLayer over a packed prompt h [T, D] (T = state bucket + question
    bucket), on one rank's heads and MLP columns, the projection weights stored [in, out];
    `reduce` sums the row-parallel outputs.
    With `gather` (sequence parallelism), h is this rank's slice of the tokens [T / ranks, D],
    `gather` all-gathers the normed input before each projection, and `reduce` is a
    reduce-scatter back to the slice: the norms and residual adds run on 1/ranks of the tokens."""
    import torch

    sb = state_len
    hd, ql, kvl, g = shape.head_dim, shape.q_local, shape.kv_local, shape.group

    x = _rms(h, ln_in, eps)
    qkv = torch.matmul(x if gather is None else gather(x), w_qkv)
    t = qkv.shape[0]
    q = qkv[:, : ql * hd].reshape(t, ql, hd)
    k = qkv[:, ql * hd : (ql + kvl) * hd].reshape(t, kvl, hd)
    v = k if shape.k_eq_v else qkv[:, (ql + kvl) * hd :].reshape(t, kvl, hd)
    q = _rope(_rms(q, q_norm, eps), cos, sin)
    k = _rope(_rms(k, k_norm, eps), cos, sin)
    v = _rms(v, None, eps)

    # [kv, group * T, hd]: the query heads of one kv head side by side, no repeated keys.
    q = q.permute(1, 0, 2).reshape(kvl, g, t, hd)
    k = k.permute(1, 0, 2)
    v = v.permute(1, 0, 2)

    zero = torch.zeros((), dtype=torch.float32, device=h.device)
    neg = torch.full((), NEG, dtype=torch.float32, device=h.device)
    if sb == 0:
        # Joint packing: one bucket of T tokens, the state first (qid STATE), then each
        # question (qid i), then padding (-1). A token sees the state's tokens; a question's
        # token also sees its own question's. The state's tokens see only the state.
        allowed = (qid[None, :] == STATE) | ((qid[:, None] == qid[None, :]) & (qid[None, :] >= 0))
        bias = torch.where(allowed, zero, neg)  # [T, T]
        chunk = q_chunk if q_chunk and q_chunk < t else t
        outs = []
        for c in range(0, t, chunk):
            n = min(chunk, t - c)
            qc = q[:, :, c : c + n].reshape(kvl, g * n, hd)
            bc = bias[None, c : c + n].expand(g, -1, -1).reshape(g * n, t)
            outs.append(_attend(qc, k, v, bc).reshape(kvl, g, n, hd))
        attn = torch.cat(outs, dim=2).reshape(ql, t, hd).permute(1, 0, 2).reshape(t, ql * hd)
        return _mlp_half(
            h,
            attn,
            w_o,
            w_gu,
            w_down,
            ln_post_attn,
            ln_pre_ff,
            ln_post_ff,
            scalar,
            eps,
            reduce,
            gather,
        )
    state_bias = torch.where(state_valid, zero, neg)[None, :]  # [1, Sb]
    # The question's queries see the state's real tokens and their own question's tokens.
    same = (qid[:, None] == qid[None, :]) & (qid[None, :] >= 0)
    q_bias = torch.cat(
        [state_bias.expand(qid.shape[0], sb), torch.where(same, zero, neg)], dim=1
    )  # [Pb, T]

    ks, vs = k[:, :sb], v[:, :sb]
    outs = []
    if nki_state:
        # The state's queries over the state's real keys with the NKI Library's flash kernel
        # (attention_cte, non-causal, each query's keys bounded to [0, len(state))): no
        # [Sb, Sb] score matrix in HBM.
        n_q = kvl * g
        bmin = torch.zeros(n_q, sb, 1, dtype=torch.float32, device=h.device)
        bmax = state_valid.to(torch.float32).sum().reshape(1, 1, 1).expand(n_q, sb, 1).contiguous()
        o = _nki_attention()[(2,)](
            q[:, :, :sb].reshape(n_q, sb, hd).contiguous(),
            ks.contiguous(),
            vs.contiguous(),
            scale=1.0,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=bmin,
            bound_max=bmax,
        )
        outs.append(o.reshape(kvl, g, sb, hd))
    else:
        chunk = q_chunk if q_chunk and q_chunk < sb else sb
        for c in range(0, sb, chunk):
            qs = q[:, :, c : c + chunk].reshape(kvl, g * min(chunk, sb - c), hd)
            outs.append(_attend(qs, ks, vs, state_bias).reshape(kvl, g, -1, hd))
    pq = t - sb
    chunk = q_chunk if q_chunk and q_chunk < pq else pq
    for c in range(0, pq, chunk):
        n = min(chunk, pq - c)
        qp = q[:, :, sb + c : sb + c + n].reshape(kvl, g * n, hd)
        qb = q_bias[None, c : c + n].expand(g, -1, -1).reshape(g * n, t)
        outs.append(_attend(qp, k, v, qb).reshape(kvl, g, n, hd))
    attn = torch.cat(outs, dim=2).reshape(ql, t, hd).permute(1, 0, 2).reshape(t, ql * hd)
    return _mlp_half(
        h, attn, w_o, w_gu, w_down, ln_post_attn, ln_pre_ff, ln_post_ff, scalar, eps, reduce, gather
    )


def _mlp_half(
    h: Any,
    attn: Any,
    w_o: Any,
    w_gu: Any,
    w_down: Any,
    ln_post_attn: Any,
    ln_pre_ff: Any,
    ln_post_ff: Any,
    scalar: Any,
    eps: float,
    reduce: Any,
    gather: Any = None,
) -> Any:
    """The rest of the layer after attention: o_proj, the residual, the MLP, the layer scalar."""
    import torch

    h = h + _rms(reduce(torch.matmul(attn, w_o)), ln_post_attn, eps)
    x = _rms(h, ln_pre_ff, eps)
    gu = torch.matmul(x if gather is None else gather(x), w_gu)
    inter = gu.shape[-1] // 2
    m = torch.nn.functional.gelu(gu[:, :inter], approximate="tanh") * gu[:, inter:]
    h = h + _rms(reduce(torch.matmul(m, w_down)), ln_post_ff, eps)
    return h * scalar


WEIGHT_NAMES = (
    "w_qkv",
    "w_o",
    "w_gu",
    "w_down",
    "ln_in",
    "ln_post_attn",
    "ln_pre_ff",
    "ln_post_ff",
    "q_norm",
    "k_norm",
    "scalar",
)


def decoder_block(
    h: Any,
    cos_s: Any,
    sin_s: Any,
    cos_f: Any,
    sin_f: Any,
    state_valid: Any,
    qid: Any,
    *weights: Any,
    shapes: tuple[LayerShape, ...],
    full: tuple[bool, ...],
    **kw: Any,
) -> Any:
    """Consecutive decoder layers in one graph; `weights` holds each layer's WEIGHT_NAMES."""
    n = len(WEIGHT_NAMES)
    for i, shape in enumerate(shapes):
        c, s = (cos_f, sin_f) if full[i] else (cos_s, sin_s)
        # mypy can't count the star-args into decoder_layer's positional parameters.
        h = decoder_layer(  # type: ignore[misc, call-arg]
            h, *weights[i * n : (i + 1) * n], c, s, state_valid, qid, shape=shape, **kw
        )
    return h


def embed(
    ids: Any,
    positions: Any,
    table: Any,
    scale: Any,
    cos_s: Any,
    sin_s: Any,
    cos_f: Any,
    sin_f: Any,
    *,
    vocab_lo: int,
    reduce: Any,
) -> tuple[Any, ...]:
    """The scaled token embeddings [T, D] from this rank's vocabulary rows (summed over ranks),
    and each layer type's cos and sin at `positions`."""
    import torch

    local = ids - vocab_lo
    ok = (local >= 0) & (local < table.shape[0])
    e = torch.index_select(table, 0, torch.where(ok, local, torch.zeros_like(local)))
    e = e * ok[:, None].to(e.dtype)
    e = reduce(e) * scale
    return (
        e,
        torch.index_select(cos_s, 0, positions),
        torch.index_select(sin_s, 0, positions),
        torch.index_select(cos_f, 0, positions),
        torch.index_select(sin_f, 0, positions),
    )


def final(h: Any, rows: Any, norm: Any, *, eps: float, gather: Any = None) -> Any:
    """The final norm at the rows read, float32 [rows, D] (h gathered first under sequence
    parallelism)."""
    import torch

    if gather is not None:
        h = gather(h)
    return _rms(torch.index_select(h, 0, rows), norm, eps).float()


def _rotary_embedding(text: Any) -> Any:
    """transformers' rotary embedding for the text config: Gemma 4's, or the 12B's
    Gemma4Unified one."""
    if text.model_type == "gemma4_unified_text":
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedTextRotaryEmbedding,
        )

        return Gemma4UnifiedTextRotaryEmbedding(text)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    return Gemma4TextRotaryEmbedding(text)


# ------------------------------------------------------------------------------ the weights


class Shards:
    """Reads one rank's slices of the checkpoint's tensors through safetensors' lazy slices."""

    def __init__(self, path: str):
        from safetensors import safe_open

        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.where = json.load(f)["weight_map"]
        else:  # a one-file checkpoint, such as the 12B
            with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
                self.where = dict.fromkeys(f.keys(), "model.safetensors")
        self.files = {
            name: safe_open(os.path.join(path, name), framework="pt")
            for name in sorted(set(self.where.values()))
        }

    def has(self, key: str) -> bool:
        return PREFIX + key in self.where

    def get(
        self, key: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None
    ) -> Any:
        sl = self.files[self.where[PREFIX + key]].get_slice(PREFIX + key)
        if rows is not None and cols is not None:
            return sl[rows[0] : rows[1], cols[0] : cols[1]]
        if rows is not None:
            return sl[rows[0] : rows[1]]
        if cols is not None:
            return sl[:, cols[0] : cols[1]]
        return sl[:]


# ------------------------------------------------------------------------------- the rank


class Rank:
    """One tensor-parallel rank: its shard on its NeuronCore and the compiled graphs."""

    def __init__(self, path: str, world: int, rank: int, opts: dict[str, Any]):
        import torch
        import transformers

        self.world, self.rank, self.opts = world, rank, opts
        self.compiled = bool(opts.get("compile", True))
        self.q_chunk = int(opts.get("q_chunk", 4096))  # query rows per attention chunk
        # Layers per compiled graph in the joint buckets (6: the 31B's 5 sliding + 1 full).
        self.per_graph = int(opts.get("layers_per_graph", 6))
        config = transformers.AutoConfig.from_pretrained(path)
        text = config.get_text_config()
        self.text = text
        self.eps = float(text.rms_norm_eps)
        self.layer_types = list(text.layer_types)
        self.shapes = {t: layer_shape(text, t, world, rank) for t in sorted(set(self.layer_types))}
        # opts device/dtype: "cpu"/"float32" runs this same decoder on the host, to check its
        # arithmetic against transformers' without a NeuronCore.
        if opts.get("device"):
            self.device = torch.device(opts["device"])
        else:
            self.device = torch.device("neuron", torch.neuron.current_device())
        dev = self.device
        self.on_neuron = self.device.type == "neuron"
        dtype = getattr(torch, opts.get("dtype", "bfloat16"))

        # Over several ranks the residual stream is split by tokens (sequence parallelism):
        # each projection's input is all-gathered and its output reduce-scattered.
        self.reduce: Any = _identity
        self.gather: Any = None
        if world > 1:
            import torch.distributed as dist
            import torch.distributed._functional_collectives as fc

            group: Any = dist.group.WORLD

            def scatter(x: Any) -> Any:
                return fc.reduce_scatter_tensor(x, "sum", 0, group)

            def gather(x: Any) -> Any:
                return fc.all_gather_tensor(x, 0, group)

            self.reduce, self.gather = scatter, gather
        # The split pass (long states) runs the state's attention with the NKI flash kernel.
        self.nki_state = bool(opts.get("nki_state", True)) and self.on_neuron
        sh = Shards(path)
        bf16 = dtype
        inter = text.intermediate_size
        if inter % world:
            raise ValueError(f"intermediate size {inter} doesn't split over {world} ranks")
        il = inter // world
        self.layers = []
        for i, lt in enumerate(self.layer_types):
            s = self.shapes[lt]
            p = f"layers.{i}."
            hd = s.head_dim
            q = sh.get(p + "self_attn.q_proj.weight", rows=(s.q_lo * hd, (s.q_lo + s.q_local) * hd))
            kv = (s.kv_lo * hd, (s.kv_lo + s.kv_local) * hd)
            parts = [q, sh.get(p + "self_attn.k_proj.weight", rows=kv)]
            if not s.k_eq_v:
                parts.append(sh.get(p + "self_attn.v_proj.weight", rows=kv))
            elif sh.has(p + "self_attn.v_proj.weight"):
                raise ValueError(f"layer {i}: attention_k_eq_v but the checkpoint has a v_proj")
            gate = sh.get(p + "mlp.gate_proj.weight", rows=(rank * il, (rank + 1) * il))
            up = sh.get(p + "mlp.up_proj.weight", rows=(rank * il, (rank + 1) * il))
            w = {
                "w_qkv": torch.cat(parts, 0),
                "w_o": sh.get(
                    p + "self_attn.o_proj.weight", cols=(s.q_lo * hd, (s.q_lo + s.q_local) * hd)
                ),
                "w_gu": torch.cat([gate, up], 0),
                "w_down": sh.get(p + "mlp.down_proj.weight", cols=(rank * il, (rank + 1) * il)),
                "ln_in": sh.get(p + "input_layernorm.weight"),
                "ln_post_attn": sh.get(p + "post_attention_layernorm.weight"),
                "ln_pre_ff": sh.get(p + "pre_feedforward_layernorm.weight"),
                "ln_post_ff": sh.get(p + "post_feedforward_layernorm.weight"),
                "q_norm": sh.get(p + "self_attn.q_norm.weight"),
                "k_norm": sh.get(p + "self_attn.k_norm.weight"),
                "scalar": sh.get(p + "layer_scalar"),
            }
            for k in ("w_qkv", "w_o", "w_gu", "w_down"):
                w[k] = w[k].t()  # [in, out]: `x @ W.t()` transposes every weight block per call
            self.layers.append({k: v.to(bf16).contiguous().to(dev) for k, v in w.items()})
        vocab = text.vocab_size
        if vocab % world:
            raise ValueError(f"vocabulary {vocab} doesn't split over {world} ranks")
        vl = vocab // world
        self.vocab_lo = rank * vl
        self.table = (
            sh.get("embed_tokens.weight", rows=(rank * vl, (rank + 1) * vl))
            .to(bf16)
            .contiguous()
            .to(dev)
        )
        # Gemma4TextScaledWordEmbedding: sqrt(hidden) in the weights' dtype (bf16: 73.5).
        self.scale = torch.tensor(text.hidden_size**0.5, dtype=torch.float32).to(bf16).to(dev)
        self.norm = sh.get("norm.weight").to(bf16).to(dev)

        # cos and sin tables from transformers' own rotary embedding, as the reference computes
        # them (float32, then bf16), for every position a bucket can hold.
        rot = _rotary_embedding(text)
        n = int(opts.get("max_positions", 32768 + 4096))
        pos = torch.arange(n).view(1, -1)
        dummy = torch.zeros(1, dtype=bf16)
        tabs = {}
        for lt in ("sliding_attention", "full_attention"):
            c, s_ = rot(dummy, pos, lt)
            tabs[lt] = (c[0].contiguous().to(dev), s_[0].contiguous().to(dev))
        self.cos_s, self.sin_s = tabs["sliding_attention"]
        self.cos_f, self.sin_f = tabs["full_attention"]

        self._fns: dict[Any, Any] = {}
        self.compile_s: dict[str, float] = {}

    # The graphs, one per (kind, bucket); static ints are bound in closures.
    def _fn(self, key: tuple[Any, ...]) -> Any:
        import functools

        import torch

        if key in self._fns:
            return self._fns[key]
        kind = key[0]
        red, gat = self.reduce, self.gather
        if kind == "embed":
            f: Any = functools.partial(embed, vocab_lo=self.vocab_lo, reduce=red)
        elif kind == "final":
            f = functools.partial(final, eps=self.eps, gather=gat)
        elif kind == "block":
            _, types, sb, _pb = key
            f = functools.partial(
                decoder_block,
                shapes=tuple(self.shapes[t] for t in types),
                full=tuple(t == "full_attention" for t in types),
                state_len=sb,
                eps=self.eps,
                reduce=red,
                q_chunk=self.q_chunk,
                gather=gat,
                nki_state=self.nki_state and sb > 0,
            )
        else:
            _, lt, sb, _pb = key
            f = functools.partial(
                decoder_layer,
                shape=self.shapes[lt],
                state_len=sb,
                eps=self.eps,
                reduce=red,
                q_chunk=self.q_chunk,
                gather=gat,
                nki_state=self.nki_state and sb > 0,
            )
        if self.compiled and self.on_neuron:
            f = torch.compile(f, backend="neuron", fullgraph=True, dynamic=False)
        self._fns[key] = f
        return f

    def forward(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray | None, dict[str, float]]:
        """One packed pass; returns the final-norm rows (rank 0) and timings."""
        import torch

        dev = self.device
        sb, pb = int(batch["state_bucket"]), int(batch["question_bucket"])
        t0 = time.time()
        ids = torch.from_numpy(batch["ids"]).to(dev)
        pos = torch.from_numpy(batch["positions"]).to(dev)
        state_valid = torch.from_numpy(batch["state_valid"]).to(dev)
        qid = torch.from_numpy(batch["qid"]).to(dev)
        rows = torch.from_numpy(batch["rows"]).to(dev)
        timings: dict[str, float] = {}
        h, cs, ss, cf, sf = self._timed(
            ("embed", sb + pb),
            timings,
            ids,
            pos,
            self.table,
            self.scale,
            self.cos_s,
            self.sin_s,
            self.cos_f,
            self.sin_f,
        )
        # Block graphs for the joint (short) buckets. The split (long) buckets keep one graph per
        # layer: block graphs for every long bucket don't fit the device memory of one process.
        if self.per_graph > 1 and sb == 0:
            for a in range(0, len(self.layer_types), self.per_graph):
                types = tuple(self.layer_types[a : a + self.per_graph])
                ws = [self.layers[i][k] for i in range(a, a + len(types)) for k in WEIGHT_NAMES]
                h = self._timed(
                    ("block", types, sb, pb), timings, h, cs, ss, cf, sf, state_valid, qid, *ws
                )
        else:
            for i, lt in enumerate(self.layer_types):
                w = self.layers[i]
                c, s = (cf, sf) if lt == "full_attention" else (cs, ss)
                h = self._timed(
                    ("layer", lt, sb, pb),
                    timings,
                    h,
                    w["w_qkv"],
                    w["w_o"],
                    w["w_gu"],
                    w["w_down"],
                    w["ln_in"],
                    w["ln_post_attn"],
                    w["ln_pre_ff"],
                    w["ln_post_ff"],
                    w["q_norm"],
                    w["k_norm"],
                    w["scalar"],
                    c,
                    s,
                    state_valid,
                    qid,
                )
        out = self._timed(("final", int(rows.shape[0]), sb + pb), timings, h, rows, self.norm)
        res = out.cpu().numpy() if self.rank == 0 else None
        if self.on_neuron:
            torch.neuron.synchronize()
        timings["forward_s"] = time.time() - t0
        return res, timings

    def _timed(self, key: tuple[Any, ...], timings: dict[str, float], *args: Any) -> Any:
        """Runs the graph for `key`, timing its first call (the compile) per bucket."""
        import torch

        name = (
            f"block{len(key[1])}/{key[2]}/{key[3]}"
            if key[0] == "block"
            else "/".join(map(str, key))
        )
        if key[0] == "embed":
            fn = self._fn(("embed",))
        elif key[0] == "final":
            fn = self._fn(("final",))
        else:
            fn = self._fn(key)
        if name in self.compile_s:
            return fn(*args)
        t = time.time()
        out = fn(*args)
        if self.on_neuron:
            torch.neuron.synchronize()
        self.compile_s[name] = time.time() - t
        timings["compile:" + name] = self.compile_s[name]
        return out


def worker_main(
    rank: int, world: int, port: int, path: str, opts: dict[str, Any], inbox: Any, outbox: Any
) -> None:
    """A rank's process: joins the process group, loads its shard, then answers the backend's
    messages until told to stop. Rank 0 replies with the rows; every rank reports errors."""
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    for k, v in (opts.get("env") or {}).items():
        os.environ[k] = str(v)
    try:
        import torch

        # 16+ ranks share the host: a few threads each, not one per core each.
        torch.set_num_threads(int(opts.get("threads", 4)))
        torch._dynamo.config.cache_size_limit = 1024  # noqa: SLF001
        torch._dynamo.config.accumulated_cache_size_limit = 4096  # noqa: SLF001
        if world > 1:
            import torch.distributed as dist

            dist.init_process_group(backend="neuron")
        torch.set_grad_enabled(False)
        r = Rank(path, world, rank, opts)
        outbox.put(("ready", rank))
    except Exception:
        outbox.put(("error", rank, traceback.format_exc()))
        return
    import queue

    parent = os.getppid()
    while True:
        try:
            msg = inbox.get(timeout=5)
        except queue.Empty:
            # The backend's process is gone (killed, or a server stopped without close()):
            # leave rather than hold the NeuronCore.
            if os.getppid() != parent:
                break
            continue
        if msg[0] == "stop":
            break
        if msg[0] == "stats":
            st: dict[str, Any] = {"rank": rank, "compile_s": dict(r.compile_s)}
            try:
                from torch.neuron import memory as mem

                torch.neuron.synchronize()
                st["max_memory_allocated"] = int(mem.max_memory_allocated())
                st["memory_allocated"] = int(mem.memory_allocated())
                st["max_memory_reserved"] = int(mem.max_memory_reserved())
            except Exception as e:
                st["memory_error"] = repr(e)
            outbox.put(("stats", rank, st))
            continue
        if msg[0] == "profile":
            # torch.profiler with a Neuron session on rank 0 (runtime and, if asked, device
            # traces for Neuron Explorer); the other ranks run the same passes unprofiled.
            _, req, batch, outdir, modes, reps = msg
            try:
                with torch.inference_mode():
                    r.forward(batch)  # warm
                    if rank == 0:
                        from torch.profiler import ProfilerActivity, profile
                        from torch_neuronx.profiling import NeuronConfig, ProfileMode

                        os.makedirs(outdir, exist_ok=True)
                        cfg = NeuronConfig(
                            modes=[ProfileMode(m) for m in modes], profile_output_dir=outdir
                        )
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                            experimental_config=cfg,
                        ) as prof:
                            times = [r.forward(batch)[1]["forward_s"] for _ in range(reps)]
                        prof.export_chrome_trace(os.path.join(outdir, "framework_trace.json"))
                        with open(os.path.join(outdir, "key_averages.txt"), "w") as f:
                            f.write(
                                prof.key_averages().table(
                                    sort_by="self_cpu_time_total", row_limit=40
                                )
                            )
                        outbox.put(("result", req, None, {"profiled_forward_s": times}))
                    else:
                        for _ in range(reps):
                            r.forward(batch)
            except Exception:
                outbox.put(("error", rank, traceback.format_exc()))
            continue
        if msg[0] == "forward":
            _, req, batch = msg
            try:
                with torch.inference_mode():
                    out, timings = r.forward(batch)
                if rank == 0:
                    outbox.put(("result", req, out, timings))
            except Exception:
                outbox.put(("error", rank, traceback.format_exc()))
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
