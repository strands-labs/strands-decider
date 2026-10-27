# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from transformers 5.17, models/qwen3_5/modeling_qwen3_5.py,
# torch_chunk_gated_delta_rule and causal_conv1d_fn's fallback: the triangular solves are
# replaced by mps_kernels' block inversion, per-chunk outputs are stacked instead of written
# into a slice, and the masking keeps -inf out of the graph. See THIRD_PARTY_NOTICES.md at the
# repository root.
"""The Qwen3.5 Gated DeltaNet kernels as static-shape graphs for AWS Neuron (`--device neuron`).

neuronx-cc compiles a whole forward ahead of time, so every op must lower to a static graph.
transformers' reference chunk rule does not: `torch.linalg.solve_triangular` has no Neuron
lowering, and writing each chunk's output into a slice of a preallocated tensor is an in-place
update of a graph input. The rule here is the reference's steps with three changes: the two
solves become `mps_kernels._unit_lower_inverse` applied twice (only batched matmuls), the
per-chunk outputs are stacked, and the pairwise decay is masked with 0 before `exp` and zeroed
after, not filled with -inf. The causal convolution is `cpu_kernels`' shifted multiply-adds,
written out of place as compiled and measured on inf2.

Both agree with transformers' reference to within float rounding in fp32
(`tests/test_neuron_kernels.py`). `install()` swaps them in for Neuron tensors only; other
devices reach whatever was bound before.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .mps_kernels import _l2norm, _unit_lower_inverse

_installed = False


def chunk_gated_delta_rule_static(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """transformers' `torch_chunk_gated_delta_rule` as a static graph. State, decay and gating
    run in fp32 whatever the input dtype, as in the reference."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query = query * k_head_dim**-0.5

    pad = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad:
        query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    num_chunks = (sequence_length + pad) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(batch_size, num_v_heads, num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(batch_size, num_v_heads, num_chunks, chunk_size)

    lower = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).tril()
    cum_decay = decay.cumsum(dim=-1)
    diff = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).masked_fill(~lower, 0.0)
    pairwise = diff.exp() * lower.to(diff.dtype)

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    ut_inv = _unit_lower_inverse(ut_system)
    new_values = ut_inv @ v_beta
    k_cumdecay = ut_inv @ decayed_k_beta

    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        state = initial_state.to(torch.float32)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    outs = []
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        outs.append(query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=2).reshape(batch_size, num_v_heads, -1, v_head_dim)
    out = out[:, :, :sequence_length].transpose(1, 2).contiguous().to(initial_dtype)
    return out, (state if output_final_state else None)


def causal_conv1d_static(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """`cpu_kernels.causal_conv1d_cpu`, out of place. [B, C, L] in, [B, C, L] out."""
    from transformers.activations import ACT2FN

    seq_len = hidden_states.shape[-1]
    width = weight.shape[-1]
    acc = torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
    x = F.pad(hidden_states.to(weight.dtype).to(acc).transpose(1, 2), (0, 0, width - 1, 0))
    w = weight.t().to(acc)
    out = x[:, width - 1 :] * w[width - 1]
    for k in range(width - 1):
        out = out + x[:, k : k + seq_len] * w[k]
    if bias is not None:
        out = out + bias.to(acc)
    out = out.transpose(1, 2).to(weight.dtype)
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def install() -> bool:
    """Route Neuron calls of the Qwen3.5 chunk rule and causal convolution through the versions
    above. Idempotent. Returns False if this transformers has no Qwen3.5 model module."""
    global _installed
    if _installed:
        return True
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    except ImportError:
        return False
    rule = q.torch_chunk_gated_delta_rule
    conv = getattr(q, "causal_conv1d_fn", None)

    def rule_dispatch(
        query: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.device.type == "neuron":
            return chunk_gated_delta_rule_static(query, *args, **kwargs)
        return rule(query, *args, **kwargs)  # type: ignore[no-any-return]

    def conv_dispatch(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if hidden_states.device.type == "neuron" or conv is None:
            return causal_conv1d_static(hidden_states, *args, **kwargs)
        out: torch.Tensor = conv(hidden_states, *args, **kwargs)
        return out

    rule_dispatch.__wrapped__ = rule  # type: ignore[attr-defined]
    conv_dispatch.__wrapped__ = conv  # type: ignore[attr-defined]
    q.torch_chunk_gated_delta_rule = rule_dispatch
    q.causal_conv1d_fn = conv_dispatch
    _installed = True
    return True
