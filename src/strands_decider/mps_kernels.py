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
# torch_chunk_gated_delta_rule: the two triangular solves are replaced by one
# block inversion applied twice, and the function was restructured around that
# change. See THIRD_PARTY_NOTICES.md at the repository root.
"""A faster Gated DeltaNet chunk rule for Apple-silicon (MPS) inference.

On macOS neither `flash-linear-attention` (Triton, Linux only) nor `causal_conv1d`
(CUDA) can be installed, so a Qwen3.5 torso runs transformers' reference PyTorch
implementation of both. Profiled on an M3 Pro at 1,024 tokens, the reference
`torch_chunk_gated_delta_rule` is 806 of 1,739 ms per forward; `causal_conv1d_fn` is
32 ms and not worth replacing.

Nearly all of that 806 ms is two `torch.linalg.solve_triangular` calls per layer, which
on MPS take ~20 ms each for a [1, 16, 16, 64, 64] unit-lower-triangular system. The
system is `I + N` with `N` strictly lower triangular, and its inverse can be built by
recursive block inversion -- six levels of two batched 64x64 matmuls, each doubling the
width of the inverted diagonal blocks -- in a few ms on MPS. The inverse is formed once
and applied to both right-hand sides, where the reference solves twice.

(A first version used the finite Neumann product (I - N)(I + N^2)(I + N^4)...; it passed
random-input tests and was wrong by up to 5e8 on real model activations, where the
powers of N grow enormous before cancelling. `tests/test_mps_kernels.py` now includes
an ill-conditioned case of that shape.)

`chunk_gated_delta_rule_mps` is modified from transformers 5.17's
`torch_chunk_gated_delta_rule` (Apache-2.0; copyright notice above): the same steps,
with the two solves replaced by the inverse and the code restructured around it.
`install()` swaps it in for MPS tensors only; other devices, and any build where a real
kernel (fla) is already bound, are left alone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_installed = False
_unit_lower_inverse_metal_library = None

_UNIT_LOWER_INVERSE_METAL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void unit_lower_inverse_64(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup float x[64 * 64];
    threadgroup float next_x[64 * 64];
    const uint base = gid * 64 * 64;

    for (uint i = tid; i < 4096; i += 256) {
        uint r = i >> 6, c = i & 63;
        x[i] = (r == c) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint b = 1; b < 64; b <<= 1) {
        uint block = b << 1;
        uint blocks = 64 / block;
        uint work = blocks * b * b;

        for (uint w = tid; w < work; w += 256) {
            uint per_block = b * b;
            uint bi = w / per_block;
            uint rem = w - bi * per_block;
            uint rr = rem / b, cc = rem - rr * b;
            uint start = bi * block;
            uint row = start + b + rr, col = start + cc;

            float sum = 0.0f;
            for (uint j = 0; j < b; ++j) {
                float left = 0.0f;
                for (uint k = 0; k < b; ++k)
                    left += x[(start+b+rr)*64 + (start+b+k)]
                          * a[base + (start+b+k)*64 + (start+j)];
                sum += left * x[(start+j)*64 + (start+cc)];
            }
            next_x[row*64 + col] = -sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint w = tid; w < work; w += 256) {
            uint per_block=b*b, bi=w/per_block, rem=w-bi*per_block;
            uint rr=rem/b, cc=rem-rr*b, start=bi*block;
            x[(start+b+rr)*64 + (start+cc)] =
                next_x[(start+b+rr)*64 + (start+cc)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i=tid; i<4096; i+=256) out[base+i]=x[i];
}
"""


def _unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit lower-triangular matrix (diagonal ignored, taken as 1).

    Recursive block inversion, done level by level on full matrices. X holds the
    inverse's diagonal blocks of width b (block-diagonal, zero elsewhere). For each
    2b-wide diagonal block [[A, 0], [C, B]], the inverse's lower-left block is
    -B^-1 C A^-1; with L_b the entries of `a` lying in those C positions, that is
    exactly -X L_b X, so X <- X - X L_b X doubles the block width. Six levels for 64
    wide, two matmuls each.

    Not the Neumann product (I-N)(I+N^2)(I+N^4)...: that is exact algebraically but
    the powers of N reach ~1e8 on real Qwen3.5 inputs before cancelling, which fp32
    cannot survive.
    """
    if a.device.type == "mps" and a.dtype == torch.float32 and a.shape[-2:] == (64, 64):
        a = a.contiguous()
        out = torch.empty_like(a)
        matrices = a.numel() // 4096
        if matrices == 0:
            return out
        global _unit_lower_inverse_metal_library
        if _unit_lower_inverse_metal_library is None:
            _unit_lower_inverse_metal_library = torch.mps.compile_shader(_UNIT_LOWER_INVERSE_METAL)
        _unit_lower_inverse_metal_library.unit_lower_inverse_64(
            a, out,
            threads=[matrices * 256, 1, 1],
            group_size=[256, 1, 1],
        )
        return out

    n = a.shape[-1]
    idx = torch.arange(n, device=a.device)
    x = torch.eye(n, dtype=a.dtype, device=a.device).expand_as(a).clone()
    b = 1
    while b < n:
        block = 2 * b
        same_block = (idx[:, None] // block) == (idx[None, :] // block)
        lower_half = (idx[:, None] % block) >= b
        upper_half = (idx[None, :] % block) < b
        coupling = (same_block & lower_half & upper_half).to(a.dtype)
        x = x - x @ (a * coupling) @ x
        b = block
    return x


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule_mps(
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
    """transformers' `torch_chunk_gated_delta_rule`, with one inverse in place of two solves."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query = query * query.shape[-1] ** -0.5

    pad = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    num_chunks = (sequence_length + pad) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # The one change from the reference: invert once, apply twice.
    ut_inv = _unit_lower_inverse(ut_system)
    new_values = ut_inv @ v_beta
    k_cumdecay = ut_inv @ decayed_k_beta

    if initial_state is None:
        state = torch.zeros(state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)
    out = torch.zeros_like(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(padded_output_shape)[:, :, :sequence_length]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, (state if output_final_state else None)


def install() -> bool:
    """Route MPS calls of the Qwen3.5 chunk rule through the version above.

    Idempotent. Returns True if the reference implementation was wrapped; False if the
    model module is absent or flash-linear-attention can run (its kernel is bound
    instead, and it would be the faster path wherever it runs).
    """
    global _installed
    if _installed:
        return True
    import importlib.util

    # fla's kernels are Triton, which has no macOS build. Without it, `fla` still imports,
    # transformers binds its reference chunk rule, and this replacement is still needed.
    find = importlib.util.find_spec
    if find("fla") is not None and find("triton") is not None:
        return False
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    except ImportError:
        return False
    current = q.torch_chunk_gated_delta_rule

    def dispatch(query: torch.Tensor, *args: Any,
                 **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.device.type == "mps":
            return chunk_gated_delta_rule_mps(query, *args, **kwargs)
        return current(query, *args, **kwargs)  # type: ignore[no-any-return]

    dispatch.__wrapped__ = current  # type: ignore[attr-defined]
    q.torch_chunk_gated_delta_rule = dispatch
    _installed = True
    return True
