"""A faster depthwise causal convolution for CPU inference of a Qwen3.5 torso.

Each Gated DeltaNet layer runs a short depthwise causal convolution (kernel width 4,
one filter per channel) over its q/k/v projections. Without the `causal_conv1d` CUDA
package, transformers' `causal_conv1d_fn` falls back to
`F.conv1d(..., groups=channels)`. On CPU, PyTorch runs that grouped convolution as one
`_slow_conv2d_forward` call per channel: 6,144 channels times 18 layers is 110,592 calls
per forward. Profiled on an M3 Pro, that was 2.0 of 2.7 s for a 70-token v19 question.

With width 4, the convolution is four shifted multiply-adds over the whole
[batch, channels, length] tensor. That takes 8 ms for all 18 layers instead of 2,026 ms,
and agrees with the reference to within float rounding (`tests/test_cpu_kernels.py`).

`install()` swaps it in for CPU tensors only, the same way `mps_kernels.install()` does
for the chunk rule. Other devices still reach whatever transformers bound.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_installed = False


def causal_conv1d_cpu(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Same contract as transformers' `causal_conv1d_fn`.

    `hidden_states` is [B, C, L] and `weight` is [C, K]. Output position t is
    sum_k weight[:, k] * x[t - (K - 1) + k], with zeros before the sequence starts.
    Accumulation is in the weight dtype, as in the reference, and the result is cast
    back to the input dtype.
    """
    from transformers.activations import ACT2FN

    seq_len = hidden_states.shape[-1]
    width = weight.shape[-1]
    x = F.pad(hidden_states.to(weight.dtype), (width - 1, 0))
    out = x[:, :, width - 1 : width - 1 + seq_len] * weight[:, width - 1, None]
    for k in range(width - 1):
        out = out + x[:, :, k : k + seq_len] * weight[:, k, None]
    if bias is not None:
        out = out + bias[:, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def install() -> bool:
    """Route CPU calls of the Qwen3.5 `causal_conv1d_fn` through the version above.

    Idempotent. Returns True if the module function was wrapped, or False if the
    Qwen3.5 model module (or that function) is not present in this transformers.
    """
    global _installed
    if _installed:
        return True
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    except ImportError:
        return False
    current = getattr(q, "causal_conv1d_fn", None)
    if current is None:
        return False

    def dispatch(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if hidden_states.device.type == "cpu":
            return causal_conv1d_cpu(hidden_states, *args, **kwargs)
        out: torch.Tensor = current(hidden_states, *args, **kwargs)
        return out

    dispatch.__wrapped__ = current  # type: ignore[attr-defined]
    q.causal_conv1d_fn = dispatch
    _installed = True
    return True
