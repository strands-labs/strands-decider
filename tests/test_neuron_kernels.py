"""The static-shape Neuron kernels against transformers' reference implementations, on CPU."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from transformers.models.qwen3_5 import modeling_qwen3_5 as q  # noqa: E402

from strands_decider import neuron_kernels  # noqa: E402


def _reference_rule():
    rule = q.torch_chunk_gated_delta_rule
    while hasattr(rule, "__wrapped__"):  # another test may have installed a dispatcher
        rule = rule.__wrapped__
    return rule


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("length", [64, 200])  # 200: not a multiple of the chunk, so padded
@pytest.mark.parametrize("with_state", [False, True])
def test_chunk_rule_matches_the_reference(length, with_state):
    torch.manual_seed(0)
    b, h, d = 1, 4, 32
    query, key, value = (torch.randn(b, length, h, d) for _ in range(3))
    g = -F.softplus(torch.randn(b, length, h))  # log-decays are negative
    beta = torch.sigmoid(torch.randn(b, length, h))
    state = torch.randn(b, h, d, d) * 0.1 if with_state else None
    kwargs = {"initial_state": state, "output_final_state": True, "use_qk_l2norm_in_kernel": True}
    want, want_state = _reference_rule()(query, key, value, g, beta, **kwargs)
    got, got_state = neuron_kernels.chunk_gated_delta_rule_static(
        query, key, value, g, beta, **kwargs
    )
    assert got.dtype == want.dtype
    assert _rel(got, want) < 1e-5
    assert _rel(got_state, want_state) < 1e-5


def test_chunk_rule_keeps_a_bf16_input_dtype():
    torch.manual_seed(0)
    query, key, value = (torch.randn(1, 96, 2, 16, dtype=torch.bfloat16) for _ in range(3))
    g = -F.softplus(torch.randn(1, 96, 2)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, 96, 2)).to(torch.bfloat16)
    want, _ = _reference_rule()(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)
    got, _ = neuron_kernels.chunk_gated_delta_rule_static(
        query, key, value, g, beta, use_qk_l2norm_in_kernel=True
    )
    assert got.dtype == torch.bfloat16
    assert _rel(got, want) < 1e-2  # both round their fp32 result to bf16 once


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_matches_a_grouped_conv(dtype):
    torch.manual_seed(0)
    channels, length, width = 16, 50, 4
    x = torch.randn(2, channels, length, dtype=dtype)
    weight = torch.randn(channels, width, dtype=dtype)
    bias = torch.randn(channels, dtype=dtype)
    want = F.silu(
        F.conv1d(
            x.float(), weight.float().unsqueeze(1), bias.float(), padding=width - 1, groups=channels
        )[..., :length]
    )
    got = neuron_kernels.causal_conv1d_static(x, weight, bias, activation="silu")
    assert got.dtype == dtype
    assert _rel(got, want) < (1e-6 if dtype == torch.float32 else 1e-2)


def test_install_leaves_cpu_tensors_to_what_was_bound(monkeypatch):
    # Install afresh, and have monkeypatch put back whatever was bound before.
    monkeypatch.setattr(neuron_kernels, "_installed", False)
    monkeypatch.setattr(q, "torch_chunk_gated_delta_rule", q.torch_chunk_gated_delta_rule)
    monkeypatch.setattr(q, "causal_conv1d_fn", q.causal_conv1d_fn)
    bound = (q.torch_chunk_gated_delta_rule, q.causal_conv1d_fn)
    assert neuron_kernels.install()
    assert neuron_kernels.install()  # idempotent
    torch.manual_seed(0)
    args = [torch.randn(1, 64, 2, 16) for _ in range(3)]
    args += [-F.softplus(torch.randn(1, 64, 2)), torch.sigmoid(torch.randn(1, 64, 2))]
    out, _ = q.torch_chunk_gated_delta_rule(*args)
    assert (q.torch_chunk_gated_delta_rule.__wrapped__, q.causal_conv1d_fn.__wrapped__) == bound
    want, _ = bound[0](*args)
    assert torch.equal(out, want)
    x, weight = torch.randn(1, 8, 20), torch.randn(8, 4)
    assert torch.equal(q.causal_conv1d_fn(x, weight), bound[1](x, weight))
