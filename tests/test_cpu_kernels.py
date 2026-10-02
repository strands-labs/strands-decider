"""The CPU causal conv must match transformers' reference `causal_conv1d_fn`. No weights needed."""

from __future__ import annotations

import pytest
import torch

q35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from strands_decider.cpu_kernels import causal_conv1d_cpu  # noqa: E402

reference = getattr(q35.causal_conv1d_fn, "__wrapped__", q35.causal_conv1d_fn)


@pytest.mark.parametrize("seq", [1, 3, 4, 70, 257])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_reference(seq, width, activation, with_bias):
    g = torch.Generator().manual_seed(seq * 10 + width)
    x = torch.randn(2, 48, seq, generator=g)
    w = torch.randn(48, width, generator=g)
    b = torch.randn(48, generator=g) if with_bias else None
    want = reference(x, w, b, activation)
    got = causal_conv1d_cpu(x, w, b, activation)
    assert got.dtype == want.dtype and got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_bf16_input_follows_weight_dtype():
    # The reference computes in the weight dtype and returns the input dtype.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, 20, generator=g).to(torch.bfloat16)
    w = torch.randn(16, 4, generator=g)
    want = reference(x, w, None, "silu")
    got = causal_conv1d_cpu(x, w, None, "silu")
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want)


def test_install_routes_cpu_and_is_idempotent():
    from strands_decider import cpu_kernels

    assert cpu_kernels.install()
    bound = q35.causal_conv1d_fn
    assert cpu_kernels.install()
    assert q35.causal_conv1d_fn is bound  # not wrapped twice
    x, w = torch.randn(1, 8, 9), torch.randn(8, 4)
    torch.testing.assert_close(bound(x, w, None, "silu"), reference(x, w, None, "silu"))
