"""The MPS chunk rule must match transformers' reference Gated DeltaNet chunk rule.

Runs on CPU (the replacement is device-agnostic; `install()` only routes MPS to it), and
on MPS as well when one is present. No model weights are needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

q35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from strands_decider.mps_kernels import (  # noqa: E402
    _unit_lower_inverse,
    chunk_gated_delta_rule_mps,
)

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
reference = getattr(q35.torch_chunk_gated_delta_rule, "__wrapped__", q35.torch_chunk_gated_delta_rule)


def _inputs(seq: int, device: str, heads: int = 4, dim: int = 32, batch: int = 2):
    g = torch.Generator().manual_seed(seq)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    # A shared direction across tokens makes the keys correlated, as in a real model;
    # independent random keys hide the conditioning problem the kernel must survive.
    shared = 2.0 * r(batch, 1, heads, dim)
    return dict(
        query=r(batch, seq, heads, dim).to(device),
        key=(r(batch, seq, heads, dim) + shared).to(device),
        value=r(batch, seq, heads, dim).to(device),
        g=(-torch.rand(batch, seq, heads, generator=g) * 0.1).to(device),
        beta=(0.5 + 0.5 * torch.rand(batch, seq, heads, generator=g)).to(device),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_unit_lower_inverse(device):
    a = torch.randn(3, 64, 64).tril(-1) * 0.2 + torch.eye(64) * 7.0  # diagonal is ignored
    inv = _unit_lower_inverse(a.to(device)).cpu()
    unit = a.tril(-1) + torch.eye(64)
    assert torch.allclose(unit @ inv, torch.eye(64).expand(3, 64, 64), atol=1e-5)


def test_unit_lower_inverse_uses_metal_fast_path(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    from strands_decider import mps_kernels

    library = torch.mps.compile_shader(mps_kernels._UNIT_LOWER_INVERSE_METAL)
    # Record the dispatch while still executing the real Metal kernel.
    kernel = Mock(wraps=library.unit_lower_inverse_64)
    monkeypatch.setattr(
        mps_kernels, "_unit_lower_inverse_metal_library",
        SimpleNamespace(unit_lower_inverse_64=kernel),
    )
    a = torch.randn(3, 64, 64, dtype=torch.float32) * 0.2 + torch.eye(64) * 7.0
    want = _unit_lower_inverse(a)  # CPU uses the existing PyTorch implementation.
    mps_a = a.to("mps").contiguous()
    got = _unit_lower_inverse(mps_a)

    kernel.assert_called_once()
    args, kwargs = kernel.call_args
    assert args[0] is mps_a and args[1] is got
    assert kwargs == {"threads": [3 * 256, 1, 1], "group_size": [256, 1, 1]}
    assert got.shape == mps_a.shape and got.dtype == mps_a.dtype and got.device == mps_a.device
    torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_unit_lower_inverse_correlated_keys(device):
    # The chunk rule's system is the strictly-lower Gram matrix of l2-normalised keys
    # (times beta and decay). When a chunk's keys share a strong common direction, as
    # real Qwen3.5 activations do, the powers of that matrix reach ~1e8 before
    # cancelling: a Neumann product (I-N)(I+N^2)... is then wrong by ~1e8 relative, and
    # so was the first version of this kernel. Random matrices do not show this.
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(4, 64, 128, generator=g) + 2.0 * torch.randn(4, 1, 128, generator=g)
    keys = torch.nn.functional.normalize(keys, dim=-1)
    a = (keys @ keys.transpose(-1, -2)).tril(-1) + torch.eye(64)
    want = torch.linalg.inv(a.double())
    got = _unit_lower_inverse(a.to(device)).cpu().double()
    assert ((got - want).abs().max() / want.abs().max()).item() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seq", [1, 63, 64, 200])
@pytest.mark.parametrize("with_state", [False, True])
def test_matches_reference(device, seq, with_state):
    x = _inputs(seq, device)
    state = torch.randn(2, 4, 32, 32).to(device) if with_state else None
    kw = dict(initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    want_out, want_state = reference(**x, **kw)
    got_out, got_state = chunk_gated_delta_rule_mps(**x, **kw)
    assert got_out.dtype == want_out.dtype and got_out.shape == want_out.shape
    torch.testing.assert_close(got_out, want_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_state, want_state, rtol=1e-4, atol=1e-4)


def test_install_routes_only_mps():
    from strands_decider import mps_kernels

    if not mps_kernels.install():
        pytest.skip("flash-linear-attention is installed; nothing to replace")
    x = _inputs(70, "cpu")
    # CPU calls still reach the reference, bit for bit.
    out, _ = q35.torch_chunk_gated_delta_rule(**x, use_qk_l2norm_in_kernel=True)
    ref, _ = reference(**x, use_qk_l2norm_in_kernel=True)
    assert torch.equal(out, ref)


@pytest.mark.parametrize(("triton", "wrapped"), [(False, True), (True, False)],
                         ids=["fla-without-triton", "fla-with-triton"])
def test_install_defers_to_fla_only_when_it_can_run(monkeypatch, triton, wrapped):
    import importlib.util

    from strands_decider import mps_kernels

    monkeypatch.setattr(mps_kernels, "_installed", False)
    monkeypatch.setattr(q35, "torch_chunk_gated_delta_rule", q35.torch_chunk_gated_delta_rule)
    original = q35.torch_chunk_gated_delta_rule
    real_find_spec = importlib.util.find_spec
    present = {"fla"} | ({"triton"} if triton else set())

    def find_spec(name, *args, **kwargs):
        if name in {"fla", "triton"}:
            return object() if name in present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert mps_kernels.install() is wrapped
    assert (q35.torch_chunk_gated_delta_rule is not original) is wrapped
