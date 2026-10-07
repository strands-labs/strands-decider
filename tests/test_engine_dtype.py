"""SystemOneEngine runs a half-precision torso in fp32 on CPU unless asked to keep it, and leaves it alone elsewhere."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from strands_decider.infer import EngineConfig, SystemOneEngine


class _TinyModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.torso = nn.Linear(4, 4).to(dtype)
        self.head = nn.Linear(4, 2)  # fp32, as in StrandsDeciderModel
        self.tokenizer = object()


def _build(dtype, device, under_inference_mode):
    if under_inference_mode:  # as load_engine does
        with torch.inference_mode():
            return SystemOneEngine(_TinyModel(dtype), EngineConfig(device=device))
    return SystemOneEngine(_TinyModel(dtype), EngineConfig(device=device))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("under_inference_mode", [True, False])
def test_cpu_upcasts_half_precision_torso(dtype, under_inference_mode):
    eng = _build(dtype, "cpu", under_inference_mode)
    assert {p.dtype for p in eng.model.torso.parameters()} == {torch.float32}
    assert {p.dtype for p in eng.model.head.parameters()} == {torch.float32}
    with torch.inference_mode():  # the engine's forward paths all run like this
        assert eng.model.torso(torch.ones(1, 4)).dtype == torch.float32


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_keeps_bf16():
    eng = _build(torch.bfloat16, "mps", under_inference_mode=True)
    assert {p.dtype for p in eng.model.torso.parameters()} == {torch.bfloat16}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cpu_checkpoint_dtype_keeps_half_precision_torso(dtype):
    with torch.inference_mode():
        eng = SystemOneEngine(_TinyModel(dtype),
                              EngineConfig(device="cpu", cpu_torso_dtype="checkpoint"))
    assert {p.dtype for p in eng.model.torso.parameters()} == {dtype}
    assert {p.dtype for p in eng.model.head.parameters()} == {torch.float32}


def test_unknown_cpu_torso_dtype_is_rejected():
    with pytest.raises(ValueError, match="cpu_torso_dtype"):
        SystemOneEngine(_TinyModel(torch.bfloat16),
                        EngineConfig(device="cpu", cpu_torso_dtype="bf16"))
