"""VLLMBackend adapts vLLM's offline encode output to HiddenStateEngine's Backend contract.

vLLM is not installed in CI, so the test stubs `vllm`, `vllm.config` and the `LLM` and
`PoolingParams` classes it uses: the stub records the prompts and returns a tensor of the
right shape. The backend must slice the hidden-state rows at the positions the head reads,
with the question's rows offset by the state's length.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from strands_decider.hidden_engine import HiddenStateEngine  # noqa: E402


class _FakePoolerConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakePoolingParams:
    pass


class _FakeOutput:
    def __init__(self, data):
        self.outputs = types.SimpleNamespace(data=data)


class _FakeLLM:
    """Stands in for vLLM's offline LLM: records constructor kwargs and encode prompts,
    returns a tensor that reads [prompt_index + position_index] at each position so the
    sliced rows can be checked numerically."""

    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.encoded = []
        _FakeLLM.instances.append(self)

    def encode(self, prompts, pooling_params, use_tqdm):
        self.encoded.append([list(p["prompt_token_ids"]) for p in prompts])
        outs = []
        for i, p in enumerate(prompts):
            n = len(p["prompt_token_ids"])
            data = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 3) + i * 1000.0
            outs.append(_FakeOutput(data))
        return outs


@pytest.fixture
def fake_vllm(monkeypatch):
    _FakeLLM.instances = []
    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.PoolingParams = _FakePoolingParams
    config = types.ModuleType("vllm.config")
    config.PoolerConfig = _FakePoolerConfig
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    return vllm


def test_backend_forward_slices_rows_offset_by_state_length(fake_vllm):
    from strands_decider.vllm_engine import VLLMBackend, VLLMBackendConfig

    backend = VLLMBackend(VLLMBackendConfig(model="stub"))
    state = [1, 2, 3, 4]
    questions = [[10, 11, 12], [20, 21]]
    rows = [[0, 2], [1]]
    out = backend.forward(state, questions, rows)
    assert len(out) == 2
    # Prompt 0 is state (4 tokens) + question 0 (3 tokens): rows 0 and 2 of the question lie
    # at positions 4 and 6 of the prompt. The fake fills column c with position c + 0.
    np.testing.assert_allclose(out[0], np.array([[4, 4, 4], [6, 6, 6]], dtype=np.float32))
    # Prompt 1 adds 1000 to every row: row 1 is position 5 -> 1005.
    np.testing.assert_allclose(out[1], np.array([[1005, 1005, 1005]], dtype=np.float32))
    # Each prompt is submitted as state + question.
    assert _FakeLLM.instances[0].encoded == [[state + questions[0], state + questions[1]]]


def test_backend_passes_pooler_config(fake_vllm):
    from strands_decider.vllm_engine import VLLMBackend, VLLMBackendConfig

    VLLMBackend(
        VLLMBackendConfig(model="stub", max_model_len=4096, extra={"trust_remote_code": True})
    )
    kwargs = _FakeLLM.instances[0].kwargs
    assert kwargs["runner"] == "pooling"
    assert kwargs["override_pooler_config"].kwargs == {
        "pooling_type": "ALL",
        "normalize": False,
        "softmax": False,
    }
    assert kwargs["max_model_len"] == 4096
    assert kwargs["trust_remote_code"] is True


def test_backend_refuses_a_wrong_shape(fake_vllm):
    """A pooler that returned one pooled vector per prompt, or truncated, is caught here."""
    from strands_decider.vllm_engine import VLLMBackend, VLLMBackendConfig

    class _ShortLLM(_FakeLLM):
        def encode(self, prompts, pooling_params, use_tqdm):
            # One row per prompt instead of per token.
            return [_FakeOutput(torch.zeros(1, 3)) for _ in prompts]

    fake_vllm.LLM = _ShortLLM
    backend = VLLMBackend(VLLMBackendConfig(model="stub"))
    with pytest.raises(RuntimeError, match="hidden rows"):
        backend.forward([1, 2], [[3, 4, 5]], [[0]])


def test_load_vllm_engine_wires_to_hidden_state_engine(tmp_path, fake_vllm, monkeypatch):
    """load_vllm_engine returns a HiddenStateEngine with a VLLMBackend attached."""
    from strands_decider.vllm_engine import VLLMBackend, VLLMBackendConfig, load_vllm_engine

    created = {}

    def fake_from_assets(path, backend, **config):
        created["path"], created["backend"], created["config"] = path, backend, config
        return "engine-sentinel"

    monkeypatch.setattr(HiddenStateEngine, "from_assets", staticmethod(fake_from_assets))
    out = load_vllm_engine(str(tmp_path), VLLMBackendConfig(model="stub"), max_request_tokens=1024)
    assert out == "engine-sentinel"
    assert created["path"] == str(tmp_path)
    assert isinstance(created["backend"], VLLMBackend)
    assert created["config"] == {"max_request_tokens": 1024}
