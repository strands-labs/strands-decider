"""The stock-attention control lives outside the package: the plugin has no switch that serves causal attention."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "causal_control"


def test_package_reads_no_environment_switch():
    for path in sorted((ROOT / "src" / "strands_decider_v2_vllm").glob("*.py")):
        text = path.read_text()
        for needle in ("os.environ", "getenv", "CAUSAL_CONTROL"):
            assert needle not in text, f"{path.name} contains {needle}"


def test_tool_is_not_packaged_and_matches_the_plugin_version():
    tomllib = pytest.importorskip("tomllib")
    import strands_decider_v2_vllm as plugin

    main = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert main["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    tool = tomllib.loads((TOOL / "pyproject.toml").read_text())["project"]
    assert tool["version"] == plugin.__version__
    assert tool["dependencies"] == [f"strands-decider-v2-vllm=={plugin.__version__}"]


@pytest.fixture
def control(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL))
    pytest.importorskip("vllm.v1.worker.gpu.model_states.default")
    import strands_decider_v2_causal_control as pkg
    from strands_decider_v2_causal_control import model

    yield pkg, model
    for name in [m for m in sys.modules if m.startswith("strands_decider_v2_causal_control")]:
        del sys.modules[name]


def test_control_uses_stock_layers_and_vllm_model_state(control):
    _, model = control
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    from strands_decider_v2_vllm.model import StrandsDeciderV2ForCausalLM
    from strands_decider_v2_vllm.model_state import PrefixBidirModelState

    cls = model.StrandsDeciderV2CausalControlForCausalLM
    assert issubclass(cls, StrandsDeciderV2ForCausalLM) and issubclass(cls, Gemma4ForCausalLM)
    state = cls.get_model_state_cls()
    assert issubclass(state, DefaultModelState) and not issubclass(state, PrefixBidirModelState)


def test_control_registers_its_own_architecture(control):
    pkg, _ = control
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    pkg.register()
    pkg.register()  # idempotent
    assert pkg.ARCH in ModelRegistry.get_supported_archs()
    assert pkg.ARCH != "StrandsDeciderV2ForCausalLM"
    assert MODELS_CONFIG_MAP[pkg.ARCH] is MODELS_CONFIG_MAP["Gemma4ForCausalLM"]
