"""The vLLM version gate (0.33.x only) and the package metadata that must agree with it."""

from __future__ import annotations

import importlib.metadata
import sys
import types
from pathlib import Path

import pytest

import strands_decider_v2_vllm as plugin
from strands_decider_v2_vllm import check_vllm_version

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "version",
    [
        "0.33.0",
        "0.33.1",
        "0.33.12",
        "0.33.0+cu130",
        "0.33.2+cu130.torch2.14",
        " 0.33.0\n",
    ],
)
def test_accepts_033_releases(version):
    check_vllm_version(version)


@pytest.mark.parametrize(
    "version",
    [
        "0.32.0",
        "0.32.3",
        "0.32.0+cu129",  # the previous minor release
        "0.31.1",
        "0.34.0",
        "1.0.0",
        "1.33.0",
        "0.3.3",
        "0.330.0",
        "10.33.0",  # other releases
        "0.33.0rc3",
        "0.33.1.dev12+g1234abc",
        "0.33.0.post1",
        "0.34.0rc1",  # pre-, dev and post releases
        "0.33",
        "0.33.x",
        "",
        "unknown",  # malformed
    ],
)
def test_refuses_everything_else(version):
    with pytest.raises(RuntimeError) as err:
        check_vllm_version(version)
    message = str(err.value)
    assert repr(version) in message, "the error names the installed version"
    assert "0.33.x" in message and plugin.SUPPORTED_VLLM in message, (
        "the error names the supported range"
    )


def test_supported_range_is_033_only():
    assert plugin.SUPPORTED_VLLM == ">=0.33.0,<0.34"


def test_register_refuses_032_before_touching_vllm(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.__version__ = "0.32.0"  # no ModelRegistry: register must stop at the version check
    monkeypatch.setitem(sys.modules, "vllm", fake)
    with pytest.raises(RuntimeError, match=r"requires vLLM 0\.33\.x"):
        plugin.register()


def test_installed_vllm_is_supported():
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vLLM is not installed")
    check_vllm_version(version)


def test_pyproject_agrees_with_package():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == plugin.__version__
    assert project["dependencies"] == [f"vllm{plugin.SUPPORTED_VLLM}"]
    assert project["entry-points"]["vllm.general_plugins"] == {
        "strands_decider_v2": "strands_decider_v2_vllm:register"
    }
