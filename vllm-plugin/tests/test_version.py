"""The vLLM version gate (0.33.x only) and the package metadata that must agree with it."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
import types
from pathlib import Path

import pytest

import strands_decider_vllm as plugin
from strands_decider_vllm import check_vllm_version, vllm_version_supported

ROOT = Path(__file__).resolve().parents[1]

SUPPORTED = ["0.33.0", "0.33.1", "0.33.12", "0.33.0+cu130", "0.33.2+cu130.torch2.14", " 0.33.0\n"]
UNSUPPORTED = [
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
]


@pytest.mark.parametrize("version", SUPPORTED)
def test_accepts_033_releases(version):
    assert vllm_version_supported(version)
    check_vllm_version(version)


@pytest.mark.parametrize("version", UNSUPPORTED)
def test_refuses_everything_else(version):
    assert not vllm_version_supported(version)
    with pytest.raises(RuntimeError) as err:
        check_vllm_version(version)
    message = str(err.value)
    assert repr(version) in message, "the error names the installed version"
    assert "0.33.x" in message and plugin.SUPPORTED_VLLM in message, (
        "the error names the supported range"
    )


def test_supported_range_is_033_only():
    assert plugin.SUPPORTED_VLLM == ">=0.33.0,<0.34"


@pytest.mark.parametrize("version", ["0.32.0", "0.34.0", "0.33.0rc1"])
def test_register_on_another_vllm_warns_and_registers_nothing(monkeypatch, caplog, version):
    # vLLM's plugin loader does not catch exceptions: raising would stop every model served from the environment.
    fake = types.ModuleType("vllm")
    fake.__version__ = version  # no ModelRegistry: register must not touch vLLM beyond the version
    monkeypatch.setitem(sys.modules, "vllm", fake)
    with caplog.at_level(logging.WARNING, logger="vllm.strands_decider"):
        plugin.register()
    (record,) = [r for r in caplog.records if r.name == "vllm.strands_decider"]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert repr(version) in message and "registered nothing" in message and plugin.ARCH in message


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
        "strands_decider": "strands_decider_vllm:register"
    }
