"""The vLLM version gate: on any vLLM but 0.34.x the plugin warns and registers nothing, since vLLM
doesn't catch a plugin's exception and raising would stop every model it serves."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from strands_decider_vllm import register


def fake_vllm(monkeypatch, version):
    fake = types.ModuleType("vllm")  # no ModelRegistry: past the gate, register fails to import it
    fake.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", fake)


@pytest.mark.parametrize(
    "version", ["0.33.0", "0.35.0", "0.34.0rc1", "0.34.1.dev12+g1a2b", "1.34.0"]
)
def test_another_vllm_registers_nothing(monkeypatch, caplog, version):
    fake_vllm(monkeypatch, version)
    with caplog.at_level(logging.WARNING, logger="vllm.strands_decider"):
        register()
    assert f"needs vLLM 0.34.x, not {version}" in caplog.text


@pytest.mark.parametrize("version", ["0.34.0", "0.34.12", "0.34.0+cpu", "0.34.0+cu130"])
def test_vllm_034_passes_the_gate(monkeypatch, version):
    fake_vllm(monkeypatch, version)
    with pytest.raises(ImportError, match="ModelRegistry"):
        register()
