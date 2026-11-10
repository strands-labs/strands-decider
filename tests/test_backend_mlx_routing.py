"""Which backend a decision checkpoint reaches on `--device mlx`, without MLX: the registry
entry, `ask` and `serve` routing to it, other devices routing to transformers, and v1
checkpoints keeping v1's MLX engine. Nothing here imports mlx, so it runs on any platform;
tests/test_backend_mlx.py runs the backend itself, on Apple silicon."""

from __future__ import annotations

import json

import pytest
from core_support import decision_config_fixture
from typer.testing import CliRunner

from strands_decider import cli, decision, server


@pytest.fixture
def ckpt(tmp_path):
    """A directory that is a decision checkpoint as far as routing can tell."""
    (tmp_path / "decision_config.json").write_text(json.dumps(decision_config_fixture()))
    return tmp_path


class _Engine:
    takes_images = True

    def __init__(self, device):
        self.device = device

    def ask(self, state, questions, images=()):
        from strands_decider.schema import SystemOneResponse

        return SystemOneResponse(model="stub", answers={}, usage={})


@pytest.fixture
def loads(monkeypatch):
    seen = []

    def fake(checkpoint, *, device, **kw):
        seen.append(device)
        return _Engine(device)

    monkeypatch.setattr(decision, "load_decision_engine", fake)
    monkeypatch.setattr(cli, "_exit_fast", False)
    return seen


def test_mlx_is_registered_without_importing_mlx():
    assert decision.BACKENDS["mlx"] == "strands_decider.backends.mlx:load_backend"
    decision.check_device("mlx")
    assert decision.BACKENDS["mps"] == decision.BACKENDS["cpu"]  # transformers, not mlx


@pytest.mark.parametrize("device", ["mlx", "cpu", "mps"])
def test_ask_routes_a_decision_checkpoint_by_device(ckpt, loads, device):
    out = CliRunner().invoke(
        cli.app, ["ask", str(ckpt), "--state", "x", "--noul", "y?", "--device", device, "--json"]
    )
    assert out.exit_code == 0, out.output
    assert loads == [device]


def test_ask_keeps_v1s_mlx_engine_for_a_v1_checkpoint(tmp_path, loads, monkeypatch):
    from strands_decider import infer

    seen = []

    class _V1:
        def ask(self, state, questions):
            from strands_decider.schema import SystemOneResponse

            seen.append("v1")
            return SystemOneResponse(model="v1", answers={}, usage={})

    monkeypatch.setattr(infer, "load_engine", lambda checkpoint, device: _V1())
    out = CliRunner().invoke(
        cli.app, ["ask", str(tmp_path), "--state", "x", "--noul", "y?", "--device", "mlx"]
    )
    assert out.exit_code == 0, out.output
    assert seen == ["v1"] and loads == []  # no decision_config.json: v1's path


def test_serve_routes_a_decision_checkpoint_on_mlx(ckpt, monkeypatch):
    seen = []
    monkeypatch.setattr(
        server,
        "load_decision_engine",
        lambda checkpoint, *, device, **kw: seen.append((device, kw["reuse_state"])),
    )
    app = server.create_app(str(ckpt), device="mlx", use_prefix_cache=False, warmup=False)
    from fastapi.testclient import TestClient

    with TestClient(app):  # the lifespan builds the engine: here, records the call
        pass
    assert seen == [("mlx", False)]
