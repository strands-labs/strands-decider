"""Device selection: what `--device` resolves to, and how a missing MLX install is reported.

Needs neither weights nor a GPU: the availability checks are replaced, so every ordering is
exercised on any machine, CI's Linux runners included.
"""

from __future__ import annotations

import pytest
import torch

from strands_decider import cli, infer


@pytest.mark.parametrize(
    ("cuda", "mlx", "mps", "expected"),
    [
        (True, True, True, "cuda"),
        (False, True, True, "mps"),  # mlx is opt-in: an installed extra does not change the default
        (False, True, False, "cpu"),
        (False, False, False, "cpu"),
    ],
)
def test_auto_device_is_cuda_then_mps_then_cpu_and_never_mlx(monkeypatch, cuda, mlx, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(infer, "mlx_available", lambda: mlx)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert cli._auto_device() == expected


def test_mlx_is_unavailable_off_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert not infer.mlx_available()


@pytest.mark.parametrize(
    ("found", "expected"),
    [
        ({"mlx.core", "mlx_lm"}, True),
        ({"mlx_lm"}, False),  # an `mlx` namespace directory left behind without mlx.core
        (None, False),  # no `mlx` at all: finding `mlx.core` raises
    ],
    ids=["installed", "namespace-leftover", "absent"],
)
def test_mlx_is_available_only_with_mlx_core(monkeypatch, found, expected):
    import importlib.util

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    def find_spec(name, *args, **kwargs):
        if found is None:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
        return object() if name in found else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert infer.mlx_available() is expected


def test_device_mlx_without_the_extra_names_it_before_loading_anything(monkeypatch):
    monkeypatch.setattr(infer, "mlx_available", lambda: False)

    def refuse(*args, **kwargs):
        raise AssertionError("nothing should load")

    monkeypatch.setattr(infer.StrandsDeciderModel, "load", refuse)
    with pytest.raises(RuntimeError, match=r"strands-decider\[mlx\]"):
        infer.load_engine("any/checkpoint", device="mlx")


def test_the_server_builds_the_mlx_engine_for_device_mlx(monkeypatch):
    from fastapi.testclient import TestClient

    from strands_decider import server

    class _Engine:
        cfg = type("Cfg", (), {"model_name": "hobson", "device": "mlx", "use_prefix_cache": True})()
        model = type(
            "Model",
            (),
            {
                "config": type(
                    "Config", (), {"base_model": "stub", "temperature": 1.0, "max_length": 512}
                )()
            },
        )()

    calls = []
    monkeypatch.setattr(
        server,
        "load_mlx",
        lambda checkpoint, config: calls.append((checkpoint, config)) or _Engine(),
    )
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("torch load")
    )
    # No warm-up: the stub engine only answers /health.
    app = server.create_app("org/hobson", device="mlx", max_batch=7, warmup=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["device"] == "mlx"
    [(checkpoint, config)] = calls
    assert checkpoint == "org/hobson"
    assert (config.device, config.use_prefix_cache, config.model_name) == ("mlx", True, "hobson")
    assert config.max_batch == 7  # the shared config reaches MLX


@pytest.mark.parametrize(
    ("decision", "cuda", "mlx", "mps", "expected"),
    [
        (True, True, True, True, "cuda"),
        (True, False, True, True, "mlx"),  # a model checkpoint on a Mac with the extra: MLX
        (True, False, False, True, "mps"),  # without it: MPS, with a note on stderr
        (True, False, False, False, "cpu"),
        (False, False, True, True, "mps"),  # a v1 checkpoint: MLX stays opt-in
    ],
)
def test_serve_runs_a_model_checkpoint_on_mlx_on_a_mac_with_the_extra(
    tmp_path, monkeypatch, decision, cuda, mlx, mps, expected
):
    import uvicorn
    from typer.testing import CliRunner

    from strands_decider import server

    if decision:
        (tmp_path / "decision_config.json").write_text('{"interface": "strands-decider"}')
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(cli, "_mlx_vlm_available", lambda: mlx)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    devices = []
    monkeypatch.setattr(server, "create_app", lambda _, *, device, **__: devices.append(device))
    monkeypatch.setattr(uvicorn, "run", lambda *_, **__: None)
    out = CliRunner().invoke(cli.app, ["serve", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert devices == [expected]
    assert ("strands-decider[mlx]" in out.stderr) == (decision and expected == "mps")
