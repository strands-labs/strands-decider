"""Device selection: what `--device` resolves to, and how a missing MLX or Neuron install is reported.

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
    app = server.create_app("org/hobson", device="mlx", strict_window=True, max_batch=7)
    assert TestClient(app).get("/health").json()["device"] == "mlx"
    [(checkpoint, config)] = calls
    assert checkpoint == "org/hobson"
    assert (config.device, config.use_prefix_cache, config.model_name) == ("mlx", True, "hobson")
    assert (config.strict_window, config.max_batch) == (True, 7)  # the shared config reaches MLX


def test_device_neuron_without_the_extra_names_it_before_loading_anything(monkeypatch):
    monkeypatch.setattr(infer, "neuron_available", lambda: False)
    monkeypatch.setattr(
        infer.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("nothing should load")
    )
    with pytest.raises(RuntimeError, match=r"strands-decider\[neuron\]"):
        infer.load_engine("any/checkpoint", device="neuron")


def test_the_server_builds_and_warms_the_neuron_engine_for_device_neuron(monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from strands_decider import server

    class _Engine:
        cfg = type(
            "Cfg", (), {"model_name": "hobson", "device": "neuron", "use_prefix_cache": False}
        )()
        model = type(
            "Model",
            (),
            {
                "config": type(
                    "Config",
                    (),
                    {"base_model": "stub", "temperature": 1.0, "max_length": 512},
                )()
            },
        )()
        warmed = False

        def warm(self, log=None):
            self.warmed = True
            log({"core": 0, "bucket": 128, "first_call_s": 1.0})

    engine = _Engine()
    calls = []
    monkeypatch.setattr(
        server,
        "load_neuron",
        lambda checkpoint, config, cache_dir: (
            calls.append((checkpoint, config, cache_dir)) or engine
        ),
    )
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("torch load")
    )
    app = server.create_app("org/hobson", device="neuron", max_batch=7, neuron_cache="/cache")
    assert engine.warmed  # every graph loads before the server answers
    assert '"bucket": 128' in capsys.readouterr().out  # and it reports progress
    assert TestClient(app).get("/health").json()["device"] == "neuron"
    [(checkpoint, config, cache_dir)] = calls
    assert (checkpoint, cache_dir, config.max_batch) == ("org/hobson", "/cache", 7)


def test_vision_is_refused_on_neuron():
    from strands_decider import server

    with pytest.raises(ValueError, match="not neuron"):
        server.create_app("org/hobson", device="neuron", vision=True)


def test_compile_neuron_without_the_extra_is_refused(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    monkeypatch.setattr(infer, "neuron_available", lambda: False)
    result = CliRunner().invoke(cli.app, ["compile-neuron", "any/checkpoint", str(tmp_path)])
    assert result.exit_code != 0
    assert "neuron extra" in result.output


def test_neuron_cache_needs_device_neuron(tmp_path):
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        cli.app, ["serve", "any/checkpoint", "--device", "cpu", "--neuron-cache", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "--device neuron" in result.output
