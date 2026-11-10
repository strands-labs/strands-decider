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
    _no_mlx_import(monkeypatch)
    return seen


def _no_mlx_import(monkeypatch):
    """Checking a device's dtypes imports its loader; give mlx a stand-in, so that these tests
    need no mlx."""

    def loader(path, **kw):
        raise AssertionError("routing only")

    loader.dtypes = ("bfloat16", "float32")
    monkeypatch.setattr(decision, "_LOADERS", {**decision._LOADERS, "mlx": loader})


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
    _no_mlx_import(monkeypatch)
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


# ---- the MLX format rules, without MLX ---------------------------------------------------------

from strands_decider.backends import mlx_format  # noqa: E402

FULL, SLIDING = "full_attention", "sliding_attention"


def _config(per_layer, **text):
    return {
        "model_type": "gemma4",
        "text_config": {
            "layer_types": [SLIDING, FULL, SLIDING, FULL],
            "head_dim": 16,
            "num_key_value_heads": 2,
            **text,
            "per_layer_config": per_layer,
        },
    }


def test_bases_form_reads_transformers_5_2x_configs():
    out = mlx_format.bases_form(_config({"1": {"head_dim": 64}, "3": {"head_dim": 64}}))
    t = out["text_config"]
    assert "per_layer_config" not in t
    assert (t["global_head_dim"], t["num_global_key_value_heads"]) == (64, 2)  # kv: the base's
    out = mlx_format.bases_form(
        _config(
            {
                "1": {"head_dim": 64, "num_key_value_heads": 1},
                "3": {"head_dim": 64, "num_key_value_heads": 1},
            }
        )
    )
    assert out["text_config"]["num_global_key_value_heads"] == 1
    assert mlx_format.bases_form({"text_config": {"head_dim": 16, "global_head_dim": 64}}) is None
    assert "per_layer_config" not in mlx_format.bases_form(_config({}))["text_config"]


@pytest.mark.parametrize(
    ("per_layer", "text", "message"),
    [
        ({"1": {"head_dim": 512}}, {}, "different head_dims: layer 1: 512, layer 3: 16"),
        ({"0": {"head_dim": 64}}, {}, "overrides layer 0, a sliding_attention layer"),
        ({"9": {"head_dim": 64}}, {}, "names no layer: '9'"),
        ({"x": {"head_dim": 64}}, {}, "names no layer: 'x'"),
        ({"1": {"sliding_window": 4}}, {}, "reads only"),
        ({"1": {"head_dim": None}, "3": {"head_dim": None}}, {}, r"\['head_dim'\] is None"),
        ({"1": {"head_dim": True}, "3": {"head_dim": True}}, {}, r"\['head_dim'\] is True"),
        ([{"head_dim": 64}], {}, "must be a mapping"),
        (
            {"1": {"head_dim": 64}, "3": {"head_dim": 64}},
            {"global_head_dim": 512},
            "global_head_dim is 512, and its per_layer_config gives the full-attention layers head_dim 64",
        ),
        (
            {"1": {"head_dim": 64}, "3": {"head_dim": 64}},
            {"num_key_value_heads": None},
            "num_key_value_heads None",
        ),
    ],
)
def test_bases_form_refuses_what_it_cannot_map(per_layer, text, message):
    with pytest.raises(ValueError, match=message):
        mlx_format.bases_form(_config(per_layer, **text))


def test_mlx_artifacts_are_told_apart():
    assert mlx_format.is_mlx_artifact({mlx_format.ARTIFACT_KEY: dict(mlx_format.ARTIFACT)})
    assert mlx_format.is_mlx_artifact({"quantization": {"bits": 4, "group_size": 64}})
    assert not mlx_format.is_mlx_artifact({"quantization_config": {"quant_method": "gptq"}})
    assert not mlx_format.is_mlx_artifact({"quantization": {"bits": 4, "quant_method": "awq"}})
    assert not mlx_format.is_mlx_artifact({"model_type": "gemma4"})


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_an_mlx_artifact_is_refused_off_mlx_before_anything_loads(ckpt, device, monkeypatch):
    """Only decision_config.json and the marked config.json are here: the refusal comes
    before the readout, the tokenizer or any backend is read."""
    (ckpt / "config.json").write_text(
        json.dumps({"model_type": "gemma4", mlx_format.ARTIFACT_KEY: mlx_format.ARTIFACT})
    )
    monkeypatch.setitem(decision.BACKENDS, "cpu", "nonexistent.module:loader")
    monkeypatch.setitem(decision.BACKENDS, "mps", "nonexistent.module:loader")
    monkeypatch.setitem(decision.BACKENDS, "cuda", "nonexistent.module:loader")
    monkeypatch.setattr(decision, "_IMPORTED", {"nonexistent.module:loader": lambda *a, **k: None})
    monkeypatch.setattr(decision, "check_dtype", lambda device, dtype: None)
    with pytest.raises(ValueError, match=r"is an MLX artifact .* use --device mlx, not"):
        decision.load_decision_engine(str(ckpt), device=device)


def test_the_conformance_claim_comes_from_the_core_and_the_marker():
    from strands_decider.core import BOUNDS
    from strands_decider.core.conformance import MIN_QUESTIONS

    e2b, marked = "google/gemma-4-E2B-it", {mlx_format.ARTIFACT_KEY: dict(mlx_format.ARTIFACT)}
    q8 = {"bits": 8, "group_size": 64, "mode": "affine"}
    want = lambda key: {  # noqa: E731
        "top1": BOUNDS[key][0],
        "mean_abs_dp": BOUNDS[key][1],
        "min_questions": MIN_QUESTIONS,
    }
    bf16 = mlx_format.conformance({}, e2b)
    assert (bf16["claim"], bf16["bounds"]) == ("bf16", want(("bf16", None)))
    assert mlx_format.conformance({}, "google/gemma-4-31B-it")["claim"] == "bf16"  # every size
    recorded = mlx_format.conformance({**marked, "quantization": q8}, e2b)
    assert (recorded["claim"], recorded["weight_format"], recorded["size"]) == (
        "recorded",
        "mlx-8bit",
        "E2B",
    )
    assert recorded["bounds"] == want(("mlx-8bit", "E2B")) and not recorded["notice"]
    for config, base, why in (
        ({"quantization": q8}, e2b, "convert-mlx did not write it"),  # no marker
        ({**marked, "quantization": q8}, "google/gemma-4-31B-it", "at E2B only, not at 31B"),
        ({**marked, "quantization": {**q8, "group_size": 32}}, e2b, "records no bounds for MLX"),
        ({**marked, "quantization": {**q8, "bits": 4}}, e2b, "records no bounds for MLX 4-bit"),
        ({**marked, "quantization": {**q8, "mode": "mxfp8"}}, e2b, "records no bounds for MLX"),
        ({**marked, "quantization": q8}, "someone/other-model", "not at someone/other-model"),
    ):
        claim = mlx_format.conformance(config, base)
        assert claim["claim"] == "none" and claim["bounds"] is None
        assert "makes no conformance claim" in claim["notice"] and why in claim["notice"]


@pytest.mark.parametrize(
    "quantization",
    [{"bits": 8}, {"bits": "8", "group_size": 64}, {"bits": 8, "group_size": True}, [8, 64]],
)
def test_a_malformed_quantization_is_refused_cleanly(quantization):
    with pytest.raises(ValueError, match=r"config\.json's quantization"):
        mlx_format.conformance({"quantization": quantization}, "google/gemma-4-E2B-it")


def test_only_the_tested_mlx_vlm_is_accepted():
    mlx_format.check_mlx_vlm(mlx_format.TESTED_MLX_VLM)
    with pytest.raises(RuntimeError, match="tested on mlx-vlm"):
        mlx_format.check_mlx_vlm("0.8.0")


def test_core_resized_size_is_what_soft_tokens_counts():
    from core_support import SOFT_TOKENS

    from strands_decider.core.images import resized_size, soft_tokens

    table = json.loads(SOFT_TOKENS.read_text())
    col = table["columns"].index("Gemma4ImageProcessor")
    for row in table["rows"]:
        w, h = resized_size(row[0], row[1])
        assert (
            w % 48 == 0
            and h % 48 == 0
            and (w // 48) * (h // 48) == row[col] == soft_tokens(row[0], row[1])
        )


def test_cpu_and_mps_load_through_the_transformers_backend():
    pytest.importorskip("torch")
    for device in ("cpu", "mps"):
        assert decision.backend_loader(device).__module__ == "strands_decider.backends.transformers"


def test_serve_keeps_v1s_mlx_engine_for_a_v1_checkpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    seen = []
    monkeypatch.setattr(server, "load_mlx", lambda checkpoint, config: seen.append(checkpoint))
    with TestClient(server.create_app(str(tmp_path), device="mlx", warmup=False)):
        pass
    assert seen == [str(tmp_path)]
