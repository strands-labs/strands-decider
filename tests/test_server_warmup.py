"""Startup warmup with stubs and tiny CPU models: no checkpoints, GPU, or network calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from strands_decider import cli, server
from strands_decider.schema import SystemOneResponse, Usage


@pytest.fixture
def engine(monkeypatch):
    model = SimpleNamespace(
        config=SimpleNamespace(
            base_model="stub",
            num_slots=2,
            max_length=512,
            temperature=1.0,
        )
    )

    class StubEngine:
        def __init__(self):
            self.model = model
            self.requests = []
            self.configs = []

        def evaluate(self, request):
            self.requests.append(request)
            self.configs.append(
                (
                    self.cfg.use_prefix_cache,
                    self.cfg.max_batch,
                    self.cfg.max_batch_tokens,
                    self.cfg.strict_window,
                )
            )
            return SystemOneResponse(model=self.cfg.model_name, answers={}, usage=Usage())

    stub = StubEngine()

    def build(model, config):
        stub.cfg = config
        return stub

    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server.StrandsDeciderModel, "load", lambda *a, **k: model)
    monkeypatch.setattr(server, "SystemOneEngine", build)
    monkeypatch.setattr(server, "load_mlx", lambda checkpoint, config: build(model, config))
    return stub


@pytest.mark.parametrize(
    ("device", "count"),
    [
        ("cuda", 4),
        ("cuda:0", 4),
        ("cuda:1", 4),
        ("cpu", 0),
        ("mps", 0),
        ("mlx", 0),
    ],
)
def test_warmup_defaults_to_cuda_only(engine, device, count):
    app = server.create_app("test/checkpoint", device=device)
    assert len(engine.requests) == count
    assert TestClient(app).get("/health").status_code == 200


@pytest.mark.parametrize(
    ("device", "enabled"),
    [
        ("cpu", True),
        ("mps", True),
        ("mlx", True),
        ("cuda", False),
        ("cuda:0", False),
    ],
)
def test_explicit_warmup_overrides_the_device_default(engine, device, enabled):
    server.create_app("test/checkpoint", device=device, warmup=enabled)
    assert len(engine.requests) == (4 if enabled else 0)


def test_warmup_covers_each_kind_alone_and_together(engine):
    server.create_app("test/checkpoint", device="cpu", warmup=True)
    assert len(engine.requests) == 4
    singles, combined = engine.requests[:3], engine.requests[3]
    assert [len(request.questions) for request in singles] == [1, 1, 1]
    assert {next(iter(request.questions.values())).type for request in singles} == {
        "noul",
        "choice",
        "score",
    }
    assert {question.type for question in combined.questions.values()} == {
        "noul",
        "choice",
        "score",
    }
    assert combined.questions == {
        name: question for request in singles for name, question in request.questions.items()
    }
    assert all(
        request.state == combined.state and not request.images for request in engine.requests
    )
    assert all(
        len(question.criteria) == 2
        for question in combined.questions.values()
        if question.type != "noul"
    )


@pytest.mark.parametrize("use_prefix_cache", [True, False])
@pytest.mark.parametrize("max_batch", [1, 2, 32])
def test_warmup_preserves_inference_settings(engine, use_prefix_cache, max_batch):
    app = server.create_app(
        "test/checkpoint",
        device="cpu",
        warmup=True,
        use_prefix_cache=use_prefix_cache,
        max_batch=max_batch,
        max_batch_tokens=64,
        strict_window=True,
    )
    expected = (use_prefix_cache, max_batch, 64, True)
    assert engine.configs == [expected] * 4
    assert (
        engine.cfg.use_prefix_cache,
        engine.cfg.max_batch,
        engine.cfg.max_batch_tokens,
        engine.cfg.strict_window,
    ) == expected
    assert TestClient(app).get("/health").json()["prefix_cache"] is use_prefix_cache


def test_warmup_uses_the_vision_loader_but_sends_text_only(engine, monkeypatch):
    from strands_decider import vision

    calls = []

    def load(checkpoint, config, **kwargs):
        calls.append(checkpoint)
        engine.cfg = config
        return engine

    monkeypatch.setattr(vision, "load_vision_engine", load)
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("text load")
    )
    app = server.create_app("test/checkpoint", device="cuda", vision=True)
    assert calls == ["test/checkpoint"]
    assert len(engine.requests) == 4
    assert all(not request.images for request in engine.requests)
    assert TestClient(app).get("/health").json()["vision"] is True


def test_serve_finishes_warmup_before_starting_uvicorn(engine, monkeypatch):
    import uvicorn

    calls = []

    def run(app, **kwargs):
        assert len(engine.requests) == 4
        assert server.get_engine() is engine
        assert TestClient(app).get("/health").status_code == 200
        calls.append(kwargs)

    monkeypatch.setattr(uvicorn, "run", run)
    server.serve("test/checkpoint", device="cpu", warmup=True, host="127.0.0.1", port=8099)
    assert calls == [{"host": "127.0.0.1", "port": 8099, "workers": 1}]


@pytest.mark.parametrize("failed_call", [1, 4])
def test_warmup_failure_prevents_server_start(engine, monkeypatch, failed_call):
    import uvicorn

    calls = 0

    def fail(request):
        nonlocal calls
        calls += 1
        assert server._engine is None
        if calls == failed_call:
            raise RuntimeError("warmup failed")
        return SystemOneResponse(model=engine.cfg.model_name, answers={}, usage=Usage())

    monkeypatch.setattr(engine, "evaluate", fail)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("server started"))
    with pytest.raises(RuntimeError, match="warmup failed"):
        server.serve("test/checkpoint", device="cuda")
    assert calls == failed_call
    assert server._engine is None


@pytest.mark.parametrize(
    ("device_args", "flag", "expected_device", "expected"),
    [
        (["--device", "cpu"], [], "cpu", None),
        (["--device", "cpu"], ["--warmup"], "cpu", True),
        (["--device", "cuda:0"], ["--no-warmup"], "cuda:0", False),
        ([], [], "cuda", None),
        ([], ["--no-warmup"], "cuda", False),
    ],
)
def test_cli_forwards_warmup(engine, monkeypatch, device_args, flag, expected_device, expected):
    calls = []
    monkeypatch.setattr(cli, "_auto_device", lambda: "cuda")
    monkeypatch.setattr(server, "serve", lambda checkpoint, **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(cli.app, ["serve", "test/checkpoint", *device_args, *flag])
    assert result.exit_code == 0, result.output
    [kwargs] = calls
    assert kwargs["device"] == expected_device
    assert kwargs["warmup"] is expected


@pytest.mark.parametrize(
    ("cache", "batch", "cap", "expected"),
    [
        (True, 32, None, ["plain"] * 3 + ["shared"]),
        (False, 32, None, ["plain"] * 4),
        (True, 2, None, ["plain"] * 3 + ["shared", "plain"]),
        (True, 1, None, ["plain"] * 6),
        (True, 32, 1, ["plain"] * 6),
    ],
)
def test_warmup_respects_real_engine_batching(monkeypatch, cache, batch, cap, expected):
    """Tiny random weights exercise real inference paths, not CUDA kernel compilation."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    core = Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", unk_token="<unk>")
    torso = Qwen3Model(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    model = StrandsDeciderModel(
        StrandsDeciderConfig(
            base_model="tiny-random",
            use_lora=False,
            head_type="pointer",
            pointer_dim=8,
            num_slots=2,
            max_length=128,
            torch_dtype="float32",
        ),
        torso,
        tok,
    )
    engine = SystemOneEngine(
        model,
        EngineConfig(
            device="cpu",
            use_prefix_cache=cache,
            max_batch=batch,
            max_batch_tokens=cap,
            strict_window=True,
        ),
    )
    paths = []

    def track(method, name):
        def forward(*args, **kwargs):
            paths.append(name)
            return method(*args, **kwargs)

        return forward

    monkeypatch.setattr(engine, "_slot_probs_batched", track(engine._slot_probs_batched, "plain"))
    monkeypatch.setattr(
        engine, "_slot_probs_shared_prefix", track(engine._slot_probs_shared_prefix, "shared")
    )
    server._warmup_engine(engine)
    assert paths == expected
    assert (engine.cfg.use_prefix_cache, engine.cfg.max_batch, engine.cfg.max_batch_tokens) == (
        cache,
        batch,
        cap,
    )
