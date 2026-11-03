"""`serve --warmup` answers every request shape before the server accepts requests.

CPU only: a stub engine records what the warm-up asks; a tiny random torso runs it end to end.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_engine_concurrency import _tokenizer

from strands_decider import server
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel


class _RecordingEngine:
    def __init__(self, fail: bool = False):
        self.requests: list = []
        self.fail = fail

    def evaluate(self, request):
        if self.fail:
            raise RuntimeError("kernel failed to compile")
        self.requests.append(request)


def _run_lifespan(app) -> None:
    with TestClient(app):
        pass


@pytest.fixture
def engine():
    return _RecordingEngine()


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "cpu", "mps"])
def test_warmup_is_on_by_default_on_every_device(engine, device):
    _run_lifespan(server.create_app("checkpoints/test", device=device, build_engine=lambda: engine))
    assert engine.requests


def test_no_warmup_skips_it(engine):
    _run_lifespan(
        server.create_app(
            "checkpoints/test", device="cuda", warmup=False, build_engine=lambda: engine
        )
    )
    assert not engine.requests


def test_warmup_asks_each_kind_and_every_question_count(engine):
    _run_lifespan(
        server.create_app(
            "checkpoints/test", device="cuda", max_batch=8, build_engine=lambda: engine
        )
    )

    kinds_alone = {
        next(iter(r.questions.values())).type for r in engine.requests if len(r.questions) == 1
    }
    assert kinds_alone == {"noul", "choice", "score"}
    assert any({q.type for q in r.questions.values()} == kinds_alone for r in engine.requests)
    # Each count compiles once on CUDA; larger requests run in chunks of at most max_batch.
    assert {len(r.questions) for r in engine.requests} == set(range(1, 9))


def test_a_failed_warmup_stops_startup():
    failing = _RecordingEngine(fail=True)
    app = server.create_app("checkpoints/test", device="cuda", build_engine=lambda: failing)
    with pytest.raises(RuntimeError, match="compile"):
        _run_lifespan(app)


def test_warmup_runs_on_a_real_engine():
    from transformers import Qwen3Config, Qwen3Model

    tok = _tokenizer()
    torso = Qwen3Model(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    cfg = StrandsDeciderConfig(
        base_model="tiny", pointer_dim=16, use_lora=False, torch_dtype="float32", max_length=512
    )
    engine = SystemOneEngine(StrandsDeciderModel(cfg, torso, tok), EngineConfig(device="cpu"))
    server.warm_up(engine, max_batch=4)


def test_vision_warmup_sends_an_image_request():
    recorder = _RecordingEngine()
    _run_lifespan(
        server.create_app(
            "checkpoints/test",
            device="cuda",
            vision=True,
            max_batch=5,
            build_engine=lambda: recorder,
        )
    )

    with_images = [r for r in recorder.requests if r.images]
    assert len(with_images) == 1
    assert {len(r.questions) for r in recorder.requests if not r.images} == set(range(1, 6))


def test_text_warmup_sends_no_images(engine):
    _run_lifespan(server.create_app("checkpoints/test", device="cuda", build_engine=lambda: engine))
    assert not any(r.images for r in engine.requests)
