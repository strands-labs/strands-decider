"""API-surface tests that stub the engine, so they need neither GPU nor weights.

These pin the wire contract. The point of matching the Jev schema is that an
existing client can be repointed at this server by changing a base URL; a silent
drift in field names or types would break that without failing any model test.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from strands_decider import server
from strands_decider.infer import _to_answer
from strands_decider.prompting import render_question
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)


class _StubCfg:
    model_name = "strands-decider-test"
    device = "cpu"
    use_prefix_cache = True


class _StubModelCfg:
    base_model = "stub"
    num_slots = 24
    temperature = 1.0
    max_length = 512


class _StubModel:
    config = _StubModelCfg()


class _StubEngine:
    """Returns a fixed, concentrated distribution: exercises the response path only."""

    cfg = _StubCfg()
    model = _StubModel()

    def evaluate(self, request):
        answers = {}
        for name, q in request.questions.items():
            rq = render_question(q)
            n = rq.n_slots
            # Concentrated on slot 0, so answers are deterministic and non-uniform.
            probs = [0.7] + [0.3 / (n - 1)] * (n - 1)
            answers[name] = _to_answer(rq, probs)
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=42, output_tokens=len(request.questions)),
        )


@pytest.fixture
def client(monkeypatch):
    """Mount the real routes against a stub engine, skipping model loading.

    SystemOneRequest must be imported at module scope: this file uses
    `from __future__ import annotations`, so FastAPI resolves the handler's
    annotation as a string against module globals. A function-local import would
    leave it unresolvable, and FastAPI would silently treat the body as a query
    parameter -- surfacing as a 422 on every valid request.
    """
    from fastapi import FastAPI

    monkeypatch.setattr(server, "_engine", _StubEngine())
    app = FastAPI(title="test")

    @app.post("/v1/systemone")
    def systemone(request: SystemOneRequest):
        return server.get_engine().evaluate(request).model_dump()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": server.get_engine().cfg.model_name}

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_identifies_checkpoint(monkeypatch):
    """The real /health names the loaded checkpoint and window, so a stale server
    still holding the port cannot pass for the one just started."""
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server.StrandsDeciderModel, "load", lambda *a, **k: _StubModel())
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _StubEngine())
    r = TestClient(server.create_app("checkpoints/strands-decider-test", device="cpu")).get(
        "/health"
    )
    assert r.status_code == 200
    assert r.json()["checkpoint"] == "checkpoints/strands-decider-test"
    assert r.json()["max_length"] == 512


def test_documented_payload_round_trips(client):
    r = client.post(
        "/v1/systemone",
        json={
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "strands-decider-latest",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "department": {
                    "type": "choice",
                    "instructions": "Route it.",
                    "criteria": {
                        "billing": "money",
                        "technical": "bugs",
                        "sales": "pricing",
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated?",
                    "criteria": ["Calm", "Frustrated", "Very angry"],
                },
            },
        },
    )
    assert r.status_code == 200
    body = r.json()

    assert set(body) == {"model", "answers", "usage"}
    assert set(body["answers"]) == {"is_urgent", "department", "frustration"}

    noul = body["answers"]["is_urgent"]
    assert noul["type"] == "noul"
    assert 0.0 <= noul["noul"] <= 1.0
    assert "confidence" not in noul  # by design: p is the uncertainty

    choice = body["answers"]["department"]
    assert choice["type"] == "choice"
    assert choice["choice"] in {"billing", "technical", "sales"}
    assert set(choice["probabilities"]) == {"billing", "technical", "sales"}
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= choice["confidence"] <= 1.0

    score = body["answers"]["frustration"]
    assert score["type"] == "score"
    assert score["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert set(score["probabilities"]) == {"0", "1", "2"}
    assert 0.0 <= score["score"] <= 2.0


def test_malformed_question_is_rejected(client):
    r = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": ""}}},
        },
    )
    assert r.status_code == 422  # choice needs >= 2 options


def test_empty_questions_rejected(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 422


def test_score_answer_is_expected_value_not_argmax():
    """The API returns a continuous score, so a split belief must land between levels."""
    q = ScoreQuestion(instructions="x", criteria=["low", "mid", "high"])
    rq = render_question(q)
    ans = _to_answer(rq, [0.0, 0.5, 0.5])
    assert ans.score == pytest.approx(1.5)


def test_noul_answer_carries_no_confidence_field():
    rq = render_question(NoulQuestion(instructions="is it?"))
    ans = _to_answer(rq, [0.1, 0.9])
    assert ans.noul == pytest.approx(0.9)
    assert not hasattr(ans, "confidence")


def test_choice_answer_picks_argmax():
    q = ChoiceQuestion(instructions="x", criteria={"a": "", "b": "", "c": ""})
    rq = render_question(q)
    ans = _to_answer(rq, [0.2, 0.5, 0.3])
    assert ans.choice == "b"
    assert ans.confidence == pytest.approx((3 * 0.5 - 1) / 2)


def test_real_create_app_routes_work(monkeypatch):
    """Exercise the shipped create_app, not a hand-rolled copy of it.

    The earlier tests mount their own routes, which would not catch a mistake in
    server.create_app itself -- for instance an unresolvable request annotation,
    which FastAPI turns into a silent 422 rather than an import error.
    """
    from strands_decider import infer

    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _StubModel())
    )
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _StubEngine())
    monkeypatch.setattr(infer, "SystemOneEngine", lambda model, cfg=None: _StubEngine())

    app = server.create_app("checkpoints/does-not-exist", device="cpu")
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    r = client.post(
        "/v1/systemone",
        json={
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
            },
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answers"]["is_urgent"]["type"] == "noul"
    # The server adds its own timing field on top of the API response.
    assert "latency_ms" in body


def test_concurrent_requests_do_not_enter_engine_in_parallel(monkeypatch):
    """Tests concurrent requests."""
    enter = threading.Event()
    release = threading.Event()
    inside = 0
    peak = 0
    lock = threading.Lock()

    class _RacingEngine(_StubEngine):
        def evaluate(self, request):
            nonlocal inside, peak
            with lock:
                inside += 1
                peak = max(peak, inside)
            enter.set()
            # Hold inside the engine long enough that a second request can race in.
            release.wait(timeout=2.0)
            try:
                return super().evaluate(request)
            finally:
                with lock:
                    inside -= 1

    monkeypatch.setattr(server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _StubModel()))
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _RacingEngine())

    app = server.create_app("checkpoints/does-not-exist", device="cpu")
    client = TestClient(app)
    payload = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}}

    def _post():
        return client.post("/v1/systemone", json=payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_post)
        # Only release once we have observed the first request inside the engine,
        # so the second request has a chance to race in while the first is held.
        assert enter.wait(timeout=2.0), "first request never entered the engine"
        f2 = pool.submit(_post)
        # Give the second request a moment to reach evaluate() if nothing serialises it.
        threading.Event().wait(0.2)
        release.set()
        r1, r2 = f1.result(timeout=5.0), f2.result(timeout=5.0)

    assert r1.status_code == 200 and r2.status_code == 200
    assert peak < 2, (
        f"two requests entered the engine concurrently (peak={peak}); "
        "GPU calls must be serialised to avoid the MPS crash"
    )


def test_too_many_options_returns_422(monkeypatch):
    """A question wider than the model's slots is caller error, not a 500."""

    class _NarrowEngine(_StubEngine):
        def evaluate(self, request):
            raise ValueError("question has 30 options but this model has 24 slots")

    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _StubModel())
    )
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _NarrowEngine())

    app = server.create_app("checkpoints/does-not-exist", device="cpu")
    r = TestClient(app).post(
        "/v1/systemone",
        json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}},
    )
    assert r.status_code == 422
    assert "24 slots" in r.json()["detail"]


def test_api_key_authentication(monkeypatch):
    from fastapi.testclient import TestClient

    from strands_decider import infer, server

    monkeypatch.setenv("STRANDS_DECIDER_API_KEY", "test-secret-key")

    monkeypatch.setattr(
        server.StrandsDeciderModel,
        "load",
        classmethod(lambda cls, *a, **k: _StubModel()),
    )
    monkeypatch.setattr(
        server,
        "SystemOneEngine",
        lambda model, cfg: _StubEngine(),
    )
    monkeypatch.setattr(
        infer,
        "SystemOneEngine",
        lambda model, cfg=None: _StubEngine(),
    )

    app = server.create_app("checkpoints/does-not-exist", device="cpu")
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    response = client.post(
        "/v1/systemone",
        json={"state": "test", "questions": {"q1": {"type": "noul", "instructions": "test"}}},
    )
    assert response.status_code == 401

    response = client.post(
        "/v1/systemone",
        json={"state": "test", "questions": {"q1": {"type": "noul", "instructions": "test"}}},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401

    response = client.post(
        "/v1/systemone",
        json={"state": "test", "questions": {"q1": {"type": "noul", "instructions": "test"}}},
        headers={"Authorization": "Bearer test-secret-key"},
    )
    assert response.status_code != 401
