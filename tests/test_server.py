"""API-surface tests that stub the engine, so they need neither GPU nor weights.

These pin the wire contract. The point of matching the Jev schema is that an
existing client can be repointed at this server by changing a base URL; a silent
drift in field names or types would break that without failing any model test.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from strands_decider import server
from strands_decider.prompting import render_question, to_answer
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
            answers[name] = to_answer(rq, probs)
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=42, output_tokens=len(request.questions)),
        )


@pytest.fixture
def client():
    """Mount routes against a stub engine on `app.state`, skipping model loading.

    SystemOneRequest, Depends, and Request must be imported at module scope: this
    file uses `from __future__ import annotations`, so FastAPI resolves handler
    annotations as strings against module globals. A function-local import would
    leave them unresolvable, and FastAPI would silently treat the body as a query
    parameter -- surfacing as a 422 on every valid request.
    """
    app = FastAPI(title="test")
    app.state.engine = _StubEngine()

    def get_engine(request: Request):
        return request.app.state.engine

    @app.post("/v1/systemone")
    def systemone(request: SystemOneRequest, eng=Depends(get_engine)):
        return eng.evaluate(request).model_dump()

    @app.get("/health")
    def health(eng=Depends(get_engine)):
        return {"status": "ok", "model": eng.cfg.model_name}

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_identifies_checkpoint():
    """The real /health names the loaded checkpoint and window, so a stale server
    still holding the port cannot pass for the one just started."""
    app = server.create_app(
        "checkpoints/strands-decider-test",
        device="cpu",
        warmup=False,
        build_engine=_StubEngine,
    )
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["checkpoint"] == "checkpoints/strands-decider-test"
    assert r.json()["max_length"] == 512


def test_sagemaker_paths_alias_the_api():
    """SageMaker and AgentCore Runtime call GET /ping and POST /invocations; the real app
    answers them as /health and /v1/systemone."""
    app = server.create_app(
        "checkpoints/strands-decider-test",
        device="cpu",
        warmup=False,
        build_engine=_StubEngine,
    )
    with TestClient(app) as client:
        assert client.get("/ping").json() == client.get("/health").json()
        body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Yes?"}}}
        r = client.post("/invocations", json=body)
        assert r.status_code == 200
        assert r.json()["answers"] == client.post("/v1/systemone", json=body).json()["answers"]
        bad = {"state": "x", "questions": {"q": {"type": "nope"}}}
        assert client.post("/invocations", json=bad).status_code == 422


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
    ans = to_answer(rq, [0.0, 0.5, 0.5])
    assert ans.score == pytest.approx(1.5)


def test_noul_answer_carries_no_confidence_field():
    rq = render_question(NoulQuestion(instructions="is it?"))
    ans = to_answer(rq, [0.1, 0.9])
    assert ans.noul == pytest.approx(0.9)
    assert not hasattr(ans, "confidence")


def test_choice_answer_picks_argmax():
    q = ChoiceQuestion(instructions="x", criteria={"a": "", "b": "", "c": ""})
    rq = render_question(q)
    ans = to_answer(rq, [0.2, 0.5, 0.3])
    assert ans.choice == "b"
    assert ans.confidence == pytest.approx((3 * 0.5 - 1) / 2)


def test_real_create_app_routes_work():
    """Exercise the shipped create_app, not a hand-rolled copy of it.

    The earlier tests mount their own routes, which would not catch a mistake in
    server.create_app itself -- for instance an unresolvable request annotation,
    which FastAPI turns into a silent 422 rather than an import error.
    """
    app = server.create_app(
        "checkpoints/does-not-exist", device="cpu", warmup=False, build_engine=_StubEngine
    )
    with TestClient(app) as client:
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


def _racing_engine_cls():
    """Engine stub that records peak concurrent entries into evaluate()."""
    state = {
        "enter": threading.Event(),
        "release": threading.Event(),
        "inside": 0,
        "peak": 0,
        "lock": threading.Lock(),
    }

    class _RacingEngine(_StubEngine):
        def evaluate(self, request):
            with state["lock"]:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
            state["enter"].set()
            # Hold inside the engine long enough that a second request can race in.
            state["release"].wait(timeout=2.0)
            try:
                return super().evaluate(request)
            finally:
                with state["lock"]:
                    state["inside"] -= 1

    return _RacingEngine, state


def _run_racing(device):
    EngineCls, state = _racing_engine_cls()
    # Warm-up would call this blocking engine before the test releases it.
    app = server.create_app(
        "checkpoints/does-not-exist", device=device, warmup=False, build_engine=EngineCls
    )
    payload = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}}

    with TestClient(app) as client:

        def _post():
            return client.post("/v1/systemone", json=payload)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_post)
            # Only release once we have observed the first request inside the engine,
            # so the second request has a chance to race in while the first is held.
            assert state["enter"].wait(timeout=2.0), "first request never entered the engine"
            f2 = pool.submit(_post)
            # Give the second request a moment to reach evaluate() if nothing serialises it.
            threading.Event().wait(0.2)
            state["release"].set()
            r1, r2 = f1.result(timeout=5.0), f2.result(timeout=5.0)

    assert r1.status_code == 200 and r2.status_code == 200
    return state["peak"]


def test_mps_serialises_engine_calls():
    """MPS crashes on concurrent GPU calls (issue #9), so requests must take turns."""
    peak = _run_racing(device="mps")
    assert peak < 2, (
        f"two requests entered the engine concurrently on MPS (peak={peak}); "
        "GPU calls must be serialised to avoid the Metal command-buffer crash"
    )


def test_cpu_allows_concurrent_engine_calls():
    """CPU (and CUDA) don't need serialisation; the lock would just block throughput."""
    peak = _run_racing(device="cpu")
    assert peak == 2, (
        f"CPU requests were serialised (peak={peak}); the lock should be lifted on "
        "devices that tolerate overlapping engine calls"
    )


def test_too_many_options_returns_422():
    """A question wider than the model's slots is caller error, not a 500."""

    class _NarrowEngine(_StubEngine):
        def evaluate(self, request):
            raise ValueError("question has 30 options but this model has 24 slots")

    # No warm-up: this engine rejects every request.
    app = server.create_app(
        "checkpoints/does-not-exist", device="cpu", warmup=False, build_engine=_NarrowEngine
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}},
        )
    assert r.status_code == 422
    assert "24 slots" in r.json()["detail"]


def test_device_map_reaches_the_loader(monkeypatch):
    from strands_decider import server

    seen = {}

    def load(*_args, **kwargs):
        seen.update(kwargs)
        return _StubModel()

    monkeypatch.setattr(server.StrandsDeciderModel, "load", load)
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: _StubEngine())
    app = server.create_app(
        "checkpoints/strands-decider-test", device="cuda", device_map="auto", warmup=False
    )
    with TestClient(app):
        pass
    assert seen["device_map"] == "auto"


@pytest.mark.parametrize("kwargs", [{"device": "cpu"}, {"device": "cuda", "vision": True}])
def test_device_map_needs_the_text_engine_on_cuda(kwargs):
    from strands_decider import server

    with pytest.raises(ValueError, match="device-map"):
        server.create_app("checkpoints/strands-decider-test", device_map="auto", **kwargs)


def test_ping_answers_while_every_worker_thread_is_busy():
    """/ping must not queue behind /invocations: with more requests in flight than the
    thread pool has threads, a health check still answers within SageMaker's 2 s."""
    import json
    import socket
    import threading
    import time
    import urllib.request

    import anyio
    import uvicorn

    from strands_decider import server

    async def _pool_size() -> int:  # Starlette runs sync handlers on anyio's default limiter
        return int(anyio.to_thread.current_default_thread_limiter().total_tokens)

    pool = anyio.run(_pool_size)
    release = threading.Event()
    blocked = threading.Semaphore(0)

    class _BlockingEngine(_StubEngine):
        def evaluate(self, request):
            blocked.release()
            release.wait(timeout=30)
            return super().evaluate(request)

    app = server.create_app(
        "checkpoints/strands-decider-test", device="cpu", warmup=False, build_engine=_BlockingEngine
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))  # handed to uvicorn, so no other process can take the port
    port = sock.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    srv_thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    srv_thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    body = json.dumps({"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}})

    def invoke():
        req = urllib.request.Request(
            f"{url}/invocations", body.encode(), {"content-type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=60).read()

    callers = [threading.Thread(target=invoke) for _ in range(pool + 5)]
    try:
        for t in callers:
            t.start()
        for _ in range(pool):  # every worker thread is now inside a request
            assert blocked.acquire(timeout=10)
        with urllib.request.urlopen(f"{url}/ping", timeout=2) as r:
            assert r.status == 200
    finally:
        release.set()
        for t in callers:
            t.join(timeout=30)
        srv.should_exit = True
        srv_thread.join(timeout=10)
        sock.close()


@contextlib.contextmanager
def _serving(app):
    """Serve `app` with uvicorn on a free local port, yield its URL, and stop it."""
    import socket
    import threading

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))  # handed to uvicorn, so no other process can take the port
    srv = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        srv.should_exit = True
        thread.join(timeout=10)
        sock.close()


def _invoke(url: str) -> int:
    body = json.dumps({"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}})
    req = urllib.request.Request(
        f"{url}/invocations", body.encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def test_max_concurrency_caps_the_requests_on_the_engine():
    import threading

    lock = threading.Lock()
    running = peak = 0

    class _CountingEngine(_StubEngine):
        def evaluate(self, request):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.2)
            with lock:
                running -= 1
            return super().evaluate(request)

    app = server.create_app(
        "checkpoints/strands-decider-test",
        device="cpu",
        warmup=False,
        build_engine=_CountingEngine,
        max_concurrency=2,
    )
    statuses = []
    with _serving(app) as url:
        callers = [
            threading.Thread(target=lambda: statuses.append(_invoke(url))) for _ in range(10)
        ]
        for t in callers:
            t.start()
        for t in callers:
            t.join(timeout=30)
    assert statuses == [200] * 10
    assert peak == 2


def test_requests_queued_by_max_concurrency_leave_ping_responsive():
    import threading

    release = threading.Event()
    started = threading.Semaphore(0)

    class _BlockingEngine(_StubEngine):
        def evaluate(self, request):
            started.release()
            release.wait(timeout=30)
            return super().evaluate(request)

    app = server.create_app(
        "checkpoints/strands-decider-test",
        device="cpu",
        warmup=False,
        build_engine=_BlockingEngine,
        max_concurrency=1,
    )
    with _serving(app) as url:
        callers = [threading.Thread(target=_invoke, args=(url,)) for _ in range(50)]
        try:
            for t in callers:
                t.start()
            assert started.acquire(timeout=10)  # one request runs, 49 wait behind the cap
            time.sleep(0.5)
            with urllib.request.urlopen(f"{url}/ping", timeout=2) as r:
                assert r.status == 200
        finally:
            release.set()
            for t in callers:
                t.join(timeout=30)


def test_max_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="max-concurrency"):
        server.create_app("checkpoints/strands-decider-test", device="cpu", max_concurrency=0)
