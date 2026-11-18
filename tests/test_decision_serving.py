"""`strands-decider serve` and `ask` on decision checkpoints (src/strands_decider/decision.py),
with tiny random-weight checkpoints made in the test (tests/decision_support.py): which
checkpoints take the decision path, the HTTP API with text and images, and the refusals. v1
checkpoints keep v1's path; tests/test_server.py and tests/test_cli_exit.py cover it."""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_support import GEMMA4, decision_checkpoint, image_b64, request  # isort: skip

from strands_decider import cli, decision, server
from strands_decider.core import Request, evaluate, render
from strands_decider.decision import (
    DECISION,
    INTERMEDIATE,
    DecisionEngine,
    checkpoint_dir,
    checkpoint_kind,
    load_decision_engine,
)
from strands_decider.schema import SystemOneResponse


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return decision_checkpoint(tmp_path_factory.mktemp("decision") / "ckpt", GEMMA4)


@pytest.fixture
def intermediate(tmp_path):
    """A directory with a decision_config.json from before the `interface` field."""
    path = tmp_path / "mid"
    path.mkdir()
    (path / "decision_config.json").write_text(json.dumps({"format_version": 1, "token_ids": []}))
    return path


@pytest.fixture(autouse=True)
def _no_exit_fast(monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)  # ask sets it; main() is not run here


def _ask(*args):
    return CliRunner().invoke(cli.app, ["ask", *args])


def _client(path, **kwargs) -> TestClient:
    return TestClient(server.create_app(str(path), device="cpu", **{"warmup": False, **kwargs}))


# ---- which checkpoints take the decision path ----------------------------------------------


def test_checkpoint_kind(ckpt, tmp_path, intermediate):
    assert checkpoint_kind(str(ckpt)) == DECISION
    assert checkpoint_kind(str(intermediate)) == INTERMEDIATE
    assert checkpoint_kind(str(tmp_path / "x")) is None  # no such path: v1 reports it
    other = tmp_path / "other"
    other.mkdir()
    assert checkpoint_kind(str(other)) is None  # no decision_config.json: v1
    (other / "decision_config.json").write_text(json.dumps({"format_id": "another-format"}))
    assert checkpoint_kind(str(other)) is None  # another decision model's file


def test_a_hub_repo_is_its_snapshot(ckpt, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    calls = []

    def snapshot(repo, revision=None, **_):
        calls.append((repo, revision))
        if repo == "org/missing":
            e = RepositoryNotFoundError.__new__(RepositoryNotFoundError)  # no HTTP response
            Exception.__init__(e, "no such repo")
            raise e
        return str(ckpt)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert checkpoint_kind("org/decision", revision="v2") == DECISION
    assert checkpoint_dir("org/missing") is None  # v1's loader reports it
    assert calls == [("org/decision", "v2"), ("org/missing", None)]


def test_a_mistyped_path_keeps_v1s_message(tmp_path):
    out = _ask(str(tmp_path / "no" / "such"), "--state", "x", "--noul", "y?", "--device", "cpu")
    assert out.exit_code != 0 and "not a local directory" in str(out.exception)


# ---- the engine ----------------------------------------------------------------------------


def test_the_engine_answers_requests(ckpt):
    engine = load_decision_engine(str(ckpt), device="cpu", model_name="tiny")
    assert isinstance(engine, DecisionEngine) and engine.image_problem is None
    out = engine.evaluate(Request.model_validate(request([image_b64(64, 48)])))
    assert isinstance(out, SystemOneResponse) and out.model == "tiny"
    assert set(out.answers) == {"late", "team", "mood"} and out.usage.output_tokens == 3
    assert out.answers["team"].type == "choice" and out.answers["mood"].type == "score"
    assert out.usage.input_tokens > 266 + 3 * 20  # the state once, then each question


def test_the_engine_answers_as_the_core_does(ckpt):
    engine = load_decision_engine(str(ckpt), device="cpu", model_name="t")
    images = [image_b64(64, 48), image_b64(50, 90, "JPEG")]
    for req in map(Request.model_validate, (request(), request(images), request(state=""))):
        want = evaluate(
            req, engine.tokenizer, engine.weight, engine.config, engine.backend, model_name="t"
        )
        assert engine.evaluate(req) == SystemOneResponse.model_validate(want)


def test_options_are_checked():
    decision.check_options("cuda:1", "bfloat16")
    with pytest.raises(ValueError, match="decision checkpoints run on cpu, cuda, mps, not 'tpu'"):
        decision.check_options("tpu", None)
    with pytest.raises(ValueError, match="--dtype is one of float32, bfloat16"):
        decision.check_options("cpu", "float16")


# ---- serve ---------------------------------------------------------------------------------


def test_serve_routes_a_decision_checkpoint(ckpt):
    app = server.create_app(str(ckpt), device="cpu", model_name="tiny")  # warm-up included
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["checkpoint"] == str(ckpt)
        assert (health["backend"], health["device"], health["dtype"]) == (
            "transformers",
            "cpu",
            "float32",
        )
        assert health["images"] is True and health["reuse_state"] is True
        assert health["model"] == "tiny"
        r = client.post("/v1/systemone", json=request([image_b64(50, 90, "JPEG")]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "tiny" and "latency_ms" in body
        assert set(body["answers"]["team"]["probabilities"]) == {"billing", "shipping", "sales"}
        bad = client.post("/v1/systemone", json=request(["not base64!"]))
        assert bad.status_code == 422 and "base64" in bad.json()["detail"]
        assert client.post("/invocations", json=request()).status_code == 200


def test_no_prefix_cache_runs_full_passes(ckpt):
    with _client(ckpt, use_prefix_cache=False) as client:
        assert client.get("/health").json()["reuse_state"] is False
        assert client.post("/v1/systemone", json=request()).status_code == 200


def test_serve_takes_a_dtype(ckpt):
    with _client(ckpt, dtype="bfloat16") as client:
        assert client.get("/health").json()["dtype"] == "bfloat16"


def test_serve_refusals_over_http(ckpt, tmp_path):
    """Any audio but null and images on a text-only checkpoint are refused with 422. The number
    of images isn't checked: five are answered, and the prompt's length bounds how many fit."""
    img = image_b64(64, 48)
    with _client(ckpt) as client:
        for audio in (["UklGRiQAAABXQVZF"], []):
            r = client.post("/v1/systemone", json={**request(), "audio": audio})
            assert r.status_code == 422 and "audio input is not supported" in r.json()["detail"]
        assert client.post("/v1/systemone", json={**request(), "audio": None}).status_code == 200
        assert client.post("/v1/systemone", json=request([img] * 5)).status_code == 200
        r = client.post("/v1/systemone", json=[request()])
        assert r.status_code == 422  # not a JSON object
    engine = load_decision_engine(str(ckpt), device="cpu")
    rendered = render(Request.model_validate(request([img] * 5)), engine.tokenizer)
    limit = max(len(q.ids) for q in rendered.questions.values()) - 1
    with _client(ckpt, max_length=limit) as client:
        r = client.post("/v1/systemone", json=request([img] * 5))
        assert r.status_code == 422 and f"exceeds the limit of {limit} tokens" in r.json()["detail"]
        assert client.post("/v1/systemone", json=request([img] * 4)).status_code == 200
    text_only = decision_checkpoint(tmp_path / "t", GEMMA4, decision={"inputs": ["text"]})
    with _client(text_only) as client:
        r = client.post("/v1/systemone", json=request([img]))
        assert r.status_code == 422
        assert r.json()["detail"].startswith("this checkpoint does not take images")
        assert client.get("/health").json()["images"] is False


def test_a_request_needs_no_state(ckpt):
    """The core's request model, not v1's: a request without a state is one with an empty one."""
    body = request()
    with _client(ckpt) as client:
        empty = client.post("/v1/systemone", json={**body, "state": ""})
        del body["state"]
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["answers"] == empty.json()["answers"]


def test_a_422_never_names_a_local_path(tmp_path, monkeypatch):
    from strands_decider.backends import transformers as tb

    path = decision_checkpoint(tmp_path / "leak", GEMMA4)

    def unreadable(p, unified):
        raise OSError(f"{p}/processor_config.json: permission denied")

    monkeypatch.setattr(tb, "load_image_processor", unreadable)
    with pytest.warns(UserWarning, match="permission denied"), _client(path) as client:
        r = client.post("/v1/systemone", json=request([image_b64(64, 48)]))
    assert r.status_code == 422 and r.json()["detail"] == tb.NO_IMAGE_PROCESSOR
    assert str(tmp_path) not in r.text


def test_serve_reports_a_runtime_failure_as_500(ckpt, monkeypatch):
    app = server.create_app(str(ckpt), device="cpu", warmup=False)
    with TestClient(app, raise_server_exceptions=False) as client:

        def fails(*_a, **_k):
            raise ValueError("an internal check")

        monkeypatch.setattr(app.state.engine.backend, "forward", fails)
        assert client.post("/v1/systemone", json=request()).status_code == 500


def test_serve_answers_concurrent_requests_alike(ckpt):
    body = request([image_b64(64, 48)])
    with _client(ckpt) as client:
        assert client.get("/health").json()["max_concurrency"] == 1
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = list(pool.map(lambda _: client.post("/v1/systemone", json=body), range(6)))
    assert all(r.status_code == 200 for r in got)
    assert len({json.dumps(r.json()["answers"], sort_keys=True) for r in got}) == 1


class _SlowBackend:
    """Hidden states of zeros, after a pause: every answer is uniform."""

    def __init__(self, path, **settings):
        self.settings = settings

    def describe(self):
        return {"backend": "slow", "device": self.settings["device"], "dtype": "float32"}

    def forward(self, state, questions, images=()):
        time.sleep(0.4)
        return np.zeros((len(questions), 64), dtype=np.float32)


def _slow_loader(path, **settings):
    return _SlowBackend(path, **settings)


def test_a_full_engine_answers_503_and_refusals_do_not_wait(ckpt, monkeypatch):
    monkeypatch.setitem(decision.BACKENDS, "slow", "test_decision_serving:_slow_loader")
    app = server.create_app(str(ckpt), device="slow", warmup=False, queue_timeout=0.1)
    with TestClient(app) as client:
        assert client.get("/health").json()["backend"] == "slow"
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/v1/systemone", json=request())
            time.sleep(0.1)  # the first holds the only slot
            second = pool.submit(client.post, "/v1/systemone", json=request())
            started = time.perf_counter()
            refused = client.post("/v1/systemone", json={**request(), "audio": ["x"]})
            assert refused.status_code == 422 and time.perf_counter() - started < 0.3
            assert second.result().status_code == 503 and "busy" in second.result().json()["detail"]
            assert first.result().status_code == 200
            team = first.result().json()["answers"]["team"]["probabilities"]
            assert team == {"billing": 0.3333, "shipping": 0.3333, "sales": 0.3333}


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="needs Apple's MPS")
def test_serve_on_mps_answers_concurrent_requests_alike(ckpt):
    body = request([image_b64(64, 48)])
    app = server.create_app(str(ckpt), device="mps", warmup=False, max_concurrency=3)
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = list(pool.map(lambda _: client.post("/v1/systemone", json=body), range(6)))
    assert all(r.status_code == 200 for r in got)
    assert len({json.dumps(r.json()["answers"], sort_keys=True) for r in got}) == 1


def test_serve_refuses_an_intermediate_checkpoint_and_a_device_map(ckpt, intermediate):
    with pytest.raises(ValueError, match="predates the decision interface"):
        server.create_app(str(intermediate), device="cpu")
    with pytest.raises(ValueError, match="device-map"):
        server.create_app(str(ckpt), device="cuda", device_map="auto")


def test_flags_that_do_not_apply_are_named(ckpt, tmp_path):
    with pytest.warns(UserWarning, match="--vision is for v1"):
        server.create_app(str(ckpt), device="cpu", vision=True, warmup=False)
    with pytest.raises(ValueError, match="decision checkpoints run on cpu, cuda, mps"):
        server.create_app(str(ckpt), device="mlx")
    with pytest.raises(ValueError, match="--dtype is for decision checkpoints"):
        server.create_app(str(tmp_path), device="cpu", dtype="bfloat16")


def test_serve_command_options(ckpt, tmp_path, intermediate, monkeypatch):
    """`serve` takes --dtype, --max-concurrency and --queue-timeout for decision checkpoints,
    refuses --dtype and --revision for v1 ones, and refuses an intermediate checkpoint, each
    before anything is announced."""
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.append((app, kw)))
    args = ["serve", str(ckpt), "--device", "cpu", "--no-warmup", "--dtype", "bfloat16",
            "--max-concurrency", "2", "--queue-timeout", "5"]  # fmt: skip
    out = CliRunner().invoke(cli.app, args)
    assert out.exit_code == 0, out.output
    [(app, kw)] = started
    assert kw == {"host": "127.0.0.1", "port": 8000, "workers": 1}
    with TestClient(app) as client:
        health = client.get("/health").json()
    assert (health["dtype"], health["max_concurrency"], health["queue_timeout"]) == (
        "bfloat16",
        2,
        5.0,
    )
    for bad, hint in (
        ([str(tmp_path), "--device", "cpu", "--dtype", "bfloat16"], "--dtype is for decision"),
        ([str(tmp_path), "--device", "cpu", "--revision", "main"], "--revision is for decision"),
        ([str(intermediate), "--device", "cpu"], "predates the decision interface"),
        ([str(ckpt), "--device", "cpu", "--dtype", "float16"], "--dtype is one of"),
        ([str(ckpt), "--device", "cpu", "--max-length", "0"], "greater than or equal to 1"),
    ):
        out = CliRunner().invoke(cli.app, ["serve", *bad])
        assert out.exit_code == 2 and "serving" not in out.output, out.output
        assert hint in " ".join(out.output.split()).replace("│ ", ""), out.output
    assert len(started) == 1


def test_serve_takes_a_max_length(ckpt, monkeypatch, capsys):
    """`--max-length` is the longest prompt served, in place of max_length_validated (4096
    here): a prompt of that many tokens is answered and a longer one refused with 422, never
    truncated; a value past the validated length warns; and a value below the warm-up prompts'
    length skips them rather than fail the start."""
    import uvicorn

    engine = load_decision_engine(str(ckpt), device="cpu")
    rendered = render(Request.model_validate(request()), engine.tokenizer)
    longest = max(len(q.ids) for q in rendered.questions.values())
    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.append(app))
    for limit, status in ((longest, 200), (longest - 1, 422)):
        args = ["serve", str(ckpt), "--device", "cpu", "--no-warmup", "--max-length", str(limit)]
        out = CliRunner().invoke(cli.app, args)
        assert out.exit_code == 0, out.output
        with TestClient(started[-1]) as client:
            assert client.get("/health").json()["max_length"] == limit
            got = client.post("/v1/systemone", json=request())
        assert got.status_code == status, got.text
        if status == 422:
            assert f"exceeds the limit of {limit} tokens" in got.json()["detail"]
    with pytest.warns(UserWarning, match="past the 4096 the checkpoint was validated at"):
        with _client(ckpt, max_length=8192):
            pass
    with TestClient(server.create_app(str(ckpt), device="cpu", max_length=8)) as client:
        assert client.get("/health").json()["max_length"] == 8
    assert "a warm-up request was skipped" in capsys.readouterr().err


# ---- ask -----------------------------------------------------------------------------------


def test_ask_on_a_decision_checkpoint(ckpt, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(image_b64(64, 48)))
    out = _ask(str(ckpt), "--state", "a b c", "--noul", "a b?", "--choice", "which?=billing,sales",
               "--image", str(img), "--device", "cpu", "--json")  # fmt: skip
    assert out.exit_code == 0, out.output
    answers = json.loads(out.stdout)["answers"]
    assert set(answers) == {"noul_0", "choice_0"}
    assert set(answers["choice_0"]["probabilities"]) == {"billing", "sales"}
    assert cli._exit_fast


def test_ask_refusals_are_usage_errors(ckpt, tmp_path, intermediate):
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(image_b64(64, 48)))
    text_only = decision_checkpoint(tmp_path / "t", GEMMA4, decision={"inputs": ["text"]})
    for args, hint in (
        ([str(intermediate), "--device", "cpu"], "predates the decision interface"),
        ([str(text_only), "--image", str(img), "--device", "cpu"], "does not take images"),
        ([str(ckpt), "--device", "mlx"], "decision checkpoints run on"),
        ([str(ckpt), "--device", "cpu", "--dtype", "float16"], "--dtype is one of"),
    ):
        out = _ask(*args[:1], "--state", "x", "--noul", "y?", *args[1:])
        assert out.exit_code == 2 and hint in " ".join(out.output.split()), out.output
    out = _ask(str(ckpt), "--state", "x", "--noul", "y?", "--device", "cpu", "--dtype", "bfloat16")
    assert out.exit_code == 0, out.output
