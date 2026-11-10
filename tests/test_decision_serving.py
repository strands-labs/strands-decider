"""`strands-decider serve` and `ask` on decision checkpoints (src/strands_decider/decision.py),
with tiny random-weight checkpoints made in the test (tests/decision_support.py): which
checkpoints take the decision path, the HTTP API with text and images, and the refusals. v1
checkpoints keep v1's path; tests/test_server.py and tests/test_cli_exit.py cover it."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from decision_support import (  # isort: skip
    GEMMA4,
    decision_checkpoint,
    image_b64,
    intermediate_checkpoint,
    request,
)

from strands_decider import cli, server
from strands_decider.decision import (
    DECISION,
    INTERMEDIATE,
    DecisionEngine,
    checkpoint_kind,
    load_decision_engine,
)
from strands_decider.schema import SystemOneResponse


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return decision_checkpoint(tmp_path_factory.mktemp("decision") / "ckpt", GEMMA4)


def test_checkpoint_kind(ckpt, tmp_path):
    assert checkpoint_kind(str(ckpt)) == DECISION
    assert checkpoint_kind(str(intermediate_checkpoint(tmp_path / "mid"))) == INTERMEDIATE
    assert checkpoint_kind(str(tmp_path)) is None  # no decision_config.json: v1
    other = tmp_path / "other"
    other.mkdir()
    (other / "decision_config.json").write_text(json.dumps({"format_id": "pplx-decider"}))
    assert checkpoint_kind(str(other)) is None  # another format's file of the same name


def test_the_engine_answers_v1_requests(ckpt):
    engine = load_decision_engine(str(ckpt), device="cpu", model_name="tiny")
    assert isinstance(engine, DecisionEngine) and engine.takes_images
    out = engine.evaluate(request([image_b64(64, 48)]))
    assert isinstance(out, SystemOneResponse) and out.model == "tiny"
    assert set(out.answers) == {"late", "team", "mood"} and out.usage.output_tokens == 3
    assert out.answers["team"].type == "choice" and out.answers["mood"].type == "score"
    # The state once, then each question.
    assert out.usage.input_tokens > 266 + 3 * 20


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
        assert health["prompt"] == "strands-chat" and health["model"] == "tiny"
        r = client.post("/v1/systemone", json=request([image_b64(50, 90, "JPEG")]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "tiny" and "latency_ms" in body
        assert set(body["answers"]["team"]["probabilities"]) == {"billing", "shipping", "sales"}
        # A request the core refuses is the caller's error.
        bad = client.post("/v1/systemone", json=request(["not base64!"]))
        assert bad.status_code == 422 and "base64" in bad.json()["detail"]
        assert client.post("/invocations", json=request()).status_code == 200


def test_no_prefix_cache_runs_full_passes(ckpt):
    app = server.create_app(str(ckpt), device="cpu", use_prefix_cache=False, warmup=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["reuse_state"] is False
        assert client.post("/v1/systemone", json=request()).status_code == 200


def test_a_text_only_checkpoint_refuses_images_with_422(tmp_path):
    path = decision_checkpoint(tmp_path / "t", GEMMA4, decision={"inputs": ["text"]})
    with TestClient(server.create_app(str(path), device="cpu", warmup=False)) as client:
        r = client.post("/v1/systemone", json=request([image_b64(64, 48)]))
        assert r.status_code == 422 and "does not take images" in r.json()["detail"]
        assert client.get("/health").json()["images"] is False


def test_serve_refuses_an_intermediate_checkpoint_and_a_device_map(ckpt, tmp_path):
    with pytest.raises(ValueError, match="intermediate checkpoint"):
        server.create_app(str(intermediate_checkpoint(tmp_path / "mid")), device="cpu")
    with pytest.raises(ValueError, match="device-map"):
        server.create_app(str(ckpt), device="cuda", device_map="auto")
    with pytest.raises(ValueError, match="cpu, cuda or mps"):
        load_decision_engine(str(ckpt), device="mlx")


def test_ask_on_a_decision_checkpoint(ckpt, tmp_path, monkeypatch):
    import base64

    from typer.testing import CliRunner

    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(image_b64(64, 48)))
    monkeypatch.setattr(cli, "_exit_fast", False)  # ask sets it; main() is not run here
    args = ["ask", str(ckpt), "--state", "a b c", "--noul", "a b?", "--choice",
            "which?=billing,sales", "--image", str(img), "--device", "cpu", "--json"]  # fmt: skip
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert set(out["answers"]) == {"noul_0", "choice_0"}
    assert set(out["answers"]["choice_0"]["probabilities"]) == {"billing", "sales"}
    assert cli._exit_fast


def test_ask_refuses_an_intermediate_checkpoint(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setattr(cli, "_exit_fast", False)
    path = intermediate_checkpoint(tmp_path / "mid")
    result = CliRunner().invoke(
        cli.app, ["ask", str(path), "--state", "x", "--noul", "y?", "--device", "cpu"]
    )
    assert result.exit_code == 2 and "intermediate" in result.output  # a usage error


def test_serve_refuses_audio_rather_than_dropping_it(ckpt):
    """v1's request model has no `audio` field; the decision engine reads the JSON the client
    sent, so the core's refusal holds on the wire too."""
    with TestClient(server.create_app(str(ckpt), device="cpu", warmup=False)) as client:
        r = client.post("/v1/systemone", json={**request(), "audio": ["UklGRiQAAABXQVZF"]})
        assert r.status_code == 422 and r.json()["detail"] == "audio input is not supported"
        assert client.post("/v1/systemone", json={**request(), "audio": []}).status_code == 200


def test_serve_reports_a_runtime_failure_as_500(ckpt, monkeypatch):
    app = server.create_app(str(ckpt), device="cpu", warmup=False)
    with TestClient(app, raise_server_exceptions=False) as client:

        def fails(*_a, **_k):
            raise ValueError("an internal check")

        monkeypatch.setattr(app.state.engine.backend, "forward", fails)
        assert client.post("/v1/systemone", json=request()).status_code == 500


def test_serve_answers_concurrent_requests_alike(ckpt):
    from concurrent.futures import ThreadPoolExecutor

    body = request([image_b64(64, 48)])
    with TestClient(server.create_app(str(ckpt), device="cpu", warmup=False)) as client:
        assert client.get("/health").json()["max_concurrency"] == 1
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = list(pool.map(lambda _: client.post("/v1/systemone", json=body), range(6)))
    assert all(r.status_code == 200 for r in got)
    assert len({json.dumps(r.json()["answers"], sort_keys=True) for r in got}) == 1


def test_flags_that_do_not_apply_are_named(ckpt, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    with pytest.warns(UserWarning, match="--vision is for v1"):
        server.create_app(str(ckpt), device="cpu", vision=True, warmup=False)
    with pytest.raises(ValueError, match="cpu, cuda or mps"):
        server.create_app(str(ckpt), device="mlx")
    with pytest.raises(ValueError, match="--dtype is for decision checkpoints"):
        server.create_app(str(tmp_path), device="cpu", dtype="bfloat16")
    monkeypatch.setattr(cli, "_exit_fast", False)
    for args, hint in (
        (["--device", "mlx"], "--device"),
        (["--device", "cpu", "--dtype", "float16"], "--dtype"),
    ):
        out = CliRunner().invoke(cli.app, ["ask", str(ckpt), "--state", "x", "--noul", "y?", *args])
        assert out.exit_code == 2 and hint in out.output, out.output
    out = CliRunner().invoke(
        cli.app,
        [
            "ask",
            str(ckpt),
            "--state",
            "x",
            "--noul",
            "y?",
            "--device",
            "cpu",
            "--dtype",
            "bfloat16",
            "--json",
        ],
    )
    assert out.exit_code == 0, out.output


def test_serve_takes_a_dtype(ckpt):
    app = server.create_app(str(ckpt), device="cpu", dtype="bfloat16", warmup=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["dtype"] == "bfloat16"


# ---- finding a decision checkpoint on the Hub ----------------------------------------------


def _hub_error(cls, message="x"):
    """A huggingface_hub error without an HTTP response behind it."""
    e = cls.__new__(cls)
    Exception.__init__(e, message)
    return e


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """huggingface_hub's cache lookup and downloads, recorded; none reaches the network."""
    import huggingface_hub

    calls = []
    state = {"cache": None, "download": None, "snapshot": None}

    def outcome(name, value):
        calls.append(name)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, fn, revision=None, **_: outcome(("cache", repo, fn, revision), state["cache"]),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, fn, revision=None, **_: outcome(
            ("download", repo, fn, revision), state["download"]
        ),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, revision=None, local_files_only=False, **_: outcome(
            ("snapshot", repo, revision, local_files_only), state["snapshot"]
        ),
    )
    dc = tmp_path / "snap" / "decision_config.json"
    dc.parent.mkdir()
    dc.write_text(json.dumps({"interface": "strands-decider"}))
    return state, calls, str(dc)


def test_hub_detection_reads_the_cache_first(hub):
    state, calls, dc = hub
    state["cache"] = dc
    assert checkpoint_kind("org/repo", revision="abc") == DECISION
    assert calls == [("cache", "org/repo", "decision_config.json", "abc")]  # no Hub request
    calls.clear()
    state["cache"] = object()  # the cache's record that the repo has no such file
    assert checkpoint_kind("org/repo") is None and len(calls) == 1


def test_hub_detection_asks_the_hub_once_and_lets_real_errors_through(hub):
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
    )

    state, calls, dc = hub
    state["download"] = dc
    assert checkpoint_kind("org/repo", revision="abc") == DECISION
    assert calls[-1] == ("download", "org/repo", "decision_config.json", "abc")
    state["download"] = _hub_error(RemoteEntryNotFoundError)  # a v1 repo: no such file
    assert checkpoint_kind("org/repo") is None
    state["download"] = _hub_error(RepositoryNotFoundError)  # v1's loader reports it
    assert checkpoint_kind("org/repo") is None
    state["download"] = _hub_error(HfHubHTTPError, "401 Unauthorized")
    with pytest.raises(HfHubHTTPError, match="401"):
        checkpoint_kind("org/repo")
    # Offline, with the file not cached on its own: the cached snapshot decides.
    state["download"] = _hub_error(LocalEntryNotFoundError)
    state["snapshot"] = str(__import__("pathlib").Path(dc).parent)
    calls.clear()
    assert checkpoint_kind("org/repo", revision="abc") == DECISION
    assert calls[-1] == ("snapshot", "org/repo", "abc", True)
    state["snapshot"] = _hub_error(LocalEntryNotFoundError)
    assert checkpoint_kind("org/repo") is None
