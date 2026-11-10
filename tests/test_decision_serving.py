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
