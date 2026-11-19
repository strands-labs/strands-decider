"""`strands-decider serve` and `ask` on decision checkpoints, with tiny random-weight checkpoints
(tests/decision_support.py): which checkpoints take the decision path, the HTTP API with text
and images, and its refusals. v1 checkpoints keep v1's path; tests/test_server.py and
tests/test_cli_exit.py cover it."""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_support import GEMMA4, decision_checkpoint, image_b64, request  # isort: skip

from strands_decider import cli, decision, server
from strands_decider.backends import transformers as tb
from strands_decider.core import Request, evaluate, load_tokenizer, render
from strands_decider.decision import is_decision_checkpoint, load_decision_engine
from strands_decider.schema import SystemOneResponse


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return decision_checkpoint(tmp_path_factory.mktemp("decision") / "ckpt", GEMMA4)


@pytest.fixture(scope="module")
def text_only(tmp_path_factory):
    path = tmp_path_factory.mktemp("text") / "ckpt"
    return decision_checkpoint(path, GEMMA4, decision={"inputs": ["text"]})


@pytest.fixture(autouse=True)
def _no_exit_fast(monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)  # ask sets it; main() is not run here


def _ask(*args):
    return CliRunner().invoke(cli.app, ["ask", *args])


def _client(path, **kwargs) -> TestClient:
    return TestClient(server.create_app(str(path), **{"device": "cpu", "warmup": False, **kwargs}))


# ---- which checkpoints take the decision path ----------------------------------------------


def test_which_checkpoints_take_the_decision_path(ckpt, tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    assert is_decision_checkpoint(str(ckpt))
    assert not is_decision_checkpoint(str(tmp_path / "x"))  # no such path: v1 reports it
    other = tmp_path / "other"
    other.mkdir()
    assert not is_decision_checkpoint(str(other))  # no decision_config.json
    (other / "decision_config.json").write_text(json.dumps({"format_id": "another-format"}))
    assert not is_decision_checkpoint(str(other))  # another decision model's

    calls = []

    def snapshot(repo, revision=None, **_):
        calls.append((repo, revision))
        if repo == "org/missing":
            e = RepositoryNotFoundError.__new__(RepositoryNotFoundError)  # no HTTP response
            Exception.__init__(e, "no such repo")
            raise e
        return str(ckpt)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert is_decision_checkpoint("org/decision", revision="v2")  # a Hub repo is its snapshot
    assert not is_decision_checkpoint("org/missing")
    assert calls == [("org/decision", "v2"), ("org/missing", None)]


# ---- the engine ----------------------------------------------------------------------------


def test_the_engine_answers_as_the_core_does(ckpt):
    engine = load_decision_engine(str(ckpt), device="cpu", model_name="t")
    images = [image_b64(64, 48), image_b64(50, 90, "JPEG")]
    for req in map(Request.model_validate, (request(), request(images), request(state=""))):
        want = evaluate(
            req, engine.tokenizer, engine.weight, engine.config, engine.backend, model_name="t"
        )
        assert engine.evaluate(req) == SystemOneResponse.model_validate(want)


# ---- serve ---------------------------------------------------------------------------------


def test_serve_answers_a_decision_checkpoint(ckpt):
    app = server.create_app(str(ckpt), device="cpu", model_name="tiny")  # warm-up included
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["checkpoint"] == str(ckpt)
        assert health["model"] == "tiny" and health["images"] is True
        assert (health["backend"], health["device"], health["dtype"]) == (
            "transformers",
            "cpu",
            "float32",
        )
        assert (health["reuse_state"], health["max_concurrency"]) == (True, 1)
        r = client.post("/v1/systemone", json=request([image_b64(50, 90, "JPEG")]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "tiny" and "latency_ms" in body
        assert set(body["answers"]["team"]["probabilities"]) == {"billing", "shipping", "sales"}
        assert body["usage"]["output_tokens"] == 3
        assert client.post("/invocations", json=request()).status_code == 200


def test_serve_refuses_what_the_interface_refuses(ckpt, text_only):
    """Any audio but null, a body that isn't an object, and images on a text-only checkpoint get
    422. A request may omit its state (the core's request model, not v1's). The number of images
    isn't checked: the prompt's length bounds it."""
    img = image_b64(64, 48)
    with _client(ckpt) as client:
        for audio in (["UklGRiQAAABXQVZF"], []):
            r = client.post("/v1/systemone", json={**request(), "audio": audio})
            assert r.status_code == 422 and "audio input is not supported" in r.json()["detail"]
        assert client.post("/v1/systemone", json={**request(), "audio": None}).status_code == 200
        assert client.post("/v1/systemone", json=[request()]).status_code == 422
        bad = client.post("/v1/systemone", json=request(["not base64!"]))
        assert bad.status_code == 422 and "base64" in bad.json()["detail"]
        body = request()
        empty = client.post("/v1/systemone", json={**body, "state": ""}).json()
        del body["state"]
        assert client.post("/v1/systemone", json=body).json()["answers"] == empty["answers"]
        assert client.post("/v1/systemone", json=request([img] * 5)).status_code == 200
    with _client(text_only) as client:
        r = client.post("/v1/systemone", json=request([img]))
        assert r.status_code == 422
        assert r.json()["detail"].startswith("this checkpoint does not take images")
        assert client.get("/health").json()["images"] is False


@pytest.mark.parametrize("cause", ["refused-settings", "unreadable"])
def test_a_processor_that_fails_refuses_images_only_and_names_no_path(tmp_path, monkeypatch, cause):
    """An image processor the core refuses, or one that can't be built, refuses image requests
    with a fixed reason; the error, which may name local paths, is only in the warning."""
    if cause == "refused-settings":
        path = decision_checkpoint(tmp_path / "p", GEMMA4, processor={"max_soft_tokens": 560})
        match = "max_soft_tokens"
    else:
        path = decision_checkpoint(tmp_path / "p", GEMMA4)

        def unreadable(p, unified):
            raise OSError(f"{p}/processor_config.json: permission denied")

        monkeypatch.setattr(tb, "load_image_processor", unreadable)
        match = "permission denied"
    with pytest.warns(UserWarning, match=match), _client(path) as client:
        r = client.post("/v1/systemone", json=request([image_b64(64, 48)]))
        assert client.post("/v1/systemone", json=request()).status_code == 200
    assert r.status_code == 422 and r.json()["detail"] == tb.NO_IMAGE_PROCESSOR
    assert str(tmp_path) not in r.text


def test_refusals_are_422_and_runtime_faults_500(ckpt, monkeypatch):
    """A file whose header reads but whose pixels don't, refused only when the backend decodes
    it, is still the caller's; any other failure answering an accepted request is the server's."""
    good = base64.b64decode(image_b64(64, 48))
    broken = base64.b64encode(good[:40] + b"\x00" * 64 + good[-12:]).decode()  # IHDR intact
    app = server.create_app(str(ckpt), device="cpu", warmup=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/v1/systemone", json=request([broken]))
        assert r.status_code == 422 and "image could not be" in r.json()["detail"]

        def fails(*_a, **_k):
            raise ValueError("an internal check")

        monkeypatch.setattr(app.state.engine.backend, "forward", fails)
        assert client.post("/v1/systemone", json=request()).status_code == 500


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS"),
        ),
    ],
)
def test_concurrent_requests_get_the_same_answers(ckpt, device):
    """Three requests at once on the backend; on MPS, where Metal aborts the process when two
    threads use it at once, the backend's calls take turns."""
    body = request([image_b64(64, 48)])
    with _client(ckpt, device=device, max_concurrency=3) as client:
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


def test_a_full_engine_answers_503_and_refusals_do_not_wait(ckpt, monkeypatch):
    monkeypatch.setitem(decision.BACKENDS, "slow", "test_decision_serving:_SlowBackend")
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


def test_serve_command_options(ckpt, tmp_path, monkeypatch):
    """`serve` takes --dtype, --no-prefix-cache, --max-concurrency and --queue-timeout for a
    decision checkpoint; options for the other kind of checkpoint are refused, each before
    anything is announced, and --vision warns."""
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.append((app, kw)))
    args = ["serve", str(ckpt), "--device", "cpu", "--no-warmup", "--dtype", "bfloat16",
            "--no-prefix-cache", "--max-concurrency", "2", "--queue-timeout", "5",
            "--max-length", "1000"]  # fmt: skip
    out = CliRunner().invoke(cli.app, args)
    assert out.exit_code == 0, out.output
    [(app, kw)] = started
    assert kw == {"host": "127.0.0.1", "port": 8000, "workers": 1}
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert client.post("/v1/systemone", json=request()).status_code == 200
    keys = ("dtype", "reuse_state", "max_concurrency", "queue_timeout", "max_length")
    assert [health[k] for k in keys] == ["bfloat16", False, 2, 5.0, 1000]
    for bad, hint in (
        ([str(tmp_path), "--dtype", "bfloat16"], "--dtype is for decision"),
        ([str(tmp_path), "--revision", "main"], "--revision is for decision"),
        ([str(tmp_path), "--max-length", "100"], "--max-length is for decision"),
        ([str(ckpt), "--dtype", "float16"], "--dtype is one of"),
        ([str(ckpt), "--max-length", "0"], "greater than or equal to 1"),
        ([str(ckpt), "--device-map", "auto"], "--device-map is for v1"),
        ([str(ckpt), "--device", "tpu"], "decision checkpoints run on cpu, cuda, mps"),
    ):
        out = CliRunner().invoke(
            cli.app, ["serve", *bad] + ["--device", "cpu"] * ("tpu" not in bad)
        )
        assert out.exit_code == 2 and "serving" not in out.output, out.output
        assert hint in " ".join(out.output.split()).replace("│ ", ""), out.output
    with pytest.warns(UserWarning, match="--vision is for v1"):
        CliRunner().invoke(cli.app, ["serve", str(ckpt), "--device", "cpu", "--vision"])
    assert len(started) == 2


def test_serve_takes_a_max_length(ckpt, capsys):
    """`--max-length` is the longest prompt served, in place of max_length_validated (4096
    here): a prompt of that many tokens is answered and a longer one refused, never truncated,
    images included; a value past the validated length warns; and one below the warm-up
    prompts' length skips them rather than fail the start."""
    req = request([image_b64(64, 48)] * 2)
    rendered = render(Request.model_validate(req), load_tokenizer(ckpt))
    longest = max(len(q.ids) for q in rendered.questions.values())
    for limit, status in ((longest, 200), (longest - 1, 422)):
        with _client(ckpt, max_length=limit) as client:
            r = client.post("/v1/systemone", json=req)
        assert r.status_code == status, r.text
        if status == 422:
            assert f"exceeds the limit of {limit} tokens" in r.json()["detail"]
    with pytest.warns(UserWarning, match="past the 4096 the checkpoint was validated at"):
        with _client(ckpt, max_length=8192):
            pass
    with TestClient(server.create_app(str(ckpt), device="cpu", max_length=8)):
        pass
    assert "a warm-up request was skipped" in capsys.readouterr().err


# ---- ask -----------------------------------------------------------------------------------


def test_ask_on_a_decision_checkpoint(ckpt, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(image_b64(64, 48)))
    out = _ask(str(ckpt), "--state", "a b c", "--noul", "a b?", "--choice", "which?=billing,sales",
               "--image", str(img), "--device", "cpu", "--dtype", "bfloat16", "--json")  # fmt: skip
    assert out.exit_code == 0, out.output
    answers = json.loads(out.stdout)["answers"]
    assert set(answers) == {"noul_0", "choice_0"}
    assert set(answers["choice_0"]["probabilities"]) == {"billing", "sales"}
    assert cli._exit_fast


def test_ask_refusals_are_usage_errors(ckpt, text_only, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(image_b64(64, 48)))
    for args, hint in (
        ([str(text_only), "--image", str(img), "--device", "cpu"], "does not take images"),
        ([str(ckpt), "--device", "tpu"], "decision checkpoints run on"),
        ([str(ckpt), "--device", "cpu", "--dtype", "float16"], "--dtype is one of"),
        ([str(tmp_path), "--device", "cpu", "--dtype", "bfloat16"], "--dtype is for decision"),
    ):
        out = _ask(*args[:1], "--state", "x", "--noul", "y?", *args[1:])
        assert out.exit_code == 2 and hint in " ".join(out.output.split()), out.output
