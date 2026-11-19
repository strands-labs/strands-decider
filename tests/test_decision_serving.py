"""`strands-decider serve` and `ask` on model checkpoints, with tiny random-weight checkpoints
(tests/decision_support.py): which checkpoints take the decision path, the HTTP API with text
and images, and its refusals. v1 checkpoints keep v1's path; tests/test_server.py and
tests/test_cli_exit.py cover it."""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor

import click
import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_support import image_b64, model_checkpoint, request  # isort: skip

from strands_decider import cli, modeling, server
from strands_decider.core import Request, evaluate, load_tokenizer, render
from strands_decider.decision import (
    NO_IMAGE_PROCESSOR,
    is_model_checkpoint,
    load_decision_engine,
)
from strands_decider.schema import SystemOneResponse

IMAGE = image_b64(64, 48)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return model_checkpoint(tmp_path_factory.mktemp("decision") / "ckpt")


@pytest.fixture(scope="module")
def text_only(tmp_path_factory):
    path = tmp_path_factory.mktemp("text") / "ckpt"
    return model_checkpoint(path, decision={"inputs": ["text"]})


@pytest.fixture(autouse=True)
def _no_exit_fast(monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)  # ask sets it; main() is not run here


def _client(path, **kwargs) -> TestClient:
    return TestClient(server.create_app(str(path), **{"device": "cpu", "warmup": False, **kwargs}))


def _cli(*args: str) -> tuple[int, str]:
    """The command's exit code, and its output on one line as rich's panels wrap it, without the
    colors typer forces under GitHub Actions."""
    out = CliRunner().invoke(cli.app, list(args))
    return out.exit_code, " ".join(click.unstyle(out.output).split()).replace("│ ", "")


def test_which_checkpoints_take_the_decision_path(ckpt, tmp_path, monkeypatch):
    assert is_model_checkpoint(str(ckpt))
    assert not is_model_checkpoint(str(tmp_path / "missing"))  # v1's loader reports it
    assert not is_model_checkpoint(str(tmp_path))  # no decision_config.json: v1's
    (tmp_path / "decision_config.json").write_text(json.dumps({"format_id": "another-format"}))
    assert not is_model_checkpoint(str(tmp_path))  # another decision model's
    calls = []
    snapshot = lambda repo, revision=None: calls.append((repo, revision)) or str(ckpt)  # noqa: E731
    monkeypatch.setattr(modeling, "snapshot_download", snapshot)
    assert is_model_checkpoint("org/decision", revision="v2")
    assert calls == [("org/decision", "v2")]


def test_the_engine_answers_as_the_core_does(ckpt):
    engine = load_decision_engine(str(ckpt), device="cpu", model_name="t")
    images = [IMAGE, image_b64(50, 90, "JPEG")]
    for req in map(Request.model_validate, (request(), request(images), request(state=""))):
        args = (engine.tokenizer, engine.weight, engine.config, engine.backend)
        want = SystemOneResponse.model_validate(evaluate(req, *args, model_name="t"))
        assert engine.evaluate(req) == want


def test_serve_answers_a_model_checkpoint(ckpt):
    with TestClient(server.create_app(str(ckpt), device="cpu", model_name="tiny")) as client:
        health = client.get("/health").json()
        r = client.post("/v1/systemone", json=request([image_b64(50, 90, "JPEG")]))
        assert client.post("/invocations", json=request()).status_code == 200
    want = {"status": "ok", "checkpoint": str(ckpt), "model": "tiny", "images": True,
            "backend": "transformers", "device": "cpu", "dtype": "float32", "reuse_state": True,
            "max_concurrency": 1}  # fmt: skip
    assert health | want == health
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "tiny" and "latency_ms" in body and body["usage"]["output_tokens"] == 3
    assert set(body["answers"]["team"]["probabilities"]) == {"billing", "shipping", "sales"}


def test_serve_refuses_what_the_interface_refuses(ckpt, text_only):
    """Any audio but null, a body that isn't an object, an image that isn't one, and images on a
    text-only checkpoint get 422. A request may omit its state (the core's request model, not
    v1's), and the number of images isn't capped: the prompt's length bounds it."""
    with _client(ckpt) as client:
        post = lambda body: client.post("/v1/systemone", json=body)  # noqa: E731
        for audio in (["UklGRiQAAABXQVZF"], []):
            r = post({**request(), "audio": audio})
            assert r.status_code == 422 and "audio input is not supported" in r.json()["detail"]
        assert post({**request(), "audio": None}).status_code == 200
        assert post([request()]).status_code == 422
        r = post(request(["not base64!"]))
        assert r.status_code == 422 and "base64" in r.json()["detail"]
        stateless = {k: v for k, v in request().items() if k != "state"}
        assert post(stateless).json()["answers"] == post(request(state="")).json()["answers"]
        assert post(request([IMAGE] * 5)).status_code == 200
    with _client(text_only) as client:
        r = client.post("/v1/systemone", json=request([IMAGE]))
        assert r.status_code == 422
        assert r.json()["detail"].startswith("this checkpoint does not take images")
        assert client.get("/health").json()["images"] is False


@pytest.mark.parametrize("cause", ["settings the core refuses", "no processor_config.json"])
def test_a_processor_that_fails_refuses_images_only_and_names_no_path(tmp_path, cause):
    """Clients get a fixed reason; the error, which may name local paths, is only warned."""
    if cause == "settings the core refuses":
        path = model_checkpoint(tmp_path / "p", processor={"max_soft_tokens": 560})
    else:
        path = model_checkpoint(tmp_path / "p")
        (path / "processor_config.json").unlink()
    with pytest.warns(UserWarning, match="image requests are refused"), _client(path) as client:
        r = client.post("/v1/systemone", json=request([IMAGE]))
        assert client.post("/v1/systemone", json=request()).status_code == 200
    assert r.status_code == 422 and r.json()["detail"] == NO_IMAGE_PROCESSOR
    assert str(tmp_path) not in r.text


def test_refusals_are_422_and_runtime_faults_500(ckpt, monkeypatch):
    """A file whose header reads but whose pixels don't, refused only when the backend decodes
    it, is still the caller's; any other failure answering an accepted request is the server's."""
    good = base64.b64decode(IMAGE)
    broken = base64.b64encode(good[:41] + b"\x00" * 64 + good[105:]).decode()  # pixel data only
    app = server.create_app(str(ckpt), device="cpu", warmup=False)

    def fails(*_):
        raise ValueError("an internal check")

    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/v1/systemone", json=request([broken]))
        assert r.status_code == 422 and "image could not be decoded" in r.json()["detail"]
        monkeypatch.setattr(app.state.engine.backend, "forward", fails)
        assert client.post("/v1/systemone", json=request()).status_code == 500


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
        ),
    ],
)
def test_concurrent_requests_get_the_same_answers(ckpt, device):
    """Three requests at once on the backend; on MPS, where Metal aborts the process when two
    threads use it at once, the backend's calls take turns."""
    with _client(ckpt, device=device, max_concurrency=3) as client, ThreadPoolExecutor(6) as pool:
        got = list(
            pool.map(lambda _: client.post("/v1/systemone", json=request([IMAGE])), range(6))
        )
    assert all(r.status_code == 200 for r in got)
    assert len({json.dumps(r.json()["answers"], sort_keys=True) for r in got}) == 1


def test_a_full_engine_answers_503_and_refusals_do_not_wait(ckpt, monkeypatch):
    app = server.create_app(str(ckpt), device="cpu", warmup=False, queue_timeout=0.1)
    forward = app.state.engine.backend.forward
    monkeypatch.setattr(
        app.state.engine.backend, "forward", lambda *a: time.sleep(0.4) or forward(*a)
    )
    with TestClient(app) as client, ThreadPoolExecutor(2) as pool:
        first = pool.submit(client.post, "/v1/systemone", json=request())
        time.sleep(0.1)  # the first holds the only slot
        second = pool.submit(client.post, "/v1/systemone", json=request())
        started = time.perf_counter()
        refused = client.post("/v1/systemone", json={**request(), "audio": ["x"]})
        assert refused.status_code == 422 and time.perf_counter() - started < 0.3
        assert second.result().status_code == 503 and "busy" in second.result().json()["detail"]
        assert first.result().status_code == 200


def test_serve_command_options(ckpt, tmp_path, monkeypatch):
    """`serve` gives a model checkpoint its options; an option for the other kind of
    checkpoint is refused before anything is announced, and --vision warns."""
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.append((app, kw)))
    code, out = _cli("serve", str(ckpt), "--device", "cpu", "--no-warmup", "--dtype", "bfloat16",
                     "--no-prefix-cache", "--max-concurrency", "2", "--queue-timeout", "5",
                     "--max-length", "1000")  # fmt: skip
    assert code == 0, out
    [(app, kw)] = started
    assert kw == {"host": "127.0.0.1", "port": 8000, "workers": 1}
    health = TestClient(app).get("/health").json()
    keys = ("dtype", "reuse_state", "max_concurrency", "queue_timeout", "max_length")
    assert [health[k] for k in keys] == ["bfloat16", False, 2, 5.0, 1000]
    for args, hint in (
        (
            [str(tmp_path), "--dtype", "bfloat16"],
            "--dtype is not supported for v1 model checkpoints",
        ),
        (
            [str(tmp_path), "--revision", "main"],
            "--revision is not supported for v1 model checkpoints",
        ),
        (
            [str(tmp_path), "--max-length", "100"],
            "--max-length is not supported for v1 model checkpoints",
        ),
        ([str(ckpt), "--dtype", "float16"], "--dtype is one of"),
        ([str(ckpt), "--max-length", "0"], "greater than or equal to 1"),
        ([str(ckpt), "--device-map", "auto"], "--device-map is only supported for v1"),
        ([str(ckpt), "--device", "tpu"], "model checkpoints run on cpu, cuda, mps"),
    ):
        code, out = _cli("serve", *args)
        assert code == 2 and "serving" not in out and hint in out, out
    with pytest.warns(UserWarning, match="--vision is only supported for v1"):
        _cli("serve", str(ckpt), "--device", "cpu", "--no-warmup", "--vision")
    assert len(started) == 2


def test_serve_takes_a_max_length(ckpt, caplog):
    """`--max-length` is the longest prompt served, in place of max_length_validated (8192
    here): a prompt of that many tokens is answered and a longer one refused, never truncated,
    images included; a value past the validated length warns; and one below the warm-up
    prompts' length skips them rather than fail the start."""
    req = request([IMAGE] * 2)
    rendered = render(Request.model_validate(req), load_tokenizer(ckpt))
    longest = max(len(q.ids) for q in rendered.questions.values())
    for limit, status in ((longest, 200), (longest - 1, 422)):
        with _client(ckpt, max_length=limit) as client:
            r = client.post("/v1/systemone", json=req)
        assert r.status_code == status, r.text
    assert f"exceeds the limit of {longest - 1} tokens" in r.json()["detail"]
    with pytest.warns(UserWarning, match="past the 8192 the checkpoint was validated at"):
        server.create_app(str(ckpt), device="cpu", warmup=False, max_length=10000)
    server.create_app(str(ckpt), device="cpu", max_length=8)
    assert "a warm-up request was skipped" in caplog.text


def test_ask_on_a_model_checkpoint(ckpt, text_only, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(IMAGE))
    out = CliRunner().invoke(cli.app, ["ask", str(ckpt), "--state", "a b c", "--noul", "a b?",
                                       "--choice", "which?=billing,sales", "--image", str(img),
                                       "--device", "cpu", "--dtype", "bfloat16", "--json"])  # fmt: skip
    assert out.exit_code == 0, out.output
    answers = json.loads(out.stdout)["answers"]
    assert set(answers) == {"noul_0", "choice_0"}
    assert set(answers["choice_0"]["probabilities"]) == {"billing", "sales"}
    assert cli._exit_fast
    for args, hint in (
        ([str(text_only), "--image", str(img), "--device", "cpu"], "does not take images"),
        (
            [str(tmp_path), "--device", "cpu", "--dtype", "bfloat16"],
            "--dtype is not supported for v1 model checkpoints",
        ),
    ):
        code, out = _cli("ask", args[0], "--state", "x", "--noul", "y?", *args[1:])
        assert code == 2 and hint in out, out
