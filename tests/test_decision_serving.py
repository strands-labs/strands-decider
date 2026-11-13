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
    (other / "decision_config.json").write_text(json.dumps({"format_id": "another-format"}))
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
    with pytest.raises(ValueError, match="decision checkpoints run on cpu, cuda, mps here"):
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
    with pytest.raises(ValueError, match="decision checkpoints run on cpu, cuda, mps here"):
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


def test_serve_refusals_over_http(ckpt, tmp_path):
    """Through create_app and HTTP: audio, five images, and images on a text-only checkpoint
    are each refused with 422 and the interface's message."""
    img = image_b64(64, 48)
    with TestClient(server.create_app(str(ckpt), device="cpu", warmup=False)) as client:
        r = client.post("/v1/systemone", json={**request(), "audio": ["UklGRiQAAABXQVZF"]})
        assert r.status_code == 422 and r.json()["detail"] == "audio input is not supported"
        assert client.post("/v1/systemone", json={**request(), "audio": []}).status_code == 200
        r = client.post("/v1/systemone", json=request([img] * 5))
        assert (
            r.status_code == 422 and r.json()["detail"] == "5 images; this server takes at most 4"
        )
        assert client.post("/v1/systemone", json=request([img] * 4)).status_code == 200
    text_only = decision_checkpoint(tmp_path / "t", GEMMA4, decision={"inputs": ["text"]})
    with TestClient(server.create_app(str(text_only), device="cpu", warmup=False)) as client:
        r = client.post("/v1/systemone", json=request([img]))
        assert r.status_code == 422
        assert r.json()["detail"].startswith("this checkpoint does not take images")


class _StubBackend:
    """A backend another runtime might register: it returns each question's probabilities."""

    outputs = "probabilities"
    image_problem = None

    def __init__(self, path, **settings):
        self.path, self.settings, self.calls = path, settings, []

    def describe(self):
        return {"backend": "stub", "device": self.settings["device"], "dtype": "none"}

    def forward(self, state, questions, images=()):
        import numpy as np

        self.calls.append((len(state), len(questions), len(images)))
        return [np.full(255, 1 / 255) for _ in questions]


def test_another_runtime_plugs_in_by_device(ckpt, monkeypatch):
    """A backend registered for a device kind gets the engine, the serve wiring and its checks:
    the raw request, the refusals, the health check."""
    from strands_decider import decision

    monkeypatch.setattr(decision, "_LOADERS", dict(decision._LOADERS))
    seen = []
    decision.register_backend(
        ["stub"], lambda path, **kw: seen.append(_StubBackend(path, **kw)) or seen[-1]
    )
    with TestClient(server.create_app(str(ckpt), device="stub:0", warmup=False)) as client:
        health = client.get("/health").json()
        assert (health["backend"], health["device"], health["images"]) == ("stub", "stub:0", True)
        r = client.post("/v1/systemone", json=request([image_b64(64, 48)]))
        assert r.status_code == 200, r.text
        assert r.json()["answers"]["team"]["probabilities"] == {
            "billing": 0.3333,
            "shipping": 0.3333,
            "sales": 0.3333,
        }
        assert client.post("/v1/systemone", json={**request(), "audio": ["x"]}).status_code == 422
    [backend] = seen
    assert backend.path == str(ckpt) and backend.settings["reuse_state"] is True
    [(state_tokens, questions, images)] = backend.calls  # the state once, its image in it
    assert (questions, images) == (3, 1) and state_tokens > 266
    with pytest.raises(ValueError, match=r"decision checkpoints run on .*stub.* here, not 'tpu'"):
        decision.check_device("tpu")


# ---- finding a decision checkpoint on the Hub ----------------------------------------------


def _hub_error(cls, message="x"):
    """A huggingface_hub error without an HTTP response behind it."""
    e = cls.__new__(cls)
    Exception.__init__(e, message)
    return e


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """huggingface_hub's cache lookup and downloads, recorded; none reaches the network.
    Online unless the test sets `offline`."""
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
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
        lambda repo, fn, revision=None, **_: outcome(("cache", revision), state["cache"]),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, fn, revision=None, **_: outcome(("download", revision), state["download"]),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, revision=None, local_files_only=False, **_: outcome(
            ("snapshot", revision, local_files_only), state["snapshot"]
        ),
    )
    dc = tmp_path / "snap" / "decision_config.json"
    dc.parent.mkdir()
    dc.write_text(json.dumps({"interface": "strands-decider"}))
    return state, calls, str(dc), lambda: monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)


COMMIT = "a" * 40


def test_hub_detection_online_asks_the_hub_unless_the_commit_is_pinned(hub):
    """A cached answer for a branch may be an old commit's: online, the Hub is asked (one
    small request, which the cache serves when current). A pinned commit never changes."""
    state, calls, dc, _ = hub
    state["cache"], state["download"] = object(), dc  # the cache's stale "no such file"
    assert checkpoint_kind("org/repo", revision="main") == DECISION
    assert calls == [("download", "main")]
    calls.clear()
    assert checkpoint_kind("org/repo", revision=COMMIT) is None  # pinned: the cache answers
    assert calls == [("cache", COMMIT)]
    calls.clear()
    state["cache"] = dc
    assert checkpoint_kind("org/repo", revision=COMMIT) == DECISION and calls == [("cache", COMMIT)]


def test_hub_detection_offline_reads_the_cache_and_the_snapshot(hub):
    state, calls, dc, go_offline = hub
    go_offline()
    state["cache"] = dc
    assert checkpoint_kind("org/repo") == DECISION and calls == [("cache", None)]
    calls.clear()
    state["cache"], state["snapshot"] = None, str(__import__("pathlib").Path(dc).parent)
    assert checkpoint_kind("org/repo", revision="v2") == DECISION
    assert calls == [("cache", "v2"), ("snapshot", "v2", True)]
    from huggingface_hub.errors import LocalEntryNotFoundError

    state["snapshot"] = _hub_error(LocalEntryNotFoundError)  # nothing cached: v1 reports it
    assert checkpoint_kind("org/repo") is None
    assert not any(c[0] == "download" for c in calls)


def test_hub_detection_takes_only_missing_files_and_repos_for_v1(hub):
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    state, _, dc, _ = hub
    for missing in (RemoteEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
        state["download"] = _hub_error(missing)
        assert checkpoint_kind("org/repo") is None, missing
    # An authorization error, and a rate limit or a 5xx, which hf_hub_download raises as a
    # LocalEntryNotFoundError caused by the Hub's answer: raised, never taken for v1.
    answered = _hub_error(LocalEntryNotFoundError, "the Hub answered 503")
    answered.__cause__ = _hub_error(HfHubHTTPError, "503 Service Unavailable")
    state["snapshot"] = str(__import__("pathlib").Path(dc).parent)  # even with a cache
    for error in (_hub_error(HfHubHTTPError, "401"), answered):
        state["download"] = error
        with pytest.raises(type(error)):
            checkpoint_kind("org/repo")


def _hub_cache(root, repo: str, files: dict[str, str], commit: str = "c" * 40) -> None:
    """A Hugging Face cache holding `repo`'s snapshot at `main` with `files`."""
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(commit)
    snapshot = folder / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name, text in files.items():
        (snapshot / name).write_text(text)


@pytest.fixture
def unreachable_hub(monkeypatch, tmp_path):
    """The real hf_hub_download and snapshot_download over a cache in `tmp_path`, online, with
    their metadata request failing: as the network does (ConnectError), or with the Hub's
    answer when the test sets `answer` to a status code."""
    import huggingface_hub.file_download as fd
    from huggingface_hub import constants
    from huggingface_hub.errors import HfHubHTTPError

    httpx = getattr(fd, "httpx2", None) or fd.httpx
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    answer = {"status": None}

    def fail(url, *_a, **_k):
        if answer["status"] is None:
            raise httpx.ConnectError("the network is down")
        response = httpx.Response(answer["status"], request=httpx.Request("HEAD", url))
        raise HfHubHTTPError(f"{answer['status']} from the Hub", response=response)

    monkeypatch.setattr(fd, "get_hf_file_metadata", fail)
    return cache, answer, httpx


def test_hub_detection_with_the_network_down_raises_when_nothing_is_cached(unreachable_hub):
    """No mock of hf_hub_download: its error reaches the caller, caused by the network's."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    _, _, httpx = unreachable_hub
    with pytest.raises(LocalEntryNotFoundError) as e:
        checkpoint_kind("org/decision-repo")
    assert isinstance(e.value.__cause__, httpx.ConnectError)


def test_hub_detection_with_the_network_down_reads_the_cached_snapshot(unreachable_hub):
    """With the Hub out of reach, a cached snapshot decides, as v1's loader then uses it: a
    cached v1 checkpoint stays v1, and a cached decision checkpoint a decision one."""
    cache, _, _ = unreachable_hub
    _hub_cache(cache, "org/v1-model", {"strands_decider_config.json": "{}"})
    _hub_cache(
        cache,
        "org/decision-model",
        {"decision_config.json": json.dumps({"interface": "strands-decider"})},
    )
    assert checkpoint_kind("org/v1-model") is None
    assert checkpoint_kind("org/decision-model") == DECISION


def test_hub_detection_raises_the_hubs_answer_even_with_a_cache(unreachable_hub):
    """A Hub that answers, with a 503 or a 429, is not out of reach: its error is raised, and
    the cache does not decide."""
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    cache, answer, _ = unreachable_hub
    _hub_cache(cache, "org/v1-model", {"strands_decider_config.json": "{}"})
    for status in (503, 429):
        answer["status"] = status
        with pytest.raises(LocalEntryNotFoundError) as e:
            checkpoint_kind("org/v1-model")
        assert isinstance(e.value.__cause__, HfHubHTTPError)


def test_a_mistyped_path_keeps_v1s_message(tmp_path):
    """Neither a directory nor a repo id: detection says v1, and v1's loader explains."""
    from typer.testing import CliRunner

    typo = str(tmp_path / "no" / "such" / "checkpoint")
    assert checkpoint_kind(typo) is None
    out = CliRunner().invoke(
        cli.app, ["ask", typo, "--state", "x", "--noul", "y?", "--device", "cpu"]
    )
    assert out.exit_code != 0
    assert "not a local directory" in str(out.exception)


# ---- serving decisions: the details, the bound, the CLI ------------------------------------


def test_a_422_never_names_a_local_path(tmp_path, monkeypatch):
    from strands_decider.backends import transformers as tb

    path = decision_checkpoint(tmp_path / "leak", GEMMA4)

    def unreadable(p, unified):
        raise OSError(f"{p}/processor_config.json: permission denied")

    monkeypatch.setattr(tb, "load_image_processor", unreadable)
    with pytest.warns(UserWarning, match="permission denied"):
        app = server.create_app(str(path), device="cpu", warmup=False)
        with TestClient(app) as client:
            r = client.post("/v1/systemone", json=request([image_b64(64, 48)]))
    assert r.status_code == 422
    assert (
        r.json()["detail"]
        == "the image processor could not be built, so image requests are refused"
    )
    assert str(tmp_path) not in r.text


def test_the_engine_answers_as_the_core_does(ckpt):
    """The engine renders once and runs the backend and readout itself; the result is
    core.evaluate's."""
    from strands_decider.core import evaluate

    engine = load_decision_engine(str(ckpt), device="cpu", model_name="t")
    for req in (
        request(),
        request([image_b64(64, 48), image_b64(50, 90, "JPEG")]),
        request(state=""),
    ):
        want = evaluate(
            req, engine.tokenizer, engine.readout, engine.config, engine.backend, model_name="t"
        )
        assert (
            engine.evaluate(req).model_dump() == SystemOneResponse.model_validate(want).model_dump()
        )


class _SlowBackend(_StubBackend):
    def forward(self, state, questions, images=()):
        import time

        time.sleep(0.4)
        return super().forward(state, questions, images)


def test_a_full_engine_answers_503_and_refusals_do_not_wait(ckpt, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from strands_decider import decision

    monkeypatch.setattr(decision, "_LOADERS", dict(decision._LOADERS))
    decision.register_backend(["slow"], lambda path, **kw: _SlowBackend(path, **kw))
    app = server.create_app(str(ckpt), device="slow", warmup=False, queue_timeout=0.1)
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/v1/systemone", json=request())
            import time

            time.sleep(0.1)  # the first holds the only slot
            second = pool.submit(client.post, "/v1/systemone", json=request())
            started = time.perf_counter()
            refused = client.post("/v1/systemone", json={**request(), "audio": ["x"]})
            assert refused.status_code == 422 and time.perf_counter() - started < 0.3
            assert second.result().status_code == 503 and "busy" in second.result().json()["detail"]
            assert first.result().status_code == 200


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="needs Apple's MPS")
def test_serve_on_mps_answers_concurrent_requests_alike(ckpt):
    from concurrent.futures import ThreadPoolExecutor

    body = request([image_b64(64, 48)])
    app = server.create_app(str(ckpt), device="mps", warmup=False, max_concurrency=3)
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=6) as pool:
            got = list(pool.map(lambda _: client.post("/v1/systemone", json=body), range(6)))
    assert all(r.status_code == 200 for r in got)
    assert len({json.dumps(r.json()["answers"], sort_keys=True) for r in got}) == 1


def test_serve_command_options(ckpt, tmp_path, monkeypatch):
    """`serve` takes --dtype, --max-concurrency, --queue-timeout and --revision for decision
    checkpoints, refuses them for v1 ones and refuses an intermediate checkpoint, each before
    anything is announced."""
    import uvicorn
    from typer.testing import CliRunner

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
        (
            [str(tmp_path), "--device", "cpu", "--max-concurrency", "2"],
            "--max-concurrency is for decision",
        ),
        ([str(tmp_path), "--device", "cpu", "--dtype", "bfloat16"], "--dtype is for decision"),
        ([str(intermediate_checkpoint(tmp_path / "mid")), "--device", "cpu"], "intermediate"),
        ([str(ckpt), "--device", "cpu", "--dtype", "float16"], "--dtype on cpu is one of"),
        ([str(ckpt), "--device", "cuda:x"], "a device index is a number"),
    ):
        out = CliRunner().invoke(cli.app, ["serve", *bad])
        assert out.exit_code == 2 and "serving" not in out.output, out.output
        assert hint in " ".join(out.output.split()).replace("│ ", ""), out.output
    assert len(started) == 1


def test_ask_reports_a_refused_request_as_a_usage_error(tmp_path, monkeypatch):
    import base64

    from typer.testing import CliRunner

    path = decision_checkpoint(tmp_path / "t", GEMMA4, decision={"inputs": ["text"]})
    img = tmp_path / "x.png"
    img.write_bytes(base64.b64decode(image_b64(64, 48)))
    monkeypatch.setattr(cli, "_exit_fast", False)
    args = [
        "ask",
        str(path),
        "--state",
        "x",
        "--noul",
        "y?",
        "--image",
        str(img),
        "--device",
        "cpu",
    ]
    out = CliRunner().invoke(cli.app, args)
    assert out.exit_code == 2 and "does not take images" in " ".join(out.output.split())


# ---- the backend registry ---------------------------------------------------------------


def test_the_registry_checks_what_it_is_given(monkeypatch):
    from strands_decider import decision

    monkeypatch.setattr(decision, "_LOADERS", dict(decision._LOADERS))
    monkeypatch.setattr(decision, "BACKENDS", dict(decision.BACKENDS))
    with pytest.raises(ValueError, match="register device kinds"):
        decision.register_backend(["cuda:1"], lambda *a, **k: None)
    with pytest.raises(ValueError, match="a device index is a number"):
        decision.check_device("cuda:x")
    decision.check_device("cuda:1")
    # Editing BACKENDS after a first load takes effect: imported loaders are kept by path.
    first = decision.backend_loader("cpu")
    decision.BACKENDS["cpu"] = "strands_decider.decision:refuse_intermediate"
    assert decision.backend_loader("cpu") is decision.refuse_intermediate is not first
    # A loader's own dtypes decide what --dtype may be.

    def loader(path, **kw):
        return _StubBackend(path, **kw)

    loader.dtypes = ("float16",)
    decision.register_backend(["half"], loader)
    decision.check_dtype("half", "float16")
    with pytest.raises(ValueError, match="one of float16"):
        decision.check_dtype("half", "bfloat16")
