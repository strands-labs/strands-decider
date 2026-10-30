"""Cross-request batching (`prepare` / `evaluate_prepared`, the server's MicroBatcher) gives
each request the answer it gets alone.

Rows of different requests share one right-padded forward. That is exact for every real
position: attention is causal with padded keys masked; Qwen3.5's Gated DeltaNet layers
zero padded inputs and run a causal convolution and recurrence; Gemma 4's per-layer
embeddings and positions are per token. These tests pin it on tiny random torsos of each
family (plain Qwen3, hybrid Qwen3.5, Gemma 4 with per-layer inputs and eager attention),
plus the planner's budgets, OOM splitting, error isolation and the server wiring.
CPU only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from fastapi.testclient import TestClient

transformers = pytest.importorskip("transformers")

from strands_decider import server  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine, plan_rows  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["<state> <question> <options> <answer> choice score noul"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<pad>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


def _torso(kind, vocab):
    torch.manual_seed(0)
    if kind == "qwen3":
        return transformers.Qwen3Model(
            transformers.Qwen3Config(
                vocab_size=vocab,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        )
    if kind == "qwen3_5":
        if not hasattr(transformers, "Qwen3_5TextConfig"):
            pytest.skip("this transformers has no Qwen3.5")
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        return transformers.Qwen3_5ForCausalLM(cfg).model.eval()
    if kind == "gemma4":
        if not hasattr(transformers, "Gemma4TextConfig"):
            pytest.skip("this transformers has no Gemma 4")
        cfg = transformers.Gemma4TextConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=vocab,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            sliding_window=8,
            num_kv_shared_layers=1,
            final_logit_softcapping=30.0,
            tie_word_embeddings=True,
            pad_token_id=0,
        )
        torso = transformers.Gemma4ForCausalLM(cfg).model.eval()
        torso.set_attn_implementation("eager")  # as served for E2B (one KV head)
        return torso
    raise ValueError(kind)


def _model(kind="qwen3"):
    tok = _tokenizer()
    cfg = StrandsDeciderConfig(
        base_model="tiny",
        head_type="pointer",
        pointer_dim=16,
        use_lora=False,
        torch_dtype="float32",
        max_length=512,
    )
    return StrandsDeciderModel(cfg, _torso(kind, len(tok)), tok)


def _engine(kind="qwen3", **kw):
    return SystemOneEngine(_model(kind), EngineConfig(device="cpu", **kw))


def _requests(n=24):
    """Single-question requests of very different lengths and all three kinds."""
    out = []
    for i in range(n):
        state = f"event {i} logged; " * (1 + (i * 7) % 13)
        if i % 3 == 0:
            q = ChoiceQuestion(
                type="choice",
                instructions="pick " * (1 + i % 5),
                criteria={f"option{k}" + "x" * ((i + k) % 7): "" for k in range(2 + i % 4)},
            )
        elif i % 3 == 1:
            q = NoulQuestion(type="noul", instructions="is it ok? " * (1 + i % 3))
        else:
            q = ScoreQuestion(
                type="score",
                instructions="rate it",
                criteria=[f"level {k}" for k in range(3 + i % 3)],
            )
        out.append(SystemOneRequest(state=state, questions={"q": q}))
    return out


def _probs(resp):
    a = resp.answers["q"].model_dump()
    return a.get("probabilities") or {"noul": a["noul"]}


def _close(a, b, tol=2e-4):  # responses are rounded to 4 decimals
    assert a.keys() == b.keys()
    for k in a:
        assert abs(a[k] - b[k]) <= tol, (k, a[k], b[k])


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5", "gemma4"])
def test_batched_answers_equal_unbatched(kind):
    eng = _engine(kind)
    reqs = _requests()
    alone = [eng.evaluate(r) for r in reqs]
    prepared = [eng.prepare(r) for r in reqs]
    lengths = sorted(p.tokens for p in prepared)
    assert lengths[-1] > 3 * lengths[0]  # real padding in the shared forwards
    together = eng.evaluate_prepared(prepared)
    for a, b in zip(alone, together, strict=True):
        assert not isinstance(b, BaseException), b
        _close(_probs(a), _probs(b))
        assert a.usage == b.usage


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5", "gemma4"])
def test_rows_match_bit_for_bit_in_fp32_tolerance(kind):
    """The raw probabilities (not rounded) of a row alone and padded among others."""
    eng = _engine(kind)
    rows = [r for p in map(eng.prepare, _requests(12)) for r in p.rows]
    together = eng._forward_rows(rows)
    for j, r in enumerate(rows):
        alone = eng._forward_rows([r])
        torch.testing.assert_close(
            together[j, : r.n_slots], alone[0, : r.n_slots], atol=1e-5, rtol=1e-5
        )


def test_plan_respects_budgets():
    lengths = [10, 300, 20, 40, 300, 5, 200, 80]
    groups = plan_rows(lengths, max_tokens=400, max_rows=3)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        assert len(g) <= 3
        assert len(g) == 1 or len(g) * max(lengths[i] for i in g) <= 400
    # quadratic cap (eager attention): rows x longest^2 <= cap
    for g in plan_rows(lengths, max_tokens=10_000, max_rows=64, quadratic_cap=300**2):
        assert len(g) == 1 or len(g) * max(lengths[i] for i in g) ** 2 <= 300**2
    # a row over the budget alone still runs, alone
    assert plan_rows([1000, 10], max_tokens=100, max_rows=8) == [[1], [0]]
    # padding waste: long rows of unlike length do not pad together; short forwards may
    assert plan_rows([1000, 3000], 10_000, 8, pad_slack=0.25, pad_free=2048) == [[0], [1]]
    assert plan_rows([2900, 3000], 10_000, 8, pad_slack=0.25, pad_free=2048) == [[0, 1]]
    assert plan_rows([100, 900], 10_000, 8, pad_slack=0.25, pad_free=2048) == [[0, 1]]


def test_tuned_caps():
    from strands_decider.infer import tuned_token_cap

    by = {256: 32, 1024: 2, 4096: 1}
    assert tuned_token_cap(by, 256) == 8192
    assert tuned_token_cap(by, 600) == 2048  # the smaller of its neighbours
    assert tuned_token_cap(by, 100) == 8192 and tuned_token_cap(by, 8000) == 4096
    groups = plan_rows([256] * 40 + [1000] * 6, 100_000, 64, rows_by_width=by)
    assert sorted(len(g) for g in groups) == [2, 2, 2, 8, 32]


def test_tune_batching_records_a_plan():
    eng = _engine()
    by = eng.tune_batching(widths=(16, 64), reps=1)
    assert set(by) == {16, 64} and all(k >= 1 for k in by.values())
    assert eng.cfg.batch_rows_by_width == by
    reqs = _requests(8)
    alone = [eng.evaluate(r) for r in reqs]
    for a, b in zip(alone, eng.evaluate_prepared([eng.prepare(r) for r in reqs]), strict=True):
        _close(_probs(a), _probs(b))


def test_prepare_refuses_like_evaluate():
    eng = _engine()
    big = SystemOneRequest(state="x " * 2000, questions={"q": _requests(1)[0].questions["q"]})
    with pytest.raises(ValueError, match="context window"):
        eng.evaluate(big)
    with pytest.raises(ValueError, match="context window"):
        eng.prepare(big)


def test_oom_in_a_shared_forward_splits_and_answers(monkeypatch):
    """A shared forward that runs out of memory is split; only a row that cannot run even
    alone gets the OOM, and its neighbours are answered."""
    eng = _engine(batch_tokens=100_000)
    reqs = _requests(8)
    alone = [eng.evaluate(r) for r in reqs]
    prepared = [eng.prepare(r) for r in reqs]
    huge = max(range(8), key=lambda k: prepared[k].tokens)
    real = SystemOneEngine._forward_rows

    def forward(self, rows):
        if len(rows) > 2 or any(r is prepared[huge].rows[0] for r in rows):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
        return real(self, rows)

    monkeypatch.setattr(SystemOneEngine, "_forward_rows", forward)
    out = eng.evaluate_prepared(prepared)
    for k in range(8):
        if k == huge:
            assert isinstance(out[k], torch.OutOfMemoryError)
        else:
            _close(_probs(alone[k]), _probs(out[k]))


def test_lost_context_stops_the_cycle(monkeypatch):
    eng = _engine(batch_tokens=64, batch_rows=2)
    prepared = [eng.prepare(r) for r in _requests(8)]
    calls = []

    def forward(self, rows):
        calls.append(len(rows))
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(SystemOneEngine, "_forward_rows", forward)
    out = eng.evaluate_prepared(prepared)
    assert len(calls) == 1  # no further forwards on a lost context
    assert all(isinstance(x, RuntimeError) for x in out)


def test_multi_question_requests_take_the_shared_prefix_path():
    eng = _engine()
    qs = {f"q{j}": r.questions["q"] for j, r in enumerate(_requests(3))}
    req = SystemOneRequest(state="a shared state " * 5, questions=qs)
    p = eng.prepare(req)
    assert p.rows is None
    (resp,) = eng.evaluate_prepared([p])
    want = eng.evaluate(req)
    assert resp.model_dump() == want.model_dump()


# ---------------------------------------------------------------- server


def _client(monkeypatch, engine_factory=None, wait_ms=5.0, kind="qwen3"):
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _model(kind))
    )
    if engine_factory is not None:
        monkeypatch.setattr(server, "SystemOneEngine", engine_factory)
    return TestClient(
        server.create_app(
            "checkpoints/does-not-exist", device="cpu", batching=True, batch_wait_ms=wait_ms
        ),
        raise_server_exceptions=False,
    )


def _body(req):
    return req.model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5"])
def test_server_batches_concurrent_requests(monkeypatch, kind):
    client = _client(monkeypatch, wait_ms=20.0, kind=kind)
    reqs = _requests(24)
    serial = [client.post("/v1/systemone", json=_body(r)).json() for r in reqs]
    with ThreadPoolExecutor(max_workers=12) as ex:
        par = list(ex.map(lambda r: client.post("/v1/systemone", json=_body(r)), reqs))
    for s, p in zip(serial, par, strict=True):
        assert p.status_code == 200, p.text
        a, b = s["answers"]["q"], p.json()["answers"]["q"]
        _close(
            a.get("probabilities") or {"noul": a["noul"]},
            b.get("probabilities") or {"noul": b["noul"]},
        )
    h = client.get("/health").json()
    assert h["batching"]["requests"] == 48
    assert h["batching"]["max_requests_per_cycle"] > 1  # requests did share forwards


def test_server_refusal_is_422_and_neighbours_are_answered(monkeypatch):
    client = _client(monkeypatch, wait_ms=20.0)
    good = _requests(6)
    bad = SystemOneRequest(state="x " * 2000, questions=good[0].questions)
    with ThreadPoolExecutor(max_workers=7) as ex:
        res = list(ex.map(lambda r: client.post("/v1/systemone", json=_body(r)), [*good, bad]))
    assert [r.status_code for r in res] == [200] * 6 + [422]
    assert client.get("/health").status_code == 200


def test_server_error_fails_only_that_request_and_keeps_serving(monkeypatch):
    """An error in the device thread fails the request(s) it belongs to (HTTP 500, as
    without batching) and the batcher keeps serving the next ones."""

    class _OomOnce(SystemOneEngine):
        hit = False

        def _forward_rows(self, rows):
            if not type(self).hit:
                type(self).hit = True
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
            return super()._forward_rows(rows)

    client = _client(monkeypatch, lambda model, cfg: _OomOnce(model, cfg))
    body = _body(_requests(1)[0])
    assert client.post("/v1/systemone", json=body).status_code == 500
    assert client.post("/v1/systemone", json=body).status_code == 200
    assert client.get("/health").status_code == 200


def test_server_without_batching_is_unchanged(monkeypatch):
    """Off by default on CPU (and MPS, MLX): requests run as before."""
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _model())
    )
    client = TestClient(server.create_app("checkpoints/does-not-exist", device="cpu"))
    assert client.get("/health").json()["batching"] is None


class _AloneEngine(SystemOneEngine):
    """Overrides `evaluate`, as the vision and MLX engines do, so it cannot batch rows."""

    def evaluate(self, request):
        return super().evaluate(request)


@pytest.mark.parametrize(
    "batching, factory, on",
    [
        (None, None, True),  # the CUDA default
        (False, None, False),  # --no-batching
        (None, _AloneEngine, False),  # an engine that scores requests alone is not queued
    ],
)
def test_batching_defaults_on_for_cuda(monkeypatch, batching, factory, on):
    """Only /health is called: the "cuda" engine's tiny model stays on the CPU, and the
    startup tune (on by default with batching on CUDA) is recorded rather than timed."""
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _model())
    )
    monkeypatch.setattr(StrandsDeciderModel, "to", lambda self, *a, **k: self)
    tuned = []

    def tune(self, *a, **k):
        tuned.append(True)
        self.cfg.batch_rows_by_width = {256: 8}
        return self.cfg.batch_rows_by_width

    monkeypatch.setattr(SystemOneEngine, "tune_batching", tune)
    if factory is not None:
        monkeypatch.setattr(server, "SystemOneEngine", factory)
    app = server.create_app(
        "checkpoints/does-not-exist",
        device="cuda",
        batching=batching,
        max_batch=16,
        max_batch_tokens=4096,
    )
    h = TestClient(app).get("/health").json()
    assert (h["batching"] is not None) is on
    assert bool(tuned) is on  # tuned exactly when batching
    if on:  # one set of forward budgets
        assert (h["batching"]["batch_rows"], h["batching"]["batch_tokens"]) == (16, 4096)
        assert h["batching"]["wait_ms"] == 0.0
        assert h["batching"]["rows_by_width"] == {"256": 8}


def test_no_batch_tune_skips_the_startup_tune(monkeypatch):
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _model())
    )
    monkeypatch.setattr(StrandsDeciderModel, "to", lambda self, *a, **k: self)
    monkeypatch.setattr(
        SystemOneEngine, "tune_batching", lambda self, *a, **k: pytest.fail("tuned")
    )
    app = server.create_app("checkpoints/does-not-exist", device="cuda", batch_tune=False)
    assert TestClient(app).get("/health").json()["batching"]["rows_by_width"] is None


def test_tune_stops_at_the_largest_forward_that_fits(monkeypatch):
    eng = _engine()
    real = SystemOneEngine._forward_rows

    def forward(self, rows):
        if len(rows) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 9 GiB")
        return real(self, rows)

    monkeypatch.setattr(SystemOneEngine, "_forward_rows", forward)
    by = eng.tune_batching(widths=(16, 64), reps=1, gain=0.0)  # gain 0: every k that runs wins
    assert by == {16: 2, 64: 2}
