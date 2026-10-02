"""The Strands tools against a scripted decider: no weights, no network, no GPU.

These pin what the tools do with an answer: input tolerance, refusals before any forward
pass, thresholds, the result shape, and the three ways an endpoint is resolved. Whether the
model answers *correctly* is the evaluation suite's question, not this file's.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

strands = pytest.importorskip("strands", reason="the tools need the strands extra")

from strands import Agent  # noqa: E402
from strands.models import BedrockModel  # noqa: E402

from strands_decider.infer import _to_answer  # noqa: E402
from strands_decider.prompting import render_question  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)
from strands_decider.tools import (  # noqa: E402
    ALL_TOOLS,
    DeciderUnavailable,
    HttpDecider,
    LocalDecider,
    _common,
    configure,
    decider_ask,
    decider_check,
    decider_choose,
    decider_classify,
    decider_fan_out,
    decider_info,
    decider_rank,
    decider_rate,
    decider_route,
    decider_sift,
    decider_usage,
)

# ------------------------------------------------------------------------------ helpers


def noul(p: float) -> NoulAnswer:
    return NoulAnswer(noul=p)


def choice(pick: str, probabilities: dict[str, float]) -> ChoiceAnswer:
    n = len(probabilities)
    conf = (n * max(probabilities.values()) - 1) / (n - 1) if n > 1 else 1.0
    return ChoiceAnswer(choice=pick, probabilities=probabilities, confidence=round(conf, 4))


def score(value: float, levels: list[str], confidence: float = 0.8) -> ScoreAnswer:
    legend = {str(i): level for i, level in enumerate(levels)}
    probs = {str(i): 0.0 for i in range(len(levels))}
    lo = max(0, min(len(levels) - 1, int(value)))
    hi = min(len(levels) - 1, lo + 1)
    probs[str(hi)] = round(value - lo, 4)
    probs[str(lo)] = round(1 - (value - lo), 4)
    return ScoreAnswer(score=value, legend=legend, probabilities=probs, confidence=confidence)


def neutral(question: Any) -> Any:
    kind = question.type
    if kind == "choice":
        options = list(question.criteria)
        share = round(1 / len(options), 4)
        return ChoiceAnswer(
            choice=options[0], probabilities={o: share for o in options}, confidence=0.0
        )
    if kind == "score":
        levels = list(question.criteria)
        return score((len(levels) - 1) / 2, levels, confidence=0.0)
    return noul(0.5)


class FakeDecider:
    """A ``DeciderEndpoint`` that answers from a script and records what it was asked.

    ``answers`` is a dict reused on every call, or a list consumed in order. ``responder`` is
    a callable ``(state, questions) -> answers`` for tests that branch on the state. Any
    question key with no scripted answer gets a neutral answer of the right type.
    """

    def __init__(
        self,
        answers: dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        responder: Any = None,
        error: Exception | None = None,
    ) -> None:
        self._answers = answers
        self._responder = responder
        self._error = error
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    @property
    def last_state(self) -> Any:
        return self.calls[-1][0]

    @property
    def last_questions(self) -> dict[str, Any]:
        return self.calls[-1][1]

    def ask(self, state: Any, questions: Any) -> SystemOneResponse:
        self.calls.append((state, dict(questions)))
        if self._error is not None:
            raise self._error
        if self._responder is not None:
            scripted = dict(self._responder(state, questions))
        elif isinstance(self._answers, list):
            scripted = dict(self._answers.pop(0)) if self._answers else {}
        else:
            scripted = dict(self._answers or {})
        for key, question in questions.items():
            scripted.setdefault(key, neutral(question))
        return SystemOneResponse(
            model="fake-decider",
            answers=scripted,
            usage=Usage(input_tokens=10, output_tokens=len(questions)),
        )


class Ctx:
    """The slice of ``ToolContext`` the tools read."""

    def __init__(self, decider: Any = None) -> None:
        self.invocation_state = {"decider": decider} if decider is not None else {}


def payload(result: dict[str, Any]) -> dict[str, Any]:
    assert result["status"] == "success", result
    return result["content"][0]["json"]


def summary(result: dict[str, Any]) -> str:
    return result["content"][-1]["text"]


def refused(result: dict[str, Any]) -> str:
    assert result["status"] == "error", result
    return result["content"][0]["text"]


@pytest.fixture(autouse=True)
def _no_default(monkeypatch):
    """No process-wide default and no environment: every test says which decider answers."""
    configure(None)
    monkeypatch.delenv("STRANDS_DECIDER_URL", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_CHECKPOINT", raising=False)
    yield
    configure(None)


# --------------------------------------------------------------------------- primitives


def test_ask_mixed_question_types_returns_every_answer_shape() -> None:
    fake = FakeDecider(
        {
            "urgent": noul(0.91),
            "team": choice("billing", {"billing": 0.8, "technical": 0.2}),
            "anger": score(1.5, ["calm", "frustrated", "angry"]),
        }
    )
    result = decider_ask(
        state="Help! My payouts have been failing for 3 days.",
        questions={
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "money", "technical": "bugs"},
            },
            "anger": {
                "type": "score",
                "instructions": "How angry?",
                "criteria": ["calm", "frustrated", "angry"],
            },
        },
        tool_context=Ctx(fake),
    )
    rows = payload(result)["answers"]
    assert rows["urgent"] == {"type": "noul", "noul": 0.91}
    assert rows["team"]["choice"] == "billing" and rows["team"]["confidence"] == 0.6
    assert rows["anger"]["legend"] == {0: "calm", 1: "frustrated", 2: "angry"}
    assert payload(result)["model"] == "fake-decider"
    assert summary(result).startswith("3 answers: urgent=0.91, team=billing (0.60), anger=1.50")
    # The fake was handed the package's own question models, in the caller's order.
    assert [q.type for q in fake.last_questions.values()] == ["noul", "choice", "score"]


def test_ask_tolerates_strings_lists_and_aliases() -> None:
    fake = FakeDecider()
    result = decider_ask(state="x", questions="Is it evidence?", tool_context=Ctx(fake))
    assert list(payload(result)["answers"]) == ["q1"]
    assert fake.last_questions["q1"].type == "noul"

    result = decider_ask(
        state="x",
        questions=json.dumps(
            {
                "k": {"type": "choice", "question": "Which?", "options": ["a", "b"]},
                "s": {"type": "score", "instructions": "How?", "levels": ["lo", "hi"]},
            }
        ),
        tool_context=Ctx(fake),
    )
    assert fake.last_questions["k"].criteria == {"a": "a", "b": "b"}
    assert fake.last_questions["s"].criteria == ["lo", "hi"]
    assert set(payload(result)["answers"]) == {"k", "s"}


def test_ask_refuses_bad_questions_before_any_forward_pass() -> None:
    fake = FakeDecider()
    assert "needs type noul, choice or score" in refused(
        decider_ask(
            state="x",
            questions={"q": {"type": "rank", "instructions": "?"}},
            tool_context=Ctx(fake),
        )
    )
    assert "at least 2 options" in refused(
        decider_ask(
            state="x",
            questions={"q": {"type": "choice", "instructions": "?", "criteria": ["only"]}},
            tool_context=Ctx(fake),
        )
    )
    assert "between 2 and 10 levels" in refused(
        decider_ask(
            state="x",
            questions={"q": {"type": "score", "instructions": "?", "criteria": ["one"]}},
            tool_context=Ctx(fake),
        )
    )
    assert "subset of {'true', 'false'}" in refused(
        decider_ask(
            state="x",
            questions={"q": {"type": "noul", "instructions": "?", "criteria": {"maybe": "x"}}},
            tool_context=Ctx(fake),
        )
    )
    assert "is empty" in refused(decider_ask(state="x", questions=[], tool_context=Ctx(fake)))
    assert fake.calls == []


def test_ask_folds_context_and_truncates_long_state() -> None:
    fake = FakeDecider()
    decider_ask(
        state={"ticket": "t"}, questions="q", context="the user asked", tool_context=Ctx(fake)
    )
    assert fake.last_state == {"context": "the user asked", "subject": {"ticket": "t"}}
    decider_ask(state="x" * 20_000, questions="q", tool_context=Ctx(fake))
    assert len(fake.last_state) <= _common.DEFAULT_MAX_STATE_CHARS
    assert fake.last_state.endswith("characters]")


def test_endpoint_failures_become_results_not_exceptions() -> None:
    down = FakeDecider(error=DeciderUnavailable("nothing on :8099"))
    text = refused(decider_check(state="x", question="q", tool_context=Ctx(down)))
    assert "not reachable" in text and "nothing on :8099" in text
    refusing = FakeDecider(error=ValueError("question has 300 options"))
    assert "refused the request" in refused(
        decider_ask(state="x", questions="q", tool_context=Ctx(refusing))
    )
    assert "nothing configured" not in refused(
        decider_ask(state="x", questions="q", tool_context=Ctx())
    )
    assert "STRANDS_DECIDER_URL" in refused(
        decider_ask(state="x", questions="q", tool_context=Ctx())
    )


def test_resolution_order_invocation_state_then_configure_then_env(monkeypatch) -> None:
    per_call = FakeDecider({"q1": noul(0.1)})
    configured = FakeDecider({"q1": noul(0.2)})
    configure(configured)
    assert (
        payload(decider_ask(state="x", questions="q", tool_context=Ctx(per_call)))["answers"]["q1"][
            "noul"
        ]
        == 0.1
    )
    assert (
        payload(decider_ask(state="x", questions="q", tool_context=Ctx()))["answers"]["q1"]["noul"]
        == 0.2
    )
    configure(None)
    monkeypatch.setenv("STRANDS_DECIDER_URL", "http://decider.test:1")
    assert isinstance(_common.resolve(None), HttpDecider)
    assert _common.resolve(None).url == "http://decider.test:1"
    with pytest.raises(TypeError, match="must be an HttpDecider"):
        _common.resolve(Ctx(object()))


def test_info_and_usage_read_the_client() -> None:
    assert "no health report" in summary(decider_info(tool_context=Ctx(FakeDecider())))
    assert "no usage tally" in refused(decider_usage(tool_context=Ctx(FakeDecider())))


# ------------------------------------------------------------------------- single state


def test_check_applies_the_threshold_and_passes_criteria() -> None:
    fake = FakeDecider({"q": noul(0.62)})
    result = decider_check(
        state="s",
        question="Urgent?",
        criteria={"true": "needs action today", "false": "can wait"},
        tool_context=Ctx(fake),
    )
    assert payload(result)["verdict"] == "yes" and payload(result)["probability"] == 0.62
    assert fake.last_questions["q"].criteria == {"true": "needs action today", "false": "can wait"}
    result = decider_check(state="s", question="Urgent?", threshold=0.7, tool_context=Ctx(fake))
    assert payload(result)["verdict"] == "no"
    assert summary(result) == "no (p=0.62, threshold 0.70)"
    assert "got a boolean" in refused(
        decider_check(state="s", question="q", threshold=True, tool_context=Ctx(fake))
    )
    assert "between 0 and 1" in refused(
        decider_check(state="s", question="q", threshold=2, tool_context=Ctx(fake))
    )


def test_choose_reports_decided_or_leaning_with_runner_up() -> None:
    fake = FakeDecider({"q": choice("billing", {"billing": 0.85, "retail": 0.09, "sales": 0.06})})
    result = decider_choose(
        state="payouts failing",
        question="Which team?",
        options={"billing": "money", "retail": "shop", "sales": "deals"},
        tool_context=Ctx(fake),
    )
    row = payload(result)
    assert (
        row["choice"] == "billing" and row["verdict"] == "decided" and row["runner_up"] == "retail"
    )
    assert list(row["probabilities"]) == ["billing", "retail", "sales"]
    result = decider_choose(
        state="s",
        question="Which team?",
        options=["billing", "retail", "sales"],
        confidence_floor=0.9,
        tool_context=Ctx(fake),
    )
    assert payload(result)["verdict"] == "leaning"
    assert "at least 2 options" in refused(
        decider_choose(state="s", question="q", options=["one"], tool_context=Ctx(fake))
    )
    assert "over the limit of 255" in refused(
        decider_choose(
            state="s", question="q", options=[str(i) for i in range(256)], tool_context=Ctx(fake)
        )
    )


def test_rate_names_the_nearest_level() -> None:
    fake = FakeDecider({"q": score(1.3, ["calm", "frustrated", "furious"])})
    result = decider_rate(
        state="s",
        question="How angry?",
        levels=["calm", "frustrated", "furious"],
        tool_context=Ctx(fake),
    )
    row = payload(result)
    assert row["score"] == 1.3 and row["level"] == 1 and row["label"] == "frustrated"
    assert row["legend"] == {0: "calm", 1: "frustrated", 2: "furious"}
    assert summary(result) == "1.30 of 2 (frustrated, confidence 0.80)"
    assert "needs between 2 and 10" in refused(
        decider_rate(state="s", question="q", levels=["one"], tool_context=Ctx(fake))
    )
    assert "needs between 2 and 10" in refused(
        decider_rate(
            state="s", question="q", levels=[str(i) for i in range(11)], tool_context=Ctx(fake)
        )
    )


# ---------------------------------------------------------------------------- batches


def _by_keyword(word: str, yes: float = 0.9, no: float = 0.1):
    def responder(state: Any, questions: Any) -> dict[str, Any]:
        text = json.dumps(state) if not isinstance(state, str) else state
        return {"q": noul(yes if word in text else no)}

    return responder


def test_sift_keeps_by_threshold_and_counts_in_code() -> None:
    fake = FakeDecider(responder=_by_keyword("ERROR"))
    lines = "INFO boot\nERROR disk full\nINFO ok\nERROR timeout"
    result = decider_sift(items=lines, question="Is this line an error?", tool_context=Ctx(fake))
    row = payload(result)
    assert row["count"] == 2 and row["total"] == 4
    assert [r["index"] for r in row["kept"]] == [1, 3]
    assert [r["index"] for r in row["dropped"]] == [0, 2]
    assert len(fake.calls) == 4 and fake.last_state == "ERROR timeout"
    assert summary(result) == "kept 2 of 4 (threshold 0.50)"
    result = decider_sift(items=lines, question="q", threshold=0.95, tool_context=Ctx(fake))
    assert payload(result)["count"] == 0


def test_sift_reports_a_failed_item_in_place() -> None:
    flaky = {"n": 0}

    def responder(state: Any, questions: Any) -> dict[str, Any]:
        flaky["n"] += 1
        if flaky["n"] == 2:
            raise ValueError("option span has no tokens left")
        return {"q": noul(0.9)}

    result = decider_sift(
        items=["a", "b", "c"], question="q", tool_context=Ctx(FakeDecider(responder=responder))
    )
    row = payload(result)
    assert row["count"] == 2 and row["failed"][0]["index"] == 1
    assert "no tokens left" in row["failed"][0]["error"]
    assert summary(result).endswith(", 1 failed)")


def test_classify_groups_and_parks_the_uncertain() -> None:
    def responder(state: Any, questions: Any) -> dict[str, Any]:
        if "refund" in state:
            return {"q": choice("billing", {"billing": 0.9, "technical": 0.1})}
        if "crash" in state:
            return {"q": choice("technical", {"billing": 0.05, "technical": 0.95})}
        return {"q": choice("billing", {"billing": 0.55, "technical": 0.45})}

    fake = FakeDecider(responder=responder)
    result = decider_classify(
        items=["I want a refund", "the app crashes", "hello there"],
        question="Which team?",
        options={"billing": "money", "technical": "bugs"},
        tool_context=Ctx(fake),
    )
    row = payload(result)
    assert row["counts"] == {"billing": 1, "technical": 1}
    assert row["groups"]["billing"][0]["index"] == 0
    assert row["uncertain"][0]["index"] == 2 and row["uncertain"][0]["choice"] == "billing"
    assert summary(result) == "3 items: billing=1, technical=1; 1 uncertain"


def test_rank_orders_by_probability_or_score_and_honours_top_k() -> None:
    fake = FakeDecider(responder=_by_keyword("python"))
    result = decider_rank(
        items=["java notes", "python guide", "python intro"],
        question="About python?",
        tool_context=Ctx(fake),
    )
    row = payload(result)
    assert row["scale"] == "probability"
    assert [r["index"] for r in row["ranked"]] == [1, 2, 0]
    assert row["ranked"][0]["rank"] == 1

    def by_length(state: Any, questions: Any) -> dict[str, Any]:
        return {"q": score(min(2.0, len(state) / 10), ["poor", "fair", "good"])}

    result = decider_rank(
        items=["x", "xxxxxxxxxx", "xxxxx"],
        question="Quality?",
        levels=["poor", "fair", "good"],
        top_k=2,
        tool_context=Ctx(FakeDecider(responder=by_length)),
    )
    row = payload(result)
    assert row["scale"] == "score" and [r["index"] for r in row["ranked"]] == [1, 2]
    assert "at least 1" in refused(
        decider_rank(items=["a"], question="q", top_k=0, tool_context=Ctx(fake))
    )


def test_batches_refuse_over_the_ceiling_without_calling() -> None:
    fake = FakeDecider()
    text = refused(decider_sift(items=["x"] * 501, question="q", tool_context=Ctx(fake)))
    assert "over the ceiling of 500" in text and fake.calls == []


# --------------------------------------------------------------------------- patterns


def test_route_sends_confident_simple_requests_to_the_handler() -> None:
    fake = FakeDecider(
        {
            "handler": choice("billing", {"billing": 0.85, "sales": 0.1, "retail": 0.05}),
            "complexity": score(0.4, ["a", "b", "c"]),
        }
    )
    result = decider_route(
        state="My payouts fail",
        handlers={"billing": "payments", "sales": "deals", "retail": "shop"},
        tool_context=Ctx(fake),
    )
    row = payload(result)
    assert row["route"] == "billing" and row["reason"] == "confident"
    assert row["complexity"]["level"] == 0
    assert set(fake.last_questions) == {"handler", "complexity"}
    assert len(fake.last_questions["complexity"].criteria) == 3


def test_route_falls_back_on_low_confidence_or_complexity() -> None:
    levels = ["a", "b", "c"]
    unsure = FakeDecider(
        {
            "handler": choice("billing", {"billing": 0.4, "sales": 0.35, "retail": 0.25}),
            "complexity": score(0.2, levels),
        }
    )
    row = payload(
        decider_route(state="?", handlers=["billing", "sales", "retail"], tool_context=Ctx(unsure))
    )
    assert (
        row["route"] == "human"
        and row["reason"] == "low_confidence"
        and row["handler"] == "billing"
    )

    hard = FakeDecider(
        {
            "handler": choice("billing", {"billing": 0.9, "sales": 0.05, "retail": 0.05}),
            "complexity": score(1.8, levels),
        }
    )
    row = payload(
        decider_route(
            state="?",
            handlers=["billing", "sales", "retail"],
            fallback="tier2",
            tool_context=Ctx(hard),
        )
    )
    assert (
        row["route"] == "tier2" and row["reason"] == "complex" and row["complexity"]["level"] == 2
    )

    custom = FakeDecider(
        {
            "handler": choice("billing", {"billing": 0.9, "sales": 0.1}),
            "complexity": score(1.0, ["easy", "medium", "hard", "impossible"]),
        }
    )
    row = payload(
        decider_route(
            state="?",
            handlers=["billing", "sales"],
            complexity_levels=["easy", "medium", "hard", "impossible"],
            tool_context=Ctx(custom),
        )
    )
    assert row["route"] == "billing", "with a custom rubric only the top level escalates"


def test_fan_out_keeps_answers_whose_premise_held() -> None:
    fake = FakeDecider(
        {
            "about_billing": noul(0.2),
            "plan": choice("pro", {"pro": 0.7, "free": 0.3}),
            "tone": choice("angry", {"angry": 0.9, "calm": 0.1}),
            "needs_apology": noul(0.8),
        }
    )
    result = decider_fan_out(
        state="the app keeps crashing and I am furious",
        questions={
            "about_billing": "Is this about billing?",
            "plan": {"type": "choice", "instructions": "Which plan?", "options": ["pro", "free"]},
            "tone": {"type": "choice", "instructions": "Tone?", "options": ["angry", "calm"]},
            "needs_apology": "Does this need an apology?",
        },
        premises={
            "plan": {"question": "about_billing", "min": 0.5},
            "needs_apology": {"question": "tone", "equals": "angry"},
        },
        tool_context=Ctx(fake),
    )
    row = payload(result)
    assert set(row["answers"]) == {"about_billing", "tone", "needs_apology"}
    assert "plan" in row["skipped"] and "wanted >= 0.50" in row["skipped"]["plan"]
    assert len(fake.calls) == 1, "one pass, every question"
    assert summary(result).endswith("; skipped plan")


def test_fan_out_refuses_a_premise_that_points_nowhere() -> None:
    fake = FakeDecider()
    assert "unknown question" in refused(
        decider_fan_out(
            state="s",
            questions={"a": "A?", "b": "B?"},
            premises={"b": {"question": "zzz", "min": 0.5}},
            tool_context=Ctx(fake),
        )
    )
    assert "cannot depend on itself" in refused(
        decider_fan_out(
            state="s",
            questions={"a": "A?"},
            premises={"a": {"question": "a", "min": 0.5}},
            tool_context=Ctx(fake),
        )
    )
    assert fake.calls == []


# ---------------------------------------------------------------------------- clients


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
    """Concentrated on slot 0, as tests/test_server.py's stub: exercises the client, not the model."""

    cfg = _StubCfg()
    model = _StubModel()

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        answers = {}
        for name, q in request.questions.items():
            rq = render_question(q)
            n = rq.n_slots
            answers[name] = _to_answer(rq, [0.7] + [0.3 / (n - 1)] * (n - 1))
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=42, output_tokens=len(request.questions)),
        )


def test_local_decider_wraps_an_engine_and_keeps_a_tally() -> None:
    local = LocalDecider(_StubEngine(), checkpoint="stub")  # type: ignore[arg-type]
    result = decider_choose(
        state="s", question="Which?", options=["a", "b", "c"], tool_context=Ctx(local)
    )
    assert payload(result)["choice"] in {"a", "b", "c"}
    info = payload(decider_info(tool_context=Ctx(local)))
    assert (
        info["checkpoint"] == "stub"
        and info["max_length"] == 512
        and info["endpoint"] == "LocalDecider('stub')"
    )
    usage = payload(decider_usage(reset=True, tool_context=Ctx(local)))
    assert usage["calls"] == 1 and usage["questions"] == 1 and usage["input_tokens"] == 42
    assert usage["latency_ms_mean"] is not None
    assert payload(decider_usage(tool_context=Ctx(local)))["calls"] == 0
    with pytest.raises(DeciderUnavailable, match="STRANDS_DECIDER_CHECKPOINT"):
        LocalDecider.load()


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_http_decider_speaks_the_wire_shape(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        body = {
            "model": "strands-decider-2B-hobson-v19",
            "answers": {"q": {"type": "noul", "noul": 0.83}},
            "usage": {"input_tokens": 17, "output_tokens": 1},
            "latency_ms": 123.4,
        }
        return _Reply(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    http = HttpDecider("http://decider.test:8099/")
    result = decider_check(state="payouts failing", question="Urgent?", tool_context=Ctx(http))
    assert (
        payload(result)["probability"] == 0.83
        and payload(result)["model"] == "strands-decider-2B-hobson-v19"
    )
    assert seen["url"] == "http://decider.test:8099/v1/systemone"
    assert seen["body"]["state"] == "payouts failing"
    assert seen["body"]["questions"]["q"] == {"type": "noul", "instructions": "Urgent?"}
    assert seen["body"]["model"] == "strands-decider-latest"
    assert http.usage.last_latency_ms == 123.4 and http.usage.input_tokens == 17


def test_http_decider_turns_connection_and_422_into_results(monkeypatch) -> None:
    def refuse(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    text = refused(
        decider_check(
            state="s", question="q", tool_context=Ctx(HttpDecider("http://decider.test:1"))
        )
    )
    assert "not reachable" in text and "strands-decider serve" in text

    def reject(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 422, "Unprocessable", {}, io.BytesIO(b'{"detail":"300 options"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)
    http = HttpDecider("http://decider.test:1")
    text = refused(decider_check(state="s", question="q", tool_context=Ctx(http)))
    assert "HTTP 422" in text and "300 options" in text and http.usage.errors == 1


# --------------------------------------------------------------------- through the SDK


def test_every_tool_has_a_spec_the_agent_can_read() -> None:
    names = [t.tool_name for t in ALL_TOOLS]
    assert len(names) == len(set(names)) == 11
    for t in ALL_TOOLS:
        spec = t.tool_spec
        assert spec["name"].startswith("decider_")
        assert len(spec["description"]) > 80
        props = spec["inputSchema"]["json"].get("properties", {})
        assert "tool_context" not in props, (
            "the injected context is not part of the model-facing schema"
        )


def test_direct_agent_call_injects_tool_context_and_reads_invocation_state() -> None:
    fake = FakeDecider({"q1": noul(0.9)})
    agent = Agent(
        model=BedrockModel(model_id="never-called", region_name="us-east-1"),
        tools=ALL_TOOLS,
        callback_handler=None,
    )
    result = agent.tool.decider_ask(state="evidence", questions=["is it evidence?"], decider=fake)
    assert len(fake.calls) == 1
    assert result["status"] == "success"
    assert result["toolUseId"].startswith("tooluse_decider_ask_")
    assert result["content"][0]["json"]["answers"]["q1"]["noul"] == 0.9
