"""The v2 readout (docs/v2-spec.md, The readout) on small arrays, and the runtime glue.

Torch-free: numpy and tokenizers. Nothing here needs Gemma 4's tokenizer; the end-to-end
test uses a toy one.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest
from v2_support import decision_config_fixture, toy_tokenizer

from strands_decider.v2 import (
    CODES,
    PLAIN,
    DecisionConfig,
    Readout,
    RenderedQuestion,
    RenderedRequest,
    answer,
    attention_mask,
    code_probabilities,
    evaluate,
    load_safetensors,
    read,
    render,
    softcap,
)

C = 30.0  # Gemma 4's final_logit_softcapping


def test_softcap():
    z = np.array([-1e6, -90.0, -1.0, 0.0, 1e-3, 1.0, 30.0, 90.0, 1e6])
    out = softcap(z, C)
    assert np.allclose(out, C * np.tanh(z / C))
    assert np.all(np.abs(out) < C + 1e-12) and out[3] == 0.0
    assert np.all(np.diff(out) > 0) or np.all(np.diff(out) >= 0)  # monotone
    assert out[-1] == pytest.approx(C) and out[0] == pytest.approx(-C)
    assert out[4] == pytest.approx(1e-3, rel=1e-6)  # near 0 the cap is the identity
    assert np.array_equal(softcap(z, None), z) and np.array_equal(softcap(z, 0.0), z)


def _manual(z, k, c, t):
    capped = [c * math.tanh(x / c) if c else x for x in z[:k]]
    e = [math.exp(x / t) for x in capped]
    return [x / sum(e) for x in e]


@pytest.mark.parametrize("c", [None, C, 2.0])
@pytest.mark.parametrize("t", [1.0, 0.5, 2.7])
@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_code_probabilities_match_the_formula(c, t, k):
    rng = np.random.default_rng(k)
    z = rng.normal(0, 8, size=255)
    p = code_probabilities(z, k, softcap_c=c, temperature=t)
    assert p.shape == (k,) and p.sum() == pytest.approx(1.0)
    assert np.allclose(p, _manual(z.tolist(), k, c, t))


def test_only_the_questions_own_codes_enter_the_softmax():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=255)
    p = code_probabilities(z, 3, softcap_c=C, temperature=1.0)
    z2 = z.copy()
    z2[3:] = 1e4  # any value on the other codes
    assert np.array_equal(code_probabilities(z2, 3, softcap_c=C, temperature=1.0), p)


def test_temperature_and_cap_order():
    z = np.array([40.0, 0.0])
    # Cap first (40 -> 30 tanh(4/3)), then divide by T: not the other way round.
    want = _manual([40.0, 0.0], 2, C, 2.0)
    assert np.allclose(code_probabilities(z, 2, softcap_c=C, temperature=2.0), want)
    assert not np.allclose(want, _manual([20.0, 0.0], 2, C, 1.0))
    hot = code_probabilities(z, 2, softcap_c=C, temperature=10.0)
    cold = code_probabilities(z, 2, softcap_c=C, temperature=0.1)
    assert hot[0] < code_probabilities(z, 2, softcap_c=C, temperature=1.0)[0] < cold[0]


def test_large_logits_are_stable():
    z = np.array([1e300, -1e300, 0.0])
    for c in (None, C):
        p = code_probabilities(z, 3, softcap_c=c, temperature=1e-3)
        assert np.all(np.isfinite(p)) and p.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        code_probabilities(np.zeros(2), 3, softcap_c=C, temperature=1.0)


# ---- safetensors and the readout file ---------------------------------------------------


def _write_safetensors(path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, blobs, offset = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        if dtype == "BF16":
            raw = (np.asarray(arr, np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raw = np.asarray(arr, {"F32": "<f4", "F16": "<f2"}[dtype]).tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps({**header, "__metadata__": {"format": "pt"}}).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_load_safetensors(tmp_path, dtype):
    # Values exact in bf16 (8 bits of mantissa), so every dtype reads them back exactly.
    w = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.5) * 0.25
    _write_safetensors(tmp_path / "t.safetensors", {"weight": (dtype, w)})
    got = load_safetensors(tmp_path / "t.safetensors")
    assert set(got) == {"weight"} and got["weight"].dtype == np.float32
    assert np.array_equal(got["weight"], w)


def test_load_safetensors_refuses_other_dtypes(tmp_path):
    p = tmp_path / "t.safetensors"
    h = json.dumps({"w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}).encode()
    p.write_bytes(struct.pack("<Q", len(h)) + h + b"\x00\x01")
    with pytest.raises(ValueError, match="dtype I8"):
        load_safetensors(p)


@pytest.mark.parametrize(
    "model_config",
    [{"final_logit_softcapping": 30.0}, {"text_config": {"final_logit_softcapping": 30.0}}],
    ids=["text-model", "multimodal-wrapper"],
)
def test_readout_load(tmp_path, model_config):
    w = np.ones((255, 8), np.float32)
    _write_safetensors(tmp_path / "readout.safetensors", {"weight": ("BF16", w)})
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    r = Readout.load(tmp_path)
    assert r.softcap == 30.0 and r.weight is not None and r.weight.shape == (255, 8)


def test_readout_load_refuses_a_config_without_the_cap(tmp_path):
    _write_safetensors(tmp_path / "readout.safetensors", {"weight": ("F32", np.ones((255, 8)))})
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        Readout.load(tmp_path)
    _write_safetensors(tmp_path / "readout.safetensors", {"w": ("F32", np.ones((255, 8)))})
    with pytest.raises(ValueError, match="one tensor"):
        Readout.load(tmp_path)


# ---- answers in v1's schema --------------------------------------------------------------


def _rq(kind, options, *, order=None, descriptions=None, name="q") -> RenderedQuestion:
    order = list(range(len(options))) if order is None else order
    k = len(options)
    return RenderedQuestion(
        name=name,
        kind=kind,
        variant=PLAIN,
        state_ids=(2, 10),
        question_ids=(11, 12, 13),
        codes=CODES[:k],
        code_token_ids=tuple(decision_config_fixture()["token_ids"][:k]),
        options=tuple(options[i] for i in order),
        request_options=tuple(options),
        descriptions=tuple((descriptions or [""] * k)[i] for i in order),
        state_text="",
        question_text="",
    )


def test_noul_answer_reads_true_wherever_it_was_rendered():
    assert answer(_rq("noul", ["false", "true"]), [0.2, 0.8]) == {"type": "noul", "noul": 0.8}
    flipped = _rq("noul", ["false", "true"], order=[1, 0])  # B is false
    assert answer(flipped, [0.8, 0.2]) == {"type": "noul", "noul": 0.8}


def test_choice_answer():
    a = answer(_rq("choice", ["billing", "technical", "sales"]), [0.234, 0.748, 0.018])
    assert a == {
        "type": "choice",
        "choice": "technical",
        "probabilities": {"billing": 0.234, "technical": 0.748, "sales": 0.018},
        "confidence": round((3 * 0.748 - 1) / 2, 4),
    }
    # Rendered in another order, reported in the request's.
    permuted = _rq("choice", ["billing", "technical", "sales"], order=[2, 0, 1])
    b = answer(permuted, [0.018, 0.234, 0.748])
    assert b == a and list(b["probabilities"]) == ["billing", "technical", "sales"]


def test_score_answer():
    rq = _rq("score", ["0", "1", "2"], descriptions=["Calm", "Frustrated", "Very angry"])
    a = answer(rq, [0.093, 0.578, 0.329])
    assert a["type"] == "score"
    assert a["score"] == pytest.approx(0.578 + 2 * 0.329, abs=1e-4)
    assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert a["probabilities"] == {"0": 0.093, "1": 0.578, "2": 0.329}
    reversed_ = _rq(
        "score", ["0", "1", "2"], order=[2, 1, 0], descriptions=["Calm", "Frustrated", "Very angry"]
    )
    assert answer(reversed_, [0.329, 0.578, 0.093]) == a


def test_answers_match_v1():
    """The same numbers as v1's `prompting.to_answer`, the confidence formulas included."""
    pytest.importorskip("pydantic")
    from strands_decider.prompting import render_question, to_answer
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

    rng = np.random.default_rng(3)
    cases = [
        (NoulQuestion(instructions="x"), ["false", "true"], "noul"),
        (ChoiceQuestion(instructions="x", criteria={"a": "1", "b": None, "c": "3", "d": None}), ["a", "b", "c", "d"], "choice"),
        (ScoreQuestion(instructions="x", criteria=["lo", "mid", "hi", "top"]), ["0", "1", "2", "3"], "score"),
    ]  # fmt: skip
    for q, options, kind in cases:
        for _ in range(20):
            p = rng.dirichlet(np.ones(len(options))).tolist()
            v1 = to_answer(render_question(q), p, ordinal_smoothing=0.1).model_dump()
            desc = list(q.criteria) if kind == "score" else None
            v2 = answer(_rq(kind, options, descriptions=desc), p, ordinal_smoothing=0.1)
            assert v2 == v1


# ---- read ------------------------------------------------------------------------------


def _config(**kw) -> DecisionConfig:
    return DecisionConfig.from_dict({**decision_config_fixture(), **kw})


def _request(*rqs: RenderedQuestion) -> RenderedRequest:
    return RenderedRequest(variant=PLAIN, model="m", questions={q.name: q for q in rqs})


def test_read_hidden_equals_read_logits():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.1, size=(255, 16)).astype(np.float32)
    h = rng.normal(0, 1, size=(2, 16))
    rendered = _request(
        _rq("noul", ["false", "true"], name="n"), _rq("choice", ["x", "y", "z"], name="c")
    )
    cfg = _config(temperature=1.3)
    by_hidden = read(h, Readout(w, C), cfg, rendered)
    by_logits = read(h @ w.T.astype(np.float64), Readout(None, C), cfg, rendered, inputs="logits")
    by_name = read({"c": h[1], "n": h[0]}, Readout(w, C), cfg, rendered)
    assert by_hidden == by_logits == by_name
    z = w.astype(np.float64) @ h[1]
    want = _manual(z.tolist(), 3, C, 1.3)
    assert by_hidden["c"]["probabilities"] == {
        o: round(p, 4) for o, p in zip("xyz", want, strict=True)
    }


def test_read_refuses_a_mismatch():
    w = np.zeros((255, 4), np.float32)
    rendered = _request(_rq("noul", ["false", "true"]))
    with pytest.raises(ValueError, match="rendered for"):
        read(np.zeros((1, 4)), Readout(w, C), _config(prompt="strands-v2-chat"), rendered)
    bad_ids = list(decision_config_fixture()["token_ids"])
    bad_ids[0], bad_ids[1] = bad_ids[1], bad_ids[0]
    with pytest.raises(ValueError, match="token_ids"):
        read(np.zeros((1, 4)), Readout(w, C), _config(token_ids=bad_ids), rendered)
    with pytest.raises(ValueError, match="one per question"):
        read(np.zeros((2, 4)), Readout(w, C), _config(), rendered)
    with pytest.raises(ValueError, match="rows"):
        read(np.zeros((1, 4)), Readout(np.zeros((254, 4)), C), _config(), rendered)
    with pytest.raises(ValueError, match="readout weight"):
        read(np.zeros((1, 4)), Readout(None, C), _config(), rendered)
    with pytest.raises(ValueError, match="inputs"):
        read(np.zeros((1, 4)), Readout(w, C), _config(), rendered, inputs="probs")


# ---- the runtime -------------------------------------------------------------------------


def test_attention_mask():
    m = attention_mask(3, 5)
    want = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(m, want)
    with pytest.raises(ValueError):
        attention_mask(0, 5)


class _FakeBackend:
    """Returns, for each question, a hidden state that points at one code; with the
    identity as readout, row i of `W h` is large on that code's index only."""

    def __init__(self, picks: dict[tuple[int, ...], int]):
        self.picks, self.calls = picks, []

    def forward(self, state, questions):
        self.calls.append((state, questions))
        rows = np.zeros((len(questions), 255))
        for i, q in enumerate(questions):
            rows[i, self.picks[tuple(q)]] = 50.0
        return rows


def test_evaluate_end_to_end_with_a_toy_tokenizer():
    tok = toy_tokenizer()
    cfg = _config(token_ids=[tok.token_to_id(c) for c in CODES])
    req = {
        "state": "a short state",
        "questions": {
            "n": {"type": "noul", "instructions": "is it short"},
            "c": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"x": None, "y": None, "z": None},
            },
        },
    }
    rendered = render(req, tok, PLAIN)
    ids = {n: q.question_ids for n, q in rendered.questions.items()}
    backend = _FakeBackend({ids["n"]: 1, ids["c"]: 2})  # n -> B (true), c -> C (z)
    resp = evaluate(
        req, tok, Readout(np.eye(255, dtype=np.float32), C), cfg, backend, model_name="t"
    )
    ((state, questions),) = backend.calls
    assert state == list(rendered.state_ids)
    assert questions == [list(q.question_ids) for q in rendered.questions.values()]
    assert resp["model"] == "t" and list(resp["answers"]) == ["n", "c"]
    assert resp["answers"]["n"]["noul"] > 0.99
    assert resp["answers"]["c"]["choice"] == "z"
    assert resp["usage"] == {"input_tokens": rendered.input_tokens, "output_tokens": 2}
    with pytest.raises(ValueError, match="exceeds the limit"):
        evaluate(
            req,
            tok,
            Readout(np.eye(255), C),
            _config(token_ids=list(cfg.token_ids), max_length_validated=5),
            backend,
        )
