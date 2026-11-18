"""The readout on small arrays: the bf16 regime, the goldens, the readout file, and the answers."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest
from core_conformance import agreement
from core_support import decision_config_fixture, readout_goldens

from strands_decider.core import (
    CODE_TOKEN_IDS,
    CODES,
    DecisionConfig,
    RenderedQuestion,
    RenderedRequest,
    attention_mask,
    load_readout,
    question_probabilities,
    read,
)
from strands_decider.core.readout import (
    answer,
    code_probabilities,
    load_safetensors,
    readout_logits,
    round_to_bf16,
)

C = 30.0  # Gemma 4's final_logit_softcapping


def _f32(x: float) -> float:
    """`x` rounded to float32 (to nearest, ties to even), as a Python float. A float32
    operation done in float64 and rounded so is the float32 operation exactly: float64 has
    more than twice float32's precision."""
    return float(struct.unpack("<f", struct.pack("<f", x))[0])


def _bf16(x: float) -> float:
    """`x` rounded to bf16 by arithmetic, not by bits: x = m 2^e with 0.5 <= |m| < 1, and bf16
    keeps 8 significant bits, so the result is round(m 2^8) 2^(e - 8), Python's round taking
    a tie to the even integer. (For normal numbers; the tests use no others.)"""
    if x == 0.0 or not math.isfinite(x):
        return x
    m, e = math.frexp(x)
    return math.ldexp(round(m * 256), e - 8)


def _after_the_gemm(z: float, c: float | None, t: float) -> float:
    """strands-decider-vllm's float32 steps on one bf16 logit: `tanh(z / c) * c`, then `z / T`."""
    if c is not None:
        z = _f32(_f32(math.tanh(_f32(z / _f32(c)))) * _f32(c))
    return _f32(z / _f32(t)) if t != 1.0 else z


def _softmax64(s) -> list[float]:
    m = max(s)
    e = [math.exp(x - m) for x in s]
    return [x / sum(e) for x in e]


def _manual(z, k, c, t) -> list[float]:
    """The readout from raw logits, one rounding at a time: bf16, the float32 cap and
    temperature, then the softmax (in float64 here; the core's float32 softmax is within
    1e-6 of it)."""
    return _softmax64([_after_the_gemm(_bf16(_f32(x)), c, t) for x in z[:k]])


# ---- bf16 and the precision regime -------------------------------------------------------


def test_round_to_bf16_rounds_to_nearest_ties_to_even():
    step = 2.0**-7  # bf16's spacing in [1, 2)
    cases = {
        1.0: 1.0,
        1 + step / 2: 1.0,  # a tie: to the even neighbour, 1.0
        1 + 3 * step / 2: 1 + 2 * step,  # a tie: to the even neighbour, 1 + 2 steps
        1 + step / 2 + 2**-23: 1 + step,  # just past the tie
        1 + step / 2 - 2**-23: 1.0,  # just short of it
        -(1 + 3 * step / 2): -(1 + 2 * step),
        16.0625: 16.0,  # between 16 and 32 the spacing is 0.125: a tie, to 16.0
        16.1875: 16.25,  # a tie, to 16.25
        24.06: 24.0,
        31.95: 32.0,
        3.3895313892515355e38: 3.3895313892515355e38,  # bf16's largest finite
        3.39e38: 3.3895313892515355e38,
        3.4e38: math.inf,  # past the midpoint to the next step: inf, as torch and CUDA give
        -3.4e38: -math.inf,
        0.0: 0.0,
        math.inf: math.inf,
        -math.inf: -math.inf,
    }
    got = round_to_bf16(np.array(list(cases), np.float32))
    assert got.dtype == np.float32
    assert got.tolist() == list(cases.values())
    assert math.copysign(1.0, float(round_to_bf16(-0.0))) == -1.0
    assert np.isnan(round_to_bf16(np.float32("nan")))
    assert round_to_bf16(np.float32(1 + step / 2)).shape == ()


def test_round_to_bf16_agrees_with_rounding_by_arithmetic():
    rng = np.random.default_rng(7)
    x = np.concatenate(
        [
            rng.normal(0, 1, 20000),
            rng.normal(0, 30, 20000),
            np.exp(rng.uniform(-80, 80, 20000)) * rng.choice([-1, 1], 20000),
        ]
    ).astype(np.float32)
    got = round_to_bf16(x)
    assert got.tolist() == [_bf16(v) for v in x.tolist()]
    assert np.array_equal(round_to_bf16(got), got)  # bf16 values are left as they are
    bits = got.view(np.uint32)
    assert np.all(bits & 0xFFFF == 0)


def test_code_probabilities_match_the_formula():
    rng = np.random.default_rng(1)
    for t in (1.0, 0.5, 2.7):
        for k in (1, 2, 7):
            z = rng.normal(0, 8, size=255)
            p = code_probabilities(round_to_bf16(z)[:k], C, t)
            assert p.dtype == np.float32 and np.allclose(p, _manual(z.tolist(), k, C, t), atol=1e-6)


def test_the_readout_goldens():
    g, w = readout_goldens()
    c = g["final_logit_softcapping"]
    for case in g["cases"]:
        h, t = np.array(case["hidden"], np.float32), case["temperature"]
        assert readout_logits(w, h).tolist() == case["z"]
        for k, want in case["probabilities"].items():
            got = code_probabilities(np.array(case["z"][: int(k)]), c, t)
            assert np.allclose(got, want, rtol=0, atol=1e-7), (case["name"], k)


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
    w = (np.arange(12, dtype=np.float32).reshape(3, 4) - 5.5) * 0.25  # exact in bf16
    _write_safetensors(tmp_path / "t.safetensors", {"weight": (dtype, w)})
    got = load_safetensors(tmp_path / "t.safetensors")
    assert got["weight"].dtype == np.float32 and np.array_equal(got["weight"], w)


def test_load_readout(tmp_path):
    _write_safetensors(
        tmp_path / "readout.safetensors", {"weight": ("BF16", np.ones((255, 8), np.float32))}
    )
    assert load_readout(tmp_path).shape == (255, 8)
    _write_safetensors(
        tmp_path / "readout.safetensors", {"weight": ("BF16", np.ones((254, 8), np.float32))}
    )
    with pytest.raises(ValueError, match="255"):
        load_readout(tmp_path)


def _rq(kind, options, *, order=None, descriptions=None, name="q") -> RenderedQuestion:
    order = list(range(len(options))) if order is None else order
    k = len(options)
    return RenderedQuestion(
        name=name,
        kind=kind,
        state_ids=(2, 10),
        question_ids=(11, 12, 13),
        codes=CODES[:k],
        code_token_ids=CODE_TOKEN_IDS[:k],
        options=tuple(options[i] for i in order),
        request_options=tuple(options),
        descriptions=tuple((descriptions or [""] * k)[i] for i in order),
        state_text="",
        question_text="",
    )


def test_answers():
    assert answer(_rq("noul", ["false", "true"], order=[1, 0]), [0.8, 0.2]) == {
        "type": "noul",
        "noul": 0.8,
    }
    a = answer(
        _rq("choice", ["billing", "technical", "sales"], order=[2, 0, 1]), [0.018, 0.234, 0.748]
    )
    assert a == {
        "type": "choice",
        "choice": "technical",
        "probabilities": {"billing": 0.234, "technical": 0.748, "sales": 0.018},
        "confidence": round((3 * 0.748 - 1) / 2, 4),
    }
    s = answer(
        _rq("score", ["0", "1", "2"], descriptions=["Calm", "Frustrated", "Very angry"]),
        [0.093, 0.578, 0.329],
    )
    assert s["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert s["score"] == pytest.approx(0.578 + 2 * 0.329, abs=1e-4)


def test_hidden_and_probabilities_give_the_same_answers():
    rng = np.random.default_rng(1)
    w = round_to_bf16(rng.normal(0, 0.5, size=(255, 16)))
    h = round_to_bf16(rng.normal(0, 1, size=(2, 16)))
    rendered = RenderedRequest(
        model="m",
        questions={
            q.name: q
            for q in (
                _rq("noul", ["false", "true"], name="n"),
                _rq("choice", list("xyz"), name="c"),
            )
        },
    )
    cfg = DecisionConfig.model_validate({**decision_config_fixture(), "temperature": 1.3})
    by_hidden = question_probabilities(h, cfg, rendered, weight=w)
    z = readout_logits(w, h)
    rows = [code_probabilities(z[0], C, 1.3), code_probabilities(z[1], C, 1.3)]
    by_p = question_probabilities(rows, cfg, rendered)
    for n in by_hidden:
        assert np.allclose(by_p[n], by_hidden[n], atol=1e-6), n
    assert read(h, cfg, rendered, weight=w)["c"]["type"] == "choice"
    with pytest.raises(ValueError, match="readout weight"):
        read(np.ones((2, 300)), cfg, rendered)  # hidden states with no weight
    with pytest.raises(ValueError, match="one per question"):
        read(h[:1], cfg, rendered, weight=w)


def test_attention_mask():
    m = attention_mask(3, 5)
    assert m[:3, :3].all() and not m[:3, 3:].any() and m[3:].all()


def test_agreement():
    a = agreement({"a": [0.9, 0.1], "b": [0.4, 0.6]}, {"a": [0.8, 0.2], "b": [0.7, 0.3]})
    assert a.questions == 2 and a.top1 == 0.5 and a.mean_abs_dp == pytest.approx(0.2)
    assert not a.conforms()


class _Backend:
    """Returns a fixed row per question, and records what it was given."""

    def __init__(self, row):
        self.row, self.calls = row, []

    def forward(self, state, questions, images=()):
        self.calls.append((state, questions, images))
        return [self.row] * len(questions)


def test_evaluate_end_to_end():
    from core_support import gemma4_tokenizer

    from strands_decider.core import Request, evaluate

    tok, cfg = gemma4_tokenizer(), DecisionConfig.model_validate(decision_config_fixture())
    request = Request.model_validate_json(
        '{"state": "I was charged twice.", "questions": {'
        '"refund": {"type": "noul", "instructions": "Is a refund requested?"}, '
        '"team": {"type": "choice", "instructions": "Which team?", '
        '"criteria": {"billing": null, "tech": null, "sales": null}}}}'
    )
    rng = np.random.default_rng(0)
    w, h = round_to_bf16(rng.normal(0, 0.5, size=(255, 8))), round_to_bf16(rng.normal(size=8))

    backend = _Backend(h)
    out = evaluate(request, tok, w, cfg, backend)
    assert set(out["answers"]) == {"refund", "team"} and out["answers"]["team"]["type"] == "choice"
    state, questions, images = backend.calls[0]
    assert len(questions) == 2 and images == ()
    assert out["usage"]["input_tokens"] == len(state) + sum(map(len, questions))

    # A backend that applies the readout itself returns probabilities, and gives the same answers.
    p = code_probabilities(readout_logits(w, h[None])[0], C, cfg.temperature)
    assert evaluate(request, tok, None, cfg, _Backend(p))["answers"] == out["answers"]

    with pytest.raises(ValueError, match="exceeds the limit"):
        evaluate(request, tok, w, cfg, _Backend(h), max_length=10)
    text_only = cfg.model_copy(update={"inputs": ("text",)})
    with_image = request.model_copy(update={"images": ["aGVsbG8="]})
    with pytest.raises(ValueError, match="does not take images"):
        evaluate(with_image, tok, w, text_only, _Backend(h))
