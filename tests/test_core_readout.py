"""The readout (docs/spec.md, section 7) on small arrays, and the runtime glue.

Torch-free: numpy, tokenizers and pydantic, and Pillow for the image requests. Nothing here
needs Gemma 4's tokenizer; the end-to-end test uses a toy one. The precision regime (a bf16
GEMM, then float32) is checked against an independent reimplementation of the arithmetic of
strands-decider-vllm, the vLLM runtime, which rounds by arithmetic where the core rounds by
bits.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest
from core_support import (
    decision_config_fixture,
    image_b64,
    image_bytes,
    readout_goldens,
    toy_tokenizer,
)

from strands_decider.core import (
    CAPPED_LOGITS,
    CODES,
    FINAL_LOGITS,
    HIDDEN,
    NO_VISION,
    PLAIN,
    PROBABILITIES,
    RAW_LOGITS,
    STAGES,
    DecisionConfig,
    Readout,
    RenderedQuestion,
    RenderedRequest,
    answer,
    attention_mask,
    code_probabilities,
    evaluate,
    load_safetensors,
    question_probabilities,
    read,
    readout_logits,
    render,
    round_to_bf16,
    softcap,
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
    """Section 7 from raw logits, one rounding at a time: bf16, the float32 cap and
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


def test_softcap():
    z = np.array([-1e6, -90.0, -1.0, 0.0, 1e-3, 1.0, 30.0, 90.0, 1e6])
    out = softcap(z, C)
    assert out.dtype == np.float32
    want = [_after_the_gemm(x, C, 1.0) for x in z.astype(np.float32).tolist()]
    assert np.allclose(out, want, rtol=0, atol=8e-6)
    assert np.all(np.diff(out) > 0)  # strictly increasing over these points
    assert out[0] == -C and out[-1] == C  # float32 saturates exactly at the cap
    assert np.all(np.abs(softcap(np.linspace(-1e4, 1e4, 20001), C)) <= C)
    assert out[3] == 0.0 and out[4] == pytest.approx(1e-3, rel=1e-6)  # near 0, the identity
    mid = softcap(np.linspace(-60, 60, 12001), C)
    assert np.all(np.diff(mid) > 0)  # strictly increasing where float32 resolves it
    assert np.array_equal(softcap(z, None), z.astype(np.float32))
    for bad in (0.0, -30.0, math.inf, math.nan, 1e300):
        with pytest.raises(ValueError, match="soft-cap"):
            softcap(z, bad)


@pytest.mark.parametrize("c", [None, C, 2.0])
@pytest.mark.parametrize("t", [1.0, 0.5, 2.7])
@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_code_probabilities_match_the_formula(c, t, k):
    rng = np.random.default_rng(k)
    z = rng.normal(0, 8, size=255)
    p = code_probabilities(z, k, stage=RAW_LOGITS, softcap_c=c, temperature=t)
    assert p.shape == (k,) and p.dtype == np.float32 and p.sum() == pytest.approx(1.0)
    assert np.allclose(p, _manual(z.tolist(), k, c, t), rtol=0, atol=1e-6)


def test_only_the_questions_own_codes_enter_the_softmax():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=255)
    for c in (None, C):
        p = code_probabilities(z, 3, stage=RAW_LOGITS, softcap_c=c, temperature=1.0)
        z2 = z.copy()
        z2[3:] = 1e4  # any value on the other codes
        p2 = code_probabilities(z2, 3, stage=RAW_LOGITS, softcap_c=c, temperature=1.0)
        assert np.array_equal(p2, p)
        assert np.array_equal(
            code_probabilities(z[:3], 3, stage=RAW_LOGITS, softcap_c=c, temperature=1.0), p
        )


def test_temperature_and_cap_order():
    z = np.array([40.0, 30.0])
    # Cap first (40 -> 30 tanh(4/3), 30 -> 30 tanh(1)), then divide by T; dividing first
    # (20 and 15, then the cap) would give 0.97 for the first code, not 0.84.
    want = _manual([40.0, 30.0], 2, C, 2.0)
    got = code_probabilities(z, 2, stage=RAW_LOGITS, softcap_c=C, temperature=2.0)
    assert np.allclose(got, want, rtol=0, atol=1e-6)
    assert got[0] == pytest.approx(0.8358, abs=1e-4)
    assert _manual([20.0, 15.0], 2, C, 1.0)[0] == pytest.approx(0.9739, abs=1e-4)

    def p0(t: float) -> float:
        return float(code_probabilities(z, 2, stage=RAW_LOGITS, softcap_c=C, temperature=t)[0])

    assert p0(10.0) < p0(1.0) < p0(0.1)


def test_large_logits_and_overflow():
    # Finite in float32 and stable: the softmax subtracts the maximum, and a gap past
    # float32's range gives exactly 0, never NaN.
    big = [3e38, -3e38, 0.0]
    p = code_probabilities(big, 3, stage=RAW_LOGITS, softcap_c=None, temperature=1)
    assert p.tolist() == [1.0, 0.0, 0.0]
    p = code_probabilities(big, 3, stage=RAW_LOGITS, softcap_c=C, temperature=1)
    assert np.allclose(p, _softmax64([C, -C, 0.0]), rtol=1e-6, atol=0)  # capped to +-30
    p = code_probabilities([1e30, -1e30, 0.0], 3, stage=RAW_LOGITS, softcap_c=C, temperature=1e-3)
    assert p.tolist() == [1.0, 0.0, 0.0]  # capped to +-30, then 3e4 apart: one-hot
    # Without a cap nothing bounds the logits. Past float32's range they are refused...
    with pytest.raises(ValueError, match="finite in float32"):
        code_probabilities([1e300, 0.0], 2, stage=RAW_LOGITS, softcap_c=None, temperature=1.0)
    for bad in (math.inf, -math.inf, math.nan):
        for c in (None, C):
            with pytest.raises(ValueError, match="finite in float32"):
                code_probabilities([bad, 0.0], 2, stage=RAW_LOGITS, softcap_c=c, temperature=1)
    # ...and so is a temperature that divides them past it.
    with pytest.raises(ValueError, match="overflow"):
        code_probabilities([1e38, 0.0], 2, stage=RAW_LOGITS, softcap_c=None, temperature=1e-3)
    # With the cap, only c / T past float32's range overflows.
    with pytest.raises(ValueError, match="overflow"):
        code_probabilities([40.0, 0.0], 2, stage=RAW_LOGITS, softcap_c=C, temperature=1.2e-38)
    with pytest.raises(ValueError, match="need a vector"):
        code_probabilities(np.zeros(2), 3, stage=RAW_LOGITS, softcap_c=C, temperature=1.0)


@pytest.mark.parametrize("t", [0, -1.0, math.nan, math.inf, 1e39, 1e300, 1e-39, True, "1"])
def test_bad_temperatures_raise(t):
    with pytest.raises(ValueError, match="temperature"):
        code_probabilities(np.zeros(3), 2, stage=RAW_LOGITS, softcap_c=C, temperature=t)


def test_code_probabilities_needs_a_named_stage():
    with pytest.raises(TypeError, match="stage"):
        code_probabilities(np.zeros(3), 2, softcap_c=C, temperature=1.0)  # type: ignore[call-arg]
    for bad in (None, HIDDEN, "logits", "probs"):
        with pytest.raises(ValueError, match="stage must be"):
            code_probabilities(np.zeros(3), 2, stage=bad, softcap_c=C, temperature=1.0)


@pytest.mark.parametrize("stage", [RAW_LOGITS, CAPPED_LOGITS, FINAL_LOGITS])
@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -30.0, "30", True])
def test_the_soft_cap_is_checked_for_every_logit_stage(stage, bad):
    """A NaN cap would silently disable the range check of capped and final logits."""
    with pytest.raises(ValueError, match="soft-cap"):
        code_probabilities([1.0, 0.0], 2, stage=stage, softcap_c=bad, temperature=1.0)
    assert code_probabilities(
        [1.0, 0.0], 2, stage=stage, softcap_c=np.float32(C), temperature=1.0
    ).shape == (2,)


def test_stages_past_the_cap_say_so():
    z = np.array([25.0, -3.0, 0.0])
    capped = softcap(z, C)
    final = capped / np.float32(0.5)
    p = code_probabilities(z, 3, stage=RAW_LOGITS, softcap_c=C, temperature=0.5)
    for stage, row in ((CAPPED_LOGITS, capped), (FINAL_LOGITS, final)):
        got = code_probabilities(row, 3, stage=stage, softcap_c=C, temperature=0.5)
        assert np.array_equal(got, p), stage
    # Raw logits handed over as capped or final, and capped logits handed over as final at
    # T > 1, can exceed what the stage allows: refused rather than capped, or divided, a
    # second time. Capped logits handed over as final at T <= 1 lie within the final range
    # (c / T >= c), so no check of the values can see them; only the stage's name can.
    with pytest.raises(ValueError, match="were they capped"):
        code_probabilities([31.0, 0.0], 2, stage=CAPPED_LOGITS, softcap_c=C, temperature=1)
    with pytest.raises(ValueError, match="were they capped"):
        code_probabilities([61.0, 0.0], 2, stage=FINAL_LOGITS, softcap_c=C, temperature=0.5)
    with pytest.raises(ValueError, match="were they capped"):
        code_probabilities([29.0, 0.0], 2, stage=FINAL_LOGITS, softcap_c=C, temperature=2.0)
    assert code_probabilities(
        [29.0, 0.0], 2, stage=FINAL_LOGITS, softcap_c=C, temperature=0.5
    ).shape == (2,)
    # Probabilities over all 255 codes renormalise to the question's k; over k, they stay.
    p255 = code_probabilities(
        np.arange(255.0) / 50, 255, stage=RAW_LOGITS, softcap_c=C, temperature=1
    )
    p3 = code_probabilities(p255, 3, stage=PROBABILITIES, softcap_c=C, temperature=1.0)
    want = code_probabilities(np.arange(3.0) / 50, 3, stage=RAW_LOGITS, softcap_c=C, temperature=1)
    assert np.allclose(p3, want, rtol=0, atol=1e-6)
    for bad in ([-0.1, 0.5], [0.0, 0.0], [math.nan, 0.5], [1.5, 0.1]):
        with pytest.raises(ValueError, match=r"probabilities|finite"):
            code_probabilities(bad, 2, stage=PROBABILITIES, softcap_c=C, temperature=1.0)


def _vllm_logits(w: list[list[float]], h: list[float], c: float, t: float) -> list[float]:
    """An independent reimplementation of strands-decider-vllm's logits at the codes' ids,
    one IEEE operation at a time:

        codes = self.readout(hidden_states.to(bf16)).float()   # a bf16 GEMM
        codes = torch.tanh(codes / self.softcap) * self.softcap  # float32
        codes = codes / self.temperature                         # float32, when T != 1

    The hidden state is rounded to bf16 first. The GEMM's products of bf16 numbers are exact
    in float64 and math.fsum adds them exactly; the sum is rounded to float32, as the
    accumulator holds it, then to bf16 by `_bf16`'s arithmetic."""
    return [_after_the_gemm(z, c, t) for z in _vllm_gemm(w, h)]


def _vllm_gemm(w: list[list[float]], h: list[float]) -> list[float]:
    """z = bf16(W bf16(h)), one rounding at a time."""
    hb = [_bf16(_f32(x)) for x in h]
    return [_bf16(_f32(math.fsum(a * b for a, b in zip(row, hb, strict=True)))) for row in w]


@pytest.mark.parametrize("t", [1.0, 0.7, 1.9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_read_matches_the_vllm_runtimes_arithmetic(seed, t):
    """Random bf16 weights and hidden states, scaled so that many logits fall between 16
    and 32 (where bf16's spacing is 0.125) and some past the cap. The core's z is the
    vLLM runtime's bit for bit; its probabilities, from every stage, are its softmax."""
    rng = np.random.default_rng(seed)
    bf16 = np.vectorize(_bf16, otypes=[np.float64])
    hidden = 64
    h = bf16(rng.normal(0, 1, hidden))
    scale = np.array([0.25, 0.5, 1.0, 2.0, 4.0])[np.arange(255) % 5, None]
    w = bf16(rng.normal(0, 1, (255, hidden)) * scale)
    z_vllm = [
        _bf16(_f32(math.fsum(a * b for a, b in zip(r, h.tolist(), strict=True))))
        for r in w.tolist()
    ]
    za = np.abs(np.array(z_vllm))
    assert np.sum((za >= 16) & (za < 32)) >= 20 and np.sum(za >= 32) >= 5, "cover the band"

    # The GEMM: bit for bit (float32 holds bf16 exactly).
    z = readout_logits(w, h)
    assert z.dtype == np.float32 and z.tolist() == z_vllm
    assert readout_logits(w, np.stack([h, h])).tolist() == [z_vllm, z_vllm]

    # The float32 cap and temperature: within a few float32 steps (numpy's float32 tanh
    # and the correctly rounded one may differ in the last bits).
    final_vllm = np.array(_vllm_logits(w.tolist(), h.tolist(), C, t))
    capped = softcap(z, C)
    final = capped / np.float32(t) if t != 1.0 else capped
    assert np.allclose(final, final_vllm, rtol=0, atol=2e-5)

    # read() from every stage, for questions of 2, 7, 40 and 255 options.
    ks = (2, 7, 40, 255)
    rendered = _request(*[_rq("choice", [f"o{i}" for i in range(k)], name=f"k{k}") for k in ks])
    cfg = _config(temperature=t)
    p255 = _softmax64(final_vllm.tolist())  # what vLLM returns for logprobs: 255
    rows = {
        HIDDEN: [h] * len(ks),
        RAW_LOGITS: [np.array(z_vllm)] * len(ks),
        CAPPED_LOGITS: [capped] * len(ks),
        FINAL_LOGITS: [final_vllm[:k] for k in ks],  # the question's own codes only
        PROBABILITIES: [np.array(p255)] * len(ks),
    }
    for stage in STAGES:
        got = question_probabilities(rows[stage], Readout(w, C), cfg, rendered, stage=stage)
        for k in ks:
            want = _softmax64(final_vllm[:k].tolist())
            assert np.allclose(got[f"k{k}"], want, rtol=0, atol=1e-5), (stage, k)

    # The regime matters: skipping the bf16 rounding (a float64 readout) moves these
    # probabilities by ten times the tolerance above, or more.
    z64 = w @ h
    loose = [_softmax64((C * np.tanh(z64[:k] / C) / t).tolist()) for k in ks]
    gap = max(
        np.max(np.abs(np.array(lo) - _softmax64(final_vllm[:k].tolist())))
        for lo, k in zip(loose, ks, strict=True)
    )
    assert gap > 1e-4


def test_the_hidden_state_is_rounded_to_bf16_before_the_gemm():
    """A float32 decoder's hidden state (llama.cpp's, or MLX's in float32) is rounded to bf16
    before the GEMM, as strands-decider-vllm rounds it (`hidden_states.to(bf16)`): its z is
    the bf16 hidden state's, which the unrounded one is not."""
    rng = np.random.default_rng(11)
    bf16 = np.vectorize(_bf16, otypes=[np.float64])
    w = bf16(rng.normal(0, 1, (255, 64)) * 2)
    h32 = rng.normal(0, 1, 64).astype(np.float32)
    assert not np.array_equal(round_to_bf16(h32), h32), "the test needs a hidden state off bf16"
    z = readout_logits(w, h32)
    assert z.tolist() == _vllm_gemm(w.tolist(), h32.tolist())
    assert np.array_equal(z, readout_logits(w, round_to_bf16(h32)))
    assert np.array_equal(readout_logits(w, h32.astype(np.float64)), z)
    unrounded = [
        _bf16(_f32(math.fsum(a * float(b) for a, b in zip(row, h32, strict=True))))
        for row in w.tolist()
    ]
    assert sum(a != b for a, b in zip(z.tolist(), unrounded, strict=True)) >= 50
    # Through read, from the hidden stage: the bf16 hidden state's answers.
    rendered = _request(_rq("choice", [f"o{i}" for i in range(40)], name="c"))
    probs = question_probabilities([h32], Readout(w, C), _config(), rendered, stage=HIDDEN)
    want = question_probabilities(
        [round_to_bf16(h32)], Readout(w, C), _config(), rendered, stage=HIDDEN
    )
    assert np.array_equal(probs["c"], want["c"])
    with pytest.raises(ValueError, match="finite in bf16"):
        readout_logits(w, np.full(64, 3.4e38, np.float32))
    with pytest.raises(ValueError, match="finite"):
        readout_logits(w, np.full(64, np.nan))


def _bf16_step(z: float) -> float:
    """The spacing of bf16 values at |z|: 8 significant bits."""
    return math.ldexp(1.0, math.frexp(abs(z))[1] - 8) if z else 2.0**-133


def test_the_readout_goldens():
    """tests/fixtures/core/readout_goldens.json, the readout apart from any decoder, by the
    spec's two checks, each from the golden's input: the GEMM on the golden's weight and h
    gives z within one bf16 step of the golden's (the core and the independent
    reimplementation of strands-decider-vllm's arithmetic both give it exactly), and the
    soft-cap, temperature and softmax applied to the golden's z give its probabilities to
    1e-5 (the core's own, to 1e-7)."""
    g, w = readout_goldens()
    c = g["final_logit_softcapping"]
    assert w.shape == (255, 32) and c == 30.0 and len(g["cases"]) >= 6
    for case in g["cases"]:
        h, t, golden_z = np.array(case["hidden"], np.float32), case["temperature"], case["z"]
        for name, z in (
            ("core", readout_logits(w, h).tolist()),
            ("vllm", _vllm_gemm(w.tolist(), h.tolist())),
        ):
            assert all(abs(a - b) <= _bf16_step(b) for a, b in zip(z, golden_z, strict=True)), name
            assert z == golden_z, name  # within one step is allowed; both are exact
        after = [_after_the_gemm(x, c, t) for x in golden_z]  # from the golden's z, not ours
        for k, want in case["probabilities"].items():
            k = int(k)
            core = code_probabilities(golden_z, k, stage=RAW_LOGITS, softcap_c=c, temperature=t)
            assert np.allclose(core, want, rtol=0, atol=1e-7), (case["name"], k)
            assert np.allclose(_softmax64(after[:k]), want, rtol=0, atol=1e-5), (case["name"], k)


def test_a_one_step_z_can_move_the_probabilities_past_the_tolerance():
    """Why the probabilities are measured from the golden's z: in the goldens, one bf16 step
    on the top code's z, at T = 0.05, moves a probability by far more than 1e-5."""
    g, _ = readout_goldens()
    c = g["final_logit_softcapping"]
    case = next(x for x in g["cases"] if x["temperature"] == 0.05)
    z = np.array(case["z"][:40])
    top = int(np.argmax(z))
    moved = z.copy()
    moved[top] -= _bf16_step(z[top])
    want = code_probabilities(z, 40, stage=RAW_LOGITS, softcap_c=c, temperature=0.05)
    got = code_probabilities(moved, 40, stage=RAW_LOGITS, softcap_c=c, temperature=0.05)
    assert np.max(np.abs(got - want)) > 1e-3


def test_the_readout_goldens_see_a_misapplied_cap_or_temperature():
    """Where the bound on model goldens may not, the readout goldens see a skipped cap, a
    doubled temperature or an unrounded hidden state, far past their 1e-5 tolerance."""
    g, w = readout_goldens()
    c = g["final_logit_softcapping"]
    worst = {"cap skipped": 0.0, "T twice": 0.0, "h unrounded": 0.0}
    for case in g["cases"]:
        h, t = np.array(case["hidden"], np.float64), case["temperature"]
        z = np.array(case["z"])
        want = np.array(case["probabilities"]["40"])
        wrong = {
            "cap skipped": _softmax64((z[:40] / t).tolist()),
            "T twice": _softmax64((c * np.tanh(z[:40] / c) / (t * t)).tolist()),
            "h unrounded": _softmax64((c * np.tanh((w @ h)[:40] / c) / t).tolist()),
        }
        for name, p in wrong.items():
            worst[name] = max(worst[name], float(np.max(np.abs(np.array(p) - want))))
    assert all(v > 1e-3 for v in worst.values()), worst


# ---- conformance: the two measures, one way for every backend -------------------------------


def test_agreement_measures_top1_and_mean_dp():
    from strands_decider.core import Agreement, agreement
    from strands_decider.core.conformance import MIN_QUESTIONS

    ref = {"a": [0.7, 0.3], "b": [0.2, 0.5, 0.3], "c": [0.5, 0.5]}
    ours = {"a": [0.6, 0.4], "b": [0.2, 0.3, 0.5], "c": [0.5, 0.5]}
    got = agreement(ours, ref)
    assert got.questions == 3 and got.top1 == pytest.approx(2 / 3)  # b's top-1 moved
    assert got.mean_abs_dp == pytest.approx((0.1 + 0.2 + 0.0) / 3)
    # A tie goes to the first code, for both sides alike.
    assert agreement({"t": [0.5, 0.5]}, {"t": [0.5, 0.5]}).top1 == 1.0
    assert agreement({"t": [0.4, 0.6]}, {"t": [0.5, 0.5]}).top1 == 0.0
    assert Agreement(1200, 0.9883, 0.0099).conforms()
    assert not Agreement(999, 1.0, 0.0).conforms()  # too few questions to claim anything
    assert (
        not Agreement(1200, 0.979, 0.001).conforms() and not Agreement(1200, 0.99, 0.021).conforms()
    )
    assert MIN_QUESTIONS == 1000
    for bad in ({"a": [0.7, 0.3]}, {**ours, "d": [1.0]}):
        with pytest.raises(ValueError, match="questions differ"):
            agreement(bad, ref)
    with pytest.raises(ValueError, match="2 probabilities for 3 codes"):
        agreement({**ours, "b": [0.5, 0.5]}, ref)
    with pytest.raises(ValueError, match="finite"):
        agreement({**ours, "a": [math.nan, 1.0]}, ref)


def test_conforms_applies_the_formats_recorded_bounds():
    """bf16's bounds for bf16 weights, at any size and whatever the activations; a quantized
    format's only at a size docs/spec.md records, and otherwise no claim at all."""
    from strands_decider.core import BOUNDS, Agreement, NoRecordedBound

    good, bad = Agreement(1200, 0.9908, 0.0104), Agreement(1200, 0.975, 0.0104)
    assert BOUNDS == {
        ("bf16", None): (0.98, 0.02),
        ("mlx-8bit", "E2B"): (0.98, 0.02),
        ("gguf-q8_0", "E2B"): (0.98, 0.02),
    }
    assert good.conforms() and good.conforms("bf16") and good.conforms("bf16", "12B")
    assert good.conforms("mlx-8bit", "E2B") and good.conforms("gguf-q8_0", "E2B")
    assert not bad.conforms() and not bad.conforms("gguf-q8_0", "E2B")
    unrecorded = [
        ("mlx-8bit",),
        ("mlx-8bit", "12B"),
        ("gguf-q8_0", "26B-A4B"),
        ("gguf-q4_k_m", "E2B"),
    ]
    for args in [*unrecorded, ("mlx-4bit", "E2B")]:
        with pytest.raises(NoRecordedBound, match=r"no conformance claim|give the size"):
            good.conforms(*args)
    assert issubclass(NoRecordedBound, ValueError)


def test_read_golden_vectors(tmp_path):
    from strands_decider.core import agreement, read_golden_vectors

    lines = [
        {
            "request": {},
            "variant": "strands-plain",
            "questions": {"q": {"probabilities": [0.1, 0.9]}},
        },
        {
            "request": {},
            "variant": "strands-plain",
            "questions": {
                "q": {"probabilities": [0.6, 0.4]},
                "r": {"probabilities": [0.2, 0.3, 0.5]},
            },
        },
    ]
    path = tmp_path / "golden_vectors.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    ref = read_golden_vectors(path)
    assert ref == {(0, "q"): [0.1, 0.9], (1, "q"): [0.6, 0.4], (1, "r"): [0.2, 0.3, 0.5]}
    assert agreement(ref, ref).top1 == 1.0 and agreement(ref, ref).mean_abs_dp == 0.0
    from strands_decider.core import agreement_with_goldens

    assert agreement_with_goldens(ref, path) == agreement(ref, ref)


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


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"dtype": "F32", "shape": [2], "data_offsets": [-8, 0]}, "data_offsets"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [8, 0]}, "data_offsets"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [8, 16]}, "outside"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}, "shape"),
        ({"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}, "shape"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [0.0, 8]}, "data_offsets"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [False, 8]}, "data_offsets"),
        ({"dtype": "F32", "shape": [2], "data_offsets": [0]}, "data_offsets"),
        ({"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}, "shape"),
        ({"dtype": "F32", "shape": "2", "data_offsets": [0, 8]}, "shape"),
        ({"dtype": "F32", "shape": [2]}, "data_offsets"),
        ("weight", "no dtype"),
    ],
)
def test_load_safetensors_refuses_bad_offsets_and_shapes(tmp_path, entry, match):
    """Offsets lie within the data, in order, and hold exactly the shape's bytes: never a
    slice from the file's end or of another tensor's bytes."""
    p = tmp_path / "t.safetensors"
    h = json.dumps({"w": entry}).encode()
    p.write_bytes(struct.pack("<Q", len(h)) + h + bytes(8))
    with pytest.raises(ValueError, match=match):
        load_safetensors(p)
    for header in (b"[1]", b"\xff"):
        p.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
        with pytest.raises(ValueError, match="header"):
            load_safetensors(p)


def test_load_safetensors_refuses_other_dtypes(tmp_path):
    p = tmp_path / "t.safetensors"
    h = json.dumps({"w": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]}}).encode()
    p.write_bytes(struct.pack("<Q", len(h)) + h + b"\x00\x01")
    with pytest.raises(ValueError, match="dtype I8"):
        load_safetensors(p)


def _checkpoint(tmp_path, *, model_config=None, weight=None, **config):
    """A checkpoint directory with readout.safetensors and decision_config.json, and a
    config.json only if `model_config` is given (a converted artifact has none)."""
    w = np.ones((255, 8), np.float32) if weight is None else weight
    _write_safetensors(tmp_path / "readout.safetensors", {"weight": ("BF16", w)})
    (tmp_path / "decision_config.json").write_text(
        json.dumps({**decision_config_fixture(), **config})
    )
    if model_config is not None:
        (tmp_path / "config.json").write_text(json.dumps(model_config))
    return tmp_path


def test_readout_load_needs_no_model_config(tmp_path):
    """The soft-cap comes from decision_config.json, so a GGUF or MLX decoder beside
    readout.safetensors and decision_config.json loads without config.json."""
    r = Readout.load(_checkpoint(tmp_path, final_logit_softcapping=25.0))
    assert r.softcap == 25.0 and r.weight is not None and r.weight.shape == (255, 8)
    cfg = DecisionConfig.load(tmp_path)
    assert Readout.load(tmp_path, cfg).softcap == 25.0


@pytest.mark.parametrize(
    "model_config",
    [
        {"final_logit_softcapping": 30.0},
        {"text_config": {"final_logit_softcapping": 30.0}},
        {},
        {"model_type": "gemma4_text"},
    ],
    ids=["text-model", "multimodal-wrapper", "empty", "without-the-key"],
)
def test_readout_load_accepts_an_agreeing_model_config(tmp_path, model_config):
    r = Readout.load(_checkpoint(tmp_path, model_config=model_config))
    assert r.softcap == 30.0


@pytest.mark.parametrize(
    "model_config",
    [
        {"final_logit_softcapping": 50.0},
        {"final_logit_softcapping": None},
        {"text_config": {"final_logit_softcapping": 20.0}},
        # Both places are read: a top-level value cannot hide text_config's, or the reverse.
        {"final_logit_softcapping": 30.0, "text_config": {"final_logit_softcapping": 20.0}},
        {"text_config": {"final_logit_softcapping": 30.0}, "final_logit_softcapping": 50.0},
    ],
)
def test_readout_load_refuses_a_disagreeing_model_config(tmp_path, model_config):
    with pytest.raises(ValueError, match="inconsistent"):
        Readout.load(_checkpoint(tmp_path, model_config=model_config))


def test_readout_load_refuses_a_bad_readout_or_config(tmp_path):
    ckpt = _checkpoint(tmp_path)
    _write_safetensors(ckpt / "readout.safetensors", {"w": ("F32", np.ones((255, 8)))})
    with pytest.raises(ValueError, match="one tensor"):
        Readout.load(ckpt)
    _write_safetensors(ckpt / "readout.safetensors", {"weight": ("F32", np.ones((254, 8)))})
    with pytest.raises(ValueError, match="255 codes"):
        Readout.load(ckpt)
    (ckpt / "decision_config.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ValueError, match="not a Strands Decider decision config"):
        Readout.load(ckpt)
    (ckpt / "decision_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        Readout.load(ckpt)


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


def test_a_tie_goes_to_the_first_option_in_request_order_however_rendered():
    names = ["billing", "technical", "sales"]
    tie = [0.4, 0.4, 0.2]
    assert answer(_rq("choice", names), tie)["choice"] == "billing"
    # Rendered as sales, technical, billing: the tie is between technical and billing, and
    # billing comes first in the request.
    permuted = _rq("choice", names, order=[2, 1, 0])
    assert answer(permuted, [0.2, 0.4, 0.4])["choice"] == "billing"
    assert answer(_rq("choice", names, order=[1, 0, 2]), [0.4, 0.4, 0.2])["choice"] == "billing"


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


def test_a_rendered_scores_legend_is_its_stripped_rubric():
    """The legend comes from render: each level's rubric text stripped, as v1's legend is,
    and not collapsed to one line as its option line is."""
    tok = toy_tokenizer()
    req = {
        "state": "s",
        "questions": {
            "s": {
                "type": "score",
                "instructions": "x",
                "criteria": ["  calm\n", "a  b\n c", "\tvery\n"],
            }
        },
    }
    rq = render(req, tok, PLAIN).questions["s"]
    assert "A. 0 — calm\nB. 1 — a b c\nC. 2 — very\n" in rq.question_text
    a = answer(rq, [0.2, 0.3, 0.5])
    assert a["legend"] == {"0": "calm", "1": "a  b\n c", "2": "very"}
    reversed_ = render(req, tok, PLAIN, option_orders={"s": [2, 1, 0]}).questions["s"]
    assert answer(reversed_, [0.5, 0.3, 0.2])["legend"] == a["legend"]


def test_answers_match_v1():
    """The same numbers as v1's `prompting.to_answer`, the confidence formulas included."""
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
            want = to_answer(render_question(q), p, ordinal_smoothing=0.1).model_dump()
            desc = list(q.criteria) if kind == "score" else None
            got = answer(_rq(kind, options, descriptions=desc), p, ordinal_smoothing=0.1)
            assert got == want


# ---- read ------------------------------------------------------------------------------


def _config(**kw) -> DecisionConfig:
    return DecisionConfig.from_dict({**decision_config_fixture(), **kw})


def _request(*rqs: RenderedQuestion) -> RenderedRequest:
    return RenderedRequest(variant=PLAIN, model="m", questions={q.name: q for q in rqs})


def test_every_stage_gives_the_same_answers():
    rng = np.random.default_rng(1)
    w = round_to_bf16(rng.normal(0, 0.5, size=(255, 16)))
    h = round_to_bf16(rng.normal(0, 1, size=(2, 16)))
    rendered = _request(
        _rq("noul", ["false", "true"], name="n"), _rq("choice", ["x", "y", "z"], name="c")
    )
    t = 1.3
    cfg = _config(temperature=t)
    z = readout_logits(w, h)
    capped = softcap(z, C)
    final = capped / np.float32(t)
    by_hidden = read(h, Readout(w, C), cfg, rendered, stage=HIDDEN)
    assert by_hidden == read({"c": h[1], "n": h[0]}, Readout(w, C), cfg, rendered, stage=HIDDEN)
    for stage, rows in (
        (RAW_LOGITS, z),
        (CAPPED_LOGITS, capped),
        (FINAL_LOGITS, final),
        (FINAL_LOGITS, [final[0][:2], final[1][:3]]),  # each question's own codes only
    ):
        assert read(rows, None, cfg, rendered, stage=stage) == by_hidden, stage
        assert read(rows, Readout(None, C), cfg, rendered, stage=stage) == by_hidden, stage
    want = _manual(z[1].tolist(), 3, C, t)
    assert by_hidden["c"]["probabilities"] == {
        o: round(p, 4) for o, p in zip("xyz", want, strict=True)
    }
    # Probabilities, ragged (k per question) as vLLM returns them, or over all 255 codes.
    by_p = question_probabilities(
        [_manual(z[0].tolist(), 2, C, t), _manual(z[1].tolist(), 255, C, t)],
        None,
        cfg,
        rendered,
        stage=PROBABILITIES,
    )
    ref = question_probabilities(h, Readout(w, C), cfg, rendered, stage=HIDDEN)
    for n in ref:
        assert np.allclose(by_p[n], ref[n], rtol=0, atol=1e-6), n


def test_read_refuses_a_mismatch():
    w = np.zeros((255, 4), np.float32)
    rendered = _request(_rq("noul", ["false", "true"]))
    with pytest.raises(TypeError):
        read(np.zeros((1, 4)), Readout(w, C), _config(), rendered)  # type: ignore[call-arg]
    for bad in (None, "logits", "probs"):
        with pytest.raises(ValueError, match="stage must be"):
            read(np.zeros((1, 4)), Readout(w, C), _config(), rendered, stage=bad)
    with pytest.raises(ValueError, match="rendered for"):
        read(
            np.zeros((1, 4)),
            Readout(w, C),
            _config(prompt="strands-chat"),
            rendered,
            stage=HIDDEN,
        )
    bad_ids = list(decision_config_fixture()["token_ids"])
    bad_ids[0], bad_ids[1] = bad_ids[1], bad_ids[0]
    with pytest.raises(ValueError, match="token_ids"):
        read(np.zeros((1, 4)), Readout(w, C), _config(token_ids=bad_ids), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="one per question"):
        read(np.zeros((2, 4)), Readout(w, C), _config(), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="no row"):
        read({"other": np.zeros(4)}, Readout(w, C), _config(), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="rows for 255 codes"):
        read(np.zeros((1, 4)), Readout(np.zeros((254, 4)), C), _config(), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="hidden state of 4"):
        read(np.zeros((1, 5)), Readout(w, C), _config(), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="readout weight"):
        read(np.zeros((1, 4)), Readout(None, C), _config(), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="readout weight"):
        read(np.zeros((1, 4)), None, _config(), rendered, stage=HIDDEN)
    with pytest.raises(ValueError, match="255 values"):
        read(np.zeros((1, 100)), None, _config(), rendered, stage=RAW_LOGITS)
    with pytest.raises(ValueError, match="a vector"):
        read([np.zeros((2, 2))], None, _config(), rendered, stage=RAW_LOGITS)
    # The readout's cap must be the checkpoint's: one source for c.
    with pytest.raises(ValueError, match="soft-cap"):
        read(np.zeros((1, 4)), Readout(w, 20.0), _config(), rendered, stage=HIDDEN)
    # Every checkpoint has a soft-cap, so a readout has one too.
    with pytest.raises(ValueError, match="softcap"):
        Readout(w, None)  # type: ignore[arg-type]


def test_the_readout_holds_its_weight_in_bf16():
    w = np.full((255, 2), 1 + 2.0**-8 + 2.0**-20, np.float32)  # just past a bf16 tie
    r = Readout(w, C)
    assert r.weight is not None and np.all(r.weight == 1 + 2.0**-7)
    with pytest.raises(ValueError, match=r"\[codes, hidden\]"):
        Readout(np.zeros(4), C)
    with pytest.raises(ValueError, match="softcap"):
        Readout(w, 0.0)


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


T = 0.8  # a checkpoint temperature other than 1, so that applying it twice would show


class _FakeBackend:
    """Returns, for each question, the stage `outputs` of a decoder whose hidden state
    points at one code: with the identity as readout, `W h` is 50 on that code only. It
    records the images it is given; with `images=False` it has no image path and refuses
    them, as a backend without a vision encoder does."""

    def __init__(
        self, picks: dict[tuple[int, ...], int], outputs: str = HIDDEN, *, images: bool = True
    ):
        self.picks, self.outputs, self.calls = picks, outputs, []
        self.vision = images
        self.images: list = []

    def forward(self, state, questions, images=()):
        if images and not self.vision:
            raise ValueError(NO_VISION)
        self.images.append(images)
        self.calls.append((state, questions))
        rows = np.zeros((len(questions), 255))
        for i, q in enumerate(questions):
            rows[i, self.picks[tuple(q)]] = 50.0
        if self.outputs in (HIDDEN, RAW_LOGITS):
            return rows
        capped = softcap(rows, C)
        if self.outputs == CAPPED_LOGITS:
            return capped
        final = capped / np.float32(T)
        if self.outputs == FINAL_LOGITS:
            return final
        # Probabilities over each question's own codes, ragged, as vLLM's logprobs give.
        ks = [2, 3]
        return [
            code_probabilities(r[:k], k, stage=FINAL_LOGITS, softcap_c=C, temperature=T)
            for r, k in zip(final, ks, strict=True)
        ]


class _DecoderBackend:
    """A decoder with a hidden state per question, of width 16, and a readout that is
    neither the identity nor square: it returns, for each question, the stage `outputs`
    computed from them the way the stage's runtime would."""

    def __init__(self, hidden: dict[tuple[int, ...], np.ndarray], weight, outputs: str):
        self.hidden, self.weight, self.outputs, self.calls = hidden, weight, outputs, []

    def forward(self, state, questions, images=()):
        assert images == ()
        self.calls.append((state, questions))
        h = np.stack([self.hidden[tuple(q)] for q in questions])
        if self.outputs == HIDDEN:
            return h
        raw = readout_logits(self.weight, h)
        if self.outputs == RAW_LOGITS:
            return raw
        capped = softcap(raw, C)
        if self.outputs == CAPPED_LOGITS:
            return capped
        final = capped / np.float32(T)
        if self.outputs == FINAL_LOGITS:
            return final
        # Probabilities over each question's own codes, ragged, as vLLM's logprobs give.
        return [_softmax64(r[:k].tolist()) for r, k in zip(final, (2, 3), strict=True)]


E2E_REQUEST = {
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


def _e2e_decoder():
    """A readout [255, 16] and two hidden states whose question codes' logits lie where the
    cap bends them (|z| about 15 to 40) with gaps of about 2."""
    rng = np.random.default_rng(21)
    w = round_to_bf16(rng.normal(0, 1, (255, 16)))
    # The smallest hidden states that give these logits on the question's codes.
    h_n = np.linalg.lstsq(w[:2], np.array([-21.0, -19.0]), rcond=None)[0]
    h_c = np.linalg.lstsq(w[:3], np.array([18.0, 20.5, 16.0]), rcond=None)[0]
    return w, round_to_bf16(h_n), round_to_bf16(h_c)


def test_the_end_to_end_numbers_can_see_a_misapplied_stage():
    """The end-to-end test's numbers move by more than its tolerance when the cap is skipped
    or applied twice, or the temperature is applied twice or not at all."""
    w, h_n, h_c = _e2e_decoder()
    z = readout_logits(w, np.stack([h_n, h_c]))
    zn, zc = z[0, :2].astype(np.float64), z[1, :3].astype(np.float64)
    assert 1.0 < abs(zn[1] - zn[0]) < 3.0 and all(10 < abs(v) < 45 for v in [*zn, *zc])

    def p(v, cap, t):
        return np.array(_softmax64([(C * math.tanh(x / C) if cap else x) / t for x in v]))

    ref = [p(zn, True, T), p(zc, True, T)]
    wrong = {
        "cap skipped": [p(zn, False, T), p(zc, False, T)],
        "cap twice": [p(C * np.tanh(zn / C), True, T), p(C * np.tanh(zc / C), True, T)],
        "T twice": [p(zn, True, T * T), p(zc, True, T * T)],
        "T skipped": [p(zn, True, 1.0), p(zc, True, 1.0)],
    }
    for name, (a, b) in wrong.items():
        gap = max(np.max(np.abs(a - ref[0])), np.max(np.abs(b - ref[1])))
        assert gap > 1e-2, name


@pytest.mark.parametrize("outputs", STAGES)
def test_evaluate_end_to_end_with_a_toy_tokenizer(outputs):
    tok = toy_tokenizer()
    cfg = _config(token_ids=[tok.token_to_id(c) for c in CODES], temperature=T)
    req = E2E_REQUEST
    rendered = render(req, tok, PLAIN)
    ids = {n: q.question_ids for n, q in rendered.questions.items()}
    w, h_n, h_c = _e2e_decoder()
    backend = _DecoderBackend({ids["n"]: h_n, ids["c"]: h_c}, w, outputs)
    readout = Readout(w, C) if outputs == HIDDEN else None
    resp = evaluate(req, tok, readout, cfg, backend, model_name="t")
    ((state, questions),) = backend.calls
    assert state == list(rendered.state_ids)
    assert questions == [list(q.question_ids) for q in rendered.questions.values()]
    assert resp["model"] == "t" and list(resp["answers"]) == ["n", "c"]
    # Every stage gives the reference's numbers: the cap and T applied once each, which
    # test_the_end_to_end_numbers_can_see_a_misapplied_stage shows this tolerance can see.
    z = readout_logits(w, np.stack([h_n, h_c]))
    want_n, want_c = _manual(z[0].tolist(), 2, C, T), _manual(z[1].tolist(), 3, C, T)
    assert resp["answers"]["n"]["noul"] == pytest.approx(want_n[1], abs=1e-4)
    got_c = resp["answers"]["c"]["probabilities"]
    assert list(got_c.values()) == pytest.approx(want_c, abs=1e-4)
    assert resp["answers"]["c"]["choice"] == "xyz"[int(np.argmax(want_c))]
    assert resp["usage"] == {"input_tokens": rendered.input_tokens, "output_tokens": 2}
    with pytest.raises(ValueError, match="exceeds the limit"):
        evaluate(
            req,
            tok,
            readout,
            _config(token_ids=list(cfg.token_ids), max_length_validated=5, temperature=T),
            backend,
        )


def test_evaluate_checks_the_configuration_before_the_forward_pass():
    """What can be checked without the decoder is checked before it runs: a misconfigured
    runtime never pays for a forward pass whose rows it cannot read."""
    tok = toy_tokenizer()
    ids = [tok.token_to_id(c) for c in CODES]
    w, _, _ = _e2e_decoder()

    class Untouchable:
        outputs = HIDDEN

        def forward(self, state, questions, images=()):
            raise AssertionError("forward ran before the configuration was checked")

    cases = [
        (None, _config(token_ids=ids), "readout weight"),
        (Readout(None, C), _config(token_ids=ids), "readout weight"),
        (Readout(w, 20.0), _config(token_ids=ids), "soft-cap"),
        (Readout(w[:254], C), _config(token_ids=ids), "rows for 255 codes"),
        (Readout(w, C), _config(), "token_ids"),  # the fixture's ids, not the toy tokenizer's
    ]
    for readout, cfg, match in cases:
        with pytest.raises(ValueError, match=match):
            evaluate(E2E_REQUEST, tok, readout, cfg, Untouchable())


IMAGE_REQUEST = {
    "state": "a short state",
    "images": [image_b64("checker-64x48.png"), image_b64("rings-90x120.webp", data_uri=True)],
    "questions": {
        "n": {"type": "noul", "instructions": "is it short"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"x": None, "y": None}},
    },
}


def test_evaluate_hands_a_backend_the_images_with_their_spans():
    tok = toy_tokenizer()
    cfg = _config(token_ids=[tok.token_to_id(c) for c in CODES])
    rendered = render(IMAGE_REQUEST, tok, PLAIN)
    ids = {n: q.question_ids for n, q in rendered.questions.items()}
    backend = _FakeBackend({ids["n"]: 1, ids["c"]: 0})
    resp = evaluate(IMAGE_REQUEST, tok, Readout(np.eye(255, dtype=np.float32), C), cfg, backend)
    ((state, _),) = backend.calls
    (images,) = backend.images
    assert state == list(rendered.state_ids) and images == rendered.images
    assert [i.data for i in images] == [
        image_bytes("checker-64x48.png"),
        image_bytes("rings-90x120.webp"),
    ]
    soft = tok.token_to_id("<|image|>")
    assert [state[i.start : i.end] for i in images] == [[soft] * 266, [soft] * 266]
    assert resp["answers"]["c"]["choice"] == "x" and resp["answers"]["n"]["noul"] > 0.99
    assert resp["usage"]["input_tokens"] == rendered.input_tokens
    # A text request reaches the same backend with no images.
    text = {k: v for k, v in IMAGE_REQUEST.items() if k != "images"}
    text_ids = {n: q.question_ids for n, q in render(text, tok, PLAIN).questions.items()}
    backend = _FakeBackend({text_ids["n"]: 1, text_ids["c"]: 0})
    evaluate(text, tok, Readout(np.eye(255, dtype=np.float32), C), cfg, backend)
    assert backend.images == [()] and len(backend.calls) == 1


def test_evaluate_refuses_images_without_an_image_path():
    """A checkpoint whose `inputs` lack "image" has its image requests refused before the
    images are decoded, as v1's text-only engine refuses them. A backend without an image
    path refuses them from `forward`, after the images' headers are checked."""
    tok = toy_tokenizer()
    ids = [tok.token_to_id(c) for c in CODES]
    readout = Readout(np.eye(255, dtype=np.float32), C)
    bad = {**IMAGE_REQUEST, "images": ["not base64!"]}
    for req in (IMAGE_REQUEST, bad):
        backend = _FakeBackend({})
        with pytest.raises(ValueError, match="does not take images") as e:
            evaluate(req, tok, readout, _config(token_ids=ids, inputs=["text"]), backend)
        assert str(e.value).endswith('its decision_config.json inputs are ["text"]')
        assert backend.calls == []
    text_only = _FakeBackend({}, images=False)
    with pytest.raises(ValueError) as e:
        evaluate(IMAGE_REQUEST, tok, readout, _config(token_ids=ids), text_only)
    assert str(e.value) == NO_VISION
    # A checkpoint that takes images checks them first, whatever the backend.
    for backend in (_FakeBackend({}), _FakeBackend({}, images=False)):
        with pytest.raises(ValueError, match="image is not valid base64"):
            evaluate(bad, tok, readout, _config(token_ids=ids), backend)


def test_a_malformed_request_with_images_is_refused_for_what_is_malformed():
    """v1 validates the request before its text-only engine refuses the images, so a request
    that is malformed and has images gets the malformed part's message from any runtime."""
    tok = toy_tokenizer()
    ids = [tok.token_to_id(c) for c in CODES]
    readout = Readout(np.eye(255, dtype=np.float32), C)
    bad = {**IMAGE_REQUEST, "questions": {"q": {"type": "noul"}}}
    for cfg, backend in (
        (_config(token_ids=ids, inputs=["text"]), _FakeBackend({})),
        (_config(token_ids=ids), _FakeBackend({}, images=False)),
    ):
        with pytest.raises(ValueError, match="instructions: Field required"):
            evaluate(bad, tok, readout, cfg, backend)


def test_evaluate_needs_the_backend_to_name_its_stage():
    tok = toy_tokenizer()
    cfg = _config(token_ids=[tok.token_to_id(c) for c in CODES])
    req = {"state": "s", "questions": {"n": {"type": "noul", "instructions": "x"}}}

    class Unnamed:
        def forward(self, state, questions, images=()):
            return np.zeros((len(questions), 255))

    for backend in (Unnamed(), _FakeBackend({}, outputs="logits")):
        with pytest.raises(ValueError, match="must say what forward returns"):
            evaluate(req, tok, Readout(np.eye(255), C), cfg, backend)  # type: ignore[arg-type]
