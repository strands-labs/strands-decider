"""From the hidden state at a question's last token to its answer (docs/spec.md, section 7).

    z = bf16(W bf16(h))        W: readout.safetensors `weight` in bf16; a bf16 GEMM
    z = tanh(z / c) * c        float32; c: decision_config.json `final_logit_softcapping`
    z = z / T                  float32; T: decision_config.json `temperature`
    p = softmax(z[:k])         float32, over the question's own k codes only
    p -> the answer            v1's response schema and confidence formulas

This is the training reference's precision regime, and that of strands-decider-vllm, the vLLM
runtime.
numpy has no bf16, so the GEMM is emulated: the hidden state is rounded to bf16 (a bf16
decoder's is unchanged by it), the products are summed in float64, and the sum is rounded to
float32 and then to bf16 by the float32's bits, round to nearest, ties to even
(`round_to_bf16`).

A runtime may hand over any stage between the hidden state and the probabilities. The stage
is always named (`stage`, `Backend.outputs`), so neither the soft-cap nor the temperature
is ever applied twice: `hidden`, `raw_logits` (W h), `capped_logits` (the cap applied),
`final_logits` (the cap and the temperature applied) or `probabilities`.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..confidence import derive_confidence, derive_score_confidence
from .config import FILENAME, DecisionConfig, check_float32_scale
from .prompt import RenderedQuestion, RenderedRequest

READOUT_FILE = "readout.safetensors"
MODEL_CONFIG_FILE = "config.json"

# What a backend returns, and so what `read` still applies. Each name says what has been done.
HIDDEN = "hidden"  # h, the final-norm hidden state at the last token: nothing applied
RAW_LOGITS = "raw_logits"  # z = W h: the readout applied, not the soft-cap or the temperature
CAPPED_LOGITS = "capped_logits"  # tanh(z / c) * c: the soft-cap applied, not the temperature
FINAL_LOGITS = "final_logits"  # tanh(z / c) * c / T: both, as in strands-decider-vllm's model
PROBABILITIES = "probabilities"  # softmax of the final logits, over the k codes or all 255
STAGES = (HIDDEN, RAW_LOGITS, CAPPED_LOGITS, FINAL_LOGITS, PROBABILITIES)

_DTYPES: dict[str, np.dtype[Any]] = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
}


def _is_index(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file as float32: F32, F16 and BF16, without the
    safetensors package (an 8-byte length, a JSON header, then the raw little-endian data).
    Each tensor's `data_offsets` must lie within the data, in order, and hold exactly its
    shape's bytes."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError(f"{path}: not a safetensors file")
    (n,) = struct.unpack("<Q", data[:8])
    if n > len(data) - 8:
        raise ValueError(f"{path}: header longer than the file")
    try:
        header = json.loads(data[8 : 8 + n])
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: the header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise ValueError(f"{path}: the header is not a JSON object")
    body = memoryview(data)[8 + n :]
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"{path}: tensor {name!r} has no dtype, shape and offsets")
        dtype = info.get("dtype")
        if dtype not in _DTYPES:
            raise ValueError(
                f"{path}: tensor {name!r} has dtype {dtype}; expected F32, F16 or BF16"
            )
        offsets, shape_ = info.get("data_offsets"), info.get("shape")
        if not (isinstance(offsets, list) and len(offsets) == 2 and all(map(_is_index, offsets))):
            raise ValueError(f"{path}: tensor {name!r} has data_offsets {offsets!r}")
        if not (isinstance(shape_, list) and all(map(_is_index, shape_))):
            raise ValueError(f"{path}: tensor {name!r} has shape {shape_!r}")
        lo, hi = offsets
        shape = tuple(shape_)
        if not lo <= hi <= len(body):
            raise ValueError(
                f"{path}: tensor {name!r} has data_offsets [{lo}, {hi}] outside its "
                f"{len(body)} bytes of data"
            )
        if hi - lo != _DTYPES[dtype].itemsize * math.prod(shape):
            raise ValueError(f"{path}: tensor {name!r} does not match its shape {shape}")
        raw = np.frombuffer(body[lo:hi], dtype=_DTYPES[dtype])
        if dtype == "BF16":  # the high half of a float32
            arr = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = raw.astype(np.float32)
        out[name] = arr.reshape(shape)
    return out


# ---- the precision regime ------------------------------------------------------------------


def round_to_bf16(x: Any) -> np.ndarray:
    """`x` as float32, rounded to the nearest bfloat16 (round to nearest, ties to even) and
    returned as float32, which holds every bf16 value exactly.

    bf16 is the high 16 bits of a float32. Adding 0x7FFF, plus 1 when bit 16 (the lowest
    kept bit) is set, and clearing the low 16 bits rounds to nearest with ties to even; a
    value past bf16's largest finite becomes inf, as in torch and CUDA's conversion. NaN
    stays NaN.
    """
    # Annotated as plain arrays: numpy's stubs type these operations differently by version.
    a: np.ndarray = np.asarray(x, dtype=np.float32)
    u: np.ndarray = np.atleast_1d(a).view(np.uint32).astype(np.uint64)  # the add cannot wrap
    u = (u + ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)) & np.uint64(0xFFFF0000)
    rounded: np.ndarray = u.astype(np.uint32).view(np.float32).reshape(a.shape)
    out: np.ndarray = np.where(np.isnan(a), np.float32(np.nan), rounded).astype(np.float32)
    return out


def readout_logits(weight: Any, hidden: Any) -> np.ndarray:
    """z = W h as a bf16 GEMM: `weight` [n_codes, hidden] and `hidden` ([hidden] or [rows,
    hidden]) each rounded to bf16, as a bf16 GEMM takes them (a hidden state wider than
    float32 is rounded to float32 first), the products summed in float64, and each sum
    rounded to float32 and then to bf16. Returns float32, [n_codes] or [rows, n_codes].

    A bf16 decoder's hidden state is unchanged by the rounding; a float32 decoder's (llama.cpp,
    MLX in float32) is rounded, as strands-decider-vllm rounds it. A GPU's bf16 GEMM sums in
    float32, in its own order, so where a sum falls next to a bf16 rounding boundary its z
    can differ from this one by one bf16 step.
    """
    w = round_to_bf16(weight).astype(np.float64)
    h64 = np.asarray(hidden, dtype=np.float64)
    if w.ndim != 2 or h64.ndim not in (1, 2) or h64.shape[-1] != w.shape[1]:
        raise ValueError(f"readout weight {w.shape} cannot take hidden states {h64.shape}")
    if not np.all(np.isfinite(h64)):
        raise ValueError("hidden states must be finite")
    with np.errstate(over="ignore"):
        h = round_to_bf16(h64).astype(np.float64)
    if not np.all(np.isfinite(h)):
        raise ValueError("hidden states must be finite in bf16")
    with np.errstate(over="ignore"):
        z32 = (h @ w.T).astype(np.float32)
    if not np.all(np.isfinite(z32)):
        raise ValueError("readout logits overflow float32")
    return round_to_bf16(z32)


def softcap(z: Any, c: float | None) -> np.ndarray:
    """Gemma's final-logit soft-cap in float32, as strands-decider-vllm computes it: `tanh(z / c) * c`,
    each operation rounded to float32. None leaves z as it is (as float32)."""
    z32 = np.asarray(z, dtype=np.float32)
    if c is None:
        return z32
    c32 = np.float32(check_float32_scale("the soft-cap", c))
    out: np.ndarray = np.tanh(z32 / c32) * c32
    return out


def _softmax32(s: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # a gap past float32's range is -inf, and exp gives 0
        e = np.exp(s - s.max())
    out: np.ndarray = e / e.sum(dtype=np.float32)
    return out


def code_probabilities(
    values: Any,
    k: int,
    *,
    stage: str,
    softcap_c: float | None,
    temperature: float,
) -> np.ndarray:
    """The probabilities of the first `k` codes, from one row of the stage `stage`.

    `raw_logits` are rounded to bf16 (a bf16 GEMM's output is unchanged by it), then capped;
    `capped_logits` are divided by the temperature; all three logit stages end in a float32
    softmax over the k codes, whose other codes' logits do not enter it. A `probabilities`
    row is renormalised over the k codes. Capped or final logits past the cap's range are
    refused: they cannot be what their stage says.

    Args:
        values: One row of the stage `stage`: the 255 codes in code order, or only the
            question's first `k`. A `probabilities` row is a softmax over the codes it holds.
        k: How many codes the question has, at least 1 and at most the row's width.
        stage: What the row is: `raw_logits`, `capped_logits`, `final_logits` or
            `probabilities` (`STAGES` without `hidden`, which goes through
            `readout_logits` first).
        softcap_c: The soft-cap c, the checkpoint's `final_logit_softcapping`, or None for
            none.
        temperature: The temperature T, the checkpoint's `temperature`.

    Returns:
        The k probabilities in code order, [k]: float32, or float64 for `probabilities`.

    Raises:
        ValueError: For `hidden` or an unknown stage, a soft-cap or temperature that is not
            a number above 0 in float32's normal range, a row that is not a vector of at
            least k values finite in float32, capped or final logits past the cap's range,
            probabilities outside [0, 1] or all 0, or logits that overflow float32 when
            divided by T.
    """
    if stage not in STAGES or stage == HIDDEN:
        raise ValueError(
            f"stage must be one of {STAGES[1:]} here, not {stage!r} (hidden states need "
            "readout_logits first)"
        )
    t = np.float32(check_float32_scale("temperature", temperature))
    c = None if softcap_c is None else check_float32_scale("the soft-cap", softcap_c)
    dtype: type[np.floating[Any]] = np.float64 if stage == PROBABILITIES else np.float32
    with np.errstate(over="ignore"):  # a float64 value past float32's range becomes inf
        v = np.asarray(values, dtype=dtype)
    if v.ndim != 1 or not 1 <= k <= v.shape[0]:
        raise ValueError(f"need a vector of at least {k} values, not shape {v.shape}")
    v = v[:k]
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{stage} must be finite in float32: {v[~np.isfinite(v)][:3]}")
    if stage == PROBABILITIES:
        if np.any(v < 0) or np.any(v > 1 + 1e-6) or v.sum() <= 0:
            raise ValueError("probabilities must lie in [0, 1], and not all be 0")
        p: np.ndarray = v / v.sum()
        return p
    if stage == RAW_LOGITS:
        s = softcap(round_to_bf16(v), c)
    elif c is not None:
        bound = np.float32(c) / (t if stage == FINAL_LOGITS else np.float32(1))
        if np.any(np.abs(v) > bound * np.float32(1 + 1e-6)):
            raise ValueError(
                f"{stage} exceed the soft-cap's range (|z| <= {bound:g}); were they capped?"
            )
        s = v
    else:
        s = v
    if stage != FINAL_LOGITS:
        with np.errstate(over="ignore"):
            s = s / t
        if not np.all(np.isfinite(s)):
            raise ValueError(f"logits divided by the temperature {temperature!r} overflow float32")
    return _softmax32(s.astype(np.float32))


# ---- the readout file -------------------------------------------------------------------------


def _softcaps_in(model_config: Any) -> list[tuple[str, Any]]:
    """Each `final_logit_softcapping` config.json gives: under `text_config` (the multimodal
    model's) and at the top level (a text model's), where present."""
    if not isinstance(model_config, Mapping):
        raise ValueError(f"{MODEL_CONFIG_FILE} must hold a JSON object")
    text = model_config.get("text_config")
    out = []
    if isinstance(text, Mapping) and "final_logit_softcapping" in text:
        out.append(("text_config.final_logit_softcapping", text["final_logit_softcapping"]))
    if "final_logit_softcapping" in model_config:
        out.append(("final_logit_softcapping", model_config["final_logit_softcapping"]))
    return out


@dataclass(frozen=True, eq=False)
class Readout:
    """The readout: `weight` [n_codes, hidden], held as float32 values rounded to bf16, and
    the soft-cap `c`, the checkpoint's `final_logit_softcapping`, which every checkpoint has.

    `weight` may be None for a runtime that applies the readout itself and passes logits or
    probabilities.
    """

    weight: np.ndarray | None
    softcap: float

    def __post_init__(self) -> None:
        if self.weight is not None:
            w = round_to_bf16(self.weight)
            if w.ndim != 2:
                raise ValueError(f"readout weight must be [codes, hidden], not {w.shape}")
            object.__setattr__(self, "weight", w)
        object.__setattr__(self, "softcap", check_float32_scale("softcap", self.softcap))

    @classmethod
    def load(
        cls, checkpoint: str | os.PathLike[str], config: DecisionConfig | None = None
    ) -> Readout:
        """From a checkpoint directory: `weight` from readout.safetensors, and the soft-cap
        from decision_config.json's `final_logit_softcapping` (`config`, if given, else the
        directory's file). A converted artifact needs no config.json; if the directory has
        one that names a different soft-cap, under `text_config` or at its top level, the
        checkpoint is inconsistent and refused."""
        d = os.fspath(checkpoint)
        cfg = config if config is not None else DecisionConfig.load(d)
        tensors = load_safetensors(os.path.join(d, READOUT_FILE))
        if set(tensors) != {"weight"}:
            raise ValueError(
                f"{READOUT_FILE} must hold one tensor, `weight`; has {sorted(tensors)}"
            )
        w = tensors["weight"]
        if w.ndim != 2 or w.shape[0] != len(cfg.codes):
            raise ValueError(
                f"readout weight must be [{len(cfg.codes)} codes, hidden], not {w.shape}"
            )
        softcap = cfg.final_logit_softcapping
        model_config = os.path.join(d, MODEL_CONFIG_FILE)
        if os.path.exists(model_config):
            with open(model_config, encoding="utf-8") as f:
                theirs = _softcaps_in(json.load(f))
            for key, value in theirs:
                if value != softcap:
                    raise ValueError(
                        f"{MODEL_CONFIG_FILE} has {key} {value!r} and {FILENAME} has "
                        f"final_logit_softcapping {softcap!r}; the checkpoint is inconsistent"
                    )
        return cls(weight=w, softcap=softcap)


# ---- answers ----------------------------------------------------------------------------------


def answer(
    rq: RenderedQuestion, probs: Sequence[float], *, ordinal_smoothing: float = 0.0
) -> dict[str, Any]:
    """One answer in v1's response schema (`schema.NoulAnswer`, `ChoiceAnswer`,
    `ScoreAnswer`), from the probabilities of the question's codes in code order.

    Values are keyed by what the request named, in the request's order, whatever order the
    options were rendered in. Rounded to 4 places, as v1 rounds.

    Args:
        rq: The rendered question (`RenderedRequest.questions`).
        probs: The probabilities of its codes, in code order: one per option.
        ordinal_smoothing: For a score, the share of each training target's probability
            mass that training put on the neighbouring levels, which its confidence corrects
            for (`confidence.derive_score_confidence`); 0 for none. Nouls and choices
            ignore it.

    Returns:
        The answer as JSON, its `type` the question's kind.

    Raises:
        ValueError: When `probs` does not hold one probability per code.
    """
    p = [float(x) for x in probs]
    if len(p) != len(rq.codes):
        raise ValueError(f"{rq.name!r}: {len(p)} probabilities for {len(rq.codes)} codes")
    by_option = dict(zip(rq.options, p, strict=True))
    if rq.kind == "noul":
        return {"type": "noul", "noul": round(by_option["true"], 4)}
    if rq.kind == "choice":
        # In request order: rq.options is the rendered order, which may be permuted.
        probs_by_name = {o: by_option[o] for o in rq.request_options}
        best = max(probs_by_name, key=lambda o: probs_by_name[o])
        return {
            "type": "choice",
            "choice": best,
            "probabilities": {o: round(v, 4) for o, v in probs_by_name.items()},
            "confidence": round(derive_confidence(list(probs_by_name.values())), 4),
        }
    levels = list(rq.request_options)  # "0" .. "L-1", ascending
    ordered = [by_option[lv] for lv in levels]
    legend = dict(zip(rq.options, rq.descriptions, strict=True))
    return {
        "type": "score",
        "score": round(sum(i * q for i, q in enumerate(ordered)), 4),
        "legend": {lv: legend[lv] for lv in levels},
        "probabilities": {lv: round(q, 4) for lv, q in zip(levels, ordered, strict=True)},
        "confidence": round(
            derive_score_confidence(ordered, ordinal_smoothing=ordinal_smoothing), 4
        ),
    }


def _rows(values: Any, names: list[str]) -> list[np.ndarray]:
    """One 1-D row per question: from a mapping by question name, an array [questions,
    width], or a sequence of rows (which may differ in width, as k does)."""
    if isinstance(values, Mapping):
        missing = [n for n in names if n not in values]
        if missing:
            raise ValueError(f"no row for questions {missing}")
        rows = [np.asarray(values[n]) for n in names]
    else:
        if isinstance(values, np.ndarray) and values.ndim != 2:
            raise ValueError(f"expected [{len(names)}, width] rows, not {values.shape}")
        rows = [np.asarray(r) for r in values]
        if len(rows) != len(names):
            raise ValueError(f"expected {len(names)} rows, one per question, not {len(rows)}")
    for n, r in zip(names, rows, strict=True):
        if r.ndim != 1:
            raise ValueError(f"{n!r}: a row must be a vector, not shape {r.shape}")
    return rows


def check_stage(stage: str, readout: Readout | None, config: DecisionConfig) -> None:
    """What reading rows of `stage` needs, checked before any row exists: a named stage, a
    readout whose soft-cap is the checkpoint's, and for hidden states a readout weight with a
    row per code. `evaluate` calls it before the forward pass; ValueError otherwise."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
    c = config.final_logit_softcapping
    if readout is not None and readout.softcap != c:
        raise ValueError(
            f"the readout's soft-cap {readout.softcap!r} is not the checkpoint's "
            f"final_logit_softcapping {c!r}"
        )
    if stage == HIDDEN:
        if readout is None or readout.weight is None:
            raise ValueError("hidden states need the readout weight")
        if readout.weight.shape[0] != len(config.codes):
            raise ValueError(
                f"readout has {readout.weight.shape[0]} rows for {len(config.codes)} codes"
            )


def check_rendered(config: DecisionConfig, rendered: RenderedRequest) -> None:
    """That `rendered` is what the checkpoint reads: its prompt, and each question's code
    token ids the checkpoint's `token_ids`. `evaluate` calls it before the forward pass."""
    if config.prompt != rendered.variant:
        raise ValueError(
            f"the request was rendered for {rendered.variant!r}, the checkpoint reads {config.prompt!r}"
        )
    for n, rq in rendered.questions.items():
        if tuple(config.token_ids[: len(rq.codes)]) != rq.code_token_ids:
            raise ValueError(
                f"{n!r}: the tokenizer's code tokens do not match the checkpoint's token_ids"
            )


def question_probabilities(
    values: Any,
    readout: Readout | None,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    stage: str,
) -> dict[str, np.ndarray]:
    """Each question's code probabilities: section 7 of docs/spec.md from the stage `stage`
    on. The soft-cap is the checkpoint's (`config.final_logit_softcapping`) and the
    temperature `config.temperature`; `stage` says which of them the rows already hold, so
    that neither is applied twice.

    Args:
        values: One row per question, of the stage `stage`: an array [questions, width], a
            sequence of rows, which may differ in width, or a mapping from question name to
            row. An array or a sequence is in `rendered.questions` order. A row of logits or
            probabilities holds the 255 codes or only the question's k, in code order.
        readout: The checkpoint's readout. `hidden` rows need its `weight`; for the other
            stages it may be None. When given, its soft-cap must be the checkpoint's.
        config: The checkpoint's decision_config.json.
        rendered: The request the rows answer, as `render` rendered it for the checkpoint's
            prompt.
        stage: What a row is: `hidden`, `raw_logits`, `capped_logits`, `final_logits` or
            `probabilities` (`STAGES`).

    Returns:
        For each question, by name in `rendered.questions` order, the probabilities of its
        k codes in code order, [k].

    Raises:
        ValueError: When the stage or the readout does not fit the checkpoint
            (`check_stage`), `rendered` was rendered for another prompt or tokenizer
            (`check_rendered`), or a row is missing or has the wrong shape or values for
            its stage (`code_probabilities`).
    """
    check_stage(stage, readout, config)
    check_rendered(config, rendered)
    c = config.final_logit_softcapping
    names = list(rendered.questions)
    rows = _rows(values, names)
    if stage == HIDDEN:
        assert readout is not None and readout.weight is not None  # check_stage
        for n, r in zip(names, rows, strict=True):
            if r.shape != (readout.weight.shape[1],):
                raise ValueError(
                    f"{n!r}: expected a hidden state of {readout.weight.shape[1]}, not {r.shape}"
                )
        rows = list(readout_logits(readout.weight, np.stack(rows)))
        stage = RAW_LOGITS
    out: dict[str, np.ndarray] = {}
    for n, row in zip(names, rows, strict=True):
        k = len(rendered.questions[n].codes)
        if row.shape not in ((len(config.codes),), (k,)):
            raise ValueError(
                f"{n!r}: expected {len(config.codes)} values (every code) or {k} (the "
                f"question's), not shape {row.shape}"
            )
        out[n] = code_probabilities(
            row, k, stage=stage, softcap_c=c, temperature=config.temperature
        )
    return out


def read(
    values: Any,
    readout: Readout | None,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    stage: str,
    ordinal_smoothing: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Every question's answer in v1's response schema: `question_probabilities`, then
    `answer`.

    Args:
        values: One row per question, of the stage `stage`, as `question_probabilities`
            takes them: an array [questions, width], a sequence of rows, or a mapping from
            question name to row.
        readout: The checkpoint's readout. `hidden` rows need its `weight`; a runtime that
            applies the readout itself and returns a later stage may pass None.
        config: The checkpoint's decision_config.json.
        rendered: The request the rows answer, as `render` rendered it for the checkpoint's
            prompt.
        stage: What a row is: `hidden`, `raw_logits`, `capped_logits`, `final_logits` or
            `probabilities` (`STAGES`).
        ordinal_smoothing: Passed to `answer`, for scores' confidence; 0 for none.

    Returns:
        For each question, by name in `rendered.questions` order, its answer as JSON: a
        `NoulAnswer`, `ChoiceAnswer` or `ScoreAnswer` of `strands_decider.schema`.

    Raises:
        ValueError: As `question_probabilities` raises it.
    """
    probs = question_probabilities(values, readout, config, rendered, stage=stage)
    return {
        n: answer(rendered.questions[n], p.tolist(), ordinal_smoothing=ordinal_smoothing)
        for n, p in probs.items()
    }
