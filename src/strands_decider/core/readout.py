"""From the hidden state at a question's last token to its answer (docs/spec.md, The readout).

    z = bf16(W h)              W: readout.safetensors `weight` in bf16; a bf16 GEMM
    z = tanh(z / c) * c        float32; c: decision_config.json `final_logit_softcapping`
    z = z / T                  float32; T: decision_config.json `temperature`
    p = softmax(z[:k])         float32, over the question's own k codes only
    p -> the answer            v1's response schema and confidence formulas

This is the training reference's precision regime, and that of strands-decider-vllm, the vLLM
runtime.
numpy has no bf16, so the GEMM is emulated: the products are summed in float64, the sum is
rounded to float32 and then to bf16 by the float32's bits, round to nearest, ties to even
(`round_to_bf16`). The hidden state is used as the decoder returns it.

A runtime may hand over any stage between the hidden state and the probabilities. The stage
is always named (`inputs`, `Backend.outputs`), so neither the soft-cap nor the temperature
is ever applied twice: `hidden`, `raw_logits` (W h), `capped_logits` (the cap applied),
`final_logits` (the cap and the temperature applied) or `probabilities`.
"""

from __future__ import annotations

import json
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
FINAL_LOGITS = "final_logits"  # tanh(z / c) * c / T: both (as strands-decider-vllm returns them)
PROBABILITIES = "probabilities"  # softmax of the final logits, over the k codes or all 255
STAGES = (HIDDEN, RAW_LOGITS, CAPPED_LOGITS, FINAL_LOGITS, PROBABILITIES)

_DTYPES: dict[str, np.dtype[Any]] = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
}


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file as float32: F32, F16 and BF16, without the
    safetensors package (an 8-byte length, a JSON header, then the raw little-endian data)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError(f"{path}: not a safetensors file")
    (n,) = struct.unpack("<Q", data[:8])
    if n > len(data) - 8:
        raise ValueError(f"{path}: header longer than the file")
    header = json.loads(data[8 : 8 + n])
    body = memoryview(data)[8 + n :]
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in _DTYPES:
            raise ValueError(
                f"{path}: tensor {name!r} has dtype {dtype}; expected F32, F16 or BF16"
            )
        lo, hi = info["data_offsets"]
        shape = tuple(info["shape"])
        raw = np.frombuffer(body[lo:hi], dtype=_DTYPES[dtype])
        if raw.size != int(np.prod(shape)):
            raise ValueError(f"{path}: tensor {name!r} does not match its shape {shape}")
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
    a = np.asarray(x, dtype=np.float32)
    u = np.atleast_1d(a).view(np.uint32).astype(np.uint64)  # 64 bits: the add cannot wrap
    u = (u + ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)) & np.uint64(0xFFFF0000)
    out = u.astype(np.uint32).view(np.float32).reshape(a.shape)
    out = np.where(np.isnan(a), np.float32(np.nan), out).astype(np.float32)
    return out


def readout_logits(weight: Any, hidden: Any) -> np.ndarray:
    """z = W h as a bf16 GEMM: `weight` [n_codes, hidden] rounded to bf16, `hidden` ([hidden]
    or [rows, hidden]) as given, the products summed in float64, and each sum rounded to
    float32 and then to bf16. Returns float32, [n_codes] or [rows, n_codes].

    A GPU's bf16 GEMM sums in float32, in its own order, so where a sum falls next to a bf16
    rounding boundary its z can differ from this one by one bf16 step.
    """
    w = round_to_bf16(weight).astype(np.float64)
    h = np.asarray(hidden, dtype=np.float64)
    if w.ndim != 2 or h.ndim not in (1, 2) or h.shape[-1] != w.shape[1]:
        raise ValueError(f"readout weight {w.shape} cannot take hidden states {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("hidden states must be finite")
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
    inputs: str,
    softcap_c: float | None,
    temperature: float,
) -> np.ndarray:
    """The probabilities of the first `k` codes, from one row of the stage `inputs`.

    A logit row (`raw_logits`, `capped_logits` or `final_logits`) holds the 255 codes in code
    order, or only the question's first k. `raw_logits` are rounded to bf16 (a bf16 GEMM's
    output is unchanged by it), then capped; `capped_logits` are divided by the temperature;
    all three end in a float32 softmax over the k codes, whose other codes' logits do not
    enter it. A `probabilities` row (the k codes or all 255, as a softmax over them gives)
    is renormalised over the k codes. Capped or final logits past the cap's range are
    refused: they cannot be what their stage says. Returns [k], float32 (float64 for
    `probabilities`).
    """
    if inputs not in STAGES or inputs == HIDDEN:
        raise ValueError(
            f"inputs must be one of {STAGES[1:]} here, not {inputs!r} (hidden states need "
            "readout_logits first)"
        )
    t = np.float32(check_float32_scale("temperature", temperature))
    with np.errstate(over="ignore"):  # a float64 value past float32's range becomes inf
        v = np.asarray(values, dtype=np.float64 if inputs == PROBABILITIES else np.float32)
    if v.ndim != 1 or not 1 <= k <= v.shape[0]:
        raise ValueError(f"need a vector of at least {k} values, not shape {v.shape}")
    v = v[:k]
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{inputs} must be finite in float32: {v[~np.isfinite(v)][:3]}")
    if inputs == PROBABILITIES:
        if np.any(v < 0) or np.any(v > 1 + 1e-6) or v.sum() <= 0:
            raise ValueError("probabilities must lie in [0, 1], and not all be 0")
        p: np.ndarray = v / v.sum()
        return p
    if inputs == RAW_LOGITS:
        s = softcap(round_to_bf16(v), softcap_c)
    elif softcap_c is not None:
        bound = np.float32(softcap_c) / (t if inputs == FINAL_LOGITS else np.float32(1))
        if np.any(np.abs(v) > bound * np.float32(1 + 1e-6)):
            raise ValueError(
                f"{inputs} exceed the soft-cap's range (|z| <= {bound:g}); were they capped?"
            )
        s = v
    else:
        s = v
    if inputs != FINAL_LOGITS:
        with np.errstate(over="ignore"):
            s = s / t
        if not np.all(np.isfinite(s)):
            raise ValueError(f"logits divided by the temperature {temperature!r} overflow float32")
    return _softmax32(s.astype(np.float32))


# ---- the readout file -------------------------------------------------------------------------

_ABSENT = object()


def _softcap_from(model_config: Mapping[str, Any]) -> Any:
    """`final_logit_softcapping`, top level (Gemma4TextModel) or under `text_config`, or
    _ABSENT."""
    for d in (model_config, model_config.get("text_config") or {}):
        if "final_logit_softcapping" in d:
            return d["final_logit_softcapping"]
    return _ABSENT


@dataclass(frozen=True, eq=False)
class Readout:
    """The readout: `weight` [n_codes, hidden], held as float32 values rounded to bf16, and
    the soft-cap `c` (None: no cap).

    `weight` may be None for a runtime that applies the readout itself and passes logits or
    probabilities.
    """

    weight: np.ndarray | None
    softcap: float | None

    def __post_init__(self) -> None:
        if self.weight is not None:
            w = round_to_bf16(self.weight)
            if w.ndim != 2:
                raise ValueError(f"readout weight must be [codes, hidden], not {w.shape}")
            object.__setattr__(self, "weight", w)
        if self.softcap is not None:
            object.__setattr__(self, "softcap", check_float32_scale("softcap", self.softcap))

    @classmethod
    def load(
        cls, checkpoint: str | os.PathLike[str], config: DecisionConfig | None = None
    ) -> Readout:
        """From a checkpoint directory: `weight` from readout.safetensors, and the soft-cap
        from decision_config.json's `final_logit_softcapping` (`config`, if given, else the
        directory's file). A converted artifact needs no config.json; if the directory has
        one that names a different soft-cap, the checkpoint is inconsistent and refused."""
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
                theirs = _softcap_from(json.load(f))
            if theirs is not _ABSENT and theirs != softcap:
                raise ValueError(
                    f"{MODEL_CONFIG_FILE} has final_logit_softcapping {theirs!r} and "
                    f"{FILENAME} has {softcap!r}; the checkpoint is inconsistent"
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


def question_probabilities(
    values: Any,
    readout: Readout | None,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    inputs: str,
) -> dict[str, np.ndarray]:
    """Each question's code probabilities, in code order ([k] per question), keyed by
    question name: section 7 of docs/spec.md from the stage `inputs` on.

    `values` holds one row per question, in `rendered.questions` order or keyed by question
    name. `inputs` names what a row is (`STAGES`) and is required, so that nothing is applied
    twice: `hidden` rows need `readout.weight`; the soft-cap is the checkpoint's
    (`config.final_logit_softcapping`) and the temperature `config.temperature`.
    """
    if inputs not in STAGES:
        raise ValueError(f"inputs must be one of {STAGES}, not {inputs!r}")
    if config.prompt != rendered.variant:
        raise ValueError(
            f"the request was rendered for {rendered.variant!r}, the checkpoint reads {config.prompt!r}"
        )
    c = config.final_logit_softcapping
    if readout is not None and readout.softcap != c:
        raise ValueError(
            f"the readout's soft-cap {readout.softcap!r} is not the checkpoint's "
            f"final_logit_softcapping {c!r}"
        )
    names = list(rendered.questions)
    rows = _rows(values, names)
    stage = inputs
    if inputs == HIDDEN:
        if readout is None or readout.weight is None:
            raise ValueError("hidden states need the readout weight")
        if readout.weight.shape[0] != len(config.codes):
            raise ValueError(
                f"readout has {readout.weight.shape[0]} rows for {len(config.codes)} codes"
            )
        for n, r in zip(names, rows, strict=True):
            if r.shape != (readout.weight.shape[1],):
                raise ValueError(
                    f"{n!r}: expected a hidden state of {readout.weight.shape[1]}, not {r.shape}"
                )
        rows = list(readout_logits(readout.weight, np.stack(rows)))
        stage = RAW_LOGITS
    out: dict[str, np.ndarray] = {}
    for n, row in zip(names, rows, strict=True):
        rq = rendered.questions[n]
        k = len(rq.codes)
        if tuple(config.token_ids[:k]) != rq.code_token_ids:
            raise ValueError(
                f"{n!r}: the tokenizer's code tokens do not match the checkpoint's token_ids"
            )
        if row.shape not in ((len(config.codes),), (k,)):
            raise ValueError(
                f"{n!r}: expected {len(config.codes)} values (every code) or {k} (the "
                f"question's), not shape {row.shape}"
            )
        out[n] = code_probabilities(
            row, k, inputs=stage, softcap_c=c, temperature=config.temperature
        )
    return out


def read(
    values: Any,
    readout: Readout | None,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    inputs: str,
    ordinal_smoothing: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Every question's answer, keyed by question name, in v1's response schema:
    `question_probabilities`, then `answer`. `inputs` (required) names the stage of the
    rows: `hidden`, `raw_logits`, `capped_logits`, `final_logits` or `probabilities`."""
    probs = question_probabilities(values, readout, config, rendered, inputs=inputs)
    return {
        n: answer(rendered.questions[n], p.tolist(), ordinal_smoothing=ordinal_smoothing)
        for n, p in probs.items()
    }
