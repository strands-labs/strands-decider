"""From the hidden state at a question's last token to its answer, in the training's precision:

z = bf16(W bf16(h))      W: readout.safetensors `weight`, [255 codes, hidden], bf16
z = tanh(z / c) * c / T  float32; c and T: decision_config.json's soft-cap and temperature
p = softmax(z[:k])       float32, over the question's own k codes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..confidence import derive_confidence, derive_score_confidence
from .config import DecisionConfig
from .prompt import RenderedQuestion, RenderedRequest

READOUT_FILE = "readout.safetensors"

# What a backend's `forward` returns: the hidden state at the last token, or (a runtime that
# applies the readout itself, like the vLLM plugin) the codes' probabilities.
HIDDEN = "hidden"
PROBABILITIES = "probabilities"
STAGES = (HIDDEN, PROBABILITIES)

_DTYPES = {"F32": "<f4", "F16": "<f2"}


def load_safetensors(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file as float32. numpy has no bf16, so a BF16 tensor's bits
    become the high half of a float32, which holds every bf16 value exactly."""
    from safetensors import deserialize  # imported here: the core imports without it

    with open(path, "rb") as f:
        tensors = deserialize(f.read())
    out = {}
    for name, t in tensors:
        if t["dtype"] == "BF16":
            arr = (np.frombuffer(t["data"], "<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(t["data"], _DTYPES[t["dtype"]]).astype(np.float32)
        out[name] = arr.reshape(t["shape"])
    return out


def round_to_bf16(x: Any) -> np.ndarray:
    """`x` rounded to the nearest bf16 (ties to even), held as float32."""
    a: np.ndarray = np.asarray(x, dtype=np.float32)
    u: np.ndarray = np.atleast_1d(a).view(np.uint32).astype(np.uint64)  # the add can't wrap
    u = (u + ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)) & np.uint64(0xFFFF0000)
    rounded: np.ndarray = u.astype(np.uint32).view(np.float32).reshape(a.shape)
    out: np.ndarray = np.where(np.isnan(a), np.float32(np.nan), rounded).astype(np.float32)
    return out


def readout_logits(weight: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """z = W h as a bf16 GEMM: both rounded to bf16, the products summed in float64, the sum
    rounded to float32 and then bf16. A GPU sums in its own order, so z can differ by one bf16
    step where a sum falls on a rounding boundary."""
    w = round_to_bf16(weight).astype(np.float64)
    h = round_to_bf16(hidden).astype(np.float64)
    return round_to_bf16((h @ w.T).astype(np.float32))


def code_probabilities(z: np.ndarray, softcap: float, temperature: float) -> np.ndarray:
    """softmax(tanh(z / c) * c / T) in float32, over the logits given (a question's k)."""
    c, t = np.float32(softcap), np.float32(temperature)
    s = np.tanh(np.asarray(z, dtype=np.float32) / c) * c / t
    e = np.exp(s - s.max())
    out: np.ndarray = e / e.sum(dtype=np.float32)
    return out


@dataclass(frozen=True, eq=False)
class Readout:
    """The readout's `weight` [255, hidden] (None for a runtime that applies it itself) and the
    soft-cap c."""

    weight: np.ndarray | None
    softcap: float

    @classmethod
    def load(
        cls, checkpoint: str | os.PathLike[str], config: DecisionConfig | None = None
    ) -> Readout:
        d = os.fspath(checkpoint)
        cfg = config if config is not None else DecisionConfig.load(d)
        w = load_safetensors(os.path.join(d, READOUT_FILE))["weight"]
        if w.shape[0] != len(cfg.codes):
            raise ValueError(f"readout weight must be [{len(cfg.codes)}, hidden], not {w.shape}")
        return cls(weight=w, softcap=cfg.final_logit_softcapping)


def check_stage(stage: str, readout: Readout | None) -> None:
    """That rows of `stage` can be read: hidden states need the readout weight."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
    if stage == HIDDEN and (readout is None or readout.weight is None):
        raise ValueError("hidden states need the readout weight")


def answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> dict[str, Any]:
    """One answer in the System One response schema, v1's, keyed by the request's option names in
    the request's order and rounded to 4 places as v1 rounds. `ordinal_smoothing` corrects a score's
    confidence for training that spread each target over neighbouring levels."""
    by_option = dict(zip(rq.options, probs, strict=True))
    if rq.kind == "noul":
        return {"type": "noul", "noul": round(by_option["true"], 4)}
    if rq.kind == "choice":
        named = {o: by_option[o] for o in rq.request_options}
        return {
            "type": "choice",
            "choice": max(named, key=lambda o: named[o]),
            "probabilities": {o: round(v, 4) for o, v in named.items()},
            "confidence": round(derive_confidence(list(named.values())), 4),
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


def question_probabilities(
    rows: np.ndarray | list[np.ndarray],
    readout: Readout | None,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    stage: str,
) -> dict[str, np.ndarray]:
    """Each question's k code probabilities, from one row per question in `rendered` order: its
    hidden state, or its probabilities over all 255 codes or its own k."""
    check_stage(stage, readout)
    names = list(rendered.questions)
    if len(rows) != len(names):
        raise ValueError(f"expected {len(names)} rows, one per question, not {len(rows)}")
    if stage == HIDDEN:
        assert readout is not None and readout.weight is not None
        rows = list(readout_logits(readout.weight, np.stack(rows)))
    out = {}
    for name, row in zip(names, rows, strict=True):
        v = np.asarray(row)[: len(rendered.questions[name].codes)]
        if stage == PROBABILITIES:
            v = v.astype(np.float64)
            out[name] = v / v.sum()
        else:
            out[name] = code_probabilities(v, config.final_logit_softcapping, config.temperature)
    return out


def read(
    rows: np.ndarray | list[np.ndarray],
    readout: Readout | None,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    stage: str,
    ordinal_smoothing: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Every question's answer, from the rows `question_probabilities` takes."""
    probs = question_probabilities(rows, readout, config, rendered, stage=stage)
    return {
        n: answer(rendered.questions[n], p.tolist(), ordinal_smoothing=ordinal_smoothing)
        for n, p in probs.items()
    }
