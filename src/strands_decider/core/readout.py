"""From the hidden state at a question's last token to its answer, in the training's precision:

z = bf16(W bf16(h))      W: readout.safetensors `weight`, [255 codes, hidden], bf16
z = tanh(z / c) * c / T  float32; c and T: decision_config.json's soft-cap and temperature
p = softmax(z[:k])       float32, over the question's own k codes
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt

from ..confidence import derive_confidence, derive_score_confidence
from .codes import CODES
from .config import DecisionConfig
from .prompt import RenderedQuestion, RenderedRequest

READOUT_FILE = "readout.safetensors"

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


def round_to_bf16(x: npt.ArrayLike) -> np.ndarray:
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
    e = np.exp(s - np.max(s))
    out: np.ndarray = e / e.sum(dtype=np.float32)
    return out


def load_readout(checkpoint: str | os.PathLike[str]) -> np.ndarray:
    """The readout weight [255 codes, hidden] from a checkpoint's readout.safetensors."""
    w = load_safetensors(os.path.join(os.fspath(checkpoint), READOUT_FILE))["weight"]
    if w.ndim != 2 or w.shape[0] != len(CODES):
        raise ValueError(f"readout weight must be [{len(CODES)}, hidden], not {list(w.shape)}")
    return w


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
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    weight: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Each question's k code probabilities, from one row per question in `rendered` order. With
    the readout `weight`, the rows are hidden states. Without it, they are probabilities over all
    255 codes or the question's own k, from a runtime that applies the readout itself."""
    names = list(rendered.questions)
    if len(rows) != len(names):
        raise ValueError(f"expected {len(names)} rows, one per question, not {len(rows)}")
    if weight is not None:
        rows = list(readout_logits(weight, np.asarray(rows)))
    out = {}
    for name, row in zip(names, rows, strict=True):
        v = np.asarray(row)
        if weight is None and v.shape[-1] > len(CODES):
            raise ValueError("rows wider than 255 are hidden states: pass the readout weight")
        v = v[: len(rendered.questions[name].codes)]
        if weight is None:
            v = v.astype(np.float64)
            out[name] = v / v.sum()
        else:
            out[name] = code_probabilities(v, config.final_logit_softcapping, config.temperature)
    return out


def read(
    rows: np.ndarray | list[np.ndarray],
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    weight: np.ndarray | None = None,
    ordinal_smoothing: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Every question's answer, from the rows `question_probabilities` takes."""
    probs = question_probabilities(rows, config, rendered, weight=weight)
    return {
        n: answer(rendered.questions[n], p.tolist(), ordinal_smoothing=ordinal_smoothing)
        for n, p in probs.items()
    }
