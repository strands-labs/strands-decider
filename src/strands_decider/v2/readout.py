"""From the hidden state at a question's last token to its answer (docs/v2-spec.md, The readout).

    z = W h                    W: readout.safetensors `weight`, [255, hidden]
    z = c * tanh(z / c)        c: the model config's `final_logit_softcapping`
    z = z / temperature        decision_config.json
    p = softmax(z[:k])         the question's own k codes only
    p -> the answer            v1's response schema and confidence formulas

numpy only. Arithmetic is float64, as in v1's numpy head (`hidden_engine.NumpyPointerHead`).
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
from .config import DecisionConfig
from .prompt import RenderedQuestion, RenderedRequest

READOUT_FILE = "readout.safetensors"
MODEL_CONFIG_FILE = "config.json"

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


def _softcap_from(model_config: Mapping[str, Any]) -> float | None:
    """`final_logit_softcapping`, top level (Gemma4TextModel) or under `text_config`."""
    for d in (model_config, model_config.get("text_config") or {}):
        if "final_logit_softcapping" in d:
            c = d["final_logit_softcapping"]
            return None if c is None else float(c)
    raise ValueError("the model config has no final_logit_softcapping")


@dataclass(frozen=True)
class Readout:
    """The readout: `weight` [n_codes, hidden] and the soft-cap `c` (None: no cap).

    `weight` may be None for a runtime that computes `z = W h` itself and passes logits.
    """

    weight: np.ndarray | None
    softcap: float | None

    @classmethod
    def load(cls, checkpoint: str | os.PathLike[str]) -> Readout:
        """From a checkpoint directory: readout.safetensors and config.json."""
        d = os.fspath(checkpoint)
        tensors = load_safetensors(os.path.join(d, READOUT_FILE))
        if set(tensors) != {"weight"}:
            raise ValueError(
                f"{READOUT_FILE} must hold one tensor, `weight`; has {sorted(tensors)}"
            )
        w = tensors["weight"]
        if w.ndim != 2:
            raise ValueError(f"readout weight must be [codes, hidden], not {w.shape}")
        with open(os.path.join(d, MODEL_CONFIG_FILE), encoding="utf-8") as f:
            softcap = _softcap_from(json.load(f))
        return cls(weight=w, softcap=softcap)


def softcap(z: np.ndarray, c: float | None) -> np.ndarray:
    """`c * tanh(z / c)`: Gemma's final-logit soft-cap. None (or 0) leaves z as it is."""
    z = np.asarray(z, dtype=np.float64)
    if not c:
        return z
    out: np.ndarray = c * np.tanh(z / c)
    return out


def code_probabilities(
    logits: np.ndarray, k: int, *, softcap_c: float | None, temperature: float
) -> np.ndarray:
    """Probabilities over the first `k` codes from raw readout logits `z = W h` (all codes).

    The softmax runs over the question's k codes only; the other codes' logits do not enter
    it. The cap and the temperature act on each logit alone, so applying them to the k is
    the same as applying them to all 255.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or not 1 <= k <= z.shape[0]:
        raise ValueError(f"need a vector of at least {k} logits, not shape {z.shape}")
    s = softcap(z[:k], softcap_c) / float(temperature)
    e = np.exp(s - s.max())
    p: np.ndarray = e / e.sum()
    return p


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


def read(
    hidden_or_logits: Any,
    readout: Readout,
    config: DecisionConfig,
    rendered: RenderedRequest,
    *,
    inputs: str = "hidden",
    ordinal_smoothing: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Every question's answer, keyed by question name, in v1's response schema.

    `hidden_or_logits` holds one row per question, in `rendered.questions` order (an array
    [questions, width]) or keyed by question name. With `inputs="hidden"` a row is the
    final-norm hidden state at the question's last token; with `inputs="logits"` it is the
    raw readout logits `W h` over all codes, before the soft-cap.
    """
    if inputs not in ("hidden", "logits"):
        raise ValueError(f"inputs must be 'hidden' or 'logits', not {inputs!r}")
    if config.prompt != rendered.variant:
        raise ValueError(
            f"the request was rendered for {rendered.variant!r}, the checkpoint reads {config.prompt!r}"
        )
    names = list(rendered.questions)
    if isinstance(hidden_or_logits, Mapping):
        rows = [np.asarray(hidden_or_logits[n], dtype=np.float64) for n in names]
    else:
        arr = np.asarray(hidden_or_logits, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(names):
            raise ValueError(
                f"expected [{len(names)}, width] rows, one per question, not {arr.shape}"
            )
        rows = list(arr)
    if inputs == "hidden":
        if readout.weight is None:
            raise ValueError("hidden states need the readout weight")
        w = np.asarray(readout.weight, dtype=np.float64)
        if w.shape[0] != len(config.codes):
            raise ValueError(f"readout has {w.shape[0]} rows for {len(config.codes)} codes")
        logits = [w @ r for r in rows]
    else:
        logits = rows
    out: dict[str, dict[str, Any]] = {}
    for n, z in zip(names, logits, strict=True):
        rq = rendered.questions[n]
        k = len(rq.codes)
        if tuple(config.token_ids[:k]) != rq.code_token_ids:
            raise ValueError(
                f"{n!r}: the tokenizer's code tokens do not match the checkpoint's token_ids"
            )
        if z.shape != (len(config.codes),):
            raise ValueError(f"{n!r}: expected {len(config.codes)} logits, not shape {z.shape}")
        p = code_probabilities(z, k, softcap_c=readout.softcap, temperature=config.temperature)
        out[n] = answer(rq, p.tolist(), ordinal_smoothing=ordinal_smoothing)
    return out
