"""Decider inference without torch: tokenizer -> libllama (ctypes shim) -> pointer head.

The prompt rendering and the answer readouts are the package's own
(`strands_decider.prompting` and `strands_decider.schema`, which import neither torch
nor transformers), so the prompts match `SystemOneEngine` exactly. Only the torso
forward is different: it runs through llama.cpp on a Q8_0 GGUF, and the pointer head
runs in numpy.

The state is decoded once per request and its memory is copied to each question, so a
request costs about the state's tokens plus each question's own tokens.
"""

from __future__ import annotations

import ctypes
import json
import os

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from strands_decider.prompting import (
    RenderedQuestion,
    read_choice,
    read_noul,
    read_score,
    render_question,
    render_state,
    score_legend,
)
from strands_decider.schema import (
    SystemOneRequest,
    derive_confidence,
    derive_score_confidence,
)

MASK_VALUE = -1e4
_F = ctypes.POINTER(ctypes.c_float)
_I = ctypes.POINTER(ctypes.c_int)


class _Shim:
    """ctypes binding for shim/decider_hidden.c."""

    def __init__(self, lib_path: str, model_path: str, threads: int):
        lib = ctypes.CDLL(lib_path)
        lib.dh_load.argtypes = [ctypes.c_char_p, ctypes.c_int, _I]
        lib.dh_load.restype = ctypes.c_void_p
        lib.dh_forward.argtypes = [ctypes.c_void_p, _I, ctypes.c_int, _F]
        lib.dh_forward.restype = ctypes.c_int
        lib.dh_forward_shared.argtypes = [
            ctypes.c_void_p,
            _I,
            ctypes.c_int,
            _I,
            _I,
            ctypes.c_int,
            ctypes.c_int,
            _F,
        ]
        lib.dh_forward_shared.restype = ctypes.c_int
        n_embd = ctypes.c_int(0)
        self.handle = lib.dh_load(model_path.encode(), threads, ctypes.byref(n_embd))
        if not self.handle:
            raise RuntimeError(f"dh_load failed for {model_path}")
        self.lib = lib
        self.n_embd = n_embd.value

    def forward(self, ids: list[int]) -> np.ndarray:
        """Hidden states for one full prompt, [len(ids), n_embd]."""
        out = np.empty((len(ids), self.n_embd), dtype=np.float32)
        rc = self.lib.dh_forward(
            self.handle, (ctypes.c_int * len(ids))(*ids), len(ids), out.ctypes.data_as(_F)
        )
        if rc != 0:
            raise RuntimeError(f"dh_forward rc={rc}")
        return out

    def forward_shared(self, state_ids: list[int], q_ids: list[list[int]]) -> list[np.ndarray]:
        """Decode the state once; return each question's suffix hidden states."""
        flat = [t for q in q_ids for t in q]
        lens = [len(q) for q in q_ids]
        out = np.empty((len(flat), self.n_embd), dtype=np.float32)
        rc = self.lib.dh_forward_shared(
            self.handle,
            (ctypes.c_int * len(state_ids))(*state_ids),
            len(state_ids),
            (ctypes.c_int * len(flat))(*flat),
            (ctypes.c_int * len(lens))(*lens),
            len(lens),
            0,
            out.ctypes.data_as(_F),
        )
        if rc != 0:
            raise RuntimeError(f"dh_forward_shared rc={rc}")
        return np.split(out, np.cumsum(lens)[:-1])


class DeciderEngine:
    def __init__(
        self,
        lib_path: str,
        gguf_path: str,
        assets_dir: str,
        threads: int,
        max_request_tokens: int | None = None,
    ):
        self.shim = _Shim(lib_path, gguf_path, threads)
        # Total tokens one request may forward (the state once, plus every question).
        # Time grows with it and the function has a timeout; None: no limit.
        self.max_request_tokens = max_request_tokens
        self.n_embd = self.shim.n_embd
        self.tok = Tokenizer.from_file(os.path.join(assets_dir, "tokenizer.json"))
        # Count untruncated tokens and reject over-length prompts: the tokenizer's own
        # truncation would silently cut off the question and its options.
        self.tok.no_truncation()
        self.head = {
            k: v.astype(np.float64)
            for k, v in load_file(os.path.join(assets_dir, "head.safetensors")).items()
        }
        with open(os.path.join(assets_dir, "head_config.json")) as f:
            cfg = json.load(f)
        self.max_tokens = int(cfg["max_length"])
        self.scale = int(cfg["pointer_dim"]) ** -0.5
        self.temperature = float(cfg["temperature"])
        self.temperature_by_kind = cfg.get("temperature_by_kind") or {}
        self.ordinal_smoothing = float(cfg.get("ordinal_smoothing", 0.0))

    # ------------------------------------------------------------- readout
    def _layernorm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * self.head["norm.weight"] + self.head["norm.bias"]

    def _probs(self, rq: RenderedQuestion, hidden: np.ndarray, opos: list[int]) -> list[float]:
        """PointerHead: query from the <answer> token (the last row), keys from the last
        token of each option line, then the calibrated softmax."""
        h = self.head
        q = self._layernorm(hidden[-1]) @ h["q.weight"].T + h["q.bias"]
        k = self._layernorm(hidden[opos]) @ h["k.weight"].T + h["k.bias"]
        logits = (
            (k @ q) * self.scale / float(self.temperature_by_kind.get(rq.kind, self.temperature))
        )
        logits[rq.n_slots :] = MASK_VALUE
        e = np.exp(logits - logits.max())
        return (e / e.sum()).tolist()

    def _answer(self, rq: RenderedQuestion, probs: list[float]) -> dict:
        if rq.kind == "noul":
            return {"type": "noul", "noul": round(read_noul(probs, rq), 4)}
        if rq.kind == "choice":
            by = read_choice(probs, rq)
            return {
                "type": "choice",
                "choice": max(by, key=by.__getitem__),
                "probabilities": {k: round(v, 4) for k, v in by.items()},
                "confidence": round(derive_confidence(list(by.values())), 4),
            }
        expected, ordered = read_score(probs, rq)
        return {
            "type": "score",
            "score": round(expected, 4),
            "legend": score_legend(rq),
            "probabilities": {k: round(v, 4) for k, v in ordered.items()},
            "confidence": round(
                derive_score_confidence(
                    list(ordered.values()), ordinal_smoothing=self.ordinal_smoothing
                ),
                4,
            ),
        }

    @staticmethod
    def _option_rows(offsets: list[tuple[int, int]], spans, base: int) -> list[int]:
        """Index of the last token inside each option line. `base` shifts the
        question-relative spans into the tokenized text's character coordinates."""
        rows = []
        for s, e in spans:
            inside = [
                j
                for j, (lo, hi) in enumerate(offsets)
                if hi > lo and lo >= base + s and hi <= base + e
            ]
            if not inside:
                raise ValueError(f"option span ({base + s},{base + e}) has no tokens")
            rows.append(inside[-1])
        return rows

    def _check_length(self, n_tokens: int) -> None:
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"prompt is {n_tokens} tokens (state + question); the model's window is "
                f"{self.max_tokens}. Shorten the state by at least "
                f"{n_tokens - self.max_tokens} tokens."
            )

    def _check_budget(self, n_state: int, q_lens: list[int]) -> None:
        n_total = n_state + sum(q_lens)
        if self.max_request_tokens and n_total > self.max_request_tokens:
            raise ValueError(
                f"request is {n_total} tokens (state {n_state} + {len(q_lens)} questions "
                f"{sum(q_lens)}); this deployment takes at most {self.max_request_tokens} "
                f"per request (MAX_REQUEST_TOKENS). Split the questions across requests: "
                f"each pays the state's {n_state} tokens again."
            )

    # ------------------------------------------------------------- evaluate
    def evaluate(self, payload: dict, model_name: str) -> dict:
        req = SystemOneRequest.model_validate(payload)
        if req.images:
            raise ValueError("this deployment serves the text model; images are not supported")
        state_text = render_state(req.state)
        names = list(req.questions)
        rqs = [render_question(req.questions[n]) for n in names]

        s_ids = self.tok.encode(state_text, add_special_tokens=False).ids
        q_encs = [self.tok.encode(rq.text, add_special_tokens=False) for rq in rqs]
        self._check_length(len(s_ids) + max(len(q.ids) for q in q_encs))
        self._check_budget(len(s_ids), [len(q.ids) for q in q_encs])

        # Sharing the state is exact only if tokenizing state and question separately
        # gives the same ids as tokenizing them together. If any question breaks that,
        # fall back to one full-prompt forward per question.
        fulls = [self.tok.encode(state_text + rq.text, add_special_tokens=False) for rq in rqs]
        shared = all(f.ids == s_ids + q.ids for f, q in zip(fulls, q_encs, strict=True))

        answers = {}
        if shared:
            for name, rq, qe, h in zip(
                names,
                rqs,
                q_encs,
                self.shim.forward_shared(s_ids, [q.ids for q in q_encs]),
                strict=True,
            ):
                opos = self._option_rows(qe.offsets, rq.option_spans, 0)
                answers[name] = self._answer(rq, self._probs(rq, h.astype(np.float64), opos))
            n_input = len(s_ids) + sum(len(q.ids) for q in q_encs)
        else:
            for name, rq, fe in zip(names, rqs, fulls, strict=True):
                h = self.shim.forward(fe.ids).astype(np.float64)
                opos = self._option_rows(fe.offsets, rq.option_spans, len(state_text))
                answers[name] = self._answer(rq, self._probs(rq, h, opos))
            n_input = sum(len(f.ids) for f in fulls)

        return {
            "model": model_name,
            "answers": answers,
            "usage": {"input_tokens": n_input, "output_tokens": len(names)},
            "state_shared": shared,
        }
