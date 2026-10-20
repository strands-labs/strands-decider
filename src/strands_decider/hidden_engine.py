"""System One evaluation over a backend that returns the torso's final hidden states.

`SystemOneEngine` runs the torso and the pointer head in torch. This engine takes the
torso's output from a backend, runs the pointer head in numpy, and imports neither torch
nor transformers, so it serves where torch does not fit, such as llama.cpp on AWS Lambda.
Everything else is `SystemOneEngine`'s: the prompts and readouts come
from prompting.py, the state is fitted to the window as `_fit` fits it, and the head is
the same function of the same weights. tests/test_hidden_engine.py holds the two engines
to the same answers.

The backend is asked only for the rows the head reads: each option's last token and the
question's last token, `<answer>`. A backend may compute the rest or skip it.

Two differences, in what is counted rather than what is answered. A backend decodes the
state once for all the questions, so `usage.input_tokens` counts it once, where
`SystemOneEngine` counts it once per batch of `max_batch` questions. And the state is
fitted once for all the questions, where `SystemOneEngine` fits it per batch: with more
than `max_batch` questions and a state too long for the window, this engine trims the
state for the longest question of all.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from tokenizers import Tokenizer

from .prompting import option_token_index, render_question, render_state, to_answer
from .schema import Answer, SystemOneRequest, SystemOneResponse, Usage

if TYPE_CHECKING:
    from .modeling import StrandsDeciderModel


class Backend(Protocol):
    def forward(
        self, state: list[int], questions: list[list[int]], rows: list[list[int]]
    ) -> list[np.ndarray]:
        """Each question continues the state on its own, seeing no other question. For
        question i, return its final (post-norm) hidden states at rows[i], positions
        counted from the question's first token: an array [len(rows[i]), hidden]."""
        ...


class NumpyPointerHead:
    """modeling.PointerHead in numpy, from its state dict."""

    def __init__(self, weights: Mapping[str, np.ndarray], pointer_dim: int, eps: float = 1e-5):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.scale = pointer_dim**-0.5
        self.eps = eps

    def _norm(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + self.eps) * self.w["norm.weight"] + self.w["norm.bias"]

    def logits(self, decide: np.ndarray, options: np.ndarray) -> np.ndarray:
        """decide [hidden], options [K, hidden] -> logits [K]."""
        q = self._norm(decide) @ self.w["q.weight"].T + self.w["q.bias"]
        k = self._norm(options) @ self.w["k.weight"].T + self.w["k.bias"]
        out: np.ndarray = (k @ q) * self.scale
        return out


@dataclass
class HiddenEngineConfig:
    max_length: int
    temperature: float = 1.0
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    ordinal_smoothing: float = 0.0
    model_name: str = "strands-decider"
    # As EngineConfig.
    strict_window: bool = False
    max_question_fraction: float = 0.75
    # The most tokens one request may forward, the state once plus every question;
    # a larger request is refused before the forward. None: no limit.
    max_request_tokens: int | None = None


class HiddenStateEngine:
    def __init__(
        self,
        tokenizer: Tokenizer,
        head: NumpyPointerHead,
        config: HiddenEngineConfig,
        backend: Backend,
    ):
        self.tok = tokenizer
        # Count every token, as `_fit` does: a tokenizer.json may be saved with
        # truncation, or with padding, which pads a batch of questions to the longest.
        self.tok.no_truncation()
        self.tok.no_padding()
        # fit() sets the tokenizer's truncation for the state, which is not safe to run
        # from two threads at once.
        self._tok_lock = threading.Lock()
        self.head = head
        self.cfg = config
        self.backend = backend

    @classmethod
    def from_assets(cls, path: str, backend: Backend, **config: Any) -> HiddenStateEngine:
        """The engine from save_assets' directory; `config` sets HiddenEngineConfig fields."""
        from safetensors.numpy import load_file

        with open(os.path.join(path, "head_config.json"), encoding="utf-8") as f:
            saved = json.load(f)
        pointer_dim = saved.pop("pointer_dim")
        head = NumpyPointerHead(load_file(os.path.join(path, "head.safetensors")), pointer_dim)
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return cls(tok, head, HiddenEngineConfig(**{**saved, **config}), backend)

    def fit(
        self, state_text: str, question_texts: list[str]
    ) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
        """SystemOneEngine._fit: state ids, question ids and each question's offsets."""
        max_len = self.cfg.max_length
        with self._tok_lock:
            encs = self.tok.encode_batch(question_texts, add_special_tokens=False)
            q = [e.ids for e in encs]
            offs = [list(e.offsets) for e in encs]
            longest = max(len(x) for x in q)
            if self.cfg.strict_window:
                s = self.tok.encode(state_text, add_special_tokens=True).ids
                if len(s) + longest > max_len:
                    raise ValueError(
                        f"prompt of {len(s) + longest} tokens exceeds the context window "
                        f"of {max_len} tokens"
                    )
                return s, q, offs
            reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
            cut = [max(0, len(x) - reserve) for x in q]
            q = [x[c:] for x, c in zip(q, cut, strict=True)]
            offs = [o[c:] for o, c in zip(offs, cut, strict=True)]
            self.tok.enable_truncation(max_length=max(1, max_len - reserve))
            try:
                s = self.tok.encode(state_text, add_special_tokens=True).ids
            finally:
                self.tok.no_truncation()
        return s, q, offs

    def evaluate(self, request: SystemOneRequest | Mapping[str, Any]) -> SystemOneResponse:
        req = (
            request
            if isinstance(request, SystemOneRequest)
            else SystemOneRequest.model_validate(request)
        )
        if req.images:
            raise ValueError("this engine has no vision tower; images are not supported")
        names = list(req.questions)
        rendered = [render_question(req.questions[n]) for n in names]
        s, q, offs = self.fit(render_state(req.state), [rq.text for rq in rendered])
        n_input = len(s) + sum(len(x) for x in q)
        if self.cfg.max_request_tokens and n_input > self.cfg.max_request_tokens:
            raise ValueError(
                f"request of {n_input} tokens (the state's {len(s)} once, then its questions) "
                f"exceeds the limit of {self.cfg.max_request_tokens} tokens per request"
            )
        # Before the forward, so a question trimmed into its options is refused at once.
        options = [
            option_token_index(o, rq.option_spans, 0) for rq, o in zip(rendered, offs, strict=True)
        ]
        rows = [[*opts, len(x) - 1] for opts, x in zip(options, q, strict=True)]
        hidden = self.backend.forward(s, q, rows)
        answers: dict[str, Answer] = {}
        for name, rq, h in zip(names, rendered, hidden, strict=True):
            h = np.asarray(h, dtype=np.float64)
            t = float(self.cfg.temperature_by_kind.get(rq.kind, self.cfg.temperature))
            logits = self.head.logits(h[-1], h[:-1]) / max(t, 1e-6)
            p = np.exp(logits - np.max(logits))
            answers[name] = to_answer(
                rq, (p / p.sum()).tolist(), ordinal_smoothing=self.cfg.ordinal_smoothing
            )
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=n_input, output_tokens=len(names)),
        )


def save_assets(model: StrandsDeciderModel, path: str, model_name: str) -> None:
    """Write what from_assets reads: the tokenizer, the pointer head and its settings
    (tokenizer.json, head.safetensors, head_config.json). The torso is the backend's."""
    from safetensors.numpy import save_file

    cfg = model.config
    if cfg.head_type != "pointer":
        raise ValueError(f"the hidden-state engine reads a pointer head, not {cfg.head_type!r}")
    os.makedirs(path, exist_ok=True)
    model.tokenizer.backend_tokenizer.save(os.path.join(path, "tokenizer.json"))
    weights = {k: v.detach().float().cpu().numpy() for k, v in model.head.state_dict().items()}
    save_file(weights, os.path.join(path, "head.safetensors"))
    saved = {
        "model_name": model_name,
        "pointer_dim": cfg.pointer_dim,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "temperature_by_kind": cfg.temperature_by_kind,
        "ordinal_smoothing": cfg.ordinal_smoothing,
    }
    with open(os.path.join(path, "head_config.json"), "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
