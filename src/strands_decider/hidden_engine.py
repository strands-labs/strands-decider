"""A System One engine that runs without torch: a backend runs the torso, numpy the head.

`SystemOneEngine` runs the whole model in torch. `HiddenStateEngine` gets the torso's final
hidden states from a backend, such as llama.cpp on AWS Lambda, and does everything else
itself, with the same code as `SystemOneEngine`: it renders the prompts, fits them to the
window and reads the answers with the functions in prompting.py, and runs the same pointer
head, with the same weights, in numpy. It imports neither torch nor transformers.
tests/test_hidden_engine.py checks that the two engines give the same answers.

The engine asks the backend only for the hidden states the head reads: at each option's
last token, and at the question's last token, `<answer>`.

The answers are the same; two things around them can differ:

- `usage.input_tokens`. This engine counts the state once, then every question.
  `SystemOneEngine` counts the state once per question on its batched path (with
  `use_prefix_cache=False`, or when its cache cannot be forked), and once per chunk of up
  to `max_batch` questions on its shared-prefix path. For a request with one question,
  all three counts agree.
- Which part of a long state is kept. `SystemOneEngine` fits the state to the window
  separately for each chunk of questions, this engine once for all of them. With more
  questions than one chunk holds and a state too long for the window, this engine trims
  the state to leave room for the longest question in the whole request.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from tokenizers import Tokenizer

from .prompting import fit_window, option_token_index, render_question, render_state, to_answer
from .schema import Answer, SystemOneRequest, SystemOneResponse, Usage

if TYPE_CHECKING:
    from torch import nn

    from .modeling import StrandsDeciderModel

# The layout of save_assets' head_config.json. A newer layout adds keys, which an older
# engine ignores.
ASSETS_FORMAT = 1


class Backend(Protocol):
    def forward(
        self, state: list[int], questions: list[list[int]], rows: list[list[int]]
    ) -> list[np.ndarray]:
        """Run each question as a continuation of the state, seeing no other question.
        For question i, return its final (post-norm) hidden states at the positions in
        rows[i], counted from the question's first token, in the order of rows[i] (each
        option's last token, then `<answer>`): an array [len(rows[i]), hidden].

        The engine may call forward from several threads at once; a backend that cannot
        run calls concurrently serializes them itself. A backend whose answers can depend
        on what it batches together should say so. Later backends may declare
        capabilities such as these, and media support, through an optional attribute."""
        ...


class NumpyPointerHead:
    """modeling.PointerHead in numpy, from its state dict."""

    def __init__(self, weights: Mapping[str, np.ndarray], pointer_dim: int, eps: float):
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
    # The side a long state loses tokens from, as the Hugging Face tokenizer's
    # `truncation_side` sets it for SystemOneEngine.
    truncation_side: str = "right"
    # As in EngineConfig.
    strict_window: bool = False
    max_question_fraction: float = 0.75
    # The most tokens one request may run through the backend, the state once plus every
    # question; a larger request is refused before the backend runs. None: no limit.
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
        # fit_window changes the tokenizer's truncation, so two threads must not run it at
        # once.
        self._tok_lock = threading.Lock()
        self.head = head
        self.cfg = config
        self.backend = backend

    @classmethod
    def from_assets(cls, path: str, backend: Backend, **config: Any) -> HiddenStateEngine:
        """The engine from a save_assets directory. `config` sets HiddenEngineConfig
        fields, over the saved ones."""
        from safetensors.numpy import load_file

        with open(os.path.join(path, "head_config.json"), encoding="utf-8") as f:
            saved = json.load(f)
        if saved.pop("readout", "pointer") != "pointer":
            raise ValueError(f"{path}: the assets are not for a pointer head")
        if saved.pop("format", ASSETS_FORMAT) > ASSETS_FORMAT:
            warnings.warn(
                f"{path}: assets from a newer strands-decider; settings this version does "
                "not know are ignored",
                stacklevel=2,
            )
        weights = load_file(os.path.join(path, "head.safetensors"))
        head = NumpyPointerHead(weights, saved.pop("pointer_dim"), saved.pop("eps", 1e-5))
        known = {f.name for f in dataclasses.fields(HiddenEngineConfig)}
        settings = {k: v for k, v in saved.items() if k in known}
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return cls(tok, head, HiddenEngineConfig(**{**settings, **config}), backend)

    def fit(
        self, state_text: str, question_texts: list[str]
    ) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
        """prompting.fit_window with this engine's settings, as SystemOneEngine._fit runs it
        with its own: state ids, question ids and each question's character offsets."""
        with self._tok_lock:
            return fit_window(
                self.tok,
                state_text,
                question_texts,
                self.cfg.max_length,
                strict=self.cfg.strict_window,
                max_question_fraction=self.cfg.max_question_fraction,
                truncation_side=self.cfg.truncation_side,
            )

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
        # Found before the backend runs, so a question whose options were trimmed away is
        # refused without the work.
        options = [
            option_token_index(o, rq.option_spans, 0) for rq, o in zip(rendered, offs, strict=True)
        ]
        rows = [[*opts, len(x) - 1] for opts, x in zip(options, q, strict=True)]
        hidden = self.backend.forward(s, q, rows)
        if len(hidden) != len(names):
            raise ValueError(f"backend returned {len(hidden)} arrays for {len(names)} questions")
        answers: dict[str, Answer] = {}
        for name, rq, h, r in zip(names, rendered, hidden, rows, strict=True):
            h = np.asarray(h, dtype=np.float64)
            if h.ndim != 2 or h.shape[0] != len(r):
                raise ValueError(
                    f"backend returned shape {h.shape} for question {name!r}; expected "
                    f"({len(r)}, hidden): one row per option, then <answer>"
                )
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
    os.makedirs(path, exist_ok=True)
    model.tokenizer.backend_tokenizer.save(os.path.join(path, "tokenizer.json"))
    weights = {k: v.detach().float().cpu().numpy() for k, v in model.head.state_dict().items()}
    save_file(weights, os.path.join(path, "head.safetensors"))
    saved = {
        "format": ASSETS_FORMAT,
        "readout": "pointer",
        "model_name": model_name,
        "pointer_dim": cfg.pointer_dim,
        "eps": cast("nn.LayerNorm", model.head.norm).eps,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "temperature_by_kind": cfg.temperature_by_kind,
        "ordinal_smoothing": cfg.ordinal_smoothing,
        "truncation_side": model.tokenizer.truncation_side,
    }
    with open(os.path.join(path, "head_config.json"), "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
