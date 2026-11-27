"""The Neuron backend: model checkpoints on AWS Trainium2 through TorchNeuron, text only.

The model is split over `tp` NeuronCores by tensor parallelism, one worker process per core
(`neuron_model.py`). This process renders nothing and touches no device: `forward` packs the
state and its questions into one prompt, pads it into a bucket, sends it to every rank, and
returns rank 0's rows. Each question attends to the state and to itself only, so its row is
what it would be alone (the `Backend` contract); the state is encoded once per pass.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core import NO_VISION, RenderedImage

# The split pass's buckets (states longer than joint_max): state tokens, question tokens (all
# of a pass's questions, packed) and questions.
STATE_BUCKETS = (
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
    10240,
    12288,
    14336,
    16384,
    20480,
    24576,
    32768,
)
QUESTION_BUCKETS = (64, 128, 256, 384, 512, 640, 768, 1024, 1536, 2048, 3072, 4096)
ROW_BUCKETS = (8, 32, 128)
_DEFAULT_TP = 16
CC_FLAGS = "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"


def _bucket(n: int, ladder: Sequence[int]) -> int:
    for b in ladder:
        if n <= b:
            return b
    raise ValueError(f"{n} tokens exceed the largest bucket, {ladder[-1]}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def pack(
    state: list[int],
    questions: list[list[int]],
    *,
    pad_id: int,
    state_buckets: Sequence[int] = STATE_BUCKETS,
    question_buckets: Sequence[int] = QUESTION_BUCKETS,
    row_buckets: Sequence[int] = ROW_BUCKETS,
) -> dict[str, Any]:
    """One pass's inputs: the state padded to its bucket, then the questions back to back,
    padded to theirs. A question's tokens sit at positions len(state) onward, as in its own
    prompt; `qid` says which question each question token belongs to (-1: padding)."""
    s = len(state)
    total = sum(len(q) for q in questions)
    sb = _bucket(s, state_buckets)
    pb = _bucket(total, question_buckets)
    nb = _bucket(len(questions), row_buckets)
    ids = np.full(sb + pb, pad_id, dtype=np.int32)
    positions = np.zeros(sb + pb, dtype=np.int32)
    ids[:s] = state
    positions[:s] = np.arange(s)
    qid = np.full(pb, -1, dtype=np.int32)
    rows = np.zeros(nb, dtype=np.int32)
    off = 0
    for i, q in enumerate(questions):
        ids[sb + off : sb + off + len(q)] = q
        positions[sb + off : sb + off + len(q)] = s + np.arange(len(q))
        qid[off : off + len(q)] = i
        rows[i] = sb + off + len(q) - 1
        off += len(q)
    state_valid = np.zeros(sb, dtype=bool)
    state_valid[:s] = True
    return {
        "ids": ids,
        "positions": positions,
        "state_valid": state_valid,
        "qid": qid,
        "rows": rows,
        "state_bucket": np.int64(sb),
        "question_bucket": np.int64(pb),
        "questions": np.int64(len(questions)),
    }


# Compiled at load, as (state tokens, question tokens, questions): five questions in each token
# bucket from 640 to 1,280, a short one-question request, and 30 questions over 1,024 tokens.
WARM = (
    *((t - 520, 520, 5) for t in (640, 768, 896, 1024, 1152, 1280)),
    (100, 28, 1),
    (1024, 1530, 30),
)
# Joint packing: the state and the questions back to back in one bucket of tokens.
TOKEN_BUCKETS = (
    tuple(range(128, 2048, 128))
    + tuple(range(2048, 4096, 256))
    + (
        4096,
        4608,
        5120,
        6144,
        7168,
        8192,
        10240,
        12288,
        14336,
        16384,
        20480,
        24576,
        28672,
        32768,
        34816,
        36864,
    )
)


def pack_joint(
    state: list[int],
    questions: list[list[int]],
    *,
    pad_id: int,
    token_buckets: Sequence[int] = TOKEN_BUCKETS,
    row_buckets: Sequence[int] = ROW_BUCKETS,
) -> dict[str, Any]:
    """One pass's inputs as one bucket: the state, each question after it (at positions
    len(state) onward, as in its own prompt), then padding; `qid` marks each token's part
    (STATE, a question's index, -1 for padding)."""
    from .neuron_model import STATE

    s = len(state)
    total = s + sum(len(q) for q in questions)
    tb = _bucket(total, token_buckets)
    nb = _bucket(len(questions), row_buckets)
    ids = np.full(tb, pad_id, dtype=np.int32)
    positions = np.zeros(tb, dtype=np.int32)
    qid = np.full(tb, -1, dtype=np.int32)
    ids[:s], positions[:s], qid[:s] = state, np.arange(s), STATE
    rows = np.zeros(nb, dtype=np.int32)
    off = s
    for i, q in enumerate(questions):
        ids[off : off + len(q)] = q
        positions[off : off + len(q)] = s + np.arange(len(q))
        qid[off : off + len(q)] = i
        rows[i] = off + len(q) - 1
        off += len(q)
    return {
        "ids": ids,
        "positions": positions,
        "state_valid": np.ones(1, dtype=bool),
        "qid": qid,
        "rows": rows,
        "state_bucket": np.int64(0),
        "question_bucket": np.int64(tb),
        "questions": np.int64(len(questions)),
    }


def passes(questions: list[list[int]], max_tokens: int, max_rows: int) -> list[list[int]]:
    """Consecutive runs of question indices whose tokens fit one pass's question bucket."""
    out: list[list[int]] = []
    cur: list[int] = []
    used = 0
    for i, q in enumerate(questions):
        if len(q) > max_tokens:
            raise ValueError(
                f"a question of {len(q)} tokens exceeds the largest bucket, {max_tokens}"
            )
        if cur and (used + len(q) > max_tokens or len(cur) >= max_rows):
            out.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += len(q)
    if cur:
        out.append(cur)
    return out


class NeuronBackend:
    """A `strands_decider.decision.DecisionBackend` on Trainium2: Gemma 4's text decoder split
    over `tp` NeuronCores. `forward` may be called from several threads; calls take turns."""

    def __init__(
        self,
        path: str,
        *,
        tp: int = _DEFAULT_TP,
        compile: bool = True,
        max_length: int = 32768,
        timeout: float = 7200.0,
        opts: dict[str, Any] | None = None,
    ):
        import multiprocessing as mp

        import transformers

        text = transformers.AutoConfig.from_pretrained(path).get_text_config()
        self.pad_id = int(text.pad_token_id or 0)
        self.tp = tp
        self.compiled = compile
        self.timeout = timeout
        self.state_buckets = (
            *(b for b in STATE_BUCKETS if b < max_length),
            _bucket(max_length, STATE_BUCKETS),
        )
        self.question_buckets = QUESTION_BUCKETS
        self.row_buckets = ROW_BUCKETS
        self.opts = {
            "compile": compile,
            "max_positions": self.state_buckets[-1] + self.question_buckets[-1],
            **(opts or {}),
        }
        self.joint_max = int((opts or {}).get("joint_max", 4096))
        self._lock = threading.Lock()
        self._req = 0
        self.last_timings: dict[str, float] = {}
        self.warm_s: dict[str, float] = {}
        # The readout is applied here (the core's weight=None path), the weight rounded to bf16
        # and widened once at load rather than on every request.
        from ..core import DecisionConfig, load_readout
        from ..core.readout import round_to_bf16

        cfg = DecisionConfig.load(path)
        self._softcap, self._temperature = cfg.final_logit_softcapping, cfg.temperature
        self._w64 = round_to_bf16(load_readout(path)).astype(np.float64).T.copy()
        self.applies_readout = bool((opts or {}).get("applies_readout", True))
        ctx = mp.get_context("spawn")
        self._out = ctx.Queue()
        self._inboxes = [ctx.Queue() for _ in range(tp)]
        port = _free_port()
        from .neuron_model import worker_main

        self._procs = [
            ctx.Process(
                target=worker_main,
                args=(r, tp, port, path, self.opts, self._inboxes[r], self._out),
                daemon=True,
            )
            for r in range(tp)
        ]
        for p in self._procs:
            p.start()
        import atexit

        atexit.register(self.close)
        ready = 0
        while ready < tp:
            msg = self._get(timeout)
            if msg[0] == "error":
                self.close()
                raise RuntimeError(f"Neuron rank {msg[1]} failed to start:\n{msg[2]}")
            if msg[0] == "ready":
                ready += 1

    def _get(self, timeout: float) -> Any:
        import queue

        deadline = time.time() + timeout
        while True:
            try:
                return self._out.get(timeout=5)
            except queue.Empty:
                dead = [r for r, p in enumerate(self._procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"Neuron rank(s) {dead} exited") from None
                if time.time() > deadline:
                    raise TimeoutError("no answer from the Neuron ranks") from None

    @property
    def takes_images(self) -> bool:
        return False

    @property
    def image_problem(self) -> str:
        return NO_VISION

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "neuron",
            "device": f"neuron (tensor parallel over {self.tp} NeuronCores)",
            "dtype": self.opts.get("dtype", "bfloat16"),
            "reuse_state": True,
            "compiled": self.compiled,
            "state_buckets": list(self.state_buckets),
            "question_buckets": list(self.question_buckets),
        }

    def forward(
        self, state: list[int], questions: list[list[int]], images: Sequence[RenderedImage] = ()
    ) -> np.ndarray:
        """Each question after the shared state under prefix_bidir, seeing no other question:
        the final-norm hidden state at each question's last token, float32, [questions, hidden]."""
        if images:
            raise ValueError(NO_VISION)
        rows: list[np.ndarray] = []
        timings: dict[str, float] = {}
        with self._lock:
            room = self.question_buckets[-1]
            if len(state) <= self.joint_max:
                room = min(room, TOKEN_BUCKETS[-1] - len(state))
            for run in passes(questions, room, self.row_buckets[-1]):
                batch = self._pack(state, [questions[i] for i in run])
                out, t = self._run(batch)
                rows.extend(out[: len(run)])
                for k, v in t.items():
                    timings[k] = timings.get(k, 0.0) + v
        self.last_timings = timings
        hidden = np.stack(rows).astype(np.float32)
        return self.readout(hidden) if self.applies_readout else hidden

    def readout(self, hidden: np.ndarray) -> np.ndarray:
        """Each row's probabilities over all 255 codes: the core's readout_logits arithmetic
        (z = bf16(W bf16(h)), products summed in float64) and code_probabilities' soft-cap and
        temperature in float32. The core keeps a question's own k codes and renormalizes them,
        which equals the softmax over those k."""
        from ..core.readout import round_to_bf16

        z = round_to_bf16((round_to_bf16(hidden).astype(np.float64) @ self._w64).astype(np.float32))
        c, t = np.float32(self._softcap), np.float32(self._temperature)
        sc = np.tanh(z / c) * c / t
        e = np.exp(sc - sc.max(axis=-1, keepdims=True))
        out: np.ndarray = (e / e.sum(axis=-1, keepdims=True, dtype=np.float32)).astype(np.float32)
        return out

    def _pack(self, state: list[int], questions: list[list[int]]) -> dict[str, Any]:
        # Joint packing up to `joint_max` state tokens; past it, the split pass, whose state
        # attention runs in the NKI flash kernel and whose padding is a small share.
        if len(state) <= self.joint_max:
            return pack_joint(state, questions, pad_id=self.pad_id, row_buckets=self.row_buckets)
        return pack(
            state,
            questions,
            pad_id=self.pad_id,
            state_buckets=self.state_buckets,
            question_buckets=self.question_buckets,
            row_buckets=self.row_buckets,
        )

    def _run(self, batch: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        self._req += 1
        for box in self._inboxes:
            box.put(("forward", self._req, batch))
        while True:
            msg = self._get(self.timeout)
            if msg[0] == "error":
                raise RuntimeError(f"Neuron rank {msg[1]} failed:\n{msg[2]}")
            if msg[0] == "result" and msg[1] == self._req:
                return msg[2], msg[3]

    def warm(self, shapes: Sequence[tuple[int, int, int]]) -> dict[str, float]:
        """Compiles (or loads from the cache) the graphs of each (state, questions' tokens,
        questions) shape, so requests in those buckets don't pay for it."""
        out = {}
        for s, qt, nq in shapes:
            per = max(1, qt // max(1, nq))
            qs = [[self.pad_id] * per for _ in range(nq)]
            t = time.time()
            self.forward([self.pad_id] * s, qs)
            out[f"{s}/{qt}/{nq}"] = time.time() - t
        return out

    def profile(
        self,
        state: list[int],
        questions: list[list[int]],
        outdir: str,
        modes: Sequence[str] = ("runtime", "device"),
        reps: int = 3,
    ) -> dict[str, Any]:
        """One pass of `questions` over `state`, profiled on rank 0 into `outdir`."""
        batch = self._pack(state, questions)
        with self._lock:
            self._req += 1
            for box in self._inboxes:
                box.put(("profile", self._req, batch, outdir, list(modes), reps))
            while True:
                msg = self._get(self.timeout)
                if msg[0] == "error":
                    raise RuntimeError(f"Neuron rank {msg[1]} failed:\n{msg[2]}")
                if msg[0] == "result" and msg[1] == self._req:
                    report: dict[str, Any] = msg[3]
                    return report

    def stats(self) -> list[dict[str, Any]]:
        """Each rank's compile times and device memory peaks."""
        with self._lock:
            for box in self._inboxes:
                box.put(("stats",))
            out: list[dict[str, Any]] = []
            while len(out) < self.tp:
                msg = self._get(self.timeout)
                if msg[0] == "stats":
                    out.append(msg[2])
                elif msg[0] == "error":
                    raise RuntimeError(f"Neuron rank {msg[1]} failed:\n{msg[2]}")
        return sorted(out, key=lambda d: d["rank"])

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for box in getattr(self, "_inboxes", []):
            box.put(("stop",))
        for p in getattr(self, "_procs", []):
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()


def _warm_shape(spec: str) -> tuple[int, int, int]:
    s, qt, nq = (int(x) for x in spec.split(":"))
    return s, qt, nq


def load_backend(
    path: str, *, tp: int | None = None, compile: bool | None = None, max_length: int = 32768
) -> NeuronBackend:
    """The model checkpoint in directory `path` on `tp` NeuronCores (STRANDS_NEURON_TP, 16 by
    default), compiled unless STRANDS_NEURON_EAGER=1."""
    tp = tp or int(os.environ.get("STRANDS_NEURON_TP", _DEFAULT_TP))
    if compile is None:
        compile = os.environ.get("STRANDS_NEURON_EAGER", "0") != "1"
    opts: dict[str, Any] = {}
    if v := os.environ.get("STRANDS_NEURON_JOINT_MAX"):
        opts["joint_max"] = int(v)
    if v := os.environ.get("STRANDS_NEURON_NKI_STATE"):
        opts["nki_state"] = v == "1"
    if v := os.environ.get("STRANDS_NEURON_LAYERS_PER_GRAPH"):
        opts["layers_per_graph"] = int(v)
    env: dict[str, str] = {}
    if v := os.environ.get("STRANDS_NEURON_CORES"):  # such as 16-31: rank r takes the r-th
        env["NEURON_RT_VISIBLE_CORES"] = v
    # neuronx-cc's collective/compute overlap is faster on short prompts but doesn't compile
    # the split pass's graphs (NCC_IBTN175), so it is opt-in.
    if os.environ.get("STRANDS_NEURON_CC_OVERLAP") == "1" and "NEURON_CC_FLAGS" not in os.environ:
        env["NEURON_CC_FLAGS"] = CC_FLAGS
    opts["env"] = env
    backend = NeuronBackend(path, tp=tp, compile=compile, max_length=max_length, opts=opts)
    # Compile (or load from the NEFF cache) the common buckets before the first request:
    # STRANDS_NEURON_WARM is "state:questions' tokens:questions,..." or "none".
    warm = os.environ.get("STRANDS_NEURON_WARM", "default")
    if compile and warm != "none":
        shapes: Sequence[tuple[int, int, int]] = WARM
        if warm != "default":
            shapes = [_warm_shape(w) for w in warm.split(",")]
        shapes = [w for w in shapes if w[0] + w[1] // max(1, w[2]) <= max_length]
        backend.warm_s = backend.warm(shapes)
    return backend
