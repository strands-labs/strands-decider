"""Serving one request: N questions against one state.

The headline property of a System One model is that asking more questions about the
same state barely costs more time. That is not automatic -- it falls out of doing
the work in the right order:

    1. Encode the state ONCE, keeping its KV cache.          <- the expensive part
    2. Broadcast that cache across the question batch.
    3. Forward only the (short) question suffixes.           <- the cheap part

For a 2000-token state and five 40-token questions, the naive approach encodes
~10,200 tokens; this encodes ~2,200. The questions genuinely evaluate in parallel,
independently -- no question can see another's text, which is what keeps the answers
decomposable in the way the API promises.

Set `use_prefix_cache=False` to fall back to plain batched encoding. The two paths
agree to within float tolerance (tests/test_prefix_cache.py pins that, for plain and hybrid torsos), so the
fallback is a safety valve rather than a different model.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from . import __version__
from .modeling import StrandsDeciderModel, apply_temperature, masked_log_softmax, pool_last_token
from .prompting import (
    RenderedQuestion,
    read_choice,
    read_noul,
    read_score,
    render_question,
    render_state,
    score_legend,
)
from .schema import (
    Answer,
    ChoiceAnswer,
    Content,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    derive_confidence,
    derive_score_confidence,
)


@dataclass
class EngineConfig:
    device: str = "cuda"
    use_prefix_cache: bool = True
    max_batch: int = 32
    model_name: str = f"strands-decider-{__version__}"
    # Also cap the tokens of one forward: a chunk of up to max_batch questions is cut
    # where its rows, fitted to the window and padded to the longest (state + question),
    # would pass this many tokens (one question at least). Questions are fitted before the
    # cut and scored independently, so answers only move by float rounding; long-context
    # Gemma torsos need it at a 32,768 window. None: off.
    max_batch_tokens: int | None = None
    # Cross-request batching (server micro-batcher, `evaluate_prepared`): one forward holds
    # at most this many padded tokens (rows x longest row) and this many rows. A row longer
    # than the budget runs alone, as it would unbatched. With eager attention (Gemma 4 E2B)
    # a forward is also held to rows x longest^2 <= window^2: its score matrices are then
    # no larger than those of one window-filling request. The server sets them from
    # --max-batch and --max-batch-tokens.
    batch_tokens: int = 16384
    batch_rows: int = 32
    # Padding waste: a row joins a forward only while its padded tokens stay within
    # (1 + batch_pad_slack) x its real tokens, unless the forward is at most
    # batch_pad_free tokens (short forwards are launch-bound: padding there is free).
    # Long prompts are compute-bound, and padding them only costs. None: no cap.
    batch_pad_slack: float | None = 0.25
    batch_pad_free: int = 4096
    # Measured by `tune_batching` (at serve startup): {row width: best rows per forward}.
    # A forward of width w then holds at most the smaller of its neighbouring widths'
    # best token counts (rows x width). Batching pays for short, launch-bound rows; for
    # long rows a torso can be memory-bound, and a larger forward only spills its
    # activations out of cache. None: only the budgets above.
    batch_rows_by_width: dict[int, int] | None = None


@dataclass
class PreparedRow:
    """One forward row of a prepared request: its tokens (state + question), its option
    positions in those tokens (pointer head), slot count and question kind."""

    ids: list[int]
    opt: list[int]
    n_slots: int
    kind: str


@dataclass
class PreparedPrefix:
    """Several questions on one state, ready for the cross-request shared-prefix path.

    `state_ids` encodes the state once. Each question has its own `question_ids` and
    option character `offsets`, both aligned with the request's `rendered`.

    Filled when the request has more than one question and the cache is forkable. The
    engine refuses any prompt that does not fit the context window, so a single
    tokenisation of the state is always deterministic across chunks.
    """

    state_ids: list[int]
    question_ids: list[list[int]]
    offsets: list[list[tuple[int, int]]]


@dataclass
class PreparedRequest:
    """A request tokenised on the host, ready for `evaluate_prepared`.

    Three ways a prepared request runs:
    - `rows` not None, `prefix` None: cross-request row batching (one row per question).
    - `rows` None, `prefix` not None: cross-request shared-prefix batching (one shared
      prefix forward, then one shared suffix forward across requests of equal state
      length).
    - both None: the request runs alone through `evaluate` (vision / MLX engines, or
      the shared-prefix path).
    """

    request: SystemOneRequest
    names: list[str]
    rendered: list[RenderedQuestion]
    rows: list[PreparedRow] | None
    prefix: PreparedPrefix | None = None

    @property
    def tokens(self) -> int:
        if self.rows is not None:
            return sum(len(r.ids) for r in self.rows)
        if self.prefix is not None:
            return len(self.prefix.state_ids) + sum(len(q) for q in self.prefix.question_ids)
        return 0


def is_cuda_fatal(exc: BaseException) -> bool:
    """A CUDA error that leaves the process's context unusable (device-side assert,
    illegal address): every later call on the device fails the same way."""
    msg = str(exc)
    return (
        "device-side assert" in msg
        or "illegal memory access" in msg
        or "CUDA error" in msg
        or "cudaError" in msg
    )


def tuned_token_cap(rows_by_width: dict[int, int] | None, width: int) -> int | None:
    """Padded tokens one forward of rows `width` wide may hold under `tune_batching`'s
    measurements: the smaller best token count (rows x width) of the measured widths
    either side of `width` (the nearest one outside the measured range)."""
    if not rows_by_width:
        return None
    ws = sorted(rows_by_width)
    lo = [w for w in ws if w <= width]
    hi = [w for w in ws if w >= width]
    near = ([lo[-1]] if lo else []) + ([hi[0]] if hi else [])
    return min(rows_by_width[w] * w for w in near)


def plan_rows(
    lengths: Sequence[int],
    max_tokens: int,
    max_rows: int,
    quadratic_cap: int | None = None,
    pad_slack: float | None = None,
    pad_free: int = 0,
    rows_by_width: dict[int, int] | None = None,
) -> list[list[int]]:
    """Group row indices into forwards, shortest first so rows of like length pad together.

    A group holds at most `max_rows` rows and `max_tokens` padded tokens (rows x its longest
    row); with `quadratic_cap`, also rows x longest^2 <= quadratic_cap (eager attention's
    score matrices); with `pad_slack`, a group over `pad_free` padded tokens also keeps its
    padded tokens within (1 + pad_slack) x its real tokens; with `rows_by_width`, at most
    `tuned_token_cap` padded tokens. A row over the caps on its own still gets a group of one.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    groups: list[list[int]] = []
    cur: list[int] = []
    real = 0
    for i in order:
        w, r = lengths[i], len(cur) + 1
        padded = r * w
        if cur and (
            r > max_rows
            or padded > max_tokens
            or (quadratic_cap is not None and r * w * w > quadratic_cap)
            or (
                pad_slack is not None
                and padded > pad_free
                and padded > (1.0 + pad_slack) * (real + w)
            )
            or (rows_by_width and padded > max(w, tuned_token_cap(rows_by_width, w) or 0))
        ):
            groups.append(cur)
            cur, real = [], 0
        cur.append(i)
        real += w
    if cur:
        groups.append(cur)
    return groups


def plan_prefix_groups(
    state_lengths: Sequence[int],
    question_counts: Sequence[int],
    max_tokens: int,
    max_rows: int,
) -> list[list[int]]:
    """Group multi-question requests into cross-request shared-prefix forwards.

    Groups indices by *equal* state length -- hybrid torsos' recurrent state and Gemma 4's
    position-based sliding window both require it (padding the prefix would corrupt
    answers). Within each length-bucket, packs so a group's suffix row count is at most
    `max_rows` and the prefix forward's padded tokens (len(group) * state_length) is at
    most `max_tokens`. A single request over the caps still gets a group of one -- the
    caller falls back to its own `evaluate` if the resulting forward runs out of memory.
    """
    assert len(state_lengths) == len(question_counts)
    by_len: dict[int, list[int]] = {}
    for i, L_s in enumerate(state_lengths):
        by_len.setdefault(L_s, []).append(i)
    groups: list[list[int]] = []
    for L_s, indices in by_len.items():
        cur: list[int] = []
        cur_rows = 0
        for i in indices:
            n = question_counts[i]
            if cur and (cur_rows + n > max_rows or (len(cur) + 1) * L_s > max_tokens):
                groups.append(cur)
                cur, cur_rows = [], 0
            cur.append(i)
            cur_rows += n
        if cur:
            groups.append(cur)
    return groups


class UnforkableCache(TypeError):
    """A KV cache layout `_expand_cache` cannot safely repeat across rows."""


# Per-row state a cache layer may hold (transformers 5): attention layers' `keys` and
# `values`; a linear-attention (Gated DeltaNet) layer's `conv_states` and
# `recurrent_states`, each a dict of tensors keyed by state index. First dim is the batch.
_ROW_STATES = ("keys", "values", "conv_states", "recurrent_states")
# Tensors a cache layer may hold that describe the layer, not a row, and so are shared
# by every fork unchanged: a sliding-window layer's window size (Gemma 4), a 0-dim tensor.
_SHARED_TENSORS = ("_sliding_window_tensor",)


def _fork_layered_cache(cache: Any, idx: torch.Tensor) -> Any:
    """Rebatch a layered cache (transformers 5) by selecting rows via `idx`.

    `idx` is a 1-D LongTensor; the output cache has `idx.numel()` rows, and output row j
    carries input row `idx[j]`. For a batch-1 cache with `idx = [0, 0, ..., 0]` of length
    n, this is the previous n-way fork. For a batch-B cache with
    `idx = [0]*n_0 + [1]*n_1 + ... + [B-1]*n_{B-1}`, each input row is repeated by its own
    count — the layout cross-request shared-prefix batching needs.

    Covers sliding-window layers (Gemma 4 E2B: four of every five), whose only extra
    tensor is the window size, shared by every output row.

    Covers hybrid torsos: Qwen3.5 mixes attention layers with Gated DeltaNet layers
    whose recurrent and convolution states must be repeated too, or the suffix forward
    fails on a shape mismatch. The fork gets new layer objects, new dicts and new
    tensors: a DeltaNet layer updates its states and flags (`has_previous_state`) in
    place, so an output that shared them would corrupt the prefix it came from. The
    approach is decider-2b's (`decider/shared_prefix.py`, Apache-2.0).

    Raises UnforkableCache for a layer holding tensors under any other name: we cannot tell
    whether such a tensor has a batch dimension, so the caller falls back to batched
    encoding rather than guess.
    """
    fork = copy.copy(cache)
    fork.layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        for name, v in list(vars(nl).items()):
            is_state = name in _ROW_STATES
            if isinstance(v, torch.Tensor):
                if name in _SHARED_TENSORS and v.dim() == 0:
                    continue  # per-layer, not per-row: every output row keeps it as it is
                if not is_state:
                    raise UnforkableCache(
                        f"cache layer {type(layer).__name__} holds tensor {name!r}"
                    )
                if v.numel():
                    setattr(nl, name, v.index_select(0, idx.to(v.device)).contiguous())
            elif isinstance(v, dict):
                if any(isinstance(t, torch.Tensor) for t in v.values()) and not is_state:
                    raise UnforkableCache(
                        f"cache layer {type(layer).__name__} holds tensors in {name!r}"
                    )
                setattr(
                    nl,
                    name,
                    {
                        k: t.index_select(0, idx.to(t.device)).contiguous()
                        if isinstance(t, torch.Tensor) and t.numel()
                        else t
                        for k, t in v.items()
                    },
                )
        fork.layers.append(nl)
    return fork


def _expand_cache(cache: Any, n: int) -> Any:
    """Repeat a batch-1 KV cache across `n` rows (the single-prefix fork)."""
    if isinstance(getattr(cache, "layers", None), list):
        return _fork_layered_cache(cache, torch.zeros(n, dtype=torch.long))
    raise UnforkableCache(f"unsupported KV cache type: {type(cache)!r}")


def _expand_cache_per_row(cache: Any, repeats: Sequence[int]) -> Any:
    """Repeat each row of a batch-B cache by `repeats[row]` times, packing the copies
    contiguously in order. `len(repeats)` is the input batch; output batch is
    `sum(repeats)`. The cross-request shared-prefix forward uses this to turn one
    prefix forward over several states into one suffix forward over all their
    questions."""
    if isinstance(getattr(cache, "layers", None), list):
        idx = torch.tensor([k for k, r in enumerate(repeats) for _ in range(r)], dtype=torch.long)
        return _fork_layered_cache(cache, idx)
    raise UnforkableCache(f"unsupported KV cache type: {type(cache)!r}")


class SystemOneEngine:
    """Stateless evaluator around a loaded StrandsDeciderModel."""

    def __init__(self, model: StrandsDeciderModel, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        if str(self.cfg.device).startswith("mps"):
            # No fla/Triton on macOS; replace the reference chunk rule's slow MPS solver.
            from .mps_kernels import install

            install()
        elif str(self.cfg.device) == "cpu":
            # The reference depthwise conv is slow on CPU (per channel without oneDNN); see cpu_kernels.
            from .cpu_kernels import install as install_cpu

            install_cpu()
        self.model = model.to(self.cfg.device).eval()
        if str(self.cfg.device) == "cpu":
            self._upcast_torso_for_cpu()
        self.tok = model.tokenizer
        self.device = self.cfg.device
        # Sticky, monotonic False->True: cache the first UnforkableCache so later
        # requests skip the shared-prefix path without racing on self.cfg.
        self._prefix_cache_unforkable = False
        # HuggingFace's fast tokenizer reconfigures its truncation/padding state on
        # every call, which is not safe to run from two threads at once (it raises
        # "Already borrowed", or silently uses another thread's max_length).
        # Tokenisation is cheap relative to the model forward, so holding this is fine.
        self._tok_lock = threading.Lock()
        # Eager attention materialises [rows, heads, L, L] scores, so a cross-request
        # forward is also capped quadratically (EngineConfig.batch_tokens).
        torso_cfg = getattr(getattr(self.model, "torso", None), "config", None)
        self._quadratic_attn = getattr(torso_cfg, "_attn_implementation", None) == "eager"

    def _upcast_torso_for_cpu(self) -> None:
        """Run a half-precision torso in fp32 on CPU.

        CPU kernels for bf16 are slower than fp32, not faster: on an M3 Pro a
        256-token v19 question takes 7.7 s in bf16 and 3.5 s in fp32, with the same
        answer. The cost is memory, about 7 GiB of torso weights instead of 3.5. The
        readout is already fp32. Done under inference_mode because `load_engine`
        builds the model there, and inference tensors cannot be modified outside it.
        """
        torso = self.model.torso
        first = next(torso.parameters(), None)  # a stub torso in the tests has none
        if first is not None and first.dtype in (torch.bfloat16, torch.float16):
            with torch.inference_mode():
                torso.to(torch.float32)

    def _temperatures(self, kinds: list[str]) -> torch.Tensor:
        """Per-row temperature, falling back to the global scalar where unfitted."""
        cfg = self.model.config
        by_kind = getattr(cfg, "temperature_by_kind", None) or {}
        return torch.tensor(
            [float(by_kind.get(k, cfg.temperature)) for k in kinds],
            device=self.device,
            dtype=torch.float32,
        )

    def _fit(
        self, state_text: str, question_texts: list[str]
    ) -> tuple[list[int], list[list[int]], list[list[tuple[int, int]]]]:
        """Tokenise state and questions, refusing any prompt that does not fit the window.

        A prompt where the state plus the longest question exceeds the model's context
        window raises ValueError (HTTP 422 at the server). Callers never see truncated
        state or questions. Returns state ids, question ids, and each question's
        character offsets, as locals rather than on the engine (the server runs requests
        concurrently and engine state would be shared across requests).
        """
        max_len = self.model.config.max_length
        with self._tok_lock:
            enc = self.tok(question_texts, add_special_tokens=False, return_offsets_mapping=True)
            q = enc["input_ids"]
            offs = enc["offset_mapping"]
            longest = max(len(x) for x in q)
            s = self.tok(state_text, add_special_tokens=True)["input_ids"]
            if len(s) + longest > max_len:
                raise ValueError(
                    f"prompt of {len(s) + longest} tokens exceeds the context window "
                    f"of {max_len} tokens"
                )
            return s, q, list(offs)

    def _option_idx(
        self,
        rendered: list[RenderedQuestion],
        base: int,
        offsets: list[list[tuple[int, int]]],
    ) -> torch.Tensor:
        """Option token positions for a pointer head, offset by `base`.

        `base` is 0 when the caller forwards only the question (the shared-prefix path,
        where `hidden` is the suffix) and the state's token count when it forwards the
        whole prompt. `offsets` are `_fit`'s, so front truncation is accounted for.
        """
        rows = self._host_option_positions(rendered, offsets)
        width = max(len(r) for r in rows)
        return torch.tensor(
            [[base + i for i in r] + [-1] * (width - len(r)) for r in rows],
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _host_option_positions(
        rendered: Sequence[RenderedQuestion],
        offsets: Sequence[Sequence[Any]],
    ) -> list[list[int]]:
        """`_option_idx`'s positions as host lists, question-relative (no `base`)."""
        return [
            _option_token_index(offs, rq.option_spans, 0)
            for rq, offs in zip(rendered, offsets, strict=True)
        ]

    def _pad(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad to a rectangle; pool_last_token finds the last real token by mask."""
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        width = max(len(x) for x in seqs)
        ids = torch.tensor([x + [pad_id] * (width - len(x)) for x in seqs], device=self.device)
        mask = torch.tensor(
            [[1] * len(x) + [0] * (width - len(x)) for x in seqs], device=self.device
        )
        return ids, mask

    # ---- low level -------------------------------------------------------

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _slot_probs_batched(
        self,
        s: list[int],
        q: list[list[int]],
        n_slots: list[int],
        kinds: list[str],
        offsets: list[list[tuple[int, int]]],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Fallback: encode each full prompt independently, from `_fit`'s state `s` and
        questions `q`.

        Previously this concatenated state and question into one string and let the
        tokeniser truncate, which cuts from the right -- removing the options and the
        `<answer>` marker the head pools at. Now it shares `_fit` with the cached path,
        so both truncate the same thing in the same direction.
        """
        ids, mask = self._pad([s + qi for qi in q])
        # This path forwards the whole prompt, so option positions sit after the state.
        assert rendered is not None
        opt_idx = self._option_idx(rendered, len(s), offsets)
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            n_slots=torch.tensor(n_slots, device=self.device),
            temperature=self._temperatures(kinds),
            opt_idx=opt_idx,
        )
        return out["log_probs"].exp(), int(mask.sum().item())

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _shared_prefix_batched(
        self,
        prefixes: Sequence[PreparedPrefix],
        rendered: Sequence[Sequence[RenderedQuestion]],
    ) -> tuple[torch.Tensor, int]:
        """Cross-request shared-prefix forward: encode B states together, then every
        state's question suffixes together.

        All prefixes MUST have `len(state_ids)` equal. Length-bucketing is the caller's
        responsibility: with different state lengths, (a) Qwen3.5's DeltaNet recurrence
        absorbs padded tokens with no mask knob to recover, and (b) Gemma 4's sliding
        window is position-based, so a short state would lose real context past
        `L_max - window`.

        Returns a [sum_k N_k, K] tensor of probabilities, flat in request order then that
        request's question order, plus the total real-token count across every forward.
        """
        B = len(prefixes)
        counts = [len(p.question_ids) for p in prefixes]
        total = sum(counts)
        L_s = len(prefixes[0].state_ids)
        assert all(len(p.state_ids) == L_s for p in prefixes), "state lengths must match"

        # One prefix forward over B states.
        prefix_ids = torch.tensor(
            [p.state_ids for p in prefixes], dtype=torch.long, device=self.device
        )
        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        # Rebatch the cache from [B, ...] to [sum N_k, ...], packing each row's forks in order.
        cache = _expand_cache_per_row(prefix_out.past_key_values, counts)

        # One suffix forward over every question of every state, right-padded.
        flat_q = [q for p in prefixes for q in p.question_ids]
        suffix_ids, suffix_mask = self._pad(flat_q)
        full_mask = torch.cat(
            [
                torch.ones(total, L_s, dtype=suffix_mask.dtype, device=self.device),
                suffix_mask,
            ],
            dim=1,
        )
        hidden = self.model.encode(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=cache,
        )
        pooled = pool_last_token(hidden, full_mask).to(torch.float32)

        # Each row's option positions are suffix-relative (`hidden` is the suffix only), as in
        # the per-request shared-prefix path.
        from .modeling import gather_options

        flat_rendered: list[RenderedQuestion] = [rq for rr in rendered for rq in rr]
        flat_offsets: list[list[tuple[int, int]]] = [o for p in prefixes for o in p.offsets]
        options = gather_options(hidden, self._option_idx(flat_rendered, 0, flat_offsets))
        raw = self.model.head(pooled, options.to(torch.float32))

        kinds = [rq.kind for rq in flat_rendered]
        n_slots = [rq.n_slots for rq in flat_rendered]
        logits = apply_temperature(raw, self._temperatures(kinds))
        log_probs = masked_log_softmax(logits, torch.tensor(n_slots, device=self.device))
        n_tokens = B * L_s + int(suffix_mask.sum().item())
        return log_probs.exp(), n_tokens

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _slot_probs_shared_prefix(
        self,
        s: list[int],
        q: list[list[int]],
        n_slots: list[int],
        kinds: list[str],
        offsets: list[list[tuple[int, int]]],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Encode the state `s` once, then all question suffixes `q` against that cache."""
        m = len(q)
        # The suffixes continue an already-tokenised sequence, so _fit adds no BOS to
        # them -- one mid-sequence would be a token training never saw.
        prefix_ids = torch.tensor([s], device=self.device)
        prefix_len = prefix_ids.size(1)

        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        cache = _expand_cache(prefix_out.past_key_values, m)

        suffix_ids, suffix_mask = self._pad(q)
        full_mask = torch.cat(
            [
                torch.ones(m, prefix_len, dtype=suffix_mask.dtype, device=self.device),
                suffix_mask,
            ],
            dim=1,
        )

        hidden = self.model.encode(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=cache,
        )
        # pool_last_token slices the mask to the forwarded tail, so it lands on the
        # last real *suffix* token -- which is `<answer>`, the position that has
        # attended to state and question alike.
        pooled = pool_last_token(hidden, full_mask).to(torch.float32)
        # `hidden` is the suffix only, so option positions are suffix-relative and
        # need no prefix offset -- the cached state never enters the gather.
        from .modeling import gather_options

        assert rendered is not None
        options = gather_options(hidden, self._option_idx(rendered, 0, offsets))
        raw = self.model.head(pooled, options.to(torch.float32))
        logits = apply_temperature(raw, self._temperatures(kinds))
        log_probs = masked_log_softmax(logits, torch.tensor(n_slots, device=self.device))
        n_tokens = prefix_len + int(suffix_mask.sum().item())
        return log_probs.exp(), n_tokens

    # ---- public ----------------------------------------------------------

    def _render(self, request: SystemOneRequest) -> tuple[list[str], list[RenderedQuestion]]:
        if request.images:
            raise ValueError("this engine has no vision tower; serve with --vision for images")
        names = list(request.questions.keys())
        questions: list[Question] = [request.questions[n] for n in names]

        rendered: list[RenderedQuestion] = [render_question(q) for q in questions]
        return names, rendered

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        names, rendered = self._render(request)
        state_text = render_state(request.state)
        n_slots = [rq.n_slots for rq in rendered]

        answers: dict[str, Answer] = {}
        total_tokens = 0

        for chunk, state_ids, question_ids, offsets in self._chunks(state_text, rendered):
            chunk_rendered = rendered[chunk]
            chunk_slots = n_slots[chunk]
            chunk_kinds = [rq.kind for rq in chunk_rendered]

            probs = None
            # One question gains nothing from a shared prefix and pays a second forward:
            # measured on JevBench (one question per task), p50 0.111 s batched, 0.204 s
            # through the prefix path, with the same answers.
            if (
                self.cfg.use_prefix_cache
                and not self._prefix_cache_unforkable
                and len(chunk_rendered) > 1
            ):
                try:
                    probs, ntok = self._slot_probs_shared_prefix(
                        state_ids,
                        question_ids,
                        chunk_slots,
                        chunk_kinds,
                        offsets,
                        rendered=chunk_rendered,
                    )
                    total_tokens += ntok
                except UnforkableCache as e:
                    print(
                        f"[strands-decider] shared-prefix cache disabled ({e}); using batched encoding"
                    )
                    self._prefix_cache_unforkable = True
            if probs is None:
                probs, ntok = self._slot_probs_batched(
                    state_ids,
                    question_ids,
                    chunk_slots,
                    chunk_kinds,
                    offsets,
                    rendered=chunk_rendered,
                )
                total_tokens += ntok

            for i, name in enumerate(names[chunk]):
                rq = chunk_rendered[i]
                row = probs[i, : rq.n_slots].tolist()
                answers[name] = _to_answer(
                    rq, row, ordinal_smoothing=self.model.config.ordinal_smoothing
                )

        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            # One slot decision per question: the output side really is this cheap.
            usage=Usage(input_tokens=total_tokens, output_tokens=len(names)),
        )

    def _chunks(
        self, state_text: str, rendered: list[RenderedQuestion]
    ) -> list[tuple[slice, list[int], list[list[int]], list[Any]]]:
        """Consecutive runs of at most max_batch questions, each with its fitted state,
        questions and option offsets. A run of max_batch is fitted to the window once; with
        max_batch_tokens set it is then cut where rows x (state + longest question) would
        pass the cap, which bounds the padded forward and leaves every input as fitted."""
        k, cap = self.cfg.max_batch, self.cfg.max_batch_tokens
        out = []
        for g in range(0, len(rendered), k):
            s, q, offsets = self._fit(state_text, [rq.text for rq in rendered[g : g + k]])
            start, longest = 0, 0
            for i, qi in enumerate(q):
                if cap and i > start and (i - start + 1) * (len(s) + max(longest, len(qi))) > cap:
                    out.append((slice(g + start, g + i), s, q[start:i], offsets[start:i]))
                    start, longest = i, 0
                longest = max(longest, len(qi))
            out.append((slice(g + start, g + len(q)), s, q[start:], offsets[start:]))
        return out

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))

    # ---- cross-request batching -------------------------------------------
    #
    # `prepare` does a request's host work (render, fit, option positions) and raises
    # ValueError for a request the engine refuses, exactly where `evaluate` would.
    # `evaluate_prepared` then scores many prepared requests together: their rows (one per
    # question: state + question, as `_slot_probs_batched` builds them) are length-sorted
    # into right-padded forwards under `batch_tokens` / `batch_rows`. Each row's tokens,
    # option positions, temperature and slot mask are its own, and right padding leaves
    # every real position's hidden state as it is alone: attention is causal and padded
    # keys are masked, Gated DeltaNet's convolution and recurrence are causal (and padded
    # inputs are zeroed), and Gemma 4's per-layer inputs and positions are per token. The
    # pooled `<answer>` position and the option positions are all real tokens, so batching
    # changes answers only by float rounding.

    def can_batch_rows(self) -> bool:
        """Whether this engine scores rows with the base class's own forward (not MLX or
        vision, which override the low-level paths and so always run requests alone)."""
        cls = type(self)
        return (
            cls._slot_probs_batched is SystemOneEngine._slot_probs_batched
            and cls.evaluate is SystemOneEngine.evaluate
        )

    def prepare(self, request: SystemOneRequest) -> PreparedRequest:
        """Host-side preparation of one request for `evaluate_prepared` (no device work)."""
        if not self.can_batch_rows():
            return PreparedRequest(request, list(request.questions), [], None)
        names, rendered = self._render(request)
        state_text = render_state(request.state)
        chunks = self._chunks(state_text, rendered)
        multi_q = any(c.stop - c.start > 1 for c, *_ in chunks)
        if self.cfg.use_prefix_cache and not self._prefix_cache_unforkable and multi_q:
            # `_fit` encodes the state the same way for every chunk (no question-dependent
            # truncation), so one PreparedPrefix carries them all.
            flat_q: list[list[int]] = []
            flat_offsets: list[list[tuple[int, int]]] = []
            state_ids = chunks[0][1]
            for _, s, q, offsets in chunks:
                assert s == state_ids
                flat_q.extend(q)
                flat_offsets.extend(offsets)
            prefix = PreparedPrefix(state_ids=state_ids, question_ids=flat_q, offsets=flat_offsets)
            return PreparedRequest(request, names, rendered, None, prefix)
        pointer = self.model.config.head_type == "pointer"
        rows: list[PreparedRow] = []
        for chunk, s, q, offsets in chunks:
            chunk_rendered = rendered[chunk]
            pos = (
                self._host_option_positions(chunk_rendered, offsets) if pointer else [[] for _ in q]
            )
            for rq, qi, p in zip(chunk_rendered, q, pos, strict=True):
                rows.append(
                    PreparedRow(
                        ids=s + qi, opt=[len(s) + i for i in p], n_slots=rq.n_slots, kind=rq.kind
                    )
                )
        return PreparedRequest(request, names, rendered, rows)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _forward_rows(self, rows: Sequence[PreparedRow]) -> torch.Tensor:
        """One right-padded forward over rows of any requests -> [rows, K] probabilities
        (on the host). The same forward as `_slot_probs_batched`."""
        ids, mask = self._pad([r.ids for r in rows])
        opt_idx = None
        if self.model.config.head_type == "pointer":
            width = max(len(r.opt) for r in rows)
            opt_idx = torch.tensor(
                [r.opt + [-1] * (width - len(r.opt)) for r in rows],
                dtype=torch.long,
                device=self.device,
            )
        out = self.model(
            input_ids=ids,
            attention_mask=mask,
            n_slots=torch.tensor([r.n_slots for r in rows], device=self.device),
            temperature=self._temperatures([r.kind for r in rows]),
            opt_idx=opt_idx,
        )
        probs: torch.Tensor = out["log_probs"].exp().float().cpu()
        return probs

    def _quadratic_cap(self) -> int | None:
        if not self._quadratic_attn:
            return None
        return int(self.model.config.max_length) ** 2

    def tune_batching(
        self, widths: Sequence[int] = (256, 1024, 4096), reps: int = 2, gain: float = 1.05
    ) -> dict[int, int]:
        """Measure, for synthetic rows of each width, the rows per forward with the best rows/s
        (k = 1, 2, 4, ... within the budgets; a larger k must gain `gain`x, and the search stops
        once rows/s falls or a forward runs out of memory), and keep them in
        cfg.batch_rows_by_width for `plan_rows`. The server runs it once at startup, before
        requests arrive, as vLLM runs a profiling forward to size its batches to the GPU."""
        import time

        sync = torch.cuda.synchronize if str(self.device).startswith("cuda") else (lambda: None)
        enc = self.tok(" the", add_special_tokens=False)["input_ids"]
        tid = int(enc[-1]) if enc else 0
        quad = self._quadratic_cap()
        window = int(self.model.config.max_length)
        out: dict[int, int] = {}
        for w in widths:
            if w > window or w < 4:
                continue
            row = PreparedRow(ids=[tid] * w, opt=[w - 3, w - 2], n_slots=2, kind="choice")
            best_k, best_rate, k = 1, 0.0, 1
            while k <= self.cfg.batch_rows and (
                k == 1 or (k * w <= self.cfg.batch_tokens and (quad is None or k * w * w <= quad))
            ):
                rows = [row] * k
                try:
                    self._forward_rows(rows)  # warm-up for this shape
                    sync()
                    t = time.perf_counter()
                    for _ in range(reps):
                        self._forward_rows(rows)
                    sync()
                except torch.OutOfMemoryError:  # the largest forward that fits is the cap
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                rate = k * reps / (time.perf_counter() - t)
                if rate > best_rate * gain:
                    best_k, best_rate = k, rate
                elif rate < best_rate:
                    break
                k *= 2
            out[w] = best_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cfg.batch_rows_by_width = out
        return out

    def _score_rows(self, rows: Sequence[PreparedRow], group: list[int], out: list[Any]) -> None:
        """Score `group` (indices into rows) into `out`: a probability row or an exception
        per row. On CUDA OOM the group is split in halves (a single row that still fails
        gets the OOM); a CUDA error that loses the context fails the whole group; any other
        error in a multi-row forward is retried row by row, so it stays with its own row."""
        try:
            probs = self._forward_rows([rows[i] for i in group])
        except torch.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(group) == 1:
                out[group[0]] = exc
                return
            h = len(group) // 2
            self._score_rows(rows, group[:h], out)
            self._score_rows(rows, group[h:], out)
            return
        except Exception as exc:  # mapped per row below
            if len(group) == 1 or is_cuda_fatal(exc):
                for i in group:
                    out[i] = exc
                return
            for i in group:
                self._score_rows(rows, [i], out)
            return
        for j, i in enumerate(group):
            out[i] = probs[j]

    def _run_prefix_group(
        self,
        prepared: Sequence[PreparedRequest],
        indices: Sequence[int],
        results: list[SystemOneResponse | BaseException | None],
    ) -> BaseException | None:
        """Score one length-bucketed group of multi-question requests with one shared
        prefix forward and one shared suffix forward. On any exception, falls back to
        running each request in the group alone through `evaluate` (so a single bad
        row cannot poison the others). Returns a CUDA-fatal exception if one was
        observed (the caller stops launching more work); otherwise None.
        """
        group = [prepared[i] for i in indices]
        prefixes = [p.prefix for p in group]
        assert all(pr is not None for pr in prefixes)
        try:
            probs, _ = self._shared_prefix_batched(
                [pr for pr in prefixes if pr is not None],
                [p.rendered for p in group],
            )
        except UnforkableCache as e:
            print(f"[strands-decider] shared-prefix cache disabled ({e}); falling back")
            self._prefix_cache_unforkable = True
            return self._run_prefix_group_alone(prepared, indices, results)
        except Exception as exc:
            if is_cuda_fatal(exc):
                for i in indices:
                    results[i] = exc
                return exc
            return self._run_prefix_group_alone(prepared, indices, results)
        # Split the flat probs tensor back into each request's answers.
        start = 0
        for i, p in zip(indices, group, strict=True):
            assert p.prefix is not None
            n = len(p.prefix.question_ids)
            mine = probs[start : start + n]
            start += n
            answers: dict[str, Answer] = {}
            for name, rq, row in zip(p.names, p.rendered, mine, strict=True):
                answers[name] = _to_answer(
                    rq,
                    row[: rq.n_slots].tolist(),
                    ordinal_smoothing=self.model.config.ordinal_smoothing,
                )
            results[i] = SystemOneResponse(
                model=self.cfg.model_name,
                answers=answers,
                usage=Usage(input_tokens=p.tokens, output_tokens=len(p.names)),
            )
        return None

    def _run_prefix_group_alone(
        self,
        prepared: Sequence[PreparedRequest],
        indices: Sequence[int],
        results: list[SystemOneResponse | BaseException | None],
    ) -> BaseException | None:
        """Fallback for a shared-prefix group: each request through `evaluate`. Returns a
        CUDA-fatal exception if observed."""
        for i in indices:
            try:
                results[i] = self.evaluate(prepared[i].request)
            except Exception as exc:
                results[i] = exc
                if is_cuda_fatal(exc):
                    return exc
        return None

    def evaluate_prepared(
        self, prepared: Sequence[PreparedRequest]
    ) -> list[SystemOneResponse | BaseException]:
        """Score prepared requests together; one response or exception per request, in order.

        Row-prepared requests go through `plan_rows` and shared row forwards; prefix-prepared
        requests go through `plan_prefix_groups` and shared prefix+suffix forwards; requests
        with neither (vision/MLX engines) run alone through `evaluate`.
        """
        results: list[SystemOneResponse | BaseException | None] = [None] * len(prepared)
        flat: list[PreparedRow] = []
        prefix_items: list[int] = []
        for k, p in enumerate(prepared):
            if p.rows is None and p.prefix is None:
                try:
                    results[k] = self.evaluate(p.request)
                except Exception as exc:  # returned to the caller
                    results[k] = exc
                continue
            if p.prefix is not None:
                prefix_items.append(k)
                continue
            flat.extend(p.rows or [])
        fatal: BaseException | None = None
        if prefix_items:
            state_lens: list[int] = []
            q_counts: list[int] = []
            for i in prefix_items:
                pr = prepared[i].prefix
                assert pr is not None
                state_lens.append(len(pr.state_ids))
                q_counts.append(len(pr.question_ids))
            prefix_groups = plan_prefix_groups(
                state_lens, q_counts, self.cfg.batch_tokens, self.cfg.batch_rows
            )
            for group in prefix_groups:
                indices = [prefix_items[g] for g in group]
                if fatal is not None:
                    for i in indices:
                        results[i] = fatal
                    continue
                fatal = self._run_prefix_group(prepared, indices, results)
        if flat:
            scored: list[Any] = [None] * len(flat)
            groups = plan_rows(
                [len(r.ids) for r in flat],
                self.cfg.batch_tokens,
                self.cfg.batch_rows,
                self._quadratic_cap(),
                self.cfg.batch_pad_slack,
                self.cfg.batch_pad_free,
                self.cfg.batch_rows_by_width,
            )
            for group in groups:
                if fatal is not None:  # the context is gone; do not launch more work
                    for i in group:
                        scored[i] = fatal
                    continue
                self._score_rows(flat, group, scored)
                fatal = next(
                    (
                        scored[i]
                        for i in group
                        if isinstance(scored[i], BaseException) and is_cuda_fatal(scored[i])
                    ),
                    None,
                )
            start = 0
            for k, p in enumerate(prepared):
                if p.rows is None:
                    continue
                mine = scored[start : start + len(p.rows)]
                start += len(p.rows)
                err = next((x for x in mine if isinstance(x, BaseException)), None)
                if err is not None:
                    results[k] = err
                    continue
                answers: dict[str, Answer] = {}
                for name, rq, row in zip(p.names, p.rendered, mine, strict=True):
                    answers[name] = _to_answer(
                        rq,
                        row[: rq.n_slots].tolist(),
                        ordinal_smoothing=self.model.config.ordinal_smoothing,
                    )
                results[k] = SystemOneResponse(
                    model=self.cfg.model_name,
                    answers=answers,
                    usage=Usage(input_tokens=p.tokens, output_tokens=len(p.names)),
                )
        return cast("list[SystemOneResponse | BaseException]", results)


def _option_token_index(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Sequence[int]],
    base: int,
) -> list[int]:
    """Last token index of each option's line, for a pointer readout.

    The last token of the span is the one that has just read the whole option under
    causal attention. Raises when an option has no surviving token: truncation has
    removed it, and scoring it from a neighbour's representation would be silently
    wrong.
    """
    out: list[int] = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was "
                "truncated through its option list"
            )
        out.append(last)
    return out


def _to_answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> Answer:
    if rq.kind == "noul":
        return NoulAnswer(noul=round(read_noul(probs, rq), 4))

    if rq.kind == "choice":
        by_name = read_choice(probs, rq)
        best = max(by_name, key=lambda k: by_name[k])
        return ChoiceAnswer(
            choice=best,
            probabilities={k: round(v, 4) for k, v in by_name.items()},
            confidence=round(derive_confidence(list(by_name.values())), 4),
        )

    score, ordered = read_score(probs, rq)
    return ScoreAnswer(
        score=round(score, 4),
        # Report the rubric in canonical ascending order, not the order rendered.
        legend=score_legend(rq),
        probabilities={k: round(v, 4) for k, v in ordered.items()},
        # Ordinal confidence, not max-probability: see derive_score_confidence.
        confidence=round(
            derive_score_confidence(list(ordered.values()), ordinal_smoothing=ordinal_smoothing),
            4,
        ),
    )


@torch.inference_mode()  # type: ignore[untyped-decorator]
def load_engine(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    attn_implementation: str | None = None,
) -> SystemOneEngine:
    """An engine on a torch device ("cuda", "mps", "cpu"), or on MLX with `device="mlx"`."""
    if device == "mlx":
        return load_mlx(checkpoint, EngineConfig(device="mlx", use_prefix_cache=use_prefix_cache))
    model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    return SystemOneEngine(model, EngineConfig(device=device, use_prefix_cache=use_prefix_cache))


def mlx_available() -> bool:
    """True on Apple silicon with the `mlx` extra installed.

    `mlx` is a namespace package: `pip uninstall mlx` can leave an importable `mlx`
    directory behind (mlx-metal's), so the check is for `mlx.core`.
    """
    import importlib.util
    import platform

    # platform.system(), not sys.platform: mypy narrows sys.platform to the checking
    # host's, which on CI's Linux runners marks the rest of the function unreachable.
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        return (
            importlib.util.find_spec("mlx.core") is not None
            and importlib.util.find_spec("mlx_lm") is not None
        )
    except ModuleNotFoundError:  # finding `mlx.core` imports `mlx`, which may be absent
        return False


def load_mlx(checkpoint: str, config: EngineConfig | None = None) -> SystemOneEngine:
    """An engine with the torso on MLX (see mlx_engine.py), configured as a torch engine is."""
    if not mlx_available():
        raise RuntimeError(
            "device 'mlx' needs Apple silicon and the mlx extra: pip install 'strands-decider[mlx]'"
        )
    from .mlx_engine import load_mlx_engine

    return load_mlx_engine(checkpoint, config)
