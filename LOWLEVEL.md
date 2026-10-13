# Strands Decider — Low-Level Design

Module-level mechanics of the serving stack, current as of 2026-10-06 (upstream at
`3e94e9d` plus the local state-cache and benchmark work). System-level structure and
decisions live in `ARCHITECTURE.md` and `adr/`.

## Module map

```
src/strands_decider/
├── schema.py      Pydantic request/answer models (Noul/Choice/Score unions)
├── prompting.py   render_state / render_question; option spans; question types
├── infer.py       SystemOneEngine (torch): _fit, batching, readout, answers
├── mlx_engine.py  MLXEngine: Metal torso, LoRA merge, state cache
├── modeling.py    StrandsDeciderModel, pointer head, temperatures, softmax helpers
├── server.py      FastAPI app factory, /v1/systemone, /health
├── cli.py         Typer: serve, ask
└── train.py, data/, distributed.py, hf_export.py, vision.py   (training/export side)
```

## The evaluation pipeline (`SystemOneEngine.evaluate`)

1. **Render.** Each question becomes text via `render_question`; `render_state` wraps
   the state (structure-aware for dict/list states). A rendered question ends at an
   `<answer>` marker — the pooling position.
2. **Slot ceilings.** Fixed-width heads cap options at `num_slots`; pointer heads have
   no ceiling (options are read from their own token positions).
3. **Fit (`_fit`).** Tokenise questions with offset mappings; the longest question
   reserves `min(longest, max_question_fraction × window)` tokens (front-truncated,
   keeping the option tail). The state then tokenises into what remains
   (`max_len − reserve`), truncated from the right. Offsets shift with truncation so
   pointer indices stay on the right tokens. `strict_window` refuses instead of
   truncating.
4. **Chunking.** Questions are encoded `max_batch` (32) per forward pass. Per chunk:
   - **1 question → whole-prompt path**: forward `state+question` in one pass
     (measured cheaper than prefix + suffix for a single question: 0.111 s vs
     0.204 s on JevBench).
   - **>1 question → shared-prefix path**: forward the state once with a KV cache,
     fan the cache out to the question count, forward all suffixes as one batch.
5. **Readout.** Pool the hidden state at each row's last real token (`<answer>`); a
   pointer head also gathers each option's last-token hidden state. The head scores
   options by comparing `<answer>` against option states; temperatures (fittable per
   question kind) scale logits; a masked softmax over each question's own slots gives
   probabilities.
6. **Answers.** `noul` → the true-probability; `choice` → argmax + per-option
   probabilities + derived confidence; `score` → ordinal expected value over the
   rubric + ordinal confidence. Usage counts prompt tokens once per state.

## MLX engine specifics (`mlx_engine.py`)

- **Load.** Base torso resolved at the provenance-pinned revision, loaded by mlx-lm
  (with a `model_type` remap for the text-only Qwen3.5 config), cast to the checkpoint
  dtype, casts evaluated eagerly (server evaluates on a thread pool — a lazy array
  would belong to the loading thread's stream).
- **LoRA merge.** PEFT names map to mlx-lm weight names; each target is
  `W + (α/r)·B·A` formed in fp32 on the CPU (Metal fp32 matmul is reduced-precision),
  rounded once to the torso dtype, swapped in immediately. Any adapter tensor that is
  not a plain lora_A/lora_B weight pair is refused, not skipped.
- **Forward.** Right-padded int32 rows into the mlx-lm decoder → last hidden states
  on Metal; the head's input positions are gathered on Metal, copied to fp32 torch
  CPU tensors through the buffer protocol, and scored by the checkpoint's own head
  module. No attention mask — both layer kinds are causal, so no real position reads
  a pad.
- **Lock.** One evaluation at a time (`threading.Lock` in `evaluate`): `_fit` stashes
  per-request option offsets on the engine, so concurrent requests would read each
  other's.

### Cross-request state cache

```
_state_cache: OrderedDict[state_token_ids -> snapshot]   # LRU, cap = state_cache_entries
snapshot = [batch-1 copy of each prompt-cache layer]     # taken AFTER encoding the state
state_encodes: int                                       # observable counter
```

- **Populate** — only in `_slot_probs_shared_prefix` (≥2 questions, every chunk):
  after `self._hidden([state], prefix)`, snapshot each layer, store under
  `tuple(state)`, evict LRU past capacity. No request pays an extra forward to fill
  the cache.
- **Read** — `_slot_probs_shared_prefix` always; `_slot_probs_batched` (single
  question) on every call first. A hit builds `batch = [merge([layer] * n_questions)]`
  from the snapshot and forwards only the question suffixes.
- **Key** — post-`_fit` state ids: exact, truncation-aware, question-set-independent.
- **Snapshot conversion (`_snapshot`)** — mlx-lm's `KVCache.merge` returns a
  `BatchKVCache` whose `offset` is a per-row mx array; merging that again slices with
  the array and raises `ValueError: Slice indices must be integers`. Snapshots of KV
  layers are converted back to a plain `KVCache` (integer `offset`, same
  keys/values); `ArraysCache` layers (Gated DeltaNet recurrent/conv state) already
  return re-mergeable caches and pass through.
- **Correctness pins** (`tests/test_mlx_engine.py`): repeated state encodes once
  (counter); single-question hit matches uncached answers ≤1e-4; LRU order at
  capacity 1; `state_cache_entries=0` disables. Full JevBench parity: 0.766 vs the
  published 0.762, within the repo's noise rule.

## Server / CLI wiring

`serve --state-cache N` → `serve(...)` → `create_app(state_cache=N)` →
`load_mlx_engine(..., state_cache_entries=N)`. `/health` reports the setting
(`getattr` — torch engines have none). Non-MLX devices ignore the flag.

## Request/response schema (authoritative shapes)

```jsonc
// POST /v1/systemone
{
  "state": "<text | structured content>",
  "images": ["<base64>"],            // --vision only, torch devices
  "questions": {
    "<name>": { "type": "noul",   "instructions": "..." },
    "<name>": { "type": "choice", "instructions": "...", "criteria": { "opt": "desc|null" } },
    "<name>": { "type": "score",  "instructions": "...", "criteria": ["low", ..., "high"] }
  }
}
// response
{ "model": "...", "answers": { "<name>": { ...typed answer + confidence... } },
  "usage": { "input_tokens": N, "output_tokens": N }, "latency_ms": F }
```

(The README's curl example omits `criteria` — this schema is authoritative.)

## Test harness

`tests/test_mlx_engine.py` builds a tiny random-weight Qwen3.5 (3 Gated DeltaNet +
1 attention layer, 64-dim) checkpoint on disk the way training saves one, with a LoRA
adapter over every v19 target projection, plus a word-level tokenizer covering the
rendered prompts. Both torch and MLX engines load it from disk — covering the loader,
the merge, both evaluation paths and now the state cache, with no downloads.
Apple-silicon-gated; CPU-device variants run the Metal-free path.

## Benchmark architecture (`bench/`)

| script | design |
| --- | --- |
| `bench_types.py` | 1000 requests/type (per-request p50/p95/p99) and 1 request × 1000 questions (server `latency_ms` is the honest clock); 20-request warmup |
| `bench_all.py` | 7 sections: length×type, qcount amortisation, choice width 2–50, mixed-vs-separate, state-cache hit/miss (populate with one multi-q request, then measure single-q repeats vs distinct states), concurrency 1/4/16, full-window steady state. `--only`/`--skip` per section |
| `bench_corpus.py` | replays `data/synthetic/*_eval.jsonl`: `record_questions` maps a corpus record (`{kind, state, instructions, options[[id, desc]], label, instruction_variants}`) to an API question; `check` scores the answer against the label. Sections: replay (latency+accuracy+confidence-when-right/wrong), grouped-by-state (pass 1 populates / pass 2 hits), paraphrase consistency |
| `jevbench_4b_letters.py` | frozen Qwen3.5-4B through the chat template; one forward per task; softmax over option-letter logits; **must append an empty-think prefix** — Qwen3.5 is a thinking model and logits at the generation prompt sit inside `<think>` (0.355 accuracy vs 0.710 at the answer position) |
| `run.sh` | corpus → all → types against one live server |

Mapping functions are pure and pinned by `tests/test_bench_corpus.py`; scripts are
operator tools and are deliberately not CI (they need a live, warmed server).
