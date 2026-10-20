# ADR-004: Cross-request state KV cache on the MLX engine

**Status**: Accepted (local implementation, 2026-10-05; not yet contributed upstream)
**Date**: 2026-10-05
**Authors**: Ken Semerkant (local work)
**Related**: ADR-003

## Context

The engine's existing `use_prefix_cache` is **intra-request**: one state, many
questions — the state encodes once and the question suffixes share it. Across HTTP
requests, nothing persisted: the same ~1400-token state sent 40 times re-encoded on
every request at ~98 ms each. Interactive agent traffic (the Strands intervention
pattern) asks follow-up questions about one conversation state in separate requests
and pays that repeatedly.

Two constraints shaped the design:

1. **A single-question request must not pay more.** The engine deliberately routes
   one-question requests to a whole-prompt forward because prefix-then-suffix
   measured ~2× slower for a single question (0.204 s vs 0.111 s on JevBench).
2. **mlx-lm caches are one-shot.** `KVCache.merge` returns a `BatchKVCache` whose
   `offset` is a per-row mx array; merging a merged cache again slices with the array
   and raises `ValueError: Slice indices must be integers`. mlx-lm never merges
   twice, so nothing upstream exercised this.

## Decision

`MLXEngine` keeps an LRU map of **state token ids (post-`_fit`) → batch-1 snapshot of
the prompt-cache layers**.

- **Populate only on the shared-prefix path** (requests with ≥2 questions, including
  every chunk of a chunked request) — the one place a state-only prefix already
  exists. No request ever pays an extra forward to warm the cache.
- **Read on both paths.** A hit fans the snapshot out with `merge([layer] × n)` and
  forwards only the question suffixes — including single-question requests, which is
  where the win lands.
- **Snapshot conversion:** KV-layer snapshots are converted back to a plain
  `KVCache` (integer offset, same keys/values) so they merge repeatedly;
  `ArraysCache` layers (Gated DeltaNet recurrent/conv state) already round-trip.
- **Capacity** `--state-cache N` (default 8, 0 disables), reported in `/health`;
  all access under the engine's existing single-evaluation lock.

## Rationale

Keying on the exact post-truncation ids makes entries question-set-independent and
truncation-safe. Population policy follows the existing routing economics rather
than fighting them. Measured on M5 Max/MLX: repeated ~1400-token states 98.2 →
18.8 ms (5.2×); real-corpus grouped states 52.3 → 22.4 ms (2.3×); hit latency is
question-type-blind.

## Consequences

**Positive:** interactive repeated-state traffic at suffix-only cost; correctness
pinned (repeat-encode counter, hit-vs-miss parity ≤1e-4, LRU order, disable flag,
and a full JevBench run at 0.766 vs published 0.762 — inside the repo's noise rule).

**Negative / risks:**

- **Capacity must exceed the client's distinct-state working set.** 60 states
  through 8 entries gained 0%; through 128 entries, 2.3×. Entry memory scales with
  state length (KV + DeltaNet state for up to 4096 tokens) — an undersized cache is
  silent dead weight, hence the `/health` field.
- One-shot workloads see zero benefit and zero cost (miss path unchanged).
- The snapshot conversion encodes an mlx-lm implementation detail; a version change
  to its cache classes needs a test-level tripwire (the tiny-checkpoint harness
  would fail loudly).

## Alternatives Considered

- **Populate on every miss (encode state-only even for single questions)** —
  rejected: doubles the first request's cost for unknown repeat benefit.
- **Server-level cache of whole responses** — rejected: question variety is the
  workload; only the state should persist.
- **Same cache on the torch engines** — deferred: different memory tradeoffs, and
  the torch `_expand_cache` machinery is per-request by design.
