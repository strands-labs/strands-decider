# ADR-005: Benchmark suite as operator tooling, not CI

**Status**: Accepted (local implementation, 2026-10-05)
**Date**: 2026-10-05
**Authors**: Ken Semerkant (local work)
**Related**: ADR-004

## Context

The repo measured engine latency directly (`evaluation/bench_local.py`) and accuracy
through JevBench, but nothing measured the **served system** end to end, nothing
measured real labelled corpora through the HTTP API, and nothing exercised the
repeated-state pattern agents actually produce. Hand-rolled curl timing is not
reproducible and cannot act as a regression tripwire for changes like the state
cache. Meanwhile, CI-grade benchmarks on this stack are flaky by nature: they need a
live server, a multi-gigabyte warmed model, and a GPU — minute-scale variance from
kernel warmup alone.

## Decision

A `bench/` directory of standalone scripts driven over HTTP against a running server,
run by one shell entry point, **deliberately outside CI**:

- `bench_types.py` — 1000 tests per question type, per-request and
  many-questions-per-request modes.
- `bench_all.py` — synthetic sweep: state length × type, questions/request,
  choice width 2–50, mixed-vs-separate, state-cache hit/miss, concurrency 1/4/16,
  full-window steady state.
- `bench_corpus.py` — replays real labelled corpora (`data/synthetic/*_eval.jsonl`)
  through the API; reports latency, accuracy, and confidence-when-right-vs-wrong;
  groups by state to exercise the cache; measures instruction-paraphrase consistency.
- `jevbench_4b_letters.py` — frozen Qwen3.5-4B comparison harness (option-letter
  logits, SemIf-style).

Only the **pure mapping functions** (corpus record → API question, answer → correct?
) are unit-tested (`tests/test_bench_corpus.py`), so a schema drift in either the
corpora or the API fails loudly in CI without needing a server.

## Rationale

- Latency claims need a realistic token distribution: the synthetic filler's
  near-zero entropy under-represents real documents (real-corpus choice questions
  measure 121 ms p50, not the 76 ms the synthetic sweep suggests at similar length).
- Accuracy reported alongside latency turns every perf run into a correctness
  tripwire — the corpus benchmark caught nothing this time, which is itself the
  finding that validated the state cache.
- CI-hostile dependencies (live server, warmed GPU) are exactly why the scripts stay
  operator tools; the seam that *can* be tested continuously (the mapping) is.

## Consequences

**Positive:** one command reproduces every number quoted in ARCHITECTURE.md §10;
regressions in the cache or engine show up as section-level deltas; the 4B harness
grounds model-improvement arguments in local measurements.

**Negative:** no automated regression gate on performance — an operator must run the
suite; synthetic benchmarks with low-entropy filler can mislead if read in isolation
(mitigated by running the corpus benchmark first in `run.sh`); the 4B harness bakes
in a thinking-model quirk (the empty-think read position) that must survive Qwen
template changes.

## Alternatives Considered

- **Benchmarks in CI with a tiny checkpoint** — covers correctness (already done by
  the engine tests) but not performance worth gating on.
- **pytest markers with `@pytest.mark.slow`** — rejected: pretends to be CI while
  actually being skipped everywhere; scripts are honest.
