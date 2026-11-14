# ADR-001: Pointer head instead of text generation

**Status**: Accepted (upstream design)
**Date**: 2026-09
**Authors**: Strands Decider maintainers
**Related**: ADR-002, ADR-003

## Context

A decision model must pick between options and rate on scales, faster than an LLM and
without per-task classifier training. The obvious implementation — prompt an LLM and
parse generated text — pays a decoding loop per answer, produces uncalibrated output,
and bakes label sets into the prompt-parsing layer. The first internal iteration used
a "slot head" mapping the final hidden state to a fixed set of 24 option slots; it
performed measurably worse and capped option counts.

## Decision

Take a pretrained decoder torso (Qwen3.5-2B-Base), **remove its language-modelling
head**, and attach a ~1M-parameter **pointer head**. One forward pass scores each
option by comparing the hidden state at a dedicated `<answer>` position against the
hidden state at that option's own last token. The torso is adapted with a rank-16
LoRA; the head runs in fp32. The same masked softmax is read three ways to produce
the three question types (`noul`, `choice`, `score`).

## Rationale

- No decoding loop: one forward pass per chunk of questions, so latency tracks prompt
  length, not answer length.
- The head holds no per-option parameters: nothing can learn positional bias, option
  count is unbounded, and label sets are defined per request rather than by weights.
- Removing generation removes the prompt-injection surface entirely — state text is
  data, never instructions.
- A derived confidence comes out of the same softmax, and it is calibrated (published:
  answers at confidence ≥0.9 are right ~95% of the time on unseen short tasks).

## Consequences

**Positive:** typed, calibrated answers at classification-model latency; option sets
defined at request time; no generation to guard.

**Negative:** single-pass architecture cannot deliberate — JevBench families needing
multi-step inference, arithmetic, or weighing competing considerations score far
below classification families (`temporal_numeric` 0.267 vs `extraction` 1.000). A
frozen Qwen3.5-4B read with option-letter logits beats the 2B decider on `probability`
(0.70 vs 0.30) and `tradeoff` (1.00 vs 0.50) — the capability exists in larger torsos
and is a distillation target, not an architectural fix.

## Alternatives Considered

- **LLM with constrained generation** — rejected: decoding latency, calibration gap,
  injection surface.
- **Slot head (fixed-width readout)** — implemented first, then replaced: fixed slot
  count, positional bias, significantly worse measured accuracy.
- **Per-task classifiers** — rejected: the point is one general model across arbitrary
  label sets.
