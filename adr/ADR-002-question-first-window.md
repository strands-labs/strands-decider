# ADR-002: Question-first claim on the context window

**Status**: Accepted (upstream design)
**Date**: 2026-09
**Authors**: Strands Decider maintainers
**Related**: ADR-001

## Context

The prompt is `state + question`, capped by a 4096-token window. The original fit
tokenised the state against the whole window first; a long state then left the
question a floor of 8 tokens — too few to hold the option list and the `<answer>`
marker the head pools at. All 19 `long_policy` items sat at exactly 1032 input tokens
scoring 0.316 against a 0.301 chance rate: the model was choosing among options it
could not see. Naïve concat-and-truncate is worse still: tokeniser truncation cuts
from the right, removing exactly the structurally required tail.

## Decision

`_fit` gives the **question first claim** on the window: the longest question
reserves `min(longest, max_question_fraction × window)` tokens, front-truncated so
the option tail and `<answer>` survive; the state tokenises into the remainder,
truncated from the right. Token offsets shift with truncation so pointer-head option
indices stay on the right tokens. `strict_window` refuses over-long prompts (HTTP
422) instead of truncating.

## Rationale

The question and its options are what make a task answerable; the state is the part
that can be sampled. Losing instruction text costs meaning; losing `<answer>` makes
the head read an arbitrary token — silently wrong rather than visibly degraded.

## Consequences

**Positive:** recovered ECE 0.115 → 0.079 on the affected tasks; long states degrade
gracefully instead of catastrophically.

**Negative:** a pathological question can starve the state — bounded by the
`max_question_fraction` cap, but the trade-off is real and tunable.

## Alternatives Considered

- **State-first fit** — the original; measured failure above.
- **Reject all over-length prompts** — available as `strict_window`; too strict for
  interactive use where a degraded answer with confidence beats a 422.
