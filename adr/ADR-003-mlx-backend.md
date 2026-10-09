# ADR-003: MLX backend with LoRA merged at load

**Status**: Accepted (upstream design)
**Date**: 2026-09
**Authors**: Strands Decider maintainers
**Related**: ADR-001, ADR-004

## Context

Serving on Apple silicon through torch/MPS is slow for this torso: the hybrid Qwen3.5
architecture (Gated DeltaNet + attention) has no fused kernels on MPS —
`flash-linear-attention` needs Triton, `causal_conv1d` needs CUDA — so every op is a
separate dispatch. Measured with v19 on an M4 Pro (median, one question at 222 /
1,118 / 4,094 input tokens): MPS 162 / 682 / 2,685 ms. mlx-lm runs the same
architecture on Metal with fused kernels including a recurrent Gated DeltaNet kernel.

## Decision

Add an `MLXEngine` that subclasses the torch engine and moves **only the torso
forward** to mlx-lm on Metal. Rendering, tokenisation, question-first windowing,
batching, temperatures, the masked softmax and the typed answers remain the torch
engine's own code; the head stays the checkpoint's own fp32 torch module on the CPU.
The LoRA adapter is **merged into the base weights at load** — `W + (α/r)·B·A` formed
in fp32 on the CPU (Metal's fp32 matmul is a reduced-precision fast path), rounded
once to the torso dtype — rather than carried unmerged as the torch path does.

## Rationale

- Same architecture, same artefacts, fused Metal kernels: the measured M4 Pro numbers
  fell to 113 / 486 / 1,764 ms — 1.4–1.6× faster than MPS at every length.
- Merging at load removes adapter plumbing from the hot path and keeps parity
  measurable: against v19 in fp32 on the CPU, no answer changes and the largest
  probability difference is 0.0138 (mostly the merge's rounding; in fp32, 0.0036).
- The merge refuses any adapter tensor it cannot represent exactly (embedding LoRA,
  DoRA magnitudes, rank patterns) rather than skipping it silently.

## Consequences

**Positive:** interactive Apple-silicon serving; one engine API across backends;
supply-chain-safe adapter handling.

**Negative:** a second inference backend to keep in behavioural parity (pinned by the
tiny-checkpoint test harness); the merge makes the numerical path differ from torch's
unmerged adapter by up to ~0.01 probability, which shows up as single-task flips on
tight benchmarks; mlx-lm must be pinned to a minor version (the backend depends on
its Qwen3.5 module layout).

## Alternatives Considered

- **MPS only** — rejected on the measured 1.4–1.6× gap.
- **CoreML / candle** — rejected: weaker fit for a hybrid recurrent torso and no
  parity story with the torch path.
