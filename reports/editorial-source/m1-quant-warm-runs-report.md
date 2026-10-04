# M1 Quantization: What Actually Helps?

We benchmarked four ways of running the Strands Decider 2B model on an
M1 MacBook Air: standard bf16, fp16, int8, and int4. We measured speed,
answer quality, and memory use across six input sizes, from short prompts
to 4000-token documents. Two findings matter:

1. **Switching from bf16 to fp16 makes inference about 20% faster** at
   every input size, with no change in answer quality. This is the only
   free speedup in the data.
2. **Quantization makes the model slower on the M1, not faster.**
   int8 is 10–35% slower than full precision; int4 is slower still. What
   quantization does deliver is memory savings (about 25% for int8,
   40% for int4), so int8 has one legitimate use: when you're tight on
   RAM and willing to pay a latency cost.

Everything below is the evidence for those two sentences.

## Speed

![Latency by precision across input sizes](img/latency.png)

Each bar is the median time for one full pass over the input, in seconds,
measured over 5 timed runs on AC power. Reading left to
right, inputs grow from 256 tokens with 1 question to 4000 tokens with
8 questions.

The blue fp16 bar is the shortest in every group. The orange (int8) and
red (int4) bars are taller than both full-precision bars everywhere —
quantization never wins on this chip, and the gap widens with input
size. At the largest input, int8 takes 15.1 seconds versus 8.4 for fp16.

Why this happens: on the M1 the bottleneck is computation, not memory
bandwidth. Quantization halves the weight data but adds decompression
math to every operation, and that math costs more than the bytes saved.
A related finding: bf16 itself carries a small compute penalty on M1
versus fp16 — identical bytes moved, ~20% slower — which is exactly why
fp16 is the fastest bar.

## Memory

![Peak memory by precision](img/memory.png)

Peak memory during the largest test. bf16 and fp16 use identical memory
(same 2 bytes per weight). int8 cuts it by about 23%, int4 by about 36%
at this size; averaged across all six input sizes the savings are 25%
and 39%. This part of quantization works exactly as advertised, and the
numbers are deterministic.

## Answer quality

No chart needed — it's simple. Across 72 test questions, bf16, fp16, and
int8 each answered 65 correctly (90.3%), with int8 changing zero answers
relative to full precision. int4 dropped to 62 correct (86.1%), flipping
3 answers. So int8 is safe on quality; int4 is measurably worse.

## What to do

- **Default to fp16 on Apple silicon.** About 20% faster than bf16,
  identical answers, identical memory.
- **Use int8 only if memory is the constraint**, accepting a 10–35%
  slowdown in exchange for ~25% less memory.
- **Don't use int4 on M1.** Slower *and* less accurate — no upside.
- These are M1-only results. NVIDIA GPUs have dedicated int8 hardware
  that changes the trade-off, so this conclusion shouldn't travel to
  CUDA without its own measurements.

## How it was measured

Two runs on a MacBook Air (M1, 16 GB), both on AC power at full charge,
about 17 minutes each: one from the standard bf16 checkpoint (bf16, int8,
int4), one from an fp16-converted checkpoint (fp16, int8, int4). MLX
0.32.3 with affine int8/int4 quantization; the fp32 decision head,
embeddings, and normalization layers were left unquantized. Each
precision ran in a fresh process with 2 warmups and 5 timed repetitions;
answer quality was scored on 72 JevBench tasks.

## Caveats

- M1 only. Don't generalize to other chips without measuring.
- Quality sample is 72 tasks — enough to separate these precisions, not
  to certify small differences.
- These numbers describe full input encodes (the model answers in a
  single forward pass), not token-by-token streaming.
