# Hardware notes (24 GiB)

These measurements were made with v1's trainer, which trained the same pointer model at micro-batch 8 and gradient
accumulation 4. The pointer configs of `training/train.py` take the same 32 rows a step in microbatches of 4.

Measured on an RTX 3090 with Qwen3-1.7B (v7's torso), `max_length` 1024, 40 optimizer
steps each; the Qwen3.5 figures follow in their own subsection.

These numbers still hold at the current 4096 default. The collator pads to the longest
sequence in each batch, not to `max_length`, and the corpus is short (median 204
tokens, p99 634, longest 1811), so only ~0.15% of examples are affected by the cap at
all and no batch grows. Raising `max_length` is an inference-side change in practice;
it costs p95 latency, not training time.

| micro-batch | grad accum | checkpointing | throughput | peak VRAM |
| --- | --- | --- | --- | --- |
| **8** | **4** | **on** | **9.6 ex/s** | **5.8 GiB** |
| 16 | 2 | on | 9.3 ex/s | 6.7 GiB |
| 32 | 1 | on | 7.7 ex/s | 9.9 GiB |
| 8 | 4 | off | 1.6 ex/s | 36.3 GiB |

Two results worth internalising before you tune anything:

**Gradient checkpointing stays on** (the trainer has no switch for it). Off, it asked for 36 GiB, and on Windows
that does not fail cleanly: WDDM silently spills the excess to host memory over
PCIe, and the run gets *6× slower* while still appearing to work. The loss curve is
identical either way (1.4087 at step 40 in both), so this is purely a memory/speed
trade and checkpointing wins it outright on this card.

**A bigger micro-batch does not help.** The GPU is already compute-bound at 100%
utilisation and ~380 W with micro-batch 8; larger batches only add memory pressure.
The headroom is there for a longer `max_length` or a larger torso, not a wider batch.

v7's corpus (97,432 training rows, 1 epoch, effective batch 32) is 3,044 optimizer
steps, 3.0 h.

## The Qwen3.5 torso (v13-v19)

Both trained under WSL2 with the fused linear-attention kernels ([Setup](README.md#setup)).

| | corpus | steps | throughput | peak VRAM | one epoch |
| --- | --- | --- | --- | --- | --- |
| v13 | 97k classification rows | 3,044 | 0.28 steps/s | 6.2 GiB | 3.1 h |
| v14 | + 12.9k multi-step rows (~1.6k tokens on average) | 3,436 | 0.19 steps/s | 12.4 GiB | ~5 h |
| v18 | + 3.8k generated document questions (~1,000-word documents) | 3,551 | 0.17 steps/s | 12.4 GiB | ~5.5 h |
| v19 | + 6.2k answer-adequacy rows | 3,738 | 0.17 steps/s | 12.4 GiB | ~6 h |

The multi-step rows add about as many tokens as the whole classification corpus, which
is where v14's extra time and memory go. Labelling them with the 4B teacher took 55
minutes. On Windows' fallback kernels the same labelling spilled past the 24 GiB and
slowed to a crawl: the labeller writes each row as it goes and resumes, and its default
batch budget is 4,000 tokens for that reason.

## Batching by length

The trainer cuts each step's rows into microbatches of similar length, under a cap on padded tokens as well as rows
(`token_budget`, training/batching.py), which does the work v1's `group_by_length` did. Measured with v1's
length-grouped sampler, which sorted within megabatches of 400 rows, in tokens:

| corpus | padding, ungrouped | grouped | padded tokens saved |
| --- | --- | --- | --- |
| v7 (`train_v5.jsonl`) | 41.9% | 18.4% | 1.41x |
| v10 (v7 + 20k long documents) | 57.9% | 15.8% | 2.00x |
| v14 (v7 + 12.9k multi-step rows) | 74.6%* | 6.1%* | |

\* measured in prompt characters, the unit that sampler sorted by; token padding runs
higher, especially grouped. `python -m training.train --plan-only` prints the trainer's own padding estimate.

## Larger torsos

Measured for Qwen3-4B, projected beyond it from those two points. Weights are exact
(2 bytes/param); activations barely move with model size once checkpointing is on
(fitted 2.47 x params^0.09 at micro-batch 8), so **weights are the binding constraint**.

| torso | weights | activations | peak @ mb=8 | throughput | 1 epoch |
| --- | --- | --- | --- | --- | --- |
| 1.7B | 3.2 GiB | 2.6 GiB | **5.8 GiB** (measured) | 9.6 ex/s | 1.2 h |
| 4B | 7.5 GiB | 2.8 GiB | **10.3 GiB** (measured) | 5.1 ex/s | 2.2 h |
| 8B | 15.3 GiB | ~3.0 GiB | ~18.3 GiB | ~3.0 ex/s | ~3.8 h |
| 14B | 27.5 GiB | ~3.2 GiB | ~30.7 GiB, **does not fit** | n/a | n/a |

14B is a hard wall, not a tight squeeze: its weights alone exceed the ~23 GiB usable,
so no batch size or sequence length recovers it. Going beyond 8B needs 4-bit QLoRA
(14B weights drop to ~7.5 GiB), but 4-bit degrades exactly the probability
calibration the confidence scores depend on, so measure ECE before trusting it.

Throughput scales as params^-0.75, not linearly: larger torsos are cheaper than a
naive FLOP count suggests.

