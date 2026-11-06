# Experiment history

This document records the experiments behind the reference recipe, from v1 to v19, and
what each one measured. Each statement is as of its own version. The current results are
in [evaluation/README.md](../evaluation/README.md). The v20 run has no narrative here: its predictions
and its outcome are in [PREREGISTRATION-v20.md](preregistrations/PREREGISTRATION-v20.md#outcome-added-after-the-run).
The list of preregistrations is in [README.md](README.md). Paths are relative to the
repository root unless they are links. A module path such as `data/synth.py` is relative
to `src/hobson/`.

## Overview

### What has moved the benchmark

Every intervention since v1, measured on JevBench, falls into one of two lists.

**Moved it:** torso size (Qwen3 1.7B to 4B, +0.043 at matched corpus and steps); the
readout (v6's KL anchor and head seeding, +0.013; v7's pointer readout, +0.017); torso
generation (Qwen3.5-2B in place of Qwen3-1.7B, v13, +0.013); multi-step documents
distilled from a teacher (v14, +0.017, with a significant gain on a held-out multi-hop
source); and generated questions over workplace documents (v16, +0.009, with a
significant gain on generated questions from unseen domains, and the hard tier's
`trap` and `tradeoff` families, but a loss on the multi-step sets); and replay toward
v14's answers on the multi-step rows (v17, +0.004 on JevBench, restoring MuSiQue,
ContractNLI and BoardgameQA past v14 but not HotpotQA); and answer-adequacy judgements,
a skill nothing in the corpus had asked for (v19, +0.013, all of it in the standard
tier, with a significant fall in Brier score: 0.388 to 0.342, p < 0.0001).

**Did not:** data volume, instruction variants and yes/no breadth (v4); generated
rule-chaining data (v5's RuleTaker, v9) and generated minimal pairs (v10); labelled
question transforms (v11a, which lost); KL toward the frozen model on changed questions
(v11b, lost); distillation on short classification (v12); public rule-application data
(v15: ShARC and ConditionalQA); an instruction-tuned torso, twice (the v6 arm, v8); and
more generated questions aimed at the weakest skills (v18: level on JevBench, although
those skills rose 0.16 on unseen domains and `long_policy` went from 7 to 10 of 19).

No single JevBench gain above is significant by itself — 231 tasks cannot resolve less
than about +0.043 ([Scaling, and the measurement that hid it](#scaling-and-the-measurement-that-hid-it)) — but the held-out
HotpotQA gain behind v14 is (p = 0.0007), v16's on unseen-domain generated questions
(p < 0.0001), and v19's on held-out adequacy judgements and on JevBench's Brier score
(both p < 0.0001). The same section explains why several early
"no change" results were unresolvable rather than negative. The runs themselves are
under [Experiments since v7](#experiments-since-v7) and [Version history](#version-history).

## What the torso knows untrained

SemIf (`TheoLeeCJ/openjev`, MIT) scores 0.810 on the same 231 tasks from a **frozen**
Qwen3.5-4B with no trained parameters. It renders the decision through the chat template
as JSON, labels the options `A`-`P`, takes one forward pass and reads the logits of those
letters. Same computational contract as Hobson — one forward pass, no decoding loop,
`enable_thinking=False`, p50 198 ms. It is not a reasoning model. (The 0.887 entry named
"OpenJev (thinking)" is a different system with a think budget.)

Read the same way here, on the same tasks:

| torso | frozen, read SemIf's way | trained here |
| --- | --- | --- |
| Qwen3-1.7B | 0.558 | 0.667 (v7) |
| Qwen3.5-2B | 0.658 | 0.680 (v13), **0.697** (v14) |
| Qwen3-8B | 0.684 | |
| Qwen3.5-4B | **0.805** | |

The Qwen3.5-4B run reproduces SemIf's published result (187/231, the same verdict on
230 of 231 tasks). The board now runs untrained controls of its own, read from raw
logits, and they agree: Qwen3-8B 0.684 (ours 0.684), Qwen3-1.7B 0.541, Qwen3-4B-Instruct
0.697, and a separate frozen Qwen3.5-4B entry (`jobe`) at 0.810. Three readings:

- **Generation matters more than size.** Qwen3.5 at 4B beats Qwen3 at 8B by 0.12, and at
  matched size Qwen3.5 is +0.10 over Qwen3. Within Qwen3.5, size still matters: 2B to
  4B is +0.147.
- **Training adds less on a better torso.** v7's recipe added about +0.11 over Qwen3's
  frozen reading and +0.02 over Qwen3.5's (v13); the multi-step corpus brought that to
  +0.04 (v14).
- **Trained and frozen readouts fail differently.** At 1.7B the trained readout won the
  classification families and the frozen instruct reading the compositional ones
  (`policy` 0.583 against 0.833, `trap` 0.625 against 0.750, measured on v5). The
  frozen *base* model — our actual starting point — scores 0.500 on `policy`, so training
  did not destroy that ability: instruction tuning supplied it, and fine-tuning an
  instruct torso loses it again (v8).

The frozen Qwen3.5-4B is also the teacher for v14's multi-step rows. A frozen Qwen3-8B,
tested for the same role, reached only 0.684 and was not used.

## Scaling, and the measurement that hid it

For most of this project the working conclusion was that parameter count is not the
lever. That conclusion was wrong, and the way it went wrong is the most useful thing
recorded in this file.

### The controlled test

`checkpoints/head-linear` and `checkpoints/head-4b-linear` were trained as a head
ablation: same corpus (`train_fix2.jsonl`), same 2170 steps, same LoRA rank, same
learning rates, same `max_length`. Only the torso differs. Neither had ever been run
against an external benchmark.

| | 1.7B | 4B | |
| --- | --- | --- | --- |
| easy (48) | 1.000 | 1.000 | saturated |
| standard (72) | 0.778 | **0.833** | +0.056 |
| hard (111) | 0.396 | **0.450** | +0.054 |
| **JevBench overall** | 0.6407 | **0.6840** | **+0.043** |
| macro | 0.6273 | 0.6755 | +0.048 |

27 tasks newly correct, 17 newly wrong, net +10 of 231. The gains land where the
theory says they should — `adversarial` +0.333, `policy` +0.167, `long_policy` +0.158,
`ambiguous` +0.143, `trap` +0.125 — and both unsaturated tiers move together.

Stated honestly: **McNemar gives p = 0.174**, not significant. Discordance is high (44
pairs) so +10 net is not enough on its own. What supports it is convergent rather than
statistical — two tiers moving together, macro accuracy moving, the gains sitting in
the predicted families, and the same direction and rough magnitude appearing in a peer's
independently published scaling curve (Open-Jev 2B to 9B, +0.130). Our 4B lands at hard
0.450, between Open-Jev 2B (0.414) and 9B (0.595). We are on that curve, not off it.

### Why it was missed

Every "no improvement" result in the [Version history](#version-history) was scored on this repo's own held-out corpus. That
corpus does not rank models:

| model | JevBench | internal held-out |
| --- | --- | --- |
| head-4b-linear (4B) | **0.6840** | 0.8450 |
| head-linear (1.7B) | 0.6407 | **0.8500** |
| arm: lm_head init | 0.6407 | 0.8375 |
| arm: KL to frozen | 0.6407 | **0.8525** |
| v5 @3072 | 0.6364 | 0.8425 |
| arm: instruct torso | 0.6190 | 0.8375 |

The internal metric ranked the 4B **below** the 1.7B, and ranked the best internal
model (KL, 0.8525) level with one scoring 0.0150 lower internally. It is dominated by
classification tasks the architecture has saturated, so it measures fit to the training
distribution rather than capability. The scaling axis was retired on the strength of it.

### The resolution limit

At 231 tasks, with the agreement rates these models actually show, McNemar cannot reach
p < 0.05 below roughly **+0.043 accuracy**:

| discordant pairs | split needed for p<0.05 | net items | accuracy |
| --- | --- | --- | --- |
| 14 | 12 vs 2 | +10 | +0.043 |
| 30 | 21 vs 9 | +12 | +0.052 |
| 44 | 29 vs 15 | +14 | +0.061 |

The readout interventions in this project produce about +0.004. They are an order of
magnitude below the resolution of the instrument, so "no significant change" was never
evidence of no change. Only effects the size of a torso jump were ever detectable.

### What this does not overturn

The corpus results stand and are if anything worse than reported: `head-linear` is two
corpus generations old (`fix2`) and scores 0.6407, matching the best v5-corpus model.
Everything v4 and v5 added — instruction variants, noul breadth, RuleTaker, ~30k
examples — bought nothing externally. What eventually did move the benchmark through
data was different in kind: multi-step documents from real sources, carrying a
teacher's distributions (v14).

### Practice going forward

Benchmark externally before drawing a conclusion; it is about four minutes per model
against a local server. Use the internal holdout for training diagnostics and
calibration fitting, not for ranking. And treat any single-run difference below +0.04
on 231 tasks as unresolved rather than as a negative result.

## Experiments since v7

Each run below changed one thing against the recipe before it and was judged against a
bar fixed before it trained. Seven runs scored above v7 — v13 to v19 — and one, v19,
cleared its pre-registered bar. v13, v14, v16, v17 and v18 were each promoted on the
strength of their other results, pending a seed replicate; v19 became the default by
its own rule.

| | **v7** | v8 instruct | v9 rule synthesis | v10 minimal pairs | v11a question transforms | v11b KL on changed questions | v12 distilled from Qwen3.5-4B | v13 Qwen3.5-2B torso | v14 v13 + multi-step documents | v15 v14 + public policy data | v16 v14 + generated documents | v17 v16 + replay toward v14 | v18 v17 + weak-skill questions | v19 v18 + answer adequacy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| JevBench accuracy | 0.6667 | 0.6580 | 0.6580 | 0.6277 | 0.6190 | 0.6234 | 0.6320 | 0.6797 | 0.6970 | 0.6926 | 0.7056 | 0.7100 | 0.7100 | **0.7229** |
| tasks vs v7 | | -2, p = 0.856 | -2, p = 0.815 | -9, p = 0.150 | -11, p = 0.007 | -10, p = 0.013 | -8, p = 0.096 | +3, p = 0.72 | +7, p = 0.31 | +6, p = 0.45 | +9, p = 0.20 | +10, p = 0.15 | +10, p = 0.14 | +13, p = 0.079 |
| macro | 0.6601 | 0.6381 | 0.6430 | 0.6196 | 0.6070 | 0.5983 | 0.6230 | 0.6847 | 0.6898 | 0.6881 | 0.7154 | 0.7195 | 0.7106 | **0.7254** |
| ECE | 0.0678 | 0.0531 | 0.0580 | 0.0605 | 0.0822 | 0.0953 | 0.0934 | 0.0853 | 0.0855 | 0.0633 | 0.0694 | 0.0828 | 0.0789 | **0.0522** |
| Brier | 0.4422 | 0.4554 | 0.4481 | 0.4696 | 0.4654 | 0.4713 | 0.4542 | 0.4645 | 0.4383 | 0.3980 | 0.3961 | 0.3924 | 0.3882 | **0.3421** |
| ordinal MAE | 0.5922 | 0.6772 | 0.5560 | 0.5931 | 0.7328 | 0.6036 | 0.5523 | 0.6211 | 0.6446 | 0.6372 | 0.6638 | 0.5708 | 0.6443 | **0.4270** |
| paraphrase consistency | 0.8333 | 0.7778 | 0.8333 | 0.8056 | 0.8333 | 0.7778 | 0.8056 | 0.8333 | 0.8611 | **0.8889** | 0.8333 | 0.8056 | 0.8333 | 0.8611 |
| held-out tasks, real questions | 0.653 | | | | 0.646 | 0.650 | **0.674** | 0.646 | 0.637 | 0.633 | 0.629 | 0.640 | 0.641 | 0.647 |

None of the JevBench differences is significant on its own ([The resolution limit](#the-resolution-limit)).

### v8: the instruct torso, retested under the pointer readout

`Qwen/Qwen3-1.7B` in place of `Qwen3-1.7B-Base`, nothing else changed. The first
instruct arm ([The v6 arms](#the-v6-arms-three-retrains-on-the-readout)) lost on `policy`, `tradeoff`,
`adversarial` and `long_policy` — the families the pointer readout later improved —
so those losses might have been the slot head's. The bar, set in advance: accuracy
holds and those four families do not fall back.

It failed. Two of the four recovered (`adversarial`, `long_policy`), confirming the
readout mattered there. The other losses reproduced **to the task**, through a
completely different readout:

| family | instruct arm (slot head) | v8 vs v7 (pointer) |
| --- | --- | --- |
| `policy` | -0.083 | -0.083 |
| `tradeoff` | -0.167 | -0.167 |
| `routing_hard` | -0.400 | -0.400 |

So those are costs of instruction tuning itself. It helps `trap` (+0.125 here, +0.250
in the first arm) and gives the best ECE recorded, 0.053, but not enough to pay for
them. Instruction tuning on this recipe is a twice-measured negative.

### Qwen3.5-2B: stopped at the spike

Same hidden size as Qwen3-1.7B, so the pointer readout would port untouched, and the
text tower loads. It stopped on three findings, any one of which makes it days of work
rather than a config change:

- **The shared-prefix cache breaks.** 18 of 24 layers are Gated DeltaNet, which carry
  a recurrent state rather than a KV cache. The cache expansion skips them, their
  `conv_states` stay at batch 1, and the question batch fails to concatenate. It
  failed loudly — the prefix-cache equivalence test would have caught a silent version.
  (Fixed since: the fork now repeats the recurrent state too; [Design decisions](../docs/architecture.md#design-decisions-worth-knowing).)
- **LoRA reaches 6 of 24 layers.** The DeltaNet projections (`in_proj_qkv`,
  `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) are not in the target list.
- **Its fused kernels are not available here** (`causal_conv1d`, `chunk_gated_delta_rule`
  fall back to reference PyTorch on this Windows machine), at an unmeasured training cost.

It would also have changed three variables at once — generation, size and instruction
tuning — against v7. Not pursued then; all three blockers were solved for v13: the
engine turns the cache off for hybrid torsos, `lora_targets` is configurable, and
training runs under WSL2.

### v9 and v10: synthetic documents

Both pre-registered before generating data or training, with predictions and the
failure signature written down: [PREREGISTRATION-v9.md](preregistrations/PREREGISTRATION-v9.md),
[PREREGISTRATION-v10.md](preregistrations/PREREGISTRATION-v10.md). Both added 20,000 generated items to
v7's corpus and changed nothing else. Ground truth is computed rather than annotated,
and every label is re-derived from the rendered document by a separate verifier
(`data/checks/verify_synth_labels.py`, `data/checks/verify_synth_pairs.py`) at 0% disagreement.

**v9** (`data/synth.py`) taught stated-rule execution — thresholds, unit conversion,
date windows, banded lookups, checking a worked calculation — in short tabular
documents, aimed at `temporal_numeric`, `multi_hop` and `judge_hard`.

**v10** (`data/synth_pairs.py`) dropped arithmetic and taught cross-referencing and
requirement checking in ~2,200-character documents with superseded tables, glossaries
and misleading requester notes. Every item is half of a **minimal pair**: the same
document with the one decisive fact changed and the answer flipped, so no surface
feature can separate the halves.

| | v9 | v10 |
| --- | --- | --- |
| targeted families (PRIMARY) | 16/50 -> **16/50** | 14/35 -> **11/35** |
| untargeted families (TRANSFER) | 35/68 -> 31/68 | 35/68 -> 31/68 |
| everything else (CONTROL) | 103/113 -> 105/113 | 103/113 -> 102/113 |
| synthetic, same distribution | 0.436 -> **0.972** | 0.441 -> **0.951** |
| synthetic, held-out variants | 0.533 -> 0.673 | 0.447 -> 0.747 |

Both hit the failure named in advance: *learned the generator, moved nothing on the
benchmark*. v9's targeted families did not change by one task — 48 of 50 verdicts
identical, the other two cancelling. v10's went backwards: `multi_hop` 7/18 -> 4/18,
and all nine lost tasks are in the hard tier.

v10's pairs rule out the obvious explanation, that the model learned the generator's
surface:

| v10, synthetic eval | item | both halves right | same answer to both |
| --- | --- | --- | --- |
| cross-reference, trained domains | 0.999 | 0.998 | 0.002 |
| cross-reference, **unseen domain** | 0.901 | 0.821 | 0.020 |
| requirement check, **unseen requirement** | 0.593 | 0.187 | 0.813 |
| *v7 on the unseen requirement* | *0.473* | *0.004* | *0.992* |

It reads the decisive fact, and carries that to vocabulary it has never seen. What it
does not carry it to is a real document. The domains vary words, but every generated
document shares one *structure*, and that structure is what was learned. Minimal pairs
defeat shortcuts inside a distribution; they cannot make the distribution wider.

**This is the third negative with the same shape**: RuleTaker (v5), rule execution
(v9), minimal pairs (v10). Each reached near its ceiling on its own generator and moved
JevBench by zero or less. Templated synthetic data does not transfer to real documents
for this model, and that should be the default expectation for a fourth attempt.

Two further findings from these runs are worth keeping:

- **`temporal_numeric` is capability-bound, not data-bound.** No non-reasoning system
  in the JevBench table passes 0.533 on it, Jev itself scores 0.27, and within v9 the
  held-out variants that transferred were rule-*reading* (a date clamp: 0.50 -> 0.91)
  while the arithmetic ones did not move (rounding mode, operation order). `multi_hop`
  and `judge_hard` are different — non-reasoning systems reach 0.88, and one 2B system
  scores 0.82 on `judge_hard` against v7's 0.41 — so the headroom is real there, just
  not reachable by this route.
- **The internal validation accuracy was misleading again.** v9 reached 0.895 against
  v7's 0.835, because 17% of its corpus was cleanly labelled synthetic data that is
  easy to fit, while JevBench fell ([Limitations](../evaluation/README.md#limitations)).

### v11: teaching the model to read the question

`evaluation/question_sensitivity.py` keeps a state and its options fixed and changes only
the question. **v7 gives the same answer to a changed question 94-99% of the time.**
Asked which option a text does *not* fit, it picks the one that fits (0.730 on held-out
tasks) where the frozen base avoids it (0.028); asked which option is listed first, it
scores 0.080 against the frozen base's 0.700. The corpus taught this: each state is
asked one fixed question per task, so state and options determine the label alone.
Where the question varies per row (boolq, MNLI), v7 reads it.

Two arms, pre-registered in [PREREGISTRATION-v11.md](preregistrations/PREREGISTRATION-v11.md), kept v5's
real states and varied only the question (`data/question_transforms.py`: is "X" the
right/wrong answer, the one of two that does NOT apply, a threshold on a rating, a
wrapper that keeps or flips a yes/no question — 30,135 rows, every label re-derived
independently at 0% disagreement). The probe's own question forms were held out.

- **v11a** replaced 30% of v5 with the transformed rows.
- **v11b** kept v5 and added the same prompts as KL-only rows, trained toward the frozen
  torso's own reading of the changed question (`kl_only_files`).

| probe, held-out choice tasks | frozen base | v7 | v11a | v11b |
| --- | --- | --- | --- | --- |
| same answer as the real question (first / last / NOT) | 0.43 / 0.35 / 0.07 | 0.95 / 0.94 / 0.95 | 0.93 / 0.91 / **0.11** | 0.95 / 0.95 / 0.92 |
| "listed first" accuracy (chance 0.184) | 0.700 | 0.080 | 0.090 | 0.062 |
| NOT picks the class that fits | 0.028 | 0.730 | **0.115** | 0.745 |

**Both arms lost on JevBench**, the first losses in this project clearly outside
task-sampling noise (see the caveat below), and by different routes — only 2 tasks were
lost by both.

**v11a learned the concept of negation and not the skill of reading questions.** NOT
transferred to a phrasing and option count it never saw; "listed first/last", a kind
of question never trained, did not move at all. And the negation it learned looks
lexical: 7 of its 13 JevBench losses have a negation word in the question, against 41
of the other 218 (Fisher p = 0.007), and all 6 lost yes/no tasks flipped yes to no.
JevBench questions are full of negation that reverses nothing ("treat unproved
conditions as not satisfied"). `policy` fell from 10/12 to 7/12.

**v11b moved nothing on choice and damaged yes/no.** The frozen torso's option-number
readout is near chance on yes/no, so anchoring to it taught the model to stop following
the original question without following the new one; on JevBench it answers yes on 60
of 74 yes/no tasks (v7 53, correct 35).

**What carries forward:**
- the probe itself, which exposed a real property no benchmark number showed;
- labelled question transforms teach their concepts keyed to their words — breadth of
  question kinds, not more of a few, is what "read the question" would need;
- the frozen torso is a usable teacher for choice at best, not for yes/no.

**A caveat for every comparison here.** McNemar treats v7's per-task answers as fixed;
it covers which tasks were sampled, not training noise. v7 has never been retrained
with a different seed, and every run since has scored below its 154 (152, 152, 145,
143, 144) — which is also what regression toward the mean would look like if v7's run
was a favourable draw. The v11 losses are large enough that seed noise is unlikely to
explain them entirely, but a seed replicate of v7 is the missing measurement. (Later
runs broke the pattern: v13 scored 157 and v14 161.)

### v12: distilling a frozen Qwen3.5-4B

A teacher had to be better than v7 at this interface. Frozen models read SemIf's way
([What the torso knows untrained](#what-the-torso-knows-untrained)) were measured against a bar fixed at 0.75 before
each run: Qwen3-8B reached 0.684 (ECE 0.305) and failed it; Qwen3.5-4B reached 0.805
(ECE 0.056, better calibrated than v7's 0.068) and passed.

v12 ([PREREGISTRATION-v12.md](preregistrations/PREREGISTRATION-v12.md)) kept v7's recipe and replaced its
KL anchor with the 4B teacher's option distributions (`data/teacher.py`, prompts checked
byte-identical to SemIf's; `teacher_file`, `teacher_weight: 1.0`), labelled on the 91,408
training rows with at most 16 options.

**Distillation worked, and did not reach JevBench.** On the held-out tasks v12 agrees with
the teacher more (0.727 against v7's 0.696) and is more accurate than any model here has
been: 0.674 against v7's 0.653, `hate_severity` 0.492 -> 0.563, `sarcasm` 0.629 -> 0.656.
On JevBench it gives the teacher's answer on 156 tasks against v7's 157 — no movement —
and scores 146/231 (5 gained, 13 lost, p = 0.096).

The reason showed up before training: on the held-out file the teacher is no better than
v7 at short classification (0.640 against gold). Its advantage is on long documents and
multi-step judgement, and the training corpus contains neither. Distillation transfers
behaviour on the prompts it is given; these carried the teacher's feel for short-text
judgements, not what makes it good at JevBench.

### v13: a Qwen3.5-2B-Base torso

v7's recipe with only the torso changed, to `Qwen3.5-2B-Base` (1.90B parameters,
[PREREGISTRATION-v13.md](preregistrations/PREREGISTRATION-v13.md)). Its 18 Gated DeltaNet layers get LoRA
on their own projections (`lora_targets`), and it trains under WSL2, where the fused
linear-attention kernels run: 0.28 steps/s, the same as v7, where the Windows fallback
path is about 2.3x slower. Serving then used batched encoding, because the shared-prefix
cache could not yet broadcast recurrent state; it can now ([Design decisions](../docs/architecture.md#design-decisions-worth-knowing)).

**157/231 (0.6797) — the first retrain above v7, and a tie.** 17 gained, 14 lost,
p = 0.72. Macro accuracy 0.685 is the best recorded; held-out accuracy 0.646 and
held-out ECE 0.056. The pre-registered bar was 162.

The hypothesis was that training adds as much on Qwen3.5 as on Qwen3 (+25 tasks over
the frozen reading). It added +5 (frozen 152, trained 157). v13 gained in `ambiguous`
(1 -> 4), `trap` (5 -> 7), `probability` (3 -> 6) and `adversarial` (4 -> 5), and lost
`tradeoff` (4 -> 1) and `long_policy` (8 -> 6); the frozen torso's lead on
`temporal_numeric` and `judge_hard` did not survive training (both back at v7's level). The
question-sensitivity probe reads the same as v7's: ignoring the question is taught by
the corpus, whatever the torso.

### v14: multi-step documents with a teacher

The failing JevBench families need three things the corpus never contained: sequential
lookups through long procedures, conditions that must all hold, and rules that override
others. v14 ([PREREGISTRATION-v14.md](preregistrations/PREREGISTRATION-v14.md)) adds 12,909 rows of them
to v13, from `data/multistep.py`:

- **ContractNLI**: full NDAs; is each of 17 claims entailed, contradicted or not
  mentioned. Balanced per claim: unbalanced, answering each claim with its usual label
  scores 0.679 without reading the contract.
- **MuSiQue** (full release): 2-4 hop questions with distractor paragraphs, each with an
  answerable and an unanswerable version; the options include the earlier hops' answers
  and "cannot be determined".
- **BoardgameQA**: conflicting rules resolved by stated preferences (generated; capped at
  a quarter of the new rows).

Each row carries its gold label and the frozen Qwen3.5-4B teacher's distribution: the
v12 mechanism, now on prompts where the teacher is clearly better than v13 (MuSiQue
0.557 against 0.202). **HotpotQA was held out entirely** as the transfer test.

| evaluation (never trained on) | v13 | teacher | v14 |
| --- | --- | --- | --- |
| **HotpotQA, held-out source** | 0.705 | 0.846 | **0.750** (+100 / -57, p = 0.0007) |
| MuSiQue | 0.202 | 0.557 | 0.852 |
| ContractNLI (per-claim guess 0.679) | 0.564 | 0.772 | 0.842 |
| BoardgameQA | 0.519 | 0.596 | 0.752 |
| JevBench | 157 | 186 | **161** (bar 163) |

**Multi-step reading transfers to a source the model never saw**: the first corpus change
in this project to show transfer outside its own generator or dataset. JevBench moved the
same way but by less: 8 gained, 4 lost against v13 (p = 0.39), hard tier 49 -> 52,
`multi_hop` 6 -> 9 of 18, with the best Brier (0.438), macro (0.690) and paraphrase
consistency (0.861) recorded. It missed the pre-registered bar of 163 by two tasks, and
was named the leader anyway, on the held-out transfer and those metrics; its JevBench
gains remain inside the noise a seed replicate would measure.

The in-distribution scores beat the teacher by so much that they were checked for
artefacts of the conversion. Option position: none (0.862 with shuffled options). An
unanswerable question's answer string is often simply absent, and where several options
share the expected answer type the task is harder; both cues explain part of the gain
(0.705 and 0.820 in the rows where they do not help) and neither explains most of it.
The held-out HotpotQA result, whose conversion is different, is the one to rely on.

The question-sensitivity probe is unchanged: documents taught reading documents, not
reading questions.

### v15: public rule-application data

decider-2b (Mapika), on the same Qwen3.5-2B-Base torso, is level with v14 at its v10
(162 against 161 on our harness) and 14 tasks ahead at its v11 (175), and the one thing
v11 adds is a stage of harder decisions over documents and policies that is not
released. v15 ([PREREGISTRATION-v15.md](preregistrations/PREREGISTRATION-v15.md)) tried the nearest
public substitute, from `data/policy.py`: **ShARC** (short rules from government
websites applied to a user's situation, balanced so that neither the dialogue's shape
nor its last answer predicts the label) and **ConditionalQA** (long gov.uk pages, answers
that hold only under a condition), 4,566 rows, gold labels only.

It learned ShARC (0.576 -> 0.752 on unseen rules) and ConditionalQA's "no" answers, but
not ConditionalQA's conditions (0.056 -> 0.111), and JevBench did not move (160/231; 7
gained, 8 lost). MuSiQue's answerable questions regressed (0.843 -> 0.806). Short rule
data teaches short rule application.

### v16: generated document questions

So v16 ([PREREGISTRATION-v16.md](preregistrations/PREREGISTRATION-v16.md)) generated the thing itself,
the way decider built it: realistic workplace documents (34 domains, 22 kinds, a median
1,056 words in numbered sections), each with 3-4 questions targeting one skill apiece —
all conditions, exception, precedence, dates, numbers, lookup chain, tradeoff,
underdetermined, surface trap, rubric — written by Qwen3.6-27B (decider's writer) with
thinking, and kept only when two answers by a different model, Qwen3.5-397B-A17B, agreed
with the writer's (`data/generators/gen_documents_openrouter.py`; 1,000 documents, 2,804 of
3,486 questions kept, about $130 in API fees at OpenRouter list prices; open-weight Apache-2.0 models only). Two
pilots chose the setup: Nova Premier writing and verifying its own questions left 7 of
30 hand-checked labels wrong; this pairing, 1 of 30. Before training, yes/no answers
were balanced within each skill, and choice rows where the right answer is also the
longest option (34% against a 23% chance rate — the writer describes the right answer
most fully) were thinned to chance (`data/generated.py`): 2,148 training rows. The raw
export, with every verifier answer, is committed in `data/generators/gen_v16/`.

| | v14 | v16 |
| --- | --- | --- |
| **generated questions, four held-out domains (350)** | 0.646 | **0.837** (+83 / -16, p < 0.0001) |
| — "yes" / "no" / choice | 0.873 / 0.435 / 0.640 | 0.873 / 0.823 / 0.831 |
| JevBench | 161 | **163** (bar 167; +10 / -8) |
| hard tier | 52 | 55 |
| HotpotQA, held-out source | **0.750** | 0.712 (+49 / -85, p = 0.002) |
| MuSiQue | **0.852** | 0.827 (p = 0.0002) |

**The generated documents taught a skill that transfers to unseen domains**, and not
as a shift in answer prior: "yes" is unchanged while "no" nearly doubles. On JevBench
it moved `trap` (6 -> 8 of 8) and `tradeoff` (2 -> 4 of 6), with the best Brier and
macro and a top confidence band right as often as it claims (v14's was not) — but by
two tasks overall, inside the noise. And it cost the multi-step sets: HotpotQA and
MuSiQue's answerable questions regressed, the named failure mode of its
pre-registration, so by the rule v14 stayed the leader; v16 was made the default on the
rest. v15 lost MuSiQue's answerable questions by the same amount with entirely
different data.

### v17: replay toward v14

decider protects each parent by training replayed rows toward the parent's own answer
distribution instead of their labels. v16 had v14's multi-step rows, with the frozen
4B's distributions on them, but nothing asked it to behave like v14 on them. v17
([PREREGISTRATION-v17.md](preregistrations/PREREGISTRATION-v17.md)) changed only that: the teacher
distributions on those 12,909 rows are v14's own (`data/replay.py`; v14 gives the gold
answer on 0.882 of them, and its distributions are soft, median top probability 0.74).

| | v14 | v16 | v17 |
| --- | --- | --- | --- |
| MuSiQue | 0.852 | 0.827 | **0.880** (+80 / -16 on v16) |
| ContractNLI | 0.842 | 0.840 | **0.865** |
| BoardgameQA | 0.752 | 0.757 | **0.817** (+71 / -17 on v16) |
| **HotpotQA, held-out source** | **0.750** | 0.712 | 0.723 (+50 / -40 on v16, p = 0.34) |
| generated questions, four held-out domains | 0.646 | 0.837 | **0.840** |
| JevBench | 161 | 163 | **164** |

**Replay restored the sources it replayed and passed v14 on all three** — v14's own
distributions made better targets than the frozen 4B's — and cost nothing on the
generated questions. **It did not restore HotpotQA**, the one pre-registered prediction
it missed: whatever v16's generated documents changed in how the model reads paragraph
sets it never saw, pinning its answers on the ones it did see did not hold it. v17 was
made the default on the rest: the best JevBench score, Brier and macro recorded, and
the best held-out short-task accuracy since v12.

### v18: questions on the weakest skills

A 100-document pilot of new skills (`data/generators/gen_mixed_pilot/`) found v16 weakest on
the rule version in force (0.53), missing facts (0.60), tie-breaks (0.67), facts across
artefacts (0.69) and running totals (0.75). v18
([PREREGISTRATION-v18.md](preregistrations/PREREGISTRATION-v18.md)) added 1,667 generated questions to
v17's recipe, most from a run that drew those five skills four times as often as the
rest (`data/generators/gen_weak/`: 692 documents, $96 on OpenRouter), written under a new rule
that the case must state every fact the deciding rules depend on. 27 of 30 hand-checked
labels were right; the other three took a fact the document leaves open as settled.

| | v17 | v18 |
| --- | --- | --- |
| **generated questions, four held-out domains, v18's set (247)** | 0.660 | **0.761** (+32 / -7, p = 0.0001) |
| — the five weak skills (133) | 0.586 | **0.744** (+26 / -5) |
| — missing facts (29) | 0.310 | **0.655** (+10 / -0) |
| generated questions, v16's set (350) | 0.840 | **0.863** |
| MuSiQue / ContractNLI | 0.880 / 0.865 | **0.892** / **0.866** |
| BoardgameQA | **0.817** | 0.801 (+26 / -40, p = 0.11) |
| **HotpotQA, held-out source** | **0.723** | 0.705 (+43 / -60, p = 0.11) |
| JevBench | 164 | 164 (+9 / -9); hard 56, standard 60 |

**The targeted skills moved a long way on documents from unseen domains, and JevBench
did not move.** The one family that did, `long_policy` (7 -> 10 of 19), is the closest
in form to the generated documents; `temporal_numeric`, which the rule-version and
running-total questions aimed at, fell from 3 to 2. The generated evaluation shares the
generator's style and label noise, so the size of the gain there overstates the skill.
HotpotQA missed its pre-registered floor by one question, so by the rule v17 stayed the
default; v18 was made the default on the rest: level on JevBench with the most hard-tier
tasks and the best Brier, ahead on both generated sets and on MuSiQue.

### v19: answer adequacy

JevBench's `adequacy` family asks whether a saved answer adequately answers a request,
and nothing in the corpus asked that. Every model here answered "adequate" to 11 of its
12 public tasks, and to about nine responses in ten on a balanced set of HelpSteer2's
human-rated responses, where it scored at chance; a frozen Qwen3.5-4B scored 0.645 there.
v19 ([PREREGISTRATION-v19.md](preregistrations/PREREGISTRATION-v19.md)) added 6,166 rows to v18's recipe,
each a request, a response and whether the response is adequate: 4,866 from HelpSteer2's
train split (adequate = helpfulness and correctness both at least 3 of 4; inadequate =
either at most 1; `data/adequacy.py`), and 1,300 generated
(`data/generators/gen_adequacy_openrouter.py`: each inadequate response carries one assigned
defect — a wrong fact, a missed constraint, a skipped part, a needless refusal, an
accepted false premise — and some adequate ones only look flawed; 1,852 of 2,280 items
kept by writer and verifier agreement, $32 on OpenRouter; 30 of 30 hand-checked
verdicts right). Gold labels only.

| | v18 | v19 |
| --- | --- | --- |
| **HelpSteer2 held-out responses (234, balanced)** | 0.483 | **0.739** (+90 / -30, p < 0.0001) |
| — inadequate / adequate recall | 0.068 / 0.897 | 0.761 / 0.718 |
| **generated requests, categories never trained on (302), balanced accuracy** | 0.577 | **0.788** |
| MuSiQue / ContractNLI / BoardgameQA | **0.892** / **0.866** / 0.801 | 0.879 / 0.862 / **0.810** |
| HotpotQA, held-out source | 0.705 | **0.726** |
| generated documents, v16's set / v18's | **0.863** / **0.761** | 0.846 / 0.757 |
| JevBench | 164 | **167** (+14 / -11); standard 63, hard 56 |
| JevBench Brier / ECE | 0.388 / 0.079 | **0.342** / **0.052** |

**All four pre-registered predictions held** — the first run here to clear its own bar.
On HelpSteer2 it went past the frozen 4B that does the task untrained, and the gain is a
judgement, not a swapped bias: it now catches three in four inadequate responses while
accepting most adequate ones. Nothing else it was measured on moved by more than about
0.02. On JevBench the skill carried beyond its own family: a yes-bias every earlier
model had shrank across all 74 yes/no tasks, `judge_hard` gained three tasks, and the
Brier score fell on 147 of 231 tasks. `long_policy` gave back v18's gain.

### What is left

- **HotpotQA.** v14 still reads an unseen multi-hop source best (0.750 against v19's
  0.726).
  HotpotQA is no longer an untouched test either: the next measure of multi-hop transfer
  needs a fresh held-out source.
- **A seed replicate**, to learn how many tasks a retrain of the same recipe moves on its
  own. Differences of 1-4 tasks, like v13 over v7, v14 over v13, v16 over v14 or v17
  over v16, cannot be read without it. (As of v19. For the retrain noise measured later,
  see [Retraining on AWS](../evaluation/results.md#retraining-on-aws).)
- **`long_policy`.** v18 raised it from 7 to 10 of 19 and v19 gave that back; which
  training rows carry it is not yet known. The task-level record below shows that this
  was not one gain disappearing: v19 retained two of v18's four gains, lost two earlier
  stable tasks, and gained another task v18 missed. It also separates the lineage from
  the two saved `decider-2b` reference runs. Regenerate the tables from the committed
  CSVs with `python research/scripts/long_policy.py`; no checkpoint is loaded.

  | task_id | v16 | v17 | v18 | v19 | decider-2b v10 | decider-2b v11 |
  | --- | --- | --- | --- | --- | --- | --- |
  | hard-opus-a-long_policy-01 | 1 | 1 | 1 | 0 | 1 | 1 |
  | hard-opus-a-long_policy-04 | 0 | 0 | 0 | 0 | 0 | 0 |
  | hard-opus-a-long_policy-08 | 1 | 1 | 1 | 1 | 0 | 1 |
  | hard-opus-a-long_policy-09 | 0 | 0 | 0 | 0 | 0 | 0 |
  | hard-opus-a-long_policy-11 | 0 | 0 | 1 | 0 | 0 | 0 |
  | hard-opus-a-long_policy-13 | 0 | 0 | 0 | 0 | 0 | 0 |
  | hard-opus-a-long_policy-17 | 1 | 1 | 1 | 0 | 0 | 0 |
  | hard-opus-a-long_policy-19 | 1 | 0 | 1 | 1 | 0 | 1 |
  | hard-opus-c-long_policy-02 | 1 | 1 | 1 | 1 | 1 | 0 |
  | hard-opus-c-long_policy-03 | 0 | 0 | 0 | 0 | 0 | 0 |
  | hard-opus-c-long_policy-04 | 1 | 1 | 1 | 1 | 1 | 1 |
  | hard-opus-c-long_policy-05 | 0 | 0 | 0 | 0 | 0 | 0 |
  | hard-opus-c-long_policy-08 | 0 | 1 | 0 | 0 | 0 | 1 |
  | hard-opus-c-long_policy-10 | 0 | 0 | 1 | 1 | 0 | 1 |
  | hard-opus-c-long_policy-11 | 0 | 0 | 0 | 0 | 0 | 1 |
  | hard-sol-b-long_policy-01 | 0 | 0 | 1 | 0 | 1 | 1 |
  | hard-sol-b-long_policy-02 | 0 | 0 | 0 | 0 | 0 | 0 |
  | hard-sol-b-long_policy-05 | 1 | 0 | 0 | 1 | 0 | 0 |
  | hard-sol-b-long_policy-06 | 1 | 1 | 1 | 1 | 1 | 0 |

  v17 to v18 changed five tasks: four gains (`a-11`, `a-19`, `c-10`, `sol-b-01`)
  and one loss (`c-08`). v18 to v19 also changed five: one gain (`sol-b-05`) and four
  losses (`a-01`, `a-11`, `a-17`, `sol-b-01`). The matching totals therefore conceal
  different task movements. A seed replicate is still needed before attributing any
  movement to particular training rows. Against the two saved peer runs, `long_policy`
  is +2 tasks against `decider-2b v10` and -1 against `decider-2b v11`; it is not the
  family that explains the eight-task overall gap to v11.
- **Reading the question.** Every model here, v19 included, gives the same answer to a
  changed question about 94% of the time (`evaluation/question_sensitivity.py`). Labelled
  transforms taught their own keywords (v11a); many real tasks, each asked several ways,
  is the untried route.
- **`score`**, weak since v1 (0.617 in the mix; 0.499 on an unseen rubric type for v19, the
  best yet).
- **Outside the <2B constraint, the torso.** Qwen3 1.7B to 4B was +0.043 at matched
  corpus and steps, and a frozen Qwen3.5-4B reads 0.805; no 4B run of the current
  recipe has been made.

**And the payoff is real.** 1.7B to 4B is worth +0.043 JevBench accuracy at matched
corpus and steps, against roughly +0.004 for the best readout intervention tried here.
For reference, a peer running the same architecture at 2B and 9B measures +0.130 across
that larger jump. On a 24 GiB card 8B is reachable at ~3.8 h per epoch
([Larger torsos](../training/hardware.md#larger-torsos)), which on this
curve is the most promising untried experiment in the project.

## Version history

Superseded versions, kept because several record results worth not repeating; v8
onward are under [Experiments since v7](#experiments-since-v7). The figures below are each version's own and
are not always comparable across sections, since the corpus, window, readout and
calibration all changed.

### What v7 changed

The readout no longer reads one position. `PointerHead` scores option *k* as a single
attention score between the `<answer>` state and the hidden state at the **last token
of option k's line** - the token that has just read that option under causal
attention:

```python
logit_k = <q(h_answer), k(h_option_k)> / sqrt(dim)
```

Three things follow. An option's logit depends on what it *says*, not where it sits.
There is no per-slot parameter, so a positional preference cannot be expressed. And
nothing caps the option count at `num_slots` any more - 255 is now the only limit, as
the API always allowed.

**No tokeniser changes.** `render_question` records the character span of each option
line, and the collator maps spans to token indices with the fast tokeniser's
`offset_mapping`. The reference implementation (kev) uses dedicated `<opt>`/`</opt>`
tokens and has to train their embeddings; reading the last token of an existing line
gets the same position for free.

**Why it was worth building.** Before touching the pipeline, a frozen-feature probe
fitted competing readouts on identical cached hidden states from v6 - same rows, same
optimiser, same LayerNorm, so only the readout differed:

| head | params | held-out accuracy |
| --- | --- | --- |
| slot, linear (v6's shape) | 53k | 0.6619 |
| slot, MLP h=512 | 1.07M | 0.6863 |
| slot, MLP h=1024 | 2.13M | 0.6937 |
| slot + mean(option states) | 2.12M | 0.7056 |
| **pointer, dim=16** | **70k** | **0.7200** |
| pointer, dim=256 | 1.05M | 0.7250 |

A pointer head at 70k parameters beats a slot head at 2.1M. Capacity is not the
explanation, and neither is information access - a slot head given the pooled option
states closes only part of the gap. Roughly half the gain is having option
representations at all, half is addressing them per option rather than pooling.

Two earlier versions of that probe were wrong in ways worth recording. The first
compared a freshly fitted pointer head against v6's *shipped* slot head, which had
never seen the holdout; that inflated the gap from +0.063 to +0.109 and inflated
`score` from +0.020 to +0.173, because the shipped head's score weakness is a transfer
problem, not a readout problem. The second selected the best epoch on dev and then
reported dev.

**What v7 gave up.** v6's `lm_head` seeding has no analogue here - there are no
per-slot rows to seed - so v7 runs `head_init: random` and carries over only the KL
anchor. It improves on v6 anyway.

**Order dependence is reduced, not eliminated.** Reversing the option list moves the
distribution by ~0.016 on an untrained head, against a slot head's much larger
sensitivity, but it is not zero: options still attend to *earlier* options under
causal attention. Making it exact needs kev's `option_isolation`, where each option
span sees only the state, the instructions and itself. That is not implemented here.

Both readouts remain available. `head_type` defaults to `"slot"`, so every v1-v6
checkpoint loads unchanged.

### v7 on JevBench

Figures below are **v7**, calibrated, at the default 3072-token window, measured with
the official harness.

| metric | **v7** | v6 | v5 |
| --- | --- | --- | --- |
| accuracy | **0.667** (154/231) | 0.649 (150) | 0.636 (147) |
| macro | **0.660** | 0.636 | 0.623 |
| ECE | **0.068** | 0.090 | 0.092 |
| Brier | **0.442** | 0.462 | 0.462 |
| ordinal MAE | **0.592** | 0.606 | 0.660 |
| latency p50 / p95 | 246 / 583 ms | 234 / 578 ms | 247 / 590 ms |
| paraphrase consistency | **0.833** (30/36) | 0.778 (28/36) | 0.639 (23/36) |

v7 improves on v6 on every axis except latency, which is unchanged within noise. **ECE
0.068 and paraphrase consistency 0.833 are the best this project has recorded**, the
latter finally matching v4 after a regression that took three versions to undo.

By family, the pointer readout helps where option content carries the decision and
hurts where it does not: `tradeoff` 0.333 -> 0.667, `policy` 0.667 -> 0.833,
`adversarial` 0.500 -> 0.667, `long_policy` 0.316 -> 0.421, against `ambiguous`
0.429 -> 0.143 and `temporal_numeric` 0.333 -> 0.133. Those two losses are 7 and 15
items, so they are two and three questions respectively, but they are the clearest
signal of what the change costs.

v6 -> v7 is +16 tasks against -12, McNemar p = 0.572 - not significant on its own.

### What v6 changed

v5 and earlier trained a randomly-initialised `Linear(hidden, 24)` and discarded
everything the LM already knew about answering a multiple-choice question. Two changes
keep that knowledge, both controlled by `TrainConfig`:

**`head_init: "lm_head"`** seeds slot *k* from the output-embedding row for the token
`"k+1"` — the digit the LM would emit after `<answer>` if it were asked for the option
number, since `<options>` is numbered from 1. Qwen3 ties embeddings, so the input table
*is* the output matrix and `AutoModel` already holds it; no extra weights are loaded.
Rows are rescaled to the variance the random init would have had, so what is under test
is the *direction* the LM head carries, not a larger initial step. Only digits 1-9 are
single tokens in this vocabulary, so slots 0-8 are seeded and the rest keep the default
init — that covers 78.6% of the training corpus and almost every real request.

**`kl_frozen_weight: 0.3`** adds `KL(frozen || student)` to the loss. The reference is
the same torso with its LoRA adapter disabled, read through those same option-number
rows, so it costs one extra forward per micro-batch and no extra weights. Rows needing
a slot above 8 are excluded from the term rather than approximated.

Neither touches inference: `slot_head.pt` is the same shape, the masked softmax is
unchanged, and a v5 checkpoint still loads. The measured effect is +0.013 JevBench
accuracy and a model that arrives very nearly calibrated — the fitted temperature falls
from 1.504 to 1.061, and `score`, historically the saturated one, from 2.401 to 1.258.

### The v6 arms: three retrains on the readout

The three candidate changes behind v6, each trained alone on the v5 corpus at 3072,
one epoch, identical otherwise. **Two were kept and combined into v6; the instruct
torso was dropped.**

| | accuracy | macro | ECE | internal val |
| --- | --- | --- | --- | --- |
| v5 baseline | 0.6364 | 0.6231 | 0.0924 (calibrated) | 0.8425 |
| instruct torso | 0.6190 | 0.5884 | 0.1216 | 0.8375 |
| `head_init: lm_head` | **0.6407** | **0.6232** | 0.0864 | 0.8375 |
| `kl_frozen_weight: 0.3` | **0.6407** | 0.6128 | **0.0851** | **0.8525** |

**No arm moved accuracy** (+1 item, p = 1.000 on both winners) — which, per
[the resolution limit](#the-resolution-limit), was never detectable either way.

**Calibration did move, and it is the clean result.** Both readout-anchored arms reach
ECE 0.085-0.086 with **no temperature fitted at all**, beating v5's 0.0924 *after*
fitting. Two independent mechanisms landing on the same number is what makes it
credible. Seeding the head from the LM's option-number rows, or constraining training
toward that readout, both produce a better-calibrated model than post-hoc scaling does.

**The KL arm behaved as its teacher predicted.** The frozen teacher was measured before
the run: it beats the trained model on `adequacy` (0.667 vs 0.583) and `long_policy`
(0.474 vs 0.368), and those are exactly the families the arm improved (0.750 and 0.421).
First prediction made before a run in this project that came true.

**The instruct torso is a clear negative** and kills its own hypothesis: `policy` fell to
0.500, precisely where the frozen base sits. Fine-tuning reaches the same ceiling from
either set of starting weights. Its one durable gain was `trap`, 0.625 -> 0.875.

Implementation: `head_init` and `kl_frozen_weight` on `TrainConfig`, described under
[What v6 changed](#what-v6-changed).

**Combining the two winners beat both.** v6 scores 0.6494 against 0.6407 for each arm
alone and 0.6364 for v5, with the best macro of any 1.7B variant (0.6355). Their
per-family strengths were complementary — `lm_head` took `ambiguous`, `ordinal` and
`intent`, KL took `adequacy` and `long_policy` — and they agreed on only 92% of tasks,
each fixing nine the other missed.

One thing did **not** compound. Each arm alone reached ECE 0.086/0.085 *uncalibrated*;
combined and uncalibrated it is 0.0950, worse than either, reaching 0.0896 only after
fitting. The accuracy effects add and the calibration effects partly cancel. No
explanation for that yet, and it is worth knowing before assuming more anchoring is
better.

### v5: compositional training teaches the task, not the capability

JevBench showed us scoring 1.000 on `fact`/`routing`/`tool_selection` and 0.167 on
`tradeoff`, 0.278 `multi_hop`, 0.316 `long_policy`. Every training task answered in
one step, so the corpus had never asked the model to compose anything.

v5 adds [RuleTaker](https://huggingface.co/datasets/tasksource/ruletaker) — apply a
stated ruleset to a case — chosen because it is the only source found that ships
*graded* compositional depth. Train on depths 0/1/2, hold out 3/5 (does shallow
composition generalise deeper?) and NatLang (same logic, natural phrasing rather than
templated).

**On RuleTaker itself the result looks excellent.**

| depth | v4 | v5 | | |
| --- | --- | --- | --- | --- |
| depth-0 | 0.732 | **0.971** | +0.239 | trained |
| depth-1 | 0.646 | **0.902** | +0.257 | trained |
| depth-2 | 0.580 | **0.860** | +0.280 | trained |
| depth-3 | 0.610 | **0.783** | **+0.173** | held out |
| depth-5 | 0.584 | **0.637** | +0.053 | held out |
| NatLang | 0.572 | **0.767** | **+0.195** | held out |

Depth-3 gains 0.173 having never been trained on, so this is not depth-specific
memorisation — the model extrapolates about one hop past its training range before
falling away by depth-5. NatLang gaining 0.195 says it learned the reasoning rather
than the template, since that split changes surface form entirely.

**On JevBench it changed nothing.**

| | v4 | v5 |
| --- | --- | --- |
| accuracy | 0.6277 | **0.6277** |
| `long_policy` | 0.316 | **0.316** |
| `temporal_numeric` | 0.267 | **0.267** |
| ECE | 0.068 | **0.135** |

Identical accuracy to four decimals, and no movement at all on the family RuleTaker
most directly targets. Per-family deltas elsewhere are one or two items — the
families are tiny (`routing_hard` n=5, `tradeoff` n=6, `adversarial` n=6), so an
apparent +0.200 is a single question, and they cancel to zero.

**So the RuleTaker gains were in-family transfer.** Held out by depth, but sharing
generator, vocabulary and structure with the training splits. The model learned to do
RuleTaker; it did not learn to reason.

The run does separate two claims that were previously entangled. *"The corpus lacks
compositional content"* is true — the model could not do RuleTaker and now can.
*"Adding compositional content yields transferable reasoning"* is false, at least
this way.

**Calibration regressed, for a traceable reason.** ECE nearly doubled because the
fitted `noul` temperature moved 1.192 → 0.863. RuleTaker is 12,000 of the 33,000
holdout, so calibration became dominated by a distribution unlike JevBench, and a
temperature below 1.0 *sharpens* confidence — right for RuleTaker where v5 is
genuinely accurate, wrong everywhere else. Adding a large, easy, unrepresentative
task to the calibration set damages calibration on everything else.

### v4: instruction diversity and noul breadth changed nothing

v4 added two things the evidence pointed at. Twelve of fourteen tasks had carried a
single instruction string, so thirteen tasks gained 5-6 phrasings each, sampled per
epoch by the collator exactly as option order is. And `noul` went from four judgement
types to eight — `vitaminc` (fact verification against evidence), `paws` (adversarial
semantic equivalence), `wnli` (coreference) and `pubmed_qa` (biomedical QA) — nearly
doubling noul from 18,620 examples to 35,449.

Neither moved held-out accuracy. On the seven-task probe set, none of which appears in
any holdout:

| | v3 | v4 | 95% noise |
| --- | --- | --- | --- |
| noul | 0.794 | 0.778 | ±0.017 |
| choice | 0.652 | 0.656 | ±0.022 |
| **overall** | **0.733** | **0.725** | |

One task moved beyond noise — `subjectivity` 0.868 → 0.825, downward. With seven
comparisons, one at p<0.05 is what chance produces.

The v3 holdout *does* show v4 ahead, 0.620 → 0.650, and that gap clears sampling
noise. But it decomposes entirely into one task:

| primitive | v3 | v4 |
| --- | --- | --- |
| score (`hate_severity`) | 0.384 | **0.486** |
| noul (`sarcasm`) | 0.606 | 0.630 |
| choice | 0.728 | 0.724 |

`hate_severity` ranges 0.355–0.504 on rewording alone
([How the question is worded moves the answer](#how-the-question-is-worded-moves-the-answer)), and v4 changed its
instruction pool. 0.486 sits inside that band, so the apparent gain is most likely a
phrasing effect rather than a capability one.

Two things did improve and are not in dispute: in-distribution ECE fell from 0.099 to
0.068, and the fitted score temperature continued to drop (6.00 saturated in v2, 4.34
in v3, **2.16** in v4), meaning the score head is progressively less overconfident.

**Five interventions failed to move held-out accuracy**: torso capacity, head capacity,
data volume, noul breadth, instruction diversity. The one apparent success — score
rubric coverage in v3 — is confounded by phrasing.

Read that list with two later findings in mind. **Torso capacity does not belong on
it** — it was scored on this same held-out corpus, which later turned out to rank a
better model below a worse one. And every entry was judged by a measurement that cannot
resolve an effect smaller than about +0.04 accuracy, so "failed to move" means "produced
nothing this instrument could see", not "produced nothing". Both are covered under
[Scaling, and the measurement that hid it](#scaling-and-the-measurement-that-hid-it).

### v3: the score failure was transfer, not difficulty

`formality` reaches **0.613 in-distribution** against a 0.25 baseline — squarely in
the normal range for score tasks here (`sst5` 0.631, `app_reviews` 0.502). So v2's held-out 0.271 was a **transfer failure, not an unlearnable task**.
v2 could not establish this, because `formality` only ever sat in its holdout.

With a register rubric in training, score transfer and calibration both improve:

| | v2 (score held out: `formality`) | v3 (score held out: `hate_severity`) |
| --- | --- | --- |
| held-out score accuracy | 0.271 (+0.021 over chance) | **0.379 (+0.129)** |
| held-out score ECE | 0.117 | **0.004** |
| confidence gap | +0.117 overconfident | **+0.003** |
| fitted score temperature | 6.00 (saturated at bound) | **4.34 (converged)** |

The calibration result is the more trustworthy half. ECE 0.004 with a +0.003 gap
means score confidence became genuinely meaningful, and the temperature no longer
pins to the search bound — the head is no longer overconfident beyond what scaling
can correct.

The accuracy gain is **confounded**: the held-out task differs between runs.
`hate_severity` is affect-adjacent and training still contains three valence rubrics,
so it is plausibly an easier transfer target than `formality` was. A clean test would
score the same held-out task under both training mixes, which is impossible when the
variable *is* which task sits in training.

**`noul` is now the weak primitive**: held-out ECE 0.208, under-confident by 0.120,
with `sarcasm` at 0.614 against a 0.500 baseline. Per-primitive temperature cannot
fix within-bin miscalibration — mean confidence tracks accuracy while individual
predictions do not.

### Early runs (v1-v3)

| | v1 | v2 | v3 |
| --- | --- | --- | --- |
| torso | Qwen3-1.7B | Qwen3-4B | Qwen3-1.7B |
| train / holdout | 40,620 / 12,000 | 72,620 / 20,000 | 71,620 / 21,000 |
| tasks | 7 train, 2 held out | 14 train, 4 held out | 14 train, 4 held out |
| held-out primitives | choice only | choice, score, noul | choice, score, noul |
| score rubrics in training | 2, both sentiment | 4, all valence | **3 valence + 1 register** |
| wall clock | 66 min | 3.8 h | 2.2 h |

v3 tests one hypothesis: that v2's score failure came from every training rubric
being a valence judgement. It swaps `formality` (register) into training and holds
out `hate_severity` instead. Run at 1.7B because v2 appeared to establish that capacity
is not the lever here — a conclusion later overturned, though it happens not to affect
this experiment, which varies the corpus rather than the torso.

### Held-out tasks — unseen label sets

Whole tasks are excluded from training, so their label sets and domains are new at
eval time. This is the honest test of whether the readout learned to bind to
prompt-described options rather than memorise categories; in-distribution accuracy can
be reached by memorising label sets.

v7's figures are in the table under [Summary](../evaluation/results.md#summary). The finding below is older - measured
across v1-v3 on an eleven-task probe set - but it is about how transfer behaves rather
than about any one checkpoint, and nothing since has contradicted it.

**Transfer is governed by task difficulty, not distance from the training mix.**
Eleven held-out tasks, deliberately chosen to span near and far from the corpus:

| held-out task | primitive | relation to training | accuracy |
| --- | --- | --- | --- |
| `massive_intent` | choice | adjacent | 0.887 |
| `subjectivity` | noul | **novel** | **0.868** |
| `rte` | noul | adjacent | 0.830 |
| `qnli` | noul | adjacent | 0.794 |
| `trec_qc` | choice | **novel** | 0.714 |
| `cola` | noul | **novel** | 0.699 |
| `arxiv_class` | choice | **novel** | 0.641 |
| `sarcasm` | noul | novel | 0.614 |
| `newsgroups` | choice | **adjacent** | 0.596 |
| `emotion` | choice | novel | 0.574 |
| `hate_severity` | score | novel | 0.38 – 0.50 |

An earlier version of the README claimed semantic proximity governed transfer. **It does
not.** The strongest new task, `subjectivity` at 0.868, is a judgement nothing in
training makes. The weakest, `newsgroups` at 0.596, is the closest analogue to three
training tasks. What separates them looks like intrinsic difficulty instead.

**Held-out `noul` is 0.794, not 0.614.** The lower figure came from a run where
`sarcasm` was the only noul holdout. Across four further tasks noul transfers well;
`sarcasm` is the worst of five and intrinsically hard — pragmatic inference over world
knowledge — rather than a transfer failure. Any conclusion resting on a single
held-out task should be treated as provisional, which is why there are now eleven.

The score failure is partly self-inflicted and v2's test was unfairly hard: training
contained `yelp`, `sst5`, `app_reviews` and `hate_severity` — all valence or
affect-adjacent rubrics — so the model learned "score = rate the valence".
`formality` is a *register* judgement, orthogonal to valence, and it was the only
genuinely non-valence rubric built, which then went into the holdout rather than
training.

### How the question is worded moves the answer

Twelve of fourteen training tasks carry exactly one instruction string, so the model
has never seen a task asked two ways. Re-running the holdout with five phrasings of
each question — everything else byte-identical — shows how much that costs:

| held-out task | spread across 5 phrasings | range |
| --- | --- | --- |
| `massive_intent` | 0.006 | 0.879 – 0.885 |
| `emotion` | 0.026 | 0.578 – 0.604 |
| `sarcasm` | 0.072 | 0.567 – 0.639 |
| `hate_severity` | **0.149** | 0.355 – **0.504** |

Overall spread is 0.042, twice the ±0.021 sampling noise at n=4,000. But the aggregate
hides the pattern: **phrasing sensitivity is inversely proportional to task
competence.** Where the model is strong, wording is irrelevant — `massive_intent`
moves by 0.006. Where it is struggling, wording dominates. With a strong signal
phrasing is noise; near chance, phrasing decides which weak cue gets picked up.

Two consequences. If you deploy this, **try several phrasings of your question** —
it is free and worth up to 0.15 on a hard task. And every held-out number here for a
*weak* task carries a phrasing band of roughly ±0.07 that sampling error does not
capture.

This also qualifies the score result in
[v3](#v3-the-score-failure-was-transfer-not-difficulty): `hate_severity` ranges 0.355–0.504 on
rewording alone, a band wider than the +0.129 improvement attributed to corpus
composition. That comparison is confounded by phrasing and is reported as suggestive
only. The calibration half of the result (ECE 0.117 → 0.004) is unaffected.

### Confidence is well calibrated, and that is what survives

| band | n | accuracy | mean confidence | gap |
| --- | --- | --- | --- | --- |
| < 0.5 | 2,754 | 0.353 | 0.343 | −0.011 |
| 0.5 – 0.9 | 1,674 | 0.695 | 0.707 | +0.012 |
| >= 0.9 | 1,572 | **0.957** | 0.972 | +0.015 |

Every band agrees with its own claim to within 0.02. 26% of held-out traffic can be
automated at 0.957 accuracy, and the model reliably flags the rest — so, on this v2-v3
held-out set, the confidence bands hold even where accuracy is poor, which matters more than the mean for
something meant to gate decisions.

### Calibration in v2-v3

Temperature is fitted per primitive against **ECE**, on the `calib` half of the
holdout, and scored on the disjoint `test` half.

| fit | overall ECE | NLL |
| --- | --- | --- |
| none (T=1) | 0.148 | 0.951 |
| one global scalar, NLL objective | 0.106 | 0.867 |
| **per-primitive, ECE objective** | **0.037** | 0.872 |

Both changes were needed, and the objective mattered more than the granularity.
Fitting `noul` to NLL chose T=2.63, which left it *under*-confident by 0.29 — worse
calibrated than no scaling at all, while the likelihood genuinely improved. NLL and
ECE are not interchangeable; thresholds depend on ECE.

Fitted values differ sharply, which is why one scalar could not serve all three:
`choice` 1.40, `noul` 1.13, `score` 6.00.

### Known problems in v2-v3

- **`score` temperature saturates.** The 6.00 above is the search bound, not a
  converged fit — the model is overconfident on score beyond what any temperature
  corrects.
- **Calibrating `score` costs accuracy.** Temperature cannot change a `choice`
  argmax, so calibration is free there. But a score is `sum(i * p_i)` over the
  distribution, so softening pulls the answer toward the midpoint: held-out score MAE
  degraded from 0.847 to 0.981. That trade is deliberate but real.
- **`noul` ECE stays at 0.190** despite a mean gap of only −0.049. Mean confidence
  tracks accuracy, but *within-bin* calibration is poor, and one scalar cannot fix
  that.
- **One calibration set cannot serve two distributions.** Fitted on held-out data
  (0.607 accuracy) and applied in-distribution (0.858), in-distribution ECE is 0.113.
  Calibrate on data resembling your production traffic.

### In-distribution, by primitive (v2)

| primitive | accuracy | ECE | note |
| --- | --- | --- | --- |
| noul | 0.963 | 0.051 | |
| choice | 0.949 | 0.062 | `lang_id` 1.000, `clinc150` 0.992, `banking77` 0.964 |
| score | 0.635 | 0.241 | MAE 0.901 levels |

`lang_id` reaching **1.000** is the architecture's clearest validation. Its 20-language
label set has no semantic relationship to the text content, so there is no shortcut
but to read the option list and bind slot *k*.

`score` is weak in-distribution too (v3: 0.615), not only on transfer — so the rubric
mix is not the whole story. Even trained directly on a rubric, 5-level ordinal
judgement sits far below what the model achieves on choice and noul.

### Slot-head era design findings (v1-v6)

**A linear readout, not an MLP — measured, not assumed.** Head *k* is one row of a
single `Linear(hidden, 24)`: 2,049 parameters, with the whole head 53,272. An
ablation trained two fresh heads on the same frozen torso and adapter, so head
capacity was the only variable — a 53K linear readout against a 1.07M
`Linear(2048→512) → GELU → Linear(512→24)`.

**20x the parameters bought nothing.** No accuracy difference exceeded sampling
noise, on any primitive, in either direction:

| | linear (53K) | MLP-512 (1.07M) | 95% noise |
| --- | --- | --- | --- |
| held-out overall | 0.6200 | 0.6225 | ±0.017 |
| held-out choice | 0.7279 | 0.7363 | ±0.021 |
| in-distribution overall | 0.8483 | 0.8483 | ±0.013 |

Calibration shows a faint overfitting signature — the MLP is slightly better
in-distribution (ECE 0.0995 → 0.0877) and slightly worse held-out (0.0522 → 0.0580) —
but the magnitudes are too small to lean on.

The original argument for keeping the head linear was that a head with capacity could
learn slot-specific structure and defeat the genericity that option-shuffling
protects. **That reasoning was wrong**: held-out choice accuracy was marginally
*higher* with the MLP, so genericity was not damaged. The real explanation is simpler
— the head is not the bottleneck, the torso's representation is. A linear probe
already extracts what is available at the pooled position. (A related prediction also
failed: the MLP came out *less* overconfident, not more, needing temperatures
1.74/2.40/5.69 against the linear head's 1.94/2.53/6.00.)

Set `head_hidden: 512` to reproduce; the default stays 0.

**Where the score failure actually lives.** A second frozen-torso probe splits it.
The 4B was LoRA-trained on a corpus with no `formality` in it and scored 0.271 on it
held out. Freezing that same torso and training only a fresh head, now with
`formality` in the head's data:

| 4B, same frozen torso and adapter | formality (chance 0.250) |
| --- | --- |
| head never saw formality | 0.271 |
| head trained on formality | **0.481** |
| *reference:* 1.7B whose LoRA did see formality | 0.625 |

Head-only training recovers +0.21, so much of the failure was the readout rather than
the features — the 4B representation already encodes a good deal of what a register
rubric needs despite never having been adapted for one. Rubric-specific LoRA
adaptation then adds a further +0.14, and it does so against a *smaller* model, so
that remainder is adaptation rather than capacity.

The practical reading: adding a rubric type to the corpus helps twice over, once by
teaching the head to bind it and again by adapting the features to it.

**Torso capacity IS the lever. This section originally concluded the opposite and was
wrong.** The error was not in these numbers but in the yardstick: every figure below is
our own held-out corpus, and that corpus cannot rank models — see
[Scaling, and the measurement that hid it](#scaling-and-the-measurement-that-hid-it). Measured externally on JevBench at matched corpus and steps,
the same two checkpoints differ by +0.043, the largest effect recorded in this project.

Head capacity (the MLP) and adapter capacity remain untested externally; only the
*torso* claim is overturned. The internal figures, kept for the record:

| | 1.7B | 4B |
| --- | --- | --- |
| in-distribution overall | 0.8483 | 0.8512 |
| in-distribution choice | 0.9442 | 0.9438 |
| held-out choice | 0.7279 | 0.7330 |
| held-out noul | 0.6060 | 0.6155 |

(The held-out *score* figures from that probe are excluded: the 4B's adapter trained
on `hate_severity`, which the v3 holdout scores as held out, so that number is
contaminated. `emotion`, `massive_intent` and `sarcasm` were held out in both
corpora and are valid.)

**And the model can already fit the task perfectly.** Training 1,494 stratified
examples for 24 epochs drives training loss to 0.001 and training accuracy to:

| | training accuracy |
| --- | --- |
| overall | **0.9933** |
| choice | **1.0000** |
| noul | 0.9955 |
| score | 0.9789 |

Option shuffling and instruction variation stayed **on** for this, so the model is not
taking a slot-memorisation shortcut — it fits the real task. Held-out loss on the same
run is 1.809 against a training loss of 0.001, the expected overfitting signature.

That closes the capacity question. A rank-16 adapter on a 1.7B torso has enough
bandwidth to fit these discriminations exactly; it simply does not learn ones that
generalise. So the accuracy plateau in full runs is label noise plus architecture, not
parameters — and a LoRA-rank sweep, the last untested capacity axis, would buy more of
something already in surplus.

The per-primitive split supports the noise reading: `score` is hardest to memorise
(0.979, with `app_reviews` lowest at 0.940), which is exactly the primitive with the
most annotator disagreement. Even memorisation struggles where the labels themselves
are ambiguous.

**`K = 24` slots, matched to the corpus.** `num_slots` must equal the largest option
count the training corpus actually contains: slots beyond that never receive a
gradient and would ship at their random initialisation, so a request with an
unusually wide option list would hit untrained heads. The default build uses
`--max-options 24` and draws each large label set down to a *random* count in
[3, 24], which both covers every slot and reinforces that N is a property of the
prompt. Questions wider than `num_slots` are rejected with a clear error rather than
silently truncated; raise both numbers together and retrain if you need more.
