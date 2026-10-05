# strands-decider-2.5B-minicpm-v21

> **Unofficial candidate.** A text decider on MiniCPM5-2B: the hobson-v20 recipe trained from
> a raw MiniCPM5-2B base. Estimated **top 3 in the 2B text class** on the JevBench board,
> and the most public tasks right of any candidate (not submitted; estimates from local
> measures). Not a published Strands release.

| | |
| --- | --- |
| What it is | The [hobson-v20](strands-decider-2B-hobson-v20.md) recipe (v19's training data, the 27B yes/no teacher, no frozen-KL anchor on yes/no rows, a pointer head, LoRA r 16, one full epoch) from the raw MiniCPM5-2B base instead of v19; MiniCPM5 won a base bake-off ([below](#the-base-bake-off)) |
| Base | `openbmb/MiniCPM5-2B` at revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, pinned in the checkpoint (a Llama decoder: 42 layers, d 2048, grouped-query attention, an untied output head) |
| Size | 2.5B dense, by the board's own description of MiniCPM5-2B (decision-2b, on the same base, is listed the same way: just over the 2B line), plus a LoRA adapter and a pointer head; the archive holds the adapter and head (115 MB) |
| Answers | text: yes/no, choice and score questions |
| Configs | [configs/experiments/strands-decider-2.5B-minicpm-v21.yaml](../../configs/experiments/strands-decider-2.5B-minicpm-v21.yaml), `-seed1`, `-seed2`: **the package is seed 0**, calibrated |
| Download | https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/strands-decider-2.5B-minicpm-v21.tar |
| sha256 | `f6d6d9346dbc3a062641384753a3ade4d68ea4bcfe8cc8eaedd87d59e873f8d2` ([weights_SHA256SUMS.txt](https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/weights_SHA256SUMS.txt)) |

## Results

Local measures, never board scores ([docs/evaluating.md](../evaluating.md)). The candidates
are measured as packaged (with the choice/score refit, "Fix A"), the published v19 as
published; v19 with the same refit scores 34.2 on the public tasks and 23.6 on the unseen
families ([hobson-v20](strands-decider-2B-hobson-v20.md#results)).

| Measure | minicpm-v21 (seed 0) | hobson-v20 | v19 (published) |
| --- | --- | --- | --- |
| JevBench v1 public, v1.5-style Intelligence | **47.1** (3 seeds: 47.1 / 46.2 / 44.2) | 42.5 | 29.5 |
| Tasks right, of 231 | **178** (3 seeds: 178 / 178 / 182) | 177 | 167 |
| Yes/no answers inside 0.2-0.8, of 74 | **21** | 35 | 48 |
| ECE, public tasks (before the choice/score refit) | 0.087 | 0.051 | 0.050 |
| Unseen families, Intelligence | 28.7 | **34.7** | 20.3 |
| Unseen, yes/no / choice / score competence | -14.8 / 57.0 / 44.0 | -10.8 / 64.7 / 50.1 | -40.8 / 60.4 / 41.4 |
| Unseen ECE | 0.101 | 0.055 | 0.073 |
| Old board measure (v1.4.2, public accuracy) | **0.771** | 0.766 | 0.723 |

- Against v19 on the public tasks (the three seeds, `v15_proxy.py --vs`, paired 95%
  intervals): +9.44 (+2.72, +16.65) Intelligence and +12.3 (+3.0, +22.3) tasks right.
- On the unseen set it scores above the two official 2B leaders run locally (decider-2b
  21.0, decision-2b 20.2), but drops more from public to unseen (-18.4) than hobson-v20
  (-7.8): much of MiniCPM5's public lead is familiarity with the public task formats.
- **Estimated board position (unofficial):** Capability 56.9 / 60.8 / 65.2 (pessimistic /
  middle / optimistic), top 3 in the 2B class; on the old v1.4.2 public-accuracy measure,
  0.771 would be #1 in the class. Method: [hobson-v20](strands-decider-2B-hobson-v20.md#results).
- The choice/score temperature refit ("Fix A",
  [docs/evaluating.md](../evaluating.md#calibration-refitting-only-some-temperatures))
  matters most here: the first calibration step gave every MiniCPM5 checkpoint a score
  temperature of 3.7-4.6; the refit adds 5 to 7 points of Intelligence on every MiniCPM5
  checkpoint and cuts its public ECE by a third to a half.

Every number per seed is in the results log:
https://github.com/Vivek0712/strands-decider/blob/ed29b08bb71aad26133bb6c45e8027648f5725ef/RESULTS.md.

## Download and serve

```bash
NAME=strands-decider-2.5B-minicpm-v21
URL=https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights
curl -fO "$URL/$NAME.tar" && curl -fO "$URL/weights_SHA256SUMS.txt"
grep " $NAME.tar\$" weights_SHA256SUMS.txt | sha256sum -c -
mkdir -p checkpoints/published && tar -xf "$NAME.tar" -C checkpoints/published   # see the note below
strands-decider serve checkpoints/published/$NAME --port 8000
```

MiniCPM5-2B downloads at the pinned revision. The archive unpacks to a directory named
like the soup the retraining commands write (`checkpoints/strands-decider-2.5B-minicpm-v21`),
which is a different checkpoint, so it is unpacked under `checkpoints/published/` here.

## Verify it yourself

On one NVIDIA GPU, with the archive unpacked as above and v19 at its pinned revision
([docs/evaluating.md](../evaluating.md#setup)):

```bash
CKPT=checkpoints/published/$NAME
# JevBench v1 public (231 tasks), scored with the local v1.5 proxy, against v19
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh $CKPT reports/jevbench/$NAME
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/$NAME --vs reports/jevbench/v19

# Unseen task families (1,050 rows), against v19
training/recipe.sh build
python evaluation/unseen/build.py --out data/unseen.jsonl
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint $CKPT --out reports/unseen/$NAME
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/v19 --out reports/unseen/v19
python evaluation/unseen/score.py reports/unseen/$NAME --vs reports/unseen/v19
```

## Retrain

Linux with NVIDIA GPUs; the recorded runs used one H200 per seed (about 2 h 30 min each).
The data and the 27B teacher file are hobson-v20's
([Retrain](strands-decider-2B-hobson-v20.md#retrain), up to `teacher_yn`); no v19
checkpoint is needed.

```bash
for s in "" -seed1 -seed2; do
  strands-decider train --config configs/experiments/strands-decider-2.5B-minicpm-v21$s.yaml
done
calibrate() {  # the two calibration steps, as for hobson-v20
  strands-decider calibrate "$1" --data data/holdout_v5_norule.jsonl
  strands-decider calibrate "$1" --kinds choice,score --objective nll --limit 900 \
    --data data/multistep_v14_eval.jsonl --data data/holdout_v5_norule.jsonl
}
calibrate checkpoints/strands-decider-2.5B-minicpm-v21-s0          # the package
# The seeds' weight soup (44.5 on the public tasks), which the -vl model starts from:
strands-decider soup checkpoints/strands-decider-2.5B-minicpm-v21-s{0,1,2} \
  --out checkpoints/strands-decider-2.5B-minicpm-v21
calibrate checkpoints/strands-decider-2.5B-minicpm-v21
```

What a Llama-family base needs, all opt-in (a Qwen3.5 base loads byte for byte as before):

| Piece | What it does |
| --- | --- |
| `base_revision` (training config) | Pins the base to a Hub commit; the checkpoint records it |
| An untied output head | MiniCPM5's `lm_head` is not tied to its input embeddings, and the torso is loaded without it; the frozen-KL readout reads the option-number rows of `lm_head.weight` once from the base's safetensors (`StrandsDeciderModel.slot_rows`) |
| One-digit option numbers only | MiniCPM5's vocabulary has single tokens for 10-24 too; the frozen readout still reads 1-9 only, so the frozen-KL term covers the same rows whatever the torso |

## The base bake-off

v21's base was chosen by training one recipe from each raw base on the same data, budget
(2,400 steps) and seed: [configs/experiments/bakeoff/](../../configs/experiments/bakeoff/),
identical but for the base, its revision and the LoRA target names (which
`tests/test_configs.py` holds). The arms were Qwen3.5-2B-Base at two seeds, MiniCPM5-2B, and
Gemma 4 E2B; the Gemma 4 E2B arm came last and was rejected, and its support is not part of
this repository.

The rule, fixed before any score: the primary measure is the v1.5 proxy Intelligence on
JevBench v1 public; a non-Qwen base wins only if it is above the higher of the two Qwen
seeds (their spread is the noise floor); between two that clear it, the higher proxy wins,
and within 1.0 point the lower Brier, then the faster p50 latency.

| Arm | Proxy Intelligence (as scored at the time) | Tasks right | Speed on one H200 |
| --- | --- | --- | --- |
| **MiniCPM5-2B** | **48.5** | **179** | 0.48 step/s |
| Qwen3.5-2B-Base, seed 0 / seed 1 | 30.0 / 40.8 | 164 / 178 | 0.56 step/s |
| Gemma 4 E2B (rejected) | 14.5 | 150 | 0.40 step/s |

Paired against the Qwen mean, as scored at the time: MiniCPM5 +13.0 (+6.3, +20.0); Gemma 4
E2B -20.9 (-32.0, -9.6). The proxy of the time graded score tasks by the top level;
re-scored by the expected level (today's `v15_proxy.py`), MiniCPM5 still clears both Qwen
seeds (37.3 against 24.1 / 35.3). Gemma 4 E2B knows the answers but hedges: in a later sweep
its best arm (the instruction-tuned E2B) got 177 tasks right yet answered 57 of 74 yes/no
tasks inside the band. The Gemma results are in the results log.

```bash
for b in qwen35-2b qwen35-2b-seed1 minicpm5-2b; do
  strands-decider train --config configs/experiments/bakeoff/$b.yaml
  strands-decider calibrate checkpoints/bakeoff-$b --data data/holdout_v5_norule.jsonl
  PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/bakeoff-$b reports/jevbench/bakeoff-$b
  python evaluation/jevbench/v15_proxy.py reports/jevbench/bakeoff-$b
done
```

The bake-off arms were calibrated with the first step only.

## Limits

- Exploratory: local runs, three seeds; v21 is the bake-off winner trained longer, chosen
  after the bake-off's results.
- 2.5B parameters: just over the 2B class line, as decision-2b is.
- The v1.5 proxy is our reimplementation of the published rules; the sealed half and
  Calibration are estimated, so the board position is an estimate.
- Its public-to-unseen drop is the largest of the text candidates; read its public lead
  with the unseen numbers beside it.
