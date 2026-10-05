# strands-decider-2B-hobson-v20

> **Unofficial candidate.** A text decider on Qwen3.5-2B-Base: v19 continued with a
> Qwen3.5-27B yes/no teacher. Estimated **#1 strictly-2B text model** on the JevBench board
> (not submitted; an estimate from local measures). Not a published Strands release.

| | |
| --- | --- |
| What it is | The published v19 continued for one full epoch on its own training data, with the 27B teacher's distributions on the yes/no rows and no frozen-KL anchor on them; three seeds averaged into one checkpoint (a weight soup), then calibrated |
| Base | `Qwen/Qwen3.5-2B-Base`, trained at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`; v19 at `bb282d786bc251fd4e3068de3ada9ddbb38127cd` |
| Size | 2B class: v19's 1.9B-parameter torso, a LoRA adapter of rank 48 (the soup concatenates three rank-16 adapters) and a pointer head; the archive holds the adapter and head (226 MB), the base downloads from Hugging Face |
| Answers | text: yes/no, choice and score questions, as v19 |
| Configs | [configs/experiments/strands-decider-2B-hobson-v20.yaml](../../configs/experiments/strands-decider-2B-hobson-v20.yaml), `-seed1`, `-seed2` |
| Download | https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/strands-decider-2B-hobson-v20.tar |
| sha256 | `6346192d738c4f096ef7abcfdda21a04e871eaad5f298bebd0d1a65efa9f11d0` ([weights_SHA256SUMS.txt](https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/weights_SHA256SUMS.txt)) |

The name is the candidate's, not a run of [research/](../../research/README.md): it is
unrelated to the `v20` catch-all run there (`configs/experiments/v20.yaml`).

## Results

Local measures, never board scores ([docs/evaluating.md](../evaluating.md) has the method).
"v1.5-style Intelligence" is the local proxy of JevBench v1.5's published rules on the 231
public v1 tasks; "unseen" is the 1,050-row set ([docs/evaluating.md](../evaluating.md#unseen-task-families)).
hobson-v20 is measured as packaged (with the choice/score refit, "Fix A"); the published v19
as published. The last column is v19 with the same refit, for a like-for-like reading.

| Measure | hobson-v20 | v19 (published) | v19 + Fix A |
| --- | --- | --- | --- |
| JevBench v1 public, v1.5-style Intelligence | **42.5** | 29.5 | 34.2 |
| Tasks right, of 231 | **177** | 167 | 167 |
| Yes/no answers inside 0.2-0.8, of 74 (count as wrong under v1.5) | **35** | 48 | 48 |
| ECE, public tasks (before the choice/score refit) | 0.051 | 0.050 | -- |
| Unseen families, Intelligence | **34.7** | 20.3 | 23.6 |
| Unseen, yes/no / choice / score competence | -10.8 / 64.7 / 50.1 | -40.8 / 60.4 / 41.4 | -40.8 / 60.4 / 51.1 |
| Unseen ECE | 0.055 | 0.073 | 0.043 |
| Old board measure (v1.4.2, public accuracy) | 0.766 | 0.723 | 0.723 |

On the unseen set it also scores above the two official 2B leaders, run locally through
their own published code and calibration: decider-2b (Mapika; 2B #1 on Intelligence) 21.0,
and decision-2b (FlyMy; 2B #1 on Capability) 20.2. Of the candidates measured there, it loses
the least between the public tasks and the unseen families (-7.8 points).

- The three seeds before averaging, against v19 on the public tasks
  (`v15_proxy.py --vs`, seeds pooled, paired 95% interval): +6.17 (+1.19, +11.47).
- **Estimated board position (unofficial):** Capability 57.2 / 61.1 / 65.5 (pessimistic /
  middle / optimistic), against 58.8 for the current 2B #1, decision-2b. The estimate takes
  the open half from the public proxy, the sealed half from the unseen score plus the
  offset measured on the two leaders (7.8 to 14.9), and Calibration as a band of 72-85,
  since it is not measurable locally.

Every number per seed is in the results log:
https://github.com/Vivek0712/strands-decider/blob/ed29b08bb71aad26133bb6c45e8027648f5725ef/RESULTS.md.

## Download and serve

```bash
NAME=strands-decider-2B-hobson-v20
URL=https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights
curl -fO "$URL/$NAME.tar" && curl -fO "$URL/weights_SHA256SUMS.txt"
grep " $NAME.tar\$" weights_SHA256SUMS.txt | sha256sum -c -
mkdir -p checkpoints && tar -xf "$NAME.tar" -C checkpoints
strands-decider serve checkpoints/$NAME --port 8000
```

It serves as v19 does, on the same API ([docs/inference.md](../inference.md)). The archive
holds the adapter, the head, the fitted temperatures, `soup.json` (the soup's inputs),
`train_config.json`, a model card and a `SHA256SUMS` of its files. Like v19's, its config
records no base revision, so the base loads from its default branch. Check that the branch
is still the revision it was trained on:

```bash
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen3.5-2B-Base').sha)"
# b1485b2fa6dfa1287294f269f5fb618e03d52d7c
```

## Verify it yourself

On one NVIDIA GPU, with the archive unpacked as above and v19 at its pinned revision
([docs/evaluating.md](../evaluating.md#setup)):

```bash
# JevBench v1 public (231 tasks), scored with the local v1.5 proxy, against v19
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/$NAME reports/jevbench/$NAME
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/$NAME --vs reports/jevbench/v19

# Unseen task families (1,050 rows), against v19
training/recipe.sh build
python evaluation/unseen/build.py --out data/unseen.jsonl
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/$NAME --out reports/unseen/$NAME
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/v19 --out reports/unseen/v19
python evaluation/unseen/score.py reports/unseen/$NAME --vs reports/unseen/v19
```

Expect agreement to within a point or two, not to the digit: kernels and library versions
move bf16 probabilities, and answers near 0.2 or 0.8 can cross the band.

## Retrain

Linux with NVIDIA GPUs; the recorded runs used one H200 per seed (about 2 h each), Python
3.11, torch 2.7.1 and transformers 5.18.0.

```bash
pip install -e ".[train,cuda]"
pip install vllm                                       # optional: TEACHER_ENGINE=vllm
hf download StrandsAgents/strands-decider-2B-hobson-v19 \
  --revision bb282d786bc251fd4e3068de3ada9ddbb38127cd --local-dir checkpoints/v19
training/recipe.sh build fetch multistep generated adequacy teacher_yn

for s in "" -seed1 -seed2; do
  strands-decider train --config configs/experiments/strands-decider-2B-hobson-v20$s.yaml
done
strands-decider soup checkpoints/strands-decider-2B-hobson-v20-s{0,1,2} \
  --out checkpoints/strands-decider-2B-hobson-v20
strands-decider calibrate checkpoints/strands-decider-2B-hobson-v20 --data data/holdout_v5_norule.jsonl
strands-decider calibrate checkpoints/strands-decider-2B-hobson-v20 --kinds choice,score --objective nll \
  --limit 900 --data data/multistep_v14_eval.jsonl --data data/holdout_v5_norule.jsonl
```

- `teacher_yn` labels the 54,857 yes/no rows of v19's training files with Qwen3.5-27B at a
  pinned revision, in bf16 (an 80 GB GPU, about 30 GPU-minutes on an H100), keeps the rows
  whose answer agrees with gold and writes `data/teacher_yn_qwen35-27b.jsonl`. The labels
  are not committed; [data/README.md](../../data/README.md) gives the hash of the file the
  recorded runs trained on. A relabel batches rows differently, so its probabilities need
  not match byte for byte.
- The three seeds share an initialisation (`init_seed`), so their weights average:
  `strands-decider soup` takes the exact mean of the LoRA updates and of the heads, and
  resets the calibration (the average of calibrated models is not calibrated).
- The two calibration steps are the usual one, then the choice/score refit of
  [docs/evaluating.md](../evaluating.md#calibration-refitting-only-some-temperatures)
  ("Fix A"), which keeps the yes/no temperature, on the two files the recorded refit used.
  The recorded fit capped the multi-step file at 900 rows and the held-out file at 1,800
  before halving; `--limit` caps each file's half, so a refit draws a similar, not
  identical, sample and gives similar, not identical, temperatures.

The options this needs are opt-in; every default trains as before:

| Option | What it does |
| --- | --- |
| `continue_from` | Trains an existing checkpoint's adapter and head (its calibration reset to 1.0), where `init_from` trains a fresh head on a frozen torso |
| `kl_frozen_skip_kinds` | No frozen-KL term on the listed row kinds; the frozen torso reads yes/no questions near chance, so anchoring yes/no rows to it pulls their answers toward 0.5 |
| `init_seed` | Seeds the initialisation apart from the data order, so seeds start from identical weights and can be averaged |
| `strands-decider soup` | Averages checkpoints that share an initialisation (`soup.json` records the inputs and library versions) |
| `python -m strands_decider.data.teacher_yn` | Labels the yes/no rows with the 27B teacher (`label`) and keeps those that agree with gold, over v14's replay distributions (`build`) |

## Lineage

1. **v19**, the published checkpoint, reproduced exactly (167 of 231).
2. **v19-yn27b**, the short run: v19 continued for 1,500 steps with the 27B yes/no teacher
   and no frozen-KL anchor on yes/no rows
   ([configs/experiments/v19-yn27b.yaml](../../configs/experiments/v19-yn27b.yaml), three
   seeds): 37.7 v1.5-style Intelligence (mean of 40.1 / 34.9 / 38.3).
3. **hobson-v20**: the same recipe for a full epoch (3,738 steps), three seeds from one
   initialisation, weight soup, then the choice/score temperature refit: 42.5.

## Limits

- Exploratory: local runs on public data, three seeds, arms chosen after earlier results;
  a published checkpoint needs a preregistered, confirmatory run
  ([CONTRIBUTING.md](../../CONTRIBUTING.md)).
- The v1.5 proxy is our reimplementation of the published rules, not the official scorer;
  the sealed half and Calibration are estimated, so the board position is an estimate.
- The unseen set is one draw of 1,050 rows: differences under about 3 points are noise.
