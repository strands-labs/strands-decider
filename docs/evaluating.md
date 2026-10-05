# Evaluating a checkpoint, and checking a claim

How to measure any Strands Decider checkpoint (a local training run, a Hub repo, or a
downloaded archive) the way the candidate models in [docs/models/](models/) were measured,
and how to check a number someone reports. Every measure here is **local**: none of them is
a JevBench board score.

Three measures, each with a paired comparison against a baseline:

| Measure | Script | What it is |
| --- | --- | --- |
| JevBench v1 public, scored with a local v1.5 proxy | `evaluation/jevbench/jevbench.sh`, then `evaluation/jevbench/v15_proxy.py` | The 231 public tasks, answered by `strands-decider serve`, graded with JevBench v1.5's published rules |
| Unseen task families | `evaluation/unseen/` | 1,050 questions: four public families no model trains on, plus RuleTaker at a held-out depth (depths 0-2 are trained on), graded with the same rules |
| Images | `evaluation/vision/run.py` | NaturalBench, POPE and the rebuildable Image JevBench preview items ([docs/vision.md](vision.md#how-well-it-does)) |

Paths are relative to the repository root unless they are links.

## Setup

A Linux host with one NVIDIA GPU for JevBench (the script serves on a GPU); the unseen set
and the image evaluation also run on CPU, slowly (pass `--device cpu` to
`evaluation/unseen/run.py`, which defaults to `cuda`; `evaluation/vision/run.py` falls back
to the CPU by itself).

```bash
pip install -e ".[train,vision,cuda]"    # train: datasets for the unseen set's build
```

A checkpoint is a local directory or a Hub repo id. A downloaded archive is unpacked first
and checked against its published hash:

```bash
NAME=strands-decider-2B-hobson-v20                     # any name in docs/models/
URL=https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights
curl -fO "$URL/$NAME.tar" && curl -fO "$URL/weights_SHA256SUMS.txt"
grep " $NAME.tar\$" weights_SHA256SUMS.txt | sha256sum -c -
mkdir -p checkpoints && tar -xf "$NAME.tar" -C checkpoints  # -> checkpoints/$NAME/
(cd "checkpoints/$NAME" && sha256sum -c SHA256SUMS)       # every file in the archive
```

The baseline below is the published v19 at its pinned revision:

```bash
hf download StrandsAgents/strands-decider-2B-hobson-v19 \
  --revision bb282d786bc251fd4e3068de3ada9ddbb38127cd --local-dir checkpoints/v19
```

## JevBench v1 public, scored with the local v1.5 proxy

```bash
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/$NAME reports/jevbench/$NAME
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/$NAME --vs reports/jevbench/v19
```

For a checkpoint served with `--vision` (an image-trained one), pass the flag to the server
it starts: `SERVE_ARGS=--vision`. JevBench's questions are text, so they take the text path,
on the weights `serve --vision` answers text with; the script checks that `/health` reports
the vision server.

[`v15_proxy.py`](../evaluation/jevbench/v15_proxy.py) applies v1.5's published rules to the
public results:

- a yes/no answer with 0.2 < P(yes) < 0.8 counts as wrong, and yes/no credit is
  chance-corrected: (right - 1/2) / (1/2);
- choice credit is chance-corrected: (right - 1/n) / (1 - 1/n);
- a score task is graded by the **expected level** of the returned distribution, against a
  level drawn at random: 100 * (1 - mean nMAE / mean nMAE of a random level), with each
  task's gold level read from the run's `all.jsonl`;
- Intelligence is the mean of the three question types.

It also reports the yes/no answers inside the band, the tasks right and the ECE. It leaves
out v1.5's tier weights and the sealed half, so it ranks runs against each other on one
machine; it is not the board's score. Given several runs before `--vs` (seeds of one
recipe), it reports their mean difference from the baseline with a paired bootstrap 95%
interval over the 231 tasks.

## Unseen task families

```bash
training/recipe.sh build             # the short-task corpus; its held-out file has the RuleTaker rows
python evaluation/unseen/build.py --out data/unseen.jsonl
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/$NAME \
  --out reports/unseen/$NAME
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/v19 --out reports/unseen/v19
python evaluation/unseen/score.py reports/unseen/$NAME --vs reports/unseen/v19
```

1,050 questions: four families no training mixture contains -- StrategyQA (250 yes/no),
CommonsenseQA (150 choice), ARC-Challenge (150 choice) and STS-B (250 score) -- plus
RuleTaker at depth 5 (250 yes/no), a depth no training mixture contains (training uses
depths 0-2). Each is pinned to a dataset revision and drawn with a fixed seed, and graded
with the same v1.5 rules. The public tasks reward familiarity with JevBench's formats; this
set asks how a checkpoint does on questions it never saw. `run.py --vision` loads a checkpoint as `serve --vision` does. Other
deciders run through their own published code or HTTP API behind an adapter
([evaluation/unseen/README.md](../evaluation/unseen/README.md)), so the same rows rank them
too. One draw of 1,050 rows: differences of a few points are within noise; read the
interval `--vs` prints.

## Calibration: refitting only some temperatures

Calibration fits temperatures on held-out rows and writes them into the checkpoint; it
never changes an answer's argmax. Two options let a refit leave the other temperatures
alone:

```bash
strands-decider calibrate CKPT --data data/holdout_v5_norule.jsonl          # as before: every temperature, by ECE
strands-decider calibrate CKPT --kinds choice,score --objective nll --limit 900 \
  --data data/multistep_v14_eval.jsonl --data data/holdout_v5_norule.jsonl
```

- `--kinds` refits only the listed question types' temperatures and keeps the checkpoint's
  others, the global one included. A yes/no temperature that makes answers decisive is then
  not softened back into the 0.2-0.8 band by a set that only needs its choice or score
  temperatures corrected.
- `--objective nll` fits by negative log-likelihood instead of ECE.
- `--data` may be repeated; each file is split and capped (`--limit`) on its own, so a large
  held-out set does not crowd out small ones.

Without these options, calibration is unchanged. The second command is the refit the
candidate models were served with (their documents call it "Fix A"): the rebuilt
`data/holdout_v5_norule.jsonl` no longer matches `data/SHA256SUMS` (an upstream held-out
dataset has changed), and on it the first step alone gave some checkpoints score
temperatures of 3.7-4.6, which flatten score answers. The recorded refit used these two
files only, the multi-step file capped at 900 rows and the held-out file at 1,800 before
halving; `--limit` caps each file's calibration half, so this command draws a similar but
not identical sample, and its temperatures need not match the recorded ones exactly. The
published archives already carry their fitted temperatures; refit only to reproduce a
checkpoint you trained yourself.

## Checking a claim

A model document in [docs/models/](models/) gives, for each number, the command that
produced it. To check one:

1. Download the archive and check its sha256 (above). The archive also holds the exact
   training config (`train_config.json`) and a `SHA256SUMS` of its files.
2. Serve it with the document's serve command and send one request
   ([docs/inference.md](inference.md)).
3. Run the measure on it and on the baseline, with the commands above, and compare with
   `--vs`. Read the interval: on 231 tasks, differences under a few points of the proxy are
   within run-to-run noise; on the unseen set, under about 3 points.
4. Expect small differences from the recorded numbers: GPU kernels, batch shapes and library
   versions move bf16 probabilities, and an answer near 0.2 or 0.8 can cross the band.
   `evaluation/unseen/run.py` records the library versions it ran with (`run_meta.json`);
   record yours for the other measures.

The recorded runs, per seed, with every table, are in the results log:
https://github.com/Vivek0712/strands-decider/blob/ed29b08bb71aad26133bb6c45e8027648f5725ef/RESULTS.md.
