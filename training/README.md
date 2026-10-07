# Training

This folder holds the v19 reference recipe and what runs it. You need one Linux or WSL2
host with 1 to 8 NVIDIA GPUs. `training/recipe.sh all` builds the corpora, trains, calibrates
and evaluates: about 11 h on one RTX 3090, or 1 h 10 min on 8x H100 with `NGPU=8 FAST=1`
through the AWS runner. Without `NGPU`, the recipe runs on one GPU. The torso is
`Qwen/Qwen3.5-2B-Base` and the teacher is `Qwen/Qwen3.5-4B`; the recipe downloads both
from Hugging Face at first use. The recipe calibrates before it evaluates; calibrate before
you serve. A retrain reproduces v19 when it scores 157 to 177 on JevBench public, is not
significantly different from v19 per task (McNemar, p ≥ 0.05), and reaches at least 0.689
on HelpSteer2 ([Retraining on AWS](../evaluation/results.md#retraining-on-aws)).
`evaluation/jevbench/jevbench.sh <checkpoint> <out_dir>` serves the checkpoint and runs the
JevBench public set against it
([Reproducing, and two caveats](../evaluation/jevbench.md#reproducing-and-two-caveats)). The Setup
commands are written for WSL2; on native Linux, skip the `wsl.exe` line and the rest applies.

- [`recipe.sh`](recipe.sh): the recipe, one stage per argument. [Usage](#usage) lists the stages.
- [`run_recipe.sh`](run_recipe.sh): the runner for an 8-GPU host. It times each stage, checks
  row counts and records the run.
- [`steps.md`](steps.md): the six steps one by one, with their commands.
- [`hardware.md`](hardware.md): memory, throughput, length-grouped batching and larger torsos.
- [`aws/`](aws/README.md): one EC2 host over SSM, and the measured stage timings.
- [`AGENTS.md`](AGENTS.md): runbooks for a coding agent that runs, operates or verifies training.
- [`recipe_v7.sh`](recipe_v7.sh) and [`legacy-v7.md`](legacy-v7.md): the v7 route on Windows.
- [`../configs/`](../configs/): `train.yaml`, its parent `train-parent.yaml`, and
  `experiments/` for the preregistered runs.
- [`../data/README.md`](../data/README.md): what the recipe downloads, and what is committed.

Evaluation is in [evaluation/README.md](../evaluation/README.md), serving in
[docs/inference.md](../docs/inference.md). Paths are relative to the repository root
unless they are links.

## Two training routes

The recipe gets its training targets in one of two ways.

- **Regenerated targets: the v19 reference recipe.** `training/recipe.sh` builds the corpora and
  labels the multi-step rows with the frozen Qwen3.5-4B teacher (`teacher`). It then
  trains a parent on v14's recipe (`parent`), labels the multi-step rows with the parent's
  own distributions (`replay`), and trains v19 toward them (`train`). This takes about
  11 h on one RTX 3090, and 1 h 10 min on 8x H100 through the AWS runner with `FAST=1`.
  `FAST` is a setting of `training/run_recipe.sh`, not of `recipe.sh`.
- **Frozen targets: the v20 route, diagnostic.** `distill` builds the teacher file from
  the 4B and replay distributions committed in `data/synthetic/`. No teacher labelling
  and no parent training are necessary. [Usage](#usage) gives the commands.

Frozen targets attach to corpus rows by position, so `recipe.sh` checks the corpus against
`data/SHA256SUMS` before each stage that uses them, and stops on a mismatch
([Reproduction contract](../data/README.md#reproduction-contract)).

## Setup

Two environments for training, one per torso family. Both need an NVIDIA GPU. Development
used one RTX 3090 (24 GiB), and the recipe also runs on 8-GPU AWS hosts
([Running the recipe on AWS](aws/README.md)). Serving also runs on an Apple-silicon Mac
([Serving on a Mac](../docs/inference.md#serving-on-a-mac)).

**Qwen3 torsos (v7 and earlier): Windows or Linux.** Python 3.11, PyTorch 2.6 + CUDA 12.4 at
the time. The package now needs PyTorch 2.7 or later:

```bash
pip install -e ".[train,dev]"
```

**The Qwen3.5 torso (v13 onwards; v19 is the default): Linux, or WSL2 on Windows.** Its Gated
DeltaNet layers need `flash-linear-attention`, whose Triton kernels run on Linux only. On
Windows transformers falls back to a PyTorch implementation that is ~2.3x slower to
train and materialises large fp32 intermediates; the 4B teacher labelling long
documents spilled out of GPU memory that way. Under WSL2 the GPU runs at native speed.

```bash
wsl.exe --install -d Ubuntu-24.04
```

Inside it (the `apt` line needs your password; Triton compiles a small C launcher at
runtime, hence `build-essential`):

```bash
sudo apt-get install -y python3.12-venv python3.12-dev build-essential
python3 -m venv ~/venvs/hobson
~/venvs/hobson/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
~/venvs/hobson/bin/pip install transformers==5.17.0 peft==0.21.0 flash-linear-attention
~/venvs/hobson/bin/pip install -e ".[train,dev]"
```

`recipe.sh` runs the active environment's `python`. Activate this environment
(`source ~/venvs/hobson/bin/activate`), or set `PY=~/venvs/hobson/bin/python`.

Keep the repository, the data and the HuggingFace cache on the Linux filesystem, not
under `/mnt/c`; reads across the boundary are slow enough to bottleneck loading.
`causal_conv1d` stays on its PyTorch path (a small depthwise convolution; the fused
kernel needs the CUDA compiler). torch 2.7 matters: `fla` wants Triton 3.3, which torch
2.6 does not ship.

### Training on several GPUs

`NGPU=8 training/recipe.sh all` runs the recipe on eight GPUs with the commands below. The same
configs train under `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m strands_decider.cli train --config configs/train-parent.yaml
torchrun --standalone --nproc_per_node=8 -m strands_decider.cli train --config configs/train.yaml
```

Each optimizer step trains the same 32 rows, in the same micro-batches, as one GPU does
(`src/strands_decider/distributed.py`). The ranks share a step's rows out by padded length, and no
forward is larger than a micro-batch. Loss terms are rescaled to the full micro-batch, so
DDP's gradient average is the single-GPU gradient, and each rank draws the collator's
augmentations for the rows it skips, so option orders and phrasings match. Rank 0 alone
validates, saves and logs; the logged losses are all-reduced. Any world size from 2 to 32
works. Without `torchrun` the original loop runs,
unchanged.

Teacher and replay labelling split into one process per GPU, then merge:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m strands_decider.data.teacher --src data/multistep_v14.jsonl \
    --out data/teacher_multistep_v14.jsonl --num-shards 8 --shard-index $i &
done; wait
python -m strands_decider.data.teacher --merge --num-shards 8 --src data/multistep_v14.jsonl \
  --out data/teacher_multistep_v14.jsonl --shift-by data/train_v5.jsonl
```

`strands_decider.data.replay` takes the same flags. Each shard labels whole batches of the
single-process batching, so the merge writes the single-process files byte for byte.

With 80 GB GPUs, `gradient_checkpointing: false` and `precompute_frozen_kl: true` train
2.2x faster at 8 GPUs. The first is exact. The second computes the frozen-KL reference in
one pass before step 1 instead of a second forward every step, which changes only bf16
rounding. Neither fits a 24 GB card, so the committed configs keep them off.

**A trained checkpoint as the frozen-KL reference.** `kl_frozen_reference: <checkpoint>`
replaces the untouched torso's option-number readout with that checkpoint's own option
distribution, at temperature 1 and without its calibration. The same rows get the term (at
most as many options as the readout covers), weighted by `kl_frozen_weight` as before. It
needs `precompute_frozen_kl: true` and `kl_frozen_weight > 0`: the reference is loaded
once, fills the table before step 1, and is freed before training starts. The
checkpoint may use another tokenizer than the student, for example a Qwen3.5 reference for
a Gemma 4 student. The reference renders each row with its own tokenizer. The collator
draws option order and instruction phrasing from its seed before it tokenises, so the
reference sees every row with the same options in the same slots as the student. Unset, the
default, training is unchanged.

**Is it the same training?** On CPU (`pytest -m distributed`: a tiny Qwen3 with LoRA and
the pointer readout, frozen KL, a partial teacher, mixed row weights, KL-only rows, across
epoch boundaries), the all-reduced gradient at 2, 3, 4 and 8 ranks matches one process to
within 4e-6 relative on every step, with the same rows rendered token for token. On 8x
H100 with the real torso, bf16, dropout off, 40 steps at 8 GPUs track 1 GPU step by step
(loss 1.661 against 1.660 at step 1, 0.912 against 0.903 at step 40). Splitting a
micro-batch across ranks changes bf16 rounding by 1e-2 to 4e-2 of the gradient; dropout,
as shipped, changes it by 0.44.

## Usage

`training/recipe.sh` builds, trains and evaluates the default, v19, end to end under WSL2; pass
one or more step names (`build`, `fetch`, `multistep`, `generated`, `adequacy`,
`catchall`, `teacher`, `distill`, `parent`, `replay`, `train`, `calibrate`, `eval`) to run
those. It trains twice: a parent on v14's recipe (`configs/train-parent.yaml`), whose
answers on the multi-step rows become the targets for v19 (`configs/train.yaml`) — about
11 h in all. `training/recipe_v7.sh` does the same for v7 on Windows.

The recipe downloads the public datasets and copies the committed synthetic files.
[Committed and downloaded data](../data/README.md#committed-and-downloaded-data) lists both.

**Retraining v20** (`configs/experiments/v20.yaml`, [PREREGISTRATION-v20.md](../research/preregistrations/PREREGISTRATION-v20.md)):

```bash
training/recipe.sh build fetch multistep generated adequacy catchall distill
TRAIN_CONFIG=configs/experiments/v20.yaml CKPT=checkpoints/hobson-2b-v20-retrain \
  training/recipe.sh train calibrate eval
```

`generated` attaches v20's checked paraphrases to the generated questions and writes the
paired consistency evals; `catchall` derives v20's catch-all rows from the short-task
corpus; `distill` builds v20's teacher file from the committed 4B and replay
distributions, so no parent is trained. `train` always writes to `$CKPT`, whatever the
config's output directory says. Given the same short-task corpus and downloads, these steps
reproduce every one of v20's training and evaluation files byte for byte; the
multi-step sets also rebuild exactly from the downloads. The catch-all and teacher files
depend on the short-task corpus ([1. Build the corpora](steps.md#1-build-the-corpora)). `eval` then
adds v20's measures: the catch-all set, and paraphrase consistency and instruction-flip
pairs (`evaluation/pair_eval.py`).

The steps, each with its command, are in [steps.md](steps.md):
[1. Build the corpora](steps.md#1-build-the-corpora),
[2. Label with the teacher](steps.md#2-label-with-the-teacher), [3. Train](steps.md#3-train),
[4. Calibrate](steps.md#4-calibrate), [5. Evaluate](steps.md#5-evaluate) and
[6. Serve](steps.md#6-serve).
