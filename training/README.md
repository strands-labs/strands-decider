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
- [`../configs/`](../configs/): `train.yaml`, its parent `train-parent.yaml`,
  `experiments/` for the preregistered runs, and `v1/` for the v1 releases
  ([The v1 release recipes](#the-v1-release-recipes)).
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

### Gemma 4 torsos

Four Gemma 4 instruction-tuned checkpoints work as torsos, all Apache-2.0. Set `base_model`
to one of them; transformers 5.15, the lowest version the package admits, has their classes.

| `base_model` | loaded through | per-layer table | peak GPU memory per rank |
| --- | --- | --- | --- |
| `google/gemma-4-E2B-it` | `Gemma4ForConditionalGeneration` | yes | 14.5 GiB |
| `google/gemma-4-E4B-it` | `Gemma4ForConditionalGeneration` | yes | 19.9 GiB |
| `google/gemma-4-12B-it` | `Gemma4UnifiedForConditionalGeneration` | no | 28.6 GiB |
| `google/gemma-4-26B-A4B-it` | `Gemma4ForConditionalGeneration` (mixture of experts) | no | 52.9 GiB |

The loader keeps the text decoder and drops the vision and audio encoders. The memory
column is what our training runs peaked at on H100s, at micro-batch 8 with
`gradient_checkpointing: true` and a 4,096-token window, so every size trains on one
8-GPU host.

Settings for a Gemma torso:

- `force_bos: true`. The -it tokenizers add no BOS token (their chat template writes
  it), and Gemma reads a prompt without it badly: the frozen E2B torso reads JevBench at
  chance without it.
- `lora_targets: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
  "down_proj"]`. Gemma has no recurrent layers, so Qwen3.5's `in_proj_*` and `out_proj`
  targets do not apply.
- `host_embeddings: true` (one GPU only) keeps the E2B/E4B per-layer embedding table
  (E2B: 2.35B parameters, 4.4 GiB in bf16) in host memory. It is a lookup table and never
  trained. Under `torchrun` it is refused, because DDP refuses parameters on the CPU.

**26B-A4B: the experts are not adapted.** Each layer has a mixture-of-experts block (128
experts, top 8) beside a dense MLP. LoRA attaches by module name, so it reaches attention
and the dense MLP. The experts are stored as 3-D parameters, not as linear modules, and
neither they nor the router get an adapter: they keep their base weights.

The frozen option-number readout and the teacher labeller apply Gemma's final logit
soft-cap, so both read the model's own distribution. At serving time, a 32,768-token
window with several long questions in one forward can run a Gemma torso out of memory:
`strands-decider serve --max-batch-tokens N` caps the tokens of one forward
([Serve](../docs/inference.md#serve)).

### Training on several GPUs

`NGPU=8 training/recipe.sh all` runs the recipe on eight GPUs with the commands below. The same
configs train under `torchrun`:

```bash
torchrun --standalone --nproc_per_node=8 -m training.train --config configs/train-parent.yaml
torchrun --standalone --nproc_per_node=8 -m training.train --config configs/train.yaml
```

Each optimizer step trains the same 32 rows, in the same micro-batches, as one GPU does
(`training/distributed.py`). The ranks share a step's rows out by padded length, and no
forward is larger than a micro-batch. Loss terms are rescaled to the full micro-batch, so
DDP's gradient average is the single-GPU gradient, and each rank draws the collator's
augmentations for the rows it skips, so option orders and phrasings match. Rank 0 alone
validates, saves and logs; the logged losses are all-reduced. Any world size from 2 to 32
works. Without `torchrun` the original loop runs,
unchanged.

Teacher and replay labelling split into one process per GPU, then merge:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m training.data.teacher --src data/multistep_v14.jsonl \
    --out data/teacher_multistep_v14.jsonl --num-shards 8 --shard-index $i &
done; wait
python -m training.data.teacher --merge --num-shards 8 --src data/multistep_v14.jsonl \
  --out data/teacher_multistep_v14.jsonl --shift-by data/train_v5.jsonl
```

`training.data.replay` takes the same flags. Each shard labels whole batches of the
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

**KL direction.** `kl_direction: reverse` turns both KL terms, the teacher's and the frozen
anchor's, from KL(target || student), which covers the target's mass, into KL(student ||
target), which seeks its mode. The default, `forward`, is the formula of every earlier run,
bit for bit. A teacher file rounds to six decimals, so an option can have probability 0;
reverse KL floors the teacher on every real option at 1e-6 and renormalises, so student
mass there is penalised rather than skipped.

**Is it the same training?** On CPU (`pytest -m distributed`: a tiny Qwen3 with LoRA and
the pointer readout, frozen KL, a partial teacher, mixed row weights, KL-only rows, across
epoch boundaries), the all-reduced gradient at 2, 3, 4 and 8 ranks matches one process to
within 4e-6 relative on every step, with the same rows rendered token for token. On 8x
H100 with the real torso, bf16, dropout off, 40 steps at 8 GPUs track 1 GPU step by step
(loss 1.661 against 1.660 at step 1, 0.912 against 0.903 at step 40). Splitting a
micro-batch across ranks changes bf16 rounding by 1e-2 to 4e-2 of the gradient; dropout,
as shipped, changes it by 0.44.

### Fine-tuning a released model with `init_from`

`init_from: <checkpoint>` starts training from a trained checkpoint instead of the base
model, and trains it on your rows. The checkpoint is a Hub id or a local directory. Its
LoRA adapter and its head, pointer or slot, start from their trained values, and both
train. The torso is not frozen, and `gradient_checkpointing` applies as usual. It works
under `torchrun` as a run from the base model does.

```yaml
init_from: "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"
train_files: ["data/my_rows.jsonl"]
val_fraction: 0.05
group_by_length: true
# Keep the answers close to the released model's own:
kl_frozen_weight: 0.3
kl_frozen_reference: "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"
precompute_frozen_kl: true
output_dir: "checkpoints/my-e4b"
```

```bash
python -m training.train --config my-finetune.yaml
python -m training.evaluate calibrate checkpoints/my-e4b --data <held-out rows>
```

- **The checkpoint fixes the architecture.** `base_model`, `num_slots`, `head_type`,
  `pointer_dim`, `head_hidden`, `use_lora`, `lora_r`, `lora_alpha`, `lora_targets` and
  `force_bos` come from the checkpoint's `strands_decider_config.json`. Leave them out of
  the config. A config that sets one of them to another value stops before the model loads,
  with one error that names every such key. A key in the YAML file counts as set even at its
  default value, so a copy of `configs/train.yaml` (`lora_r: 16`) stops on a released soup.
  `max_length` is the checkpoint's unless the config sets it. Everything else (data,
  learning rates, epochs, `head_dropout`, `ordinal_smoothing`, the KL terms) is the
  config's, and is saved with the result. `head_lr` defaults to 1e-3, a value chosen for a
  randomly initialised head; here the head is already trained.
- **Released models are soups** ([Seeds and soups](steps.md#seeds-and-soups)): a rank-N·r
  adapter and a pointer head with `pointer_dim` N·d. They load and train as any checkpoint.
- **Temperatures** are reset to 1 at the start, as for a soup: calibrate the result before
  you evaluate or serve it.
- **The frozen-KL anchor.** With `kl_frozen_reference` set to the `init_from` checkpoint,
  the KL term starts near 0 (exactly 0 with dropout off) and keeps the fine-tune close to
  the released model's answers.
  Without `kl_frozen_reference`, the anchor is the base torso's own option-number readout,
  with the adapter disabled, as in a run from the base model.
- **The result is a normal checkpoint**: `strands-decider serve` and `ask`,
  `python -m training.evaluate` and `training.soup` read it. To soup fine-tunes, train
  several seeds with one config (the same `init_from` and settings, other `seed` values):
  `soup` refuses checkpoints whose configs differ in more than their temperatures. The soup
  of N fine-tunes of a rank-R checkpoint has rank N·R: three seeds on a rank-48
  soup give a rank-144 adapter.
- `freeze_torso: true` trains the head alone, on the checkpoint's frozen adapter.
  `head_init` must stay `random`: the trained head is kept.

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

## The v1 release recipes

`configs/v1/` holds one config for each released v1 model: the seed-0 config of its
training run, with only `output_dir` and the comments changed. Each release is a soup of
three runs of that config, seeds 0, 1 and 2, which differ only in `seed`. The soup was
calibrated afterwards. The header of each config lists its inputs, their sha256, and the
rerun commands: `recipe.sh train` once per seed, then `python -m training.soup` and
`python -m training.evaluate calibrate`. All five configs keep the answers close to the
released v19 (`kl_frozen_reference: checkpoints/v19-ref`); `hf download
StrandsAgents/strands-decider-2B-hobson-v19 --local-dir checkpoints/v19-ref` gets it.

| release | config | base model | data | teacher | to rerun it |
| --- | --- | --- | --- | --- | --- |
| `strands-decider-2B-qwen3.5-v1-2610` | `configs/v1/strands-decider-2B-qwen3.5-v1-2610.yaml` | `Qwen/Qwen3.5-2B-Base` | the v1 mix, 246,678 rows | gemma-4-31B-it (`data/teacher_g28.jsonl`) | the four mix files that are not v21's (`train_v5_nc`, `code_g28`, `gapfill_g28`, `rule_apply_g28`) and the teacher file: none is in this repository or built by it |
| `strands-decider-E2B-gemma4-v1-2610` | `configs/v1/strands-decider-E2B-gemma4-v1-2610.yaml` | `google/gemma-4-E2B-it` | v21's data plus code tasks, 135,602 rows | Qwen3.5-4B (`data/teacher_v20.jsonl`, from `recipe.sh distill`) | the code-task rows, `data/code_v27.jsonl`, which are built outside this repository |
| `strands-decider-E4B-gemma4-v1-2610` | `configs/v1/strands-decider-E4B-gemma4-v1-2610.yaml` | `google/gemma-4-E4B-it` | the same | the same | the same |
| `strands-decider-12B-gemma4-v1-2610` | `configs/v1/strands-decider-12B-gemma4-v1-2610.yaml` | `google/gemma-4-12B-it` | the same | the same | the same |
| `strands-decider-26B-A4B-gemma4-v1-2610` | `configs/v1/strands-decider-26B-A4B-gemma4-v1-2610.yaml` | `google/gemma-4-26B-A4B-it` | the same | the same | the same |

Everything else the five configs read is built by `training/recipe.sh` (`build`, `fetch`,
`multistep`, `generated`, `adequacy`, `distill`). The config headers and `data/SHA256SUMS` give the sha256 of
each file the runs read, so a rebuild can be checked against them. The runs used the code of the development repository, not this repository's `main`; the settings
they use are on `main` (`kl_frozen_reference`, the Gemma 4 torsos and the soup). The
four Gemma 4 configs differ only in `base_model` and `output_dir`
(`training/tests/test_configs.py`).
