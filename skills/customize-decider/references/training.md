# Training a customized decider

Read `training/README.md` (setup, Gemma 4 settings, several GPUs) and `training/AGENTS.md`
(host runbooks and ground rules) before the first run. This file says what to change for a
user's own decisions.

## What to train

Fine-tune the released decider of the size the user will serve: `init_from` starts from a trained
checkpoint (a Hub id or a local directory) and trains its LoRA adapter and its head on the user's
rows. `training/README.md`, "Fine-tuning a released model with `init_from`", has the details. The
points that matter here:

- The checkpoint fixes the architecture. Leave `base_model`, `num_slots`, `head_type`,
  `pointer_dim`, `head_hidden`, `use_lora`, `lora_r`, `lora_alpha`, `lora_targets` and `force_bos`
  out of the config: a config that sets one of them stops before the model loads, even at a value
  that looks harmless. Do not start from a copy of `configs/train.yaml`.
- Released models are soups, so the adapter has rank 48 (three seeds of rank 16). A soup of three
  fine-tunes of it has rank 144.
- Temperatures reset to 1: calibrate the result.

Training from the base model instead (the recipes in `configs/v1/`) is for a user who wants a
different recipe, not only their data added. It needs the full corpus and teacher files
(`training/README.md`, "The v1 release recipes") and many more GPU hours.

## Environment

- An NVIDIA GPU. Qwen3.5 torsos need Linux (or WSL2) and `flash-linear-attention`; Gemma 4 torsos
  need transformers 5.15 or later. Install with `pip install -e ".[train,dev]"` plus the pinned
  packages of `training/README.md` "Setup".
- `init_from` the released model of the size the user will serve (Phase 4).

## Hosts

Peak GPU memory per rank at micro-batch 8, gradient checkpointing on, 4,096-token window
(`training/README.md`), and a one-GPU host that fits it:

| torso | peak memory | one GPU that fits | notes |
|---|---|---|---|
| Qwen3.5 2B | under 24 GiB | 24 GB (RTX 3090/4090, L4) or larger | about 11 h for the full v19 recipe on one RTX 3090 |
| Gemma 4 E2B | 14.5 GiB | 24 GB or larger | `host_embeddings: true` saves 4.4 GiB on one GPU |
| Gemma 4 E4B | 19.9 GiB | 48 GB (L40S) is safe; 24 GB is tight | `host_embeddings: true` on one GPU |
| Gemma 4 12B | 28.6 GiB | 48 GB (L40S) or 80 GB (H100) | |
| Gemma 4 26B-A4B | 52.9 GiB | 80 GB (H100) | |

On AWS, `training/aws/scripts/launch-host.sh` defaults to 8-GPU shapes (`SHAPES` in
`training/aws/README.md`). For one GPU, pass a one-GPU shape, for example
`SHAPES="g6e.xlarge"` (one L40S, 48 GB) or a one-H100 shape where the region has one. Look up the
hourly price for the shape and region, give the user the shape, region, hours and price, and wait
for a yes (`training/AGENTS.md` section 0). Eight GPUs train one run about 8 times faster with
`torchrun` (below), at about 8 times the hourly price.

Time: training prints the number of optimizer steps; the smoke run gives the time per step.

## Data to the host

`sync-code.sh` ships one git commit, and the user's data must never be committed. Copy the frozen
`work/` files through a bucket the user owns, after the user agrees that the data may go there:

```bash
# on your machine
aws s3 cp --recursive work/train s3://<user-bucket>/<run>/work/train
aws s3 cp --recursive work/eval  s3://<user-bucket>/<run>/work/eval
aws s3 cp work/train.yaml s3://<user-bucket>/<run>/work/train.yaml
# on the host (through ssm-run.sh), into the code directory
aws s3 cp --recursive s3://<user-bucket>/<run>/work /opt/hobson/code/<code-dir>/work
(cd /opt/hobson/code/<code-dir>/work/train && sha256sum -c SHA256SUMS)
```

The host's IAM role must be able to read that bucket. Copy checkpoints back the same way, and
delete the data from the host and the bucket when the user asks.

## Build the public corpus

The public corpus keeps the general skills of the model and its calibration on unfamiliar input.
Use the `data build` command of `training/steps.md` "1. Build the corpora", with three changes:
`--out work/train/public.jsonl`; no `--holdout` options (they write a second file); and
`--max-examples K` to cap the rows per recipe. The command names 28 recipes, so K = (customer rows)
/ 28 gives about one public row per customer row. Its sources and licences are in
`data/sources.md`; the user decides whether those licences suit their use.

## Config template

Save as `work/train.yaml`. Fields are those of `TrainConfig` in `training/train.py`;
`configs/train.yaml` explains each in its comments (but do not copy its architecture keys).

```yaml
init_from: "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"   # the size the user will serve
train_files: ["work/train/train.jsonl", "work/train/public.jsonl"]
val_fraction: 0.03
val_split_seed: 0          # every seed validates on the same rows
# host_embeddings: true    # Gemma E2B and E4B on ONE GPU only; never under torchrun

# Keep the answers close to the released model's own. Lower the weight (0.1) if the user's
# rows do not move enough on the dev half; raise it if the general-skills set falls.
kl_frozen_weight: 0.3
kl_frozen_reference: "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"
precompute_frozen_kl: true # needs group_by_length
group_by_length: true

epochs: 1
micro_batch_size: 8
grad_accum: 4              # 32 rows per optimizer step
lr: 0.0001
head_lr: 0.0001            # the head is already trained; 1e-3 (the default) is for a new head
weight_decay: 0.01
warmup_ratio: 0.03
max_grad_norm: 1.0
gradient_checkpointing: true
shuffle_options: true      # never turn this off
head_dropout: 0.05
ordinal_smoothing: 0.1
reverse_score_prob: 0.5

seed: 0
output_dir: "work/ckpt/s0"
log_every: 20
eval_every: 500
```

No teacher file: the user's rows have gold labels, and a teacher file attaches to rows by
position. The learning rates are starting points, not measured optima for fine-tuning; if the
smoke run's loss does not fall, or the first full seed moves the dev half less than expected,
try `lr` 2e-4 on one seed before you train three.

## Smoke run

```bash
sed -e 's/^log_every: .*/log_every: 2/' work/train.yaml > work/train-smoke.yaml
python -m training.train --config work/train-smoke.yaml --output-dir work/ckpt/smoke --max-steps 20
```

Check the printed lines:

- `trainable N / M`: a small share (LoRA plus the head), not 0 and not 100%. From a released
  Qwen3.5 2B soup it is 53.6M of 1.94B (2.8%).
- `train=T val=V`: the row counts you expect.
- A loss line every 2 steps, and a loss that falls over the 20 steps.
- The time per step (wall time / 20). With `--max-steps` the printed step count is 20, so compute
  the full run: steps ≈ T / (micro_batch_size × grad_accum) × epochs. Full time ≈ steps × time
  per step. Give the user that time and its cost before the full runs.

## Three seeds

```bash
for s in 0 1 2; do
  sed -e "s/^seed: .*/seed: $s/" -e "s#^output_dir: .*#output_dir: \"work/ckpt/s$s\"#" \
    work/train.yaml > work/train-s$s.yaml
  python -m training.train --config work/train-s$s.yaml
done
# or one run on eight GPUs (without host_embeddings)
torchrun --standalone --nproc_per_node=8 -m training.train --config work/train-s0.yaml
```

On a cloud host, run long jobs in the background as `training/AGENTS.md` describes, not in a
held-open session.

## Monitoring

Training prints a loss line every `log_every` steps (and `kl` when `kl_frozen_weight` > 0), and
`eval @ <step>: {'val_loss': …, 'val_acc': …}` every `eval_every` steps. At the end it writes
`history.json` into `output_dir`. Watch for:

- **Healthy:** the loss falls fast in the first few hundred steps, then slowly; `val_loss` falls
  and `val_acc` rises, then both flatten.
- **Divergence:** the loss rises and stays up, or becomes NaN. Stop the run; do not soup it. Retry
  with half the `lr`, and report it.
- **Overfitting:** `val_loss` rises while the training loss still falls. With one epoch this is
  rare; if it happens, the customer rows are probably repeated too often.
- **Seeds disagree:** compare the three `history.json` files. One seed far from the other two is a
  failed run, not a lucky one.

## Calibration quality

After training, for each seed, score the uncalibrated model:

```bash
python -m training.evaluate eval work/ckpt/s0 --data work/eval/eval.jsonl --limit 1000000 --raw
```

Then soup, calibrate and evaluate (SKILL.md Phase 7). Check:

- ECE on the test half falls after calibration, and stays below the released model's ECE on the
  same half.
- The accuracy at the Phase 8 thresholds meets the user's target for automatic action.
- The `by_confidence_band` table: accuracy should rise with the band. A band whose accuracy is
  below a lower band means the confidence is not usable there.
- `noul` answers near 0.5: many more than the released model means the model hedges; it fails the
  user's automation even when accuracy is the same.

Calibration cannot change a `choice` pick, so choice accuracy does not change. It does change
`noul` and `score` values.

## Soup

`python -m training.soup` averages the three seeds exactly (LoRA updates and head logits) and resets
the temperatures; calibrate it before you evaluate or serve it. The seeds must share one recipe and
one base revision (`training/steps.md` "Seeds and soups").
