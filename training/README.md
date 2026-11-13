# Training

One trainer, `python -m training.train`, trains both kinds of Strands Decider model. Both are LoRA adapters on a
frozen decoder plus a trained readout, and differ in the readout and the prompt:

- The token readout (`head: token_readout`) is the readout of the Strands Decider v2 models: Gemma 4's text decoder
  under the `prefix_bidir` attention mask, prompts rendered by `strands_decider.core` with the `strands-chat`
  template, and a [255, hidden] readout at the last prompt token. A run's adapter is merged into the base, the
  seeds' merged checkpoints are averaged, and the average is exported as the decision interface's checkpoint
  ([`docs/spec.md`](../docs/spec.md), "the spec" below).
- The pointer head (`head: pointer`) is v1's readout: a Qwen3, Qwen3.5 or Gemma 4 torso under the causal mask, v1's
  numbered prompt, and the PointerHead, which scores each option from the hidden state of its own line. Its saves
  are v1 checkpoints, which `strands-decider serve` and v1's other engines load as they load v1's releases.

"v1" below means the models released before v2 (v19 and the `-v1-2610` releases) and the trainer that trained them,
the `training/train.py` this one replaced. That trainer is at commit `79bf6e7` of `main`, the last before this
change. "v1's corpora" are the files this repository builds and v20 trained on (`training/steps.md`); the `-v1-2610`
releases trained on other files too, which are not in the repository.

The trainer runs in one process on one GPU, or under `torchrun --standalone` on the GPUs of one machine. The token
readout path is the code that trained the v2 models, checked against that trainer bit for bit
([How the token-readout path was checked](#how-the-token-readout-path-was-checked)). The pointer path drives v1's
own model, render and optimizer through the same loop; [What differs from v1's trainer](#what-differs-from-v1s-trainer)
lists where its training differs.

Paths in text are relative to the repository root; the table names files in `training/`.

| File | What it does |
|---|---|
| [`train.py`](train.py) | The trainer, for both heads |
| [`configs/`](configs) | One YAML per Gemma 4 size for the token readout (`e2b`, `e4b`, `12b`, `26b-a4b`, `31b`), and the pointer head's (`pointer-2b`, `pointer-e2b`) |
| [`convert.py`](convert.py) | Decision rows, or v1's examples, into the trainer's rows, each with its prompt length |
| [`batching.py`](batching.py), [`targets.py`](targets.py), [`schedule.py`](schedule.py), [`loss.py`](loss.py) | The step plan, the targets and option shuffling, the learning rate, the loss |
| [`folds.py`](folds.py), [`dev_eval.py`](dev_eval.py), [`metrics.py`](metrics.py) | The training rows and held-out folds, the dev evaluation, its metrics |
| [`model.py`](model.py), [`prompt.py`](prompt.py), [`attention.py`](attention.py) | The token readout's model, its rendering through the core, the prefix_bidir mask and the split attention |
| [`lora.py`](lora.py), [`expert_lora.py`](expert_lora.py) | The token readout's adapters and their merge arithmetic |
| [`merge.py`](merge.py), [`average.py`](average.py), [`export.py`](export.py) | The token readout's merge, average of runs and export |
| [`pointer.py`](pointer.py) | The pointer head: v1's model, render, augmentation and checkpoint behind the trainer's interface |
| [`distributed.py`](distributed.py) | The GPUs of one machine under torchrun |
| [`soup.py`](soup.py), [`evaluate.py`](evaluate.py), [`hf_export.py`](hf_export.py) | The pointer head's average of seeds, calibration and evaluation, and Hugging Face export |
| [`data/`](data), [`recipe.sh`](recipe.sh), [`steps.md`](steps.md) | v1's corpus builders, and v1's data with the pointer head end to end |
| [`evaluation.md`](evaluation.md) | Evaluating a token-readout export |
| [`hardware.md`](hardware.md) | Memory and throughput notes for the pointer head |
| [`tests/`](tests) | pytest on the CPU; `tests/gpu/check_model.py` on one GPU |

## Setup

Linux with NVIDIA GPUs, and Python 3.10 or newer:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[train,dev]"
pip install -r training/requirements.txt    # the token readout: the versions it was trained and checked with
```

`training/requirements.txt` holds the versions the token readout was trained and checked with (Python 3.12, torch
2.14.0, transformers 5.17.0); the package itself admits transformers 5.15 or newer, on which the CPU tests pass too.
If Hugging Face asks you to accept the Gemma 4 terms, accept them and log in (`hf auth login`) before downloading the
bases. The Qwen3.5 torso of `training/configs/pointer-2b.yaml` also needs `flash-linear-attention` (`pip install -e
".[cuda]"`, Linux only), or its Gated DeltaNet layers fall back to a slower PyTorch path.

## The data

The trainer reads JSONL training rows, one question about one state a line:

```
{"id", "suite", "family", "state", "question": {"type", "criteria", "instructions"}, "target", "label",
 "source": {"dataset", "lane", "input_tokens", "fixed_order"?, "score"?, "instruction_variants"?}}
```

- `question.type` is `choice` or `noul`. A choice's `criteria` maps each option's name to its description or
  `null`; a noul's options are `false` and `true`.
- `target` is a list of probabilities in criteria order, P(true) for a noul, or the name of the right option.
  `label` is the one right answer the dev metrics check.
- `family` groups rows that are one item: the trainer refuses a training row that shares an id, or a (dataset,
  family), with a dev row.
- `source.input_tokens` is the row's prompt length under the run's prompt and tokenizer. The trainer batches and
  filters by it, so the rows must come from `training/convert.py`, which measures it and writes a manifest beside the
  rows that the trainer checks (the prompt, the tokenizer's sha256).
- `source.fixed_order` marks a choice whose options are never shuffled. `source.score` marks v1's score questions,
  stored as fixed-order choices named by their level index, which the pointer head renders as v1 did.
  `source.instruction_variants` lists other phrasings, which the pointer head samples from.

`training/convert.py` writes these rows from either of two inputs. Decision rows, one JSON object a line,

```
{"id", "lane", "source", "group", "type", "state", "instructions", "options": [{"key", "text"}], "target": [p, ...],
 "meta"}
```

convert for the token readout: `--train-files <files>` and `--dev-file <file>`, or `--mix <dir>` for a directory of
`train/part-*.jsonl.gz` and `dev.jsonl.gz`. A score question becomes a fixed-order choice of its levels. v1's examples
(`training/data/format.py`, what `python -m training.data` and v1's builders write) convert for either head with
`--examples`; the converter's docstring gives every mapping, and [`steps.md`](steps.md) builds v1's corpora.

This repository does not ship the training data of the v2 models, and where it will be published is not decided.
To try the token readout without it, convert decision rows of your own; to try the pointer head, build v1's corpora.

## The token readout

### Start to finish, on one machine

For the E2B; every size has its config, and the commands are the same.

1. The base, at the config's revision:

   ```bash
   hf download google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 --local-dir work/e2b/base
   ```

2. The data. Convert decision rows for the base's tokenizer. One conversion serves every size, since the five bases
   share one tokenizer.

   ```bash
   python -m training.convert --train-files <train rows ...> --dev-file <dev rows> --base work/e2b/base \
     --out work/e2b/data
   ```

3. Training, once per seed of the config, each run in a directory of its own:

   ```bash
   for seed in $(python -c 'import yaml; print(*yaml.safe_load(open("training/configs/e2b.yaml"))["seeds"])'); do
     torchrun --standalone --nproc_per_node 8 -m training.train --config training/configs/e2b.yaml \
       --base work/e2b/base --train work/e2b/data/train.jsonl --dev work/e2b/data/dev.jsonl \
       --output work/e2b/runs/seed-$seed --seed $seed --resume
   done
   ```

   On one GPU, `python -m training.train` takes the same flags. The trainer scores the dev fold every `eval_every`
   steps (`eval_log.jsonl`) and saves every `save_every`; `--resume` continues a stopped run on the GPU count it
   started with. When a run ends, the trainer merges its last checkpoint into the base on the CPU, as
   `<run>/merged/step-NNNNN`: the base's checkpoint with the trained decoder, plus `readout.safetensors` and
   `training.json`. `python -m training.merge --base <base> --adapter <run>/step-NNNNN/adapter --out <dir>` merges
   any other step.

4. Average the seeds' merged checkpoints. The average is the model:

   ```bash
   python -m training.average --out work/e2b/model work/e2b/runs/seed-*/merged/step-*[0-9]
   ```

5. Export it as the spec's checkpoint, with its `decision_config.json`, NOTICE and model card:

   ```bash
   python -m training.export --checkpoint work/e2b/model --out work/e2b/export/strands-decider-e2b \
     --release strands-decider-e2b --source https://github.com/strands-labs/strands-decider
   ```

6. Evaluate the export with a decision runtime, as [`evaluation.md`](evaluation.md) describes. The runtimes that load
   the spec's checkpoint are open pull requests of this repository (#130, #136, #137, #138); until one is merged,
   serve the export from its branch.

### The model

- Prompt. Each training row is one question about one state, rendered by `strands_decider.core.render` with the
  `strands-chat` template: `<bos><|turn>user\n` + a `<state>` block and a `<question>` block + `<turn|>\n<|turn>model\n`
  (spec section 3). The options are printed one per line, each after its code: the core writes `f"{code}."`, a space
  and the option's name, then `strands_decider.core.prompt.DESCRIPTION_SEPARATOR` and the description when the option
  has one. That separator is `" \u2014 "`, a space, U+2014 EM DASH and a space. The k options of a question take the
  first k of the 255 codes (A to IV without GZ). `--prompt strands-plain` trains on the spec's other template.
- Attention. Under `prefix_bidir` (spec section 6) a state token sees every state token, before or after it, but no
  question token; a question token sees the whole prompt. Every layer uses this mask, sliding-window layers
  included. Since the state never sees the question, its hidden states are shared by every question on it, and a
  server computes them once.
- Readout. The final hidden state at the prompt's last token goes through a [255, hidden] readout, the soft-cap
  `tanh(z / 30) * 30`, and a mask that gives the codes past the question's options -1e9. The untrained readout is the
  base's tied LM-head rows of the 255 codes. It is trained with the adapters, held as a float32 master and computed
  in bf16.
- Adapters. LoRA of rank 64, alpha 128, dropout 0.05 on every linear of the decoder layers (`q, k, v, o, gate, up,
  down`, plus `per_layer_input_gate` and `per_layer_projection` on E2B and E4B), float32 masters on the frozen bf16
  base, B = 0 at the start. The 26B-A4B also adapts every routed expert's fused `gate_up_proj` and `down_proj` with a
  LoRA of rank 8 (64 over its top-8 routing), the router frozen ([`expert_lora.py`](expert_lora.py)).
- The base. The decoder is read straight from the Hugging Face checkpoint, `google/gemma-4-*-it` at the config's
  revision, as transformers' own text model for it (`Gemma4TextModel`; `Gemma4UnifiedTextModel` for the 12B). The
  vision and audio encoders are not loaded; the export carries them unchanged.

### A training step

- Rows per step. The plan ([`batching.py`](batching.py)) shuffles the rows with the seed, sorts each window of one
  step's rows by length, cuts them into microbatches of at most `batch_size` rows and `token_budget` padded tokens,
  and shuffles the epoch's microbatches, so a step holds microbatches of similar-length rows from all over the data.
  A step is `rows_per_step / batch_size` microbatches (3,072 rows at most), `accum` of them on each GPU. The plan
  depends only on the rows' lengths, the seed, the epochs and these settings, so it is the same on every GPU count
  that divides a step's microbatches.
- Loss. Cross-entropy between the model's distribution over the question's options and the row's target (a
  distribution, P(true) for a yes/no question, or one option), averaged over the step's rows ([`loss.py`](loss.py)).
- Augmentation. Each choice row's options are shuffled, the target moving with its option, by a generator seeded
  from the seed, the step and the microbatch, so a row's order does not depend on the GPU it runs on.
- Optimizer. AdamW, weight decay 0, gradients clipped to norm 1.0. The learning rate rises linearly over the first 5%
  of the steps, then follows a cosine from the peak down to `lr_floor` of it, 0.1 ([`schedule.py`](schedule.py)).
- Throughput. Gradient checkpointing is on. `--attn-split N` computes the mask as two unmasked attentions for
  microbatches at least N tokens wide; it changes no step's rows, and equals the masked attention up to bf16 rounding.
- Dev fold. Every `eval_every` steps the trainer reports accuracy, NLL, Brier and ECE, and with `--temperature-data`
  a fitted temperature and the dev metrics at it ([`metrics.py`](metrics.py), [`dev_eval.py`](dev_eval.py)).

### The per-size settings

`training/configs/<size>.yaml` holds each size's settings; any `training.train` flag overrides them. A step's microbatches
set the GPU counts a run can use on one machine, and the peak memory is what one run of the settings measured:

**Table 1.** The token readout's settings per size, and what they allow on one machine.

| Size | LR | Microbatch (rows / padded tokens) | Microbatches a step | GPU counts | Peak GB per GPU |
|---|---:|---|---:|---|---:|
| E2B | 1e-4 | 32 / 32,768 | 96 | 1, 2, 3, 4, 6, 8 | 24.8 |
| E4B | 1e-4 | 32 / 32,768 | 96 | 1, 2, 3, 4, 6, 8 | 36.8 |
| 12B | 3e-5 | 16 / 16,384 | 192 | 1, 2, 3, 4, 6, 8 | 42.8 |
| 26B-A4B (expert LoRA) | 1e-4 | 8 / 8,192 | 384 | 1, 2, 3, 4, 6, 8 | 69.5 |
| 31B | 1e-4 | 8 / 8,192 | 384 | 1, 2, 3, 4, 6, 8 | 88.9 |

Every size takes at most 3,072 rows a step, and every GPU count up to 8 but 5 and 7 divides its microbatches. The
12B collapses at a learning rate of 1e-4, hence 3e-5. The 31B's peak is more than an 80 GB GPU holds, and this
trainer does not shard the model, so the 31B needs GPUs with more memory. Each size averages the runs of the
config's `seeds`: three, or two for the 12B.

### Checkpoints

**Table 2.** What the token-readout path writes.

| What | Where | Holds |
|---|---|---|
| A run | `--output` | `step-NNNNN/adapter/{adapter.safetensors, adapter_config.json}` every `save_every` steps and at the end, `trainer_state.pt` (the last two), `last`, `run_config.json`, `train_log.jsonl`, `eval_log.jsonl` |
| A merged checkpoint | `<run>/merged/step-NNNNN` or `training.merge --out` | the base's checkpoint with the merged decoder (bf16, the base's names and shards), `readout.safetensors`, `training.json` |
| The model | `training.average --out` | the same layout, every tensor averaged; `training.json` names the checkpoints and seeds averaged |
| The export | `training.export --out` | the spec's checkpoint (section 9): no code, no chat template, no training record |

`adapter_config.json` and `training.json` record the arguments, the data's and the code's sha256 and the step plan's.
Each config pins its base by `base_sha256` (over the base's `config.json`, `tokenizer.json` and tensor headers) and
its tokenizer by `tokenizer_sha256`, so the trainer refuses another size or revision; to train on another revision,
set `revision`, `base_sha256` and, if the tokenizer changed, `tokenizer_sha256` in a copy of the config. The merge is
float32 on the CPU, `W' = bf16(W + (B @ A) * alpha / rank)`, one base shard at a time, and every tensor it does not
merge is the base's, byte for byte. The average adds each tensor in float32 and stores it in bf16. The export keeps
the base's `config.json` as it is, writes `decision_config.json` (spec section 8: `inputs` are the ones the run
trained on, `[text]`; `max_length_validated` is the training max length unless `--max-length-validated` says
otherwise), the NOTICE, the Apache-2.0 LICENSE and a model card, and checks the result with the core's
`DecisionConfig.load` and `Readout.load`. It does not make the spec's golden vectors (`--goldens` copies a file made
by a reference run).

### How the token-readout path was checked

The trainer that trained the v2 models ran beside this one on the same inputs.

- Each reimplemented piece, on the CPU: the row lengths, the microbatches, the plan and its placement, the target
  matrix, the option shuffling, the schedule, the loss, the dev metrics and the temperature calibration give the
  same outputs bit for bit. [`tests/fixtures/previous_trainer.json`](tests/fixtures/README.md) keeps that trainer's
  outputs. The tests compare bit for bit when `STRANDS_DECIDER_EXACT_FIXTURES=1`, which needs the versions of
  `training/requirements.txt` and which a CI job sets; elsewhere they allow the last bit's rounding of a float sum
  (`training/tests/trainer_support.py` says why).
- The whole path, outside the suite, on the CPU, on a small random Gemma 4 decoder with the real tokenizer, in one
  process: the same converted rows, step plan, loss and gradient norm at every step, dev metrics, adapters, merged
  checkpoints, the average of two seeds and every exported file, bit for bit. The same holds with the split attention, with expert
  LoRA on a mixture-of-experts decoder, and for a stopped and resumed run. That trainer could also place a step's
  microbatches on the GPUs by their estimated cost, which reorders them even on one GPU and with them the dropout
  draws. This one cannot, so it equals that trainer's runs without that placement, not the runs its configs made.
- The core: [`tests/fixtures/render.json`](tests/fixtures/README.md) records this repository's core's renders. They
  equal the renders of the core the v2 models were trained with on every row but those whose state holds NaN or
  Infinity, which the newer core refuses, and a lone surrogate, which both refuse with different errors.

The next two checks were measured on GPUs, outside the test suite:

- On one 8-GPU machine, on the E2B, with PyTorch's deterministic algorithms and SDPA's math attention kernel, the two
  trainers gave the same loss at each of 12 steps, the same dev metrics and the same adapter, bit for bit. With the
  default kernels they differ by less than that trainer differs from itself (0.054 in loss over 12 steps), since the
  default CUDA kernels are nondeterministic.
- The 12B and 31B bases, read from the Hugging Face checkpoints (the 12B as transformers' `Gemma4UnifiedTextModel`),
  give the decoder tensors, readout, final hidden states and logits that trainer computed, bit for bit, on 40
  prompts.

A known limit: score questions are trained as choices whose options keep their order, while the spec renders a score
question with its own header and its levels named by their index. A served score question therefore reads
differently from anything the model was trained on.

## The pointer head

### Start to finish, on one machine

[`steps.md`](steps.md) builds v1's corpora and trains the pointer head on them, step by step, and
`training/recipe.sh all` runs every step: the corpus build and downloads, the multi-step, generated, adequacy and
catch-all rows, v20's teacher file, the base, the conversion, training, calibration and evaluation. It trains v20's
data and teacher with v1's settings, without the frozen-KL anchor, so it reproduces no released model. Its `parent`
and `replay` stages (v19's route) and `PARENT_CONFIG` and `TRAIN_CONFIG` are gone, and `CKPT` is now the run's
`last/checkpoint`. Serving, calibrating and evaluating a checkpoint load its base from the Hub by the id and revision
it names, not from the local copy training used. In short:

```bash
training/recipe.sh build fetch multistep generated adequacy catchall distill base convert
for seed in 0 1 2; do SEED=$seed NGPU=8 training/recipe.sh train; done
python -m training.soup checkpoints/pointer-2b-seed-{0,1,2}/last/checkpoint --out checkpoints/pointer-2b-soup
python -m training.evaluate calibrate checkpoints/pointer-2b-soup --data data/holdout_v5_norule.jsonl
strands-decider serve checkpoints/pointer-2b-soup --port 8099
```

`training/configs/pointer-2b.yaml` holds v1's settings on Qwen3.5 2B, and `pointer-e2b.yaml` the same on Gemma 4 E2B.
For the E4B, 12B and 26B-A4B, copy `pointer-e2b.yaml`, set its `base_model` and `revision`, and set or delete its
`tokenizer_sha256` and `base_sha256`, which pin the E2B. Use a pointer config, or give every setting: the trainer's
defaults are the token readout's (rank 64, a cosine to 0.1, 3,072 rows a step). The settings are
rank 16 and alpha 32, the head at a learning rate of 1e-3 and the adapter at 1e-4, weight decay 0.01 on the head's
matrices, 3% warmup, a cosine to zero, 32 rows a step in 8 microbatches of 4, and a 4,096-token window. The 8
microbatches divide over 1, 2, 4 or 8 GPUs, which therefore all train the same steps. Each save,
`<run>/step-NNNNN/checkpoint/`, is a v1 checkpoint: `strands_decider_config.json`, which names the base on the Hub at
its revision, `head.safetensors`, `lora/`, the tokenizer, and the record as `training.json`. `last` points at the
newest.

### What it is

[`pointer.py`](pointer.py) holds strands-decider's own `StrandsDeciderModel`
([`src/strands_decider/modeling.py`](../src/strands_decider/modeling.py)) behind the trainer's interface.

- The model is v1's torso loader, a PEFT LoRA adapter and the `PointerHead`, built by the same call v1's trainer made
  right after seeding, so a run starts from the weights v1's trainer started from (`training/tests/test_pointer.py`).
- Each row is rendered with v1's numbered prompt (`strands_decider.prompting.build_prompt`) and tokenized as v1's
  `SystemOneCollator` tokenizes it, so the ids, the masks and each option's last-token position are the collator's.
  `training/tests/test_pointer.py` checks this, and that the logits are v1's for the same weights. A prompt over `max_length`
  is refused, never cut.
- Per row, the augmentation draws what the collator draws, in its order: a choice's or a noul's options shuffled, a
  score's levels reversed with probability `reverse_score_prob`, and one phrasing among the instructions and their
  variants.
- The targets are the row's, in the order its options were rendered. v1's teacher files and ordinal smoothing come
  in through the conversion (below); the checkpoint records the smoothing, which serving reads.
- The optimizer has v1's groups: the head's matrices at `head_lr` with `weight_decay`, its norm gains and biases at
  `head_lr` without, the adapter at `lr` without, and AdamW betas 0.9 and 0.95.

`python -m training.convert --examples` writes v1's examples as the trainer's rows. Their teacher files mix into the
targets: a row that a `--teacher W <files>` group covers gets (label + W * teacher) / (1 + W), several groups summed
the same way, and the dev rows keep their label's target. Per row, the mixture's loss has the gradient direction of
v1's label loss plus W times the teacher's KL, at 1 / (1 + W) of its size, and every row weighs alike: v1 gave rows
with a teacher more weight, since it averaged their KL over the micro-batch's teacher rows only. Score targets are
smoothed over the neighbouring levels by `--ordinal-smoothing` (0.1, v1's), and the dev fold is v1's split
(`--val-fraction` of each task, chosen with `--val-split-seed`) or `--val-files`.

### What differs from v1's trainer

v1's trainer and this one were run on the CPU, outside the test suite, on the same tiny Qwen3 torso, the same 480
examples and teacher file, and v1's settings at 8 rows a step. Both took 108 steps over two epochs and
held out the same 48 dev rows. From a tiny random model on random labels, both ended near chance on them: accuracy
0.31 for v1's and 0.35 for this one, NLL 1.286 and 1.285. No run of the pointer head on a real torso and v1's corpora
has been made yet; `training/tests/test_pointer.py` checks that it learns a learnable toy task. They are not the same
training, by design.

**Table 3.** Where this trainer's pointer head trains differently from v1's trainer.

| | v1's trainer | This trainer |
|---|---|---|
| Steps | `micro_batch_size` x `grad_accum` rows; micro-batches cut from megabatches of 400 rows sorted by prompt characters | at most `rows_per_step` rows, in microbatches capped by rows and by padded tokens, cut from each step's rows sorted by measured tokens |
| Loss | the label's cross-entropy as a weighted mean over each micro-batch, plus `teacher_weight` times KL(teacher, student) as a mean over the micro-batch's rows with a teacher | cross-entropy against each row's target, the label mixed with the teacher, as a mean over the step's rows |
| Row weights | a weighted mean | every row weighs 1; the converter refuses other weights |
| Schedule | `LambdaLR`: the first step at a learning rate of 0, a cosine to 0 | the first step at 1/warmup of the peak, a cosine to `lr_floor` (0 in the pointer configs) |
| Augmentation | one random stream for the run | a stream per step and microbatch, seeded, so a resumed run and every GPU count draw the same orders |
| Dev fold | each seed held out its own `val_fraction` unless `val_split_seed` was set | fixed by the conversion, so every seed trains on the same rows |
| Long rows | truncated at `max_length` | dropped when the prompt plus 16 tokens exceeds `max_length` |
| Several GPUs | any count from 2 to 32, micro-batches split across ranks | a count that divides a step's microbatches, whole microbatches per GPU |
| Dev evaluation | at most 50 micro-batches of the validation rows | the whole dev fold, on the first GPU while the others wait |
| Options | any number | at most 255, the codes' count |
| Gradient checkpointing | a setting (`gradient_checkpointing: false` was 2.2x faster on 80 GB GPUs) | always on |

Not in this trainer, and open questions on the change that made it the one trainer: the frozen-KL anchor
(`kl_frozen_weight`, `kl_frozen_reference`, `precompute_frozen_kl`, `kl_direction`, `kl_only_files`), EMA,
`init_from`, `freeze_torso` and `host_embeddings`. An approximation of the anchor is the reference checkpoint's own
distributions as one more teacher group ([`steps.md`](steps.md), "3. Convert and train"); the KL terms' reverse
direction has no equivalent. The top-level `configs/` keeps the configs v1's trainer read, as the records of the
runs it made (v19, v20 and the v1 releases). This trainer does not read them, and the pointer configs restate their
settings.

## Several GPUs on one machine

```bash
torchrun --standalone --nproc_per_node <N> -m training.train --config <config> ...
```

Each process is one rank on one GPU ([`distributed.py`](distributed.py)). Microbatch j of a step runs on rank j % N,
the gradients of the adapters and the readout are averaged over the ranks (the frozen decoder is never
synchronized), and the loss is weighted so the average is the per-row mean over the whole step, as on one GPU. N must
divide a step's microbatches: 1, 2, 3, 4, 6 or 8 for the token readout's configs, 1, 2, 4 or 8 for the pointer
head's. Dropout is drawn per rank, so the weights equal a one-GPU run's only with dropout off, and then up to the
rounding of sums taken in another order. `pytest -m distributed` checks it: `training/tests/test_ddp.py` for the
pointer head (losses and weights within 1e-4 relative, gradient norms within 1e-3, the key bias aside, whose gradient
is zero up to rounding, so Adam turns rounding into steps that change no logit) and `test_train_readout.py` for the
token readout (losses within 1e-5, gradient norms and weights within 1e-4). The rendezvous listens on the machine's
network interfaces without authentication: train on machines whose ports only you can reach. A world that spans
several machines is refused.

## Left for later

- Training on several machines. The trainer uses the GPUs of one machine, and does not shard the model across GPUs,
  which the 31B needs on 80 GB GPUs.
- Training at the v2 models' scale. One run of a size's settings over about 6,400 steps of 3,072 rows took about
  1,100 H100-hours for the E2B, 1,400 for the E4B, 1,500 for the 12B and 1,700 for the 26B-A4B, and about 1,700
  B200-hours for the 31B. On one 8-GPU machine an E2B run takes about six days. A guide to training at that scale
  comes later.
- Serving an export from this repository's `main`. The decision runtimes are open pull requests (#130, #136, #137,
  #138).
- The data. Where the v2 models' training data will be published is not decided. Until then the token readout trains
  on decision rows you bring, and the pointer head on v1's corpora, which [`steps.md`](steps.md) builds.

## Tests

```bash
python -m pytest -q training/tests                                       # CPU, no network
STRANDS_DECIDER_GEMMA4_TOKENIZER=<Gemma 4's tokenizer.json, sha256 cc8d3a0c...> python -m pytest -q training/tests
python -m pytest -q -m distributed training/tests                        # two ranks against one, on the CPU
python -m training.tests.gpu.check_model work/e2b/base                   # one GPU: mask, boundary, padding
```
