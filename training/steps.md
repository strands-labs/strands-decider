# v1's data and the pointer head, step by step

The steps of `training/recipe.sh`: v1's corpora, built from public datasets and the committed synthetic files, and
the pointer head trained on them by `training/train.py`, with the command and what each writes. [README.md](README.md)
says how the trainer works and what differs from v1's trainer. Paths are relative to the repository root unless they
are links.

## 1. Build the corpora

Recipes convert public classification datasets into `(state, question, answer)` triples.
The mix is deliberately varied — topic, intent, sentiment, entailment, ordinal ratings,
yes/no, rule chaining — because a corpus drawn from one task teaches memorisation, not
binding.

```bash
python -m training.data build --out data/train_v5.jsonl --max-options 24 \
  -r ag_news -r banking77 -r clinc150 -r dbpedia -r lang_id -r yahoo_topics \
  -r spam -r toxicity -r mnli_entail -r boolq \
  -r paws -r vitaminc -r wnli -r pubmed_qa \
  -r yelp_stars -r sst5_sentiment -r app_reviews -r formality \
  -r ruletaker_d0 -r ruletaker_d1 -r ruletaker_d2 \
  -r emotion -r massive_intent -r sarcasm -r hate_severity \
  -r ruletaker_d3 -r ruletaker_d5 -r ruletaker_natlang \
  --holdout emotion --holdout massive_intent --holdout sarcasm --holdout hate_severity \
  --holdout ruletaker_d3 --holdout ruletaker_d5 --holdout ruletaker_natlang
```

The build writes `data/train_v5.jsonl` (100,449 rows, 21 tasks) and
`data/train_v5.holdout.jsonl` in about two minutes. It is deterministic: a fresh build
gives the same files on a Mac and on AWS (`data/train_v5.jsonl` sha256 `3d17d148…`, full hashes in
[Reproduction contract](../data/README.md#reproduction-contract)). The committed teacher targets align with this build row by row. Whether it is byte-identical
to the original v5 file that v7 trained on is not verified.

`--holdout` routes whole tasks to the `.holdout.jsonl` file. Held-out *tasks* — not
just held-out rows — are the honest test of whether the readout learned to bind to
prompt-described options, since in-distribution accuracy can be reached by memorising
label sets. Of the seven held out, the four with unseen label sets and rubrics are the
generalisation test; the three RuleTaker depths share a generator with the trained ones,
so `recipe.sh build` drops them into `data/holdout_v5_norule.jsonl` for calibration and
evaluation.

Datasets with large label sets (banking77's 77 intents) are subsampled per example to
`--max-options`, always keeping the gold option. Varying N across examples is itself
useful signal: it stops the readout from assuming a fixed number of live options.

```bash
python -m training.data recipes                     # what is available
python -m training.data stats data/train_v5.jsonl   # composition
python -m training.data peek  data/train_v5.jsonl   # exactly what the model will see
```

The multi-step rows come from two downloaded releases and two HuggingFace datasets:

```bash
training/recipe.sh fetch                  # ContractNLI, MuSiQue and HelpSteer2 into data/raw/ (~350 MB)
python -m training.data.multistep    # data/multistep_v14.jsonl and _eval.jsonl
```

That writes 12,909 training rows and 4,084 evaluation rows from splits never trained
on, including 959 from HotpotQA, a source held out of training entirely. ContractNLI is
balanced per claim, because its 17 claims are asked of every contract and answering each
with its usual label scores 0.679 without reading; MuSiQue keeps each question's
answerable and unanswerable versions together. Rows longer than the window are
dropped, never truncated. `tests/test_multistep.py` checks both properties.

## 2. Label with the teacher

```bash
python -m training.data.teacher --src data/multistep_v14.jsonl \
  --out data/teacher_multistep_v14.jsonl --shift-by data/train_v5.jsonl
```

A frozen Qwen3.5-4B reads each multi-step row the way SemIf reads it — chat template,
JSON, one forward pass, softmax over option letters — with prompts checked
byte-identical to SemIf's own (`tests/test_teacher.py`, which needs SemIf's code and skips
without it). Rows are written as they are
labelled and a rerun resumes. `--shift-by` also writes the labels indexed into the
concatenated training files, which is what training reads. About an hour under WSL2.

## 3. Convert and train

The trainer reads its own rows (training/README.md, "The data"), so the examples are converted first. The converter
measures each row's prompt under the numbered render with the base's tokenizer, folds the teacher's distributions
into the targets, smooths the score targets as v1's collator did, and holds out the dev fold with v1's split:

```bash
training/recipe.sh base      # Qwen/Qwen3.5-2B-Base at the config's revision, into checkpoints/base-qwen3.5-2b
python -m training.convert --examples data/train_v5.jsonl data/multistep_v14.jsonl data/generated_v16p.jsonl \
  data/generated_v18p.jsonl data/adequacy_hs2.jsonl data/adequacy_gen.jsonl data/catchall_v20.jsonl \
  data/flips_v20.jsonl --teacher 1.0 data/teacher_v20.jsonl --base checkpoints/base-qwen3.5-2b \
  --prompt numbered --max-length 4096 --out data/rows-v20
python -m training.train --config training/configs/pointer-2b.yaml --base checkpoints/base-qwen3.5-2b \
  --train data/rows-v20/train.jsonl --dev data/rows-v20/dev.jsonl --output checkpoints/pointer-2b-seed-0 --seed 0
```

`training/recipe.sh convert train` runs the same. The examples go in the order of the teacher file's row indices, as
v1's `train_files` did. Training writes a v1 checkpoint at every save, `checkpoints/pointer-2b-seed-0/step-NNNNN/
checkpoint/`, and `last` points at the newest. On eight GPUs of one machine, `NGPU=8 training/recipe.sh train` runs
the same command under `torchrun --standalone`; the config's 8 microbatches a step divide over 1, 2, 4 or 8 GPUs.

`training/configs/pointer-e2b.yaml` is the same recipe on Gemma 4 E2B (with `force_bos`, and `--force-bos` at the
conversion), and the E4B, 12B and 26B-A4B take it with their own `--base-model` and `--revision`. v1's frozen-KL anchor
toward a reference checkpoint is no term of this trainer: an approximation is the reference's own distributions as
one more teacher group, from `python -m training.data.replay <reference> --src <file> --out <file>` per training file
(with `--shift-by` the files before it), given as `--teacher 0.3 <files>` (training/README.md, "Open questions").

### Seeds and soups

Two seeds of one recipe differ by about 3 tasks on JevBench (seed SD 3.1 over 18 seeds of
v26's c0001 recipe). That is the spread you get when each of the ~60 items near the decision
boundary falls either way at its own rate, so the top seed by JevBench is a lucky draw, not a
better model. A release is therefore made from 3 seeds, not by picking one. Train the recipe
three times, with `SEED=0`, `1` and `2` (`training/recipe.sh train` writes
`checkpoints/pointer-2b-seed-N`), then soup and calibrate them:

```bash
python -m training.soup checkpoints/pointer-2b-seed-{0,1,2}/last/checkpoint --out checkpoints/pointer-2b-soup
python -m training.evaluate calibrate checkpoints/pointer-2b-soup --data data/holdout_v5_norule.jsonl
```

`soup` writes one checkpoint whose weights average the seeds' by what they compute, not
tensor by tensor:

- **LoRA**: for each adapted layer, the soup's update ΔW = s · B @ A (s = lora_alpha / r)
  is exactly the mean of the seeds' updates. Averaging A and B separately would mean nothing,
  since each seed's adapter is defined only up to an invertible r × r mixing. The soup stores
  the mean as a rank-N·r adapter: the A matrices stacked, the B matrices stacked and divided
  by N, and lora_alpha scaled so that s is unchanged. Three rank-16 seeds give a rank-48
  adapter, which the loader, `serve` and the Hugging Face export read as any adapter.
- **Head**: the soup's head logit is exactly the mean of the seeds' head logits. A pointer
  head's logit is a dot product q · k, unchanged by any rotation of the q/k space, so heads
  initialised independently have unrelated bases and averaging their tensors gives a
  different, wrong head. Each head's LayerNorm is folded into its q and k projections and
  the heads are stacked (pointer_dim N · d). A linear slot head is averaged directly; an
  MLP slot head stacks its hidden units.
- **Temperatures** are reset to 1: the seeds' fitted temperatures do not describe the soup,
  so calibrate it before you evaluate or serve it.

The checkpoints must share one recipe: their configs may differ only in the fitted
temperatures, they must resolve to one base revision (the config's `base_revision`, else
their `provenance.json`'s, as the loader resolves it), and their adapters must be plain LoRA
(no DoRA, rank or alpha patterns, `modules_to_save` or `target_parameters`) over the same
modules. Otherwise `soup` refuses them. The soup's config pins that base revision. `--out`
must not overlap an input and must be absent or empty; `--replace` deletes what it holds
first. It writes `soup.json` with the input paths; the Hugging Face export leaves that file
out.

Every seed trains on the same rows: the conversion fixes the dev fold once, where v1's trainer held out each seed's
own `val_fraction` unless `val_split_seed` was set. Initialisation, data order, option order and dropout still
follow `--seed`.

## 4. Calibrate

```bash
python -m training.evaluate calibrate checkpoints/pointer-2b-seed-0/last/checkpoint --data data/holdout_v5_norule.jsonl
```

Fits a temperature **per primitive**, minimising **ECE**, and writes them into the
checkpoint. **Run this before serving.**

It cannot change a `choice` argmax, so choice accuracy is untouched. It *does* change
the value returned for `noul` and `score`, since those are read off the distribution
itself rather than its argmax — softening a score pulls it toward the midpoint. Skip
calibration entirely and the thresholds of the
[routing convention](../docs/architecture.md#the-routing-convention) will not hold, which is the failure
mode that matters: a model that is 70% accurate but reports 0.99 confidence fails
silently in exactly the place you were relying on it.

Fitted against ECE rather than NLL on purpose. The two disagree: fitting `noul` to NLL
here chose T=2.63, which left it under-confident by 0.29 — worse calibrated than no
scaling at all, even though likelihood improved.

`calibrate` uses only the `calib` half of the file and `eval` defaults to the `test`
half — a deterministic, disjoint partition of the same file, so the reported
calibration error is one the model would actually achieve on rows the temperature was
never fitted on. Fitting and scoring on the same rows reports a flatteringly low ECE
that evaporates in production. Pass `--split all` to either command to override.

## 5. Evaluate

`training/recipe.sh eval` runs the held-out classification and multi-step evaluations. The commands
and what they measure are in [Internal evaluations](../evaluation/README.md#internal-evaluations).

JevBench, the public benchmark the recipe is ranked on, runs against the served checkpoint
([Reproducing, and two caveats](../evaluation/jevbench.md#reproducing-and-two-caveats)). On a machine with one GPU, the
third argument a run in `research/data/jevbench_results.csv` (for example `v19`) for the paired test:

```bash
GPU=0 evaluation/jevbench/jevbench.sh checkpoints/pointer-2b-soup <out dir> v19
python evaluation/jevbench/paired.py --a research/data/jevbench_results.csv --a-run v19 --b <out dir>/results.jsonl --list
```

The script serves the checkpoint on 127.0.0.1 and stops by itself when the port already answers (the benchmark would
score an old server) or `/health` names another checkpoint or window. Its `run_meta.json` should say `n_attempted`
231, `n_failed` 0 and `schema_validity_strict` 1.0.

## 6. Serve and export

`strands-decider serve` answers requests over HTTP. See [Serve](../docs/inference.md#serve).

`training.hf_export` writes a checkpoint as a Hugging Face model-repo folder: no pickles, a model card with
provenance, and a `MANIFEST.sha256` written last. Exporting the same inputs twice gives the same bytes, the export
never changes weights, and nothing in it talks to the Hub:

```bash
python -m training.hf_export export checkpoints/pointer-2b-soup <export dir> --jevbench <out dir> \
  --repo-url https://github.com/strands-labs/strands-decider
python -m training.hf_export verify <export dir>
hf upload <org>/<repo> <export dir> . --commit-message "<what changed and why>"
```
