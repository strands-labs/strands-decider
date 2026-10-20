# Agents that operate training

Runbooks for a coding agent that runs, operates or verifies this repository's training on
its user's behalf. The human documents say how the recipe works; this file says what an
agent must do and must not do with the scripts. It describes only what the code and those
documents show. Decisions that cost money (a host, a reservation) belong to the user.
Paths are relative to the repository root. Replace every `<...>` placeholder.

| runbook | when |
| --- | --- |
| [0. Ground rules](#0-ground-rules) | always |
| [1. Get a GPU host](#1-get-a-gpu-host) | before any GPU work |
| [2. Set up and verify a host](#2-set-up-and-verify-a-host) | after every launch, and after every code change |
| [3. Run the recipe](#3-run-the-recipe) | a retrain, or one stage of it |
| [4. Verify a code change before you trust a retrain](#4-verify-a-code-change-before-you-trust-a-retrain) | the code changed since the published checkpoint |
| [5. Evaluate on JevBench](#5-evaluate-on-jevbench) | a checkpoint to score |
| [6. Export and publish a checkpoint](#6-export-and-publish-a-checkpoint) | a retrain to keep |
| [7. Tear down](#7-tear-down) | the end of every session |

Human documents: [training/README.md](README.md) (the recipe, the reproduction bar),
[training/aws/README.md](aws/README.md) (the scripts, every variable, timings, costs),
[training/steps.md](steps.md) (the six steps), [evaluation/jevbench.md](../evaluation/jevbench.md)
(the benchmark), [data/README.md](../data/README.md) (what is committed, what is downloaded).

## 0. Ground rules

- **Spend is the user's decision.** Launching a host, buying a Capacity Block and creating
  a reservation all cost money. Ask your user before each one, with the shape, region and
  duration you intend to use. Do not retry a launch or buy anything without that answer.
- **No secrets on a host, in an SSM command or in a log.** Never `echo`, `cat` or paste a
  token or key. `huggingface_hub` reads its token by itself on your machine. The host
  needs no Hugging Face token (the models are public) and no AWS key (its IAM role).
- **One `HOBSON_NAME_TAG` and one `HOBSON_STATE_DIR` per agent.** The scripts find hosts by
  the `Name` tag and `teardown.sh` terminates every host with it. Two agents on one tag
  find and terminate each other's hosts. Set both before the first script call:

  ```bash
  # training/aws/scripts/local.env is git-ignored and sourced first. Every line needs `export`:
  # common.sh does not export what it sources, so a plain AWS_PROFILE=... never reaches the CLI.
  export AWS_PROFILE=<profile>
  export HOBSON_BUCKET=<bucket>                         # shared by every agent of the account
  export HOBSON_NAME_TAG=hobson-<agent>                 # unique per agent
  export HOBSON_JOB_TAG=<job>
  export HOBSON_STATE_DIR=training/aws/scripts/.state/<agent>
  export HOBSON_REGIONS="<every region you may launch in>"
  ```

- **Tear down every host you launch and confirm `terminated`.** Cancel every reservation
  you create. Report instance ids, launch and termination times, and reservation ids to
  your user.
- **Long jobs run in `hobson-bg`, never in a held-open SSM call.** `ssm-run.sh` has a
  one-hour default timeout, and a finished call is the only way to get its output.
- **Poll, do not spin.** One `ssm-run.sh` status call every few minutes; the AWS API
  throttles tight loops.
- **Ship commits, not working trees.** `sync-code.sh` archives one commit. Uncommitted
  changes never reach the host, and the host records the commit as `git_rev`.
- **Record every deviation** from a documented command: what, when, why, what you changed.
- **Never modify the benchmark**, and never change a script to make a check pass.

## 1. Get a GPU host

**When.** Before any stage, evaluation or gate that needs a GPU. How many GPUs each route
needs, and how long it takes, is in [training/README.md](README.md) and
[Measured stage timings](aws/README.md#measured-stage-timings). Read
[What you need](aws/README.md#what-you-need) once: credentials, default VPC, vCPU quota.
Ask your user before you launch.

**Commands.** `launch-host.sh` launches one On-Demand host ([2. A host](aws/README.md#2-a-host)).
It tries every shape in `SHAPES` in every AZ of every region in `REGIONS`, moves on at
each capacity error, and stops at any other error. It is idempotent: if a host with your
tag exists, it prints that host and exits.

```bash
training/aws/scripts/launch-host.sh                                  # defaults: see the variable table in aws/README.md
SHAPES="<shape ...>" REGIONS="<region ...>" ROOT_GB=<GiB> training/aws/scripts/launch-host.sh
```

Every region in `REGIONS` must be in `HOBSON_REGIONS` too, or `teardown.sh` does not find
the host. `ROOT_GB` is the size of the gp3 root volume (default 1024 GiB); the host's
data, checkpoints and model cache live on the NVMe scratch, not on it. Every attempt is
logged in `$HOBSON_STATE_DIR/launch-attempts.log`.

`launch-capacity-block.sh` launches one host into an EC2 Capacity Block that already
exists. It waits until the reservation is `active`, launches into it, and sets the safety
poweroff 30 min before the block ends (`SHUTDOWN_AT` overrides). It does not buy the
block: that is `aws ec2 purchase-capacity-block`, a prepaid purchase with no refund, and
the user's decision.

```bash
CR_ID=cr-<id> REGION=<region> training/aws/scripts/launch-capacity-block.sh
```

The scripts have no other reservation path.

**Checks.** The launcher prints `LAUNCHED i-... <shape> <region>/<az>` and `SSM Online`,
and writes `$HOBSON_STATE_DIR/host.env`. `training/aws/scripts/ssm-run.sh nvidia-smi`
lists the GPUs you asked for.

**Stop.** A non-capacity error from `run-instances` (quota, IAM) stops the launcher: fix
the cause, do not retry. The safety timer powers the host off after `SHUTDOWN_MIN`
(default 720 min), and poweroff terminates the host. Compare that with the timings of
the stages you plan; launch with a larger `SHUTDOWN_MIN`, or run `hobson-extend <minutes>`
on the host before the timer fires. Anything not copied to S3 is lost with the host.

## 2. Set up and verify a host

**When.** After every launch. After every code change, run `sync-code.sh` again, then
`verify-host.sh`. Details: [3. Code and environment](aws/README.md#3-code-and-environment).

**Commands.** Pick a code name that is yours. `<name>` is one S3 key per bucket
(`code/<name>.tgz`): a second agent that syncs under the same name replaces your archive,
and `hobson-pull` then extracts their commit over your code on the host.

```bash
training/aws/scripts/sync-code.sh . hobson-<agent> --upload-only     # one commit, HEAD by default; --rev <commit> for another
training/aws/scripts/ssm-run.sh -t 1800 -f training/aws/image/setup-host.sh hobson-<agent>
training/aws/scripts/ssm-run.sh -t 1800 -f training/aws/image/verify-host.sh hobson-<agent>
```

The same name goes to all three. The code lands in `/opt/hobson/code/hobson-<agent>`, with
`data/` and `checkpoints/` as links to the NVMe scratch. `setup-host.sh` needs the training
code in the archive: it reads the teacher revision from `training/data/teacher.py`.

**Checks.** `sync-code.sh` prints the file count and the full commit SHA; compare the SHA to
`git rev-parse <rev>`. `setup-host.sh` ends with `DONE setup`. `verify-host.sh` ends with
`VERIFY RESULT: PASS`: the pinned versions, a forward and backward pass on the fla
kernels with no Gated DeltaNet fallback, and the CPU test suite.

**Stop.** `VERIFY RESULT: FAIL`, or `sync-code.sh` warns about uncommitted changes to
tracked files (commit them first; they are not shipped).

## 3. Run the recipe

**When.** A retrain, or named stages of it. Read [4. The recipe](aws/README.md#4-the-recipe)
for the runner and [Measured stage timings](aws/README.md#measured-stage-timings) for how
long each stage takes.

**Commands.** Inside `hobson-bg`, from the code directory, with `PY` and `S3_PREFIX` set
(`S3_PREFIX=none` disables the S3 copy; otherwise use a prefix of your own):

```bash
training/aws/scripts/ssm-run.sh 'hobson-bg <job> "cd /opt/hobson/code/hobson-<agent> && PY=/opt/hobson/venv/bin/python S3_PREFIX=s3://$HOBSON_BUCKET/results/<run-id> training/run_recipe.sh all"'
training/aws/scripts/ssm-run.sh 'tail -n 5 /opt/hobson/logs/<job>.log; cat /opt/hobson/code/hobson-<agent>/runs/recipe/stages.jsonl'   # poll
training/aws/scripts/ssm-run.sh 'systemctl status hobson-<job> --no-pager; cat /opt/hobson/logs/<job>.rc'                              # state, exit code
```

Stages and groups: `cpu` (`build`, `fetch`, `multistep`, `adequacy`, `generated`, side by
side), `gpu` (`teacher`, `parent`, `replay`, `train`, `calibrate`, `eval`), `all`. `catchall`
and `distill` run only when named. The runner checks every name before the first stage.
`NGPU` defaults to every GPU the host has.

Rules the runner enforces:

- `FAST=1` only with `NGPU=8`; any other `NGPU` exits with code 2.
- A `RUN_DIR` (default `runs/recipe`) holds one run. The config copies it writes for
  `SEED` or `FAST` are never overwritten: a stage whose config would differ (a plain run
  after `FAST=1`) stops. Use another `RUN_DIR`.
- `CKPT` must be a path under `checkpoints/`; `train` always writes there.
- `S3_PREFIX/<run-id>` is yours alone. Never write under another run's prefix.

**Resume on a new host.** Set up the host ([2.](#2-set-up-and-verify-a-host)), then copy
the finished outputs back and name the stages that are left:

```bash
training/aws/scripts/ssm-run.sh 'cd /opt/hobson/code/hobson-<agent> && aws s3 sync --only-show-errors s3://$HOBSON_BUCKET/results/<run-id>/data/ data/ && aws s3 sync --only-show-errors s3://$HOBSON_BUCKET/results/<run-id>/checkpoints/ checkpoints/ && aws s3 sync --only-show-errors s3://$HOBSON_BUCKET/results/<run-id>/run/ runs/recipe/'
```

**Checks.** After each stage, `stages.jsonl` gets one line with `exit_code`, `wall_s`,
`gpus_used`, `host_shape` and `git_rev`. The stage log (`runs/recipe/logs/<stage>.log`)
ends with the row-count lines: every `OK <file>: <n> rows`, no `FAIL`. `sha256.txt` holds
the sha256 of every data file; `sha256sum -c --strict data/SHA256SUMS` passes for the
committed files. `git_rev` is the commit you synced.

**Stop.** Any `exit_code` other than 0 in `stages.jsonl`; a `FAIL` row count (never set
`ALLOW_COUNT_MISMATCH=1` without your user); `git_rev` not the commit you meant; the
safety timer within reach of the stages left.

## 4. Verify a code change before you trust a retrain

**When.** The code changed since the published checkpoint and someone asks "does it still
train the same model?". A retrain is not bit-deterministic: the measured run-to-run noise
is in [Retraining on AWS](../evaluation/results.md#retraining-on-aws). A score difference
proves nothing either way. Prove identity statically and with deterministic gates first;
retrain last.

**Step 1: static identity, no GPU.** Take T, the commit in the published export's
`provenance.json` (`code_commit`, also `git_rev` on every line of its
`training/stages.jsonl`), and R, the commit under test.

1. Exact blobs: `git ls-tree -r` both, path by path, through any rename map (paths and
   tokens, applied in reverse to R). Classify `identical`, `differs`, `only-in-one`.
2. Normalize what differs: Python by AST with docstrings and comments dropped, then with
   annotations dropped; YAML parsed; everything else by sha256 after the rename map.
   Classify `same-logic`, `typing-only`, `differs`.
3. Trace every residual to one commit (`git log -S`, `git blame`) and say what it changes.
   Restrict the judgment to the route under test: the static call graph of
   `training/recipe.sh all`.

Result: the list of unexplained residuals is empty, or you stop here and report them.

**Step 2: deterministic gates, one host.** Both trees on the host, one venv each with the
same pins ([2.](#2-set-up-and-verify-a-host) twice, two code names). The published run's
record (its `training/data_sha256.txt`, `training/stages.jsonl`, `eval/jevbench-*/results.jsonl`)
and its frozen inputs (parent checkpoint, replay file) come from the export, not from a
new run.

| gate | what runs | pass |
| --- | --- | --- |
| corpora | `run_recipe.sh cpu` with R; `sha256sum -c --strict data/SHA256SUMS` | every sha256 in `runs/recipe/sha256.txt` equals the record. Some `data/synthetic/*.jsonl` are committed with CRLF line endings, so for those compare rows (`json.loads` of every line, in order) and record the raw mismatch |
| teacher | `run_recipe.sh teacher` with R | `data/teacher_multistep_v14*.jsonl` byte-identical to the record |
| replay | `training.data.replay` with R from the frozen reference parent | byte-identical to the record's replay file |
| bounded training | `python -m training.train` on 1 GPU, T and R, one config copy with a small `max_steps` and `log_every: 1`, frozen targets, PyTorch's deterministic settings, one warm-up run so Triton autotune choices are recorded and reused | initial weights, every step's gradient and weight digests, losses and final weights bit-identical for T vs T and for R vs T. The T vs T pair proves the setup is deterministic; without it, R vs T means nothing |
| serving | `jevbench.sh` with R's code on the reference checkpoint ([5.](#5-evaluate-on-jevbench)) | `n_correct` equals the record and the per-task probabilities are identical |

The repository has no gate tool. Keep yours outside the tree, record its sha256 in the
report, and change it only with a note in the report.

**Step 3: the full retrain** ([3.](#3-run-the-recipe)), judged by the reproduction bar
in [training/README.md](README.md).

**Stop.** An unexplained residual in step 1, any gate that is not exact in step 2. Report
them; your user decides whether a retrain is still worth running.

## 5. Evaluate on JevBench

**When.** A checkpoint to score, or the serving gate of [4.](#4-verify-a-code-change-before-you-trust-a-retrain).
Read [Reproducing, and two caveats](../evaluation/jevbench.md#reproducing-and-two-caveats)
and [5. JevBench](aws/README.md#5-jevbench).

**Commands.** On the host, inside `hobson-bg`. `GPU` is `CUDA_VISIBLE_DEVICES` and
defaults to 7: set `GPU=0` on a 1-GPU host. `PORT` defaults to 8099. The third argument
is a run in `research/data/jevbench_results.csv` (for example `v19`) for the paired test.

```bash
training/aws/scripts/ssm-run.sh 'hobson-bg jev "cd /opt/hobson/code/hobson-<agent> && PY=/opt/hobson/venv/bin/python GPU=0 evaluation/jevbench/jevbench.sh checkpoints/<ckpt> /opt/hobson/scratch/jev/<run-id> v19"'
training/aws/scripts/ssm-run.sh 'cat /opt/hobson/scratch/jev/<run-id>/paired.txt; aws s3 sync --only-show-errors /opt/hobson/scratch/jev/<run-id> s3://$HOBSON_BUCKET/results/<run-id>/jev'
```

The recipe saves a 4096-token window and `strands-decider serve` has no flag to change it.
To score at 3072 (the window v19 was pre-registered at), copy the checkpoint directory,
set `max_length` to 3072 in its `strands_decider_config.json` (`hobson_config.json` on a
checkpoint saved before the rename), and run `jevbench.sh` on the copy. Weights unchanged;
keep that config json for `hf_export --window-config`.

```bash
python evaluation/jevbench/paired.py --a research/data/jevbench_results.csv --a-run v19 --b <out_dir>/results.jsonl --list
```

**Checks.** The script stops by itself when the port already answers (the stale-server
trap: the benchmark would score the old model) and when `/health` names another
checkpoint or window; it rechecks `/health` after the run. `run_meta.json` says
`n_attempted` 231, `n_failed` 0, `schema_validity_strict` 1.0, easy tier 48/48, and
`hobson_git_rev` is your commit.

**Stop.** Any of those checks fails; `paired.txt` shows p < 0.05 against the baseline
for a run that claims to reproduce it.

## 6. Export and publish a checkpoint

**When.** A retrain to keep. The export is a Hugging Face model-repo folder: no pickles
(`slot_head.pt` becomes `head.safetensors`, tensors checked), everything else copied byte
for byte, a model card with provenance, and `MANIFEST.sha256` written last. Exporting the
same inputs twice gives the same bytes. The export never changes weights. Nothing in it
talks to the Hub.

**Commands.** Paths may be local or `s3://`. The run id is `results/<run-id>/` in the
checkpoint path, else `--run-id`.

```bash
python -m training.hf_export export s3://$HOBSON_BUCKET/results/<run-id>/checkpoints/hobson-2b-recipe \
  s3://$HOBSON_BUCKET/hf/<name>/<run-id> --run-dir s3://$HOBSON_BUCKET/results/<run-id>/run \
  --reports s3://$HOBSON_BUCKET/results/<run-id>/reports --jevbench s3://$HOBSON_BUCKET/results/<run-id>/jev \
  [--jevbench <dir of the 3072 arm> --window-config <its config json>] --repo-url <code repository URL>
python -m training.hf_export verify s3://$HOBSON_BUCKET/hf/<name>/<run-id>
python -m training.hf_export index  s3://$HOBSON_BUCKET/hf                     # INDEX.json and INDEX.md of every export
```

The exporter refuses an output that holds a different export or other files unless you
pass `--replace`, and always refuses an output that is an input or inside one.

**Publishing is the user's decision**: the Hub repo id, and whether it is private or
public. Publish from your own machine, where `huggingface_hub` already has the token.
Never copy a token to a host or into an SSM command.

```bash
hf upload <org>/<repo> <local export> . --commit-message "<what changed and why>"   # unchanged files are skipped; one commit
hf download <org>/<repo> --local-dir <fresh dir>
python -m training.hf_export verify <fresh dir>
```

**Checks.** `verify` prints `ok`: every file matches `MANIFEST.sha256` and the required
files are present. After publishing, `verify` on a fresh download passes and the Hub
commit lists only the files you changed.

**Stop.** `verify` fails, or the Hub commit would delete or replace files you did not
export. Do not pass `--replace` or re-upload to fix it; report.

## 7. Tear down

**When.** The end of every session, and before you stop for a question that may take
hours. Copy everything you need to S3 first: the root volume and the NVMe scratch are
deleted with the host. See [6. Teardown](aws/README.md#6-teardown).

**Commands.** With your tag, state directory and every region you launched in:

```bash
training/aws/scripts/teardown.sh          # dry run: lists what it would terminate
training/aws/scripts/teardown.sh --now    # terminates and waits for `terminated`
aws ec2 describe-instances --region <region> --filters "Name=tag:Name,Values=$HOBSON_NAME_TAG" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text     # confirm, per region
```

`teardown.sh` deletes only the hosts. What AWS bills before and after it, and how to
delete the rest, is in [Costs and cleanup](aws/README.md#costs-and-cleanup). Cancel any
reservation you created with `aws ec2 cancel-capacity-reservation`.

**Checks.** Every instance id you launched reads `terminated`; `teardown.sh --now` removed
`$HOBSON_STATE_DIR/host.env`; every reservation you created reads `cancelled`.

**Stop.** A host with your tag that you did not launch: do not terminate it; report it.
