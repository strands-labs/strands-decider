---
name: benchmark-decider
description: >
  Measure how fast and how accurately Strands Decider models run on different engines, precisions,
  hosts and serving setups (a local checkpoint, any /v1/systemone server, AWS Lambda, SageMaker),
  and turn the measurements into figures and a written analysis. Use when a user asks "how fast is
  the decider on X", wants time to first token by input length, throughput or tail latency under
  load, cost per million decisions, a comparison of engines, instance types or model sizes, or a
  check that an optimization kept the model's accuracy (JevBench, Decision Index). Also use to make
  figures such as latency against input length, p95 against throughput, or throughput per GPU
  across hosts and model sizes.
---

# Benchmark Strands Decider

The purpose: answer a performance question with numbers that hold up. That means the same inputs
and the same load for every system compared, a recorded host for every number, and accuracy checked
next to speed whenever the numerics change.

The tool is [`evaluation/decider_bench.py`](../../evaluation/decider_bench.py). It runs on the host
where the client should be, measures whatever `--model` points at, and never starts anything
elsewhere. When the user wants several hosts, you run it on each host (or have the user do so) and
combine the run directories afterwards.

The user owns two kinds of decisions; ask before each one:

- **Money:** every instance, its type, region and hours; every Lambda or SageMaker invocation
  campaign beyond a smoke test; a full Decision Index run (about 140,000 requests).
- **Their cloud resources:** creating, updating or deleting functions, endpoints, instances or
  quotas. The tool only reads (invokes, and describes for the host label); anything that changes a
  resource is yours to propose and theirs to approve.

Read each reference file when its phase starts.

| phase | what you produce | gate to pass before the next phase |
|---|---|---|
| 1. Question | `work/plan.md` | the user agrees with what varies, what is fixed, and the cost |
| 2. Smoke | one tiny run per target | every target answers, its host is right, token counts match |
| 3. Runs | one run directory per host | no unexplained errors or refusals; key points repeated |
| 4. Figures | standard and custom figures | each figure answers a question in the plan |
| 5. Write-up | `work/results.md` | every number traces to a run directory and a command |

Keep a dated log as you go (`work/LOG.md`): each command, where it ran, what it printed, the run
directory, and what you concluded. Log failures and corrections too, with their evidence.

## The tool in one screen

```bash
python evaluation/decider_bench.py --model [NAME=]MODEL [--model ...] \
    [--metrics ttft,jevbench,decision-index] \
    [--input-length 256,1024,4096] [--dataset requests.jsonl] \
    [--concurrency 1,8,32] [--rps 2,8] [--num-requests 50] [--out runs/NAME]
python evaluation/decider_bench.py plot runs/a runs/b ...
```

- `--model` takes a checkpoint directory or Hub id (served on this host by `strands-decider serve`,
  with `--device`; `--serve-cmd` runs any other server), an `http(s)://` URL, `lambda:FUNCTION[:ALIAS][@REGION]`
  or `sagemaker:ENDPOINT[@REGION]`. `NAME=` names it in tables and figures.
- `ttft` is time to first token, which is a decider's whole latency because it decodes nothing.
  It is measured for every input and every load, with the decisions per second achieved.
- Inputs: `--input-length` sends synthetic requests of exact input-token counts; `--dataset` sends the
  user's own requests (one `{"state", "questions"}` object per line). Either or both.
- Load: `--concurrency N` keeps N requests in flight (a closed loop, which finds the throughput
  ceiling); `--rps R` sends R requests per second at Poisson times (an open loop, which shows latency
  under independent traffic, queueing included). Either or both.
- `jevbench` runs the 231 public tasks through JevBench's own harness; `decision-index` runs the
  public index through its kit and needs `--di-suite`. Neither uses the input or load flags.
- Each run writes `latency.csv`, `accuracy.csv`, `summary.json`, `requests.jsonl` (every timed
  request) and standard figures. `plot` combines runs, one column per host.

Requirements: the strands-decider package for local checkpoints, `boto3` for Lambda and SageMaker,
`matplotlib` for figures, `git` for JevBench, and the Decision Index kit for `decision-index`.

## Phase 1. Frame the question

Write `work/plan.md` with the user before running anything. It names:

1. **The comparison.** What varies (engine, precision, model size, host, packaging) and what is
   held fixed (the model, the inputs, the load, where the client runs). Two systems that differ in
   more than the factor under test do not answer the question.
2. **Inputs.** `--input-length` when the question is how cost scales with input length; `--dataset` when
   it is how the system behaves on real traffic. The user's own requests beat any synthetic mix.
   Input lengths above the model's window (4,096 tokens for the released models) are skipped.
3. **Load.** `--concurrency 1` gives the latency floor. A concurrency sweep (1, 4, 16, 32, 64) finds
   the throughput ceiling. An `--rps` sweep at about 25, 50, 75 and 90% of that ceiling shows latency
   under realistic traffic; get the ceiling from a first run.
4. **Accuracy.** Any change to the numerics (quantization, a new kernel or engine, a new host
   architecture) needs JevBench at least. A release-grade claim needs the Decision Index.
5. **Samples.** 50 requests per point for medians; 200 or more for p95 and p99 claims; open loops
   need more than closed loops.
6. **Cost.** Instance hours, invocations and their rough price ([references/analysis.md](references/analysis.md)).

## Phase 2. Smoke test

Run every target once with `--input-length 256 --num-requests 5`. Check that each model started, that
its `host` in `latency.csv` is what you expect, that `tokens_mean` is close to the requested length,
and that there are no errors. Fix anything odd now; a full run on a broken target wastes hours.

## Phase 3. Runs

- **Run the client where the claim is about.** For a model served on a GPU instance, run the script
  on that instance. For Lambda or SageMaker, run it from an instance in the same region when the
  latency itself matters; a laptop adds tens of milliseconds and jitter to every request.
- **One run directory per host**, named for it (`runs/g6e.xlarge`, `runs/lambda-arm64`). Put all
  the models measured on that host in the same run with distinct `NAME=`s.
- **Name models so the names carry the factors**, for example `2b-released-bf16` and `2b-vllm-fp8`,
  so that `figures.split_name` can turn them into columns for a grid later.
- **Orchestrating from a laptop:** copy the script to each host, run it there, copy the run
  directory back. Stop every instance you started when its runs are done; the script stops any
  server it launched itself.
- **Repeat before you claim.** A difference under about 5% between two runs on different days, or
  on different instances, needs a repeat in one session before it counts.

[references/pitfalls.md](references/pitfalls.md) lists what has gone wrong before. Read it before
the first full run.

## Phase 4. Figures

`python evaluation/decider_bench.py plot runs/*` draws the standard set:

- `ttft_vs_input_length.png`: time to first token against input length, one line per model and load.
- `ttft_vs_throughput.png`: p95 time to first token against decisions per second, one row per input,
  one column per host.
- `throughput_vs_input_length.png`: the highest decisions per second measured against input length.
- `accuracy.png`: JevBench and the Decision Index.

For anything else, use [`scripts/figures.py`](scripts/figures.py) over `latency.csv` and
`accuracy.csv`, for example a grid with model sizes in rows, hosts in columns and engines as lines.
[references/figures.md](references/figures.md) has the recipes and the style rules every figure
follows.

## Phase 5. Write-up

`work/results.md` answers the question in the plan in its first paragraph, then supports it.
[references/analysis.md](references/analysis.md) covers first-principles checks, cost per million
decisions and the break-even rate, and how to state uncertainty. Write numbers as a value with its
difference from the baseline in parentheses, for example "0.61 s (−2.89 s vs. the released
engine's 3.50 s)". Caption every table and figure, refer to each in the text, and say what the
measurements do not cover.
