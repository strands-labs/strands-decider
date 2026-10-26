# Benchmarking changes to the inference server

This document is for someone changing the serving path — the engine, the HTTP server, batching,
prefix caching, device code, quantization, a new deployment target — and who needs to prove the
change made the server faster (or at least no slower, and no less correct) before shipping it.

Accuracy benchmarks for a *checkpoint* live elsewhere: see [README.md](README.md) and
[jevbench.md](jevbench.md). The scripts below measure the *serving system* around a checkpoint
and treat correctness as a gate: did the perf change move any answers?

## The problem

A change to the inference server can affect three different things, and the tests have to tell
them apart:

1. **Speed per request.** How long one request takes end-to-end, from the client's point of view.
   This is what a user feels. On a decider it is time to first token (TTFT); the model decodes
   nothing, so TTFT is the whole latency.
2. **Capacity.** How many requests per second the server can finish under sustained load, and how
   latency grows as that load rises. A change that is 10% faster per request is worthless if it
   halves the sustainable throughput.
3. **Correctness.** Whether the change moved any answers. Quantization, kernel changes,
   batching math, device swaps and precision changes all put correctness at risk, so we re-check
   it alongside the perf numbers.

Every run records the host it ran on (EC2 instance type for local, function memory for Lambda,
endpoint instance type for SageMaker). A number without a host is not comparable.

## Metrics collected

Every row below comes from [`decider_bench.py`](decider_bench.py), one row per
(model × input length × load) cell, written to `runs/<name>/latency.csv`. Correctness rows go to
`accuracy.csv` (the file keeps its historical name; the column `score` is what you read).

### Latency (per-request)

| Column | What it is | What drives it down |
| --- | --- | --- |
| `p50_ms` | Median TTFT, as the client sees it | Smaller model, faster device, better kernels |
| `p90_ms`, `p95_ms`, `p99_ms` | Tail TTFT. **p95 is the number to quote.** | Reduce queueing (batching strategy, admission control) |
| `mean_ms` | Mean TTFT. Report only alongside p50; the mean on its own hides tails. | — |
| `server_p50_ms` | What the server itself reported as `latency_ms` for the same request | Diff against `p50_ms` → the client/server gap is network + queueing |
| `tokens_mean` | Actual input tokens sent, averaged over the cell | Sanity check: did `--input-length 1024` really send ~1024? |

### Throughput and load

| Column | What it is | What drives it up |
| --- | --- | --- |
| `decisions_per_s` | Finished requests per second in that cell | Larger batches, more parallelism, prefix cache hits |
| `load` / `concurrency` / `rps` | How the cell was driven (see [Tests](#tests)) | — |
| `n`, `ok` | Requests attempted and succeeded | Fix anything causing `ok < n` before trusting the perf numbers |

### Errors

| Column | Meaning |
| --- | --- |
| `refused` | HTTP 422: prompt longer than the context window. Expected if you benchmark above `max_length`; a problem if you did not. |
| `errors` | Any other non-2xx. Any non-zero count invalidates the row — fix the server, re-run. |

### Correctness (gate, not a goal)

Written to `accuracy.csv` when you pass `--metrics ttft,jevbench` or `--metrics ttft,decision-index`.
These are the server's correctness check: the same labelled benchmark run before and after the
change, to prove it did not move answers.

| Column | What it is |
| --- | --- |
| `score` | Fraction correct on the benchmark (0 to 1) |
| `correct`, `of` | Numerator and denominator behind the score |
| `complete` | Whether the run finished every task |

The release rule: a perf change must not drop `score` on JevBench by more than the noise floor
(SD ≈ 3 tasks out of 231, so about 0.013). See
[README.md, Limitations](README.md#limitations) for how to size the effect. One run's drop
inside that noise is not a failure; outside it is.

## Tests

Every test is run twice — once on the baseline (today's server, same commit) and once on the
change — and the pair is compared. Running the change on its own tells you nothing: hosts drift
and queue depth is noisy. All of these are shapes of one `decider_bench.py` command.

### 1. Short vs long input

Decider latency scales with input tokens; a change can help short prompts and hurt long ones
(or the other way around). Measure both.

```bash
python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --input-length 256,1024,4096 \
    --concurrency 1 --num-requests 50 \
    --out runs/length-sweep
```

- `--input-length` sweeps synthetic requests trimmed to exactly that many tokens.
- `--concurrency 1` keeps load out of it so the number reflects pure per-request latency.
- Both servers must be reachable at the same time (two ports, or baseline + a `sagemaker:` target).

Figure to read: `ttft_vs_input_length.png` (one line per model). The useful comparison is the
*slope*, not just the intercept — if the change is faster at 256 and slower at 4096, say so.

### 2. Hot vs cold

The first request after the server comes up pays for weight load, VRAM allocator warm-up and
(for serverless) container start. Steady-state TTFT is a separate number. A change that improves
one can regress the other.

Three ways to measure, in order of fidelity:

- **Steady-state only (default).** `decider_bench.py` always discards a warm-up round. Every
  cell is already a hot number.
- **Cold first-request.** The dedicated script is [`jevbench/jevbench_cold_warm.py`](jevbench/jevbench_cold_warm.py):
  one task, fired twice, reports the first vs the second (MPS). Generalize it or write a small
  wrapper if you need the same test against Lambda or SageMaker.
- **Serverless cold start.** For Lambda or a SageMaker Serverless endpoint, scale to zero
  before the test (idle long enough, or redeploy) and run `decider_bench.py` with
  `--num-requests 1 --concurrency 1`. The single TTFT measurement is the cold start.

A perf change that only ever runs once per cold boot (e.g. a one-time compile) should be
benchmarked by running the full suite twice: once immediately after start, once after N minutes.

### 3. Load (closed loop vs open loop)

The two load knobs answer different questions. Run both.

**Closed loop: `--concurrency N`** — N requests in flight at all times; each simulated user sends
its next request when the previous answer arrives. The highest N shows the server's maximum
sustainable throughput. Use this to size the server and to find batch-size and parallelism
regressions.

```bash
python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --input-length 1024 --concurrency 1,8,32,64 \
    --num-requests 100 \
    --out runs/throughput-sweep
```

Figure to read: `throughput_vs_input_length.png` and `ttft_vs_throughput.png`. The latter is the
plot that quotes a p95 at a chosen decisions-per-second — the one to put in a design doc.

**Open loop: `--rps R`** — R requests per second at Poisson arrival times, regardless of whether
answers keep up. TTFT then *includes* any queueing, which is what a real user with no backoff
sees. Use this to find the throughput at which p95 crosses your SLA.

```bash
python evaluation/decider_bench.py \
    --model change=http://127.0.0.1:8000 \
    --input-length 1024 --rps 2,8,24 \
    --num-requests 200 \
    --out runs/rps-sweep
```

A closed-loop p95 at N concurrency is almost always lower than the open-loop p95 at the same
sustained throughput. Reporting one when you meant the other is a common source of surprised
pagers.

### 4. Correctness gate

Every perf change that touches numerics (quantization, kernels, batching, device, precision)
runs JevBench on both baseline and change through the same bench:

```bash
python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --metrics ttft,jevbench \
    --input-length 1024 --num-requests 20 \
    --out runs/correctness-gate
```

- JevBench uses its own inputs; the TTFT part uses the ones above. Both land in the same run
  directory.
- Change ships only if the JevBench delta is within noise (about ±3 tasks, see
  [the limitations section](README.md#limitations)).
- For a device swap or a Mac-vs-CUDA question, [`device_parity.py`](device_parity.py) is the
  stronger check: it reports the largest probability difference and lists the answers that
  changed, not just a scalar score.

A perf change that does *not* touch numerics (HTTP tweaks, process-level batching, admission
control) can skip this test, but should say so explicitly in the change description.

### 5. Cross-host comparison

For "what instance should we run on?" or "is Lambda cheaper than SageMaker for our shape?",
run the same bench against each target. One `--model` per target:

```bash
python evaluation/decider_bench.py \
    --model g5=sagemaker:strands-decider-g5@us-east-1 \
    --model g6=sagemaker:strands-decider-g6@us-east-1 \
    --model lambda=lambda:strands-decider-live@us-east-1 \
    --input-length 256,1024,4096 --concurrency 1,8,32 \
    --num-requests 50 \
    --out runs/host-compare
```

Each host is auto-labelled by instance type (SageMaker) or function memory (Lambda), so the
figures group correctly. For hosts you cannot reach from one machine, run the bench on each
host and combine the directories at plot time:

```bash
python evaluation/decider_bench.py plot runs/g5 runs/g6 runs/lambda
```

The [benchmark-decider skill](../skills/benchmark-decider/SKILL.md) walks through planning and
writing up a multi-host comparison in phases, with gates between them.

## Driving metrics down

The order these pay off, from a server-perf change:

1. **`p50_ms` at a representative input length and `--concurrency 1`.** Floor on how fast one
   request can be. Baseline-to-change delta here is the clearest signal.
2. **`decisions_per_s` at the highest `--concurrency` the server survives.** Headroom on peak
   throughput. Harder to move than p50 and gated by batching.
3. **`p95_ms` on `--rps` at the throughput you need to serve.** The number a design doc should
   quote. If this is bad, latency improvements at p50 are cosmetic.
4. **`server_p50_ms` vs `p50_ms`.** If the gap grows under load, your bottleneck is queueing
   (or network), not model compute. Fix the batching/admission policy; model work will not
   help.
5. **Cold-start TTFT.** Only matters for serverless or auto-scaled fleets, where every scale-up
   costs a user a slow request.

The one trap: before trusting any of the above, verify `ok == n` and `refused == 0` on the row
you are reading. A 10% latency improvement that comes with silent errors is a regression.

## Running the full suite

The quickest end-to-end check for a server change:

```bash
# baseline
python evaluation/decider_bench.py --model baseline=http://127.0.0.1:8000 \
    --metrics ttft,jevbench --input-length 256,1024,4096 \
    --concurrency 1,8,32 --num-requests 50 --out runs/baseline

# change (same shape)
python evaluation/decider_bench.py --model change=http://127.0.0.1:8001 \
    --metrics ttft,jevbench --input-length 256,1024,4096 \
    --concurrency 1,8,32 --num-requests 50 --out runs/change

# combined figures and tables
python evaluation/decider_bench.py plot runs/baseline runs/change
```

Needs `matplotlib` for the figures (`pip install matplotlib`); the CSVs are written without it.

Outputs per run: `latency.csv`, `accuracy.csv`, `requests.jsonl`, `summary.json`,
`server.log` (if the bench ran its own server), and `ttft_vs_input_length.png`,
`ttft_vs_throughput.png`, `throughput_vs_input_length.png` once plotted.
