# Benchmarking changes to the inference server

This document is for someone changing the serving path — the engine, the HTTP server, batching,
prefix caching, device code, quantization — and who wants to prove on their own machine that the
change made the server faster (or at least no slower, and no less correct) before pushing it.

Scope is **one host, baseline vs change**. The point is incremental: measure what today's server
does, apply the change, re-measure. Multi-host comparisons ("Lambda vs SageMaker g6 vs the Mac")
are a separate exercise — see [`skills/benchmark-decider/`](../skills/benchmark-decider/SKILL.md),
which walks through planning and writing one up in phases.

Accuracy benchmarks for a *checkpoint* also live elsewhere: see [README.md](README.md) and
[jevbench.md](jevbench.md). The scripts below measure the serving system around a checkpoint and
treat correctness as a gate — did the perf change move any answers?

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

## Shared rules

Every test below is one call to [`decider_bench.py`](decider_bench.py), writes to
`runs/<name>/latency.csv` (perf) and `accuracy.csv` (correctness, when `--metrics` names a
benchmark), and comes with figures via `decider_bench.py plot runs/...`.

Four invariants that make the numbers trustworthy:

- **Pin the host.** Baseline and change run on the same hardware in the same session. A number
  from a different host does not count. `decider_bench.py` records `host`, `gpus`, `cpu`, `spec`
  on every row so you can see when this slipped.
- **Pair every run.** Baseline (today's server, same commit) and change, same inputs, same load.
  A single run tells you nothing — queue depth and warm-up noise swamp small effects. The
  typical shape is two local servers on `:8000` and `:8001`, with `--model baseline=http://...`
  and `--model change=http://...` in one command so both are measured against the same clock.
- **Gate on `n`, `ok`, `refused`, `errors` first.** Any non-zero `errors`, or `ok < n`, means the
  perf numbers for that row do not count. `refused` (HTTP 422, prompt over the context window)
  is expected only when you deliberately exceed `max_length`.
- **Server-side has only p50.** `decider_bench.py` reports `server_p50_ms` from the model's own
  `latency_ms` field but no `server_p90/95/99_ms`. The full server distribution is in
  `requests.jsonl` if you want to re-aggregate; adding server tail columns is a four-line patch.

## Tests

### 1. Short vs long input

Does the change help short prompts, long prompts, or both? Decider latency scales with input
tokens, so a change can help one end and hurt the other.

```bash
python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --input-length 256,1024,4096 \
    --concurrency 1 --num-requests 50 \
    --out runs/length-sweep
```

| Column | What it is here | Read for |
| --- | --- | --- |
| `input_length` | The target length for this cell | Row key |
| `tokens_mean` | Actual tokens sent, averaged | Sanity: did 1024 really hit? |
| `p50_ms` | Pure per-request TTFT at that length | The floor on how fast one request can be |
| `p95_ms` | Tail TTFT at that length | How much the slow requests slow down |
| `server_p50_ms` | Server-reported model-only p50 | Gap vs `p50_ms` = network overhead at zero load |

**Figure:** `ttft_vs_input_length.png` — one line per model. The slope matters more than the
intercept: a change faster at 256 and slower at 4096 is a different change than a flat shift.

**Pass/tune:** this test's `p50_ms` at a representative input length is the clearest signal of
whether a change made one request faster. Drive it down by shrinking the model, swapping to a
faster device, or improving kernels. If `p95_ms` is high at `--concurrency 1`, something is wrong
beyond load — look at GC, allocator warm-up, or a one-off slow path.

### 2. Hot vs cold

The first request after a server starts pays for weight load, VRAM allocator warm-up and (for
some devices) kernel compilation. Steady-state TTFT is a separate number, and a change can help
one and regress the other.

```bash
# steady-state is the default; decider_bench discards a warm-up round every run.
# For a dedicated cold-vs-warm number against a local checkpoint:
python evaluation/jevbench/jevbench_cold_warm.py <checkpoint>
```

| Column | What it is here | Read for |
| --- | --- | --- |
| `p50_ms` (cold run, n=1) | The one cold TTFT | Cold-start cost on this host |
| `p50_ms` (steady suite) | TTFT after warm-up | Steady-state cost |

**Figure:** none standard — report the pair as two numbers in the write-up.

**Pass/tune:** cold start matters where the server restarts often (dev loop, local kill-and-run,
auto-restart on crash). Drive it down by pre-loading weights, caching compiled kernels, or
reducing the warm-up round's work. A change that only pays off on cold start should be verified
against a steady-state suite — some wins amortize to zero after the first second.

### 3. Load — closed loop (`--concurrency N`)

What is the server's maximum sustainable throughput, and how does latency grow as load rises?
Closed loop keeps `N` requests in flight at all times; each simulated user sends its next request
when the previous answer arrives.

```bash
python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --input-length 1024 --concurrency 1,8,32,64 \
    --num-requests 100 \
    --out runs/throughput-sweep
```

| Column | What it is here | Read for |
| --- | --- | --- |
| `concurrency` | In-flight requests for this cell | Row key |
| `decisions_per_s` | Throughput at this concurrency | Peak = highest value across the sweep |
| `p50_ms` | TTFT including time spent in queue | How badly the server slows under load |
| `p95_ms`, `p99_ms` | Tail under load | Pager risk |
| `server_p50_ms` | Server-reported model-only p50 | **`p50_ms − server_p50_ms` growing with N = queue cost, not compute** |

**Figure:** `throughput_vs_input_length.png` (peak) and `ttft_vs_throughput.png` (p95 at each
decisions-per-second — the plot to put in a design doc).

**Pass/tune:** peak `decisions_per_s` is harder to move than p50; it is gated by batching
strategy, request-coalescing and allocator behaviour. The queue-cost diagnostic
(`p50_ms − server_p50_ms`) tells you where to spend time — if the gap grows with `N`, batching
and admission control pay off; if it stays flat, only model-side work (quantization, kernels,
smaller model) will help.

### 4. Load — open loop (`--rps R`)

What throughput can we serve before p95 crosses our SLA? Open loop sends `R` requests per second
at Poisson arrival times, regardless of whether answers keep up — queue time counts toward TTFT.

```bash
python evaluation/decider_bench.py \
    --model change=http://127.0.0.1:8000 \
    --input-length 1024 --rps 2,8,24 \
    --num-requests 200 \
    --out runs/rps-sweep
```

| Column | What it is here | Read for |
| --- | --- | --- |
| `rps` | Target request rate for this cell | Row key |
| `p95_ms` | TTFT with queueing, at that rate | **The number a design doc quotes against an SLA** |
| `p99_ms` | Worst-case tail at that rate | Pager risk under real traffic |
| `decisions_per_s` | What the server actually delivered | Below `rps` ⇒ the server fell behind |

**Figure:** `ttft_vs_throughput.png`.

**Pass/tune:** closed-loop p95 at `N` concurrency is almost always lower than open-loop p95 at
the same sustained throughput; reporting one when you meant the other is a common source of
surprised pagers. Drive down by shortening the critical path (same levers as test 3) or by
rejecting load before it queues (admission control / timeouts).

### 5. Correctness gate

Did the change move any answers? Run JevBench on both baseline and change through the same bench.

```bash
python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --metrics ttft,jevbench \
    --input-length 1024 --num-requests 20 \
    --out runs/correctness-gate
```

| Column | File | What it is here | Read for |
| --- | --- | --- | --- |
| `score` | accuracy.csv | Fraction correct on JevBench's 231 tasks | Baseline-vs-change delta |
| `correct`, `of` | accuracy.csv | Numerator and denominator | Sanity on `score` |
| `complete` | accuracy.csv | Whether every task ran | **If false, the row does not count** |

**Figure:** none; this is a scalar gate.

**Pass/tune:** the ship rule is "within noise" — SD ≈ 3 tasks of 231, so about 0.013. One run's
drop inside that band is not a failure; outside it is. See
[README.md, Limitations](README.md#limitations) for sizing. For a change that *must not* move
answers (same math, different code path), [`device_parity.py`](device_parity.py) is the stronger
check — it reports the largest probability difference and lists specific requests whose argmax
changed, which is a legitimate hard pass/fail. A change that does not touch numerics (HTTP
tweaks, process-level batching, admission control) can skip this test but should say so in the
change description.

## Running the full suite

The quickest end-to-end check for a server change on your machine:

```bash
# serve baseline on :8000 and change on :8001 before running (two shells, or your usual pattern).

python evaluation/decider_bench.py \
    --model baseline=http://127.0.0.1:8000 \
    --model change=http://127.0.0.1:8001 \
    --metrics ttft,jevbench \
    --input-length 256,1024,4096 \
    --concurrency 1,8,32 --num-requests 50 \
    --out runs/change-vs-baseline

python evaluation/decider_bench.py plot runs/change-vs-baseline
```

Needs `matplotlib` for the figures (`pip install matplotlib`); the CSVs are written without it.

Outputs per run: `latency.csv`, `accuracy.csv`, `requests.jsonl`, `summary.json`,
`server.log` (if the bench ran its own server), and `ttft_vs_input_length.png`,
`ttft_vs_throughput.png`, `throughput_vs_input_length.png` once plotted.

For *host* comparisons (which GPU, which instance type, serverless vs always-on) rather than
*change* comparisons, see [`skills/benchmark-decider/`](../skills/benchmark-decider/SKILL.md).
