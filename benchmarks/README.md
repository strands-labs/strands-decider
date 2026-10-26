# Benchmarks

Server-side latency and throughput measurements per `(host, model)` pair.

Each measured pair is a directory under `runs/<host>-<device>-<model>/` holding the raw
`latency.csv`, `accuracy.csv`, `summary.json`, `requests.jsonl` from
[`evaluation/decider_bench.py`](../evaluation/decider_bench.py), plus the three standard
figures: `ttft_vs_input_length.png`, `ttft_vs_throughput.png`, `throughput_vs_input_length.png`.

> For *how* to benchmark a change (metrics, tests, pass/tune), see
> [`benchmarking.md`](benchmarking.md) in this directory.

## Related benchmarks (sizing and cross-instance)

This directory is for **incremental testing of changes to the serving stack on one host**. For
the complementary question — *which instance should I deploy on, with which model?* — there is
a separate benchmark run against stock vLLM, the current strands_decider engine, and the vLLM
"Decision 2.0" family across g5, g6, g6e and inf2. Two figures:

- `$ per 1M decisions` vs model size, by input length, per (instance × serving stack).
- Saturation `decisions/s` vs model size, same layout.

Takeaways relevant to this directory:

- The current `strands_decider` serving engine is **batch of 1**. On the data drop, adding
  batching and flash linear attention raises throughput ~10×. A PR is in flight.
- `inf2.8xlarge` with a ported Neuron kernel gives the lowest `$ / 1M decisions` for strands E2B
  (~$5 vs ~$6.50 on the next best, `g6e.xlarge`), at 256 input tokens.
- Stock vLLM on the backbone model (no decider head) is 5–10× faster than the current
  strands_decider engine. This is the ceiling our engine should chase.

The sweep below (`--concurrency 1,8,32`) is chosen so our numbers compare directly against the
data drop's saturation point (`c=32`). When the batching PR lands, re-run every measured host.

## Host × serving-stack matrix

A row is a `(host, serving stack)` pair. A single host can appear more than once as the engine
evolves (today's batch-of-1 engine; the batching PR; stock vLLM for a non-decider baseline).

### Measured

| Host | Serving stack | Model | Run directory | Status |
| --- | --- | --- | --- | --- |
| Apple M4 Pro | strands_decider (batch of 1) | 2B hobson-v21 | [`runs/m4-pro-mlx-2b-v21/`](runs/m4-pro-mlx-2b-v21/) | Indicative — `--concurrency 1,8,32`, 50 reqs/cell |

### Supported deploy targets, not yet measured

| Host | Serving stack | Reach as | Likely run directory |
| --- | --- | --- | --- |
| NVIDIA RTX 3090 (WSL2) | strands_decider | `http://...` after `strands-decider serve --device cuda` | `runs/rtx-3090-cuda-2b-v21/` |
| SageMaker, NVIDIA L4 (`ml.g6.xlarge`) | strands_decider | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-g6-xlarge-2b-v21/` |
| SageMaker, NVIDIA A10G (`ml.g5.xlarge`) | strands_decider | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-g5-xlarge-2b-v21/` |
| SageMaker, NVIDIA L40S (`ml.g6e.xlarge`) | strands_decider | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-g6e-xlarge-2b-v21/` |
| SageMaker, NVIDIA H100 (`ml.p5.4xlarge`) | strands_decider | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-p5-4xlarge-26b-v21/` |
| SageMaker CPU (`ml.m5.2xlarge`, etc.) | strands_decider | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-m5-2xlarge-2b-v21/` |
| AWS Lambda | strands_decider | `lambda:FUNCTION[:ALIAS][@REGION]` | `runs/lambda-<mem>mb-2b-v21/` |
| Apple M3 Pro | strands_decider | self-hosted | `runs/m3-pro-mps-2b-v21/` |
| AWS Inferentia2 (`inf2.*`) | strands_decider + Neuron port (forked, upstream pending) | self-hosted | `runs/inf2-xlarge-neuron-2b-v21/` |
| any host above | strands_decider + batching (PR pending) | self-hosted | append `-batched` suffix |
| any host above | stock vLLM on the backbone (reference, no decider head) | self-hosted | append `-vllm` suffix |

### Unsupported

The deploy stack refuses these; `strands-decider serve` has no backend for them.

| Host | Why |
| --- | --- |
| Graviton (arm64 EC2) | Torso has no arm64 fused-kernel path |
| AWS Inferentia 1 (`inf1`) | No Neuron SDK integration; Qwen3.5 / Gemma 4 not supported |
| AWS Trainium (`trn1`, `trn2`) | Training-only accelerator |

## Adding a host

1. **Deploy** the target per [`deploy/`](../deploy/) (SageMaker, Lambda) or run
   `strands-decider serve <checkpoint> --device <cuda|mps|mlx|cpu>` on the host.
2. **Measure** with [`evaluation/decider_bench.py`](../evaluation/decider_bench.py). The
   standard sweep this directory expects:
   ```bash
   python evaluation/decider_bench.py \
       --model this-host=<URL or sagemaker:... or lambda:...> \
       --metrics ttft,jevbench \
       --input-length 256,1024,4096 \
       --concurrency 1,8,32 \
       --num-requests 50 \
       --out benchmarks/runs/<host-slug>
   ```
3. **Plot** the standard figures into the same directory:
   ```bash
   python evaluation/decider_bench.py plot benchmarks/runs/<host-slug>
   python benchmarks/plot_throughput.py benchmarks/runs/<host-slug>
   ```
   The first writes `ttft_vs_input_length.png`, `ttft_vs_throughput.png` and
   `throughput_vs_input_length.png` (stock, peak throughput only). The second writes
   `throughput_by_concurrency.png` — one line per concurrency across input lengths, each point
   annotated with its p50 latency, so a backend that scales with load reads differently from a
   batch-of-1 backend.
4. **Update this README's host matrix** so the row moves from *not yet measured* to *measured*.
   If you want a cost figure, apply the formulas in [Cost](#cost) below and add the number to
   your host's run README.

## Cost

Cost per decision is a function of the host's price and the throughput the measurements show.
`decider_bench.py` records `decisions_per_s` for every (input length, load) cell in
`latency.csv`; combine that with the host's price to get cost per million decisions. Three cases.

**Always-on servers** (your own hardware, SageMaker real-time, EC2 Linux CUDA):

```
$ per 1M decisions = (hourly $) / (sustained decisions/s) × 1,000,000 / 3,600
```

Pick which `decisions_per_s` to use before quoting:

- **Peak** — the highest cell across the sweep. Assumes 100% utilization. Honest for a saturated
  server; a floor on cost, not the expected cost.
- **Sustained at your SLA concurrency** — the `--concurrency N` cell where p95 is still under
  your target. More realistic for traffic that isn't saturated; "N" is roughly the number of
  simultaneous users you can serve.

Prices: SageMaker on-demand at <https://aws.amazon.com/sagemaker-ai/pricing/>, EC2 at your actual
invoice, `$0` for own hardware (or amortize purchase ÷ hours used if you need a figure).

**AWS Lambda** (per-request billing):

```
$ per request = per_invoke + gb_second × memory_GB × p50_latency_seconds
$ per 1M      = $ per request × 1,000,000
```

Memory is set at deploy time and recorded in `summary.json`'s host label; `p50_latency_seconds`
is `p50_ms / 1000` from `latency.csv` (use `server_p50_ms` if you want to exclude client-side
network from the billed duration — but Lambda bills the invocation wall clock, so `p50_ms` is
usually what you want). Current rates at <https://aws.amazon.com/lambda/pricing/>.

Cold-start cost: Lambda bills GB-seconds for the whole invocation including init, which can be
5–10× a warm request on first invoke. If your traffic has gaps that scale the function to zero,
measure cold start separately (benchmarking.md, test 2) and weight accordingly.

**Own hardware** (M3 Pro, M4 Pro, RTX 3090 on your box): zero marginal cost per decision. If you
need a number anyway, divide the hardware's purchase price by its useful life in hours and treat
that as the hourly rate.

### Caveats

- One host, one model, one input length: changing any of those changes `decisions_per_s` and
  therefore the cost. Quote the shape, not just the number.
- Utilization matters more than peak throughput for always-on hosts. A SageMaker endpoint at
  10% utilization costs 10× its peak-throughput row.
- Prices change. Record the source URL and date next to any cost number you commit.
