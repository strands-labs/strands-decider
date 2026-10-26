# Benchmarks

Server-side latency, throughput and cost measurements per `(host, model)` pair.

Each measured pair is a directory under `runs/<host>-<device>-<model>/` holding the raw
`latency.csv`, `accuracy.csv`, `summary.json`, `requests.jsonl` from
[`evaluation/decider_bench.py`](../evaluation/decider_bench.py), plus the three standard
figures: `ttft_vs_input_length.png`, `ttft_vs_throughput.png`, `throughput_vs_input_length.png`.

Cross-host comparisons (cost per 1M decisions, latency/price frontier) go in `figures/`,
produced by `cost.py` from `runs/` and `prices.yaml`.

> For *how* to benchmark a change (metrics, tests, pass/tune), see
> [`evaluation/benchmarking.md`](../evaluation/benchmarking.md). This directory holds the data
> and figures those tests produce; `evaluation/` holds the methodology.

## Host matrix

### Measured

| Host | Device | Model | Run directory | Status |
| --- | --- | --- | --- | --- |
| Apple M4 Pro | mlx | 2B hobson-v21 | [`runs/m4-pro-mlx-2b-v21/`](runs/m4-pro-mlx-2b-v21/) | Indicative smoke (20 reqs/cell, 2 input lengths, 2 loads) |

### Supported deploy targets, not yet measured

| Host | Device | Reach as | Likely run directory |
| --- | --- | --- | --- |
| NVIDIA RTX 3090 (WSL2) | cuda | `http://...` after `strands-decider serve --device cuda` | `runs/rtx-3090-cuda-2b-v21/` |
| SageMaker, NVIDIA L4 (`ml.g6.xlarge`) | cuda | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-g6-xlarge-2b-v21/` |
| SageMaker, NVIDIA A10G (`ml.g5.xlarge`) | cuda | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-g5-xlarge-2b-v21/` |
| SageMaker, NVIDIA L40S (`ml.g6e.xlarge`) | cuda | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-g6e-xlarge-2b-v21/` |
| SageMaker, NVIDIA H100 (`ml.p5.4xlarge`) | cuda | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-p5-4xlarge-26b-v21/` |
| SageMaker CPU (`ml.m5.2xlarge`, etc.) | cpu | `sagemaker:ENDPOINT@REGION` | `runs/sagemaker-m5-2xlarge-2b-v21/` |
| AWS Lambda | cpu (x86) | `lambda:FUNCTION[:ALIAS][@REGION]` | `runs/lambda-<mem>mb-2b-v21/` |
| Apple M3 Pro | mps | self-hosted | `runs/m3-pro-mps-2b-v21/` |

### Unsupported

The deploy stack refuses these; `strands-decider serve` has no backend for them.

| Host | Why |
| --- | --- |
| Graviton (arm64 EC2) | Torso has no arm64 fused-kernel path |
| AWS Inferentia (`inf1`, `inf2`) | No Neuron SDK integration |
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
   ```
4. **Price** the host in [`prices.yaml`](prices.yaml) — the hourly rate, the source and date.
5. **Re-render** the cross-host figures:
   ```bash
   python benchmarks/cost.py
   ```
6. **Update this README's host matrix** so the row moves from *not yet measured* to *measured*.

## Figures

- `figures/cost_per_1m_decisions.png` — $/1M decisions at peak sustained throughput, one bar
  per host. **Assumes 100% utilization.** Underestimates the real cost of any always-on host
  that is not saturated.
- `figures/latency_vs_price_frontier.png` — p95 TTFT (at a representative input length and
  concurrency) against $/hour. Pareto frontier is the useful read.

Produced by [`cost.py`](cost.py); regenerate after each new run or price update.
