# Apple M4 Pro, MLX, Strands Decider 2B (hobson-v21)

**Indicative run** — 15 requests per cell, four input lengths, three concurrencies and four rps
targets. Short of the standard sweep (50 requests, three input lengths, three concurrencies with
`jevbench`), but enough to show throughput, latency and goodput shape on this backend.

- Host: Apple M4 Pro
- Device: `mlx` (via `strands-decider serve --device mps` — auto-selected mlx on this box)
- Checkpoint: `StrandsAgents/strands-decider-2B-hobson-v21`
- Date: 2026-10-26
- Command:
  ```bash
  python evaluation/decider_bench.py \
      --model m4-pro-mlx=http://127.0.0.1:8000 \
      --input-length 256,512,1024,2048 \
      --concurrency 1,4,16 --rps 1,2,4,8 \
      --num-requests 15 \
      --out benchmarks/runs/m4-pro-mlx-2b-v21
  ```

All 420 requests succeeded; `ok == n`, `refused == 0`, `errors == 0`.

## Closed-loop numbers (`--concurrency`)

| Input tokens | In flight | p50 latency | p95 latency | Decisions/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 143 ms | 150 ms | 7.12 |
| 256 | 4 | 544 ms | 579 ms | 7.36 |
| 256 | 16 | 2.11 s | 2.19 s | 7.47 |
| 512 | 1 | 243 ms | 246 ms | 4.33 |
| 512 | 4 | 920 ms | 945 ms | 4.34 |
| 512 | 16 | 3.70 s | 3.70 s | 4.32 |
| 1024 | 1 | 430 ms | 431 ms | 2.32 |
| 1024 | 4 | 1.72 s | 1.72 s | 2.33 |
| 1024 | 16 | 6.87 s | 6.87 s | 2.33 |
| 2048 | 1 | 876 ms | 927 ms | 1.14 |
| 2048 | 4 | 3.47 s | 3.50 s | 1.16 |
| 2048 | 16 | 13.8 s | 13.9 s | 1.15 |

## Open-loop numbers (`--rps`)

Target rps vs delivered decisions/s shows whether the server kept up: delivered below target
means the server fell behind and p95 is queueing-dominated.

| Input tokens | Target rps | p50 latency | p95 latency | Delivered dps |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 160 ms | 234 ms | 1.02 |
| 256 | 2 | 145 ms | 260 ms | 2.75 |
| 256 | 4 | 148 ms | 348 ms | 4.09 |
| 256 | 8 | 277 ms | 500 ms | 7.53 |
| 512 | 1 | 259 ms | 385 ms | 1.10 |
| 512 | 2 | 258 ms | 438 ms | 2.11 |
| 512 | 4 | 290 ms | 671 ms | 2.94 |
| 512 | 8 | 977 ms | 2.12 s | 4.02 |
| 1024 | 1 | 477 ms | 933 ms | 0.96 |
| 1024 | 2 | 830 ms | 1.58 s | 1.65 |
| 1024 | 4 | 2.68 s | 3.70 s | 2.07 |
| 1024 | 8 | 3.66 s | 5.51 s | 2.03 |
| 2048 | 1 | 916 ms | 2.75 s | 0.54 |
| 2048 | 2 | 2.60 s | 6.11 s | 1.01 |
| 2048 | 4 | 6.46 s | 11.3 s | 1.04 |
| 2048 | 8 | 7.74 s | 13.0 s | 1.02 |

## Read

**Throughput is flat across concurrency** (`throughput_by_concurrency.png`). On this backend
(MLX, batch of 1), queueing more requests does not raise decisions/s; p50 grows roughly
linearly with the number in flight. At 2048 input tokens, concurrency 1 and 16 both deliver
~1.15 decisions/s but p50 goes from 876 ms to 13.8 s. If a workload benefits from batching,
this is not the serving path for it.

**Goodput ceiling matches the closed-loop peak** (`rps_ramp.png`). The sustainable rps at each
input length is roughly the measured peak decisions/s:

| Input tokens | Peak dps (closed loop) | Sustainable rps (open loop, p95 < 1 s) |
| ---: | ---: | ---: |
| 256 | 7.5 | ~8 |
| 512 | 4.3 | ~4 |
| 1024 | 2.3 | ~2 |
| 2048 | 1.2 | ~1 |

Past that rps the delivered dps caps at the ceiling, requests queue, and p95 jumps by 2–10×.

**Latency doubles per doubling of input tokens** at `concurrency 1`: 143 → 243 → 430 → 876 ms
for 256 → 512 → 1024 → 2048. Decisions/s halves correspondingly: 7.1 → 4.3 → 2.3 → 1.1.

## Figures

- `throughput_by_concurrency.png` — closed-loop sweep, one line per concurrency, p50 annotated.
  From `../../plot_throughput.py`.
- `rps_ramp.png` — open-loop goodput curve, p95 vs target rps, one line per input length,
  delivered dps annotated. From `../../plot_rps_ramp.py`.
- `ttft_vs_input_length.png`, `ttft_vs_throughput.png`, `throughput_vs_input_length.png` —
  stock figures from `decider_bench.py plot` (peak-only, less informative than the two above).

## To replace this with a full run

```bash
python evaluation/decider_bench.py \
    --model m4-pro-mlx=http://127.0.0.1:8000 \
    --metrics ttft,jevbench \
    --input-length 256,512,1024,2048 \
    --concurrency 1,8,32 --rps 1,2,4,8 \
    --num-requests 50 \
    --out benchmarks/runs/m4-pro-mlx-2b-v21
python evaluation/decider_bench.py plot benchmarks/runs/m4-pro-mlx-2b-v21
python benchmarks/plot_throughput.py benchmarks/runs/m4-pro-mlx-2b-v21
python benchmarks/plot_rps_ramp.py benchmarks/runs/m4-pro-mlx-2b-v21
```
