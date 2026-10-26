# Apple M4 Pro, MLX, Strands Decider 2B (hobson-v21)

**Indicative run** — 20 requests per cell, four input lengths, three concurrencies. Short of
the standard sweep (50 requests, three input lengths, three concurrencies with `jevbench`),
but enough to show the throughput and latency shape on this backend.

- Host: Apple M4 Pro
- Device: `mlx` (via `strands-decider serve --device mps` — auto-selected mlx on this box)
- Checkpoint: `StrandsAgents/strands-decider-2B-hobson-v21`
- Date: 2026-10-26
- Command:
  ```bash
  python evaluation/decider_bench.py \
      --model m4-pro-mlx=http://127.0.0.1:8000 \
      --input-length 256,512,1024,2048 --concurrency 1,4,16 \
      --num-requests 20 \
      --out benchmarks/runs/m4-pro-mlx-2b-v21
  ```

## Numbers

| Input tokens | In flight | p50 latency | p95 latency | Decisions/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 136 ms | 139 ms | 7.57 |
| 256 | 4 | 513 ms | 538 ms | 7.90 |
| 256 | 16 | 2.10 s | 2.17 s | 7.49 |
| 512 | 1 | 242 ms | 243 ms | 4.23 |
| 512 | 4 | 916 ms | 943 ms | 4.40 |
| 512 | 16 | 3.72 s | 3.85 s | 4.20 |
| 1024 | 1 | 428 ms | 431 ms | 2.33 |
| 1024 | 4 | 1.71 s | 1.71 s | 2.34 |
| 1024 | 16 | 6.84 s | 6.84 s | 2.34 |
| 2048 | 1 | 867 ms | 872 ms | 1.16 |
| 2048 | 4 | 3.49 s | 3.56 s | 1.15 |
| 2048 | 16 | 13.9 s | 14.1 s | 1.15 |

All 240 requests succeeded; `ok == n`, `refused == 0`, `errors == 0`.

## Read

**Throughput is flat across concurrency** — see `throughput_by_concurrency.png`. On this backend
(MLX, batch of 1), queueing more requests does not raise decisions/s; latency grows roughly
linearly with the number in flight. At 2048 input tokens, concurrency 1 and 16 both deliver
~1.15 decisions/s, but p50 goes from 867 ms to 13.9 s. If a workload benefits from batching,
this is not the serving path for it.

**Latency roughly doubles per doubling of input tokens** at `concurrency 1`: 136 → 242 → 428 →
867 ms for 256 → 512 → 1024 → 2048. Decisions/s halves correspondingly: 7.6 → 4.2 → 2.3 → 1.2.

## Figures

- `throughput_by_concurrency.png` — enhanced chart from `../../plot_throughput.py`, one line per
  concurrency with p50 annotated at each point.
- `ttft_vs_input_length.png`, `ttft_vs_throughput.png`, `throughput_vs_input_length.png` —
  stock figures from `decider_bench.py plot`.

## To replace this with a full run

```bash
python evaluation/decider_bench.py \
    --model m4-pro-mlx=http://127.0.0.1:8000 \
    --metrics ttft,jevbench \
    --input-length 256,512,1024,2048 \
    --concurrency 1,8,32 \
    --num-requests 50 \
    --out benchmarks/runs/m4-pro-mlx-2b-v21
python evaluation/decider_bench.py plot benchmarks/runs/m4-pro-mlx-2b-v21
python benchmarks/plot_throughput.py benchmarks/runs/m4-pro-mlx-2b-v21
```
