# Apple M4 Pro, MLX, Strands Decider 2B (hobson-v21)

**Indicative run** — 20 requests per cell, four input lengths, three concurrencies. Short of
the standard sweep (50 requests with `jevbench`), but enough to show throughput and latency
shape on this backend.

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

All 240 requests succeeded; `ok == n`, `refused == 0`, `errors == 0`.

## Numbers

| Input tokens | In flight | p50 latency | p95 latency | Decisions/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 136 ms | 137 ms | 7.62 |
| 256 | 4 | 520 ms | 540 ms | 7.80 |
| 256 | 16 | 2.05 s | 2.07 s | 7.81 |
| 512 | 1 | 243 ms | 249 ms | 4.22 |
| 512 | 4 | 920 ms | 946 ms | 4.34 |
| 512 | 16 | 3.70 s | 3.78 s | 4.26 |
| 1024 | 1 | 430 ms | 431 ms | 2.32 |
| 1024 | 4 | 1.72 s | 1.72 s | 2.33 |
| 1024 | 16 | 6.87 s | 6.87 s | 2.33 |
| 2048 | 1 | 871 ms | 876 ms | 1.15 |
| 2048 | 4 | 3.46 s | 3.49 s | 1.16 |
| 2048 | 16 | 13.9 s | 13.9 s | 1.15 |

## Read

**Throughput is flat across concurrency** (`throughput_by_concurrency.png`). On this backend
(MLX, batch of 1), queueing more requests does not raise decisions/s; p50 grows roughly
linearly with the number in flight. At 2048 input tokens, concurrency 1 and 16 both deliver
~1.15 decisions/s but p50 goes from 871 ms to 13.9 s. If a workload benefits from batching,
this is not the serving path for it.

**Latency doubles per doubling of input tokens** at `concurrency 1`: 136 → 243 → 430 → 871 ms
for 256 → 512 → 1024 → 2048. Decisions/s halves correspondingly: 7.6 → 4.2 → 2.3 → 1.2.

## Figures

- `throughput_by_concurrency.png` — closed-loop sweep, one line per concurrency, p50 annotated.
  From `../../plot_throughput.py`.
- `ttft_vs_input_length.png`, `ttft_vs_throughput.png`, `throughput_vs_input_length.png` —
  stock figures from `decider_bench.py plot` (peak-only, less informative than the one above).

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
