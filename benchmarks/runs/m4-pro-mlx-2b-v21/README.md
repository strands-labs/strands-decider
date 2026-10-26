# Apple M4 Pro, MLX, Strands Decider 2B (hobson-v21)

**Indicative smoke only** — 20 requests per cell, two input lengths, two load levels. Not a full
benchmark run. Use the shape, not the magnitudes, until this is rerun with the standard sweep.

- Host: Apple M4 Pro
- Device: `mlx` (via `strands-decider serve --device mps` — auto-selected mlx on this box)
- Checkpoint: `StrandsAgents/strands-decider-2B-hobson-v21`
- Date: 2026-10-26
- Command:
  ```bash
  python evaluation/decider_bench.py \
      --model local=http://127.0.0.1:8000 \
      --input-length 256,1024 --concurrency 1,4 --num-requests 20 \
      --out benchmarks/runs/m4-pro-mlx-2b-v21
  ```

## Numbers

| Input tokens | In flight | p50 latency | p95 latency | Decisions/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 119.5 ms | 140.0 ms | 8.04 |
| 256 | 4 | 483.2 ms | 524.3 ms | 8.18 |
| 1024 | 1 | 429.7 ms | 470.4 ms | 2.30 |
| 1024 | 4 | 1738.0 ms | 1788.6 ms | 2.29 |

All 80 requests succeeded; `ok == n`, `refused == 0`, `errors == 0`.

## To replace this with a full run

```bash
python evaluation/decider_bench.py \
    --model m4-pro-mlx=http://127.0.0.1:8000 \
    --metrics ttft,jevbench \
    --input-length 256,1024,4096 \
    --concurrency 1,8,32 \
    --num-requests 50 \
    --out benchmarks/runs/m4-pro-mlx-2b-v21
python evaluation/decider_bench.py plot benchmarks/runs/m4-pro-mlx-2b-v21
```
