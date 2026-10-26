# Apple M4 Pro, MLX, Strands Decider 2B (hobson-v21)

**Indicative run** — 50 requests per cell, four input lengths, three concurrencies (`1,8,32`
to match the sibling data drop's saturation point). Short of the standard sweep (which also
runs `jevbench`), but enough to show throughput and latency shape on this backend.

- Host: Apple M4 Pro
- Device: `mlx` (via `strands-decider serve --device mps` — auto-selected mlx on this box)
- Checkpoint: `StrandsAgents/strands-decider-2B-hobson-v21`
- Date: 2026-10-26
- Command:
  ```bash
  python evaluation/decider_bench.py \
      --model m4-pro-mlx=http://127.0.0.1:8000 \
      --input-length 256,512,1024,2048 --concurrency 1,8,32 \
      --num-requests 50 \
      --out benchmarks/runs/m4-pro-mlx-2b-v21
  ```

All 600 requests succeeded; `ok == n`, `refused == 0`, `errors == 0`.

## Numbers

| Input tokens | In flight | p50 latency | p95 latency | Decisions/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 135 ms | 136 ms | 7.73 |
| 256 | 8 | 1.02 s | 1.07 s | 7.82 |
| 256 | 32 | 4.07 s | 4.14 s | 7.83 |
| 512 | 1 | 243 ms | 244 ms | 4.30 |
| 512 | 8 | 1.84 s | 1.89 s | 4.31 |
| 512 | 32 | 7.48 s | 7.59 s | 4.33 |
| 1024 | 1 | 430 ms | 431 ms | 2.32 |
| 1024 | 8 | 3.44 s | 3.44 s | 2.33 |
| 1024 | 32 | 13.9 s | 14.3 s | **0.14** ⚠ |
| 2048 | 1 | 875 ms | 879 ms | 1.15 |
| 2048 | 8 | 6.95 s | 6.98 s | 1.15 |
| 2048 | 32 | 27.8 s | 27.9 s | 1.15 |

⚠ The `1024 tokens, 32 in flight` cell measured `decisions_per_s` at 0.14, 16× below what the
surrounding cells and the p50 latency predict (p50 ≈ 13.9 s with N=32 implies ~2.3 dps). The
cell took 6:26 wall clock vs an expected ~22 s. This is a measurement artifact — likely a
transient slowdown on this box, or MLX recompiling during the sliding window `decider_bench.py`
measures over. The tool warned at the start: `--num-requests 50 is under twice the concurrency
[32]; throughput there is a rough estimate`. Re-run to clean this up before quoting the N=32
cell's throughput specifically. Latency numbers for that cell look consistent and are trusted.

## Read

**Throughput is flat across concurrency** (`throughput_by_concurrency.png`). On this backend
(MLX, batch of 1), queueing more requests does not raise decisions/s; p50 grows linearly with
the number in flight. At 2048 input tokens, concurrency 1 and 32 both deliver ~1.15 decisions/s
but p50 goes from 875 ms to 27.8 s. If a workload benefits from batching, this is not the
serving path for it — this matches the batch-of-1 observation in the sibling cross-instance
benchmarks across every host measured.

**Latency doubles per doubling of input tokens** at `concurrency 1`: 135 → 243 → 430 → 875 ms
for 256 → 512 → 1024 → 2048. Decisions/s halves correspondingly: 7.7 → 4.3 → 2.3 → 1.2.

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
    --num-requests 100 \
    --out benchmarks/runs/m4-pro-mlx-2b-v21
python evaluation/decider_bench.py plot benchmarks/runs/m4-pro-mlx-2b-v21
python benchmarks/plot_throughput.py benchmarks/runs/m4-pro-mlx-2b-v21
```

`--num-requests 100` resolves the "under twice concurrency" warning for N=32 and will produce
a trustworthy throughput number for that cell.
