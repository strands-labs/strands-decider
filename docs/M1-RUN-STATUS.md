# M1 benchmark checkpoint

Branch: `codex/m1-inference-quant-bench`.

Implementation and partial results are committed locally. No push has been requested.
Hardware: Apple M1 MacBook Air, 16 GB RAM, macOS 26.5.1.

## Current run

`reports/m1-local/`: checkpoint bfloat16 baseline, affine int8, affine int4.
Bfloat16, int8 and int4 latency and quality evaluations are complete.
The checkpoint-default experiment is complete. The explicit fp16 follow-up is starting; final conclusions remain pending.
The JSON files are saved after every completed shape and every quality task.
The early bfloat16 measurements began on battery; later runs are on AC power.
An explicit fp16 comparison is starting in `reports/m1-fp16`.

Execution uses `/private/tmp/strands-decider-m1-run` because Desktop cloud placeholders
initially stalled reads. The source changes and exact benchmark source snapshot are also
saved in this repository. Git access has been restored here.

## Resume if interrupted

Read per-precision JSON files before rerunning. Do not overwrite an existing run directory.
The standard command repeats all three precisions in fresh processes:

```sh
.venv/bin/python evaluation/bench_quant.py --out reports/m1-resumed
```

To finish only a missing precision into the current directory (advanced worker mode):

```sh
.venv/bin/python evaluation/bench_quant.py --precision 4 --out reports/m1-local
```

The worker writes that precision's JSON, but the top-level runner is what generates the
combined CSV and quality-summary JSON. Preserve already-completed JSON files before using
the same precision name. Full setup and fp16 instructions are in `docs/m1-quantization.md`.
The saved dataset and its upstream revision/license are in `evaluation/fixtures/jevbench-original`.

Validation so far: 24 MLX/quantization tests passed on the M1 GPU; lint passed.
