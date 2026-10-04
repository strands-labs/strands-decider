# M1 quantization experiment

Compare the checkpoint's floating-point MLX torso with affine int8 and int4 (group size 64).
LoRA is merged in fp32 before quantization. An explicit projection allowlist leaves embeddings,
normalizations, convolutions, DeltaNet dynamics and the separate fp32 pointer head unchanged.
`load_engine(..., device="mlx", quant_bits=4|8, quant_group_size=64)` enables it.
The normal engine defaults remain unchanged. Quantization is not yet exposed by `ask`/`serve`.

## Reproduce

Requires Apple silicon, Python 3.12 and sufficient free disk space for weights (~4 GB),
the environment and swap. Use AC power and close competing GPU workloads.

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r evaluation/requirements-m1.txt
uv pip install --python .venv/bin/python --no-deps -e .
git clone https://github.com/fstandhartinger/jevbench .bench/jevbench
# Use the JevBench commit recorded in the run manifest to reproduce a saved run.
.venv/bin/python evaluation/bench_quant.py --out reports/m1-quant
# Short first pass:
.venv/bin/python evaluation/bench_quant.py --lengths 256,1024 --reps 3 --out reports/m1-short
```

Each precision loads in a fresh process. Every shape gets two warmups, then five timed,
explicitly synchronized end-to-end engine calls. Multiple questions use shared-prefix inference.
The requested state length is distinct from actual input tokens (prompt overhead and truncation);
both are recorded. Load and quantization time are excluded from warm latency and reported separately.
MLX peak allocation excludes CPU allocations; RSS peak includes loading and is a process high-water
mark. Swap is machine-wide, not attributable exclusively to this process.

Outputs include per-precision JSON saved incrementally, all raw latency samples, a CSV with
speedups, a quality summary, and a manifest with source revision/diff, packages, dataset revision,
power and swap state. Do not reuse an output directory for distinct experiments.

Quality uses all 72 JevBench `original` tasks, upstream request conversion and scoring. Accuracy is
argmax exact-label accuracy, including score tasks; Brier is the multiclass sum (two-class sum for
noul); ECE uses ten confidence bins. Checkpoint temperatures remain enabled and are recorded in
config. These calibrated results are not directly comparable to the report's pre-temperature
numbers. Probability deltas compare against MLX floating point, not a CPU fp32 oracle.

The input report's supporting scripts/patch were not supplied. This is an independent implementation,
not a byte-for-byte reproduction. The report's hypothesis is tested, not assumed: quantization may
reduce memory without accelerating the large forward-pass matmuls. A single M1 run cannot establish
that quantization is useless on every Mac. Repeat in reversed order if small speed differences could
be caused by the fanless Air's thermal throttling.

## Validation

```sh
.venv/bin/pytest tests/test_quantization.py tests/test_mlx_engine.py -q
```

Tests use tiny generated checkpoints, require Metal GPU access, and download no model weights.
They exercise quantized single/shared-prefix inference, protected modules, fp32 head and invalid
configuration. They are correctness checks, not performance evidence for the production model.

## Explicit fp16 comparison

The published v19 checkpoint specifies **bfloat16**, although the supplied report calls its baseline
fp16. The benchmark records the actual dtype. To also test fp16, create a local configuration override:

```sh
.venv/bin/python evaluation/prepare_mlx_dtype.py /path/to/huggingface/snapshot .bench/fp16-checkpoint --dtype float16
.venv/bin/python evaluation/bench_quant.py --checkpoint .bench/fp16-checkpoint --out reports/m1-fp16
```

The helper links assets to the source snapshot and copies its config, leaving the Hub cache unchanged.
It excludes the original checksum manifest because the config differs. Keep the source snapshot
available, or recreate the override after moving machines. The same merge-then-quantize ordering
applies to this fp16 experiment.

If Desktop cloud storage offloads files and Git/imports stall, run an equivalent local checkout outside
cloud-managed folders with `PYTHONPATH` pointing to that checkout's `src`. Save the source snapshot,
revision and results together. The M1 run used `/private/tmp/strands-decider-m1-run` for this reason.
