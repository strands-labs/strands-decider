# Profile FP16 and INT8 on Apple Silicon

`evaluation/profile_mlx.py` compares the same checkpoint and requests in separate processes. Both models start from FP16 weights; INT8 uses the existing affine, group-size-64 projection quantization after merging LoRA. The output head stays FP32.

## Run a comparison

Create a local FP16 checkpoint using `evaluation/prepare_mlx_dtype.py` as described in [the benchmark setup](m1-quantization.md). From the repository root:

```sh
PYTHONPATH=src .venv/bin/python evaluation/profile_mlx.py \
  --checkpoint /private/tmp/strands-fp16-checkpoint \
  --out reports/profile-forward \
  --lengths 256 1024 --questions 1 8 --warmup 2 --reps 5
```

Choose a fresh output directory for each run. Use AC power and avoid competing GPU workloads. Repeat with `--order int8 fp16` and another output directory to check order and thermal effects. Add `4000` to lengths for the longest benchmark case.

## Read the measurements

- `fp16.json` and `int8.json` contain uninstrumented, synchronized wall-clock samples, peak MLX allocation, and separate instrumented passes.
- `comparison.json` compares baseline medians and phase medians. `_hidden` includes input preparation, decoder execution, and GPU completion. For eight questions, its two events correspond to the shared prefix and batched question suffixes. `_probs` includes selected-state transfer and CPU head scoring. Remaining time includes tokenization, cache preparation, and response assembly.
- `*-python.txt` contains cumulative Python call timings; `.prof` files can be explored with Python's `pstats`. GPU work is asynchronous, so Python call timings are not individual GPU kernel timings.
- `manifest.json` records source revision, script hash, package versions, power, thermal, and swap information. Results are checkpointed after each shape.

Phase timings come from additional passes with explicit synchronization and must not be substituted for the normal latency samples. Independently calculated phase medians need not sum to a median request latency. Peak MLX allocation excludes general host-process memory.

## Capture GPU kernels

Add `--capture` to record one warmed inference per shape and precision, after the timing runs. The parent enables `MTL_CAPTURE_ENABLED=1` before launching the workers. Captures can be large; start with `--lengths 256 --questions 1` and keep traces outside Git.

```sh
PYTHONPATH=src .venv/bin/python evaluation/profile_mlx.py \
  --checkpoint /private/tmp/strands-fp16-checkpoint \
  --out /private/tmp/strands-metal-traces \
  --lengths 256 --questions 1 --capture
```

Open each `.gputrace` in Xcode and profile its replay. Compare aggregate GPU duration and invocation counts for quantized matrix multiplication, dense matrix multiplication, attention, and recurrent-state kernels. Check whether the same request shape changes dispatch count or time spent waiting between kernels. CPU phase measurements locate the regression broadly; GPU traces are needed to attribute it to particular kernels.

The installed Command Line Tools do not currently provide `xctrace` or `metalperftrace`. Full Xcode is needed for the documented interactive trace analysis. MLX capture itself is exposed by the Python API.

INT8 calls MLX `quantized_matmul`; this does not imply that the full model is first dequantized into a separate dense copy. A slower forward pass alone cannot distinguish unpacking cost, kernel selection, arithmetic throughput, or scheduling overhead. Do not describe dequantization as a proven root cause without kernel evidence.

References: [MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html), [Apple Metal workload analysis](https://developer.apple.com/documentation/xcode/analyzing-your-metal-workload).
