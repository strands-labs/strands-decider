"""Isolated FP16/INT8 latency, phase, Python, and optional Metal profiling."""
from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=20)
    return (result.stdout + result.stderr).strip()


def worker(args):
    import mlx.core as mx
    import torch
    from bench_local import make_questions, make_state
    from strands_decider.infer import load_engine
    from strands_decider.schema import SystemOneRequest

    torch.set_num_threads(1)
    engine = load_engine(args.checkpoint, device="mlx",
                         quant_bits=8 if args.precision == "int8" else None)
    if engine.model.config.torch_dtype != "float16":
        raise ValueError("Both arms require a float16 checkpoint; use prepare_mlx_dtype.py")
    result = {"precision": args.precision, "dtype": "float16", "device": mx.device_info(),
              "quantized_paths": engine.quantized_paths, "shapes": []}
    output = args.out / f"{args.precision}.json"
    for length in args.lengths:
        for count in args.questions:
            req = SystemOneRequest(state=make_state(engine.tok, length), questions=make_questions(count))
            for _ in range(args.warmup):
                engine.evaluate(req)
            mx.synchronize()
            mx.reset_peak_memory()
            samples = []
            for _ in range(args.reps):
                mx.synchronize()
                start = time.perf_counter()
                engine.evaluate(req)
                mx.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            row = {"state_tokens": length, "questions": count, "samples_ms": samples,
                   "median_ms": statistics.median(samples), "peak_mlx_gib": mx.get_peak_memory() / 2**30}
            # Separate instrumented passes: never mix these timings into the baseline.
            phases = []
            originals = {name: getattr(engine, name) for name in ("_hidden", "_probs")}
            events = []
            def wrap(name, fn):
                def measured(*a, **kw):
                    mx.synchronize()
                    start = time.perf_counter()
                    value = fn(*a, **kw)
                    mx.synchronize()
                    events.append({"phase": name, "ms": (time.perf_counter() - start) * 1000,
                                   "shape": [len(a[0]), max(map(len, a[0]))] if name == "_hidden" else None})
                    return value
                return measured
            try:
                for name, fn in originals.items():
                    setattr(engine, name, wrap(name, fn))
                for _ in range(args.reps):
                    events = []
                    start = time.perf_counter()
                    engine.evaluate(req)
                    mx.synchronize()
                    total = (time.perf_counter() - start) * 1000
                    phases.append({"total_ms": total, "events": events,
                                   "other_ms": total - sum(e["ms"] for e in events)})
            finally:
                for name, fn in originals.items():
                    setattr(engine, name, fn)
            row["instrumented_passes"] = phases
            stem = f"{args.precision}-{length}-q{count}"
            profiler = cProfile.Profile()
            profiler.runcall(engine.evaluate, req)
            profiler.dump_stats(str(args.out / f"{stem}.prof"))
            with (args.out / f"{stem}-python.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(60)
            if args.capture:
                trace = (args.out / f"{stem}.gputrace").resolve()
                mx.synchronize()
                mx.metal.start_capture(str(trace))
                try:
                    engine.evaluate(req)
                    mx.synchronize()
                finally:
                    mx.metal.stop_capture()
                row["metal_trace"] = str(trace)
            result["shapes"].append(row)
            save(output, result)
            print(args.precision, length, count, round(row["median_ms"], 2), "ms", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--capture", action="store_true", help="Save a Metal GPU trace per shape (large files)")
    ap.add_argument("--order", nargs=2, choices=["fp16", "int8"], default=["fp16", "int8"])
    ap.add_argument("--precision", choices=["fp16", "int8"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if min(args.lengths + args.questions + [args.warmup, args.reps]) < 1:
        ap.error("Lengths, questions, warmup and reps must be positive")
    if set(args.order) != {"fp16", "int8"}:
        ap.error("Order must contain each precision once")
    if args.precision:
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "manifest.json", {
        "args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(), "commit": command("git", "rev-parse", "HEAD"),
        "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "torch"]},
        "power": command("pmset", "-g", "batt"), "thermal": command("pmset", "-g", "therm"),
        "swap": command("sysctl", "vm.swapusage")})
    for precision in args.order:
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--precision", precision],
                       check=True, env={**os.environ, "MTL_CAPTURE_ENABLED": "1",
                                        "TOKENIZERS_PARALLELISM": "false"})
    results = {p: json.loads((args.out / f"{p}.json").read_text()) for p in args.order}
    comparison = []
    for fp, quant in zip(results["fp16"]["shapes"], results["int8"]["shapes"], strict=True):
        entry = {"state_tokens": fp["state_tokens"], "questions": fp["questions"],
                 "int8_over_fp16": quant["median_ms"] / fp["median_ms"]}
        for precision, row in [("fp16", fp), ("int8", quant)]:
            passes = row["instrumented_passes"]
            entry[precision] = {"baseline_ms": row["median_ms"],
                "hidden_ms": statistics.median(sum(e["ms"] for e in p["events"] if e["phase"] == "_hidden") for p in passes),
                "head_ms": statistics.median(sum(e["ms"] for e in p["events"] if e["phase"] == "_probs") for p in passes),
                "other_ms": statistics.median(p["other_ms"] for p in passes)}
        comparison.append(entry)
    save(args.out / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
