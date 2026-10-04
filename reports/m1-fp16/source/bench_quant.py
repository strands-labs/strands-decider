"""Compare MLX floating point / int8 / int4 in isolated processes.

Run from the repository root; see docs/m1-quantization.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def command(*args):
    try:
        return subprocess.check_output(args, text=True, timeout=15).strip()
    except subprocess.TimeoutExpired:
        return "UNAVAILABLE: metadata command timed out: " + " ".join(args)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def worker(args):
    import mlx.core as mx
    import torch
    from bench_local import make_questions, make_state, pct

    from strands_decider.infer import load_engine
    from strands_decider.schema import SystemOneRequest

    torch.set_num_threads(1)
    start = time.perf_counter()
    engine = load_engine(args.checkpoint, device="mlx", quant_bits=None if args.precision == "fp" else int(args.precision))
    result = {"precision": args.precision, "load_s": time.perf_counter() - start,
              "torso_dtype": engine.model.config.torch_dtype,
              "config": vars(engine.model.config), "quantized_paths": engine.quantized_paths,
              "device": mx.device_info(), "latency": [], "quality": []}
    output = args.out / f"{args.precision}.json"
    save(output, result)
    for length in map(int, args.lengths.split(",")):
        for count in map(int, args.questions.split(",")):
            req = SystemOneRequest(state=make_state(engine.tok, length), questions=make_questions(count))
            for _ in range(args.warmup):
                engine.evaluate(req)
            mx.synchronize()
            mx.reset_peak_memory()
            times = []
            for _ in range(args.reps):
                mx.synchronize()
                start = time.perf_counter()
                response = engine.evaluate(req)
                mx.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            row = {"state_tokens_requested": length, "questions": count,
                   "input_tokens": response.usage.input_tokens,
                   "path": "single" if count == 1 else "prefix",
                   "median_ms": statistics.median(times), "p90_ms": pct(times, .9),
                   "samples_ms": times, "mlx_peak_gib": mx.get_peak_memory() / 2**30,
                   "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
                   "swap": command("sysctl", "vm.swapusage")}
            result["latency"].append(row)
            save(output, result)
            print(args.precision, row, flush=True)
    sys.path.insert(0, str(args.jevbench.resolve()))
    from jevbench.adapters.base import build_question
    from jevbench.metrics import brier_score, ece_top_label
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    dataset = args.jevbench / "datasets/public/original.jsonl"
    result["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    for task in load_jsonl(str(dataset)):
        req = SystemOneRequest(state=task.state, questions={"decision": build_question(task)})
        answer = engine.evaluate(req).answers["decision"]
        probs = {"yes": answer.noul, "no": 1 - answer.noul} if answer.type == "noul" else answer.probabilities
        scored = score_task(probs, task)
        if not scored["valid"]:
            raise ValueError(f"Invalid prediction for {task.id}: {scored}")
        row = {"id": task.id, "type": answer.type, **scored,
               "brier": brier_score(scored["probs"], str(task.expected), task.labels)}
        result["quality"].append(row)
        save(output, result)
    rows = result["quality"]
    result["metrics"] = {"n": len(rows), "accuracy": statistics.mean(r["correct"] for r in rows),
        "brier": statistics.mean(r["brier"] for r in rows),
        "ece": ece_top_label([(max(r["probs"].values()), r["correct"]) for r in rows])}
    save(output, result)
    print(args.precision, result["metrics"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="StrandsAgents/strands-decider-2B-hobson-v19")
    ap.add_argument("--jevbench", type=Path, default=Path(".bench/jevbench"))
    ap.add_argument("--out", type=Path, default=Path("reports/m1-quant"))
    ap.add_argument("--lengths", default="256,1024,4000")
    ap.add_argument("--questions", default="1,8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--order", default="fp,8,4", help="precision order, e.g. 4,8,fp for thermal cross-check")
    ap.add_argument("--precision", choices=["fp", "8", "4"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.reps < 1 or args.warmup < 1 or any(int(n) < 1 for n in (args.lengths + "," + args.questions).split(",")):
        ap.error("reps, warmup, lengths and questions must be positive")
    if sorted(args.order.split(",")) != ["4", "8", "fp"]:
        ap.error("order must contain fp,8,4 exactly once")
    if not args.precision and (args.out / "manifest.json").exists():
        ap.error("output directory already contains a run; choose a fresh --out")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.precision:
        worker(args)
        return
    manifest = {"platform": platform.platform(), "machine": platform.machine(),
        "commit": command("git", "rev-parse", "HEAD"), "diff": command("git", "diff"),
        "jevbench_commit": command("git", "-C", str(args.jevbench), "rev-parse", "HEAD"),
        "packages": {p: importlib.metadata.version(p) for p in ["mlx", "mlx-lm", "torch", "transformers", "peft"]},
        "args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "power": command("pmset", "-g", "batt"), "swap_before": command("sysctl", "vm.swapusage")}
    save(args.out / "manifest.json", manifest)
    for precision in args.order.split(","):
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--precision", precision], check=True,
                       env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
    results = {p: json.loads((args.out / f"{p}.json").read_text()) for p in ["fp", "8", "4"]}
    comparison = []
    for p, data in results.items():
        for baseline, row in zip(results["fp"]["latency"], data["latency"], strict=True):
            comparison.append({"precision": p, **{k: v for k, v in row.items() if k not in ("samples_ms", "swap")},
                               "speedup_vs_fp": baseline["median_ms"] / row["median_ms"]})
        pairs = list(zip(results["fp"]["quality"], data["quality"], strict=True))
        data["metrics"]["changed_predictions_vs_fp"] = sum(a["predicted"] != b["predicted"] for a, b in pairs)
        data["metrics"]["max_probability_delta_vs_fp"] = max(abs(a["probs"][k] - b["probs"][k]) for a, b in pairs for k in a["probs"])
    with (args.out / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    save(args.out / "quality-summary.json", {p: d["metrics"] for p, d in results.items()})
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
