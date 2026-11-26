"""A checkpoint's golden_vectors.jsonl through the decision image's llama.cpp, one CPU module at
a time: the agreement tests/core_conformance.py defines (top-1, mean |dp|), per module.

  python conformance_container.py --golden golden_vectors.jsonl --lib /var/task/libdecision.so \\
      --module graviton4 --max-state 5000 --threads 32 --s3 s3://bucket/conformance/

--module NAME copies the libraries beside --lib into a directory with only libggml-cpu-NAME.so,
so the shim can load nothing else (a static --lib has no modules: leave --module out). Text
rows only; their state and question token ids are the golden file's own.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", required=True, help="a golden_vectors.jsonl, or s3://")
    p.add_argument("--lib", default="/var/task/libdecision.so")
    p.add_argument("--module", help="force libggml-cpu-MODULE.so")
    p.add_argument("--model-dir", default="/opt/model")
    p.add_argument(
        "--gguf", help="another decoder GGUF than MODEL_DIR/model.gguf (e.g. requantized)"
    )
    p.add_argument("--max-state", type=int, default=5000, help="skip rows with longer states")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--label", default="")
    p.add_argument("--s3", help="s3://bucket/prefix/ for the results")
    args = p.parse_args()

    sys.path.insert(0, os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
    sys.path.append("/var/runtime")
    import numpy as np  # the image installs it in LAMBDA_TASK_ROOT

    from strands_decider.backends.llama_cpp import LlamaCppBackend
    from strands_decider.core import DecisionConfig, load_readout
    from strands_decider.core.readout import code_probabilities, readout_logits

    lib = args.lib
    if args.module:
        src = os.path.dirname(os.path.abspath(lib))
        work = tempfile.mkdtemp()
        for f in os.listdir(src):
            if ".so" in f and not f.startswith("libggml-cpu-"):
                shutil.copy2(os.path.join(src, f), work, follow_symlinks=False)
        shutil.copy2(os.path.join(src, f"libggml-cpu-{args.module}.so"), work)
        lib = os.path.join(work, "libdecision.so")

    config = DecisionConfig.load(args.model_dir)
    weight = load_readout(args.model_dir)
    backend = LlamaCppBackend(
        lib, args.gguf or os.path.join(args.model_dir, "model.gguf"), threads=args.threads
    )
    loaded = backend.cpu_backend
    if args.module and loaded != f"libggml-cpu-{args.module}.so":
        print(json.dumps({"module": args.module, "skipped": f"loaded {loaded}"}))
        return

    rows = [json.loads(line) for line in _read(args.golden).splitlines() if line.strip()]
    started = time.perf_counter()
    n = top1 = 0
    dps: list[float] = []
    per_question: dict[str, list[float]] = {}
    for i, r in enumerate(rows):
        if r["images"] or len(r["state_ids"]) > args.max_state:
            continue
        names = list(r["questions"])
        hidden = backend.forward(r["state_ids"], [r["questions"][q]["question_ids"] for q in names])
        logits = readout_logits(weight, np.asarray(hidden))
        for name, z in zip(names, logits, strict=True):
            golden = np.asarray(r["questions"][name]["probabilities"], dtype=np.float64)
            k = len(golden)
            got = code_probabilities(
                z[:k], config.final_logit_softcapping, config.temperature
            ).astype(np.float64)
            n += 1
            top1 += int(np.argmax(got) == np.argmax(golden))
            dps.append(float(np.max(np.abs(got - golden))))
            per_question[f"{i}:{name}"] = [round(float(x), 6) for x in got]
    out = {
        "label": args.label,
        "module": args.module or loaded,
        "gguf": os.path.basename(args.gguf) if args.gguf else "model.gguf",
        "cpu_part": _cpu_part(),
        "questions": n,
        "top1": round(top1 / n, 4) if n else None,
        "mean_abs_dp": round(float(np.mean(dps)), 5) if dps else None,
        "max_abs_dp": round(float(np.max(dps)), 5) if dps else None,
        "seconds": round(time.perf_counter() - started, 1),
        "threads": args.threads,
    }
    print(json.dumps(out), flush=True)
    if args.s3:
        import uuid

        import boto3

        bucket, prefix = args.s3[5:].split("/", 1)
        body = json.dumps({**out, "per_question": per_question}).encode()
        key = f"{prefix}{args.label or 'run'}-{out['module']}-{out['gguf']}-{uuid.uuid4()}.json"
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)


def _read(path: str) -> str:
    if path.startswith("s3://"):
        import boto3

        bucket, key = path[5:].split("/", 1)
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    with open(path) as f:
        return f.read()


def _cpu_part() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next(line for line in f if line.startswith("CPU part")).split(":")[1].strip()
    except (OSError, StopIteration):
        return "?"


if __name__ == "__main__":
    main()
