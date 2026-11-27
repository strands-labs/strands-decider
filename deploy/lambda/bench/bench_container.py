# ruff: noqa: SLF001  -- the bench reads the handler's own module state, as the function logs it
"""The decision function's image, benchmarked in place: on EC2 (docker run), on ECS (a task), or
anywhere the image runs. It imports the Lambda handler, so the model, the shim and its CPU
backend are exactly the function's, and runs a request set through it with a fixed number of
threads (Lambda at 10 GB has 6 vCPUs). It prints one JSON line per request, and with --s3 uploads
them; the first line is the host and the load.

  docker run --rm --cpuset-cpus 0-5 -m 10g -e BENCH_THREADS=6 \\
      -v $PWD:/bench --entrypoint python <image> /bench/bench_container.py \\
      --requests /bench/requests.json --label variants --rounds 3

requests.json is graviton_bench.py's mix: `python graviton_bench.py requests golden_vectors.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--requests", required=True, help="a JSON list of {name, rid, request}, or s3://"
    )
    p.add_argument("--label", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--s3", help="s3://bucket/key for the results; a key ending in / gets a unique name"
    )
    p.add_argument("--module", help="load only libggml-cpu-MODULE.so (a variants image)")
    args = p.parse_args()

    threads = int(os.environ.get("BENCH_THREADS", "6"))
    # The handler sizes llama.cpp's threads by os.cpu_count(), which on a host bigger than the
    # function counts every core, not the container's.
    os.cpu_count = lambda: threads  # type: ignore[assignment]
    root = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
    sys.path.insert(0, root)
    sys.path.append("/var/runtime")  # boto3, as the Lambda base image ships it
    if args.module:
        # The handler loads the libdecision.so beside it, and the shim the modules beside that.
        import shutil
        import tempfile

        work = tempfile.mkdtemp()
        for f in os.listdir(root):
            if (".so" in f and not f.startswith("libggml-cpu-")) or f == "handler.py":
                shutil.copy2(os.path.join(root, f), work, follow_symlinks=False)
        shutil.copy2(os.path.join(root, f"libggml-cpu-{args.module}.so"), work)
        sys.path.insert(0, work)

    requests = json.loads(_read(args.requests))
    started = time.perf_counter()
    import handler  # loads the model

    model = handler._model.result()
    if args.module and model.llama.cpu_backend != f"libggml-cpu-{args.module}.so":
        print(
            json.dumps({"label": args.label, "name": "skipped", "loaded": model.llama.cpu_backend})
        )
        return
    load_s = time.perf_counter() - started
    rows = [
        {
            "label": args.label,
            "name": "host",
            "cpu_part": handler._CPU_PART,
            "cpu_features": handler._CPU_FEATURES,
            "cpu_backend": model.llama.cpu_backend,
            "threads": threads,
            "load_s": round(load_s, 1),
            "instance": _instance_type(),
            "platform": platform.platform(),
            **_memory(),  # after the load: the weights, ggml's repacked copies, KleidiAI's
        }
    ]
    print(json.dumps(rows[0]), flush=True)
    handler.handler(requests[0]["request"], None)  # warm: the first request pages the model in
    for rnd in range(args.rounds):
        for req in requests:
            t = time.perf_counter()
            payload = handler.handler(req["request"], None)
            row = {
                "label": args.label,
                "round": rnd,
                "name": req["name"],
                "rid": req["rid"],
                "wall_ms": round((time.perf_counter() - t) * 1000, 1),
                "latency_ms": payload["latency_ms"],
                "input_tokens": payload["usage"]["input_tokens"],
                "timings_ms": model.llama.last_timings,
                "answers": _numbers(payload["answers"]),
                "cpu_part": handler._CPU_PART,
                "cpu_backend": model.llama.cpu_backend,
                "instance": rows[0]["instance"],
                **_memory(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.s3:
        import boto3

        bucket, key = args.s3[5:].split("/", 1)
        if key.endswith("/"):
            import uuid

            # ecs_bench.py gives each run an id, so it collects that run's results and no other's.
            run = os.environ.get("BENCH_RUN_ID")
            key += f"{run}/" if run else ""
            key += f"{args.label}-{uuid.uuid4()}.jsonl"
        body = "".join(json.dumps(r) + "\n" for r in rows).encode()
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=body)


def _read(path: str) -> str:
    if path.startswith("s3://"):
        import boto3

        bucket, key = path[5:].split("/", 1)
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    with open(path) as f:
        return f.read()


def _numbers(answers: dict) -> dict[str, float]:
    out = {}
    for q, a in answers.items():
        if a.get("type") == "noul":
            out[q] = a["noul"]
        for o, v in (a.get("probabilities") or {}).items():
            out[f"{q}.{o}"] = v
    return out


def _memory() -> dict[str, int]:
    """This process's memory in MB, from /proc/self/status. RssAnon is what the kernel can't
    reclaim, the weights' repacked copies and the state's buffers; RssFile is the mapped GGUF,
    which the page cache can drop; VmHWM is the peak RSS so far (both together)."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key in ("RssAnon", "RssFile", "VmHWM", "VmRSS"):
                    out[f"{key.lower()}_mb"] = int(value.split()[0]) // 1024
    except OSError:
        pass
    return out


def _instance_type() -> str:
    """The EC2 instance type (EC2, ECS on EC2), else the ECS task's launch type and size
    (Fargate, which has no instance metadata), else "?"."""
    import urllib.request

    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            token = r.read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode()
    except Exception:
        pass
    meta = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
    try:
        if meta:
            with urllib.request.urlopen(meta + "/task", timeout=2) as r:
                task = json.load(r)
            return f"{task.get('LaunchType', 'ECS')}:{task.get('Limits', {})}"
    except Exception:
        pass
    return "?"


if __name__ == "__main__":
    main()
