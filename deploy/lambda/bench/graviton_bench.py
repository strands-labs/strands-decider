"""Latency of decision functions per Graviton generation, from requests in a checkpoint's
golden_vectors.jsonl, and
whether the builds answer alike.

Lambda picks the host, so the bench samples many execution environments: each round updates an
environment variable on every function's $LATEST, which retires its environments, then sends
`--concurrency` workers through the request set at once, so Lambda starts that many new ones. Each
response's log tail names the host's CPU (cpu_part) and the CPU backend llama.cpp loaded
(cpu_backend); results go to a JSONL file, one line per request, and `--report` summarizes it.

  python graviton_bench.py run --golden golden_vectors.jsonl --out results.jsonl \
      --function baseline=arn:aws:lambda:us-east-1:...:function:name \
      --function variants=arn:... --rounds 4 --concurrency 6
  python graviton_bench.py report results.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

CPU = {"0xd0c": "Graviton2", "0xd40": "Graviton3", "0xd4f": "Graviton4", "0xd84": "Graviton5"}
_LOG = re.compile(r"request (\{.*\})")


def pick_requests(path: str) -> list[dict]:
    """A fixed mix: two ~40-token texts, the text nearest 1,100 tokens and nearest 4,100, and the
    first one-image request."""
    rows = [json.loads(line) for line in open(path)]
    text = [r for r in rows if not r["images"]]
    img = [r for r in rows if len(r["images"]) == 1]
    near = lambda n: min(text, key=lambda r: abs(len(r["state_ids"]) - n))  # noqa: E731
    short = sorted(text, key=lambda r: abs(len(r["state_ids"]) - 40))[:2]
    chosen = [
        ("text40", short[0]),
        ("text40", short[1]),
        ("text1k", near(1100)),
        ("text4k", near(4100)),
    ]
    if img:
        chosen.append(("image1", img[0]))
    out = []
    for name, r in chosen:
        out.append({"name": name, "rid": len(out), "request": r["request"]})
    return out


def _numbers(answers: dict) -> dict[str, float]:
    """Every probability in a response's answers, flattened: question.option -> value."""
    out = {}
    for q, a in answers.items():
        if a.get("type") == "noul":
            out[q] = a["noul"]
        for o, v in (a.get("probabilities") or {}).items():
            out[f"{q}.{o}"] = v
    return out


def invoke(client, arn: str, request: dict) -> tuple[dict | None, dict | None, float, str | None]:
    started = time.time()
    try:
        r = client.invoke(FunctionName=arn, Payload=json.dumps(request).encode(), LogType="Tail")
    except Exception as e:
        return None, None, time.time() - started, repr(e)
    wall = time.time() - started
    body = json.loads(r["Payload"].read() or b"null")
    tail = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
    m = _LOG.search(tail)
    info = json.loads(m.group(1)) if m else None
    err = r.get("FunctionError")
    return body, info, wall, (f"{err}: {str(body)[:300]}" if err else None)


def recycle(client, arn: str, epoch: str) -> None:
    """New environment variable on $LATEST: Lambda retires the environments that had the old one."""
    name = arn.split(":function:")[1].split(":")[0]
    cfg = client.get_function_configuration(FunctionName=name)
    env = dict(cfg.get("Environment", {}).get("Variables", {}))
    env["BENCH_EPOCH"] = epoch
    client.update_function_configuration(FunctionName=name, Environment={"Variables": env})
    client.get_waiter("function_updated_v2").wait(FunctionName=name)


def run(args) -> None:
    import boto3
    from botocore.config import Config

    requests = pick_requests(args.golden)
    functions = [f.split("=", 1) for f in args.function]
    lock = threading.Lock()
    out = open(args.out, "a")
    cfg = Config(
        read_timeout=960,
        connect_timeout=10,
        retries={"max_attempts": 0},
        max_pool_connections=args.concurrency * len(functions) + 4,
    )
    clients = {
        arn: boto3.client("lambda", region_name=arn.split(":")[3], config=cfg)
        for _, arn in functions
    }
    # Invokes never retry (a retry would land on another environment); control calls do.
    control_cfg = Config(retries={"max_attempts": 10, "mode": "standard"})
    control = {
        arn: boto3.client("lambda", region_name=arn.split(":")[3], config=control_cfg)
        for _, arn in functions
    }

    def latest(arn: str) -> str:
        return arn.split(":function:")[0] + ":function:" + arn.split(":function:")[1].split(":")[0]

    def worker(label: str, arn: str, rnd: int, w: int) -> None:
        client = clients[arn]
        for i, req in enumerate([requests[0], *requests]):  # the first is the cold one, kept apart
            body, info, wall, err = invoke(client, latest(arn), req["request"])
            row = {
                "label": label,
                "region": arn.split(":")[3],
                "round": rnd,
                "worker": w,
                "name": "cold" if i == 0 else req["name"],
                "rid": req["rid"],
                "wall_s": round(wall, 2),
                "error": err,
            }
            if info:
                row.update(
                    cpu_part=info.get("cpu_part"),
                    cpu=CPU.get(info.get("cpu_part"), info.get("cpu_part")),
                    cpu_backend=info.get("cpu_backend"),
                    cpu_features=info.get("cpu_features"),
                    latency_ms=info.get("latency_ms"),
                    load_wait_ms=info.get("load_wait_ms"),
                    input_tokens=info.get("input_tokens"),
                    timings_ms=info.get("timings_ms"),
                )
            if body and not err and "answers" in body:
                row["answers"] = _numbers(body["answers"])
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    f"[{label} r{rnd} w{w}] {row['name']:7} {row.get('cpu', '?'):9} "
                    f"{row.get('cpu_backend', '?'):28} {row.get('latency_ms')} ms {err or ''}",
                    flush=True,
                )
            if err and i == 0:
                return  # an environment that never loaded: the round moves on without it

    for rnd in range(args.rounds):
        epoch = f"{int(time.time())}-{rnd}"
        for _, arn in functions:
            recycle(control[arn], latest(arn), epoch)
        with ThreadPoolExecutor(args.concurrency * len(functions)) as pool:
            for label, arn in functions:
                for w in range(args.concurrency):
                    pool.submit(worker, label, arn, rnd, w)


def report(args) -> None:
    # Lambda rows (this file's run), and bench_container.py's from ECS on EC2 and Fargate, whose
    # first line per task is the host's ("host"): its features go with that task's rows.
    rows, seen = [], set()
    for path in args.results:
        for line in open(path):
            if line.strip() and line not in seen:  # the same results collected twice count once
                seen.add(line)
                rows.append(json.loads(line))
    features = {}
    for r in rows:
        r.setdefault("where", "lambda")
        r["cpu"] = CPU.get(r.get("cpu_part"), r.get("cpu") or r.get("cpu_part"))
        if r["name"] == "host":
            features[(r["where"], r["label"], r.get("instance"), r["cpu"])] = r.get("cpu_features")
    for r in rows:
        r.setdefault(
            "cpu_features", features.get((r["where"], r["label"], r.get("instance"), r["cpu"]))
        )
    ok = [
        r
        for r in rows
        if not r.get("error")
        and r.get("latency_ms") is not None
        and r["name"] not in ("cold", "host")
    ]
    groups: dict[tuple, list[float]] = defaultdict(list)
    backends: dict[tuple, set] = defaultdict(set)
    for r in ok:
        groups[(r["where"], r["cpu"], r["name"], r["label"])].append(r["latency_ms"])
        backends[(r["where"], r["cpu"], r["label"])].add(
            f"{r.get('cpu_backend')} [{r.get('cpu_features')}]"
        )
    order = ["baseline", "variants", "kleidiai"]
    labels = sorted(
        {r["label"] for r in ok}, key=lambda x: (order.index(x) if x in order else 9, x)
    )
    # Every request name in the results, the usual mix first in its order, then any others.
    usual = ["text40", "text1k", "text4k", "text10k", "image1"]
    present = {r["name"] for r in ok}
    names = [n for n in usual if n in present] + sorted(present - set(usual))
    print("Median server latency (ms) [n], and speedup against the first label\n")
    print("| where | CPU | request | " + " | ".join(labels) + " |")
    print("|---|---|---|" + "---|" * len(labels))
    for where in sorted({r["where"] for r in ok}):
        for cpu in sorted({r["cpu"] for r in ok if r["where"] == where}):
            for name in names:
                cells, base = [], None
                for label in labels:
                    xs = groups.get((where, cpu, name, label))
                    if not xs:
                        cells.append("-")
                        continue
                    med = statistics.median(xs)
                    base = base or med
                    cells.append(
                        f"{med:,.0f} [{len(xs)}]" + (f" ({base / med:.2f}x)" if med != base else "")
                    )
                if any(c != "-" for c in cells):
                    print(f"| {where} | {cpu} | {name} | " + " | ".join(cells) + " |")
    memory = [r for r in rows if r.get("rssanon_mb") is not None]
    if memory:
        print(
            "\nMemory, MB (bench_container.py): RssAnon after load, the most RssAnon and peak RSS seen\n"
        )
        print("| where | CPU | label | RssAnon after load | most RssAnon | peak RSS (VmHWM) |")
        print("|---|---|---|---|---|---|")
        mem: dict[tuple, dict] = defaultdict(lambda: {"load": [], "anon": 0, "hwm": 0})
        for r in memory:
            m = mem[(r["where"], r["cpu"], r["label"])]
            if r["name"] == "host":
                m["load"].append(r["rssanon_mb"])
            m["anon"] = max(m["anon"], r["rssanon_mb"])
            m["hwm"] = max(m["hwm"], r.get("vmhwm_mb") or 0)
        for (where, cpu, label), m in sorted(mem.items()):
            load = statistics.median(m["load"]) if m["load"] else "-"
            print(f"| {where} | {cpu} | {label} | {load} | {m['anon']} | {m['hwm']} |")
    print("\nCPU backends loaded:")
    for (where, cpu, label), b in sorted(backends.items()):
        print(f"  {where:8} {cpu:9} {label:10} {sorted(map(str, b))}")
    print(f"\nAnswers against {labels[0]}'s for the same request (largest probability difference):")
    ref = {}
    for r in ok:
        if r["label"] == labels[0] and "answers" in r:
            ref.setdefault(r["rid"], r["answers"])
    for label in labels:
        diffs = [
            max(abs(v - ref[r["rid"]].get(k, 0.0)) for k, v in r["answers"].items())
            for r in ok
            if r["label"] == label and "answers" in r and r["rid"] in ref
        ]
        if diffs:
            print(
                f"  {label:12} max {max(diffs):.4f}, mean {statistics.mean(diffs):.4f} over {len(diffs)}"
            )
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n{len(errs)} errors, e.g. {errs[0]['error'][:200]}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--golden", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--function", action="append", required=True, help="label=function ARN")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--concurrency", type=int, default=4)
    q = sub.add_parser("requests", help="write the request mix as JSON, for bench_container.py")
    q.add_argument("golden")
    s = sub.add_parser("report")
    s.add_argument("results", nargs="+")
    args = p.parse_args()
    if args.cmd == "requests":
        print(json.dumps(pick_requests(args.golden)))
        return
    (run if args.cmd == "run" else report)(args)


if __name__ == "__main__":
    sys.exit(main())
