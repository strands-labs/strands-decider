"""``strands-decider-vllm serve``: the plugin's server (``server``; README, Serving)."""

from __future__ import annotations

import argparse
import os

#: What vLLM reads from the environment when it starts: Model Runner V2, which the mask needs, and
#: its internal sockets on loopback, whatever address the HTTP server binds.
ENVIRONMENT = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_HOST_IP": "127.0.0.1",
    "GLOO_SOCKET_IFNAME": "lo",
    "NCCL_SOCKET_IFNAME": "lo",
}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strands-decider-vllm")
    s = ap.add_subparsers(dest="command", required=True).add_parser(
        "serve",
        help="the System One server on vLLM's engine",
        description="Serve a checkpoint (a directory or a Hub repo id). The server has no "
        "authentication and binds 127.0.0.1 by default: put the deployment's own authentication "
        "(SageMaker's IAM-signed endpoint, an authenticating proxy) in front of any wider bind.",
    )
    s.add_argument("model", help="the checkpoint directory or Hub repo id")
    s.add_argument("--revision", help="the Hub revision (default: main)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--served-model-name", default="strands-decider")
    s.add_argument(
        "--max-length",
        type=int,
        help="refuse longer prompts (default: decision_config.json's max_length_validated)",
    )
    s.add_argument(
        "--max-model-len", type=int, help="the engine's context (default: --max-length + 1)"
    )
    s.add_argument(
        "--max-num-batched-tokens", type=int, help="the step's budget (at least --max-model-len)"
    )
    s.add_argument("--max-num-seqs", type=int)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    s.add_argument("--block-size", type=int)
    s.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="float32 only on a CPU build of vLLM, for tests",
    )
    s.add_argument("--enforce-eager", action="store_true")
    s.add_argument(
        "--max-body-mb",
        type=float,
        help="larger bodies get HTTP 413 (default: 93, four 16 MiB images in base64 and 8 MiB of "
        "text; 8 for a checkpoint that takes no images)",
    )
    s.add_argument(
        "--max-buffered-mb",
        type=float,
        default=256,
        help="the body bytes that requests over 1 MiB may buffer at once, granted in arrival order",
    )
    s.add_argument(
        "--body-read-timeout",
        type=float,
        default=30.0,
        help="seconds a body may go without a chunk (HTTP 408), or wait for --max-buffered-mb (503)",
    )
    s.add_argument(
        "--limit-concurrency",
        type=int,
        default=128,
        help="uvicorn's limit on open connections, /ping included (README, behind SageMaker)",
    )
    s.add_argument(
        "--preprocess-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="threads that render requests, and as many that decode images",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    for key, value in ENVIRONMENT.items():
        os.environ.setdefault(key, value)
    from .server import serve

    serve(a)
