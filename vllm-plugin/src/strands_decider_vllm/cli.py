"""``strands-decider-vllm serve``: the plugin's System One server (``server``; README, Serving)."""

from __future__ import annotations

import argparse
import os

from strands_decider.core import TOKEN_BUDGETS

#: vLLM's internal sockets on loopback, whatever address the HTTP server binds.
_LOOPBACK = {
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
        "--max-image-token-budget",
        type=int,
        choices=TOKEN_BUDGETS,
        help="the largest token budget a request may give an image (default: the checkpoint's, "
        "processor_config.json's max_soft_tokens)",
    )
    s.add_argument(
        "--max-num-batched-tokens", type=int, help="the step's budget (default: --max-length + 1)"
    )
    s.add_argument("--max-num-seqs", type=int)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    s.add_argument("--tensor-parallel-size", type=int, default=1)
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
    args = parser().parse_args(argv)
    for key, value in _LOOPBACK.items():
        os.environ.setdefault(key, value)
    from .server import serve

    serve(args)
