"""``strands-decider-vllm``: the plugin's command line. ``serve`` starts :mod:`strands_decider_vllm.server`."""

from __future__ import annotations

import argparse
import os


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="strands-decider-vllm",
        description="Serve a Strands Decider checkpoint on vLLM 0.33: POST /v1/systemone, images included.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "serve",
        help="the System One server on vLLM's engine in process",
        description="Serve a checkpoint (a local directory or a Hub repo id). The server has no authentication and "
        "binds 127.0.0.1 by default; reach it from elsewhere through an SSH port forward, or put the deployment's "
        "own authentication in front of it (SageMaker's IAM-signed endpoint, an authenticating proxy).",
    )
    s.add_argument("model", help="the checkpoint directory or Hub repo id")
    s.add_argument(
        "--revision", default=None, help="the Hub revision (default: the repo's main branch)"
    )
    s.add_argument("--host", default="127.0.0.1", help="the address to bind (default 127.0.0.1)")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument(
        "--served-model-name", default="strands-decider", help="the model name in responses"
    )
    s.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="the engine's context, in tokens (default: --max-length plus 1, the answer token's position; vLLM "
        "refuses a prompt that fills the context)",
    )
    s.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="refuse prompts longer than this many tokens (default: decision_config.json's max_length_validated); "
        "raising it past the validated length is the operator's decision",
    )
    s.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="the step's token budget (default and minimum: --max-model-len)",
    )
    s.add_argument("--max-num-seqs", type=int, default=None)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument(
        "--no-enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="turn vLLM's prefix cache off (on by default)",
    )
    s.add_argument(
        "--mm-processor-cache-gb",
        type=float,
        default=4.0,
        help="vLLM's cache of preprocessed images, in GiB (0 turns it off)",
    )
    s.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="bfloat16, the checkpoints' and the specification's precision; float32 on a CPU build of vLLM only, "
        "for tests",
    )
    s.add_argument(
        "--block-size", type=int, default=None, help="vLLM's KV-cache block size (default: vLLM's)"
    )
    s.add_argument(
        "--enforce-eager", action="store_true", help="no CUDA graphs (slower; for debugging)"
    )
    s.add_argument(
        "--max-body-mb",
        type=float,
        default=None,
        help="refuse request bodies larger than this, in MiB, with HTTP 413 (default: 187, four 32 MiB image files "
        "in base64 plus 16 MiB of text; 16 for a checkpoint that takes no images)",
    )
    s.add_argument(
        "--limit-concurrency",
        type=int,
        default=128,
        help="uvicorn's limit on open connections and requests, /ping and /health included; past it uvicorn answers "
        "HTTP 503 (default 128)",
    )
    s.add_argument(
        "--preprocess-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="threads that parse, render and decode requests, and so the requests decoded at once (default: 4, or "
        "the CPU count if lower)",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    a = parser().parse_args(argv)
    if a.command == "serve":
        from .server import prepare_environment

        prepare_environment()
        from .server import serve

        serve(a)
