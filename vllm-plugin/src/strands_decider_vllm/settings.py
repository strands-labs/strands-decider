"""What the server starts with: the CLI's flags checked against the checkpoint, and vLLM's engine
arguments for them."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_decider import core
from strands_decider.core.images import MAX_IMAGE_BYTES
from vllm.engine.arg_utils import AsyncEngineArgs

from . import ARCHITECTURES, SERVER_CONFIG_KEY

#: The checkpoint's files the server reads, before the engine loads the rest.
_CONFIG_FILES = ("config.json", "decision_config.json", "processor_config.json", "tokenizer.json")
_TEXT_BYTES = 8 * 2**20
#: One of the core's largest image files in base64, with room for a data URI and quotes.
_IMAGE_BYTES = 4 * math.ceil(MAX_IMAGE_BYTES / 3) + 64


@dataclass(frozen=True)
class Settings:
    config: core.DecisionConfig
    tokenizer: Any
    engine_args: AsyncEngineArgs
    served_name: str
    max_length: int
    max_body: int
    max_buffered: int
    body_read_timeout: float
    preprocess_workers: int

    @property
    def takes_images(self) -> bool:
        return "image" in self.config.inputs


def settings(args: argparse.Namespace) -> Settings:
    """The server's settings, or SystemExit for a checkpoint it can't serve."""
    try:
        path, revision = _config_files(args.model, args.revision)
        config = core.DecisionConfig.load(path)
        tokenizer = core.load_tokenizer(path)
        if "image" in config.inputs:
            core.check_image_processor(path)
        base = json.loads((path / "config.json").read_text("utf-8"))["architectures"][0]
        if base not in ARCHITECTURES:
            raise ValueError(f"config.json names {base}, which is no Strands Decider layout")
    except (OSError, ValueError) as e:  # pydantic's ValidationError is a ValueError
        raise SystemExit(f"strands-decider-vllm serve: {args.model}: {e}") from None
    max_length = args.max_length or config.max_length_validated
    takes_images = "image" in config.inputs
    if args.max_body_mb:
        max_body = int(args.max_body_mb * 2**20)
    else:  # 8 MiB of text, and four of the largest images (93 MiB)
        max_body = _TEXT_BYTES + (4 * _IMAGE_BYTES if takes_images else 0)
    return Settings(
        config=config,
        tokenizer=tokenizer,
        engine_args=_engine_args(args, base, revision, max_length, takes_images),
        served_name=args.served_model_name,
        max_length=max_length,
        max_body=max_body,
        max_buffered=max(int(args.max_buffered_mb * 2**20), max_body),  # room for any one body
        body_read_timeout=args.body_read_timeout,
        preprocess_workers=args.preprocess_workers,
    )


def _config_files(model: str, revision: str | None) -> tuple[Path, str | None]:
    """The directory of the checkpoint's config files, and the revision the engine loads: for a Hub
    repo, the commit those files were read at."""
    if Path(model).is_dir():
        return Path(model), revision
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, revision=revision, allow_patterns=list(_CONFIG_FILES)))
    return path, path.name  # .../snapshots/<commit>


def _engine_args(
    args: argparse.Namespace, base: str, revision: str | None, max_length: int, takes_images: bool
) -> AsyncEngineArgs:
    # vLLM refuses a prompt that fills the context: it leaves no position for the answer.
    max_model_len = max_length + 1
    # prefix_bidir replaces the "vision" mode of the 12B, 26B-A4B and 31B bases (model.py).
    vision_off = {"use_bidirectional_attention": None}
    multimodal = base != "Gemma4TextModel"
    return AsyncEngineArgs(
        model=args.model,
        revision=revision,
        served_model_name=[args.served_model_name],
        dtype="bfloat16",
        seed=0,
        hf_overrides={
            "architectures": [ARCHITECTURES[base][0]],
            **({"text_config": vision_off} if multimodal else vision_off),
        },
        max_model_len=max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens or max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256",
        generation_config="vllm",
        logprobs_mode="processed_logprobs",
        max_logprobs=255,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        # vLLM's custom all-reduce failed at startup at TP=2 on the H100 nodes tested.
        disable_custom_all_reduce=args.tensor_parallel_size > 1,
        additional_config={SERVER_CONFIG_KEY: True},
        # No count of images: the prompt's length bounds them. ({} is vLLM's default.)
        limit_mm_per_prompt=(
            {"image": max_model_len if takes_images else 0, "video": 0, "audio": 0}
            if multimodal
            else {}
        ),
    )
