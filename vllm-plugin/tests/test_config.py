"""The startup refusals on configs vLLM builds from a checkpoint (its config.json and tokenizer), so
that a renamed vLLM attribute fails here. Set STRANDS_DECIDER_TEST_CHECKPOINT to run them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

CHECKPOINT = os.environ.get("STRANDS_DECIDER_TEST_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="STRANDS_DECIDER_TEST_CHECKPOINT is not set")
model = pytest.importorskip("strands_decider_vllm.model")
from strands_decider_vllm import ARCHITECTURES  # noqa: E402


def flags():
    """The README's serving flags for the checkpoint's layout, at an 8,192-token context."""
    config = json.loads((Path(CHECKPOINT) / "config.json").read_text())
    vision_off = {"use_bidirectional_attention": None}
    return dict(
        hf_overrides={
            "architectures": [ARCHITECTURES[config["architectures"][0]][0]],
            **({"text_config": vision_off} if "text_config" in config else vision_off),
        },
        dtype="bfloat16",
        max_model_len=8192,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        logprobs_mode="processed_logprobs",
        max_logprobs=255,
    )


@pytest.fixture(autouse=True)
def _model_runner_v2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def engine_config(**overrides):
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(model=CHECKPOINT, **{**flags(), **overrides}).create_engine_config()


def refused(config):
    with pytest.raises(ValueError, match="cannot serve with this configuration") as e:
        model.check_config(config)
    return str(e.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"enable_prefix_caching": False, "prefix_caching_hash_algo": "xxhash"},
        {"prefix_caching_hash_algo": "sha256_cbor"},
        {"long_prefill_token_threshold": 8192},  # can't cut a prompt of max_model_len tokens
        {"kv_cache_dtype": "bfloat16"},
        {"compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}},
        {"compilation_config": {"cudagraph_mode": "PIECEWISE"}},
        {"enforce_eager": True},
    ],
)
def test_the_serving_flags_pass(overrides):
    model.check_config(engine_config(**overrides))


@pytest.mark.parametrize(
    "overrides, problem",
    [
        ({"enable_chunked_prefill": True}, "--no-enable-chunked-prefill"),
        (
            {"max_num_scheduled_tokens": 4096},
            "--max-num-scheduled-tokens 4096 < --max-model-len 8192",
        ),
        ({"long_prefill_token_threshold": 256}, "--long-prefill-token-threshold 256"),
        ({"compilation_config": {"cudagraph_mode": "FULL"}}, "cudagraph_mode FULL"),
        ({"logprobs_mode": "raw_logprobs"}, "--logprobs-mode raw_logprobs"),
        ({"dtype": "float16"}, "--dtype torch.float16"),
        ({"kv_cache_dtype": "fp8"}, "--kv-cache-dtype fp8"),
        ({"enable_prompt_embeds": True}, "--enable-prompt-embeds"),
        ({"prefix_caching_hash_algo": "xxhash"}, "--prefix-caching-hash-algo"),
    ],
)
def test_unsafe_flags_are_refused(overrides, problem):
    assert problem in refused(engine_config(**overrides))


@pytest.mark.parametrize(
    "where, name, value, problem",
    [
        ("model_config", "quantization", "fp8", "--quantization fp8"),
        # The 12B, 26B-A4B and 31B bases' "vision" mode, left on.
        ("model_config", "is_mm_prefix_lm", True, "PrefixLM"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline parallelism"),
        ("parallel_config", "prefill_context_parallel_size", 2, "context parallelism"),
        ("cache_config", "kv_sharing_fast_prefill", True, "--kv-sharing-fast-prefill"),
        (None, "speculative_config", object(), "speculative decoding"),
        (None, "kv_transfer_config", object(), "KV connectors"),
        (None, "ec_transfer_config", object(), "encoder-cache connectors"),
    ],
)
def test_settings_one_machine_cant_express_are_refused(where, name, value, problem):
    config = engine_config()
    vars(config if where is None else getattr(config, where))[name] = value  # past validation
    assert problem in refused(config)


def test_micro_batching_under_data_parallelism_is_refused():
    config = engine_config()
    vars(config.parallel_config)["enable_dbo"] = True
    model.check_config(config)  # vLLM micro-batches only under data parallelism
    vars(config.parallel_config)["data_parallel_size"] = 2
    assert "micro-batching" in refused(config)
