"""Startup checks on real vLLM configs: engine settings under which prefix_bidir would be served incorrectly, or under
which one request could stop the engine, are refused.

The configs are built by vLLM's own ``EngineArgs.create_engine_config`` from a checkpoint directory (only its
config.json and tokenizer are read), so ``check_config`` reads the attributes vLLM really sets: a renamed attribute
fails here. Set ``STRANDS_DECIDER_TEST_CHECKPOINT`` to a checkpoint directory to run them; settings that
``EngineArgs`` cannot express on one machine (data parallelism, a quantized checkpoint, speculative decoding) are set
on the built config.
"""

from __future__ import annotations

import os

import pytest

model = pytest.importorskip("strands_decider_vllm.model")
from strands_decider_vllm import ARCH  # noqa: E402

CHECKPOINT = os.environ.get("STRANDS_DECIDER_TEST_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="STRANDS_DECIDER_TEST_CHECKPOINT is not set")

#: The README's serving flags, at an 8,192-token context.
BASE = dict(
    hf_overrides={"architectures": [ARCH], "use_bidirectional_attention": None},
    dtype="bfloat16",
    seed=0,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    enable_chunked_prefill=False,
    enable_prefix_caching=True,
    generation_config="vllm",
    logprobs_mode="processed_logprobs",
    max_logprobs=255,
)


@pytest.fixture(autouse=True)
def _model_runner_v2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


def engine_config(**overrides):
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(model=CHECKPOINT, **{**BASE, **overrides}).create_engine_config()


def refused(config):
    with pytest.raises(ValueError, match="cannot serve with this configuration") as err:
        model.check_config(config)
    return str(err.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"enable_prefix_caching": False},
        {"prefix_caching_hash_algo": "sha256_cbor"},
        {"max_model_len": 4096, "max_num_batched_tokens": 8192},
        {"max_num_scheduled_tokens": 8192},
        {
            "long_prefill_token_threshold": 8192
        },  # cannot cut a prompt of at most max_model_len tokens
        {"kv_cache_dtype": "bfloat16"},
        {"logprobs_mode": "processed_logits"},
        {"compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"}},
        {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}},
        {"compilation_config": {"cudagraph_mode": "PIECEWISE"}},
        {"compilation_config": {"cudagraph_mode": "NONE"}},
        {"enforce_eager": True},
    ],
)
def test_serving_flags_pass(overrides):
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
        ({"logprobs_mode": "raw_logits"}, "--logprobs-mode raw_logits"),
        ({"dtype": "float16"}, "--dtype torch.float16"),
        ({"dtype": "float32"}, "--dtype torch.float32"),
        ({"kv_cache_dtype": "fp8"}, "--kv-cache-dtype fp8"),
        ({"kv_cache_dtype": "fp8_e5m2"}, "--kv-cache-dtype fp8_e5m2"),
        ({"enable_prompt_embeds": True}, "--enable-prompt-embeds"),
        ({"prefix_caching_hash_algo": "xxhash"}, "--prefix-caching-hash-algo"),
    ],
)
def test_unsafe_flags_are_refused(overrides, problem):
    assert problem in refused(engine_config(**overrides))


def set_attr(obj, name, value):
    object.__setattr__(obj, name, value)  # past pydantic's assignment validation


@pytest.mark.parametrize(
    "where, name, value, problem",
    [
        # vLLM itself refuses this budget with chunked prefill off; the plugin checks it again.
        (
            "scheduler_config",
            "max_num_batched_tokens",
            4096,
            "--max-num-batched-tokens 4096 < --max-model-len 8192",
        ),
        ("model_config", "quantization", "fp8", "--quantization fp8"),
        ("model_config", "is_mm_prefix_lm", True, "use_bidirectional_attention"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("parallel_config", "decode_context_parallel_size", 2, "decode_context_parallel_size"),
        ("parallel_config", "prefill_context_parallel_size", 2, "prefill_context_parallel_size"),
        ("cache_config", "kv_sharing_fast_prefill", True, "--kv-sharing-fast-prefill"),
        (None, "speculative_config", object(), "speculative decoding"),
        (None, "kv_transfer_config", object(), "KV connectors"),
    ],
)
def test_settings_set_on_the_built_config_are_refused(where, name, value, problem):
    config = engine_config()
    target = config if where is None else getattr(config, where)
    if name == "is_mm_prefix_lm":
        target.__dict__[name] = value  # a cached_property
    else:
        set_attr(target, name, value)
    assert problem in refused(config)


@pytest.mark.parametrize("dbo, ubatch", [(True, 0), (False, 2)])
def test_data_parallel_micro_batching_is_refused(dbo, ubatch):
    config = engine_config()
    pc = config.parallel_config
    set_attr(pc, "enable_dbo", dbo)
    set_attr(pc, "ubatch_size", ubatch)
    model.check_config(config)  # one engine: vLLM micro-batches only with data parallelism
    set_attr(pc, "data_parallel_size", 2)
    assert "micro-batching" in refused(config)


def test_model_runner_v1_is_refused(monkeypatch):
    config = engine_config()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert "Model Runner V2" in refused(config)


def test_every_problem_is_reported_at_once():
    message = refused(engine_config(enable_chunked_prefill=True, logprobs_mode="raw_logprobs"))
    assert message.count("\n  - ") == 2


def test_xxhash_is_allowed_without_prefix_caching():
    model.check_config(
        engine_config(enable_prefix_caching=False, prefix_caching_hash_algo="xxhash")
    )
