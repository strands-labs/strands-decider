"""Startup checks: engine settings under which prefix_bidir would be served incorrectly are refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

model = pytest.importorskip("strands_decider_v2_vllm.model")


def config(**overrides):
    """A VllmConfig stand-in with the attributes check_config reads, set to the README's serving flags."""
    values = dict(
        is_mm_prefix_lm=False,
        use_v2_model_runner=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=32768,
        max_model_len=32768,
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256",
        speculative_config=None,
        kv_transfer_config=None,
        kv_sharing_fast_prefill=False,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    values.update(overrides)
    v = SimpleNamespace(**values)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_mm_prefix_lm=v.is_mm_prefix_lm, max_model_len=v.max_model_len
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=v.enable_chunked_prefill,
            max_num_batched_tokens=v.max_num_batched_tokens,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=v.enable_prefix_caching,
            prefix_caching_hash_algo=v.prefix_caching_hash_algo,
            kv_sharing_fast_prefill=v.kv_sharing_fast_prefill,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=v.pipeline_parallel_size,
            decode_context_parallel_size=v.decode_context_parallel_size,
            prefill_context_parallel_size=v.prefill_context_parallel_size,
        ),
        use_v2_model_runner=v.use_v2_model_runner,
        speculative_config=v.speculative_config,
        kv_transfer_config=v.kv_transfer_config,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"enable_prefix_caching": False},
        {"prefix_caching_hash_algo": "sha256_cbor"},
        {"max_model_len": 12288, "max_num_batched_tokens": 12288},
        {"max_model_len": 8192, "max_num_batched_tokens": 32768},
    ],
)
def test_serving_flags_pass(overrides):
    model.check_config(config(**overrides))


@pytest.mark.parametrize(
    "overrides, problem",
    [
        ({"is_mm_prefix_lm": True}, "use_bidirectional_attention"),
        ({"use_v2_model_runner": False}, "Model Runner V2"),
        ({"enable_chunked_prefill": True}, "--no-enable-chunked-prefill"),
        ({"max_num_batched_tokens": 8192}, "--max-num-batched-tokens 8192 < --max-model-len 32768"),
        ({"prefix_caching_hash_algo": "xxhash"}, "--prefix-caching-hash-algo"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV connectors"),
        ({"kv_sharing_fast_prefill": True}, "--kv-sharing-fast-prefill"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"decode_context_parallel_size": 2}, "decode_context_parallel_size"),
        ({"prefill_context_parallel_size": 2}, "prefill_context_parallel_size"),
    ],
)
def test_unsafe_settings_are_refused(overrides, problem):
    with pytest.raises(ValueError, match="cannot serve with this configuration") as err:
        model.check_config(config(**overrides))
    assert problem in str(err.value)


def test_every_problem_is_reported_at_once():
    with pytest.raises(ValueError) as err:
        model.check_config(config(enable_chunked_prefill=True, use_v2_model_runner=False))
    assert str(err.value).count("\n  - ") == 2


def test_xxhash_is_allowed_without_prefix_caching():
    model.check_config(config(enable_prefix_caching=False, prefix_caching_hash_algo="xxhash"))
