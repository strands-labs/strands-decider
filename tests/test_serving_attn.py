from types import SimpleNamespace

from strands_decider.modeling import serving_attn_implementation


def test_gemma4_with_one_kv_head_is_served_with_eager_attention():
    cfg = lambda t, n: SimpleNamespace(model_type=t, num_key_value_heads=n)  # noqa: E731
    assert serving_attn_implementation(cfg("gemma4_text", 1)) == "eager"  # E2B's text decoder
    assert serving_attn_implementation(cfg("gemma4_text", 2)) is None  # E4B
    assert serving_attn_implementation(cfg("qwen3_5_text", 1)) is None
