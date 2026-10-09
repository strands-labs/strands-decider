from types import SimpleNamespace

from strands_decider.modeling import serving_attn_implementation


def test_gemma4_with_one_kv_head_is_served_with_eager_attention():
    cfg = lambda t, n: SimpleNamespace(model_type=t, num_key_value_heads=n)  # noqa: E731
    assert serving_attn_implementation(cfg("gemma4_text", 1)) == "eager"  # E2B's text decoder
    assert serving_attn_implementation(cfg("gemma4_text", 2)) is None  # E4B
    assert serving_attn_implementation(cfg("qwen3_5_text", 1)) is None


def test_a_config_with_per_layer_head_counts_keeps_the_default():
    class PerLayer:  # transformers' gemma4_unified / 26B-A4B text configs raise on the global read
        model_type = "gemma4_text"

        @property
        def num_key_value_heads(self):
            raise AttributeError("per-layer attribute")

    assert serving_attn_implementation(PerLayer()) is None
