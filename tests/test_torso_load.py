"""Real, tiny CPU checkpoints exercise decoder loading without Hub downloads."""

import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file

from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel


def _base_model(family):
    text = dict(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        tie_word_embeddings=True,
    )
    if family == "qwen3":
        model = transformers.Qwen3Model(transformers.Qwen3Config(**text))
        return model, model, ""
    if family.startswith("qwen3_5"):
        text.update(
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
        )
        if family == "qwen3_5_text":
            model = transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(**text))
            return model, model.model, "model."
        config = transformers.Qwen3_5Config(
            text_config=text,
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=32,
                patch_size=4,
                spatial_merge_size=2,
                temporal_patch_size=2,
                num_position_embeddings=16,
            ),
            tie_word_embeddings=True,
        )
        model = transformers.Qwen3_5ForConditionalGeneration(config)
    else:
        text.update(
            global_head_dim=16,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            layer_types=["full_attention"],
        )
        name = "Gemma4Unified" if family == "gemma4_unified" else "Gemma4"
        if not hasattr(transformers, name + "Config"):
            pytest.skip(f"this transformers has no {name}")
        config = getattr(transformers, name + "Config")(
            text_config=text, vision_config=None, audio_config=None
        )
        model = getattr(transformers, name + "ForConditionalGeneration")(config)
    return model, model.model.language_model, "model.language_model."


@pytest.fixture(params=["qwen3", "qwen3_5_text", "qwen3_5", "gemma4", "gemma4_unified"])
def base_checkpoint(request, tmp_path):
    model, decoder, prefix = _base_model(request.param)
    model.save_pretrained(tmp_path)
    return tmp_path, decoder.state_dict(), prefix


def _load(path):
    return StrandsDeciderModel._load_torso(
        StrandsDeciderConfig(base_model=str(path), torch_dtype="float32"), None, None
    )


def test_complete_decoder_loads_every_saved_tensor(base_checkpoint):
    path, expected, _ = base_checkpoint
    actual = _load(path).state_dict()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], value) for key, value in expected.items())


def test_a_missing_decoder_weight_is_refused(base_checkpoint):
    path, expected, prefix = base_checkpoint
    decoder_key = next(key for key in expected if key.endswith(".self_attn.q_proj.weight"))
    key = prefix + decoder_key
    weights = load_file(path / "model.safetensors")
    del weights[key]
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})

    with pytest.raises(ValueError, match="missing decoder weights") as exc:
        _load(path)
    assert str(path) in str(exc.value)
    assert decoder_key in str(exc.value)


def test_an_unmapped_qwen35_checkpoint_layout_is_refused(tmp_path):
    model, _, _ = _base_model("qwen3_5")
    # Qwen3_5Model saves language_model.*, unlike the Hub wrapper's model.language_model.*.
    model.model.save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="missing decoder weights") as exc:
        _load(tmp_path)
    assert str(tmp_path) in str(exc.value)


def test_gemma4_can_omit_the_discarded_encoders_and_tied_head(tmp_path):
    model, _, prefix = _base_model("gemma4")
    model.config.vision_config = transformers.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        position_embedding_size=16,
    )
    model.config.audio_config = transformers.Gemma4AudioConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=[4, 4],
        output_proj_dims=16,
    )
    model = transformers.Gemma4ForConditionalGeneration(model.config)
    model.save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    assert "lm_head.weight" not in weights
    discarded = [key for key in weights if not key.startswith(prefix)]
    assert any("vision" in key for key in discarded)
    assert any("audio" in key for key in discarded)
    save_file(
        {key: value for key, value in weights.items() if key.startswith(prefix)},
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )

    actual = _load(tmp_path).state_dict()
    expected = model.model.language_model.state_dict()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
