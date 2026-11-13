"""The decoder loader, the prefix_bidir mask, the split attention and the adapters, on a tiny random Gemma 4 (per-layer
inputs, a KV-shared layer, sliding and full layers) built in the test: CPU, float32, no tokenizer."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from training import attention, lora, model

LAYERS = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig, Gemma4VisionConfig

    torch.manual_seed(0)
    text = Gemma4TextConfig(
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=1,
        sliding_window=8,
        layer_types=LAYERS,
        max_position_embeddings=1024,
        final_logit_softcapping=30.0,
    )
    vision = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
    )
    m = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=None)
    )
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.1)
    path = tmp_path_factory.mktemp("tiny") / "base"
    m.save_pretrained(str(path))
    return path


def batch(width=12, lengths=(12, 9, 5), states=(5, 4, 2), seed=1):
    """Left-padded random ids, the padding mask and each row's boundary."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for i, n in enumerate(lengths):
        ids[i, width - n :] = torch.randint(1, 512, (n,), generator=g)
        mask[i, width - n :] = 1
    ends = torch.tensor([width - n + s for n, s in zip(lengths, states, strict=True)])
    return ids, mask, ends


def run(decoder, owner, ids, mask, ends):
    owner._prefix_end = ends
    try:
        with torch.no_grad():
            return decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    finally:
        owner._prefix_end = None


def test_loader_reads_the_base_layout(base):
    decoder = model.load_decoder(base, torch.float32)
    tensors = load_file(str(base / "model.safetensors"))
    state = decoder.state_dict()
    for name, value in state.items():
        assert torch.equal(value, tensors["model.language_model." + name].float()), name


def test_loader_refuses_a_missing_or_stray_tensor(base, tmp_path):
    import shutil

    for change in ("missing", "stray"):
        copy = tmp_path / change
        shutil.copytree(base, copy)
        tensors = load_file(str(copy / "model.safetensors"))
        if change == "missing":
            del tensors["model.language_model.layers.1.mlp.down_proj.weight"]
        else:
            tensors["model.language_model.layers.1.extra.weight"] = torch.zeros(2, 2)
        save_file(tensors, str(copy / "model.safetensors"), metadata={"format": "pt"})
        with pytest.raises(ValueError, match="did not load cleanly"):
            model.load_decoder(copy, torch.float32)


def test_only_the_kv_shared_layers_may_carry_unread_k_v(base, tmp_path):
    import shutil

    config = model.text_config(base)
    assert model.kv_shared_unused(config) == {
        f"model.language_model.layers.3.self_attn.{m}.weight"
        for m in ("k_proj", "v_proj", "k_norm")
    }
    assert (
        model.kv_shared_unused(SimpleNamespace(num_hidden_layers=48, num_kv_shared_layers=0))
        == set()
    )
    for layers, shared, first in ((35, 20, 15), (42, 18, 24)):  # the E2B's and the E4B's
        names = model.kv_shared_unused(
            SimpleNamespace(num_hidden_layers=layers, num_kv_shared_layers=shared)
        )
        assert len(names) == 3 * shared and {int(n.split(".")[3]) for n in names} == set(
            range(first, layers)
        )
    for layer, accepted in ((3, True), (9, False)):
        copy = tmp_path / f"layer-{layer}"
        shutil.copytree(base, copy)
        tensors = load_file(str(copy / "model.safetensors"))
        tensors[f"model.language_model.layers.{layer}.self_attn.v_proj.weight"] = torch.zeros(
            16, 32
        )
        save_file(tensors, str(copy / "model.safetensors"), metadata={"format": "pt"})
        if accepted:
            model.load_decoder(copy, torch.float32)
        else:
            with pytest.raises(ValueError, match="did not load cleanly"):
                model.load_decoder(copy, torch.float32)


def test_state_is_independent_of_the_question(base):
    """Under prefix_bidir the state's hidden states do not depend on what follows it."""
    decoder = model.load_decoder(base, torch.float32)
    owner = SimpleNamespace(_prefix_end=None)
    attention.install_mask(decoder, owner)
    ids, mask, ends = batch(lengths=(12, 12), states=(6, 6))
    ids[1, :6] = ids[0, :6]  # the same state, another question
    hidden = run(decoder, owner, ids, mask, ends)
    assert torch.allclose(hidden[0, :6], hidden[1, :6], atol=1e-6)
    assert not torch.allclose(hidden[0, 6:], hidden[1, 6:])


def test_split_attention_equals_the_masked_attention(base):
    masked = model.load_decoder(base, torch.float32)
    split = model.load_decoder(base, torch.float32)
    owner_m, owner_s = SimpleNamespace(_prefix_end=None), SimpleNamespace(_prefix_end=None)
    attention.install_mask(masked, owner_m)
    attention.install_mask(split, owner_s)
    attention.install_split(owner_s, split, min_length=1)
    ids, mask, ends = batch()
    a, b = run(masked, owner_m, ids, mask, ends), run(split, owner_s, ids, mask, ends)
    valid = mask.bool()
    assert owner_s._split["split_calls"] > 0
    assert torch.allclose(a[valid], b[valid], atol=1e-5)


def test_adapters_start_at_the_base_and_merge_into_it(base, tmp_path):
    decoder = model.load_decoder(base, torch.float32)
    owner = SimpleNamespace(_prefix_end=None)
    attention.install_mask(decoder, owner)
    ids, mask, ends = batch()
    reference = run(decoder, owner, ids, mask, ends)
    holder = model.LanguageModelWrapper(decoder)
    torch.manual_seed(3)
    attached = lora.attach(holder, rank=4, alpha=8, dropout=0.0, targets=lora.DEFAULT_TARGETS)
    assert attached and torch.equal(run(decoder, owner, ids, mask, ends), reference)  # B = 0
    with torch.no_grad():
        for module in attached.values():
            module.lora_b.normal_(0, 0.05)
    adapted = run(decoder, owner, ids, mask, ends)
    assert not torch.allclose(adapted, reference)
    # fold the adapters into the base tensors and load the result: the same hidden states
    tensors = load_file(str(base / "model.safetensors"))
    for name, module in attached.items():
        key = f"model.{name}.weight"
        tensors[key] = lora.merged_weight(
            tensors[key].float(), module.lora_a.detach(), module.lora_b.detach(), module.scale
        )
    out = tmp_path / "merged"
    out.mkdir()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text((base / "config.json").read_text())
    merged = model.load_decoder(out, torch.float32)
    owner2 = SimpleNamespace(_prefix_end=None)
    attention.install_mask(merged, owner2)
    assert torch.allclose(
        run(merged, owner2, ids, mask, ends)[mask.bool()], adapted[mask.bool()], atol=1e-4
    )


def test_text_config_refuses_other_models(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    with pytest.raises((ValueError, KeyError, OSError)):
        model.text_config(tmp_path)
