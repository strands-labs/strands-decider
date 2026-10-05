"""A tiny random-weight Qwen3.5 multimodal base and a Strands Decider checkpoint on it.

Nothing is downloaded: the checkpoint, the tokeniser and the image processor are all
written to the given directory. The tokeniser is a real `Qwen3_5Tokenizer` (byte-level
BPE over the 256 byte symbols, no merges, plus the Qwen vision tokens), so it reloads as
itself next to a qwen3_5 config and tokenises every character of a state or question.
Shared by every test that needs a Qwen3.5 torso.
"""

from __future__ import annotations

import re

import pytest
import torch
import transformers

# The tiny Qwen3.5 (and image input) need transformers >= 5.18; the package needs 5.15.
needs_tiny_qwen35 = pytest.mark.skipif(
    tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18),
    reason="the tiny Qwen3.5 needs transformers >= 5.18",
)

VISION_TOKENS = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


def tokenizer():
    import transformers
    from tokenizers import pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, **{c: i + 1 for i, c in enumerate(alphabet)}}
    tok = transformers.Qwen3_5Tokenizer(vocab=vocab, merges=[])
    tok.add_special_tokens({"additional_special_tokens": list(VISION_TOKENS)})
    return tok


def save_base(d: str) -> str:
    """3 Gated DeltaNet + 1 attention layer, 2-layer ViT, saved to `d`."""
    import transformers

    tok = tokenizer()
    start, pad, end = (tok.convert_tokens_to_ids(t) for t in VISION_TOKENS)
    cfg = transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128, "head_dim": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": len(tok),
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 4, "linear_num_value_heads": 4,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
            "attn_output_gate": True, "tie_word_embeddings": True, "mtp_num_hidden_layers": 0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        },
        vision_config={"depth": 2, "hidden_size": 64, "num_heads": 4, "intermediate_size": 128,
                       "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "num_position_embeddings": 2304},
        image_token_id=pad, vision_start_token_id=start, vision_end_token_id=end,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    transformers.Qwen2VLImageProcessorPil(
        patch_size=16, temporal_patch_size=2, merge_size=2,
        size={"shortest_edge": 65536, "longest_edge": 16777216},
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
    ).save_pretrained(d)
    return d


def save_checkpoint(base: str, d: str) -> str:
    """A text Strands Decider checkpoint on `base`, with a non-trivial adapter."""
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    cfg = StrandsDeciderConfig(base_model=base, head_type="pointer", pointer_dim=32,
                               torch_dtype="float32", max_length=1024, lora_targets=TARGETS,
                               temperature_by_kind={"noul": 0.9, "choice": 0.7})
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    with torch.no_grad():
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    model.save_pretrained(d)
    return d
