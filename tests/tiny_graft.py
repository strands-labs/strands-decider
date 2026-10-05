"""Tiny random-weight pieces of a grafted MiniCPM5 (src/strands_decider/graft.py), written to
directories, nothing downloaded:

* `save_siglip`: a full SigLIP checkpoint (text and vision towers, as
  google/siglip2-so400m-patch16-384 is) with a PIL image processor; 32 px images in 4 px
  patches, so an 8 x 8 grid that unshuffles to 16 slots per image.
* `save_minicpm_like`: the Llama of tests/tiny_bases.py (untied output head, `<s>`
  prepended) with `<unused_token_0>` added as an ordinary added token, as MiniCPM5's
  vocabulary holds it.
* `save_text_checkpoint`: a pointer-head text Strands Decider on it, its adapter non-trivial.
* `save_stage1`: a projector directory as stage 1 writes it.
"""

from __future__ import annotations

import torch
from tiny_bases import ATTENTION_MLP, llama_tokenizer

SLOT = "<unused_token_0>"


def save_siglip(d: str) -> str:
    import transformers

    cfg = transformers.SiglipConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                     "num_attention_heads": 2, "vocab_size": 64, "max_position_embeddings": 16},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "image_size": 32, "patch_size": 4},
    )
    torch.manual_seed(0)
    transformers.SiglipModel(cfg).save_pretrained(d)
    transformers.SiglipImageProcessorPil(size={"height": 32, "width": 32}, image_mean=[0.5] * 3,
                                         image_std=[0.5] * 3).save_pretrained(d)
    return d


def save_minicpm_like(d: str) -> str:
    import transformers

    tok = llama_tokenizer()
    tok.add_tokens([SLOT])
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
        bos_token_id=0, eos_token_id=1, pad_token_id=1)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def save_text_checkpoint(base: str, d: str) -> str:
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    cfg = StrandsDeciderConfig(base_model=base, head_type="pointer", pointer_dim=32, torch_dtype="float32",
                               max_length=1024, lora_targets=ATTENTION_MLP,
                               temperature_by_kind={"noul": 0.9, "choice": 0.7})
    torch.manual_seed(1)
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    with torch.no_grad():
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    model.save_pretrained(d)
    return d


def projector_config(encoder: str):
    from strands_decider.graft import ProjectorConfig

    return ProjectorConfig(encoder=encoder, encoder_revision=None, encoder_dtype="float32",
                           vision_hidden=32, text_hidden=64, image_size=32, patch_size=4, unshuffle=2)


def save_stage1(encoder: str, d: str) -> str:
    from strands_decider.graft import Projector, save_projector

    torch.manual_seed(2)
    cfg = projector_config(encoder)
    save_projector(Projector(cfg), cfg, d)
    return d
