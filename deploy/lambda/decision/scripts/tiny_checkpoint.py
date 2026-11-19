"""A tiny random Gemma 4 model checkpoint, and transformers' hidden states on it under any mask:
the reference shim_check.py holds the shim to.

The checkpoint, from a fixed seed, is shaped like E2B but tiny: 15 decoder layers, four sliding
then one full, three times, with KV sharing in the last five, so that, as in E2B, two
full-attention layers come before the shared ones (llama.cpp runs the shared layers on the output
row only, so only a full layer before them can show a full layer's mask); per-layer embeddings,
Gemma 4's whole 262,144-token vocabulary, a vision tower and an audio tower, a random readout,
and a synthetic tokenizer with Gemma 4's control-token and code ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from strands_decider.core import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS

SEED = 20261110
VOCAB = 262144
_HIDDEN = 64
# What `Reference.hidden` runs under: prefix_bidir, then a causal state, a causal question, causal
# throughout, an 8-token sliding window (shorter than the prompts), and causal full-attention layers
# with bidirectional sliding ones, which is what llama.cpp gives without patches/0001.
MASKS = ("prefix_bidir", "causal_state", "causal_question", "causal", "window_8", "swa_only")

_TEXT_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": False, "bos_token_id": 2,
    "enable_moe_block": False, "eos_token_id": 1, "final_logit_softcapping": 30.0, "global_head_dim": 32,
    "head_dim": 16, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": _HIDDEN,
    "hidden_size_per_layer_input": 16, "initializer_range": 0.1, "intermediate_size": 128,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 3,
    "max_position_embeddings": 4096, "model_type": "gemma4_text", "num_attention_heads": 2,
    "num_hidden_layers": 15, "num_key_value_heads": 1, "num_kv_shared_layers": 5, "pad_token_id": 0,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "sliding_window": 8, "tie_word_embeddings": True, "use_bidirectional_attention": None,
    "use_double_wide_mlp": True, "vocab_size": VOCAB, "vocab_size_per_layer_input": VOCAB,
}  # fmt: skip
_VISION_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "default_output_length": 280, "global_head_dim": 16,
    "head_dim": 16, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 32, "initializer_range": 0.1,
    "intermediate_size": 64, "max_position_embeddings": 4096, "model_type": "gemma4_vision",
    "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 2, "patch_size": 16,
    "pooling_kernel_size": 3, "position_embedding_size": 10240, "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}, "standardize": False,
    "use_clipped_linears": True,
}  # fmt: skip
_AUDIO_CONFIG = {
    "attention_chunk_size": 12, "attention_context_left": 13, "attention_context_right": 0,
    "attention_invalid_logits_value": -1e9, "attention_logit_cap": 50.0, "conv_kernel_size": 5,
    "gradient_clipping": 1e10, "hidden_act": "silu", "hidden_size": 32, "initializer_range": 0.1,
    "model_type": "gemma4_audio", "num_attention_heads": 2, "num_hidden_layers": 1,
    "output_proj_dims": _HIDDEN, "residual_weight": 0.5, "rms_norm_eps": 1e-6,
    "subsampling_conv_channels": [8, 4], "use_clipped_linears": True,
}  # fmt: skip
_PROCESSOR_CONFIG = {
    "image_processor": {
        "do_convert_rgb": True, "do_normalize": False, "do_rescale": True, "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0], "image_processor_type": "Gemma4ImageProcessor", "image_seq_length": 280,
        "image_std": [1.0, 1.0, 1.0], "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3,
        "resample": 3, "rescale_factor": 1 / 255,
    },
    "image_seq_length": 280, "processor_class": "Gemma4Processor",
}  # fmt: skip
_SPECIAL_TOKENS = {
    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<|turn>": 105, "<turn|>": 106,
    "<|image>": 255999, "<|audio>": 256000, "<|image|>": 258880, "<|audio|>": 258881,
    "<image|>": 258882, "<audio|>": 258883,
}  # fmt: skip


def _tokenizer_json() -> dict:
    """A byte-fallback BPE tokenizer of Gemma 4's size with its special tokens and its answer codes
    at Gemma 4's ids (two-letter codes by merges), and the printable ASCII characters."""
    assert all(_SPECIAL_TOKENS[k] == v for k, v in CONTROL_TOKEN_IDS.items())
    vocab: dict[str, int] = {**_SPECIAL_TOKENS, **dict(zip(CODES, CODE_TOKEN_IDS, strict=True))}
    reserved = set(vocab.values())
    ids = iter(i for i in range(VOCAB) if i not in reserved)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = next(ids)
    for c in [chr(i) for i in range(32, 127)] + ["\n"]:
        if c not in vocab:
            vocab[c] = next(ids)
    for i in ids:
        vocab[f"<unused{i}>"] = i
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t, i in _SPECIAL_TOKENS.items()]  # fmt: skip
    merges = [[code[0], code[1]] for code in CODES if len(code) == 2]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None, "pre_tokenizer": None, "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": True,
                  "ignore_merges": False, "vocab": vocab, "merges": merges},
    }  # fmt: skip


def make_checkpoint(d: Path) -> Path:
    """The tiny checkpoint in a model checkpoint's layout, in `d`."""
    import torch
    import transformers
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    config = transformers.Gemma4Config(
        text_config=_TEXT_CONFIG, vision_config=_VISION_CONFIG, audio_config=_AUDIO_CONFIG,
        boi_token_id=255999, eoi_token_id=258882, image_token_id=258880, boa_token_id=256000,
        eoa_token_id=258883, audio_token_id=258881, initializer_range=0.1, tie_word_embeddings=True,
    )  # fmt: skip
    transformers.Gemma4ForConditionalGeneration(config).save_pretrained(d, safe_serialization=True)
    (d / "processor_config.json").write_text(json.dumps(_PROCESSOR_CONFIG, indent=1))
    (d / "tokenizer.json").write_text(json.dumps(_tokenizer_json()))
    # The other control tokens are extra special tokens, as in Gemma 4's own tokenizer_config.json,
    # so that the converter types them CONTROL and llama.cpp (and mtmd) match them.
    extra = ["<mask>", "<|turn>", "<turn|>", "<|image>", "<|image|>", "<image|>", "<|audio>",
             "<|audio|>", "<audio|>"]  # fmt: skip
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>",
        "pad_token": "<pad>", "unk_token": "<unk>", "model_max_length": 4096,
        "extra_special_tokens": extra,
    }))  # fmt: skip
    readout = torch.randn(len(CODES), _HIDDEN, generator=torch.Generator().manual_seed(SEED)) * 0.5
    save_file({"weight": readout.to(torch.bfloat16)}, str(d / "readout.safetensors"))
    (d / "decision_config.json").write_text(json.dumps({
        "interface": "strands-decider", "format_version": 1, "prompt": "strands-chat",
        "inputs": ["text", "image"], "codes": list(CODES), "token_ids": list(CODE_TOKEN_IDS),
        "temperature": 1.0, "final_logit_softcapping": 30.0, "attention_mode": "prefix_bidir",
        "pooling": "last", "max_length_validated": 4096,
        "provenance": {"base_model": "google/gemma-4-E2B-it",
                       "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7"},
    }))  # fmt: skip
    return d


class Reference:
    """transformers' Gemma4ForConditionalGeneration on the tiny checkpoint, float32 on CPU, one
    prompt per pass, with the attention mask given per layer type (which bypasses Gemma 4's own)."""

    def __init__(self, ckpt: Path):
        import torch
        import transformers

        self._torch = torch
        conf = transformers.AutoConfig.from_pretrained(ckpt)
        conf.audio_config = None
        self._model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            ckpt, config=conf, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        self._processor = transformers.Gemma4ImageProcessorPil.from_pretrained(ckpt)

    def hidden(self, state, question, mask="prefix_bidir", pixels=()) -> np.ndarray:
        """The final-norm hidden state at the prompt's last token under one of MASKS, with
        `pixels`, the images already at the processor's size, in the state's placeholders."""
        torch = self._torch
        ids = [*state, *question]
        n, s = len(ids), len(state)
        i, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
        in_state, causal = i < s, j <= i
        bidir = (in_state & (j < s)) | ~in_state
        full, sliding = {  # each mask for the full-attention layers, then the sliding ones
            "prefix_bidir": (bidir, bidir),
            "causal_state": ((in_state & causal) | ~in_state,) * 2,
            "causal_question": ((in_state & (j < s)) | (~in_state & causal),) * 2,
            "causal": (causal, causal),
            "window_8": (bidir, bidir & ((i - j).abs() < 8)),
            "swa_only": (causal, bidir),
        }[mask]
        kw = {}
        if pixels:
            px = self._processor(images=list(pixels), do_resize=False, return_tensors="pt")
            kw = {k: v for k, v in px.items() if k != "num_soft_tokens_per_image"}
        with torch.inference_mode():
            out = self._model.model(
                input_ids=torch.tensor([ids]),
                attention_mask={"full_attention": full[None, None], "sliding_attention": sliding[None, None]},
                use_cache=False, return_dict=True, **kw,
            )  # fmt: skip
        return out.last_hidden_state[0, -1].float().numpy()
