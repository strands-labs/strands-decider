"""Tiny random-weight decision checkpoints for the backend tests (no download).

`decision_checkpoint` writes a directory in a decision checkpoint's layout: the base's
multimodal class saved by transformers in bf16, processor_config.json, a word-level tokenizer
with Gemma 4's control tokens and codes at their ids (so the core's `prepare_tokenizer` accepts
it), readout.safetensors and decision_config.json. The decoder's sliding window of 2 is shorter
than any prompt here, so a runtime that kept the window would answer differently.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from strands_decider.core import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS  # noqa: E402

GEMMA4 = "gemma4"  # E2B's design: per-layer embeddings, KV-shared layers
# 26B-A4B's and 31B's: the same class with bidirectional image blocks, no per-layer embeddings
# and no KV sharing
GEMMA4_VISION = "gemma4-vision"
UNIFIED = "unified"  # the 12B's encoder-free class, also with bidirectional image blocks
KINDS = (GEMMA4, GEMMA4_VISION, UNIFIED)
HAS_UNIFIED = hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")
BASE = {
    GEMMA4: ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    GEMMA4_VISION: ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    UNIFIED: ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d"),
}


def tokenizer():
    """A word-level tokenizer with Gemma 4's ids where the interface fixes them: the control
    tokens at `CONTROL_TOKEN_IDS`, `<eos>` at 1, and each code at its id in `CODE_TOKEN_IDS`.
    The other words take the free ids from 0 up."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    fixed = {"<eos>": 1, **CONTROL_TOKEN_IDS}
    codes = dict(zip(CODES, CODE_TOKEN_IDS, strict=True))
    taken = {*fixed.values(), *codes.values()}
    free = (i for i in range(10**6) if i not in taken)
    vocab = {w: next(free) for w in dict.fromkeys(words) if w not in codes}
    vocab.update(codes)
    vocab.update(fixed)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    tok.add_special_tokens([AddedToken(t, special=True) for t in fixed])
    return tok


def _model(kind: str, tok, text: dict):
    vocab = max(tok.get_vocab().values()) + 1  # 258883: the image placeholder's id is the largest
    ids = {
        "boi_token_id": tok.token_to_id("<|image>"),
        "image_token_id": tok.token_to_id("<|image|>"),
        "eoi_token_id": tok.token_to_id("<image|>"),
    }
    text = {
        "vocab_size": vocab,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 32,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "sliding_window": 2,
        "final_logit_softcapping": 30.0,
        "pad_token_id": 0,
        **text,
    }
    if kind == UNIFIED:
        text_config = transformers.Gemma4UnifiedTextConfig(
            **text, use_bidirectional_attention="vision"
        )
        vision = transformers.Gemma4UnifiedVisionConfig(mm_embed_dim=32, output_proj_dims=32)
        config = transformers.Gemma4UnifiedConfig(
            text_config=text_config, vision_config=vision, audio_config=None, **ids
        )
        return transformers.Gemma4UnifiedForConditionalGeneration(config)
    per_layer = {"hidden_size_per_layer_input": 8, "num_kv_shared_layers": 1}
    if kind == GEMMA4_VISION:
        per_layer = {
            "use_bidirectional_attention": "vision",
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
        }
    text_config = transformers.Gemma4TextConfig(
        **{"vocab_size_per_layer_input": vocab, **per_layer, **text}
    )
    vision = transformers.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )
    config = transformers.Gemma4Config(
        text_config=text_config, vision_config=vision, audio_config=None, **ids
    )
    return transformers.Gemma4ForConditionalGeneration(config)


def decision_checkpoint(
    path: Path,
    kind: str = GEMMA4,
    *,
    seed: int = 0,
    decision: dict | None = None,
    processor: dict | None = None,
    text: dict | None = None,
) -> Path:
    """A tiny decision checkpoint of `kind` at `path`; `decision`, `processor` and `text`
    override fields of decision_config.json, of processor_config.json's image_processor, and
    of the decoder's configuration."""
    if kind == UNIFIED and not HAS_UNIFIED:
        pytest.skip("this transformers has no Gemma4UnifiedForConditionalGeneration")
    from safetensors.torch import save_file

    tok = tokenizer()
    torch.manual_seed(seed)
    model = _model(kind, tok, text or {}).to(torch.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tok.save(str(path / "tokenizer.json"))
    image_processor = {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "image_processor_type": "Gemma4UnifiedImageProcessor"
        if kind == UNIFIED
        else "Gemma4ImageProcessor",
        "max_soft_tokens": 280,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1 / 255,
        **(processor or {}),
    }
    (path / "processor_config.json").write_text(
        json.dumps({"image_processor": image_processor, "processor_class": "Gemma4Processor"})
    )
    base_model, base_revision = BASE[kind]
    config = {
        "interface": "strands-decider",
        "format_version": 1,
        "prompt": "strands-chat",
        "inputs": ["text", "image"],
        "codes": list(CODES),
        "token_ids": list(CODE_TOKEN_IDS),
        "temperature": 1.7,
        "final_logit_softcapping": 30.0,
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "max_length_validated": 4096,
        "provenance": {"base_model": base_model, "base_revision": base_revision},
        **(decision or {}),
    }
    (path / "decision_config.json").write_text(json.dumps(config))
    # Large enough that the soft-cap and the bf16 rounding both matter.
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(len(CODES), model.config.get_text_config().hidden_size, generator=g) * 0.6
    save_file({"weight": w.to(torch.bfloat16)}, str(path / "readout.safetensors"))
    return path


def image_b64(width: int, height: int, fmt: str = "PNG", seed: int = 0) -> str:
    """A noisy RGB image of `width` x `height` as base64, so that resizing moves pixels."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def request(
    images: list[str] | None = None, state: object = "The order shipped late, twice."
) -> dict:
    """A request with one question of each kind, the choice's options of different lengths."""
    return {
        "state": state,
        "images": images or [],
        "questions": {
            "late": {"type": "noul", "instructions": "Was the order late?"},
            "team": {
                "type": "choice",
                "instructions": "Which team handles it?",
                "criteria": {"billing": "refunds and invoices", "shipping": None, "sales": "deals"},
            },
            "mood": {
                "type": "score",
                "instructions": "How upset is the customer?",
                "criteria": ["calm", "annoyed", "angry", "furious"],
            },
        },
    }
