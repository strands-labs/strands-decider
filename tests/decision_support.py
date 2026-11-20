"""Tiny random-weight model checkpoints for the backend tests (no download).

`model_checkpoint` writes a model checkpoint's files: the base's multimodal class saved by
transformers in bf16, processor_config.json, a word-level tokenizer with Gemma 4's control tokens
and codes at their ids (so the core's `prepare_tokenizer` accepts it), readout.safetensors and
decision_config.json. The decoder's sliding window of 2 is shorter than any prompt here, so a
runtime that kept the window would answer differently.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from core_support import PROCESSOR_SETTINGS, decision_config_fixture

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from strands_decider.core import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS  # noqa: E402

GEMMA4 = "gemma4"  # E2B's design: per-layer embeddings, KV-shared layers
# 26B-A4B's and 31B's: bidirectional image blocks, no per-layer embeddings, no KV sharing
GEMMA4_VISION = "gemma4-vision"
UNIFIED = "unified"  # the 12B's encoder-free class, also with bidirectional image blocks
KINDS = (GEMMA4, GEMMA4_VISION, UNIFIED)
_TEXT = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "global_head_dim": 32,
    "layer_types": ["sliding_attention", "full_attention"] * 2,
    "sliding_window": 2,
    "pad_token_id": 0,
}
_IMAGE_IDS = {
    "boi_token_id": CONTROL_TOKEN_IDS["<|image>"],
    "image_token_id": CONTROL_TOKEN_IDS["<|image|>"],
    "eoi_token_id": CONTROL_TOKEN_IDS["<image|>"],
}


def _tokenizer():
    """Gemma 4's ids where the interface fixes them, and a few words at the free ids from 0 up."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    fixed = {"<eos>": 1, **CONTROL_TOKEN_IDS}
    vocab = dict(zip(CODES, CODE_TOKEN_IDS, strict=True)) | fixed
    taken = set(vocab.values())
    free = (i for i in range(10**6) if i not in taken)
    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    vocab |= {w: next(free) for w in words}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    tok.add_special_tokens([AddedToken(t, special=True) for t in fixed])
    return tok


def _model(kind: str, vocab: int, text: dict):
    text = {"vocab_size": vocab, **_TEXT, **text}
    if kind == UNIFIED:
        if not hasattr(transformers, "Gemma4UnifiedForConditionalGeneration"):
            pytest.skip("this transformers has no Gemma4UnifiedForConditionalGeneration")
        config = transformers.Gemma4UnifiedConfig(
            text_config=transformers.Gemma4UnifiedTextConfig(
                **text, use_bidirectional_attention="vision"
            ),
            vision_config=transformers.Gemma4UnifiedVisionConfig(
                mm_embed_dim=32, output_proj_dims=32
            ),
            audio_config=None,
            **_IMAGE_IDS,
        )
        return transformers.Gemma4UnifiedForConditionalGeneration(config)
    layout = (
        {"hidden_size_per_layer_input": 8, "num_kv_shared_layers": 1}
        if kind == GEMMA4
        else {
            "use_bidirectional_attention": "vision",
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
        }
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
        text_config=transformers.Gemma4TextConfig(
            vocab_size_per_layer_input=vocab, **layout | text
        ),
        vision_config=vision,
        audio_config=None,
        **_IMAGE_IDS,
    )
    return transformers.Gemma4ForConditionalGeneration(config)


def model_checkpoint(
    path: Path,
    kind: str = GEMMA4,
    *,
    decision: dict | None = None,
    processor: dict | None = None,
    text: dict | None = None,
) -> Path:
    """A tiny model checkpoint of `kind` at `path`; `decision`, `processor` and `text` override
    fields of decision_config.json, of processor_config.json's image_processor, and of the
    decoder's configuration."""
    from safetensors.torch import save_file

    tok = _tokenizer()
    torch.manual_seed(0)
    model = _model(kind, max(tok.get_vocab().values()) + 1, text or {}).to(torch.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tok.save(str(path / "tokenizer.json"))
    image_processor = {
        "image_processor_type": "Gemma4UnifiedImageProcessor"
        if kind == UNIFIED
        else "Gemma4ImageProcessor",
        **PROCESSOR_SETTINGS,
        **(processor or {}),
    }
    (path / "processor_config.json").write_text(json.dumps({"image_processor": image_processor}))
    config = decision_config_fixture() | {"temperature": 1.7} | (decision or {})
    (path / "decision_config.json").write_text(json.dumps(config))
    # Large enough that the soft-cap and the bf16 rounding both matter.
    g = torch.Generator().manual_seed(0)
    w = torch.randn(len(CODES), model.config.get_text_config().hidden_size, generator=g) * 0.6
    save_file({"weight": w.to(torch.bfloat16)}, str(path / "readout.safetensors"))
    return path


def image_b64(width: int, height: int, fmt: str = "PNG", seed: int = 0) -> str:
    """A noisy RGB image as base64, so that resizing moves pixels."""
    from PIL import Image

    arr = np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
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
