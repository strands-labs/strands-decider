"""Tiny random-weight decision checkpoints for the backend tests (no download).

`decision_checkpoint` writes a directory in docs/spec.md's layout (section 9): the base's
multimodal class saved by transformers (`Gemma4ForConditionalGeneration`, or
`Gemma4UnifiedForConditionalGeneration` for the 12B's encoder-free design) in bf16,
`processor_config.json`, the toy tokenizer of tests/core_support.py, `readout.safetensors`
and `decision_config.json`. The decoder has a sliding window of 2, shorter than any prompt
here, so a runtime that kept the window would answer differently; the Gemma 4 variant has
E2B's KV-shared layers and per-layer embeddings. `intermediate_checkpoint` writes the layout
the intermediate checkpoints have: a bare `Gemma4TextModel` and a decision_config.json from
before `interface`, `inputs`, `prompt` and `provenance`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from core_support import toy_tokenizer  # noqa: E402

from strands_decider.core import CODES  # noqa: E402

GEMMA4 = "gemma4"
UNIFIED = "unified"
KINDS = (GEMMA4, UNIFIED)
HAS_UNIFIED = hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")
LAYERS = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
HIDDEN = 64
SOFTCAP = 30.0
TEMPERATURE = 1.7
BASE = {
    GEMMA4: ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    UNIFIED: ("google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d"),
}
PROCESSOR_TYPES = {GEMMA4: "Gemma4ImageProcessor", UNIFIED: "Gemma4UnifiedImageProcessor"}


def tokenizer():
    return toy_tokenizer()


def _text_kwargs(vocab: int) -> dict:
    return {
        "vocab_size": vocab,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": len(LAYERS),
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "global_head_dim": 32,
        "layer_types": list(LAYERS),
        "sliding_window": 2,
        "final_logit_softcapping": SOFTCAP,
        "pad_token_id": 0,
    }


def gemma4_text_config(vocab: int, **overrides):
    return transformers.Gemma4TextConfig(
        **{**_text_kwargs(vocab), **overrides},
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=vocab,
        num_kv_shared_layers=1,
    )


def model_config(kind: str, tok, *, vision: bool = True, text: dict | None = None):
    ids = {
        "boi_token_id": tok.token_to_id("<|image>"),
        "image_token_id": tok.token_to_id("<|image|>"),
        "eoi_token_id": tok.token_to_id("<image|>"),
    }
    vocab = tok.get_vocab_size()
    if kind == GEMMA4:
        vis = transformers.Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
        return transformers.Gemma4Config(
            text_config=gemma4_text_config(vocab, **(text or {})),
            vision_config=vis if vision else None,
            audio_config=None,
            **ids,
        )
    text = transformers.Gemma4UnifiedTextConfig(
        **{**_text_kwargs(vocab), **(text or {})}, use_bidirectional_attention="vision"
    )
    vis = transformers.Gemma4UnifiedVisionConfig(mm_embed_dim=32, output_proj_dims=32)
    return transformers.Gemma4UnifiedConfig(
        text_config=text, vision_config=vis if vision else None, audio_config=None, **ids
    )


def decision_config(kind: str, tok, **overrides) -> dict:
    base_model, base_revision = BASE[kind]
    d = {
        "interface": "strands-decider",
        "format_version": 1,
        "prompt": "strands-chat",
        "inputs": ["text", "image"],
        "codes": list(CODES),
        "token_ids": [tok.token_to_id(c) for c in CODES],
        "temperature": TEMPERATURE,
        "final_logit_softcapping": SOFTCAP,
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "max_length_validated": 4096,
        "provenance": {"base_model": base_model, "base_revision": base_revision},
    }
    d.update(overrides)
    return d


def processor_config(kind: str, **overrides) -> dict:
    ip = {
        "do_convert_rgb": True,
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0],
        "image_std": [1.0, 1.0, 1.0],
        "image_processor_type": PROCESSOR_TYPES[kind],
        "max_soft_tokens": 280,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "resample": 3,
        "rescale_factor": 1 / 255,
    }
    ip.update(overrides)
    return {"image_processor": ip, "image_seq_length": 280, "processor_class": "Gemma4Processor"}


def _readout(path: Path, hidden: int, seed: int) -> None:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    # Large enough that the soft-cap and the bf16 rounding both matter.
    w = (torch.randn(len(CODES), hidden, generator=g) * 0.6).to(torch.bfloat16)
    save_file({"weight": w}, str(path / "readout.safetensors"))


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
    tok = tokenizer()
    torch.manual_seed(seed)
    cls = (
        transformers.Gemma4UnifiedForConditionalGeneration
        if kind == UNIFIED
        else transformers.Gemma4ForConditionalGeneration
    )
    model = cls(model_config(kind, tok, text=text)).to(torch.bfloat16)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    tok.save(str(path / "tokenizer.json"))
    (path / "processor_config.json").write_text(
        json.dumps(processor_config(kind, **(processor or {})))
    )
    (path / "decision_config.json").write_text(
        json.dumps(decision_config(kind, tok, **(decision or {})))
    )
    _readout(path, (text or {}).get("hidden_size", HIDDEN), seed)
    return path


def intermediate_checkpoint(path: Path, *, seed: int = 0, decision: dict | None = None) -> Path:
    """A bare Gemma4TextModel with a decision_config.json from before the interface field."""
    tok = tokenizer()
    torch.manual_seed(seed)
    model = transformers.Gemma4TextModel(gemma4_text_config(tok.get_vocab_size()))
    path.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(str(path))
    tok.save(str(path / "tokenizer.json"))
    dc = {
        "format_version": 1,
        "base_model": BASE[GEMMA4][0],
        "revision": BASE[GEMMA4][1],
        "codes": list(CODES),
        "token_ids": [tok.token_to_id(c) for c in CODES],
        "temperature": TEMPERATURE,
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "finetune": {"epochs": 1.0},
    }
    dc.update(decision or {})
    (path / "decision_config.json").write_text(json.dumps(dc))
    _readout(path, HIDDEN, seed)
    return path


def image_b64(width: int, height: int, fmt: str = "PNG", seed: int = 0) -> str:
    """A noisy RGB image of `width` x `height` as base64, so that resizing moves pixels."""
    import base64
    import io

    import numpy as np
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
