"""Tiny random-weight Strands Decider checkpoints for the GPU tier: the real layouts and tokenizer, a few MB of layers.

    python tiny.py BASE_DIR OUT_DIR [--layout multimodal|unified|decoder] [--inputs text,image] [--seed N]
                   [--vision-bidir] [--hetero-heads]

BASE_DIR holds a Gemma 4 base's config.json, processor_config.json, tokenizer.json and tokenizer_config.json: those of
google/gemma-4-E2B-it for the multimodal and decoder layouts, google/gemma-4-12B-it for the unified one (tests/gpu/ci.sh
fetches them at pinned revisions). The checkpoint keeps the base's architecture, every config field the plugin and
vLLM read, the image processor and the 262,144-token vocabulary, and shrinks the layers:

- multimodal (Gemma4ForConditionalGeneration, as E2B): 6 decoder layers, sliding and full (sliding window 64, so a
  window left on would change the answers), the last 2 sharing their KV with earlier layers as E2B's do, per-layer
  embeddings, a 2-layer vision encoder, no audio encoder;
- unified (Gemma4UnifiedForConditionalGeneration, as the 12B): the same decoder without KV sharing or per-layer
  embeddings, keys equal to values as in the 12B, and the encoder-free patch embedder;
- decoder: the intermediate checkpoints' bare Gemma4TextModel (text only).

``--vision-bidir`` keeps (multimodal) or sets the base text config's ``use_bidirectional_attention: "vision"``, as the
12B, 26B-A4B and 31B carry it; the unified base has it already. ``--hetero-heads`` gives the full-attention layers a
head size twice the sliding layers', as every Gemma 4 size has, which on Hopper selects FlashAttention 4 (vLLM's Gemma 4
hook) and elsewhere a backend the plugin refuses; without it both head sizes are 64, which FlashAttention 2 and 3 run.

The weights are transformers' own random initialization (seeded) with the attention's q/k norms scaled so that bf16
rounding does not decide the answers (:func:`condition_attention`), saved by transformers, plus a random readout of
standard deviation 0.25 and a decision_config.json in the specification's schema with prompt strands-chat.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CODES_FILE = (
    None  # the codes come from the core library when installed, else from the tokenizer below
)

SMALL_TEXT = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=6,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    global_head_dim=64,
    sliding_window=64,
)


def codes() -> list[str]:
    """The 255 answer codes: A to Z, AA to IV without GZ (specification, section 2)."""
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    out = list(letters)
    for a in letters:
        for b in letters:
            out.append(a + b)
            if a + b == "IV":
                return [c for c in out if c != "GZ"]
    raise AssertionError


#: The standard deviation of the tiny models' attention logits at initialization (see condition_attention).
LOGIT_STD = 1.5


def condition_attention(model) -> int:
    """Set every q_norm and k_norm weight so that attention logits have a standard deviation of LOGIT_STD.

    Gemma 4 normalizes each head's queries and keys and scales their product by 1.0, so with its norms at their
    initial weight of 1 a random model's attention logits have a standard deviation of sqrt(head size), 8 at 64: a
    near-hard maximum over the keys, which bf16 rounding flips (the tiny models' bf16 answers then differ from their
    own float32 answers on 15% of questions). A trained model learns smaller norm weights; these tiny ones get
    (LOGIT_STD / sqrt(d)) ** 0.5 for each, which keeps attention smooth, so bf16 and float32 agree, and still far
    from uniform, so the mask changes the answers."""
    import math

    import torch

    n = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            w = getattr(m, "weight", None)
            if name.endswith(("q_norm", "k_norm")) and w is not None:
                w.fill_((LOGIT_STD / math.sqrt(w.numel())) ** 0.5)
                n += 1
    return n


def text_config(
    base: dict,
    layout: str,
    hetero: bool,
    bidir: bool,
    head_dim: int | None = None,
    rotary: float | None = None,
) -> dict:
    tc = dict(base)
    tc.update(SMALL_TEXT)
    if head_dim:
        tc["head_dim"] = tc["global_head_dim"] = head_dim
    if hetero:
        tc["global_head_dim"] = 2 * tc["head_dim"]
    if rotary is not None:
        rp = {k: dict(v) for k, v in tc["rope_parameters"].items()}
        rp["full_attention"]["partial_rotary_factor"] = rotary
        tc["rope_parameters"] = rp
    if layout == "multimodal" or layout == "decoder":
        tc["num_kv_shared_layers"] = 2
        tc["hidden_size_per_layer_input"] = 16
    else:
        tc["num_kv_shared_layers"] = 0
        tc["num_global_key_value_heads"] = 1
    if bidir and layout != "decoder":
        tc["use_bidirectional_attention"] = "vision"
    tc.pop("per_layer_config", None)
    return tc


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=("multimodal", "unified", "decoder"), default="multimodal")
    ap.add_argument("--inputs", default="text,image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vision-bidir", action="store_true")
    ap.add_argument("--hetero-heads", action="store_true")
    ap.add_argument("--head-dim", type=int, default=None, help="both head sizes (default 64)")
    ap.add_argument(
        "--rotary",
        type=float,
        default=None,
        help="the full layers' partial_rotary_factor (base: 0.25)",
    )
    ap.add_argument("--base-model", default="google/gemma-4-E2B-it")
    ap.add_argument("--base-revision", default="3e22461f65e89153144f8adb70e3b8c2cc9845a7")
    a = ap.parse_args()

    import torch
    import transformers
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    torch.manual_seed(a.seed)
    base = json.loads((a.base / "config.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    tc = text_config(
        base["text_config"], a.layout, a.hetero_heads, a.vision_bidir, a.head_dim, a.rotary
    )
    if a.layout == "decoder":
        cfg = transformers.Gemma4TextConfig(**{k: v for k, v in tc.items() if k != "model_type"})
        cfg.architectures = ["Gemma4TextModel"]
        model = transformers.Gemma4TextModel(cfg)
    else:
        c = dict(base)
        c["text_config"] = tc
        c["audio_config"] = None
        vc = dict(base["vision_config"])
        if a.layout == "multimodal":
            vc.update(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=32,
                global_head_dim=32,
            )
            c["vision_config"] = vc
            cfg = transformers.Gemma4Config(**{k: v for k, v in c.items() if k != "model_type"})
            model = transformers.Gemma4ForConditionalGeneration(cfg)
        else:
            vc.update(
                mm_embed_dim=SMALL_TEXT["hidden_size"], output_proj_dims=SMALL_TEXT["hidden_size"]
            )
            c["vision_config"] = vc
            cfg = transformers.Gemma4UnifiedConfig(
                **{k: v for k, v in c.items() if k != "model_type"}
            )
            model = transformers.Gemma4UnifiedForConditionalGeneration(cfg)
        cfg.architectures = base["architectures"]
    conditioned = condition_attention(model)
    model = model.to(torch.bfloat16)
    model.save_pretrained(a.out, safe_serialization=True)
    (a.out / "generation_config.json").unlink(missing_ok=True)

    for f in ("processor_config.json", "tokenizer.json"):
        if (a.base / f).exists() and (a.layout != "decoder" or f == "tokenizer.json"):
            shutil.copy(a.base / f, a.out / f)
    tok_cfg = json.loads((a.base / "tokenizer_config.json").read_text())
    tok_cfg.pop("chat_template", None)
    (a.out / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))
    (a.out / "chat_template.jinja").unlink(missing_ok=True)

    hidden = SMALL_TEXT["hidden_size"]
    readout = (torch.randn(255, hidden) * 0.25).to(torch.bfloat16)
    save_file({"weight": readout.contiguous()}, str(a.out / "readout.safetensors"))
    tok = Tokenizer.from_file(str(a.out / "tokenizer.json"))
    cs = codes()
    ids = [tok.token_to_id(c) for c in cs]
    assert all(i is not None for i in ids) and len(set(ids)) == 255, "a code is not one token"
    dc = {
        "interface": "strands-decider",
        "format_version": 1,
        "prompt": "strands-chat",
        "inputs": a.inputs.split(","),
        "codes": cs,
        "token_ids": ids,
        "temperature": 1.0,
        "final_logit_softcapping": tc["final_logit_softcapping"],
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "max_length_validated": 4096,
        "provenance": {"base_model": a.base_model, "base_revision": a.base_revision},
    }
    if (
        a.layout == "decoder"
    ):  # an intermediate checkpoint: its decision_config.json predates these fields
        for k in ("interface", "inputs", "provenance", "prompt", "max_length_validated"):
            dc.pop(k)
    (a.out / "decision_config.json").write_text(json.dumps(dc, indent=2))
    n = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "out": str(a.out),
                "layout": a.layout,
                "parameters": n,
                "inputs": dc.get("inputs"),
                "conditioned_norms": conditioned,
            }
        )
    )


if __name__ == "__main__":
    main()
