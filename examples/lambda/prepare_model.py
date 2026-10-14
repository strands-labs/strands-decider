"""Export a 2B Qwen3.5 decider checkpoint for the llama.cpp Lambda recipe.

Writes, under OUT (default examples/lambda/build/model):
  merged-hf/      the torso with the LoRA folded in, as a plain fp16 Qwen3_5ForCausalLM
                  checkpoint that llama.cpp's convert_hf_to_gguf.py reads
  assets/         what the Lambda app reads besides the GGUF: tokenizer.json,
                  head.safetensors (the pointer head) and head_config.json

usage: python examples/lambda/prepare_model.py [CKPT] [--out OUT]

CKPT is a checkpoint path or Hub id (default StrandsAgents/strands-decider-2B-qwen3.5-v1-2610).
Needs the package with torch, transformers and peft (`pip install -e .`); runs on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import transformers
from safetensors.torch import save_file

from strands_decider.modeling import StrandsDeciderModel

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("ckpt", nargs="?", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "build", "model"))
    args = ap.parse_args()

    merged = os.path.join(args.out, "merged-hf")
    assets = os.path.join(args.out, "assets")
    os.makedirs(assets, exist_ok=True)

    with torch.inference_mode():
        model = StrandsDeciderModel.load(args.ckpt)
        cfg = model.config
        if cfg.head_type != "pointer":
            raise SystemExit(f"this recipe reads a pointer head; {args.ckpt} has {cfg.head_type!r}")
        torso = (
            model.torso.merge_and_unload()
            if hasattr(model.torso, "merge_and_unload")
            else model.torso
        )
        if not isinstance(torso.config, transformers.Qwen3_5TextConfig):
            raise SystemExit(
                f"this recipe converts Qwen3.5 torsos; got {type(torso.config).__name__}"
            )

        # The torso is the decoder without its LM head. Wrap it back into the causal LM
        # so convert_hf_to_gguf sees a standard checkpoint; llama.cpp's embeddings output
        # is the post-final-norm hidden state, which is what the head reads.
        lm = transformers.Qwen3_5ForCausalLM(torso.config)
        lm.model.load_state_dict(torso.state_dict(), strict=True)
        lm.to(torch.float16).save_pretrained(merged, safe_serialization=True)
        model.tokenizer.save_pretrained(merged)
        print(f"merged torso -> {merged}")

        head = {k: v.detach().float().contiguous() for k, v in model.head.state_dict().items()}
        save_file(head, os.path.join(assets, "head.safetensors"))
        with open(os.path.join(assets, "head_config.json"), "w") as f:
            json.dump(
                {
                    "checkpoint": args.ckpt,
                    "pointer_dim": cfg.pointer_dim,
                    "temperature": cfg.temperature,
                    "temperature_by_kind": cfg.temperature_by_kind,
                    "max_length": cfg.max_length,
                    "ordinal_smoothing": cfg.ordinal_smoothing,
                    "hidden_size": StrandsDeciderModel.hidden_size(model.torso),
                },
                f,
                indent=2,
            )
        shutil.copyfile(
            os.path.join(merged, "tokenizer.json"), os.path.join(assets, "tokenizer.json")
        )
        print(f"head, head_config.json, tokenizer.json -> {assets}")


if __name__ == "__main__":
    main()
