"""Write what the Lambda image serves from a strands-decider checkpoint.

    python prepare_model.py MODEL REVISION OUT_DIR [VISION] [DECODER_PRECISION]

MODEL is a Hugging Face Hub id. The script writes OUT_DIR/torso, the torso with its LoRA folded
in, as the Qwen3_5ForCausalLM checkpoint llama.cpp's convert_hf_to_gguf.py reads, and
OUT_DIR/assets, the tokenizer, pointer head and settings HiddenStateEngine.from_assets reads.
With VISION `true`, OUT_DIR/vision links to the base model at the revision MODEL records: its
vision tower, which the adapter leaves as it is, for convert_hf_to_gguf.py --mmproj. The model is
named for DECODER_PRECISION (q8_0 by default), the precision the Dockerfile gives the decoder.
The Dockerfile runs it.
"""

from __future__ import annotations

import os
import sys

import torch
import transformers

from strands_decider.hidden_engine import save_assets
from strands_decider.modeling import StrandsDeciderModel, checkpoint_dir


def main(
    model_id: str, revision: str, out: str, vision: str = "false", decoder: str = "q8_0"
) -> None:
    model = StrandsDeciderModel.load(checkpoint_dir(model_id, revision))
    torso = (
        model.torso.merge_and_unload() if hasattr(model.torso, "merge_and_unload") else model.torso
    )
    if not isinstance(torso.config, transformers.Qwen3_5TextConfig):
        raise SystemExit(
            f"{model_id}: a {type(torso.config).__name__} torso; only Qwen3.5 is supported"
        )
    # The torso is the decoder without its LM head, and the converter reads a causal LM. llama.cpp's
    # embeddings are the decoder's final normed hidden states, which the head reads.
    lm = transformers.AutoModelForCausalLM.from_config(torso.config, dtype=torch.float16)
    lm.model.load_state_dict(torso.state_dict())
    lm.save_pretrained(os.path.join(out, "torso"))
    model.tokenizer.save_pretrained(os.path.join(out, "torso"))
    # Named as `strands-decider serve` names it, and for the decoder's precision in the image.
    save_assets(model, os.path.join(out, "assets"), f"{model_id.split('/')[-1]}-{decoder}")
    if vision == "true":
        from huggingface_hub import snapshot_download

        # The weights, the config, and the image processor's preprocessor_config.json.
        base = snapshot_download(
            model.config.base_model,
            revision=model.config.base_revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        os.symlink(base, os.path.join(out, "vision"))
    elif vision != "false":
        raise SystemExit(f"VISION must be true or false, not {vision!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
