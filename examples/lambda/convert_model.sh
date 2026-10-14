#!/bin/bash
# Convert a decider checkpoint into the files the Lambda recipe serves. No weights are
# committed to the repo; everything this writes lands in the git-ignored build/ directory.
#
#   examples/lambda/convert_model.sh [CKPT]
#
# CKPT is a checkpoint path or Hub id (default: prepare_model.py's default, the v1 release).
# Writes, under examples/lambda/build/:
#   llama.cpp/                   llama.cpp at the pinned commit (its GGUF converter and sources)
#   model/merged-hf/             the torso with the LoRA folded in, fp16 (about 4 GB)
#   model/assets/                tokenizer.json, head.safetensors, head_config.json
#   model/decider-f16.gguf       the torso as an f16 GGUF (about 4 GB)
# build.sh then quantizes the f16 GGUF to Q8_0 with the llama-quantize it builds.
#
# Needs the package with its torch dependencies (`pip install -e .`) and git. Loading and
# merging the checkpoint peaks at about 10-12 GB of RAM: close other heavy processes (a
# container VM, for one) before running it.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
# The llama.cpp commit this recipe was tested with (qwen35 support, llama_memory_seq_cp).
LLAMA_CPP_REV=6184e92c57dcd34de8a3e381d7641a5e75250d5f
M=build/model

if [ "$(git -C build/llama.cpp rev-parse HEAD 2>/dev/null)" != "$LLAMA_CPP_REV" ]; then
  echo ">> llama.cpp @ ${LLAMA_CPP_REV:0:7}"
  rm -rf build/llama.cpp && mkdir -p build/llama.cpp
  git -C build/llama.cpp init -q
  git -C build/llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_REV"
  git -C build/llama.cpp checkout -q FETCH_HEAD
fi

echo ">> merge the LoRA and export the head"
"$PYTHON" prepare_model.py "$@" --out "$M"

echo ">> convert merged torso -> f16 GGUF"
PYTHONPATH=build/llama.cpp/gguf-py "$PYTHON" build/llama.cpp/convert_hf_to_gguf.py \
  "$M/merged-hf" --outtype f16 --outfile "$M/decider-f16.gguf"

ls -la "$M/decider-f16.gguf" "$M/assets"
echo "next: examples/lambda/build.sh"
