#!/bin/bash
# Build the Lambda function zip, and quantize the model to Q8_0 on the way.
#
#   examples/lambda/convert_model.sh      # first, once per checkpoint
#   examples/lambda/build.sh
#
# Builds llama.cpp, the shim and the Python dependencies in amazonlinux:2023 (linux/arm64),
# quantizes build/model/decider-f16.gguf to build/model/decider-q8_0.gguf (about 2 GB) with the
# llama-quantize built there, and writes build/function.zip (about 50 MB). Needs an
# arm64-capable container runtime (CONTAINER=finch or docker; default finch).
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
CONTAINER=${CONTAINER:-finch}
M=build/model

for f in "$M/decider-f16.gguf" "$M/assets/head_config.json" build/llama.cpp/CMakeLists.txt; do
  [ -e "$f" ] || { echo "missing $f -- run: examples/lambda/convert_model.sh"; exit 1; }
done

echo ">> build in amazonlinux:2023 (linux/arm64)"
"$CONTAINER" run --rm --platform linux/arm64 \
  -v "$HERE:/recipe:ro" \
  -v "$HERE/../../src/strands_decider:/pkgsrc:ro" \
  -v "$HERE/build/llama.cpp:/llama:ro" \
  -v "$HERE/$M:/model" \
  -v "$HERE/build:/out" \
  amazonlinux:2023 bash /recipe/container_build.sh

ls -la "$M/decider-q8_0.gguf" build/function.zip
