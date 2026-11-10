#!/bin/bash
# The CI tier: the plugin end to end on a CPU build of vLLM 0.33, with no GPU (GitHub's hosted Linux runners).
#
# usage: tests/cpu/ci.sh OUT_DIR
#
# It is tests/gpu/ci.sh with DEVICE=cpu: tiny random-weight checkpoints in every layout, the plugin's server and vLLM's,
# text and image requests, refusals, the image-content cache test and state reuse, and every answer against the
# transformers reference, in float32 with 32-token KV blocks, one server at a time. FULL=1 adds the text-only server and
# the cross-server comparison. Needs vLLM's x86 CPU wheel (README, Tests), the plugin, the core library, Pillow, curl,
# g++ (torch's compiler) and libnuma.
set -u
DEVICE=cpu exec bash "$(cd "$(dirname "$0")" && pwd)/../gpu/ci.sh" "$@"
