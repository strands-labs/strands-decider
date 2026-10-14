#!/bin/bash
# Run the packaged function zip once in amazonlinux:2023 (linux/arm64), with the GGUF
# from build/model instead of S3, and print the answers to client.py's request.
#
#   examples/lambda/local_test.sh
set -euo pipefail
cd "$(dirname "$0")"
CONTAINER=${CONTAINER:-finch}
"$CONTAINER" run --rm --platform linux/arm64 \
  -v "$PWD/build:/build:ro" -v "$PWD/client.py:/client.py:ro" \
  amazonlinux:2023 bash -c '
    dnf -q -y install python3.12 unzip >/dev/null
    mkdir /task && cd /task && unzip -q /build/function.zip
    PYTHONPATH=/task/app:/ LD_LIBRARY_PATH=/task/app/libs python3.12 -c "
import json, os, time
from client import REQUEST
from engine import DeciderEngine
e = DeciderEngine(\"app/libs/libdecider_hidden.so\", \"/build/model/decider-q8_0.gguf\",
                  \"app/assets\", os.cpu_count())
t0 = time.perf_counter()
out = e.evaluate(REQUEST, \"local\")
print(json.dumps(out, indent=1))
print(f\"{(time.perf_counter() - t0) * 1000:.0f} ms\")
"'
