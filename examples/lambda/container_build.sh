#!/bin/bash
# Runs INSIDE an arm64 Amazon Linux 2023 container (glibc 2.34, as the Lambda
# python3.12 arm64 runtime). Called by build.sh, which sets up these mounts:
#   /recipe  examples/lambda (ro)
#   /pkgsrc  src/strands_decider (ro)
#   /llama   llama.cpp source at the pinned commit (ro)
#   /model   build/model (rw): reads decider-f16.gguf, writes decider-q8_0.gguf
#   /out     build (rw): writes function.zip
set -euo pipefail

echo "=== toolchain ==="
dnf -q -y install cmake gcc gcc-c++ make binutils zip python3.12 python3.12-pip >/dev/null

echo "=== llama.cpp ==="
# armv8.2-a+fp16+dotprod and NO i8mm: Lambda mixes Graviton2 and Graviton3 hosts, a
# SnapStart snapshot can restore on either, and an i8mm instruction is SIGILL on
# Graviton2. dotprod (sdot) is on both.
rm -rf /build && cp -a /llama /build
cmake -S /build -B /build/b \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv8.2-a+fp16+dotprod" \
  -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON >/dev/null
cmake --build /build/b --target llama llama-quantize -j"$(nproc)" >/dev/null
n_smmla=$(objdump -d /build/b/bin/libggml-cpu.so | grep -c smmla || true)
[ "$n_smmla" = 0 ] || { echo "FATAL: libggml-cpu.so has $n_smmla i8mm (smmla) instructions"; exit 1; }
echo "no i8mm instructions"

if [ ! -f /model/decider-q8_0.gguf ] || [ /model/decider-f16.gguf -nt /model/decider-q8_0.gguf ]; then
  echo "=== quantize Q8_0 ==="
  /build/b/bin/llama-quantize /model/decider-f16.gguf /model/decider-q8_0.gguf Q8_0 2>&1 | tail -2
fi

echo "=== package ==="
P=/pkg/app
rm -rf /pkg && mkdir -p "$P/libs" "$P/assets" "$P/strands_decider"
cp -P /build/b/bin/*.so* "$P/libs/"
# The python3.12 runtime has no libgomp, and ggml-cpu is built with OpenMP.
cp -L /usr/lib64/libgomp.so.1 /usr/lib64/libstdc++.so.6 /usr/lib64/libgcc_s.so.1 "$P/libs/"
gcc -O3 -fPIC -shared -o "$P/libs/libdecider_hidden.so" /recipe/shim/decider_hidden.c \
  -I/build/include -I/build/ggml/include -L"$P/libs" -lllama

cp /recipe/app/main.py /recipe/app/engine.py /recipe/app/loader.py "$P/"
cp /recipe/app/run.sh /pkg/run.sh && chmod +x /pkg/run.sh
cp /model/assets/tokenizer.json /model/assets/head.safetensors /model/assets/head_config.json "$P/assets/"
# The package's prompt rendering and readouts import only pydantic: ship those modules,
# not the package (which pulls in torch).
cp /pkgsrc/__init__.py /pkgsrc/schema.py /pkgsrc/prompting.py "$P/strands_decider/"

# boto3 is in the Lambda runtime. Pinned to the versions this recipe was tested with.
python3.12 -m pip install -q --root-user-action=ignore --no-compile --only-binary=:all: --target "$P" \
  fastapi==0.143.0 pydantic==2.14.0 uvicorn==0.54.0 \
  numpy==2.5.3 tokenizers==0.23.2 safetensors==0.8.0
find "$P" -name __pycache__ -prune -exec rm -rf {} +

for so in "$P"/libs/*.so*; do
  [ -L "$so" ] && continue
  if LD_LIBRARY_PATH="$P/libs" ldd "$so" | grep -q "not found"; then
    echo "FATAL: unresolved libraries in $so"; LD_LIBRARY_PATH="$P/libs" ldd "$so"; exit 1
  fi
done
PYTHONPATH="$P" LD_LIBRARY_PATH="$P/libs" python3.12 -c "import fastapi, uvicorn, engine; print('imports OK')"
echo "unzipped: $(du -sh /pkg | cut -f1) (Lambda limit 250 MB)"

rm -f /out/function.zip
(cd /pkg && zip -q -r -X /out/function.zip .)
echo "function.zip: $(du -h /out/function.zip | cut -f1)"
