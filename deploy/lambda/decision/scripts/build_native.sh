#!/bin/sh
# Build llama.cpp at the pinned commit with patches/, and shim.c against it, into one library.
#
#   build_native.sh SRC BUILD [--lambda]
#
# SRC is where the llama.cpp source goes: an empty or missing directory is fetched at the pin
# (LLAMA_CPP_REV, beside patches/), and an existing checkout must already be at it. Each patch is
# checked, then applied, and the build stops at the first that does not apply. SRC/LLAMA_CPP_REV
# records the commit for the image build's artifact.json. BUILD gets the static libraries,
# bin/llama-quantize and libdecision.so.
#
# --lambda builds for every Graviton Lambda runs (armv8.2-a+fp16+dotprod) and checks the library:
# it must be arm64 and hold none of the instructions Graviton2 lacks (i8mm, bf16, SVE), which kill
# the process there. Without it the build is native, for the machine it runs on (tests, the cluster).
set -eu

DECISION=$(cd "$(dirname "$0")/.." && pwd)
SRC=$1
BUILD=$2
LAMBDA=${3:-}
REV=$(tr -d ' \n' < "$DECISION/LLAMA_CPP_REV")
JOBS=${JOBS:-$(nproc)}

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$SRC"
    git -C "$SRC" init -q
    git -C "$SRC" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$REV"
    git -C "$SRC" checkout -q FETCH_HEAD
fi
if [ "$(git -C "$SRC" rev-parse HEAD)" != "$REV" ]; then
    echo "build_native.sh: $SRC is at $(git -C "$SRC" rev-parse HEAD), not the pinned $REV" >&2
    exit 1
fi
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
    echo "build_native.sh: $SRC has local changes; give it a clean checkout at the pin" >&2
    exit 1
fi
for p in "$DECISION"/patches/*.patch; do
    git -C "$SRC" apply --check "$p" || { echo "build_native.sh: $(basename "$p") does not apply at $REV" >&2; exit 1; }
    git -C "$SRC" apply "$p"
    echo "applied $(basename "$p")"
done
echo "$REV" > "$SRC/LLAMA_CPP_REV"

if [ "$LAMBDA" = "--lambda" ]; then
    ARCH_FLAGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod"
elif [ -n "$LAMBDA" ]; then
    echo "build_native.sh: unknown option $LAMBDA" >&2
    exit 1
else
    ARCH_FLAGS="-DGGML_NATIVE=ON"
fi
# shellcheck disable=SC2086
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON $ARCH_FLAGS -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build "$BUILD" --target llama llama-quantize mtmd -j"$JOBS"
# --no-undefined: a symbol no archive defines fails the build, not the library's first load.
# shellcheck disable=SC2046
gcc -O3 -fPIC -shared -o "$BUILD/libdecision.so" "$DECISION/shim.c" \
    -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/tools/mtmd" -Wl,--no-undefined \
    -Wl,--start-group $(find "$BUILD" -name '*.a') -Wl,--end-group \
    -fopenmp -lstdc++ -lm -lpthread -ldl
for f in sd_load sd_free sd_load_vision sd_vision_info sd_n_embd_inp sd_encode_image sd_forward; do
    nm -D --defined-only "$BUILD/libdecision.so" | grep -q " T $f\$" || { echo "build_native.sh: no $f" >&2; exit 1; }
done

if [ "$LAMBDA" = "--lambda" ]; then
    objdump -f "$BUILD/libdecision.so" | grep -q "architecture: aarch64" \
        || { echo "build_native.sh: libdecision.so is not arm64" >&2; exit 1; }
    # i8mm (smmla, ummla, usmmla, usdot, sudot), bf16 (bfdot, bfmmla, bfmlal*, bfcvt*), and SVE,
    # whose instructions name z or p registers: none of them runs on Graviton2.
    objdump -d --no-show-raw-insn "$BUILD/libdecision.so" > "$BUILD/libdecision.s"
    if grep -qiE '\s(smmla|ummla|usmmla|usdot|sudot|bfdot|bfmmla|bfmlal[bt]|bfcvtn?2?)\s' "$BUILD/libdecision.s" \
        || grep -qE '\s[a-z0-9.]+\s+[^;]*\b(z[0-9]{1,2}\.[bhsdq]|p[0-9]{1,2}/[mz])\b' "$BUILD/libdecision.s"; then
        echo "build_native.sh: libdecision.so has instructions Graviton2 lacks" >&2
        exit 1
    fi
    echo "libdecision.so: arm64, no i8mm, bf16 or SVE"
fi
echo "built $BUILD/libdecision.so against llama.cpp $REV"
