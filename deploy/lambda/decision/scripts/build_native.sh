#!/bin/sh
# Build llama.cpp at the pinned commit with patches/, and shim.c against it, into one library.
#
#   build_native.sh SRC BUILD [--lambda | --lambda-variants]
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
#
# --lambda-variants builds a CPU backend module per Arm tier and per Graviton generation (needs GCC 14); see docs/deployment/graviton-cpu-variants.md.
# GRAVITON_SVE=1 adds SVE Graviton modules, off by default: why, and when to change that, is in that doc.
# KLEIDIAI=1 adds Arm's KleidiAI kernels; the CPU modules then hold its runtime-dispatched SVE/SME code and skip the instruction checks.
set -eu

DECISION=$(cd "$(dirname "$0")/.." && pwd)
SRC=$1
BUILD=$2
LAMBDA=${3:-}
REV=$(tr -d ' \n' < "$DECISION/LLAMA_CPP_REV")
JOBS=${JOBS:-$(nproc)}
CC=${CC:-gcc}
CXX=${CXX:-g++}
export CC CXX

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

SHARED=OFF
if [ "$LAMBDA" = "--lambda" ]; then
    # Graviton2 and later. Lambda picks the host's generation (Graviton2, 3 or 4) for each execution
    # environment, so the library may use no instruction Graviton2 lacks (i8mm, bf16, SVE); the check
    # below enforces it. --lambda-variants is faster on Graviton3 to 5 from the same image: it picks
    # each host's own module at load (docs/deployment/graviton-cpu-variants.md).
    ARCH_FLAGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod"
elif [ "$LAMBDA" = "--lambda-variants" ]; then
    if [ "$("$CC" -dumpversion | cut -d. -f1)" -lt 14 ] || [ "$("$CXX" -dumpversion | cut -d. -f1)" -lt 14 ]; then
        echo "build_native.sh: --lambda-variants needs GCC 14 or later (CC=$CC, CXX=$CXX)" >&2
        exit 1
    fi
    SHARED=ON
    ARCH_FLAGS="-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CPU_GRAVITON_VARIANTS=ON"
    if [ "${GRAVITON_SVE:-0}" = 1 ]; then
        ARCH_FLAGS="$ARCH_FLAGS -DGGML_CPU_GRAVITON_SVE=ON"
    fi
elif [ -n "$LAMBDA" ]; then
    echo "build_native.sh: unknown option $LAMBDA" >&2
    exit 1
else
    ARCH_FLAGS="-DGGML_NATIVE=ON"
fi
if [ "${KLEIDIAI:-0}" = 1 ]; then
    ARCH_FLAGS="$ARCH_FLAGS -DGGML_CPU_KLEIDIAI=ON"
fi
# shellcheck disable=SC2086
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=$SHARED \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON $ARCH_FLAGS -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build "$BUILD" --target llama llama-quantize mtmd -j"$JOBS"
if [ "$SHARED" = ON ]; then
    # The CPU variants are built with libggml, which depends on each of them; nothing links to them.
    "$CC" -O3 -fPIC -shared -o "$BUILD/bin/libdecision.so" "$DECISION/shim.c" \
        -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/tools/mtmd" -Wl,--no-undefined \
        -L"$BUILD/bin" -Wl,-rpath,'$ORIGIN' -lmtmd -lllama -lggml -lggml-base \
        -lstdc++ -lm -lpthread -ldl
    LIB="$BUILD/bin/libdecision.so"
    ls "$BUILD"/bin/libggml-cpu-*.so > /dev/null \
        || { echo "build_native.sh: no libggml-cpu-*.so variants were built" >&2; exit 1; }
else
    # --no-undefined: a symbol no archive defines fails the build, not the library's first load.
    # shellcheck disable=SC2046
    "$CC" -O3 -fPIC -shared -o "$BUILD/libdecision.so" "$DECISION/shim.c" \
        -I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/tools/mtmd" -Wl,--no-undefined \
        -Wl,--start-group $(find "$BUILD" -name '*.a') -Wl,--end-group \
        -fopenmp -lstdc++ -lm -lpthread -ldl
    LIB="$BUILD/libdecision.so"
fi

# i8mm (smmla, ummla, usmmla, usdot, sudot), bf16 (bfdot, bfmmla, bfmlal*, bfcvt*), and SVE,
# whose instructions name z or p registers: none of them runs on Graviton2.
graviton2_clean() {
    objdump -f "$1" | grep -q "architecture: aarch64" \
        || { echo "build_native.sh: $(basename "$1") is not arm64" >&2; exit 1; }
    objdump -d --no-show-raw-insn "$1" > "$1.s"
    if grep -qiE '\s(smmla|ummla|usmmla|usdot|sudot|bfdot|bfmmla|bfmlal[bt]|bfcvtn?2?)\s' "$1.s" \
        || grep -qE '\s[a-z0-9.]+\s+[^;]*\b(z[0-9]{1,2}\.[bhsdq]|p[0-9]{1,2}/[mz])\b' "$1.s"; then
        echo "build_native.sh: $(basename "$1") has instructions Graviton2 lacks" >&2
        exit 1
    fi
    rm -f "$1.s"
    echo "$(basename "$1"): arm64, no i8mm, bf16 or SVE"
}
# The Graviton4 and 5 modules are +nosve (docs/deployment/graviton-cpu-variants.md): no z or p register.
no_sve() {
    objdump -d --no-show-raw-insn "$1" > "$1.s"
    if grep -qE '\s[a-z0-9.]+\s+[^;]*\b(z[0-9]{1,2}\.[bhsdq]|p[0-9]{1,2}/[mz])\b' "$1.s"; then
        echo "build_native.sh: $(basename "$1") holds SVE instructions; it is meant to be +nosve" >&2
        exit 1
    fi
    rm -f "$1.s"
    echo "$(basename "$1"): arm64, no SVE"
}
if [ "$LAMBDA" = "--lambda" ]; then
    graviton2_clean "$LIB"
elif [ "$LAMBDA" = "--lambda-variants" ]; then
    for so in "$BUILD"/bin/*.so; do
        case $(basename "$so") in
            # Loaded only where the host has SVE, i8mm or SME (ggml-cpu/arch/arm/cpu-feats.cpp).
            libggml-cpu-armv8.2_3.so | libggml-cpu-armv8.6_*.so | libggml-cpu-armv9.*.so) ;;
            libggml-cpu-graviton[45]_s*.so) ;;  # GRAVITON_SVE=1's
            libggml-cpu-graviton3.so) ;;  # SVE, loaded only on Graviton3
            libggml-cpu-graviton[45].so) [ "${KLEIDIAI:-0}" = 1 ] || no_sve "$so" ;;
            libggml-cpu-*.so) [ "${KLEIDIAI:-0}" = 1 ] || graviton2_clean "$so" ;;
            *) graviton2_clean "$so" ;;
        esac
    done
fi
echo "built $LIB against llama.cpp $REV"
