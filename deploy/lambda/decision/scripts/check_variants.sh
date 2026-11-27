#!/bin/sh
# shim_check.py once per CPU module this host loads, each alone: check_variants.sh SRC LIBDIR (docs/deployment/graviton-cpu-variants.md).
set -eu

SRC=$1
LIBDIR=$2
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
status=0

for module in "$LIBDIR"/libggml-cpu-*.so; do
    name=$(basename "$module" .so)
    dir="$WORK/$name"
    mkdir -p "$dir"
    for f in "$LIBDIR"/*.so*; do
        case $(basename "$f") in libggml-cpu-*) ;; *) cp -a "$f" "$dir/" ;; esac
    done
    cp -a "$module" "$dir/"
    loaded=$(python3 - "$dir/libdecision.so" 2>/dev/null <<'EOF'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.sd_cpu_backend.restype = ctypes.c_char_p
lib.sd_load.restype = ctypes.c_void_p
n = ctypes.c_int()
lib.sd_load(b"/nonexistent.gguf", 1, ctypes.byref(n), ctypes.byref(n))  # loads the backends, then fails
print(lib.sd_cpu_backend().decode())
EOF
    )
    if [ "$loaded" != "$name.so" ]; then
        echo "$name: skipped (this host does not load it)"
        continue
    fi
    # 0.015, not 0.01: docs/deployment/graviton-cpu-variants.md#the-image-builds-tiny-model-check
    if python3 "$SCRIPTS/shim_check.py" --llama-cpp "$SRC" --lib "$dir/libdecision.so" --golden-dp 0.015 > "$WORK/$name.json" 2> "$WORK/$name.err"; then
        echo "$name: passed"
    else
        echo "$name: FAILED"; tail -5 "$WORK/$name.err"
        python3 -c 'import json,sys; t=open(sys.argv[1]).read(); r,_=json.JSONDecoder().raw_decode(t[t.find("{"):]); print("  ", r.get("failures"))' "$WORK/$name.json" || true
        status=1
    fi
done
exit $status
