#!/usr/bin/env bash
# Eyes for MiniCPM5-2B: a frozen SigLIP2 so400m encoder and a trained projector grafted onto
# the v21 MiniCPM5 text checkpoint (src/strands_decider/graft.py), end to end. Linux with an
# NVIDIA GPU. Exploratory.
#
#   1. fetch   captions_train2014.json (COCO annotations_trainval2014.zip; annotations CC BY 4.0)
#              into data/raw/coco/, and SigLIP2 so400m and MiniCPM5-2B at their pinned commits
#              into the Hub cache. Stage 2 trains on the image rows of training/recipe_images.sh:
#              run its `fetch build dedupe` first (data/image/build/*.jsonl).
#   2. build   80,000 caption rows, one caption per image, train2014 only (data/image/captions.py
#              refuses val2014, POPE's images), and those images from images.cocodataset.org
#              (~13 GB; images already in data/image/build/coco/ are kept)
#   3. dedupe  the caption images against every evaluation image (data/checks/dedupe_images.py)
#   4. align   stage 1, the projector alone, on the frozen base
#              (configs/align/strands-decider-2.5B-minicpm-v21-vl.yaml)
#   5. train   stage 2 for each of SEEDS (configs/vision/strands-decider-2.5B-minicpm-v21-vl*.yaml)
#   6. eval    each stage-2 checkpoint: NaturalBench, POPE and, with IJB_JSONL set, the rebuilt
#              Image JevBench preview items, with and without the image; image temperatures fitted
#              on NaturalBench groups 300-599 and applied; the 899-row text check; and JevBench
#              (text) through `strands-decider serve --vision`, then the v1.5-rule proxy
#
# Usage: training/recipe_minicpm_vision.sh STEP [STEP ...]
# Env: INIT (stage 2's text checkpoint, default the configs' checkpoints/strands-decider-2.5B-minicpm-v21,
# the v21 soup), ALIGN (stage 1's output, default the align config's output_dir), SEEDS (default
# "0 1 2"), PY (python), GPU and PORT (JevBench's server, default 0 and 8099), IJB_JSONL
# (optional), JEVBENCH=0 to skip JevBench.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # every path below is relative to the repo root
export PY="${PY:-python}" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
RAW=data/raw
DATA=data/image/build
ALIGN_CONFIG=configs/align/strands-decider-2.5B-minicpm-v21-vl.yaml
ALIGN="${ALIGN:-checkpoints/strands-decider-2.5B-minicpm-v21-vl-align}"
SEEDS="${SEEDS:-0 1 2}"
N_CAPTIONS="${N_CAPTIONS:-80000}"
SIGLIP=google/siglip2-so400m-patch16-384
SIGLIP_REV=dd658faac399427308559e2c3ac1e99cbe43845d
MINICPM=openbmb/MiniCPM5-2B
MINICPM_REV=f97400052a43d642bbc6e9975e2397e3ae6a6b52

download() { [ -f "$1" ] || { curl -fsSL -o "$1.part" "$2" && mv "$1.part" "$1"; }; }
config() {
  local c=configs/vision/strands-decider-2.5B-minicpm-v21-vl
  if [ "$1" = 0 ]; then echo "$c.yaml"; else echo "$c-seed$1.yaml"; fi
}
ckpt() { "$PY" -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$(config "$1")"; }

fetch() {
  mkdir -p "$RAW/coco"
  download "$RAW/coco/annotations_trainval2014.zip" \
    http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip -qo "$RAW/coco/annotations_trainval2014.zip" annotations/captions_train2014.json -d "$RAW/coco"
  "$PY" - "$SIGLIP" "$SIGLIP_REV" "$MINICPM" "$MINICPM_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
enc, enc_rev, lm, lm_rev = sys.argv[1:]
# the vision half is read from the full checkpoint; the tokenizer files are not needed
snapshot_download(enc, revision=enc_rev, allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors"])
snapshot_download(lm, revision=lm_rev)
print(f"{enc} @ {enc_rev}, {lm} @ {lm_rev}")
EOF
}

build() {
  mkdir -p "$DATA/coco"
  "$PY" data/image/captions.py --captions "$RAW/coco/annotations/captions_train2014.json" --out "$DATA" \
    --n "$N_CAPTIONS"
  # The train2014 images the rows use, and no others.
  sed 's#^coco/##' "$DATA/captions_images.txt" |
    xargs -P 64 -I{} sh -c '[ -f "$0/coco/$1" ] || curl -fsS --retry 3 -o "$0/coco/$1" "http://images.cocodataset.org/train2014/$1"' "$DATA" {}
  local missing
  missing=$(sed 's#^#'"$DATA"'/#' "$DATA/captions_images.txt" | while read -r f; do [ -s "$f" ] || echo "$f"; done | wc -l)
  [ "$missing" -eq 0 ] || { echo "$missing caption images failed to download; rerun build" >&2; exit 1; }
  (cd "$DATA" && sha256sum captions.jsonl)
}

dedupe() {
  "$PY" data/checks/dedupe_images.py --data-root "$DATA" ${IJB_JSONL:+--ijb-dir "$(dirname "$IJB_JSONL")"} \
    --files captions.jsonl
}

align() { "$PY" -m strands_decider.graft_align "$ALIGN_CONFIG" output_dir="$ALIGN"; }

train() {
  [ -f "$ALIGN/projector_config.json" ] || { echo "no stage-1 projector in $ALIGN; run align" >&2; exit 2; }
  local s
  for s in $SEEDS; do
    "$PY" -m strands_decider.vision_train "$(config "$s")" ${INIT:+"init_from=$INIT"} projector_from="$ALIGN"
  done
}

evaluate() {
  local s ck out run
  for s in $SEEDS; do
    ck=$(ckpt "$s")
    out="reports/$(basename "$ck")"
    run=(--systems strands --device cuda --checkpoint "$ck" --long-side 0 --max-pixels 400000)
    mkdir -p "$out"
    "$PY" evaluation/vision/run.py --out "$out/eval" "${run[@]}" --nb-groups 300 --pope 600 \
      ${IJB_JSONL:+--ijb-jsonl "$IJB_JSONL"}
    "$PY" evaluation/vision/run.py --out "$out/cal" "${run[@]}" --nb-start 300 --nb-groups 300 --pope 0 --no-blind
    "$PY" evaluation/vision/temps.py fit --run "$out/cal" --out "$out/image_temps.json"
    "$PY" evaluation/vision/temps.py apply --run "$out/eval" --temps "$out/image_temps.json" --out "$out/eval-T"
    "$PY" evaluation/vision/text_check.py --checkpoint "$ck" --out "$out/text_check.json"
    if [ "${JEVBENCH:-1}" != 0 ]; then
      # text questions through the vision server: they take the text path, unchanged
      PY="$(command -v "$PY")" GPU="${GPU:-0}" PORT="${PORT:-8099}" SERVE_ARGS=--vision \
        bash evaluation/jevbench/jevbench.sh "$ck" "$out/jevbench"
      "$PY" evaluation/jevbench/v15_proxy.py "$out/jevbench" --json "$out/jevbench/proxy.json" \
        > "$out/jevbench/proxy.txt"
    fi
  done
}

[ $# -gt 0 ] || { echo "usage: $0 STEP [STEP ...] (fetch build dedupe align train eval)" >&2; exit 2; }
for STEP in "$@"; do
  case "$STEP" in
    fetch|build|dedupe|align|train) "$STEP" ;;
    eval) evaluate ;;
    *) echo "unknown step: $STEP" >&2; exit 2 ;;
  esac
done
