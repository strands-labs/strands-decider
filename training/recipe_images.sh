#!/usr/bin/env bash
# Continuing a text checkpoint on images (configs/vision/), end to end: v19-images.yaml (v19),
# strands-decider-2B-hobson-v20-vl*.yaml (the v20 soup). Linux with an NVIDIA GPU;
# docs/models/strands-decider-2B-hobson-v20-balanced.md has the commands.
#
#   1. fetch   the training sources, TRAIN splits only, into data/raw/ (~45 GB, mostly
#              Multimodal-Mind2Web): VQAv2 questions, annotations and complementary pairs,
#              COCO 2014 instance annotations, TabFact at a pinned commit, and the
#              Multimodal-Mind2Web train shards at a pinned revision
#   2. build   the six row files and their images in data/image/build/ (data/image/README.md),
#              with the ~10,000 COCO train2014 images the rows use
#   3. dedupe  every training image against every evaluation image (data/checks/dedupe_images.py)
#   4. train   python -m strands_decider.vision_train $CONFIG (~46 min on one H100)
#   5. eval    $CKPT on NaturalBench, POPE and, with IJB_JSONL set, the rebuilt Image JevBench
#              preview items, with and without the image; image temperatures fitted on
#              NaturalBench groups 300-599 and applied; the 899-row text check
#
# Usage: training/recipe_images.sh STEP [STEP ...], each one of fetch build dedupe train eval.
# CONFIG (default configs/vision/v19-images.yaml) and CKPT (default its output_dir) choose
# the run; OUT (default reports/$(basename $CKPT)) holds the evaluation. INIT_FROM, when set,
# replaces the config's init_from (the text checkpoint to continue):
#   INIT_FROM=/path/to/text-ckpt CONFIG=configs/vision/v19-images.yaml \
#     training/recipe_images.sh train eval
# The builders need matplotlib, pyarrow, curl and the DejaVu and Liberation fonts; dedupe
# needs imagehash and pandas; eval needs pandas and pyarrow.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # every path below is relative to the repo root
export PY="${PY:-python}" PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
CONFIG="${CONFIG:-configs/vision/v19-images.yaml}"
CKPT="${CKPT:-$("$PY" -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")}"
OUT="${OUT:-reports/$(basename "$CKPT")}"
RAW=data/raw
DATA=data/image/build
W="${W:-16}"  # builder processes
M2W_REVISION=1b4c6a8cf9f77b7a5e0d641959935c80c4a05889
TABFACT_COMMIT=2ab782ba42b5808076ac91fec846473aa5315a79

download() { [ -f "$1" ] || { curl -fsSL -o "$1.part" "$2" && mv "$1.part" "$1"; }; }

fetch() {
  mkdir -p "$RAW/vqa" "$RAW/coco" "$RAW/m2w"
  local f
  for f in v2_Questions_Train_mscoco v2_Annotations_Train_mscoco v2_Complementary_Pairs_Train_mscoco; do
    download "$RAW/vqa/$f.zip" "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/$f.zip"
    unzip -qo "$RAW/vqa/$f.zip" -d "$RAW/vqa"
  done
  download "$RAW/coco/annotations_trainval2014.zip" \
    http://images.cocodataset.org/annotations/annotations_trainval2014.zip
  unzip -qo "$RAW/coco/annotations_trainval2014.zip" annotations/instances_train2014.json -d "$RAW/coco"
  [ -d "$RAW/tabfact" ] || git clone -q https://github.com/wenhuchen/Table-Fact-Checking "$RAW/tabfact"
  git -C "$RAW/tabfact" checkout -q "$TABFACT_COMMIT"
  "$PY" - "$RAW/m2w" "$M2W_REVISION" <<'EOF'
import sys
from huggingface_hub import HfApi, hf_hub_download
out, rev = sys.argv[1:]
files = [s.rfilename for s in HfApi().dataset_info("osunlp/Multimodal-Mind2Web", revision=rev).siblings
         if s.rfilename.startswith("data/train-")]
for f in files:
    hf_hub_download("osunlp/Multimodal-Mind2Web", f, repo_type="dataset", revision=rev, local_dir=out)
print(f"Multimodal-Mind2Web @ {rev}: {len(files)} train shards")
EOF
}

build() {
  mkdir -p "$DATA/coco"
  local b=data/image
  "$PY" "$b/vqa.py" --vqa-dir "$RAW/vqa" --out "$DATA"
  "$PY" "$b/count.py" --instances "$RAW/coco/annotations/instances_train2014.json" --out "$DATA" \
    --prefer-images "$DATA/vqa_images.txt"
  # The COCO train2014 images the rows use, and no others.
  cat "$DATA/vqa_images.txt" "$DATA/count_images.txt" | sort -u | sed 's#^coco/##' |
    xargs -P 32 -I{} sh -c '[ -f "$0/coco/$1" ] || curl -fsS -o "$0/coco/$1" "http://images.cocodataset.org/train2014/$1"' "$DATA" {}
  "$PY" "$b/charts.py" --out "$DATA" --n-charts 2600 --workers "$W"
  "$PY" "$b/documents.py" --out "$DATA" --n-docs 1500 --workers "$W"
  "$PY" "$b/tabfact.py" --repo "$RAW/tabfact" --out "$DATA" --n-tables 1300 --workers "$W"
  "$PY" "$b/m2w.py" --parquets "$RAW"/m2w/data/train-*.parquet --out "$DATA" --per-file 230 --workers "$W"
  (cd "$DATA" && sha256sum vqa.jsonl count.jsonl m2w.jsonl charts.jsonl docs.jsonl tabfact.jsonl)
}

dedupe() {
  "$PY" data/checks/dedupe_images.py --data-root "$DATA" ${IJB_JSONL:+--ijb-dir "$(dirname "$IJB_JSONL")"} \
    --files vqa.jsonl count.jsonl m2w.jsonl charts.jsonl docs.jsonl tabfact.jsonl
}

train() { "$PY" -m strands_decider.vision_train "$CONFIG" ${INIT_FROM:+"init_from=$INIT_FROM"}; }

evaluate() {
  local run=(--systems strands --device cuda --checkpoint "$CKPT" --long-side 0 --max-pixels 400000)
  mkdir -p "$OUT"
  "$PY" evaluation/vision/run.py --out "$OUT/eval" "${run[@]}" --nb-groups 300 --pope 600 \
    ${IJB_JSONL:+--ijb-jsonl "$IJB_JSONL"}
  "$PY" evaluation/vision/run.py --out "$OUT/cal" "${run[@]}" --nb-start 300 --nb-groups 300 --pope 0 --no-blind
  "$PY" evaluation/vision/temps.py fit --run "$OUT/cal" --out "$OUT/image_temps.json"
  "$PY" evaluation/vision/temps.py apply --run "$OUT/eval" --temps "$OUT/image_temps.json" --out "$OUT/eval-T"
  "$PY" evaluation/vision/text_check.py --checkpoint "$CKPT" --out "$OUT/text_check.json"
}

[ $# -gt 0 ] || { echo "usage: $0 STEP [STEP ...] (fetch build dedupe train eval)" >&2; exit 2; }
for STEP in "$@"; do
  case "$STEP" in
    fetch|build|dedupe|train) "$STEP" ;;
    eval) evaluate ;;
    *) echo "unknown step: $STEP" >&2; exit 2 ;;
  esac
done
