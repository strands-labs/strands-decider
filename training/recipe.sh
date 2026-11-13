#!/usr/bin/env bash
# v1's data, and the pointer head trained on it with training/train.py, end to end. Run it on Linux or WSL2 (see
# training/README.md, "The pointer head"): the Qwen3.5 torso's fused kernels need Linux.
#
#   1. build     the v5 corpus and the held-out calibration file. A fresh build today
#                gives the sha256 recorded in data/SHA256SUMS. Identity with the
#                original v5 file is not verified.
#   2. fetch     the ContractNLI, MuSiQue and HelpSteer2 releases into data/raw/ (~350 MB)
#   3. multistep the 12,909 multi-step training rows and the evaluation sets
#   4. generated the generated document questions and their held-out-domain eval sets --
#                v16's 2,148 and v18's 1,667 -- copied from data/synthetic/, the exact
#                files every model since v16 used, and the same rows with v20's checked
#                paraphrases attached (plus the paired consistency evals and v20's
#                instruction-flip rows). No generator is rerun.
#   5. adequacy  the answer-adequacy rows and eval sets: HelpSteer2 (4,866 rows, built from
#                the download) and the generated items (1,300 rows, data/synthetic/)
#   6. catchall  v20's catch-all rows, from the short-task corpus
#   7. distill   v20's teacher file, from the committed Qwen3.5-4B and replay distributions
#                (`teacher` relabels the multi-step rows with the frozen Qwen3.5-4B instead, ~1 h)
#   8. base      Qwen/Qwen3.5-2B-Base at training/configs/pointer-2b.yaml's revision, into $BASE
#   9. convert   v20's training files and teacher into the trainer's rows (training/convert.py), into $ROWS
#  10. train     training/configs/pointer-2b.yaml on those rows, into $RUN; the checkpoint is $CKPT
#  11. calibrate, then eval: held-out short tasks, the multi-step sets (HotpotQA among
#      them as the held-out transfer test), the generated held-out-domain sets and the
#      two answer-adequacy sets
#
# `train` trains v20's data and teacher with v1's settings, without the frozen-KL anchor v20 named, which this
# trainer does not have (training/README.md says what else differs from v1's trainer): it reproduces no released
# model. SEED picks the run's seed (default 0); a model is three seeds souped (`python -m training.soup`).
# Usage: training/recipe.sh STEP [STEP ...], each one of
#   build fetch multistep generated adequacy catchall teacher distill base convert train calibrate eval all
#
# Synthetic data is committed; public data is not. data/synthetic/ holds every generated
# row and every model-produced label the recipe trains on, as the exact files used: the
# generated questions and adequacy items (built from the data/generators/gen_*/ exports by
# training.data.generated), v20's instruction flips, the frozen Qwen3.5-4B's distributions
# on the short-task corpus and the multi-step rows, and v14's replay distributions.
# Public sources are downloaded and converted by `build`, `fetch`, `multistep` and
# `adequacy`, and v20's catch-all rows are derived from them by `catchall`.
#
# NGPU=N trains on N GPUs of this machine under torchrun (1, 2, 4 or 8: the config's 8 microbatches a step
# divide over them), and labels with `teacher` one shard per GPU, then a merge.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # every path below is relative to the repo root
export PY="${PY:-python}"  # the active environment
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
NGPU="${NGPU:-1}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-training/configs/pointer-2b.yaml}"
BASE="${BASE:-checkpoints/base-qwen3.5-2b}"
ROWS="${ROWS:-data/rows}"
RUN="${RUN:-checkpoints/pointer-2b-seed-$SEED}"
CKPT="${CKPT:-$RUN/last/checkpoint}"
# v20's training files, in the order the teacher file's row indices count them
EXAMPLES=(data/train_v5.jsonl data/multistep_v14.jsonl data/generated_v16p.jsonl data/generated_v18p.jsonl
  data/adequacy_hs2.jsonl data/adequacy_gen.jsonl data/catchall_v20.jsonl data/flips_v20.jsonl)

# label MODULE ARGS...: one process, or with NGPU > 1 one shard per GPU and then --merge.
label() {
  [ "$NGPU" -gt 1 ] || { "$PY" -m "$@"; return; }
  local i bad=0 pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i "$PY" -m "$@" --num-shards "$NGPU" --shard-index "$i" & pids+=("$!")
  done
  for i in "${pids[@]}"; do wait "$i" || bad=1; done
  [ "$bad" = 0 ] || { echo "$1: a shard failed" >&2; return 1; }
  "$PY" -m "$@" --merge --num-shards "$NGPU"
}

# verify FILE...: stop unless data/SHA256SUMS has an entry for each FILE and FILE has that
# sha256. Teacher and replay targets attach to corpus rows by position, so they are correct
# only for these exact bytes. A changed upstream download also stops here. To check every
# recorded file by hand: sha256sum -c data/SHA256SUMS
verify() {
  local f line sums=""
  for f; do
    line="$(awk -v f="$f" '$2 == f' data/SHA256SUMS)"
    [ -n "$line" ] || { echo "$f: no entry in data/SHA256SUMS" >&2; return 1; }
    sums+="$line"$'\n'
  done
  printf '%s' "$sums" | sha256sum -c --quiet --strict - ||
    { echo "stop: the files above do not match data/SHA256SUMS" >&2; return 1; }
}

build() {
  # Seven whole tasks are held out: four with label sets and rubrics never seen in
  # training, and three RuleTaker depths (in-family transfer, reported separately).
  "$PY" -u -m training.data build --out data/train_v5.jsonl --max-options 24 \
    -r ag_news -r banking77 -r clinc150 -r dbpedia -r lang_id -r yahoo_topics \
    -r spam -r toxicity -r mnli_entail -r boolq \
    -r paws -r vitaminc -r wnli -r pubmed_qa \
    -r yelp_stars -r sst5_sentiment -r app_reviews -r formality \
    -r ruletaker_d0 -r ruletaker_d1 -r ruletaker_d2 \
    -r emotion -r massive_intent -r sarcasm -r hate_severity \
    -r ruletaker_d3 -r ruletaker_d5 -r ruletaker_natlang \
    --holdout emotion --holdout massive_intent --holdout sarcasm --holdout hate_severity \
    --holdout ruletaker_d3 --holdout ruletaker_d5 --holdout ruletaker_natlang
  # The calibration and evaluation file: the held-out tasks but RuleTaker, whose depths share a
  # generator with the trained ones. Filtered as bytes so each line keeps its own ending.
  "$PY" - <<'PYTHON'
import json
src = open("data/train_v5.holdout.jsonl", "rb").read().splitlines(keepends=True)
keep = [ln for ln in src if not json.loads(ln)["task"].startswith("ruletaker")]
open("data/holdout_v5_norule.jsonl", "wb").write(b"".join(keep))
print(f"data/holdout_v5_norule.jsonl: {len(keep):,} of {len(src):,} held-out rows")
PYTHON
}

# download FILE URL: get URL into FILE, unless FILE exists. curl fails on an HTTP error and
# writes to FILE.part. Only a complete download becomes FILE, so a rerun never keeps a failed
# or partial file. fetch checks the downloads against data/SHA256SUMS before it extracts them.
download() { [ -f "$1" ] || { curl -fsSL -o "$1.part" "$2" && mv "$1.part" "$1"; }; }

fetch() {
  mkdir -p data/raw/helpsteer2 && cd data/raw
  download contract-nli.zip \
    https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip
  # MuSiQue's authors distribute through Google Drive (see their download_data.sh).
  download musique_data_v1.0.zip \
    "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"
  for f in train validation; do
    download "helpsteer2/$f.jsonl.gz" \
      "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/$f.jsonl.gz"
  done
  cd - >/dev/null
  verify data/raw/contract-nli.zip data/raw/musique_data_v1.0.zip \
    data/raw/helpsteer2/train.jsonl.gz data/raw/helpsteer2/validation.jsonl.gz
  cd data/raw
  "$PY" - <<'EOF'
import zipfile
z = zipfile.ZipFile("contract-nli.zip")
z.extractall(".", [m for m in z.namelist() if m.endswith((".json", "LICENSE", "TERMS", "README.md"))])
z = zipfile.ZipFile("musique_data_v1.0.zip")
for m in ("data/musique_full_v1.0_train.jsonl", "data/musique_full_v1.0_dev.jsonl"):
    z.extract(m, "musique")
print("extracted ContractNLI and MuSiQue (full)")
EOF
  cd - >/dev/null
}

multistep() { "$PY" -m training.data.multistep; }

# The committed files were built from the exports by
#   python -m training.data.generated [--src data/generators/gen_v16]              (v16)
#   python -m training.data.generated --src data/generators/gen_pilot_qwen data/generators/gen_mixed_pilot \
#     data/generators/gen_weak --out data/generated_v18.jsonl --eval-out data/generated_v18_eval.jsonl
#   python -m training.data.generated --src data/generators/gen_adequacy --out data/adequacy_gen.jsonl \
#     --eval-out data/adequacy_gen_eval.jsonl
#   python -m training.data.generated --src data/generators/gen_flips --out data/flips_v20.jsonl \
#     --eval-out data/flips_v20_eval.jsonl
# A rebuild today need not match them byte for byte, which is why they are committed.
generated() {
  local v
  for v in generated_v16 generated_v16_eval generated_v18 generated_v18_eval flips_v20 flips_v20_eval; do
    cp "data/synthetic/$v.jsonl" "data/$v.jsonl"
  done
  for v in v16 v18; do
    "$PY" -m training.data.generated --attach "data/generated_$v.jsonl" \
      --attach-eval "data/generated_${v}_eval.jsonl" --paraphrases data/generators/gen_paraphrases/paraphrases.jsonl \
      --out "data/generated_${v}p.jsonl" --pairs-out "data/para_pairs_${v}_eval.jsonl"
  done
}

adequacy() {
  "$PY" -m training.data.adequacy --raw data/raw/helpsteer2 \
    --out data/adequacy_hs2.jsonl --eval-out data/adequacy_hs2_eval.jsonl
  cp data/synthetic/adequacy_gen.jsonl data/synthetic/adequacy_gen_eval.jsonl data/
}

catchall() { "$PY" -m training.data.catchall; }  # v20: from data/train_v5.jsonl and the held-out set

teacher() {
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  label training.data.teacher --src data/multistep_v14.jsonl \
    --out data/teacher_multistep_v14.jsonl --shift-by data/train_v5.jsonl
}

# v20's teacher file: the committed 4B distributions on the short-task corpus (labelled
# for v12 by `python -m training.data.teacher --src data/train_v5.jsonl --out
# data/teacher_v5_qwen35-4b.jsonl`; the committed copy omits the score rows, which
# `distill` never kept), kept where they agree with gold, then v14's replay
# distributions on the multi-step rows (`python -m training.data.replay
# checkpoints/hobson-2b-v14 ...`).
distill() {
  verify data/train_v5.jsonl data/synthetic/teacher_v5_qwen35-4b.jsonl \
    data/synthetic/replay_v14_multistep.jsonl
  cp data/synthetic/teacher_v5_qwen35-4b.jsonl data/synthetic/replay_v14_multistep.jsonl data/
  "$PY" -m training.data.distill --corpus data/train_v5.jsonl --teacher data/teacher_v5_qwen35-4b.jsonl \
    --append data/replay_v14_multistep.jsonl --out data/teacher_v20.jsonl
}

base() {
  local model revision
  model="$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["base_model"])' "$CONFIG")"
  revision="$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["revision"])' "$CONFIG")"
  hf download "$model" --revision "$revision" --local-dir "$BASE"
}

convert() {
  verify data/train_v5.jsonl data/multistep_v14.jsonl
  "$PY" -m training.convert --examples "${EXAMPLES[@]}" --teacher 1.0 data/teacher_v20.jsonl \
    --base "$BASE" --prompt numbered --max-length 4096 --out "$ROWS"
}

train() {
  local args=(--config "$CONFIG" --base "$BASE" --train "$ROWS/train.jsonl" --dev "$ROWS/dev.jsonl"
    --output "$RUN" --seed "$SEED" --resume)
  if [ "$NGPU" -gt 1 ]; then
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" -m training.train "${args[@]}"
  else "$PY" -u -m training.train "${args[@]}"; fi
}

calibrate() { verify data/holdout_v5_norule.jsonl; "$PY" -u -m training.evaluate calibrate "$CKPT" --data data/holdout_v5_norule.jsonl; }

evaluate() {
  verify data/holdout_v5_norule.jsonl data/multistep_v14_eval.jsonl
  mkdir -p reports
  "$PY" -u -m training.evaluate eval "$CKPT" --data data/holdout_v5_norule.jsonl --limit 6000 --out reports/recipe_heldout.json
  "$PY" -m evaluation.multistep_eval "$CKPT"
  "$PY" -m evaluation.multistep_eval "$CKPT" --data data/generated_v16_eval.jsonl
  "$PY" -m evaluation.multistep_eval "$CKPT" --data data/generated_v18_eval.jsonl
  "$PY" -m evaluation.multistep_eval "$CKPT" --data data/adequacy_hs2_eval.jsonl
  "$PY" -m evaluation.multistep_eval "$CKPT" --data data/adequacy_gen_eval.jsonl
  # v20's evaluations, where their files have been built (`catchall`, `generated`)
  if [ -f data/catchall_v20_eval.jsonl ]; then
    "$PY" -m evaluation.multistep_eval "$CKPT" --data data/catchall_v20_eval.jsonl
  fi
  if [ -f data/flips_v20_eval.jsonl ]; then
    "$PY" -m evaluation.pair_eval "$CKPT" data/para_pairs_v16_eval.jsonl data/para_pairs_v18_eval.jsonl \
      data/flips_v20_eval.jsonl
  fi
}

[ $# -gt 0 ] || set -- all
for STEP in "$@"; do
  case "$STEP" in
    build|fetch|multistep|generated|adequacy|catchall|teacher|distill|base|convert|train|calibrate) "$STEP" ;;
    eval) evaluate ;;
    all) build; fetch; multistep; generated; adequacy; catchall; distill; base; convert; train; calibrate; evaluate ;;
    *) echo "unknown step: $STEP" >&2; exit 2 ;;
  esac
done
