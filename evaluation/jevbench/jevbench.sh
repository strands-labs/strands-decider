#!/usr/bin/env bash
# Run the JevBench v1 public set (231 tasks) against `strands-decider serve <ckpt>` on one GPU.
#
#   PY=~/venvs/hobson/bin/python evaluation/jevbench/jevbench.sh <checkpoint> <out_dir> [baseline_run]
#
# Env (all optional except PY):
#   PY               python of the strands-decider venv (required); JevBench is installed into it
#   GPU              CUDA_VISIBLE_DEVICES for the server            (default 7)
#   PORT             server port                                    (default 8099)
#   JEVBENCH_DIR     where JevBench is cloned                       (default $HOME/jevbench)
#   JEVBENCH_COMMIT  pinned JevBench commit                         (default 1bcc55e...)
#   EXPECT_MAX_LENGTH  window /health must report     (default: the checkpoint's own)
#   MODEL_LABEL      --model / --run-label passed to JevBench       (default basename of ckpt)
#   HEALTH_TIMEOUT_S seconds to wait for /health                    (default 1200)
#   SERVE_ARGS       extra `serve` flags, e.g. --vision to answer JevBench through the
#                    vision torso (text requests take the text path; /health must then
#                    report "vision": true)                         (default none)
#   baseline_run     a run in research/data/jevbench_results.csv (e.g. v17); if given,
#                    evaluation/jevbench/paired.py writes <out_dir>/paired.{txt,json}
#
# Writes under <out_dir>: all.jsonl (task file), health.json, server.log, results.jsonl,
# raw/, ledger.jsonl, manifest.json, summary.json, run_meta.json, jevbench.log, and
# paired.txt/json. Refuses to reuse a non-empty <out_dir> (JevBench opens its outputs
# with O_EXCL). Fails if the port is already served (the stale-server trap in
# evaluation/jevbench.md#reproducing-and-two-caveats) or if /health names a different
# checkpoint than the one requested.
set -euo pipefail

CKPT_ARG=${1:?usage: jevbench.sh <checkpoint> <out_dir> [baseline_run]}
OUT_ARG=${2:?usage: jevbench.sh <checkpoint> <out_dir> [baseline_run]}
BASELINE_RUN=${3:-}
: "${PY:?set PY to the strands-decider venv python}"
GPU=${GPU:-7}
PORT=${PORT:-8099}
JEVBENCH_DIR=${JEVBENCH_DIR:-$HOME/jevbench}
JEVBENCH_COMMIT=${JEVBENCH_COMMIT:-1bcc55eb6c8cffde2306b3db03ede39b61c6152a}
HEALTH_TIMEOUT_S=${HEALTH_TIMEOUT_S:-1200}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HOBSON=${HOBSON:-$(dirname "$PY")/strands-decider}
URL="http://127.0.0.1:$PORT"

log() { echo "[jevbench.sh $(date -u +%H:%M:%SZ)] $*"; }
die() { log "FAIL: $*"; exit 1; }

[ -x "$PY" ] || die "PY=$PY is not executable"
[ -x "$HOBSON" ] || die "strands-decider CLI not found at $HOBSON (set HOBSON)"
[ -d "$CKPT_ARG" ] || die "checkpoint dir $CKPT_ARG not found"
# strands_decider_config.json, or hobson_config.json for checkpoints saved before the rename.
CFG_NAME=strands_decider_config.json
[ -f "$CKPT_ARG/$CFG_NAME" ] || CFG_NAME=hobson_config.json
[ -f "$CKPT_ARG/$CFG_NAME" ] || die "$CKPT_ARG has no strands_decider_config.json (or legacy hobson_config.json)"
CKPT=$(cd "$CKPT_ARG" && pwd -P)          # absolute; this exact string is served and checked
EXPECT_MAX_LENGTH=${EXPECT_MAX_LENGTH:-$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["max_length"])' "$CKPT/$CFG_NAME")}
MODEL_LABEL=${MODEL_LABEL:-$(basename "$CKPT")}

if [ -e "$OUT_ARG" ] && [ -n "$(ls -A "$OUT_ARG" 2>/dev/null)" ]; then
  die "$OUT_ARG exists and is not empty; use a fresh output dir"
fi
mkdir -p "$OUT_ARG"
OUT=$(cd "$OUT_ARG" && pwd -P)
exec > >(tee -a "$OUT/jevbench.log") 2>&1
T0=$(date +%s)

# ---- 1. JevBench at the pinned commit, installed into the venv -----------------------
if [ ! -d "$JEVBENCH_DIR/.git" ]; then
  log "cloning JevBench into $JEVBENCH_DIR"
  git clone -q https://github.com/fstandhartinger/jevbench "$JEVBENCH_DIR"
fi
if ! git -C "$JEVBENCH_DIR" cat-file -e "$JEVBENCH_COMMIT^{commit}" 2>/dev/null; then
  git -C "$JEVBENCH_DIR" fetch -q origin
fi
git -C "$JEVBENCH_DIR" checkout -q --detach "$JEVBENCH_COMMIT"
[ -z "$(git -C "$JEVBENCH_DIR" status --porcelain --untracked-files=no)" ] \
  || die "$JEVBENCH_DIR has local modifications; the benchmark must be unmodified"
JB_HEAD=$(git -C "$JEVBENCH_DIR" rev-parse HEAD)
# Editable, no deps: seconds. uv first (uv-made venvs have no pip), else the venv's pip.
uv pip install -q --python "$PY" --no-deps -e "$JEVBENCH_DIR" 2>/dev/null || "$PY" -m pip install -q --no-deps -e "$JEVBENCH_DIR"
JB_IMPORT=$("$PY" -c 'import jevbench,os;print(os.path.dirname(os.path.dirname(os.path.abspath(jevbench.__file__))))')
[ "$JB_IMPORT" = "$(cd "$JEVBENCH_DIR" && pwd -P)" ] || die "jevbench imports from $JB_IMPORT, not $JEVBENCH_DIR"
case "$OUT/" in "$(cd "$JEVBENCH_DIR" && pwd -P)"/*) die "out dir must be outside the JevBench checkout";; esac
log "JevBench $JB_HEAD"

# ---- 2. the public task file: original (72) + easy (48) + hard (111), README order ---
TASKS="$OUT/all.jsonl"
cat "$JEVBENCH_DIR"/datasets/public/{original,easy,hard}.jsonl > "$TASKS"
N_TASKS=$(grep -c . "$TASKS")
[ "$N_TASKS" = 231 ] || die "task file has $N_TASKS tasks, expected 231"
"$PY" - "$TASKS" "$REPO/research/data/jevbench_tasks.csv" <<'EOF' || die "task ids differ from research/data/jevbench_tasks.csv"
import csv, json, sys
ids = [json.loads(l)["id"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
ref = [r["task_id"] for r in csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8"))]
assert len(set(ids)) == 231 and set(ids) == set(ref), (len(ids), len(set(ids) ^ set(ref)))
EOF
TASKS_SHA=$(sha256sum "$TASKS" | cut -d' ' -f1)

# ---- 3. start the server on one GPU, after proving the port is free ------------------
if curl -s -m 3 -o /dev/null "$URL/health"; then
  die "something already answers on $URL; refusing to measure a stale server"
fi
log "serving $CKPT on GPU $GPU port $PORT"
SERVE_ARGS=${SERVE_ARGS:-}
# shellcheck disable=SC2086  # SERVE_ARGS is a list of flags
CUDA_VISIBLE_DEVICES=$GPU nohup "$HOBSON" serve "$CKPT" --host 127.0.0.1 --port "$PORT" $SERVE_ARGS \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping server pid $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

T_LOAD0=$(date +%s)
until curl -s -m 5 "$URL/health" > "$OUT/health.json" 2>/dev/null && [ -s "$OUT/health.json" ]; do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -30 "$OUT/server.log"; die "server exited before /health answered"; }
  [ $(( $(date +%s) - T_LOAD0 )) -lt "$HEALTH_TIMEOUT_S" ] || die "/health not up after ${HEALTH_TIMEOUT_S}s"
  sleep 5
done
LOAD_S=$(( $(date +%s) - T_LOAD0 ))
"$PY" - "$OUT/health.json" "$CKPT" "$EXPECT_MAX_LENGTH" "$SERVE_ARGS" <<'EOF' || die "/health does not match the requested checkpoint (see health.json)"
import json, os, sys
h = json.load(open(sys.argv[1]))
want, want_len = sys.argv[2], int(sys.argv[3])
got = h.get("checkpoint")
print("health:", json.dumps(h))
ok = h.get("status") == "ok" and got is not None \
    and (got == want or os.path.realpath(got) == os.path.realpath(want)) \
    and h.get("max_length") == want_len \
    and bool(h.get("vision")) == ("--vision" in sys.argv[4].split())
sys.exit(0 if ok else 1)
EOF
log "/health ok (checkpoint and max_length match) after ${LOAD_S}s"

# ---- 4. run all 231 tasks, then summarize ------------------------------------------
# LC_ALL=C: byte order, as strands_decider.hf_export.fingerprint() recomputes it (a UTF-8 locale sorts
# tokenizer_config.json before tokenizer.json).
CKPT_REV=$(cd "$CKPT" && find -L . -type f ! -name '*.log' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
T_RUN0=$(date +%s)
"$PY" -m jevbench.cli run --tasks "$TASKS" --adapter typesafe --endpoint "$URL" --key-env '' \
  --model "$MODEL_LABEL" --run-label "$MODEL_LABEL" --revision "ckpt-sha256:$CKPT_REV" \
  --cost-basis no_billable_account_public_endpoint --reserve-usd 0 \
  --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" --ledger "$OUT/ledger.jsonl" \
  --manifest "$OUT/manifest.json"
RUN_S=$(( $(date +%s) - T_RUN0 ))
"$PY" -m jevbench.cli summarize --tasks "$TASKS" --results "$OUT/results.jsonl" \
  --ledger "$OUT/ledger.jsonl" --public-export "$OUT/summary.json" > "$OUT/summarize.log"

# Re-check /health after the run: the server that answered must still be ours.
curl -s -m 5 "$URL/health" > "$OUT/health_after.json"
cmp -s "$OUT/health.json" "$OUT/health_after.json" || die "/health changed during the run"
kill -0 "$SERVER_PID" 2>/dev/null || die "server pid $SERVER_PID died during the run"
cleanup
trap - EXIT
if curl -s -m 3 -o /dev/null "$URL/health"; then die "port $PORT still answers after stopping the server"; fi

# ---- 5. metadata, sanity gates, optional paired comparison --------------------------
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo unknown)
HOBSON_REV=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || cat "$REPO/.hobson-rev" 2>/dev/null || echo unknown)
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  HOBSON_DIRTY=$(git -C "$REPO" status --porcelain -- src | wc -l | tr -d ' ')
else
  HOBSON_DIRTY=unknown
fi
# The package the server actually imported (the host may have another tree installed).
HOBSON_PKG=$("$PY" -c 'import strands_decider,os;print(os.path.dirname(os.path.abspath(strands_decider.__file__)))')
HOBSON_SRC_SHA=$(cd "$HOBSON_PKG" && find . -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
TOTAL_S=$(( $(date +%s) - T0 ))
"$PY" - "$OUT" <<EOF
import json, sys
out = sys.argv[1]
s = json.load(open(f"{out}/summary.json"))
recs = [json.loads(l) for l in open(f"{out}/results.jsonl") if l.strip()]
meta = {
  "checkpoint": "$CKPT", "checkpoint_files_sha256_16": "$CKPT_REV",
  "jevbench_commit": "$JB_HEAD", "tasks_sha256": "$TASKS_SHA", "n_tasks": $N_TASKS,
  "hobson_git_rev": "$HOBSON_REV", "hobson_src_dirty_files": "$HOBSON_DIRTY",
  "hobson_pkg": "$HOBSON_PKG", "hobson_pkg_py_sha256_16": "$HOBSON_SRC_SHA",
  "gpu_index": "$GPU", "gpu": "$GPU_NAME", "port": $PORT, "serve_args": "$SERVE_ARGS",
  "server_load_s": $LOAD_S, "jevbench_run_s": $RUN_S, "total_wall_s": $TOTAL_S,
  "n_attempted": s.get("n_attempted"), "n_failed": sum(1 for r in recs if not r["ok"]),
  "n_correct": s.get("n_correct"), "accuracy": s.get("accuracy"),
  "schema_validity_strict": s.get("schema_validity_strict"),
  "latency": s.get("latency"),
}
import csv
tier = {r["task_id"]: r["tier"] for r in csv.DictReader(open("$REPO/research/data/jevbench_tasks.csv", newline=""))}
meta["tiers"] = {t: [sum(bool(r.get("correct")) for r in recs if tier.get(r["task_id"]) == t),
                     sum(1 for r in recs if tier.get(r["task_id"]) == t)] for t in ("easy", "standard", "hard")}
json.dump(meta, open(f"{out}/run_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
EOF
if [ -n "$BASELINE_RUN" ]; then
  "$PY" "$HERE/paired.py" --a "$REPO/research/data/jevbench_results.csv" --a-run "$BASELINE_RUN" \
    --b "$OUT/results.jsonl" --list --json "$OUT/paired.json" | tee "$OUT/paired.txt"
fi
"$PY" - "$OUT" <<'EOF' || die "sanity gate failed (see run_meta.json)"
import json, sys
m = json.load(open(f"{sys.argv[1]}/run_meta.json"))
assert m["n_attempted"] == 231 and m["n_failed"] == 0, "not 231/231 attempted with 0 failed"
assert m["schema_validity_strict"] == 1.0, "strict schema validity below 1.0"
if m["tiers"]["easy"] != [48, 48]:
    print("WARNING: easy tier", m["tiers"]["easy"], "- every model since v5 scores 48/48; suspect the harness or the checkpoint")
EOF
log "done: n_correct=$("$PY" -c "import json;print(json.load(open('$OUT/summary.json'))['n_correct'])")/231, run ${RUN_S}s, total ${TOTAL_S}s -> $OUT"
