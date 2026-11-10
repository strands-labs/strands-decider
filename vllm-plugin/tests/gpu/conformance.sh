#!/bin/bash
# Full conformance: every size against its own model goldens, one server per GPU, for nightly and release runs on an
# 8-GPU H100 node (the team cluster's release gate; README, Tests).
#
# usage: tests/gpu/conformance.sh OUT_DIR CKPT_E2B CKPT_E4B CKPT_12B CKPT_26B_A4B CKPT_31B
#   each CKPT_* a released checkpoint directory (the multimodal layout) holding golden_vectors.jsonl
# env: GPUS (default "0 1 2 3 4 5"), PORT_BASE (8900), READY_S (2400)
#
# The 31B runs on two GPUs (TP=2) so that contexts up to 32K fit; the others on one each. Each server is the plugin's
# own (strands-decider-vllm serve) at the checkpoint's max_length_validated, bound to 127.0.0.1. tests/gpu/
# conformance.py checks section 10's criteria per size: the rendering (exact), then the core's agreement against
# the reference, at least 98% top-1 and a mean |dp| of at most 0.02 (bf16) on at least 1,000 questions. Until the
# checkpoints publish golden_vectors.jsonl, conformance.py exits 3 (STUB) for every size and this script reports
# STUB, not PASS.
set -u
OUT=$1; shift; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a GPU <<< "${GPUS:-0 1 2 3 4 5}"
P=${PORT_BASE:-8900}; READY_S=${READY_S:-2400}
NAMES=(E2B E4B 12B 26B-A4B 31B)
[ $# -eq 5 ] || { echo "usage: $0 OUT_DIR CKPT_E2B CKPT_E4B CKPT_12B CKPT_26B_A4B CKPT_31B"; exit 2; }
[ ${#GPU[@]} -ge 6 ] || { echo "need 6 GPUs (the 31B on two)"; exit 2; }
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 15; for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done; }
trap cleanup EXIT
CKPTS=("$@")
for i in 0 1 2 3 4; do
  n=${NAMES[$i]}; mkdir -p "$OUT/$n"
  if [ "$n" = 31B ]; then g="${GPU[4]},${GPU[5]}"; extra=(--tensor-parallel-size 2 --gpu-memory-utilization 0.92)
  else g=${GPU[$i]}; extra=(--gpu-memory-utilization 0.90); fi
  env CUDA_VISIBLE_DEVICES="$g" setsid python -m strands_decider_vllm serve "${CKPTS[$i]}" --host 127.0.0.1 \
    --port $((P + i)) "${extra[@]}" > "$OUT/$n/server.log" 2>&1 &
  PIDS+=($!); echo $! > "$OUT/$n/pid"
done
RC=0; STUB=0
for i in 0 1 2 3 4; do
  n=${NAMES[$i]}; pid=$(cat "$OUT/$n/pid"); up=0
  for _ in $(seq "$READY_S"); do
    curl -sf -m 2 "http://127.0.0.1:$((P + i))/health" > /dev/null && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || break; sleep 1
  done
  if [ $up = 0 ]; then echo "$n: server did not start"; tail -20 "$OUT/$n/server.log"; RC=1; continue; fi
  python -I "$HERE/conformance.py" "${CKPTS[$i]}" --url "http://127.0.0.1:$((P + i))" --out "$OUT/$n/result.json" \
    > "$OUT/$n/conformance.txt" 2>&1
  rc=$?; echo "$n: $(tail -1 "$OUT/$n/conformance.txt")"
  case $rc in 0) ;; 3) STUB=1 ;; *) RC=1 ;; esac
done
if [ $RC -ne 0 ]; then echo "conformance: FAIL"; exit 1; fi
if [ $STUB -ne 0 ]; then echo "conformance: STUB (no readable golden_vectors.jsonl yet)"; exit 3; fi
echo "conformance: PASS"
