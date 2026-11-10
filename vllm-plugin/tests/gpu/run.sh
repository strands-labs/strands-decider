#!/bin/bash
# GPU integration check for strands-decider-vllm: the crash scenarios of the 0.2.0 review, on one node.
#
# usage: tests/gpu/run.sh MODEL_DIR ROWS_JSONL OUT_DIR
#   MODEL_DIR   a Strands Decider checkpoint directory (E2B is enough)
#   ROWS_JSONL  rendered decision requests for that checkpoint (token_ids, prefix_end, label_token_ids; optionally
#               probabilities from a reference, which gates top-1 agreement), prompts of about 1K tokens
#   OUT_DIR     logs and JSON summaries, one directory per server
# env: GPUS (default "0 1 2 3 4 5 6 7"; at least 6), PORT_BASE (8100), MAX_MODEL_LEN (8192), READY_S (1500)
#      MM_MODEL       the same checkpoint in the multimodal layout, inputs ["text", "image"]: runs the image checks
#      MM_TEXT_MODEL  a multimodal-layout checkpoint with inputs ["text"] (the image checks' text-only server)
#
# Run it with the vLLM environment active (vllm and ninja on PATH) and the plugin installed, and with the core
# library (strands_decider.core) importable for the image checks. Every server binds 127.0.0.1, with vLLM's internal
# sockets on loopback, runs in its own process group and is stopped on exit.
#
# Servers, one GPU each unless noted:
#   main      the README flags: malformed requests get 400 and the server stays healthy; cache and mixed batches
#   nocache   --no-enable-prefix-caching: the same refusals, the same answers as main
#   hub       the checkpoint as a Hugging Face Hub repo id, from a local HF cache, offline: the same answers as main
#   preempt   a 600-block KV cache: concurrent requests succeed
#   resume    the same small cache, max_tokens check bypassed (tests/gpu/bypass_max_tokens): requests preempted
#             mid-generation and resumed, without stopping the engine
#   refusals  each unsafe flag in turn: the server must stop at startup with the plugin's message
# Then, with MM_MODEL set, on the first GPUs once those servers have stopped:
#   images    strands-decider-vllm serve on MM_MODEL (tests/gpu/server_client.py images): text, one, two and four
#             images, refusals with the server still healthy, and the image-content cache test (a different image of
#             the same size reuses nothing; a further question on the same image reuses its state)
#   textonly  strands-decider-vllm serve on MM_TEXT_MODEL: images refused with the specification's message
#   mmtext    vllm serve on MM_MODEL with the multimodal architecture: the text rows of main answered alike
set -u
MODEL=$(cd "$1" && pwd); ROWS=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a GPU <<< "${GPUS:-0 1 2 3 4 5 6 7}"
PORT_BASE=${PORT_BASE:-8100}; MAXLEN=${MAX_MODEL_LEN:-8192}; READY_S=${READY_S:-1500}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
[ ${#GPU[@]} -ge 6 ] || { echo "need at least 6 GPUs, got ${#GPU[@]}"; exit 2; }
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
FLAGS=(--served-model-name m --dtype bfloat16 --seed 0
  --hf-overrides '{"architectures": ["StrandsDeciderForCausalLM"], "use_bidirectional_attention": null}'
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAXLEN" --no-enable-chunked-prefill
  --enable-prefix-caching --enable-prompt-tokens-details --prefix-cache-retention-interval None
  --generation-config vllm --logprobs-mode processed_logprobs --max-logprobs 255 --gpu-memory-utilization 0.85)
SMALL=(--max-model-len 2048 --max-num-batched-tokens 2048 --num-gpu-blocks-override 600)
PIDS=()
FAILED=()

start() {  # NAME GPUS PORT [ENV=VALUE ...] -- MODEL flags...  (server in its own process group)
  local name=$1 gpus=$2 port=$3; shift 3
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p "$OUT/$name"
  env CUDA_VISIBLE_DEVICES="$gpus" "${envs[@]}" setsid vllm serve "$@" --host 127.0.0.1 --port "$port" \
    > "$OUT/$name/server.log" 2>&1 &
  echo $! > "$OUT/$name/pid"; PIDS+=($!)
}
stop() {  # PID: the whole process group (API server, engine core, workers)
  kill -TERM -- -"$1" 2>/dev/null
  for _ in $(seq 30); do kill -0 -- -"$1" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- -"$1" 2>/dev/null
}
cleanup() { for p in "${PIDS[@]}"; do stop "$p"; done; }
trap cleanup EXIT
ready() {  # NAME PORT: until /health answers; fails if the server exits
  local pid; pid=$(cat "$OUT/$1/pid")
  for i in $(seq "$READY_S"); do
    curl -sf -m 2 "http://127.0.0.1:$2/health" > /dev/null && { echo "$1: ready after ${i}s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "$1: server exited"; tail -30 "$OUT/$1/server.log"; return 1; }
    sleep 1
  done
  echo "$1: not ready after $READY_S s"; return 1
}
client() {  # NAME SCENARIO PORT [args]
  local name=$1 scenario=$2 port=$3; shift 3
  python -I "$HERE/client.py" "$scenario" --url "http://127.0.0.1:$port" --rows "$ROWS" \
    --out "$OUT/$name/client.json" "$@" > "$OUT/$name/client.txt" 2>&1
  local rc=$?; tail -1 "$OUT/$name/client.txt"; [ $rc -eq 0 ] || FAILED+=("$name")
}
audit() {  # every listening TCP socket of this user's processes must be on loopback
  ss -ltnpH 2>/dev/null | grep "users:((" | awk '{print $4}' \
    | grep -v -E '^127\.0\.0\.1:|^\[::1\]:|^\[::ffff:127\.0\.0\.1\]:' > "$OUT/listeners_off_loopback.txt"
  if [ -s "$OUT/listeners_off_loopback.txt" ]; then echo "LISTENER AUDIT FAILED"; cat "$OUT/listeners_off_loopback.txt"; FAILED+=(listeners)
  else echo "listener audit: every socket on loopback"; fi
}

# A Hub repo id served from a local HF cache: snapshot links to the checkpoint's files.
HUB=$OUT/hub; REPO=example/strands-decider-test; COMMIT=0123456789abcdef0123456789abcdef01234567
SNAP=$HUB/models--example--strands-decider-test/snapshots/$COMMIT
mkdir -p "$SNAP" "$HUB/models--example--strands-decider-test/refs"
echo -n $COMMIT > "$HUB/models--example--strands-decider-test/refs/main"
for f in "$MODEL"/*; do ln -sfn "$f" "$SNAP/$(basename "$f")"; done

P=$PORT_BASE
start main    "${GPU[0]}" $((P + 0)) VLLM_SERVER_DEV_MODE=0 -- "$MODEL" "${FLAGS[@]}"
start nocache "${GPU[1]}" $((P + 1)) VLLM_SERVER_DEV_MODE=0 -- "$MODEL" "${FLAGS[@]}" --no-enable-prefix-caching
start hub     "${GPU[2]}" $((P + 2)) HF_HUB_CACHE="$HUB" HF_HUB_OFFLINE=1 -- "$REPO" "${FLAGS[@]}"
start preempt "${GPU[3]}" $((P + 3)) VLLM_SERVER_DEV_MODE=0 -- "$MODEL" "${FLAGS[@]}" "${SMALL[@]}"
start resume  "${GPU[4]}" $((P + 4)) PYTHONPATH="$HERE/bypass_max_tokens${PYTHONPATH:+:$PYTHONPATH}" -- \
  "$MODEL" "${FLAGS[@]}" "${SMALL[@]}"

# Startup refusals, on the remaining GPUs, a few at a time: NAME, expected message, extra flags.
REFUSALS=(
  "longpf|--long-prefill-token-threshold 256|--long-prefill-token-threshold 256"
  "fullcg|cudagraph_mode FULL|--compilation-config {\"cudagraph_mode\":\"FULL\"}"
  "rawlogprobs|--logprobs-mode raw_logprobs|--logprobs-mode raw_logprobs"
  "fp8kv|--kv-cache-dtype fp8|--kv-cache-dtype fp8"
  "fp16|--dtype torch.float16|--dtype float16"
  "embeds|--enable-prompt-embeds|--enable-prompt-embeds"
  "sched|--max-num-scheduled-tokens 4096|--max-num-scheduled-tokens 4096"
  "quant|--quantization fp8|--quantization fp8"
  "chunked|--no-enable-chunked-prefill|--enable-chunked-prefill"
)
refuse_on() {  # GPU PORT, then refusal specs; each must exit at startup with its message
  local gpu=$1 port=$2; shift 2
  PIDS=(); trap cleanup EXIT  # runs in a subshell: stop only this wave's servers
  for spec in "$@"; do
    IFS='|' read -r name want extra <<< "$spec"
    port=$((port + 1))
    # shellcheck disable=SC2086
    start "refuse-$name" "$gpu" $port VLLM_SERVER_DEV_MODE=0 -- "$MODEL" "${FLAGS[@]}" $extra
    local pid; pid=$(cat "$OUT/refuse-$name/pid") log=$OUT/refuse-$name/server.log
    for _ in $(seq 900); do kill -0 "$pid" 2>/dev/null || break; curl -sf -m 2 "http://127.0.0.1:$port/health" > /dev/null && break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
      echo "FAIL  refuse-$name: the server started"; stop "$pid"; echo "refuse-$name" >> "$OUT/refusals_failed.txt"
    elif grep -q "strands-decider cannot serve with this configuration" "$log" && grep -qF -- "$want" "$log"; then
      echo "ok    refuse-$name: refused at startup ($want)"
    else
      echo "FAIL  refuse-$name: exited without the expected message"; tail -5 "$log"; echo "refuse-$name" >> "$OUT/refusals_failed.txt"
    fi
  done
}
rm -f "$OUT/refusals_failed.txt"
( refuse_on "${GPU[5]}" $((P + 20)) "${REFUSALS[@]:0:5}" ) > "$OUT/refusals_a.txt" 2>&1 &
RA=$!
if [ ${#GPU[@]} -ge 7 ]; then
  ( refuse_on "${GPU[6]}" $((P + 40)) "${REFUSALS[@]:5}" ) > "$OUT/refusals_b.txt" 2>&1 & RB=$!
else
  RB=""; REFUSALS_LATER=("${REFUSALS[@]:5}")
fi

ready main $((P + 0)) && client main requests $((P + 0)) --n 60
ready nocache $((P + 1)) && client nocache compare $((P + 1)) --expect "$OUT/main/client.json"
ready hub $((P + 2)) && client hub compare $((P + 2)) --expect "$OUT/main/client.json"
ready preempt $((P + 3)) && client preempt preempt $((P + 3))
ready resume $((P + 4)) && client resume resume $((P + 4))
for n in main nocache hub preempt resume; do [ -s "$OUT/$n/client.txt" ] || FAILED+=("$n"); done
audit
wait $RA; [ -n "$RB" ] && wait $RB
[ -z "$RB" ] && ( refuse_on "${GPU[5]}" $((P + 40)) "${REFUSALS_LATER[@]}" ) > "$OUT/refusals_b.txt" 2>&1
cat "$OUT/refusals_a.txt" "$OUT/refusals_b.txt"
[ -s "$OUT/refusals_failed.txt" ] && FAILED+=(refusals)
for n in main nocache hub preempt resume; do
  grep -E "Traceback|EngineDeadError|EngineCore encountered" "$OUT/$n/server.log" > /dev/null && { echo "$n: engine error in server.log"; FAILED+=("$n-log"); }
done

if [ -n "${MM_MODEL:-}" ]; then
  # Phase 2, on GPUs the first phase has released.
  cleanup; PIDS=()
  MM=$(cd "$MM_MODEL" && pwd)
  SERVE=(python -m strands_decider_vllm serve)
  startserve() {  # NAME GPU PORT MODEL [flags]: the plugin's own server
    local name=$1 gpu=$2 port=$3 model=$4; shift 4
    mkdir -p "$OUT/$name"
    env CUDA_VISIBLE_DEVICES="$gpu" setsid "${SERVE[@]}" "$model" --host 127.0.0.1 --port "$port" \
      --max-length "$MAXLEN" --gpu-memory-utilization 0.85 "$@" > "$OUT/$name/server.log" 2>&1 &
    echo $! > "$OUT/$name/pid"; PIDS+=($!)
  }
  startserve images "${GPU[0]}" $((P + 60)) "$MM"
  [ -n "${MM_TEXT_MODEL:-}" ] && startserve textonly "${GPU[1]}" $((P + 61)) "$(cd "$MM_TEXT_MODEL" && pwd)"
  start mmtext "${GPU[2]}" $((P + 62)) VLLM_SERVER_DEV_MODE=0 -- "$MM" "${FLAGS[@]}" \
    --hf-overrides '{"architectures": ["StrandsDeciderForConditionalGeneration"], "text_config": {"use_bidirectional_attention": null}}' \
    --limit-mm-per-prompt '{"image": 4, "video": 0, "audio": 0}'
  TEXTONLY=()
  if [ -n "${MM_TEXT_MODEL:-}" ] && ready textonly $((P + 61)); then TEXTONLY=(--text-only "http://127.0.0.1:$((P + 61))"); fi
  if ready images $((P + 60)); then
    python -I "$HERE/server_client.py" images --url "http://127.0.0.1:$((P + 60))" "${TEXTONLY[@]}" \
      --out "$OUT/images/client.json" > "$OUT/images/client.txt" 2>&1 || FAILED+=(images)
    tail -1 "$OUT/images/client.txt"
  else FAILED+=(images); fi
  ready mmtext $((P + 62)) && client mmtext compare $((P + 62)) --expect "$OUT/main/client.json"
  audit
  for n in images mmtext; do
    grep -E "Traceback|EngineDeadError|EngineCore encountered" "$OUT/$n/server.log" > /dev/null && { echo "$n: engine error in server.log"; FAILED+=("$n-log"); }
  done
fi
if [ ${#FAILED[@]} -eq 0 ]; then echo "GPU checks: PASS"; else echo "GPU checks: FAIL (${FAILED[*]})"; exit 1; fi
