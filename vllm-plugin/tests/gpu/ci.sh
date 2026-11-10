#!/bin/bash
# The plugin end to end with tiny random-weight checkpoints: no cluster, no dataset, no real weights.
#
# usage: tests/gpu/ci.sh OUT_DIR             one GPU of 24 GB or more (the release gate's GPU tier)
#        DEVICE=cpu tests/gpu/ci.sh OUT_DIR  a CPU build of vLLM, no GPU (the CI tier; tests/cpu/ci.sh)
# env:   GPU (default: CUDA_VISIBLE_DEVICES as the scheduler set it, else 0), PORT_BASE (8700), READY_S (1200)
#        STRANDS_DECIDER_GEMMA4_BASE          a directory with google/gemma-4-E2B-it's config.json, processor_config.json,
#                                             tokenizer.json and tokenizer_config.json (default: fetched from the Hub at
#                                             E2B_REVISION into OUT_DIR/base/e2b; set HF_TOKEN if the repo is gated)
#        STRANDS_DECIDER_GEMMA4_UNIFIED_BASE  the same four files of google/gemma-4-12B-it (default: fetched at
#                                             UNIFIED_REVISION)
#        STRANDS_DECIDER_SMOKE_CHECKPOINT     optional, GPU only: a real checkpoint in the multimodal layout, served
#                                             once for a smoke test of text and image requests
#        HETERO_HEADS=1                       GPU only: give the tiny models Gemma 4's two head sizes (FlashAttention 4,
#                                             Hopper); without it both are 64, which FlashAttention 2 and 3 run
#        FULL=1                               on CPU, also the text-only server and the cross-server comparison
#
# Needs vLLM 0.33 with the plugin installed, the core library (strands_decider.core), transformers, Pillow and curl.
# On a GPU everything runs in bf16, as the plugin serves; on a CPU in float32 (a CPU build of vLLM accepts it, and the
# hosted runners have no bf16 arithmetic), with 32-token KV blocks. Steps, each a PASS/FAIL line:
#   1. tiny checkpoints (tests/gpu/tiny.py): multimodal (Gemma4ForConditionalGeneration, with the base's "vision" mode
#      on), the same with inputs ["text"], unified (Gemma4UnifiedForConditionalGeneration), and a bare decoder;
#   2. requests (tests/gpu/make_requests.py) and the transformers reference (tests/gpu/reference.py) on each, in the
#      serving precision;
#   3. strands-decider-vllm serve on the multimodal checkpoint: text, images, refusals, the image-content cache test
#      and state reuse on exact counts of cached tokens (tests/gpu/server_client.py images), the text-only
#      checkpoint's refusal of images, every answer against the reference, and the conformance check
#      (tests/gpu/conformance.py) on goldens written from that reference; then the unified checkpoint against its
#      reference;
#   4. the server's request path in process on the multimodal checkpoint (tests/gpu/engine_checks.py): a prompt of
#      exactly --max-length tokens is answered, one token more is refused, vLLM refuses a prompt that fills the
#      context, an engine refusal raises VLLMValidationError (HTTP 400), and image requests through the server's
#      HTTP app have vLLM's image preprocessing off the event loop while /ping is timed;
#   5. vllm serve on the bare decoder: tests/gpu/client.py requests (refusals, exact cache counts, mixed batches, the
#      reference) and, on a second server without prefix caching, compare (the cross-server comparison);
#   6. with STRANDS_DECIDER_SMOKE_CHECKPOINT, the smoke test.
# Every server binds 127.0.0.1 with vLLM's internal sockets on loopback, runs in its own process group, and is stopped
# on exit; the listening sockets are audited.
set -u
OUT=$1; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
DEVICE=${DEVICE:-gpu}; P=${PORT_BASE:-8700}; READY_S=${READY_S:-1200}
E2B_REVISION=3e22461f65e89153144f8adb70e3b8c2cc9845a7
UNIFIED_REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
TOKENIZER_SHA256=cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f
export VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
# Every tolerance below is a measured difference with headroom, never a widened gate: the source of each is in its
# comment. "Cold" is an answer computed from scratch, "cached" one that reuses prefix-cache blocks, and |dp| the
# largest difference of one question's probabilities. The cache checks themselves are exact counts of cached tokens.
if [ "$DEVICE" = cpu ]; then
  # float32 on a CPU build of vLLM, against transformers in float32: answers agree to rounding (measured on this
  # tier, twice: top-1 1.0 at every layout, mean |dp| at most 0.0001, max 0.0009 (unified) and 0.0006 (decoder),
  # the residue of the readout's bf16 rounding of slightly different float32 hidden states). The gates: top-1 1.0,
  # mean 0.001, max 0.005 (about five times the largest measured).
  DTYPE=float32; BLOCK=32; REF=(--device cpu --dtype float32); MEM=()
  GATE=(--min-top1 1.0 --max-mean-dp 0.001 --max-dp 0.005)
  # A cached, batched or repeated answer is the cold one exactly: CPU attention computes the same bits whatever the
  # cache and the batch (measured: |dp| 0 for every repeat, cached, further-question and mixed-batch check, and
  # 7e-8 for logprobs 255 against allowed_token_ids, a renormalization in float32).
  TOLS=(--repeat-tol 1e-6 --same-server-tol 1e-6); SAME=(--same-tol 1e-6)
  export VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-2} CUDA_VISIBLE_DEVICES=
else
  # bf16 on the GPU, against transformers in bf16 (both as the plugin serves). A tiny random model's own bf16 and
  # float32 answers differ by a mean |dp| of 0.003 to 0.009 and on up to 2.5% of top-1 answers, while a wrong mask
  # (causal) moves them by a mean of 0.15 to 0.36. Measured on H100 over four runs of this tier: top-1 at least
  # 0.983, mean |dp| at most 0.0081, max 0.038. The gates: top-1 0.95 (twice the 2.5%) and mean 0.015.
  DTYPE=bfloat16; BLOCK=16; REF=(); MEM=(--gpu-memory-utilization 0.18)
  GATE=(--min-top1 0.95 --max-mean-dp 0.015)
  # A cached, repeated or batched answer runs other attention kernels than a cold one. Measured on H100 over four
  # runs: a cached repeat against the cold answer 0.0138 (on every repeat path: an exact repeat, the request after a
  # refusal, logprobs 255), a mixed batch against the request alone at most 0.0149, a further question 0.0003, an
  # image request's repeat 0.0064; two separately started servers at most 0.0273, with one top-1 flip in four runs.
  # The tolerances: 0.02 on one server, 0.01 for the image repeat, 0.035 across servers, each about 1.3 to 1.6 times
  # the largest measured; a cross-server flip counts as a near-tie only where the first server's top two are within
  # 0.07, twice the cross-server tolerance, the most two such answers can move apart.
  TOLS=(--repeat-tol 0.02 --same-server-tol 0.02 --cross-server-tol 0.035 --tie-gap 0.07); SAME=(--same-tol 0.01)
  export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
fi
FAILED=(); PIDS=()
T0=$(date +%s)
step() { echo "[$(( $(date +%s) - T0 ))s] $*"; }
result() { if [ "$1" = 0 ]; then echo "PASS  $2"; else echo "FAIL  $2"; FAILED+=("$2"); fi; }
stop() { kill -TERM -- -"$1" 2>/dev/null; for _ in $(seq 30); do kill -0 -- -"$1" 2>/dev/null || return 0; sleep 1; done; kill -KILL -- -"$1" 2>/dev/null; }
cleanup() { for p in "${PIDS[@]}"; do stop "$p"; done; PIDS=(); }
trap cleanup EXIT

fetch() {  # REPO REVISION DIR ENV_NAME
  mkdir -p "$3"
  local auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  for f in config.json processor_config.json tokenizer.json tokenizer_config.json; do
    [ -s "$3/$f" ] && continue
    code=$(curl -sSL --retry 3 ${auth[@]+"${auth[@]}"} -o "$3/$f" -w '%{http_code}' "https://huggingface.co/$1/resolve/$2/$f")
    [ "$code" = 200 ] || { echo "fetch $1/$f: HTTP $code (set HF_TOKEN for a gated repo, or $4)"; return 1; }
  done
}
BASE=${STRANDS_DECIDER_GEMMA4_BASE:-$OUT/base/e2b}
UBASE=${STRANDS_DECIDER_GEMMA4_UNIFIED_BASE:-$OUT/base/12b}
[ -n "${STRANDS_DECIDER_GEMMA4_BASE:-}" ] || fetch google/gemma-4-E2B-it $E2B_REVISION "$BASE" STRANDS_DECIDER_GEMMA4_BASE || exit 2
[ -n "${STRANDS_DECIDER_GEMMA4_UNIFIED_BASE:-}" ] || fetch google/gemma-4-12B-it $UNIFIED_REVISION "$UBASE" STRANDS_DECIDER_GEMMA4_UNIFIED_BASE || exit 2
echo "$TOKENIZER_SHA256  $BASE/tokenizer.json" | sha256sum -c --quiet - || { echo "tokenizer.json is not the pinned Gemma 4 tokenizer"; exit 2; }

step "1. tiny checkpoints ($DEVICE, $DTYPE)"
H=(); [ "${HETERO_HEADS:-0}" = 1 ] && H=(--hetero-heads)
C=$OUT/ckpt
python "$HERE/tiny.py" "$BASE" "$C/mm" --layout multimodal --vision-bidir "${H[@]}" > "$OUT/tiny.txt" 2>&1 &&
python "$HERE/tiny.py" "$BASE" "$C/mm-text" --layout multimodal --inputs text "${H[@]}" >> "$OUT/tiny.txt" 2>&1 &&
python "$HERE/tiny.py" "$UBASE" "$C/unified" --layout unified --base-model google/gemma-4-12B-it \
  --base-revision $UNIFIED_REVISION "${H[@]}" >> "$OUT/tiny.txt" 2>&1 &&
python "$HERE/tiny.py" "$BASE" "$C/decoder" --layout decoder "${H[@]}" >> "$OUT/tiny.txt" 2>&1
result $? "tiny checkpoints"; [ ${#FAILED[@]} -eq 0 ] || { tail -20 "$OUT/tiny.txt"; exit 1; }

step "2. requests and the transformers reference"
python "$HERE/make_requests.py" images "$OUT/requests.jsonl" --n 30 > /dev/null &&
python "$HERE/make_requests.py" rows "$BASE/tokenizer.json" "$OUT/rows-in.jsonl" --n 60 > /dev/null &&
python "$HERE/reference.py" "$C/mm" --requests "$OUT/requests.jsonl" --out "$OUT/ref-mm.jsonl" "${REF[@]}" > "$OUT/ref.txt" 2>&1 &&
python "$HERE/reference.py" "$C/unified" --requests "$OUT/requests.jsonl" --out "$OUT/ref-unified.jsonl" "${REF[@]}" >> "$OUT/ref.txt" 2>&1 &&
python "$HERE/reference.py" "$C/decoder" --rows "$OUT/rows-in.jsonl" --annotate --out "$OUT/rows.jsonl" "${REF[@]}" >> "$OUT/ref.txt" 2>&1
result $? "transformers reference"; [ ${#FAILED[@]} -eq 0 ] || { tail -20 "$OUT/ref.txt"; exit 1; }

ready() {  # NAME PORT PID
  for i in $(seq "$READY_S"); do
    curl -sf -m 2 "http://127.0.0.1:$2/health" > /dev/null && { step "$1 ready after ${i}s"; return 0; }
    kill -0 "$3" 2>/dev/null || { echo "$1: server exited"; tail -30 "$OUT/$1/server.log"; return 1; }
    sleep 1
  done
  echo "$1: not ready after $READY_S s"; return 1
}
serve() {  # NAME PORT MODEL [flags]: the plugin's server
  local name=$1 port=$2 model=$3; shift 3; mkdir -p "$OUT/$name"
  setsid python -m strands_decider_vllm serve "$model" --host 127.0.0.1 --port "$port" \
    --dtype $DTYPE --block-size $BLOCK ${MEM[@]+"${MEM[@]}"} "$@" > "$OUT/$name/server.log" 2>&1 &
  PIDS+=($!); ready "$name" "$port" $!
}
vllm_serve() {  # NAME PORT MODEL [flags]
  local name=$1 port=$2 model=$3; shift 3; mkdir -p "$OUT/$name"
  setsid vllm serve "$model" --host 127.0.0.1 --port "$port" --served-model-name m --dtype $DTYPE --seed 0 \
    --hf-overrides '{"architectures": ["StrandsDeciderForCausalLM"], "use_bidirectional_attention": null}' \
    --max-model-len 4096 --max-num-batched-tokens 4096 --no-enable-chunked-prefill --enable-prompt-tokens-details \
    --generation-config vllm --logprobs-mode processed_logprobs --max-logprobs 255 --block-size $BLOCK \
    ${MEM[@]+"${MEM[@]}"} "$@" > "$OUT/$name/server.log" 2>&1 &
  PIDS+=($!); ready "$name" "$port" $!
}
audit() {
  ss -ltnpH 2>/dev/null | grep "users:((" | awk '{print $4}' \
    | grep -v -E '^127\.0\.0\.1:|^\[::1\]:|^\[::ffff:127\.0\.0\.1\]:' > "$OUT/listeners_off_loopback.txt"
  [ ! -s "$OUT/listeners_off_loopback.txt" ]; result $? "every listening socket on loopback"
}
answers() {  # NAME PORT REFERENCE
  python -I "$HERE/server_client.py" answers --url "http://127.0.0.1:$2" --requests "$OUT/requests.jsonl" \
    --expect "$3" "${GATE[@]}" --out "$OUT/$1/answers.jsonl" > "$OUT/$1/answers.txt" 2>&1
  result $? "$1: answers against the reference ($(grep -o 'top-1 [0-9.]*.*max |dp| [0-9.]*' "$OUT/$1/answers.txt" | head -1))"
}

step "3. strands-decider-vllm serve: multimodal, text-only, unified"
TEXTONLY=()
if [ "$DEVICE" = gpu ] && serve mm-text $((P + 2)) "$C/mm-text"; then TEXTONLY=(--text-only "http://127.0.0.1:$((P + 2))"); fi
if serve mm $((P + 1)) "$C/mm"; then
  python -I "$HERE/server_client.py" images --url http://127.0.0.1:$((P + 1)) ${TEXTONLY[@]+"${TEXTONLY[@]}"} \
    --block-size $BLOCK "${SAME[@]}" --out "$OUT/mm/images.json" > "$OUT/mm/images.txt" 2>&1
  result $? "multimodal: text, images, refusals, the image-content cache test (exact cached tokens)"
  answers mm $((P + 1)) "$OUT/ref-mm.jsonl"
  # The conformance tier's check (section 10's fields and bounds) on goldens written from the tiny checkpoint's
  # reference: 90 questions, so the core's 1,000-question minimum is lowered for this wiring check only.
  python "$HERE/goldens.py" "$C/mm" "$OUT/requests.jsonl" "$OUT/ref-mm.jsonl" > "$OUT/mm/goldens.txt" 2>&1 &&
    python "$HERE/conformance.py" "$C/mm" --url http://127.0.0.1:$((P + 1)) --min-questions 90 \
      --out "$OUT/mm/conformance.json" > "$OUT/mm/conformance.txt" 2>&1
  result $? "conformance check on the tiny checkpoint's goldens: $(tail -1 "$OUT/mm/conformance.txt")"
else result 1 "multimodal server started"; fi
audit; cleanup
if [ "$DEVICE" = cpu ] && [ "${FULL:-0}" = 1 ]; then
  if serve mm-text $((P + 2)) "$C/mm-text"; then
    python -I "$HERE/server_client.py" textonly --url http://127.0.0.1:$((P + 2)) > "$OUT/mm-text/textonly.txt" 2>&1
    result $? "text-only checkpoint: images refused, text answered"
  else result 1 "text-only server started"; fi
  cleanup
fi
if serve unified $((P + 3)) "$C/unified"; then answers unified $((P + 3)) "$OUT/ref-unified.jsonl"
else result 1 "unified server started"; fi
cleanup

step "4. the server's request path on a real engine, in process"
mkdir -p "$OUT/engine"
python "$HERE/engine_checks.py" serve "$C/mm" --max-length 1024 --dtype $DTYPE --block-size $BLOCK --enforce-eager \
  ${MEM[@]+"${MEM[@]}"} > "$OUT/engine/server.log" 2>&1
rc=$?  # the multimodal checkpoint takes images, so the image check must have run, not been skipped
grep -q "image preprocessing ran off the event loop" "$OUT/engine/server.log" || rc=1
result $rc "engine checks: the length boundary, the engine's refusal, images off the loop ($(grep -c '^PASS' "$OUT/engine/server.log") passed)"
grep -E '^(PASS|FAIL)' "$OUT/engine/server.log"

step "5. vllm serve on the bare decoder: /v1/completions"
if vllm_serve decoder $((P + 4)) "$C/decoder" --enable-prefix-caching --prefix-cache-retention-interval None; then
  python -I "$HERE/client.py" requests --url http://127.0.0.1:$((P + 4)) --rows "$OUT/rows.jsonl" --n 60 \
    "${GATE[@]}" "${TOLS[@]}" --block-size $BLOCK --out "$OUT/decoder/client.json" > "$OUT/decoder/client.txt" 2>&1
  result $? "decoder: refusals, exact cache counts, mixed batches, the reference ($(grep -o 'top-1 [0-9.]*' "$OUT/decoder/client.txt" | head -1))"
else result 1 "decoder server started"; fi
[ "$DEVICE" = cpu ] && cleanup  # one server at a time on a CPU
if [ "$DEVICE" = gpu ] || [ "${FULL:-0}" = 1 ]; then
  if vllm_serve decoder-nocache $((P + 5)) "$C/decoder" --no-enable-prefix-caching; then
    python -I "$HERE/client.py" compare --url http://127.0.0.1:$((P + 5)) --rows "$OUT/rows.jsonl" \
      --expect "$OUT/decoder/client.json" "${TOLS[@]}" --out "$OUT/decoder-nocache/client.json" \
      > "$OUT/decoder-nocache/client.txt" 2>&1
    result $? "decoder without prefix caching: the same answers (cross-server)"
  else result 1 "decoder server without prefix caching started"; fi
fi
audit; cleanup

if [ -n "${STRANDS_DECIDER_SMOKE_CHECKPOINT:-}" ] && [ "$DEVICE" = gpu ]; then
  step "6. smoke test of a real checkpoint"
  mkdir -p "$OUT/smoke"
  setsid python -m strands_decider_vllm serve "$STRANDS_DECIDER_SMOKE_CHECKPOINT" --host 127.0.0.1 --port $((P + 6)) \
    --max-length 8192 --gpu-memory-utilization 0.85 > "$OUT/smoke/server.log" 2>&1 &
  PIDS+=($!)
  if ready smoke $((P + 6)) $!; then
    python -I "$HERE/server_client.py" images --url http://127.0.0.1:$((P + 6)) --out "$OUT/smoke/images.json" \
      > "$OUT/smoke/images.txt" 2>&1
    result $? "real checkpoint: text, images, refusals, the image-content cache test"
  else result 1 "real checkpoint served"; fi
  audit; cleanup
fi

# An engine error while serving fails the tier. Stopping a server (SIGTERM to its process group) can make the API
# server log EngineDeadError as the engine goes first, so only the log before the shutdown counts.
for f in "$OUT"/*/server.log; do
  sed '/\[shutdown\]\|Shutting down/,$d' "$f" | grep -E "Traceback|EngineDeadError|EngineCore encountered" > /dev/null &&
    result 1 "no engine error in $f"
done
step "done"
if [ ${#FAILED[@]} -eq 0 ]; then echo "$DEVICE tier: PASS"; else echo "$DEVICE tier: FAIL (${FAILED[*]})"; exit 1; fi
