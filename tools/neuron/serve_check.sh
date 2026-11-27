#!/usr/bin/env bash
# Serve a checkpoint on Neuron (loopback only), ask a short request, an over-long prompt
# (expect 422) and an image request (expect 422, NO_VISION), then stop the server.
#   tools/neuron/serve_check.sh [MODEL] [OUT]
set -u
MODEL="${1:?usage: serve_check.sh MODEL [OUT]}"
OUT="${2:-results/neuron-serve}"; mkdir -p "$OUT"
t0=$(date +%s)
strands-decider serve "$MODEL" --device neuron --port 8000 > "$OUT/serve.log" 2>&1 &
pid=$!
until curl -sf http://127.0.0.1:8000/health > "$OUT/health.json" 2>/dev/null; do
  kill -0 $pid 2>/dev/null || { echo "server exited"; tail -30 "$OUT/serve.log"; exit 1; }
  sleep 5
done
echo "ready after $(( $(date +%s) - t0 )) s" | tee "$OUT/ready.txt"
cat "$OUT/health.json"; echo
curl -s -X POST http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days!", "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' | tee "$OUT/blog_response.json"; echo
for i in 1 2 3 4 5; do curl -s -o /dev/null -w "%{time_total}\n" -X POST http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days!", "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'; done | tee "$OUT/blog_times_s.txt"
python - <<'PY' > "$OUT/long.json"
import json; print(json.dumps({"state": "word " * 40000, "questions": {"q": {"type": "noul", "instructions": "Is this long?"}}}))
PY
curl -s -w "\nHTTP %{http_code}\n" -X POST http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d @"$OUT/long.json" | tail -c 400 | tee "$OUT/too_long_response.txt"; echo
IMG=$(python -c "import base64,io;from PIL import Image;b=io.BytesIO();Image.new('RGB',(64,64)).save(b,'PNG');print(base64.b64encode(b.getvalue()).decode())")
curl -s -w "\nHTTP %{http_code}\n" -X POST http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' \
  -d "{\"state\": \"A photo.\", \"images\": [\"$IMG\"], \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"Is it dark?\"}}}" | tee "$OUT/image_response.txt"; echo
kill $pid; wait $pid 2>/dev/null
echo SERVE_CHECK_DONE
