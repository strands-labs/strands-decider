#!/usr/bin/env bash
# Every local benchmark against one running server (start it with --device mlx):
#   .venv/bin/strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --device mlx --port 8000
#   bash bench/run.sh
# Each script can also run alone; see its --help.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY bench/bench_corpus.py "$@"   # real corpora: latency + accuracy + cache behaviour
$PY bench/bench_all.py           # synthetic sweep: length, qcount, options, concurrency
$PY bench/bench_types.py         # 1000 tests per question type
