#!/usr/bin/env bash
# Run the real strands-agents/harness-sdk Astro site locally, with this
# checkout substituted for the published strands-decider product. This renders
# the docs the way strandsagents.com does (Starlight theme, nav, search, code
# blocks), so you can sanity-check docs/ edits before pushing.
#
# Astro's dev server daemonizes itself, so this script also owns teardown:
#   scripts/preview_docs_site.sh start   # default — clones if needed, starts
#   scripts/preview_docs_site.sh stop    # astro dev stop
#   scripts/preview_docs_site.sh status  # astro dev status
#   scripts/preview_docs_site.sh restart
#
# Env:
#   HARNESS_SDK=~/src/harness-sdk   reuse an existing harness-sdk checkout
#   PORT=4322                       dev server port (default 4321)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache"
HARNESS_SDK="${HARNESS_SDK:-${CACHE_DIR}/harness-sdk}"
PORT="${PORT:-4321}"
SITE_DIR="${HARNESS_SDK}/site"

astro() {
  if [ ! -d "${SITE_DIR}" ]; then
    echo "[preview-site] no harness-sdk site checkout at ${SITE_DIR} — nothing to do" >&2
    return 0
  fi
  (cd "${SITE_DIR}" && npx --no-install astro "$@")
}

cmd_start() {
  if [ ! -d "${HARNESS_SDK}" ]; then
    mkdir -p "$(dirname "${HARNESS_SDK}")"
    echo "[preview-site] cloning strands-agents/harness-sdk into ${HARNESS_SDK}"
    git clone --depth=1 https://github.com/strands-agents/harness-sdk.git "${HARNESS_SDK}"
  else
    echo "[preview-site] reusing ${HARNESS_SDK}"
  fi

  if [ ! -d "${SITE_DIR}" ]; then
    echo "[preview-site] error: ${SITE_DIR} not found — is this a harness-sdk checkout?" >&2
    exit 1
  fi

  cd "${SITE_DIR}"
  if [ ! -d node_modules ]; then
    echo "[preview-site] installing site dependencies (first run)…"
    npm install
  fi

  export EXTERNAL_DOCS_STRANDS_DECIDER_SOURCE="${REPO_ROOT}"
  echo "[preview-site] starting Astro dev server on :${PORT}"
  echo "[preview-site] Decider docs: http://localhost:${PORT}/docs/labs/strands-decider/overview/"
  echo "[preview-site] stop with: scripts/preview_docs_site.sh stop"
  npm run dev -- --port "${PORT}" --host 127.0.0.1
}

cmd_stop() {
  echo "[preview-site] stopping Astro dev server"
  astro dev stop || true
}

cmd_status() {
  astro dev status || true
}

case "${1:-start}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  *)
    echo "usage: $0 [start|stop|restart|status]" >&2
    exit 2
    ;;
esac
