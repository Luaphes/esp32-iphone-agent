#!/usr/bin/env bash
# Local legacy workbench. The ESP32 connects to TCP 7300; the UI stays on localhost.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/server"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export PATH="$ROOT/.tools/bin:$PATH"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export HIDAGENT_DATA="${HIDAGENT_DATA:-$ROOT/.local/hidagent}"
export HOST="${HOST:-127.0.0.1}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
mkdir -p "$HIDAGENT_DATA"
[[ -x .venv/bin/python ]] || { echo "Python environment is missing" >&2; exit 1; }
command -v uxplay >/dev/null || { echo "UxPlay is missing" >&2; exit 1; }
exec .venv/bin/python -m hidagent
