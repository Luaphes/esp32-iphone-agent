#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT/server"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export PATH="$PROJECT_ROOT/.tools/bin:$PATH"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export HIDAGENT_DATA="$PROJECT_ROOT/.local/device-baseline"
export HOST="${HOST:-127.0.0.1}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
mkdir -p "$HIDAGENT_DATA"
[[ -x .venv/bin/python ]] || { echo "Python environment is missing" >&2; exit 1; }
command -v uxplay >/dev/null || { echo "UxPlay is missing" >&2; exit 1; }
exec .venv/bin/python -m hidagent
