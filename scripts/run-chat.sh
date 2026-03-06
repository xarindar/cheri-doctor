#!/usr/bin/env bash
# Start the Metro Manual chat server (UI + API on port 8000).
# Run from project root:  bash scripts/run-chat.sh

set -e
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  echo "Run setup first: bash scripts/setup.sh"
  exit 1
fi
source .venv/bin/activate
exec python -m uvicorn tools.chat_backend.serve:app --reload --host 0.0.0.0 --port 8000
