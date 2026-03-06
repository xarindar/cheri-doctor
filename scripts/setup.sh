#!/usr/bin/env bash
# Metro Project — one-time setup on a fresh Linux box.
# Run from project root:  bash scripts/setup.sh

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

echo "=== Metro Project setup ==="

# System deps (Debian/Ubuntu)
if ! command -v python3 &>/dev/null; then
  echo "python3 not found. Install it first."
  exit 1
fi
if ! python3 -m venv --help &>/dev/null; then
  echo "python3-venv not found. Install with:"
  echo "  sudo apt update && sudo apt install -y python3.12-venv python3-pip"
  exit 1
fi

# Venv
if [[ ! -d .venv ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# Pip deps (CPU by default; use requirements.txt + PyTorch CUDA index for GPU)
echo "Installing Python dependencies (CPU)..."
pip install --upgrade pip
pip install -r requirements-cpu.txt

# Optional: tesseract for pipeline (not needed for chat only)
if ! command -v tesseract &>/dev/null; then
  echo "Note: tesseract not found. Install for pipeline: sudo apt install tesseract-ocr"
fi

echo ""
echo "Done. Activate and run:"
echo "  source .venv/bin/activate"
echo "  make run-chat"
echo "Then open http://localhost:8000"
