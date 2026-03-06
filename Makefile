# Metro Project — standard dev targets
# Usage: make install | run-pipeline | run-chat | run-chat-bg

PYTHON ?= python3
VENV = .venv
BIN = $(VENV)/bin
PIP = $(BIN)/pip
PY = $(BIN)/python
UVICORN = $(BIN)/uvicorn

.PHONY: venv install run-pipeline run-chat run-chat-bg help

help:
	@echo "Metro Project targets:"
	@echo "  make venv       — create .venv"
	@echo "  make install    — venv + pip install -r requirements.txt"
	@echo "  make run-pipeline — run pipeline B..G (activate venv first)"
	@echo "  make run-chat   — start FastAPI chat on :8000"
	@echo "  make run-chat-bg — start chat server in background"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-cpu.txt
	@echo "Done. Run: source $(VENV)/bin/activate"

run-pipeline:
	$(PY) -m src.pipeline --stages "B,C,D,E,F,G"

run-chat:
	$(UVICORN) tools.chat_backend.serve:app --reload --host 0.0.0.0 --port 8000

run-chat-bg:
	$(UVICORN) tools.chat_backend.serve:app --host 0.0.0.0 --port 8000 &
