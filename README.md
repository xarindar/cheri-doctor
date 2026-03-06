# Metro Manual — Cheri's Service Manual Chat

RAG-powered chat over the 1990 Geo Metro factory service manual.
Ask natural-language questions, get answers cited directly from the manual with figures.

**Stack:** Python 3.12+ · PyMuPDF · Surya layout · Tesseract + EasyOCR · rank-bm25 · sentence-transformers · Claude Sonnet 4.6 · FastAPI

---

## Linux / New machine (quick serve)

From project root:

```bash
# 1. System packages (Debian/Ubuntu, one-time)
sudo apt update && sudo apt install -y python3.12-venv python3-pip

# 2. Create venv and install deps (CPU; for GPU see requirements.txt)
bash scripts/setup.sh
source .venv/bin/activate

# 3. Ensure .env has ANTHROPIC_API_KEY=...

# 4. Run chat server (UI at /, API at /api/chat)
make run-chat
```

Open **http://localhost:8000**. The same server serves the frontend and API.

---

## Quick Start (Windows / existing env)

### 1. Prerequisites

- Python 3.13
- Tesseract OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `.env` file in project root with your API key:
  ```
  ANTHROPIC_API_KEY=sk-ant-...
  ```

### 2. Environment Setup

```bash
# Create venv
python -m venv .venv

# Activate (bash / Git Bash)
source .venv/Scripts/activate

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA first (required before requirements.txt)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Install everything else
pip install -r requirements.txt
```

> The system Python (`C:\Python313`) already has all packages installed — skip the above if using it directly.

---

## Chat UI — Start / Stop / Restart

### Start

```bash
# From project root (bash)
python -m tools.chat_backend.serve
```

Open **http://localhost:8000** in your browser.

To run in the background and keep the terminal free:

```bash
# Background (bash)
nohup python -m tools.chat_backend.serve > server.log 2>&1 &
echo $! > server.pid
```

### Stop

```bash
# If you have the PID file
kill $(cat server.pid)

# Find and kill by port (bash)
netstat -ano | grep ":8000" | grep LISTENING
# Note the PID in the last column, then:
taskkill //F //PID <PID>
```

```powershell
# PowerShell — kill whatever is on port 8000
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
```

### Restart

```bash
# Kill then start fresh
kill $(cat server.pid) 2>/dev/null; python -m tools.chat_backend.serve
```

---

## Pipeline — Running the Ingestion

All commands run from `D:\Metro Project\`.

> **Note:** Stage arguments must be quoted in PowerShell: `--stages "C,D"` not `--stages C,D`

### Full pipeline run (all pages, vision on)

```bash
python -m src.pipeline
```

### Common partial runs

```bash
# Specific page range
python -m src.pipeline --pages 5-34

# Specific stages only (e.g. layout + extraction)
python -m src.pipeline --pages 40-45 --stages "C,D"

# Rebuild chunks + index only (fast, no OCR)
python -m src.pipeline --stages "F,G"

# Skip vision (no Claude API calls, much faster)
python -m src.pipeline --skip-vision

# Skip vision, specific page range
python -m src.pipeline --pages 30-50 --skip-vision
```

### Pipeline stages reference

| Stage | Flag | What it does |
|-------|------|--------------|
| A | `A` | Ingest — rasterize PDF to PNG at 350 DPI |
| B | `B` | Preprocess — crop, deskew, CLAHE, border removal |
| C | `C` | Layout — Surya region detection + CV fallback |
| D | `D` | Extract — OCR text, tables (vision-first), figures |
| E | `E` | Structure — heading hierarchy, section paths |
| F | `F` | Chunker — type-aware chunk generation → `build/chunks.jsonl` |
| G | `G` | Index — BM25 + embeddings → `tools/rag_index/` |

After changing the system prompt or chat logic, only stages F and G need to re-run.
After adding/fixing pages, run the full pipeline or the relevant stages for those pages.

### Rebuild index only (after prompt/config changes)

```bash
python -m src.pipeline --stages "F,G"
```

---

## Build Outputs

```
build/
  manifest.json       — run metadata (doc_id, DPI, page count, run_id)
  document.json       — CANONICAL TRUTH: all pages → blocks → relations
  chunks.jsonl        — retrieval units for RAG
  figures.jsonl       — figure metadata + asset links
  pages/              — rasterized PNGs (350 DPI)
  pages_pre/          — preprocessed PNGs
  assets/             — figure/diagram WebP crops
  tables/             — CSV + WebP table crops
tools/rag_index/
  bm25_index.pkl
  embeddings.npy
  chunk_ids.json
  chunk_lookup.json
```

---

## Project Layout

```
D:\Metro Project\
  src/                    — pipeline stages (A–H)
  configs/
    default.yaml          — all pipeline + chat settings
    chat_system_prompt.txt — vehicle profile, citation rules
  tools/
    chat_backend/
      serve.py            — FastAPI server (port 8000)
    rag_index/            — built indices
  build/                  — all pipeline outputs
  Reference Manuals/      — source PDFs + hand-curated TOC/CSV
  legacy/                 — original pipeline modules (imported)
  frontend/
    index.html            — chat UI (served by FastAPI)
  .env                    — API key (never commit)
  requirements.txt        — Python dependencies
```

---

## Current Index State

- **Pages processed:** 5–34
- **Chunks:** 162 (includes TOC + maintenance schedule)
- **Figures:** 25
- **Tables:** 7 (vision rows → CSV)
