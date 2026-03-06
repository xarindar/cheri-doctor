# Metro Project — Complete Handoff Guide

> This document captures everything a new Claude instance needs to know to continue working on this project. Written March 2026.

---

## 1. What This Project Is

A **RAG-powered "chat with the manual"** system for a 1990 Geo Metro factory service manual (771 pages). The owner's car is named **Cheri**. The chat assistant is called **Cheri Doctor**.

The user (Abe) scanned the entire factory service manual PDF, and this project extracts, structures, chunks, indexes, and serves it via a chat interface so he can ask questions about his car and get accurate answers with citations and figure images.

### The Car
- **Name:** Cheri
- **VIN:** JG1MR3362LK769576
- **Year/Model:** 1990 Geo Metro, 3-door hatchback
- **Engine:** 1.0L 3-cylinder (G10) — NOT the 1.3L G13
- **Transmission:** Automatic transaxle
- **Steering:** Manual (no power steering)
- **A/C:** Yes (retrofitted from R-12 to R-134a in 2007)
- **Mileage:** ~80,000 miles
- **Maintenance schedule:** Schedule I (severe service)

---

## 2. Project Structure

```
D:\Metro Project\
├── configs/
│   ├── default.yaml              # Master config for all pipeline stages
│   └── chat_system_prompt.txt    # "Cheri Doctor" system prompt
├── src/                          # Pipeline source code
│   ├── ingest.py                 # Stage A: PDF → page PNGs
│   ├── preprocess.py             # Stage B: crop, deskew, normalize
│   ├── layout.py                 # Stage C: Surya layout detection
│   ├── pipeline.py               # Stage D: OCR + vision + table/figure extraction
│   ├── structure.py              # Stage E: section path assignment
│   ├── chunker.py                # Stage F: blocks → retrieval chunks
│   ├── index_build.py            # Stage G: BM25 + embedding index
│   ├── chat.py                   # Stage H: RAG chat orchestration
│   ├── ocr_text.py               # Tesseract + EasyOCR consensus
│   ├── table_extract.py          # img2table + OpenCV table extraction
│   ├── figure_extract.py         # Figure detection and captioning
│   ├── vision_classify.py        # Claude Vision region classification
│   ├── vision_extract.py         # Vision-based text extraction
│   ├── gemini_api.py             # Gemini API wrapper
│   ├── models.py                 # Dataclasses (ChunkRecord, ChatResponse, etc.)
│   └── utils.py                  # load_jsonl, save_json, load_config, etc.
├── tools/
│   ├── chat_backend/
│   │   └── serve.py              # FastAPI server (port 8000)
│   ├── audit_chunks.py           # Chunk quality audit script
│   ├── inspect_index.py          # Index inspection utility
│   └── ...                       # Various one-off tools
├── frontend/
│   └── index.html                # Chat UI (single-file, served by FastAPI)
├── legacy/                       # Old/deprecated scripts (kept for reference)
│   ├── vision_describe.py        # Has load/save_description_cache
│   └── automotive_dictionary.txt # SymSpell dictionary for OCR correction
├── build/                        # All pipeline outputs (gitignored)
│   ├── pages/                    # 771 page PNGs at 350 DPI (Stage A)
│   ├── pages_pre/                # Preprocessed pages (Stage B)
│   ├── assets/                   # Extracted figures as .webp
│   ├── tables/                   # Extracted table CSVs
│   ├── document.json             # Full structured document (Stages D+E)
│   ├── chunks.jsonl              # 4226 retrieval chunks (Stage F)
│   ├── figures.jsonl             # 1462 figure records (Stage F)
│   ├── vision_classify_cache.json # Cached vision API classify results
│   ├── full_page_vision_cache.json
│   └── manifest.json
├── tools/rag_index/              # Search index (Stage G)
│   ├── bm25_index.pkl            # BM25Okapi object
│   ├── embeddings.npy            # (4048, 384) float32 embeddings
│   ├── chunk_ids.json            # Ordered chunk ID list
│   └── chunk_lookup.json         # chunk_id → full chunk record
├── Reference Manuals/
│   ├── 1990 Geo Metro Manual.pdf # Source PDF (771 pages)
│   ├── TABLE OF CONTENTS.md      # Hand-converted TOC
│   └── maintenance-schedule.csv  # Hand-curated maintenance schedule
├── requirements.txt              # Python 3.13 + CUDA 12.8
└── .env                          # API keys (ANTHROPIC_API_KEY, GEMINI_API_KEY)
```

---

## 3. Pipeline Stages

| Stage | Script | What It Does | Output | Status |
|-------|--------|-------------|--------|--------|
| **A** | `src/ingest.py` | Rasterize PDF at 350 DPI | `build/pages/*.png` | DONE — 771 pages, skip always |
| **B** | `src/preprocess.py` | Crop, deskew, CLAHE normalize, denoise | `build/pages_pre/` | DONE |
| **C** | `src/layout.py` | Surya layout detection (text, table, figure boxes) | `build/layout_results.json` | DONE |
| **D** | `src/pipeline.py` | OCR text, extract tables/figures, vision classify | `build/document.json`, `build/assets/`, `build/tables/` | DONE |
| **E** | `src/structure.py` | Assign section_path from TOC headings | Updates `build/document.json` | DONE |
| **F** | `src/chunker.py` | Split blocks into retrieval chunks | `build/chunks.jsonl`, `build/figures.jsonl` | DONE — 4226 chunks |
| **G** | `src/index_build.py` | Build BM25 + sentence-transformer index | `tools/rag_index/` | DONE — 4048 indexed |
| **H** | `src/chat.py` | Full RAG orchestration (not a "stage" per se) | Live chat responses | Active |

### Running the Pipeline

```bash
# Full pipeline (skips Stage A by default):
python -m src.pipeline

# Specific stages:
python -m src.pipeline --stages "F,G" --reprocess

# Single page for debugging:
python -m src.pipeline --pages 224

# Just re-chunk and re-index (most common during iteration):
python -m src.pipeline --stages "F,G" --reprocess
```

### Running the Chat Server

```bash
# Start:
python tools/chat_backend/serve.py
# or:
uvicorn tools.chat_backend.serve:app --host 0.0.0.0 --port 8000

# Access UI: http://localhost:8000
# API endpoint: POST /api/chat  {query: "...", conversation: [...]}
# Figure endpoint: GET /api/figures/{figure_id}
```

---

## 4. Current State (March 2026)

### Chunk Statistics
- **Total chunks:** 4,226 (4,048 indexed — 178 near-empty figure chunks filtered from index)
- **Types:** figure (1432), procedure (991), paragraph (736), table (540), notice (313), caution (76), important (66), toc (46), note (26)
- **Token range:** 2–917, mean 70
- **56 multi-block procedure stitches** (cross-page procedures merged)
- **1,462 figure records** with extracted .webp images

### Models in Use
| Purpose | Model | Notes |
|---------|-------|-------|
| Chat LLM | `gemini-2.5-flash` | Set in `configs/default.yaml` `chat.model` — was Claude, changed to Gemini for cost |
| Vision classify/describe | `claude-sonnet-4-6` | For figure/region classification during pipeline Stage D |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, via sentence-transformers |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` | Neural reranking in chat.py |
| Query rewrite | Same as chat model | Rewrites follow-up queries into standalone search queries |

### API Keys Required
```
ANTHROPIC_API_KEY=sk-ant-...   # For Claude Vision (pipeline) and Claude chat (if selected)
GEMINI_API_KEY=...             # For Gemini chat model and fallback vision
```

### Dependencies
- Python 3.13
- PyTorch 2.10 + CUDA 12.8 (install separately before pip install)
- Tesseract OCR at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- All other deps in `requirements.txt`

---

## 5. Retrieval Architecture (chat.py)

The retrieval pipeline has evolved significantly. Here's the full flow:

```
User Query
    │
    ├─► LLM Query Rewrite (for follow-up messages only)
    │       Uses Gemini Flash to rewrite "what about the belt?"
    │       into "AC compressor drive belt inspection tension"
    │
    ├─► Query Expansion
    │       1. Abbreviation expansion (VIN → vehicle identification number)
    │       2. Component synonym aliases (belt → drive belt V-belt fan belt)
    │       3. Maintenance supplements (oil change → capacity torque viscosity)
    │
    ├─► System Detection (_detect_system)
    │       Scores query against SYSTEM_KEYWORDS dict (27 systems)
    │       Longest matching keywords win (prevents "oil" from always matching engine)
    │       Returns: "engine", "maintenance", "brakes", etc. or None
    │
    ├─► Figure Intent Detection
    │       Checks for "diagram", "figure", "show me", "where is" etc.
    │
    ├─► Multi-Pass Retrieval (index_build.py RetrievalIndex)
    │       Pass 1: System-filtered retrieval (top-20, within detected system)
    │       Pass 2: Broad (unfiltered) merge (catches cross-section content)
    │       Pass 3: Figure-targeted retrieval (only if figure intent detected)
    │       Result: ~40 candidate chunks
    │
    ├─► Neural Reranking (_neural_rerank)
    │       Cross-encoder (ms-marco-MiniLM-L-6-v2) scores each (query, chunk) pair
    │       BUT: uses RRF fusion of original-rank + neural-rank
    │       (prevents garbled OCR table text from overriding strong BM25 matches)
    │       Result: top-30 candidates
    │
    ├─► Type-Boost Reranking (_rerank)
    │       Adds score boosts by chunk type and query intent:
    │       - Procedural queries: procedure +0.25, warning +0.20
    │       - Figure queries: figure +0.40
    │       - Informational: table +0.10, warning/caution +0.10
    │       - Diagnostic intent: table with CONDITION: content gets +0.15 extra
    │       Suppresses irrelevant figures (below 50% of top non-figure score)
    │       Enforces type diversity (max 2-4 figures depending on intent)
    │       Result: top-10 evidence chunks
    │
    ├─► Context Expansion
    │       1. Same-page context: pulls in related chunks from already-retrieved pages
    │       2. Dependency resolution: follows figure_refs, "Figure X" text refs,
    │          "diagnostic chart" mentions to pull in referenced figures
    │       3. Procedure continuation: links multi-page procedures by heading match
    │
    ├─► Figure Collection
    │       Gathers .webp image data (base64) for figure chunks, max 4 images
    │
    ├─► LLM Generation
    │       Builds prompt with EVIDENCE blocks + figure images
    │       Calls chat model (Gemini or Claude) with system prompt
    │
    └─► Response Parsing
            Extracts citations [p{page} | {chunk_id}]
            Resolves figure IDs (chunk-style → asset-style)
            Strips citation markers from displayed answer
```

### Key Design Decisions

1. **RRF-over-RRF neural reranking**: The cross-encoder scores are blended with original BM25+embedding ranks via Reciprocal Rank Fusion, not used as a pure replacement. This is critical because garbled OCR table text (common in this manual) can confuse the cross-encoder even when BM25 keyword matching correctly identifies the chunk.

2. **System-filtered + broad merge**: Every query gets both system-filtered results AND unfiltered results merged together. This ensures cross-section content (e.g., oil capacity in the maintenance section when querying about "engine oil") is never completely lost.

3. **Diagnostic table splitting**: Tables with CONDITION/CAUSE/CORRECTION columns are split into one chunk per condition group, regardless of row count. This enables precise retrieval ("low oil pressure causes" returns just the low oil pressure rows, not the entire multi-condition table).

4. **Figure relevance gating**: When the user doesn't ask for a figure, figures must score at least 50% of the top non-figure chunk's score to be included. Prevents irrelevant ESD labels and VIN diagrams from appearing in every response.

---

## 6. Chunk Types and Their Sources

| Type | Count | How Created | Notes |
|------|-------|-------------|-------|
| `figure` | 1432 | `vision_description` from figure blocks | Only 1254 indexed (178 <10 tokens filtered) |
| `procedure` | 991 | Ordered list blocks, merged cross-page | 56 multi-block stitches |
| `paragraph` | 736 | Text blocks, merged small adjacent ones | Min 200 tokens target |
| `table` | 540 | Table blocks, diagnostic tables split by condition | 36 diagnostic tables identified |
| `notice` | 313 | NOTICE: blocks | |
| `caution` | 76 | CAUTION: blocks, kept whole | |
| `important` | 66 | IMPORTANT: blocks | |
| `toc_entry` | 39 | Table of contents entries | |
| `note` | 26 | NOTE: blocks | |

### Chunk ID Format
- `para_{section}_{page}_{seq}` — paragraphs
- `proc_{section}_{page}_{seq}` — procedures
- `tbl_{section}_{page}_{seq}` — tables
- `fig_{section}_{page}_{seq}` — figure descriptions
- `caut_{section}_{page}_{seq}` — cautions
- `warn_{section}_{page}_{seq}` — warnings
- `maint_{section}_{schedule}_{seq}` — maintenance schedule entries

### Section Codes (from manual structure)
| Code | Section |
|------|---------|
| 0A | General Information |
| 0B | Maintenance and Lubrication |
| 1A | Heater and Ventilation |
| 1B | Air Conditioner |
| 2A/2B | Suspension Front/Rear |
| 3A/3B | Manual/Power Steering |
| 4A | Drive Axle |
| 5A/5B | Brakes / Brakes Hydraulic |
| 6A | Engine Mechanical |
| 6B | Engine Cooling |
| 6C | Engine Fuel |
| 6D | Engine Electrical |
| 6E | Emission Controls |
| 7A/7B | Automatic/Manual Transaxle |
| 8A/8B | Electrical / Electrical Wiring |

---

## 7. Known Issues and Gotchas

### OCR Quality
- **Garbled table text**: Some tables have broken first-column OCR where every row shows the same value (e.g., "Air Cleaner Element:" for all rows in the capacities table `tbl_0b_p20_0`). The RRF-blended neural reranking was specifically designed to handle this — BM25 can still match on keywords like "Crankcase" or "3.7 Quarts" even when the cross-encoder is confused by the garbled text.
- **Garble detection**: `src/ocr_text.py` drops blocks with >3% em-dash/symbol ratio (min 3 chars). This catches OCR'd diagram borders that produce garbage text.
- **Windows encoding**: All print statements use ASCII characters only (no unicode arrows, em-dashes, etc.) to avoid cp1252 crashes on Windows console.

### Pipeline
- **Stage A is expensive and already done**: 771 pages at 350 DPI. Never re-run unless the source PDF changes.
- **Vision API caching**: `build/vision_classify_cache.json` persists Claude Vision classify results. Re-running Stage D uses cached results (0 API calls for already-processed figures).
- **PAUSE mechanism**: `build/PAUSE` file exists — the pipeline checks for this and pauses between pages. Delete it or use `tools/resume.py`.

### Retrieval
- **System detection edge cases**: The `_detect_system()` function uses keyword length scoring. "oil capacity" was a specific regression that was fixed by adding it to maintenance keywords. Similar edge cases may exist for other cross-section queries.
- **Config says `gemini-2.5-flash` but memory says `claude-sonnet-4-6`**: Check `configs/default.yaml` `chat.model` for the actual current value. The model was changed during development — the config file is the source of truth.
- **Cross-encoder max_length=512 tokens**: Chunks are truncated to 500 chars for cross-encoder scoring. This can miss data in long table chunks where the relevant info is past position 500. The RRF blend mitigates this.

### Frontend
- The UI is a single `frontend/index.html` file served by FastAPI.
- CSS/JS are served from `/static/style.css` and `/static/app.js` but currently there's only `index.html` (CSS/JS likely inline or external CDN).
- Figure citations `[p5 | fig: fig_id]` are rendered as clickable links in the UI.

---

## 8. All Fixes Made (Chronological)

### Phase 1: Pipeline Quality (Fixes 1-7)
1. **Figure ID resolution** — chunker uses page-level figure IDs not regex-guessed IDs
2. **Figure vision chunks** — `vision_description` indexed as searchable text chunks
3. **Garble detection** — drops blocks with >3% em-dash/symbol ratio
4. **Section heading normalization** — truncates to canonical name from config
5. **Vision classify cache** — persists `classify_region` results across runs
6. **Pipeline unicode fixes** — ASCII-only print statements for Windows
7. **Single-page `--pages N` support** — accepts single page number

### Phase 2: Chunk Quality (Fixes 8-11, 13, 16)
8. **Plural figure mention pattern** — matches "Figures" not just "Figure"
9. **Procedures always get figures** — service manual procedures are always illustrated
10. **source_label date corruption** — rejects MM-DD-YY dates, handles sub-section letters
11. **Zombie/empty chunks filter** — drops chunks with <10 chars of text
13. **Decontextualized figure titles** — uses caption_map instead of fallback heading
16. **Decontextualized table rows** — removed table_row type, single-step procedures → paragraphs

### Phase 3: Retrieval Quality (Fixes 12, 14-15, 17)
12. **Figure flooding** — type diversity caps, dependency expansion limits
14. **Precise figure matching** — fig_num_to_id map per page, specific ID resolution
15. **Section 1D system mapping** — 1D is AC not maintenance, 1A is AC not general_info
17. **System prompt improvements** — "Cheri Doctor" identity, natural citations, mechanic tone

### Phase 4: Retrieval Precision (Recent Session)
18. **Diagnostic table splitting** — `_is_diagnostic_table()` splits CONDITION/CAUSE/CORRECTION tables by condition group regardless of row count
19. **Orphan row inheritance** — `prev_condition` carries across pages for split tables
20. **Reranking rebalance** — tables +0.10 for informational queries, +0.15 for diagnostic intent; figures get 0 boost for non-figure queries
21. **Index noise filtering** — near-empty figure chunks (<10 tokens) excluded from index; "CONDITION: CONDITION" header-as-data noise chunks dropped
22. **Cross-encoder reranker** — `ms-marco-MiniLM-L-6-v2` neural reranking blended with RRF
23. **Component synonym expansion** — 25 alias mappings (belt → drive belt V-belt, gas → fuel gasoline, etc.)
24. **Irrelevant figure suppression** — figures below 50% of top non-figure score dropped
25. **Oil capacity system detection** — "oil capacity" added to maintenance keywords; RRF-over-RRF blend prevents garbled table text from being dropped by cross-encoder

---

## 9. Config Reference (default.yaml)

Key settings you'll most likely need to change:

```yaml
pipeline:
  source_pdf: "Reference Manuals/1990 Geo Metro Manual.pdf"
  build_dir: "build"
  front_matter_pages: 4          # Skip cover/foreword pages
  toc_markdown: "Reference Manuals/TABLE OF CONTENTS.md"
  skip_pages: [11, 12, 15, 16]  # Metric/conversion tables, schedule pages

ocr:
  tesseract_cmd: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
  # UPDATE THIS PATH on the new machine

vision:
  primary: "claude"
  model: "claude-sonnet-4-6"     # For pipeline Stage D vision classification

chat:
  model: "gemini-2.5-flash"     # Chat LLM — check this is what you want
  provider: "gemini"
  top_k_retrieve: 20
  top_n_rerank: 10
  temperature: 0.1

chunking:
  min_tokens: 200
  max_tokens: 600

indexing:
  embeddings:
    model: "all-MiniLM-L6-v2"
    dimension: 384
```

---

## 10. Quick Start on New Machine

1. **Clone/copy the project** including the `build/` directory (contains all pipeline outputs)

2. **Install Python 3.13** and create a venv

3. **Install PyTorch with CUDA first:**
   ```bash
   pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Install remaining deps:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Tesseract OCR** and update the path in `configs/default.yaml` if different

6. **Set up API keys** in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=...
   ```

7. **Start the chat server:**
   ```bash
   python tools/chat_backend/serve.py
   ```

8. **Open http://localhost:8000** in your browser

### If You Need to Rebuild the Index
```bash
python -m src.pipeline --stages "F,G" --reprocess
```
This re-chunks from `build/document.json` and rebuilds the search index. Takes ~2 minutes.

### If You Need to Re-run the Full Pipeline
```bash
python -m src.pipeline --stages "B,C,D,E,F,G" --reprocess
```
Stage D (OCR + vision) is the most expensive. Vision classify results are cached in `build/vision_classify_cache.json`.

**Do NOT re-run Stage A** unless you lost the `build/pages/` directory. It just rasterizes the PDF.

---

## 11. Verified Test Queries

These queries have been tested and produce good results:

| Query | Expected Answer |
|-------|----------------|
| "What is the oil capacity for the engine?" | 3.7 Quarts (3.5 Liters) — cites tbl_0b_p20_0 |
| "What causes engine overheating?" | Comprehensive cause list from diagnostic table tbl_6_p221_3 |
| "How do I replace the thermostat?" | Full procedure with figures |
| "What is the spark plug gap?" | 1.0-1.1mm (0.039-0.043 in.) — cites para_6d4_p336_0 |
| "What causes engine surging?" | Diagnostic table causes from tbl_6_p221_0 |
| "What are the maintenance intervals?" | Specific intervals from maint_0b_* schedule tables |
| "What are the causes of low oil pressure?" | Focused chunk from split table tbl_6_p224_1 |
| "What is the coolant capacity?" | Should return from tbl_0b_p20_0 (4.2 Quarts / 4.0 Liters) |

---

## 12. Architecture Patterns Worth Knowing

### Chunker (src/chunker.py)
- **Procedure stitching**: Adjacent ordered_list blocks with the same heading are merged into one procedure chunk, even across pages. `starting_step` tracks where numbering begins.
- **Fill-down**: For diagnostic tables, empty cells in the CONDITION column are filled from the last non-empty value above.
- **Cross-page inheritance**: `prev_condition` parameter carries the last condition value from one table to the next (handles page-break splits).
- **Figure chunk creation**: When a figure block has `vision_description`, a text chunk of type "figure" is created alongside the figure record. This makes figure descriptions searchable.

### Index (src/index_build.py)
- **BM25 tokenization** (`_tokenize`): Includes text + section_path + source_label. Figure chunks get extra "figure diagram illustration" keywords.
- **Embedding text** (`_embed_text`): Prepends section_path. Figure chunks get "Figure/Diagram in {section}" prefix.
- **Near-empty figure filter**: Chunks with type "figure" and <10 tokens are excluded from the index (still in chunks.jsonl for dependency resolution).

### Chat (src/chat.py)
- **SYSTEM_KEYWORDS**: 27 system categories, each with a keyword list. Scoring uses total matched keyword character length. Short keywords (<=4 chars) require word boundaries.
- **QUERY_EXPANSIONS**: Abbreviation → full form (VIN, ECM, TBI, etc.)
- **MAINTENANCE_QUERY_SUPPLEMENTS**: Adds spec-related terms to common maintenance queries.
- **COMPONENT_ALIASES**: 25 synonym mappings appended (not replaced) to queries.
- **Cross-encoder**: Loaded lazily as a global singleton `_cross_encoder`.

---

## 13. Things That Could Be Improved (Future Work)

Based on the most recent audit:

1. **Upgrade embedding model**: `all-MiniLM-L6-v2` (384-dim) is dated. Models like `bge-small-en-v1.5` or `gte-small` may perform better.
2. **Contextual retrieval** (Anthropic approach): Prepend LLM-generated context to each chunk before embedding. Would help garbled table text significantly.
3. **Weighted RRF**: Currently BM25 and embedding get equal weight in RRF. Tuning the weights could improve precision.
4. **Fix garbled table `tbl_0b_p20_0`**: The capacity table has garbled first-column OCR ("Air Cleaner Element:" for every row). Could be fixed by re-extracting with vision or manually correcting the table in document.json.
5. **Streaming responses**: The chat server currently waits for the full LLM response before returning. Streaming would improve UX.
6. **Conversation memory**: The chat has basic conversation context (last 6 messages for query rewrite) but no persistent memory of the user's maintenance history.
