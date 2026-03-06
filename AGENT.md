# Metro Manual — Chat With The Manual

## GOAL
Build a "chat with the manual" system optimized for LLM troubleshooting grounded in a scanned PDF manual.
Output MUST preserve structure (sections, procedures, cautions, figures, legends, tables) and support RAG + Claude Vision evidence injection.
Markdown is NOT canonical; JSON is canonical. HTML/MD/CSV are derived.

## DELIVERABLES
1) `build/manifest.json`
   - doc_id, source_pdf, dpi, page_count, run_id, timestamps
2) `build/document.json`
   - full structured representation of pages -> blocks -> relations (canonical truth)
3) `build/chunks.jsonl`
   - retrieval units (atomic text chunks) with metadata for RAG
4) `build/figures.jsonl`
   - figure metadata + links to exported image assets
5) `build/assets/`
   - cropped images for figures/diagrams/tables (webp preferred)
6) `build/tables/`
   - per-table CSV (when grid detected) + table image crop
7) `tools/rag_index/`
   - embeddings index + BM25 index + mappings chunk_id -> record
8) `tools/chat_backend/`
   - retrieve->rerank->attach figure images->answer with citations

## CONSTRAINTS / QUALITY BARS
- Never OCR a whole page blindly. Always: raster -> content crop -> classify regions -> OCR per region.
- Avoid narrow strips: ensure text crops are upscaled (2x–3x) before OCR.
- Preserve ordered list step numbers and cautions as first-class objects.
- Figures and wiring schematics are evidence images; do not "convert wiring diagrams" to tables.
- Require citations in responses (chunk_id + page + label) and include figure_ids when using images.

## PIPELINE OVERVIEW
```
A) Ingest & Rasterize
B) Page Preprocessing (crop margins, deskew, normalize)
C) Layout Segmentation (Complexity Classifier: SIMPLE vs COMPLEX)
D) Region-Specific Extraction
   - COMPLEX Page: Claude Vision full-page structured extraction
   - SIMPLE Page:
     - Text: column detection -> OCR with correct PSM
     - Tables: grid detection -> OCR per cell -> CSV + text rendering
     - Figures: crop/export image + caption detection + legend association
E) Structure Reconstruction
   - Page Continuation Stitching (merge multi-page procedures)
   - heading hierarchy -> section_path for each block
F) Retrieval Corpus Build (chunks.jsonl + system mapping + engine variants)
G) Index Build (BM25 + embeddings)
H) Chat Orchestration (Keyword pre-filter -> System-aware Retrieve -> Intent-based Rerank -> attach figures -> answer w/ citations)
```

## IMPLEMENTATION PLAN (STEP-BY-STEP)
... (rest of the file)

### STEP 0 — Repo Layout
Create a repo with:
```
src/
  ingest.py
  preprocess.py
  layout.py
  ocr_text.py
  table_extract.py
  figure_extract.py
  structure.py
  chunker.py
  index_build.py
  chat.py
build/ (output)
data/ (pdf inputs)
configs/
  default.yaml
```

### STEP 1 — Rasterize PDF consistently
- Use 300–400 DPI (start at 350)
- Output per-page images (PNG or TIFF) to `build/pages/page_###.png`
- Record page dimensions in `manifest.json`

### STEP 2 — Preprocess each page
For each raster page:
1) Compute content bbox from image (NOT Surya big region):
   - grayscale -> adaptive threshold -> connected components -> union bbox
   - add margin (40–80px)
2) Deskew:
   - estimate skew angle using Hough lines or projection profile
3) Normalize:
   - light denoise
   - contrast normalize (CLAHE)

Output: `build/pages_pre/page_###.png`
Store preproc transforms in `document.json` page metadata.

### STEP 3 — Layout segmentation (region detection)
Goal: blocks = {heading, paragraph, list, table, legend, figure/schematic}

Approach:
- Use Surya/LayoutParser or existing detector, but run it on the PREPROCESSED page crop.
- Post-process regions:
  - Merge overlapping text boxes into blocks
  - Identify likely "legend" blocks: dense short lines with key/value patterns OR numbered list near a figure
  - Identify "table" blocks: strong horizontal/vertical line structure OR repeated aligned columns

Store each region as:
- block_id (stable)
- type (initial classification)
- bbox in page coordinates
- order index (approx reading order)

### STEP 4 — Column detection for text regions
For each text block likely multi-column:
- Build vertical projection histogram of ink density
- Find valley in middle band (40–60% width) => gutter
- Split into columns accordingly
- Add padding (30–80px) around column crops
- Upscale crops 2x–3x (Lanczos)

### STEP 5 — OCR for text
For each column crop:
- Tesseract oem=1 (LSTM)
- Use psm=4 for single column, psm=6 for single block if needed
- Preserve line breaks; run light cleanup to rejoin wrapped lines conservatively
- Detect ordered list numbering patterns and store as `ordered_list.steps[]` when present
- Detect WARNING/CAUTION/NOTE blocks and store separately (type=warning/caution/note)

Write results into `document.json` blocks:
- `paragraph.text` OR `ordered_list.steps[]` OR `caution.text`, etc.

### STEP 6 — Figure extraction and linking
For each figure/schematic region:
- Export crop to `build/assets/page_###_fig_##.webp`
- Find caption:
  - nearest text block below within N pixels
  - look for `"Figure \d+"` or `"Fig."` patterns
- Find legend mapping:
  - nearby text block with numbered items (1..N) OR key/value list near figure
- Store figure block:
  - figure_id, page, bbox, asset_path
  - caption_text (if found)
  - legend_items[] (if parseable)
  - relations: caption_of, legend_for

**IMPORTANT:** Wiring schematics remain figure blocks. Do not attempt graph reconstruction in v1.

### STEP 7 — Table extraction (v1 pragmatic)
Detect tables (ruled):
- Use line detection (morphology) to detect grid
- If grid found:
  - segment into cells
  - OCR per cell (psm=7 or 6 depending)
  - output:
    - `build/tables/page_###_table_##.csv`
    - table block in `document.json` with `rows[][]` + csv_path + asset_path
- If unruled table:
  - treat as text block + preserve alignment heuristically
  - still export crop image and store as table_image evidence

Also generate a "retrieval rendering" for each table:
- key/value style lines OR "Row i: ..." lines for embeddings/BM25.

### STEP 8 — Heading hierarchy and section paths
For each page:
- Identify headings by typography cues:
  - all caps
  - larger font
  - leading section codes like "1B-7"
- Build a running section stack:
  - chapter -> section -> subsection
- Assign section_path + source_label to each block and chunk:
  - e.g. `"AIR CONDITIONER > Recovery Tank Installation"`
  - source_label might include `"1B-6"` if reliably detected

### STEP 9 — Build retrieval chunks (chunks.jsonl)
Chunking rules:
- One chunk per ordered procedure list
- One chunk per caution/warning/note
- Paragraphs chunked into ~200–600 tokens
- Legends as key/value chunks
- Tables as "rendered text" chunks + link to CSV + image asset
- Each chunk record includes:
  - chunk_id (stable)
  - doc_id
  - page
  - bbox (optional)
  - type: procedure|warning|caution|note|paragraph|legend|table
  - section_path
  - source_label
  - text (rendered string)
  - steps[] (if procedure)
  - kv[] (if legend)
  - figure_refs[] (figure_ids)
  - asset_refs[] (paths if any)

### STEP 10 — Build indices
- BM25 over chunk.text + section_path + source_label
- Embeddings index over chunk.text
- Store mapping chunk_id -> record + file offsets for fast lookup

### STEP 11 — Chat orchestration (Claude Vision)
Given user question + conversation state:
1) Retrieve top K (BM25 + embeddings union)
2) Rerank (either:
   - simple weighted score; or
   - LLM rerank prompt; or
   - cross-encoder if available)
3) Collect figure_refs from top N evidence chunks
4) Attach corresponding images (asset_paths) to Claude Vision call
5) Prompt rules:
   - Answer ONLY using provided evidence
   - Cite chunk_id + page + source_label
   - If figure used, cite figure_id + page
   - If evidence insufficient, say what's missing and ask 1 targeted question
6) Return:
   - diagnosis/troubleshooting steps
   - citations
   - optional "next check" question

### STEP 12 — Evaluation harness
Create a small test set:
- 20–50 user symptom queries
- Expected relevant sections/figures

Metrics:
- retrieval hit rate (correct chunk in top K)
- citation correctness (chunk/page alignment)
- hallucination rate (claims not in evidence)
- latency and cost

## OUTPUT EXAMPLE (what "good" looks like)
Response citations format:
- `[p12 AIR CONDITIONER 1B-7 | chunk: proc_1b7_filter_drier_steps]`
- `[p12 | fig: fig_8_control_panel]`

## IMPLEMENTATION NOTES (PRAGMATIC DEFAULTS)
- DPI: 350
- Upscale: 2x for text crops; 3x if font is tiny
- Tesseract:
  - text columns: `--oem 1 --psm 4 -l eng`
  - single block: `--psm 6`
  - single cell: `--psm 7`
- Image export: WebP (quality ~80–90), keep PNG for debug
- Don't overfit gutter split; compute per block/page with projection valley

## WHAT TO BUILD FIRST (ORDER)
1) Preprocess + content crop + deskew
2) Region detection + figure export
3) Text block column split + OCR (procedures + cautions)
4) Chunker + indices
5) Chat retrieval + citations
6) Tables (ruled) extraction
7) Legend parsing improvements
8) HTML renderer (nice-to-have, derived)

## DONE CRITERIA
- A user asks a symptom question
- System retrieves 5–10 relevant chunks
- Claude answers with a step-by-step troubleshooting path
- Citations include chunk_id/page + figure references when used
- If a diagram is relevant, it is attached and cited
- No answer content without evidence

## EXECUTION NOTES
Prefer correctness and stable IDs over cleverness. Keep outputs deterministic and reproducible (manifest includes run_id and config hash).

## EXISTING STACK (from V2 pipeline)
| Component | Library | Notes |
|-----------|---------|-------|
| PDF rendering | PyMuPDF (fitz) | Keep |
| Image preprocessing | OpenCV (contrib) | CLAHE, adaptive threshold, deskew, border removal |
| Layout classification | Surya | Use for region detection on preprocessed crops |
| Column detection | Custom (NumPy/OpenCV) | Vertical projection profile gutter detection |
| Primary OCR | Tesseract 5.4 | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| Secondary OCR | EasyOCR (GPU) | Neural net OCR for consensus |
| Table extraction | img2table | Structural table detection with per-cell OCR |
| Spell correction | SymSpell | Custom automotive dictionary (~300 terms) |
| Diagram descriptions | Claude Vision API (Sonnet) | GPT-4o fallback |
| GPU | PyTorch + CUDA 12.8 | RTX 2080 |
| Image handling | Pillow | Image manipulation |

## ENVIRONMENT
- Python 3.13.5, Windows 11
- Working directory: `D:\Metro Project`
- PDFs: `1990 Geo Metro Manual.pdf` (771 pages, 2.1GB), `Vert Sup.pdf` (913MB)
