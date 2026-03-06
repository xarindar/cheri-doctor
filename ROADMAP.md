# Metro Manual — Chat With The Manual: Project Roadmap

## Project Summary
Convert the 1990 Geo Metro factory service manual (771-page scanned PDF) into a
RAG-powered chat system. The user can ask natural-language questions and receive
evidence-cited answers drawn directly from the manual, including figures.

**Stack:** Python 3.13 · PyMuPDF · Surya layout · Tesseract + EasyOCR (consensus)
· SymSpell correction · rank-bm25 · sentence-transformers (all-MiniLM-L6-v2)
· Claude Sonnet 4.6 (chat + vision) · FastAPI · Windows 11 · RTX 2080

**Project root:** `D:\Metro Project\`
**PDF:** `Reference Manuals/1990 Geo Metro Manual.pdf` (771 pages)
**Car name:** Cheri · **Assistant name:** Cheri Doctor

---

## Vehicle Profile (baked into system prompt)
- **VIN:** JG1MR3362LK769576
- **Year/Model:** 1990 Geo Metro, 3-door hatchback
- **Engine:** 1.0L 3-cylinder G10 (NOT the G13 4-cyl)
- **Transmission:** Automatic transaxle (Section 7A)
- **Steering:** Manual (Section 3A)
- **A/C:** Yes — converted from R-12 to R-134a in 2007 (manual specs obsolete)
- **Maintenance schedule:** Schedule I (severe service) — 3,000 mi / 3 month oil changes
- **Mileage:** ~80,000 miles
- **Service history:** Unknown (recently acquired from previous owner)

---

## Repo Layout
```
D:\Metro Project\
  src/
    __init__.py
    pipeline.py          # Main orchestrator — stages A-H
    ingest.py            # Stage A: PDF rasterization (350 DPI -> PNG)
    preprocess.py        # Stage B: crop, deskew, CLAHE, border removal
    layout.py            # Stage C: Surya layout + CV box detection
    ocr_text.py          # Stage D.1: OCR + block type classification + garble filter
    table_extract.py     # Stage D.2: table grid detect + per-cell OCR
    figure_extract.py    # Stage D.3: figure crop + caption + legend + vision
    structure.py         # Stage E: heading hierarchy + section_path
    chunker.py           # Stage F: type-aware chunk generation
    index_build.py       # Stage G: BM25 + embedding indices
    chat.py              # Stage H: retrieve -> rerank -> cite -> answer
    models.py            # Shared dataclasses
    utils.py             # JSON I/O, timing, path helpers
  configs/
    default.yaml                # All pipeline settings
    chat_system_prompt.txt      # System prompt with vehicle profile + Cheri persona
  build/
    manifest.json               # Stage A output (kept across runs)
    page_metas.json             # Stage A output (kept across runs)
    pages/                      # Rasterized PNGs — 771 pages at 350 DPI (COMPLETE)
    vision_cache.json           # describe_diagram cache (persist across runs)
    vision_classify_cache.json  # classify_region cache (persist across runs, saves $)
    document.json               # CANONICAL TRUTH (all processed pages)
    chunks.jsonl
    figures.jsonl
    pages_pre/                  # Preprocessed PNGs (Stage B)
    assets/                     # Figure WebP crops (Stage D)
    tables/                     # CSV + WebP table crops (Stage D)
  tools/
    rag_index/                  # bm25_index.pkl · embeddings.npy · mappings
    chat_backend/
      serve.py                  # FastAPI server (port 8000)
  Reference Manuals/
    1990 Geo Metro Manual.pdf
    TABLE OF CONTENTS.md        # Manually converted TOC (-> 51 RAG chunks)
    maintenance-schedule.csv    # Pages 15-16 combined (Schedule I + II)
  legacy/                       # Original pipeline modules (imported, not copied)
    ocr_engine.py
    layout_analysis.py
    text_correction.py
    table_extract.py
    table_extract_v2.py
    vision_describe.py          # describe_diagram + load/save_description_cache
  .env                          # ANTHROPIC_API_KEY=sk-... (never commit)
```

---

## Stages

### ✅ Phase 1 — Scaffold, Config, Ingest, Preprocess
- [x] `configs/default.yaml` — all settings
- [x] `src/models.py` — added `procedure_type`, `system`, `engine_variant`
- [x] `src/ingest.py` — skip-if-exists rasterization
- [x] `src/preprocess.py` — skip-if-exists preprocessing

### ✅ Phase 2 — Layout + Figure Export
- [x] `src/layout.py` — added `Complexity Classifier` (reasons for routing)
- [x] `src/vision_extract.py` — added `Full Page Vision Extraction` for COMPLEX pages
- [x] `src/pipeline.py` — added `Page Continuation Stitching` across boundaries

### ✅ Phase 3 — Text OCR + Block Classification
- [x] `src/ocr_text.py` — added `procedure_type` detection and `Domain Vocabulary Validator`
- [x] `src/chunker.py` — added `Header-Only Chunk Prevention` and `Table Row Misclassification` fixes

### ✅ Phase 4 — Table Extraction (vision-first)
- [x] Improved vision-first routing and transcription caching

### ✅ Phase 5 — Structure Reconstruction
- [x] `src/structure.py` — mapping section_id to system

### ✅ Phase 6 — Pipeline Orchestrator + document.json
- [x] `src/pipeline.py` — coordinated COMPLEX/SIMPLE routing

### ✅ Phase 7 — Chunker + Indices
- [x] `src/chunker.py` — populated `system` and `engine_variant` (G10/G13)
- [x] `src/index_build.py` — added pre-filtering by system and engine

### ✅ Phase 8 — Chat Backend + UI
- [x] `src/chat.py` — implemented hybrid retrieval with keyword pre-filter and intent-based reranking

### ✅ Phase 8 — Chat Backend + UI
- [x] `src/chat.py` — retrieve -> rerank -> Claude API -> parse citations
  - [x] Fixed: citation format simplified to `[p{page} | {chunk_id}]`
  - [x] Fixed: inline citations stripped from answer text; deduplicated
- [x] `configs/chat_system_prompt.txt` — vehicle profile, evidence-only rules, Cheri persona
- [x] `tools/chat_backend/serve.py` — FastAPI on port 8000 (was 8001)
- [x] `frontend/index.html` — user messages right-aligned, renamed to "Cheri Doctor",
      avatar "CD", figure cards display inline

---

## Current State

### Pipeline Build Status
| Stage | Status | Notes |
|-------|--------|-------|
| A — Rasterize | **COMPLETE** | 771 pages, kept in `build/pages/` |
| B — Preprocess | Wiped, ready | Runs B-G fresh |
| C — Layout | Wiped, ready | |
| D — Extract | Wiped, ready | Vision caches preserved (31 entries) |
| E — Structure | Wiped, ready | |
| F — Chunk | Wiped, ready | |
| G — Index | Wiped, ready | |

### Ready to Run
```powershell
python -m src.pipeline --stages "B,C,D,E,F,G"
```

---

## What's Left

### Completed (Audit 1–10)
- [x] Chunk ID consistency
- [x] source_label populated (646 distinct per-page labels, 0 null)
- [x] CAUTION block isolation + double-prefix fix
- [x] Table rows with full condition/cause/correction context
- [x] Index table chunking (flat topic lists, no single-letter fragments)
- [x] Procedure chunks back in index (1172 chunks, was 0)
- [x] Figure chunks searchable (1433 chunks, was 32)
- [x] Figure captions populated (92% — 1332/1448)
- [x] Vision type mapping (procedure/header/notice/important → chunker types)
- [x] Page header bleed-in stripped from chunk text
- [x] Small spec tables kept as single chunks (< 15 rows)
- [x] Full-page vision cache (`build/full_page_vision_cache.json`, 571 pages)
- [x] All `mixed` and `text` pages routed through Vision (93 additional pages)
- [x] **Orphaned procedure fragments** (fixed by cross-page stitching and `starting_step`)
- [x] **Decontextualized small table chunks** (fixed by `last_caption`/`source_label` fallback)
- [x] **Step renumbering artifacts** (fixed by regex cleaning and sub-step detection)
- [x] **Missing procedure_type** (fixed by keyword heuristic for OCR-path)

### Completed — Audit 11 Fixes

- [x] **source_label corruption (REGRESSION, fixed)**
  Root cause: `page_label` override in structure.py was updating the
  SectionStack from unreliable vision OCR, corrupting all subsequent pages.
  Fix: proper section-code labels (with letter, e.g. `7C-3`) update the
  stack; ambiguous ones (e.g. `5-18`) only accepted if prefix matches stack.
  Result: 4192/4261 chunks in proper SECTION-PAGE format, 0 null labels.

- [x] **Procedure misclassification (fixed)**
  Symptom lists with short items + SYMPTOM/DIAGNOSTIC in title now
  reclassified as paragraph instead of procedure.

- [x] **Cross-section retrieval filtering (fixed)**
  Expanded SECTION_SYSTEM_MAP from 22 to 36 entries (all manual sections).
  Expanded SYSTEM_KEYWORDS from 9 to 19 systems. Raised broadening
  threshold from 3 to 5. Increased cross-reference penalty from -0.1 to -0.25.
  System distribution now balanced across 23 systems (was concentrated in 3).

- [x] **Figure retrieval absent (fixed)**
  Added `figure: 0.15` to TYPE_BOOST in reranker. Enriched figure
  embed text with `"Figure/Diagram in {section}"` prefix. Added
  `figure diagram illustration` keywords to BM25 tokenizer. Fixed
  `_collect_figures` to include images from figure-type chunks directly.

### Open Issues — Pipeline (Stages D/E, may need API)

- [ ] **116 figures with null captions (8%)**
  Mostly figures where caption is embedded in the image (wiring diagrams
  with "CODE 21" title bars). Would need OCR on figure crop bottom or
  vision description to extract. Current fallback uses nearest text block
  below figure.

- [ ] **Unicode artifacts in 3 source_labels**
  Section divider pages produce labels like `ENGINE \ufffd 6`. Fix: strip
  replacement chars in structure.py's `PAGE_LABEL_RE` or in page_label
  extraction.

### Future
- [ ] **Process second manual** — Vert Sup (supplemental, path TBD)
- [ ] **Evaluation harness** — 20–50 test queries with expected sections/figures

---

## Quick Reference

### Pipeline Commands
```powershell
# Full run from B onward (A already done)
python -m src.pipeline --stages "B,C,D,E,F,G"

# Specific page range
python -m src.pipeline --pages 40-45 --stages "D,E,F,G"

# Single page (reprocess)
python -m src.pipeline --pages 17 --stages "D,E,F,G" --reprocess

# Rebuild chunks + index only (after fixing chunker/chat)
python -m src.pipeline --stages "F,G"

# Skip vision (faster, no figure descriptions)
python -m src.pipeline --stages "B,C,D,E,F,G" --skip-vision
```

### Chat Server
```powershell
# Start
python tools/chat_backend/serve.py

# Open browser
start http://localhost:8000
```

---

## Key Bugs Fixed (cumulative)
1. Column detection failure — running headers masked gutter. Fix: skip top 10% + bottom 3%.
2. Gutter expansion zero-width — threshold below min. Fix: `min_val * 1.5` fallback.
3. Spell correction destroys paragraph structure. Fix: split on `\n\n`, correct per-para.
4. UnicodeDecodeError on Windows JSON. Fix: `encoding="utf-8"` everywhere.
5. Re-rasterization on every run. Fix: `out_path.exists()` check in ingest.py.
6. Vision API key not found. Fix: load dotenv at top of pipeline.py.
7. Wrong image space for extraction. Fix: pass `pre_img` to extractors.
8. Figure IDs wrong (fig_3 vs fig_p0027_000). Fix: page-level ID collection in chunker.
9. VIN location not findable. Fix: vision_description creates searchable chunk.
10. Oil viscosity chart OCR'd as heading. Fix: garble detector with em-dash detection.
11. {2} false positive in garble filter. Fix: min absolute count of 3 garbage chars.
12. Section headings absorbing 500+ chars of content. Fix: normalize to config section_names.
13. classify_region results not cached. Fix: vision_classify_cache.json persisted per figure.
14. Unicode crashes in pipeline print statements. Fix: replaced all non-ASCII chars.
15. --pages N single page caused IndexError. Fix: handle single-element split.
16. Vision blocks with `text: null` crash chunker. Fix: `block.get("text") or ""` everywhere.
17. Vision full-page cache for COMPLEX pages. Fix: `full_page_vision_cache.json` (571 pages).
18. `block.get("text", "")` returns None for explicit null. Fix: `or ""` pattern throughout.
19. Vision type mismatch — `procedure`/`header`/`notice`/`important` silently dropped.
    Fix: type normalization map at top of chunker block loop.
20. Zero procedure chunks (1445 blocks dropped). Fix: map `procedure` → `ordered_list` in chunker.
21. Figure captions always None. Fix: fallback to nearest text block below in figure_extract.py.
22. Figure chunks not searchable (32/1448). Fix: create chunk from caption+legend+vision_desc.
23. Index tables over-chunked to single letters. Fix: `_is_index_table()` + flat rendering.
24. Table groupby producing "SECTION: O" fragments. Fix: expanded `INDEX_REF_RE` for alpha refs.
25. CAUTION double-prefix ("CAUTION: CAUTION:"). Fix: check if text already starts with prefix.
26. Page header bleed-in ("8-30 BODY AND CHASSIS"). Fix: `PAGE_HEADER_RE` strip in `_make_chunk`.
27. source_label coarse (16 values for 763 pages). Fix: use `page_label` per-page in structure.py.
28. Small spec tables split per-row. Fix: tables < 15 data rows kept as single chunk.
29. `last_heading` reset per page. Fix: moved outside page loop to carry across pages.
30. Mixed pages (61) routed to OCR. Fix: `page_type == "mixed"` → Vision in pipeline.py.
31. text/complex=False pages (32) with broken columns. Fix: route to Vision in pipeline.py.
32. Vision title field dropped for paragraph blocks. Fix: `text or title or None` in pipeline.py.
33. Cross-page stitching fails when figures precede continuation. Fix: search for `continues_from` block in `_stitch_blocks`.
34. Stale `source_label` on figure-only pages. Fix: update `SectionStack` from `page_label` in `structure.py`.
35. Orphaned procedures lose starting step. Fix: export `starting_step` metadata and use keyword heuristic for `procedure_type`.
36. Table/Procedure context lost on figure-heavy pages. Fix: track `last_caption` and fallback to `source_label`.
37. source_label regression — SectionStack corrupted by misread page_labels. Fix: only update stack from proper section codes (with letter); validate ambiguous labels against stack.
38. Procedure misclassification — symptom lists typed as procedure. Fix: detect SYMPTOM/DIAGNOSTIC title + short steps pattern, reclassify as paragraph.
39. Figure retrieval absent (5 audits). Fix: add TYPE_BOOST for figures, enrich embed text, add figure keywords to BM25, collect images from figure-type chunks.
40. Cross-section retrieval unfiltered (11 audits). Fix: expand SECTION_SYSTEM_MAP (22→36), SYSTEM_KEYWORDS (9→19), raise broadening threshold (3→5), increase cross-ref penalty (0.1→0.25).
41. Config file lost pipeline/ocr/structure sections. Fix: restored from backup, merged with new provider config.
