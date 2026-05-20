# Geo Metro RAG Index Audit - Compiled Recommendations

This document compiles the findings and recommendations from the RAG index audit performed by Copilot, Codex, Claude, and Gemini-API.

---

## 1. Embedding & Model Improvements

### Upgrade Embedding Model
*   **Observation:** The current index uses `all-MiniLM-L6-v2` (384-dim). This small, general-purpose model underperforms on dense automotive technical text and abbreviations.
*   **Recommendation:** Swap to `msmarco-distilbert-base-tas-b` or `all-mpnet-base-v2` (768-dim) for better semantic capture.
*   **Source:** Copilot, Claude

### Bridge Terminology Gaps (OBD/DTC)
*   **Observation:** The 1990 manual uses terms like "ALDL" and "trouble code," while users search for "OBD" and "DTC."
*   **Recommendation:** Add synonym bridges in `index_build.py`: `OBD` ↔ `ALDL`, `DTC` ↔ `trouble code`, `PCM` ↔ `ECM`.
*   **Source:** Claude

### Expand 'System' Metadata
*   **Observation:** The `system` field (e.g., `ac`, `steering`) is used for filtering but is missing from searchable text.
*   **Recommendation:** Map system codes to human-friendly terms (e.g., `ac` → `air conditioning`) and include them in the chunk's searchable metadata.
*   **Source:** Gemini-API

---

## 2. Chunking & Structural Integrity

### Raise the Floor on Chunk Size
*   **Observation:** Over 700 chunks are 10 tokens or shorter (e.g., "Weather strips."), which provides poor context for retrieval.
*   **Recommendation:** Extend short-chunk filters to non-figure types or merge sub-20-token fragments into neighboring chunks in `src/chunker.py`.
*   **Source:** Codex

### Fix Body Service Subsections (Chapter 10)
*   **Observation:** Chapter 10 content often collapses to a generic `section_code=10`, losing specific subsection granularity for doors, seats, etc.
*   **Recommendation:** Promote subsection codes from `source_label` (e.g., `10-4-1`) to `section_code` during indexing.
*   **Source:** Codex

### Implement "Criterion 2" for Diagnostic Tables
*   **Observation:** `_is_diagnostic_table` in `src/chunker.py` only checks headers, missing many "Condition/Cause/Correction" tables with poor OCR.
*   **Recommendation:** Implement the logic to detect multiple non-empty values in the first column to capture these diagnostic blocks.
*   **Source:** Gemini-API

---

## 3. Metadata Enrichment

### Re-derive Missing `info_types`
*   **Observation:** ~19% of non-figure chunks (paragraphs, tables, notices) have empty `info_types`, missing out on synonym expansion.
*   **Recommendation:** Run a one-shot rebuild using `INFO_TYPE_PATTERNS` to populate empty fields.
*   **Source:** Copilot

### Inject Referenced Figure Captions
*   **Observation:** Procedures often refer to parts by number (e.g., "Adjust screw (1)"), with the part name only appearing in the figure caption.
*   **Recommendation:** Pull `caption_text` from referenced figures into the procedure's searchable metadata.
*   **Source:** Gemini-API

### Enrich Caption-Only Figures
*   **Observation:** 145 figures have only a short label (e.g., "Figure 8 Blower Motor") without descriptions, making them hard to find conceptually.
*   **Recommendation:** Run vision extraction or inject nearest heading/procedure context into the figure's embedding text.
*   **Source:** Claude

---

## 4. Content Noise & Filtering

### Default Engine Variant to G10
*   **Observation:** The manual is G10-specific, but all chunks are tagged as `both`.
*   **Recommendation:** Default `engine_variant` to `G10` to make the retrieval filter meaningful.
*   **Source:** Copilot

### Deduplicate and Normalize Tables/Advisories
*   **Observation:** Repeated boilerplate (SIR cautions, "Notes on Fault Tree") and pipe-heavy OCR table text inflate noise.
*   **Recommendation:** Add near-exact-text deduplication for notices/cautions and normalize table rendering.
*   **Source:** Codex
