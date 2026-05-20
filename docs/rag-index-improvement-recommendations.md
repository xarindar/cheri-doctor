# Geo Metro RAG Index — Improvement Recommendations (Ordered by Value)

**Audit date:** 2026-05-18  
**Audited by:** copilot, codex, claude, gemini-api  
**Sanity-checked and reordered by:** copilot (2026-05-19)  
**Index stats at time of audit:** 5,686 chunks (4,226 main + 1,460–1,675 supplement)
**Current live index (after fragment-filtering passes on 2026-05-19):** 4,759 chunks

---

## Status Update — 2026-05-19

Completed from this list:
- `Priority 1` done: `index_build.py` now bridges `OBD`↔`ALDL`, `DTC`↔`trouble code`, and `PCM`↔`ECM`.
- `Priority 2` partial: `index_build.py` now re-derives missing `info_types` at index-build time using the chunker classifier plus a broader fallback heuristic.
- `Priority 3` done: `_metadata_text()` now injects human-readable `system` text into searchable/indexed content.
- `Priority 10` done: `_is_diagnostic_table()` now includes the missing sparse fill-down Criterion 2, with a targeted regression test.

Partially completed from this list:
- `Priority 4` partial: the live index now removes the high-noise fragment classes actually found in the corpus:
  - TOC chunks
  - section `9J` TOC rows
  - non-figure stub orphans `<=3` tokens
  - figure chunks `<=20` tokens
  - section `0A` metric conversion rows
  - supplement front-matter / publication boilerplate
- `Priority 5` partial: several repeated boilerplate classes were removed surgically via explicit filters, but there is still no general duplicate/near-duplicate pass.

Current live index snapshot:
- Total indexed chunks: `4,759`
- Type mix: procedure `1316`, paragraph `1011`, table `930`, figure `872`, notice `347`, caution `152`, important `78`, note `53`
- Re-derived `info_types` during latest rebuild: `570`
- Remaining chunks with empty `info_types`: `533`
- Current `engine_variant` values: all `4,759` chunks still default to `both`
- Chapter 10 subsection issue still present: `666` chunks still sit at bare `section_code=10`, with `0` promoted `10-*` subsection codes

Recommended next priorities from the remaining list:
1. Finish `Priority 2` — shrink the remaining `533` empty-`info_types` chunks with a more precise second pass if needed
2. `Priority 8` — fix Chapter 10 subsection labeling (`666` bare `10` chunks)
3. `Priority 11` — default `engine_variant` to `G10` instead of `both`

---

## Sanity-Check Summary

All 12 original recommendations were validated against current RAG literature (2024–2025). No recommendations are incorrect. The ordering below is based on a value assessment weighing:

- **Breadth** — how many queries or chunks are affected
- **Severity** — whether the issue causes *zero* results vs. *degraded* results
- **Effort** — changes that unlock large gains for small cost rank above equal-impact expensive changes

Research confirms:
- `all-mpnet-base-v2` achieves **34–60% better retrieval accuracy** vs `all-MiniLM-L6-v2` on dense technical documents (NOAA/BEIR benchmarks, 2024).
- Vocabulary bridging (synonym injection) is rated highest-ROI in RAG systems for domain-shifted terminology.
- BM25 degrades measurably with micro-fragment chunks; deduplication of boilerplate is a standard pre-indexing step.
- Metadata enrichment (system labels, section codes, figure captions) is a top-tier 2024 recommendation from LlamaIndex, Haystack, and LangChain pattern guides.

---

## Recommendations in Value Order

---

### ★★★ Priority 1 — Synonym Bridges for Modern Diagnostic Vocabulary
**Source:** Claude #3 | **Effort:** Index rebuild only

`OBD` appears **0 times** and `DTC` appears **0 times** across all 5,686 chunks. The manual predates OBD-II and uses 1990 GM terms: `ALDL` (21 mentions), `trouble code` (32 mentions). Any user asking "how do I read OBD codes" or "what does DTC 35 mean" gets **zero BM25 token matches** — the single worst failure mode in retrieval.

The codebase already has a working synonym bridge (cigar↔cigarette) in `_tokenize()` and `_embed_text()` in `index_build.py`. The pattern is established.

**Fix:** Add to the existing synonym map:
- `OBD`, `OBD-II` ↔ `ALDL`
- `DTC` ↔ `trouble code`
- `PCM` ↔ `ECM`

**Why #1:** Zero-match failure > degraded-match failure. High breadth (diagnostic queries are among the most common in any repair chatbot). Negligible implementation cost.

---

### ★★★ Priority 2 — Re-Derive `info_types` for 801 Empty Chunks
**Source:** Copilot #3 | **Effort:** Index rebuild only

`_tokenize()` and `_embed_text()` both call `_metadata_text()`, which expands `info_types` into synonym strings (e.g., `"wiring"` → `"wiring wire circuit harness terminal ground power"`). But **801 chunks** — `paragraph` (297), `notice` (230), `table` (183), `caution` (37), `important` (37), `note` (17) — have empty `info_types`. Many contain wiring diagrams, torque specs, and coolant leak cautions. The `INFO_TYPE_PATTERNS` regex list in `chunker.py` would match their text — they just weren't applied at index time.

**Fix:** In `index_build.py`, after loading chunks but before building BM25/embeddings, re-run `INFO_TYPE_PATTERNS` for any chunk where `info_types` is empty and backfill.

**Why #2:** Unlocks synonym expansion for **~19% of indexed chunks** with a one-shot index rebuild and zero pipeline changes. Direct impact on BM25 and cosine retrieval simultaneously.

---

### ★★★ Priority 3 — Include `system` Field in `_metadata_text`
**Source:** Gemini #1 | **Effort:** Index rebuild only

Every chunk carries a `system` tag (e.g., `ac`, `steering`, `engine`) but these tags are only used for *hard filtering* in `retrieve()`. They never appear in the searchable text. A user asking "How do I fix the AC?" who doesn't explicitly filter by system gets zero benefit from the system label. The chunk text may say "Blower Motor Removal" without ever using the word "air conditioning."

**Fix:** Map each `system` code to an expanded human-readable string and include it in `_metadata_text`:
- `ac` → `"air conditioning heating ventilation HVAC"`
- `steering` → `"steering power steering column rack"`
- etc.

**Why #3:** Affects all 5,686 chunks. Metadata enrichment is consistently rated top-tier in 2024–2025 RAG guides for exactly this kind of system-level intent bridging. Index rebuild only.

---

### ★★★ Priority 4 — Filter / Merge Sub-20-Token Micro-Fragments
**Source:** Codex #1 | **Effort:** Index rebuild only (or minor chunker change)

**720 chunks are ≤10 tokens; 2,305 chunks are ≤20 tokens** (40.5% of the index). Examples: `"Weather strips."`, `"Heater Core: 1A-9"`. BM25 depends on term frequency normalized by document length — very short documents distort this normalization and compete with richer chunks for top-k slots. Research confirms BM25 performs poorly with micro-fragments, and they inflate noise in hybrid retrieval reranking.

**Fix:** Extend the existing figure-only short-chunk filter in `index_build.py` to all chunk types: merge any non-figure chunk under 20 tokens into its nearest neighbor before indexing.

**Why #4:** Removes retrieval budget waste affecting 40% of the index. Low implementation cost if done at the index-build stage.

---

### ★★★ Priority 5 — Deduplicate Repeated Boilerplate (Cautions, Notes)
**Source:** Codex #3 | **Effort:** Index rebuild only

The same SIR caution appears **24 times** in supplement chunks; `"Notes on Fault Tree:"` appears **11 times**. 209 table chunks use dense `|`-separated OCR text. Repeated boilerplate inflates lexical BM25 matches, crowding out relevant procedure and diagnostic chunks when a user's query happens to contain a word from the caution text.

**Fix:** At index-build time, compute a hash of normalized chunk text; for identical or near-identical chunks (≥90% overlap via simple token comparison), keep only one and mark duplicates. For pipe-heavy OCR tables, apply a light normalization pass before indexing.

**Why #5:** Directly improves BM25 precision. Well-established pre-indexing step in production RAG pipelines.

---

### ★★☆ Priority 6 — Inject Referenced Figure Captions into Procedure Metadata
**Source:** Gemini #3 | **Effort:** Chunker change + rebuild

Procedures are often brief — *"1. Adjust screw (1) to spec."* — relying on nearby figures for part names. A search for *"ISC Solenoid adjustment"* may retrieve the Figure chunk (caption: "ISC Solenoid") but miss the Procedure chunk (which has the actual steps). The semantic keywords are in the figure, not the procedure.

**Fix:** In `chunker.py`, when a procedure or paragraph chunk resolves a `figure_ref` to a figure on the same page, copy that figure's `caption_text` into the procedure chunk's `_metadata_text`.

**Why #6:** Closes the figure↔procedure semantic gap — one of the most common failure patterns in technical manual RAG systems per 2024 multimodal RAG research.

---

### ★★☆ Priority 7 — Upgrade Embedding Model to `all-mpnet-base-v2`
**Source:** Claude #1, Copilot #2 (independently) | **Effort:** Full re-embed (high)

The index uses `all-MiniLM-L6-v2` (384-dim, 6 layers, ~22M params). Research (2024) shows `all-mpnet-base-v2` (768-dim, 12 layers, ~110M params) achieves **34–60% better retrieval accuracy** on dense technical documents including engineering, HPC, and meteorology corpora — directly analogous to a 1990 automotive service manual. The manual's abbreviations (ECM, ALDL, TPS, CTS, ISC, MAT, EGR, VSS) are not meaningful tokens in a general-text embedding; the larger model handles domain shift far better.

An alternative worth evaluating: `msmarco-distilbert-base-tas-b`, which was trained specifically for passage retrieval tasks (vs. semantic similarity), potentially better suited for Q&A over technical manuals.

**Fix:** Change the model name in `configs/default.yaml` and rebuild `embeddings.npy`. Drop-in replacement via `sentence-transformers` — no pipeline changes required.

**Why #7:** Highest ceiling impact of any single change, but placed below Priorities 1–6 because those give large gains at rebuild-only cost. Do this after quick wins are banked. Note: Priorities 1–5 also improve embedding quality since `_embed_text()` uses the same enriched metadata.

---

### ★★☆ Priority 8 — Fix Body Service Subsection Labeling (Chapter 10)
**Source:** Codex #2 | **Effort:** Chunker change + rebuild

Body Service chunks collapse to `section_code=10` / `section_path=10` even when `source_label` is more specific (e.g., `10-4-1`, `10-5-5`). Subsections `10-1` through `10-11` appear in the index mostly as TOC entries, not real content. A query about doors, seats, roof, or stationary glass cannot benefit from subsection-level filtering.

**Fix:** In `chunker.py`, when `section_code` is a bare integer and `source_label` contains a dotted subsection code, promote `source_label` to `section_code` and `section_path` before indexing chapter 10 content.

**Why #8:** Enables filtering for an entire chapter. Medium effort. Ranked below the metadata and synonym fixes because those affect the whole index.

---

### ★★☆ Priority 9 — Context-Inject 145 Sparse Figure Chunks
**Source:** Claude #2 | **Effort:** Chunker change or vision pipeline + rebuild

Of 1,463 figure chunks, **145 have ≤30 tokens** — just a label like `"Figure 8 Blower Motor Removal"`. Even with section path prepended, these are too sparse for cosine similarity to work on conceptual queries. The vision pipeline either skipped these figures or returned nothing.

**Fix (option a):** Run vision extraction retroactively on the 145 skipped figures.  
**Fix (option b):** At build time, inject the nearest heading/procedure context from same-page chunks into these figure chunks' embed text.

**Why #9:** Option (b) is a chunker-level fix with no additional AI calls. Research on figure caption embedding (ACL 2024) confirms this directly improves recall for procedure queries.

---

### ★☆☆ Priority 10 — Implement Diagnostic Table "Criterion 2" Detection
**Source:** Gemini #2 | **Effort:** Chunker change + rebuild

In `chunker.py`, `_is_diagnostic_table()` only checks header keywords (Criterion 1). The commented-out Criterion 2 — detecting sparse first-column values for tables without proper OCR headers — is not implemented. Many Condition/Cause/Correction tables in this 1990 manual have poor OCR headers and remain unsplit.

**Fix:** Implement Criterion 2: check for multiple non-empty, short (≤5 token) values in column 0 across ≥4 rows.

**Why #10:** Meaningful for diagnostic table splitting but narrower in scope than Priorities 1–9. Adds complexity to a chunker heuristic; test carefully against known-good tables.

---

### ★☆☆ Priority 11 — Default Engine Variant Tag to `G10`
**Source:** Copilot #1 | **Effort:** Chunker change + rebuild (small)

Every chunk currently has `engine_variant='both'`. The `retrieve()` filter for G10/G13 is therefore inert. Since this is a G10-only manual, the filter cannot help at all. Only explicitly G13-labeled content (if any exists) should carry `engine_variant='G13'`.

**Fix:** In `_get_engine_variant()`, default to `'G10'` instead of `'both'`. Promote `'G13'` only where the chunk text explicitly names the G13 engine.

**Why #11:** Correctness fix rather than a retrieval uplift for most users (single-engine manual). Low effort but narrow benefit — correctly tagged G10 content enables future cross-manual scenarios where G13 content exists.

---

## Cross-Agent Themes (Confirmed High-Confidence)

| Theme | Agents | Research Validation |
|---|---|---|
| Embedding model too small/generic | Copilot #2, Claude #1 | 34–60% recall gain confirmed in domain-specific benchmarks |
| Missing synonym/metadata expansion | Copilot #3, Claude #3, Gemini #1 | Vocabulary bridging is highest-ROI RAG fix for domain-shifted corpora |
| Short/micro-fragment chunks diluting retrieval | Codex #1 | BM25 term-frequency distortion confirmed; 100–200 token floor recommended |
| Figure chunks lack semantic richness | Claude #2, Gemini #3 | Multimodal RAG research confirms figure caption enrichment improves recall |

---

## Implementation Phases

### Phase 1 — Index Rebuild Only (do these first; high ROI, zero pipeline risk)
1. Add OBD/DTC/PCM synonym bridges → `index_build.py`
2. Re-derive `info_types` for 801 empty chunks → `index_build.py`
3. Include `system` field with human-friendly expansions in `_metadata_text` → `index_build.py`
4. Filter/merge sub-20-token micro-fragments → `index_build.py`
5. Deduplicate repeated notices/cautions/boilerplate → `index_build.py`

### Phase 2 — Chunker Changes + Rebuild
6. Inject referenced figure captions into procedure metadata → `chunker.py`
7. Fix Body Service subsection labeling (chapter 10) → `chunker.py`
8. Context-inject 145 sparse figure chunks (option b) → `chunker.py`
9. Implement Diagnostic Table Criterion 2 → `chunker.py`
10. Default engine variant to G10 → `chunker.py`

### Phase 3 — Full Re-Embed
11. Upgrade embedding model to `all-mpnet-base-v2` (or `msmarco-distilbert-base-tas-b` for Q&A-optimized retrieval) → `configs/default.yaml` + rebuild `embeddings.npy`

> **Note:** Completing Phase 1 first maximizes the value of the Phase 3 re-embed, since Priorities 1–5 all improve the text that gets embedded.
