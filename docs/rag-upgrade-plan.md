# RAG Upgrade Plan
_Consensus plan from Claude, Codex, and Copilot review of chunker.py, chat.py, and models.py_

---

## Consensus Priorities

All three agents agree on the following order and scope:

| # | Change | Rebuild? | Risk |
|---|--------|----------|------|
| 1 | Add `section_path` to evidence prompt header | No | Trivial |
| 2 | Use relation graph directly in `_expand_dependencies()` | No | Low |
| 3 | Dynamic TYPE_BOOST — extend existing `_rerank()` hooks | No | Low |
| 4 | Inline figure captions into procedure chunk text | Yes | Low |
| 5 | Sentence-boundary splitting in `_chunk_paragraphs()` | Yes | Low |
| 6 | Add `table_type` metadata (start small) | Yes | Medium |

**Explicitly out of scope:** spaCy NER, general-purpose NER, mini-summarization before LLM.

---

## Item 1 — Add `section_path` to evidence prompt header
**File:** `src/chat.py` · `_build_messages()` (~line 2483)

**Problem:** The evidence block header shows `chunk_id | page | source_label | type`. The `section_path` (e.g. `DRIVEABILITY AND EMISSIONS > Section 6E2 > ECM Connector Identification`) is in chunk metadata but never shown to the LLM.

**Change:** Add `section_path` to the header:
```
--- {chunk_id} | page: {page} | {section_path} | type: {type} ---
```

**Why first:** Zero risk, no rebuild, immediate improvement to Claude's context and citation quality.

---

## Item 2 — Use relation graph directly in `_expand_dependencies()`
**File:** `src/chat.py` · `_expand_dependencies()` (~line 2207)

**Problem:** The function uses `figure_refs` correctly but then falls through to scanning all of `index.lookup` heuristically for chart/diagram text patterns. It ignores `related_figure_ids`, `related_table_ids`, and `same_page_figure_ids` — which were built specifically for this traversal.

**Change:**
- **Same-page figures:** use `chunk["related_figure_ids"]` and `chunk["same_page_figure_ids"]` as first-class graph edges — direct lookups via `index.lookup[id]` instead of scanning.
- **Same-page tables:** use `chunk["related_table_ids"]` for table cross-referencing (e.g. diagnostic chart key + flowchart).
- **Adjacent-page charts:** keep a narrow page±1 fallback *only* for the specific facing-page diagnostic flowchart case (common in section 6E2). Do not use `same_page_chunk_ids` for this — that field is same-page only and stores chunk IDs, not figure IDs.
- Keep text-pattern fallback for chunks that lack populated graph edges (OCR pages with incomplete metadata).

**Why second:** Biggest precision and performance win with no index rebuild. Fixes the O(index) scan and reduces accidental unrelated figure pulls.

---

## Item 3 — Dynamic TYPE_BOOST via existing `_rerank()` hooks
**File:** `src/chat.py` — extend the existing `_rerank()` intent signals

**Problem:** `TYPE_BOOST` is a static dict. A diagnostic query should surface tables more aggressively; a how-to query should surface procedures more aggressively.

**Constraint (Copilot):** `chat.py` already has `is_procedural`, `is_diagnostic`, and some special-case boost logic in `_rerank()`. A new parallel intent-boost system would double-count and get brittle. Extend the existing hooks rather than adding a new layer.

**Change:** Inside the existing intent-detection path in `_rerank()`, adjust per-type boost values based on detected intent:
- Diagnostic signal → raise `table` boost, add `diagnostic` info_type boost
- How-to / procedural signal → raise `procedure` boost
- Connector/pinout signal → raise `pinout` table boost (once `table_type` exists)

**Why third:** No rebuild needed, and it's lower-risk than structural changes.

---

## Item 4 — Inline figure captions into procedure chunk text
**File:** `src/chunker.py` · `_chunk_procedure()` (~line 704)

**Problem:** When a procedure step says "see Figure 3," the `figure_refs` link is stored but the figure's caption text is absent from the chunk. The LLM must infer context or issue a separate lookup.

**Change:** After resolving `figure_refs`, look up each figure's `caption_text` from the page's `caption_map` and append a compact inline reference to the chunk text:
```
[Figure 3: Fuel injector assembly — throttle body components]
```
Only do this for figures with substantive captions (>15 chars). Do not inline `vision_description` (too long).

**Requires:** Index rebuild.

---

## Item 5 — Sentence-boundary splitting in `_chunk_paragraphs()`
**File:** `src/chunker.py` · `_chunk_paragraphs()` (~line 1231)

**Problem:** The function merges whole paragraph blocks and splits only at the character-count limit, cutting mid-sentence. This degrades embedding quality for oversized merged blocks.

**Change:** When a paragraph block would exceed `max_chars`, split at the nearest sentence boundary before the limit. Use a deterministic heuristic — period/exclamation/question mark followed by whitespace and a capital letter, or a newline. **No spaCy** — OCR text is too noisy for general sentence parsers, and the manual's prose is already fairly well-delimited.

**Requires:** Index rebuild. Be conservative — only split oversized merges, don't fragment normally-sized paragraphs.

---

## Item 6 — Add `table_type` metadata (start small)
**Files:** `src/models.py` · `src/chunker.py` · `src/chat.py`

**Problem:** Every table gets a flat `+0.10` TYPE_BOOST regardless of whether it's a diagnostic troubleshooting table, a torque spec, or a cross-reference index. The code already detects diagnostic tables (`_is_diagnostic_table()`) and index tables (`_is_index_table()`) but discards those labels after use.

**Change:**
- Add `table_type: str | None` to `ChunkRecord` in `models.py`
- In `_chunk_table()`, classify using the already-present detection functions plus new heuristics:
  - `diagnostic` — already detected by `_is_diagnostic_table()`
  - `spec` — header contains VOLTAGE / RESISTANCE / TORQUE / PRESSURE / CLEARANCE
  - `pinout` — header contains PIN / TERMINAL / CIRCUIT / WIRE COLOR
  - `index` — already detected by `_is_index_table()`
- Start with these four types. Expand to `maintenance`, `torque`, `general` only after validating retrieval improvement.
- In `chat.py`, reference `table_type` in the boost logic added in Item 3.

**Requires:** Schema change to `models.py` + full index rebuild.

---

## Notes

- Items 1–3 can ship as a batch with no rebuild.
- Items 4–5 can ship together (one rebuild).
- Item 6 ships alone (schema change + rebuild).
- After Item 6, revisit the boost values in Item 3 with real `table_type` signal available.
