# Geo Metro RAG Upgrade Plan

**Date:** 2026-05-19
**Prepared by:** Codex
**Scope:** `src/chunker.py`, `src/chat.py`, `src/index_build.py`, retrieval index rebuilds, and targeted evaluation

## Goal

Improve answer quality for the Python chat system by prioritizing upgrades that increase retrieval precision and recall on the current manual corpus, not just code cleanliness or implementation convenience.

## Planning Principles

1. Prefer fixes that improve the existing corpus and ranking signals before adding more model complexity.
2. Treat rebuild-only changes as the fastest path to measurable gains.
3. Use deterministic metadata and graph signals before introducing costly NLP or summarization layers.
4. Require evaluation after each phase so the team does not stack speculative changes without evidence.

## Current State

The code already has several capabilities that should be preserved and extended rather than reimplemented:

- Query expansion already exists in `src/chat.py`.
- Deep research retrieval already exists in `src/chat.py`.
- Same-page and related-artifact metadata is already populated in `src/chunker.py`.
- Evidence expansion already exists in `src/chat.py`, but it still relies heavily on heuristic rescans instead of directly using the relation fields as the primary graph.

That means the best upgrade plan is not a greenfield redesign. It is a focused tightening of metadata quality, retrieval edges, chunk labeling, and evaluation.

## Priority Order

### Phase 1: Highest-ROI Rebuild and Retrieval Improvements

These should be done first because they improve the live system quickly with low architecture risk.

1. Finish metadata backfill and searchable-text enrichment in `src/index_build.py`
   - Re-derive remaining empty `info_types`.
   - Keep the existing synonym bridges and extend only where gaps are proven by failed queries.
   - Ensure human-readable `system` context is consistently injected into indexed text.
   - Success check: fewer empty `info_types`, better recall on known diagnostic/spec/wiring queries.

2. Reduce obvious retrieval noise in the index
   - Continue removing tiny orphan fragments and repeated boilerplate that compete for BM25 slots.
   - Keep filtering surgical; do not merge or delete chunks that contain unique procedural steps.
   - Success check: fewer irrelevant top-k hits on audit queries and smaller live index without recall loss.

3. Make relation metadata drive evidence expansion in `src/chat.py`
   - Change `_expand_dependencies()` to prefer `related_figure_ids`, `related_table_ids`, and `same_page_chunk_ids` before doing broad heuristic scans through the entire lookup.
   - Preserve heuristic fallback only when explicit links are missing.
   - Success check: more precise figure/table attachment and fewer unrelated dependency pulls.

### Phase 2: Chunk Semantics and Labeling Fixes

These are the next best upgrades because they improve what gets indexed and how reranking understands it.

1. Add explicit `table_type` metadata
   - Extend the chunk model and chunker so tables can be typed as `diagnostic`, `spec`, `pinout`, `maintenance`, `torque`, or `wiring` where possible.
   - Use deterministic heuristics from headers, section context, and row structure.
   - Feed `table_type` into reranking boosts instead of relying on raw text patterns like `"CONDITION:"`.
   - Success check: better ranking on spec lookups, pinout queries, and diagnostic questions.

2. Expose hierarchical context more explicitly
   - Include `section_path` more visibly in the final evidence packaging sent to the model.
   - If needed, prepend condensed section hierarchy to chunk text only for chunk classes that are commonly decontextualized.
   - Do not blindly duplicate long headings into every chunk.
   - Success check: fewer answers that confuse subsection context or mix nearby topics.

3. Finish chapter-10 subsection promotion and engine tagging correctness
   - Keep the chapter-10 subsection promotion so body-service retrieval can use real subsection codes.
   - Correct `engine_variant` defaults only if the corpus and routing logic support that cleanly.
   - Success check: better filtering for body-service and engine-specific lookups.

4. Improve figure and procedure semantic coupling
   - Inject referenced figure caption context into procedure/paragraph indexing text when the reference is explicit.
   - Focus on cases where the step text uses numeric callouts and the semantic part name only exists in the figure.
   - Success check: procedure retrieval improves for component-name queries that currently retrieve only figures.

### Phase 3: Retrieval Model and Ranking Upgrades

These are valuable, but only after the metadata and chunk graph are in better shape.

1. Upgrade the embedding model after the index text is cleaned up
   - Evaluate `all-mpnet-base-v2` and any retrieval-tuned alternative against the current baseline.
   - Do not swap models blindly before Phase 1 and Phase 2 are measured.
   - Success check: measurable improvement on the evaluation set, not just better theoretical benchmarks.

2. Make reranking intent-aware using structured metadata
   - Replace some hard-coded text heuristics in `_rerank()` with boosts tied to `table_type`, `info_types`, procedure type, and query intent.
   - Keep the current approach interpretable.
   - Success check: better top-n ordering across procedural, diagnostic, and specification queries.

### Phase 4: Only If Earlier Phases Plateau

These ideas are lower priority because they add cost and failure modes.

1. Advanced NLP segmentation
   - Sentence-boundary or semantic merge logic should be evaluated only if deterministic chunking still leaves obvious fragmentation after earlier fixes.

2. General-purpose NER
   - Only worth doing if regex/entity coverage proves insufficient on a measured set of real queries.

3. Pre-answer evidence summarization
   - Only introduce summarization if context limits become a measured blocker after retrieval quality is improved.

## Implementation Sequence

1. Finish the remaining rebuild-only metadata and noise work in `src/index_build.py`.
2. Refactor `src/chat.py` dependency expansion to use the explicit relation graph first.
3. Add `table_type` and wire it into chunking plus reranking.
4. Improve evidence packaging with clearer section hierarchy.
5. Apply targeted figure/procedure context enrichment.
6. Run a measured embedding-model bakeoff.

## Evaluation Plan

Create a fixed regression set before making large retrieval changes.

Include at least these categories:

- Diagnostic lookup queries using modern terms like `OBD`, `DTC`, `PCM`, and `check engine`.
- Specification lookups such as torque, capacity, resistance, and voltage.
- Pinout and connector-identification queries.
- Procedure queries where steps depend on nearby figures.
- Body-service queries inside chapter 10 subsections.
- Deep-research prompts spanning multiple related subsystems.

For each phase, record:

- Top-k retrieval relevance by query.
- Top-n evidence ordering quality.
- Whether the right figures/tables were attached.
- Whether final answers cited the right chunk family.

## Deliverables

- Updated index-building and retrieval code.
- Rebuilt index artifacts.
- A small regression-query pack with before/after results.
- A short post-phase report showing which upgrades actually moved answer quality.

## Recommended First Ticket Set

1. Refactor dependency expansion to use explicit relation fields first.
2. Add `table_type` metadata and reranking support.
3. Finish the remaining metadata backfill and fragment cleanup pass.
4. Improve evidence prompt headers to include clearer section hierarchy.
