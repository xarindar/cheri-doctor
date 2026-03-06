"""Stage G: Search Index Construction.

Builds two indices over chunks.jsonl for hybrid retrieval:
1. BM25 (keyword matching) via rank_bm25
2. Dense embeddings (semantic search) via sentence-transformers

Retrieval uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import load_jsonl, save_json


def build_indices(chunks_path: "Path | list[Path]", config: dict, index_dir: Path):
    """Build BM25 and embedding indices from one or more chunks.jsonl files.

    Args:
        chunks_path: A single Path or list of Paths to chunks.jsonl files.
                     When multiple files are provided they are merged in order
                     (supplement files should come last so their chunk_ids are
                     distinct and the lookup contains both corpora).

    Saves to:
      index_dir/bm25_index.pkl       — BM25Okapi object
      index_dir/embeddings.npy        — (N, D) float32 embeddings
      index_dir/chunk_ids.json        — ordered list of chunk_ids
      index_dir/chunk_lookup.json     — chunk_id -> full chunk record
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(chunks_path, Path):
        chunks_paths = [chunks_path]
    else:
        chunks_paths = list(chunks_path)

    chunks = []
    for cp in chunks_paths:
        batch = load_jsonl(cp)
        chunks.extend(batch)
        print(f"  Loaded {len(batch)} chunks from {cp.name}")
    if not chunks:
        print("  No chunks found, skipping index build.")
        return

    # Filter out near-empty figure chunks (<10 tokens) from the search index.
    # These are caption-only entries like "Figure 7 Blower Case Removal" that
    # add BM25 noise. They remain in chunks.jsonl for dependency expansion.
    MIN_FIG_TOKENS = 10
    before = len(chunks)
    chunks = [c for c in chunks if not (c.get("type") == "figure" and c.get("token_count", 0) < MIN_FIG_TOKENS)]
    dropped = before - len(chunks)
    if dropped:
        print(f"  Filtered {dropped} near-empty figure chunks (<{MIN_FIG_TOKENS} tokens) from index")

    print(f"  Building indices for {len(chunks)} chunks...")

    # ── BM25 ──────────────────────────────────────────────────────
    cfg_bm25 = config.get("indexing", {}).get("bm25", {})
    corpus   = [_tokenize(c) for c in chunks]
    bm25     = BM25Okapi(corpus, k1=cfg_bm25.get("k1", 1.5), b=cfg_bm25.get("b", 0.75))
    with open(index_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("  BM25 index saved.")

    # ── Embeddings ────────────────────────────────────────────────
    cfg_emb    = config.get("indexing", {}).get("embeddings", {})
    model_name = cfg_emb.get("model", "all-MiniLM-L6-v2")
    batch_size = cfg_emb.get("batch_size", 64)

    print(f"  Encoding {len(chunks)} chunks with {model_name}...")
    model  = SentenceTransformer(model_name)
    texts  = [_embed_text(c) for c in chunks]
    embs   = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)

    np.save(str(index_dir / "embeddings.npy"), embs.astype(np.float32))

    # ── Mappings ──────────────────────────────────────────────────
    chunk_ids = [c["chunk_id"] for c in chunks]
    save_json(chunk_ids, index_dir / "chunk_ids.json", indent=0)

    lookup = {c["chunk_id"]: c for c in chunks}
    save_json(lookup, index_dir / "chunk_lookup.json")

    print(f"  Embeddings saved: {embs.shape}")
    print(f"  Index complete — {len(chunks)} chunks indexed.")


def _tokenize(chunk: dict) -> list[str]:
    """Tokenize a chunk for BM25: text + section_path + source_label.

    For figure chunks, add 'figure diagram' keywords so they match queries
    about figures/diagrams even when the caption text is very short.
    """
    parts = [
        chunk.get("text", ""),
        chunk.get("section_path", ""),
        chunk.get("source_label", "") or "",
    ]
    if chunk.get("type") == "figure":
        parts.append("figure diagram illustration")
        parts.append(chunk.get("section_path", ""))  # double-weight section context
    combined = " ".join(p for p in parts if p)
    # Simple whitespace tokenizer, lowercased, alphanumeric only
    return re.findall(r"[a-z0-9]+", combined.lower())


def _embed_text(chunk: dict) -> str:
    """Text to embed: section_path prefix + chunk text.

    For figure chunks, prepend 'Figure/Diagram' and the section context
    to strengthen embeddings (bare captions like 'Figure 3' are too short).
    """
    prefix = chunk.get("section_path", "")
    text   = chunk.get("text", "")
    ctype  = chunk.get("type", "")

    if ctype == "figure":
        # Enrich short figure text with section context for better embedding
        parts = [f"Figure/Diagram in {prefix}" if prefix else "Figure/Diagram"]
        if text:
            parts.append(text)
        return " - ".join(parts)

    return f"{prefix}: {text}" if prefix else text


class RetrievalIndex:
    """Unified BM25 + embedding retrieval with RRF merge."""

    RRF_K = 60          # Standard RRF constant
    SUPPLEMENT_BOOST = 0.15  # Added to RRF score for supplement chunks (source_doc="supplement")

    def __init__(self, index_dir: Path, config: dict | None = None):
        self.index_dir = index_dir

        with open(index_dir / "bm25_index.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        self.embeddings: np.ndarray = np.load(str(index_dir / "embeddings.npy"))

        with open(index_dir / "chunk_ids.json", encoding="utf-8") as f:
            self.chunk_ids: list[str] = json.load(f)

        with open(index_dir / "chunk_lookup.json", encoding="utf-8") as f:
            self.lookup: dict = json.load(f)

        cfg_emb    = (config or {}).get("indexing", {}).get("embeddings", {})
        model_name = cfg_emb.get("model", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 20, 
                 system: str | None = None,
                 engine_variant: str | None = "G10") -> list[dict]:
        """Hybrid retrieval: BM25 + embedding, merged via RRF.

        Args:
            query: User's question.
            top_k: Number of results to return.
            system: Optional system name to filter by (e.g. 'ac', 'steering').
            engine_variant: Engine to filter by ('G10', 'G13', or 'both').
                           Defaults to 'G10' (user's car).

        Returns list of chunk records, ordered by relevance.
        """
        # Pre-filter chunk IDs if filtering is requested
        valid_ids = None
        if system or engine_variant:
            valid_ids = set()
            for cid, chunk in self.lookup.items():
                if system and chunk.get("system") != system:
                    continue
                if engine_variant:
                    chunk_ev = chunk.get("engine_variant", "both")
                    if engine_variant != "both" and chunk_ev != "both" and chunk_ev != engine_variant:
                        continue
                valid_ids.add(cid)
            
            # If filtering too strictly, return few results.
            # The chat orchestration can decide to broaden if needed.
            if not valid_ids:
                return []

        bm25_ranked = self._bm25_search(query, top_k * 2, valid_ids)
        emb_ranked  = self._embedding_search(query, top_k * 2, valid_ids)
        merged      = self._rrf_merge(bm25_ranked, emb_ranked, top_k)

        # Boost supplement chunks so they surface above main-manual chunks
        # when both cover the same topic — supplement is source of truth.
        results = []
        for cid, score in merged:
            if cid not in self.lookup:
                continue
            chunk = self.lookup[cid]
            if chunk.get("source_doc") == "supplement":
                score += self.SUPPLEMENT_BOOST
            results.append({"chunk": chunk, "score": score})

        # Re-sort after boost
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def get(self, chunk_id: str) -> dict | None:
        return self.lookup.get(chunk_id)

    # ── Internal ──────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int, valid_ids: set[str] | None = None) -> list[tuple[str, float]]:
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        scores = self.bm25.get_scores(tokens)
        
        if valid_ids is not None:
            # Mask scores for invalid IDs
            mask = np.array([cid in valid_ids for cid in self.chunk_ids])
            scores = scores * mask
            
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def _embedding_search(self, query: str, top_k: int, valid_ids: set[str] | None = None) -> list[tuple[str, float]]:
        q_emb  = self.model.encode([query], normalize_embeddings=True)[0]
        sims   = self.embeddings @ q_emb   # cosine similarity (normalized)
        
        if valid_ids is not None:
            # Mask sims for invalid IDs (set to -1.0)
            mask = np.array([cid in valid_ids for cid in self.chunk_ids], dtype=float)
            sims = sims * mask + (1.0 - mask) * -1.0
            
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(self.chunk_ids[i], float(sims[i])) for i in ranked if sims[i] > -1.0]

    def _rrf_merge(self,
                   bm25_results: list[tuple[str, float]],
                   emb_results:  list[tuple[str, float]],
                   top_k: int) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for rank, (cid, _) in enumerate(bm25_results):
            scores[cid] = scores.get(cid, 0) + 1 / (self.RRF_K + rank + 1)
        for rank, (cid, _) in enumerate(emb_results):
            scores[cid] = scores.get(cid, 0) + 1 / (self.RRF_K + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
