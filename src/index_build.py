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

from src.chunker import _infer_info_types
from src.utils import load_jsonl, save_json


REVIEWED_INFO_TYPES_PATH = Path(__file__).resolve().parents[1] / "tmp" / "rag_empty_info_types_review_2026-05-19.jsonl"
ALLOWED_INFO_TYPES = {
    "connector_face",
    "cross_reference",
    "diagnostic",
    "diagram",
    "location",
    "pinout",
    "procedure",
    "safety",
    "spec",
    "wiring",
}


FALLBACK_INFO_TYPE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("pinout", re.compile(r"\b(?:pinout|terminal|cavity|pin\s*\d+|connector\s+[a-z]\d*)\b", re.IGNORECASE)),
    ("connector_face", re.compile(r"\b(?:connector face|end view|terminal end view|face layout)\b", re.IGNORECASE)),
    ("wiring", re.compile(r"\b(?:wire|wiring|circuit|harness|ground|relay|connector|switch)\b", re.IGNORECASE)),
    ("diagnostic", re.compile(r"\b(?:inspect|check|test|fault|faulty|continuity|diagnos(?:is|tic)|symptom|crank only|should\b|leak|leakage|condition|cause|correction|recovery|recycling)\b", re.IGNORECASE)),
    ("location", re.compile(r"\b(?:location|located|mounted|behind|under|below|above|left side|right side|column|position)\b", re.IGNORECASE)),
    ("spec", re.compile(r"\b(?:torque|viscosity|resistance|voltage|pressure|clearance|capacity|n·m|lb\.?ft|psi|kpa|mm|inch|oil filter|air cleaner element|filter drier|refrigerant-12|standards?|approved)\b", re.IGNORECASE)),
    ("cross_reference", re.compile(r"\b(?:see\s+[\"']?.+?procedure|see section|refer to|in this section)\b", re.IGNORECASE)),
    ("safety", re.compile(r"\b(?:caution|warning|notice|important|safety|hazard|injury|personal|risk|protective|goggles|wear)\b", re.IGNORECASE)),
)


SECTION_REF_RE = re.compile(r"\b\d+[A-Z]?(?:-\d+)+(?:-\d+)?\b")
MICRO_FRAGMENT_TOKEN_LIMIT = 20
ADVISORY_TYPES = {"warning", "caution", "notice", "important", "note"}
EMPTY_ADVISORY_RE = re.compile(r"^(?:warning|caution|notice|important|note):?\s*$", re.IGNORECASE)
NOTE_STUB_RE = re.compile(r"^note:\s*(?:o\s+operated|inspect)\s*$", re.IGNORECASE)
FAULT_TREE_NOTE_RE = re.compile(r"^notes?\s+on\s+fault\s+tree:?\s*$", re.IGNORECASE)
SUPPLEMENT_BOILERPLATE_NOTICE_RE = re.compile(
    r"^notice:\s+whenever it becomes necessary to perform procedures not included",
    re.IGNORECASE,
)


def _normalized_chunk_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_micro_fragment(chunk: dict) -> bool:
    if chunk.get("type") == "figure":
        return False
    return (chunk.get("token_count") or 0) < MICRO_FRAGMENT_TOKEN_LIMIT


def _is_empty_advisory(chunk: dict) -> bool:
    if chunk.get("type") not in ADVISORY_TYPES:
        return False
    return bool(EMPTY_ADVISORY_RE.fullmatch((chunk.get("text") or "").strip()))


def _is_fault_tree_note_fragment(chunk: dict) -> bool:
    if chunk.get("type") != "note":
        return False
    if chunk.get("source_doc") != "supplement" or chunk.get("section_code") != "9J":
        return False
    return bool(FAULT_TREE_NOTE_RE.fullmatch(_normalized_chunk_text(chunk.get("text", ""))))


def _is_navigation_fragment(chunk: dict) -> bool:
    ctype = chunk.get("type")
    if ctype not in {"paragraph", "table"}:
        return False

    text = _normalized_chunk_text(chunk.get("text", ""))
    if not text:
        return False

    info_types = set(chunk.get("info_types") or [])
    table_type = chunk.get("table_type")
    semantic_info_types = {"spec", "diagnostic", "pinout", "wiring", "location", "procedure", "safety"}

    if info_types & semantic_info_types:
        return False

    if ctype == "table":
        if table_type == "index" or "cross_reference" in info_types:
            return True
        return bool(SECTION_REF_RE.search(text))

    if "cross_reference" in info_types:
        return True
    if text.endswith(":"):
        return True
    if SECTION_REF_RE.search(text) and any(phrase in text for phrase in ("see ", "refer to", "installation", "removal", "procedure")):
        return True
    return False


def _should_drop_micro_fragment(chunk: dict) -> bool:
    """Conservative narrow filter for explicit junk/navigation fragments."""
    if _is_empty_advisory(chunk):
        return True
    if NOTE_STUB_RE.fullmatch(_normalized_chunk_text(chunk.get("text", ""))):
        return True
    if _is_fault_tree_note_fragment(chunk):
        return True
    if not _is_micro_fragment(chunk):
        return False

    ctype = chunk.get("type")

    # Procedures are more likely to carry distinct action steps even when short.
    if ctype in {"procedure", "figure"} or ctype in ADVISORY_TYPES:
        return False

    return _is_navigation_fragment(chunk)


def _dedupe_key(chunk: dict) -> tuple[str, str, str | int | None, str] | None:
    ctype = chunk.get("type")
    normalized = _normalized_chunk_text(chunk.get("text", ""))
    if not normalized:
        return None

    if ctype == "notice" and chunk.get("source_doc") == "supplement" and SUPPLEMENT_BOILERPLATE_NOTICE_RE.match(normalized):
        return ("supplement_notice", ctype, chunk.get("source_doc"), normalized)

    if ctype in ADVISORY_TYPES:
        page = chunk.get("page")
        if page is not None:
            return ("same_page_advisory", ctype, page, normalized)
        return None

    if ctype == "table" and _is_navigation_fragment(chunk):
        return ("navigation_table", ctype, chunk.get("section_code"), normalized)

    return None


def _load_reviewed_info_type_overrides() -> dict[str, list[str]]:
    if not REVIEWED_INFO_TYPES_PATH.exists():
        return {}

    overrides: dict[str, list[str]] = {}
    with REVIEWED_INFO_TYPES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chunk_id = row.get("chunk_id")
            suggested = [
                info_type
                for info_type in row.get("suggested_info_types", [])
                if info_type in ALLOWED_INFO_TYPES
            ]
            if chunk_id and suggested:
                overrides[chunk_id] = suggested

    return overrides


def _rederive_info_types(chunk: dict) -> list[str]:
    text = chunk.get("text", "") or ""
    ctype = chunk.get("type", "")
    table_type = chunk.get("table_type")
    section_path = chunk.get("section_path", "") or ""
    source_label = chunk.get("source_label", "") or ""

    inferred = _infer_info_types(chunk.get("type", ""), text)
    recovered = list(inferred) if inferred else []

    # Advisory chunks should always carry the safety semantic label.
    if ctype in ADVISORY_TYPES and "safety" not in recovered:
        recovered.append("safety")

    # Table typing is useful semantic information even when raw regex
    # inference misses because of OCR layout or chunk formatting.
    if table_type == "diagnostic" and "diagnostic" not in recovered:
        recovered.append("diagnostic")
    elif table_type == "pinout" and "pinout" not in recovered:
        recovered.append("pinout")
    elif table_type == "spec" and "spec" not in recovered:
        recovered.append("spec")
    elif table_type == "index" and "cross_reference" not in recovered:
        recovered.append("cross_reference")

    # Maintenance schedule/capacity tables are retrieval-critical and should
    # be searchable as specification/reference content even without explicit
    # chunker labels.
    maintenance_context = "\n".join(part for part in (text, section_path, source_label) if part)
    if (
        ctype == "table"
        and "spec" not in recovered
        and re.search(
            r"\b(?:maintenance schedule|scheduled maintenance|schedule i|schedule ii|"
            r"every \d|miles|months|engine oil and oil filter change)\b",
            maintenance_context,
            re.IGNORECASE,
        )
    ):
        recovered.append("spec")

    # If inferred/recovered already has content, still check for safety if it's missing.
    if recovered:
        if "safety" not in recovered:
            safety_pat = next(pat for name, pat in FALLBACK_INFO_TYPE_PATTERNS if name == "safety")
            if safety_pat.search(text):
                recovered.append("safety")
        return recovered

    context = "\n".join(
        part for part in (
            text,
            section_path,
            source_label,
        ) if part
    )

    recovered = []
    for info_type, pattern in FALLBACK_INFO_TYPE_PATTERNS:
        if pattern.search(context):
            recovered.append(info_type)

    # Short topic-to-section rows like "Heater Core: 1A-9" are cross-references.
    if (
        "cross_reference" not in recovered
        and chunk.get("type") == "table"
        and SECTION_REF_RE.search(context)
        and chunk.get("token_count", 0) <= 20
    ):
        recovered.append("cross_reference")

    if (
        ctype == "table"
        and "spec" not in recovered
        and re.search(
            r"\b(?:maintenance schedule|scheduled maintenance|schedule i|schedule ii|"
            r"every \d|miles|months|engine oil and oil filter change)\b",
            context,
            re.IGNORECASE,
        )
    ):
        recovered.append("spec")

    if ctype in ADVISORY_TYPES and "safety" not in recovered:
        recovered.append("safety")

    return recovered

def _metadata_text(chunk: dict) -> str:
    parts: list[str] = []

    # Map system codes to human-friendly searchable terms
    system_map = {
        "ac": "air conditioning heating ventilation hvac",
        "steering": "steering suspension",
        "engine": "engine motor powertrain",
        "brakes": "brakes braking system",
        "electrical": "electrical wiring circuit",
        "fuel_system": "fuel system gasoline",
        "transmission": "transmission transaxle",
        "body": "body chassis exterior interior",
    }
    system = chunk.get("system")
    if system in system_map:
        parts.append(system_map[system])

    for info_type in chunk.get("info_types") or []:
        parts.append(info_type.replace("_", " "))
        if info_type == "connector_face":
            parts.append("connector face terminal end view")
        elif info_type == "pinout":
            parts.append("pinout pin terminal cavity")
        elif info_type == "diagram":
            parts.append("diagram schematic illustration")
        elif info_type == "spec":
            parts.append("spec specification voltage resistance torque pressure capacity")
        elif info_type == "location":
            parts.append("location component position")
        elif info_type == "diagnostic":
            parts.append("diagnostic symptom cause correction")
        elif info_type == "wiring":
            parts.append("wiring wire circuit harness")
        elif info_type == "cross_reference":
            parts.append("cross reference mapping index")
        elif info_type == "safety":
            parts.append("safety warning caution hazard risk injury personal safety protective goggles")

    table_type = chunk.get("table_type")
    if table_type:
        parts.append(f"table type {table_type.replace('_', ' ')}")
        if table_type == "pinout":
            parts.append("pinout terminal cavity wire color connector")
        elif table_type == "spec":
            parts.append("spec specification torque voltage resistance pressure clearance")
        elif table_type == "diagnostic":
            parts.append("diagnostic condition cause correction troubleshooting")
        elif table_type == "index":
            parts.append("index cross reference lookup")

    entities = chunk.get("entities") or []
    if entities:
        parts.append(" ".join(entities))

    body_styles = chunk.get("body_styles") or []
    if body_styles:
        parts.append("body style " + " ".join(body_styles))

    trim_variants = chunk.get("trim_variants") or []
    if trim_variants:
        parts.append("trim variant " + " ".join(trim_variants))

    sir_equipped = chunk.get("sir_equipped")
    if sir_equipped is True:
        parts.append("sir equipped airbag")
    elif sir_equipped is False:
        parts.append("without sir non airbag")

    # Include injected figure captions for semantic bridging
    figure_captions = chunk.get("figure_captions") or []
    if figure_captions:
        parts.append("referenced figures: " + " ".join(figure_captions))

    return " ".join(parts)


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

    # Recover missing semantic labels so metadata enrichment applies to
    # chunks that were built before the current chunker inference logic.
    reviewed_overrides = _load_reviewed_info_type_overrides()
    reviewed_info_types = 0
    heuristic_info_types = 0
    for c in chunks:
        if c.get("info_types"):
            continue
        reviewed = reviewed_overrides.get(c.get("chunk_id"))
        if reviewed:
            c["info_types"] = reviewed
            reviewed_info_types += 1
            continue

        inferred = _rederive_info_types(c)
        if inferred:
            c["info_types"] = inferred
            heuristic_info_types += 1
    if reviewed_info_types or heuristic_info_types:
        print(
            "  Re-derived info_types for "
            f"{reviewed_info_types + heuristic_info_types} chunks "
            f"({reviewed_info_types} reviewed, {heuristic_info_types} heuristic)"
        )

    # --- Surgical Index Filtering (Navigation + explicit garbage) ---
    before = len(chunks)
    filtered_chunks = []
    dropped_specific = 0
    for c in chunks:
        ctype = c.get("type")
        text = (c.get("text") or "").upper()
        text_lower = (c.get("text") or "").lower()
        scode = c.get("section_code")
        tokens = c.get("token_count", 0)

        # Rule 1: TOC entries/categories (Navigation artifacts)
        if ctype in {"toc_entry", "toc_category"}:
            dropped_specific += 1
            continue
        
        # Rule 2: Section 0A metric conversion split rows
        if scode == "0A" and ctype == "table" and "FASTENER STRENGTH IDENTIFICATION" in text:
            dropped_specific += 1
            continue
            
        # Rule 3: Section 9J SIR table-of-contents rows
        if scode == "9J" and ctype == "table" and text.startswith("9J-"):
            dropped_specific += 1
            continue
            
        # Rule 4: Stub orphans
        if ctype != "figure" and tokens <= 3:
            dropped_specific += 1
            continue

        # --- LLM-known content (Bucket 2): Generic knowledge the model already has ---
        # These chunks don't add vehicle-specific signal and compete with real repair content.

        # Rule 6: Generic ESD handling procedure (universal electronics safety — not Metro-specific)
        if scode == "0A" and ctype in ("notice", "procedure") and "esd sensitive" in text_lower:
            dropped_specific += 1
            continue

        # Rule 7: Generic bolt grade/strength identification charts (ISO/DIN standards)
        if scode == "0A" and (
            (ctype == "figure" and ("bolt strength marking" in text_lower or "metric bolt class" in text_lower))
            or (ctype == "paragraph" and text_lower.startswith("bolt and nut strength identification"))
        ):
            dropped_specific += 1
            continue

        # Rule 8: Thread notation diagram (standard metric/inch notation — textbook knowledge)
        if scode == "0A" and ctype == "figure" and "thread notation" in text_lower:
            dropped_specific += 1
            continue

        # Rule 9: Tire size format diagram (industry-standard P-metric schema, not vehicle spec)
        if scode == "3E" and ctype == "figure" and "metric tire size format" in text_lower:
            dropped_specific += 1
            continue

        # Rule 10: Abbreviations glossary in 6E (LLM knows ECM, EGR, A/T, MAP, TPS, etc.)
        if scode == "6E" and text_lower.startswith("abbreviations used in this section"):
            dropped_specific += 1
            continue

        # Rule 11: Supplement front-matter / publication boilerplate
        if c.get("source_doc") == "supplement" and not scode:
            dropped_specific += 1
            continue

        filtered_chunks.append(c)

    micro_dropped = 0
    chunks = []
    for c in filtered_chunks:
        if _should_drop_micro_fragment(c):
            micro_dropped += 1
            continue
        chunks.append(c)

    dedup_dropped = 0
    deduped_chunks = []
    seen_texts: dict[tuple[str, str, str | int | None, str], str] = {}
    for c in chunks:
        key = _dedupe_key(c)
        if key is not None:
            if key in seen_texts:
                dedup_dropped += 1
                continue
            seen_texts[key] = c.get("chunk_id", "")
        deduped_chunks.append(c)

    chunks = deduped_chunks
    dropped = before - len(chunks)
    if dropped:
        print(
            "  Filtered "
            f"{dropped} noise/fragment chunks from search index "
            f"({dropped_specific} surgical, {micro_dropped} micro-fragment, {dedup_dropped} dedup)"
        )

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
        _metadata_text(chunk),
    ]
    if chunk.get("type") == "figure":
        parts.append("figure diagram illustration")
        parts.append(chunk.get("section_path", ""))  # double-weight section context
    combined = " ".join(p for p in parts if p).lower()
    
    # Bridge vocabulary gaps (1990 legacy vs 2026 modern terms)
    # Cigar -> Cigarette
    if "cigar" in combined and "cigarette" not in combined:
        combined += " cigarette"
    elif "cigarette" in combined and "cigar" not in combined:
        combined += " cigar"
    
    # OBD -> ALDL
    if "aldl" in combined and "obd" not in combined:
        combined += " obd diagnostic"
    elif "obd" in combined and "aldl" not in combined:
        combined += " aldl"
        
    # DTC -> trouble code
    if "trouble code" in combined and "dtc" not in combined:
        combined += " dtc"
    elif "dtc" in combined and "trouble code" not in combined:
        combined += " trouble code"
        
    # PCM -> ECM
    if "ecm" in combined and "pcm" not in combined:
        combined += " pcm"
    elif "pcm" in combined and "ecm" not in combined:
        combined += " ecm"

    # Simple whitespace tokenizer, lowercased, alphanumeric only
    return re.findall(r"[a-z0-9]+", combined)


def _embed_text(chunk: dict) -> str:
    """Text to embed: section_path prefix + chunk text.

    For figure chunks, prepend 'Figure/Diagram' and the section context
    to strengthen embeddings (bare captions like 'Figure 3' are too short).
    """
    prefix = chunk.get("section_path", "")
    text   = chunk.get("text", "")
    ctype  = chunk.get("type", "")
    metadata = _metadata_text(chunk)

    if ctype == "figure":
        # Enrich short figure text with section context for better embedding
        parts = [f"Figure/Diagram in {prefix}" if prefix else "Figure/Diagram"]
        if metadata:
            parts.append(metadata)
        if text:
            parts.append(text)
        out = " - ".join(parts)
    else:
        enriched_text = f"{metadata} - {text}" if metadata and text else (metadata or text)
        out = f"{prefix}: {enriched_text}" if prefix else enriched_text

    # Bridge vocabulary gaps in embeddings too
    low = out.lower()
    if "cigar" in low and "cigarette" not in low:
        out += " (cigarette)"
    elif "cigarette" in low and "cigar" not in low:
        out += " (cigar)"
        
    if "aldl" in low and "obd" not in low:
        out += " (obd diagnostic)"
    elif "obd" in low and "aldl" not in low:
        out += " (aldl)"
        
    if "trouble code" in low and "dtc" not in low:
        out += " (dtc)"
    elif "dtc" in low and "trouble code" not in low:
        out += " (trouble code)"
        
    if "ecm" in low and "pcm" not in low:
        out += " (pcm)"
    elif "pcm" in low and "ecm" not in low:
        out += " (ecm)"
    
    return out


class RetrievalIndex:
    """Unified BM25 + embedding retrieval with RRF merge."""

    RRF_K = 60          # Standard RRF constant
    SUPPLEMENT_BOOST = 0.05  # Added to RRF score for supplement chunks (source_doc="supplement")

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
