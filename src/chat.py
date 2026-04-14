"""Stage H: Chat Orchestration.

Implements the full RAG pipeline:
  1. Retrieve top-K chunks (BM25 + embeddings, RRF merge)
  2. Rerank by weighted score (type boost for procedures/warnings)
  3. Collect referenced figure images
  4. Build Claude API prompt with text evidence + figure images
  5. Call Claude Sonnet, parse response with citations
"""

import base64
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


import anthropic
from sentence_transformers import CrossEncoder
from src.gemini_api import call_gemini, build_gemini_assistant_parts, build_tool_result_parts

# Load .env file from project root if present (ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.index_build import RetrievalIndex
from src.models import ChatResponse, Citation
from src.utils import load_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOC_TYPES = {"toc_category", "toc_entry"}
FETCH_CHUNKS_MAX_DIRECT = 20
FETCH_CHUNKS_MAX_KEYWORDS = 8
FETCH_CHUNKS_TEXT_LIMIT = 300
FETCH_CHUNKS_MAX_TOOL_ITERATIONS = 10
FETCH_CHUNKS_MAX_LEGACY_REQUESTS = 8
FETCHING_STATUS_LABEL = "Checking the manual..."
SUPPLEMENT_MATCH_HEAD_TOKENS = 14
SUPPLEMENT_MATCH_TOKEN_LIMIT = 80
SUPPLEMENT_MATCH_MIN_SHARED = 8
SUPPLEMENT_MATCH_MIN_OVERLAP = 0.75


def _resolve_project_path(path_value: str | Path | None, fallback: str | Path) -> Path:
    raw = path_value if path_value not in (None, "") else fallback
    path = raw if isinstance(raw, Path) else Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _index_dir_from_config(config: dict | None) -> Path:
    if config:
        override = config.get("_index_dir")
        if override:
            return _resolve_project_path(override, "tools/rag_index")
        chat_cfg = config.get("chat", {})
        if chat_cfg.get("index_dir"):
            return _resolve_project_path(chat_cfg["index_dir"], "tools/rag_index")
    return _resolve_project_path(None, "tools/rag_index")


def _system_prompt_path_from_config(config: dict | None) -> Path:
    if config:
        override = config.get("_system_prompt_path")
        if override:
            return _resolve_project_path(override, "configs/chat_system_prompt.txt")
        chat_cfg = config.get("chat", {})
        if chat_cfg.get("system_prompt"):
            return _resolve_project_path(chat_cfg["system_prompt"], "configs/chat_system_prompt.txt")
    return _resolve_project_path(None, "configs/chat_system_prompt.txt")


def _chunk_lookup_path_from_config(config: dict | None) -> Path:
    if config:
        override = config.get("_chunk_lookup_path")
        if override:
            return _resolve_project_path(override, "tools/rag_index/chunk_lookup.json")
    return _index_dir_from_config(config) / "chunk_lookup.json"

FETCH_CHUNKS_TOOL = {
    "name": "fetch_chunks",
    "description": (
        "Fetch manual chunks when you need more context. "
        "Use section_code when you know the section from the table of contents. "
        "Use page when a citation or cross-reference points to a specific page. "
        "Use keywords as a fallback BM25 search when you do not have a specific section or page. "
        "Provide only one parameter per call. "
        "This tool is internal: never mention tool calls or emit placeholder tags like [CHUNK_REQUEST: ...] to the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "section_code": {
                "type": "string",
                "description": "Manual section code from the TOC, such as 6D1 or 1B.",
            },
            "page": {
                "type": "integer",
                "minimum": 1,
                "description": "Exact manual page number to fetch.",
            },
            "keywords": {
                "type": "string",
                "description": "Fallback keyword search when you do not know the section code or page.",
            },
        },
        "additionalProperties": False,
    },
}
SAVE_NOTE_TOOL = {
    "name": "save_note",
    "description": (
        "Save a note to the owner's notebook. Use this to record important findings, "
        "measurements, diagnostic results, specs, or anything the owner might need later. "
        "Notes persist across conversations. Use a clear, descriptive title. "
        "Tag notes by category (e.g. 'diagnostic', 'spec', 'measurement', 'finding', 'procedure'). "
        "This tool is internal: do not mention tool calls to the user, but DO tell them you saved a note."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short descriptive title for the note.",
            },
            "content": {
                "type": "string",
                "description": "The note content. Include specific values, specs, readings, or findings.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Category tags like 'diagnostic', 'spec', 'measurement', 'finding', 'procedure'.",
            },
        },
        "required": ["title", "content"],
        "additionalProperties": False,
    },
}
RETRIEVE_NOTES_TOOL = {
    "name": "retrieve_notes",
    "description": (
        "Retrieve saved notes from the owner's notebook. "
        "Use this when the owner asks about previous findings, what has already been ruled out, "
        "prior measurements, past plans, or saved specs. "
        "By default, notes come from the current project notebook when the chat belongs to a project; "
        "otherwise they come from the current chat notebook. "
        "You may optionally narrow by keywords, tags, or scope. "
        "This tool is internal: never mention tool calls to the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "Words or phrases to match in note titles, content, or tags.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to filter by, such as diagnostic, measurement, or plan.",
            },
            "scope": {
                "type": "string",
                "enum": ["auto", "project", "chat"],
                "description": "Use auto by default. Use project to search all notes in the current project or chat to search only this conversation's notes.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
                "description": "Maximum number of notes to return.",
            },
        },
        "additionalProperties": False,
    },
}

LEGACY_CHUNK_REQUEST_RE = re.compile(r"\[CHUNK_REQUEST:\s*([^\]]+)\]", re.IGNORECASE)
LOOKUP_INTENT_RE = re.compile(
    r"\b(?:let me|i(?:'|’)ll|i will)\s+(?:go get|get|pull|look up|check|fetch|grab)\b",
    re.IGNORECASE,
)


@dataclass
class AgentToolCall:
    id: str
    name: str
    input: dict[str, object] = field(default_factory=dict)


@dataclass
class AgentTurn:
    provider: str
    model: str
    stop_reason: str | None
    text: str
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    raw_content: Any = None


@dataclass
class ModelCapabilities:
    provider: str
    supports_tools: bool
    supports_images: bool = True
    supports_streaming: bool = False
    max_tool_iterations: int = FETCH_CHUNKS_MAX_TOOL_ITERATIONS


@dataclass
class ModelAdapter:
    provider: str
    model: str
    capabilities: ModelCapabilities
    run_turn: Callable[[dict, list[dict]], AgentTurn]
    assistant_message: Callable[[AgentTurn], dict]


def _format_page_span(page_span: tuple[int, int] | None) -> str:
    if not page_span:
        return "-"
    start, end = page_span
    if start == end:
        return f"p. {start}"
    return f"pp. {start}-{end}"


def _update_page_span(page_spans: dict[str, list[int]], section_code: str, page: int) -> None:
    bounds = page_spans.setdefault(section_code, [page, page])
    bounds[0] = min(bounds[0], page)
    bounds[1] = max(bounds[1], page)


def _section_root(section_path: str) -> str:
    return section_path.split(" > ", 1)[0].strip() if section_path else ""


def _chunk_source_doc(chunk: dict) -> str:
    return chunk.get("source_doc") or "main"


def _page_sort_value(page: object) -> int:
    return page if isinstance(page, int) else 10**9


def _page_label(page: object) -> str:
    return f"p.{page}" if isinstance(page, int) else "p.-"


def _supplement_match_tokens(text: str) -> list[str]:
    cleaned = (text or "").lower()
    cleaned = re.sub(r"figure\s+\d+[a-z0-9-]*", " ", cleaned)
    cleaned = re.sub(
        r"\b(remove or disconnect|install or connect|adjust|inspect|check|"
        r"caution|notice|note|specifications)\b",
        " ",
        cleaned,
    )
    return re.findall(r"[a-z0-9]+", cleaned)[:SUPPLEMENT_MATCH_TOKEN_LIMIT]


def _supplement_head_signature(chunk: dict) -> str:
    tokens = _supplement_match_tokens(chunk.get("text", ""))
    return " ".join(tokens[:SUPPLEMENT_MATCH_HEAD_TOKENS])


def _is_authoritative_duplicate(main_chunk: dict, supplement_chunk: dict) -> bool:
    if _chunk_source_doc(main_chunk) == "supplement":
        return False
    if _chunk_source_doc(supplement_chunk) != "supplement":
        return False

    main_section = (main_chunk.get("section_code") or "").upper()
    supplement_section = (supplement_chunk.get("section_code") or "").upper()
    if not main_section or main_section != supplement_section:
        return False
    if main_chunk.get("type") != supplement_chunk.get("type"):
        return False

    main_sig = _supplement_head_signature(main_chunk)
    supplement_sig = _supplement_head_signature(supplement_chunk)
    if main_sig and main_sig == supplement_sig:
        return True

    main_tokens = set(_supplement_match_tokens(main_chunk.get("text", "")))
    supplement_tokens = set(_supplement_match_tokens(supplement_chunk.get("text", "")))
    if not main_tokens or not supplement_tokens:
        return False

    shared = len(main_tokens & supplement_tokens)
    overlap = shared / min(len(main_tokens), len(supplement_tokens))
    return shared >= SUPPLEMENT_MATCH_MIN_SHARED and overlap >= SUPPLEMENT_MATCH_MIN_OVERLAP


def _prefer_supplement_direct_chunks(chunks: list[dict]) -> list[dict]:
    supplements = [chunk for chunk in chunks if _chunk_source_doc(chunk) == "supplement"]
    return supplements if supplements else chunks


def _apply_supplement_authority_to_chunk_list(chunks: list[dict]) -> list[dict]:
    supplements = [chunk for chunk in chunks if _chunk_source_doc(chunk) == "supplement"]
    if not supplements:
        return chunks

    filtered: list[dict] = []
    emitted_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        source_doc = _chunk_source_doc(chunk)
        if source_doc == "supplement":
            if chunk_id not in emitted_ids:
                filtered.append(chunk)
                emitted_ids.add(chunk_id)
            continue

        match = next((supp for supp in supplements if _is_authoritative_duplicate(chunk, supp)), None)
        if match:
            match_id = match.get("chunk_id")
            if match_id not in emitted_ids:
                filtered.append(match)
                emitted_ids.add(match_id)
            continue

        filtered.append(chunk)
        if chunk_id:
            emitted_ids.add(chunk_id)

    return filtered


def _apply_supplement_authority_to_results(results: list[dict], *, label: str) -> list[dict]:
    supplements = [row for row in results if _chunk_source_doc(row["chunk"]) == "supplement"]
    if not supplements:
        return results

    filtered: list[dict] = []
    emitted_ids: set[str] = set()
    dropped = 0

    for row in results:
        chunk = row["chunk"]
        chunk_id = chunk.get("chunk_id")
        source_doc = _chunk_source_doc(chunk)
        if source_doc == "supplement":
            if chunk_id not in emitted_ids:
                filtered.append(row)
                emitted_ids.add(chunk_id)
            continue

        match = next((supp for supp in supplements if _is_authoritative_duplicate(chunk, supp["chunk"])), None)
        if match:
            dropped += 1
            match_id = match["chunk"].get("chunk_id")
            if match_id not in emitted_ids:
                filtered.append(match)
                emitted_ids.add(match_id)
            continue

        filtered.append(row)
        if chunk_id:
            emitted_ids.add(chunk_id)

    if dropped:
        print(f"  [chat] Dropped {dropped} main chunk(s) in favor of supplement authority during {label}")

    return filtered


def _truncate_tool_text(text: str, max_chars: int = FETCH_CHUNKS_TEXT_LIMIT) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _serialize_tool_chunk(chunk: dict) -> dict[str, object]:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "section_code": chunk.get("section_code"),
        "section_path": chunk.get("section_path"),
        "page": chunk.get("page"),
        "type": chunk.get("type"),
        "text": _truncate_tool_text(chunk.get("text", "")),
    }


def format_fetch_chunks_result(result: dict) -> str:
    if result.get("error"):
        return f"fetch_chunks error: {result['error']}"

    lines = [
        "fetch_chunks results",
        f"mode: {result.get('mode', '-')}",
        f"returned: {len(result.get('chunks', []))}",
        f"total_available: {result.get('total_available', 0)}",
    ]
    if result.get("truncated"):
        lines.append("truncated: true")

    if result.get("section_code"):
        lines.append(f"section_code: {result['section_code']}")
    if result.get("page") is not None:
        lines.append(f"page: {result['page']}")
    if result.get("keywords"):
        lines.append(f"keywords: {result['keywords']}")

    for chunk in result.get("chunks", []):
        section_code = chunk.get("section_code") or "-"
        chunk_type = chunk.get("type") or "-"
        lines.append(f"[{section_code} | {_page_label(chunk.get('page'))} | {chunk_type}] {chunk.get('text', '')}")

    return "\n".join(lines)


def fetch_chunks(index: RetrievalIndex,
                 section_code: str | None = None,
                 page: int | None = None,
                 keywords: str | None = None) -> dict[str, object]:
    """Fetch compact chunk payloads for tool use."""
    normalized_section = (section_code or "").strip().upper() or None
    normalized_keywords = (keywords or "").strip() or None

    normalized_page = None
    if page is not None:
        try:
            normalized_page = int(page)
        except (TypeError, ValueError):
            return {
                "error": "page must be an integer",
                "mode": "page",
                "chunks": [],
                "truncated": False,
                "total_available": 0,
            }
        if normalized_page < 1:
            return {
                "error": "page must be >= 1",
                "mode": "page",
                "chunks": [],
                "truncated": False,
                "total_available": 0,
            }

    provided = [
        name for name, value in (
            ("section_code", normalized_section),
            ("page", normalized_page),
            ("keywords", normalized_keywords),
        )
        if value is not None
    ]
    if len(provided) != 1:
        return {
            "error": "Provide exactly one of section_code, page, or keywords",
            "mode": "invalid",
            "chunks": [],
            "truncated": False,
            "total_available": 0,
        }

    if normalized_section is not None:
        mode = "section_code"
        matches = [
            chunk for chunk in index.lookup.values()
            if chunk.get("type") not in TOC_TYPES
            and (chunk.get("section_code") or "").upper() == normalized_section
        ]
        matches = _prefer_supplement_direct_chunks(matches)
        matches.sort(key=lambda chunk: (_page_sort_value(chunk.get("page")), chunk.get("chunk_id") or ""))
        total_available = len(matches)
        limited = matches[:FETCH_CHUNKS_MAX_DIRECT]
    elif normalized_page is not None:
        mode = "page"
        matches = [
            chunk for chunk in index.lookup.values()
            if chunk.get("type") not in TOC_TYPES
            and chunk.get("page") == normalized_page
        ]
        matches = _prefer_supplement_direct_chunks(matches)
        matches.sort(key=lambda chunk: (chunk.get("section_code") or "", chunk.get("chunk_id") or ""))
        total_available = len(matches)
        limited = matches[:FETCH_CHUNKS_MAX_DIRECT]
    else:
        mode = "keywords"
        tokens = re.findall(r"[a-z0-9]+", normalized_keywords.lower())
        if not tokens:
            return {
                "error": "keywords must contain at least one alphanumeric token",
                "mode": mode,
                "chunks": [],
                "truncated": False,
                "total_available": 0,
                "keywords": normalized_keywords,
            }

        # BM25-only retrieval is intentional here: this tool is the model's
        # fallback keyword search when it lacks a concrete section or page.
        ranked = index._bm25_search(normalized_keywords, top_k=len(index.chunk_ids))
        matches = [
            index.lookup[cid]
            for cid, _score in ranked
            if cid in index.lookup and index.lookup[cid].get("type") not in TOC_TYPES
        ]
        matches = _apply_supplement_authority_to_chunk_list(matches)
        total_available = len(matches)
        limited = matches[:FETCH_CHUNKS_MAX_KEYWORDS]

    return {
        "mode": mode,
        "section_code": normalized_section,
        "page": normalized_page,
        "keywords": normalized_keywords,
        "truncated": total_available > len(limited),
        "total_available": total_available,
        "chunks": [_serialize_tool_chunk(chunk) for chunk in limited],
    }


@lru_cache(maxsize=8)
def _get_toc_chunks_cached(lookup_path_str: str) -> list[dict[str, str]]:
    lookup_path = Path(lookup_path_str)
    if not lookup_path.exists():
        return []
    lookup = load_json(lookup_path)
    if not isinstance(lookup, dict):
        return []

    page_spans: dict[str, list[int]] = {}
    categories: list[dict] = []
    entries: list[dict] = []

    for chunk in lookup.values():
        chunk_type = chunk.get("type")
        if chunk_type == "toc_category":
            categories.append(chunk)
            continue
        if chunk_type == "toc_entry":
            entries.append(chunk)
            continue

        # Keep page spans aligned to the main manual TOC.
        if chunk.get("source_doc") == "supplement":
            continue

        section_code = chunk.get("section_code")
        page = chunk.get("page")
        if section_code and isinstance(page, int):
            _update_page_span(page_spans, section_code, page)

    entry_pages = {
        section_code: (bounds[0], bounds[1])
        for section_code, bounds in page_spans.items()
    }
    entries_by_category: dict[str, list[dict]] = {}
    for entry in entries:
        entries_by_category.setdefault(_section_root(entry.get("section_path", "")), []).append(entry)

    rows: list[dict[str, str]] = []
    seen_entry_ids: set[str] = set()

    for category in categories:
        category_path = category.get("section_path") or category.get("source_label") or "-"
        category_entries = entries_by_category.get(category_path, [])

        category_span: tuple[int, int] | None = None
        for entry in category_entries:
            entry_span = entry_pages.get(entry.get("section_code", ""))
            if not entry_span:
                continue
            if category_span is None:
                category_span = entry_span
            else:
                category_span = (
                    min(category_span[0], entry_span[0]),
                    max(category_span[1], entry_span[1]),
                )

        rows.append({
            "section_code": "-",
            "section_path": category_path,
            "pages": _format_page_span(category_span),
        })

        for entry in category_entries:
            seen_entry_ids.add(entry.get("chunk_id", ""))
            rows.append({
                "section_code": entry.get("section_code") or "-",
                "section_path": entry.get("section_path") or entry.get("source_label") or "-",
                "pages": _format_page_span(entry_pages.get(entry.get("section_code", ""))),
            })

    for entry in entries:
        chunk_id = entry.get("chunk_id", "")
        if chunk_id in seen_entry_ids:
            continue
        rows.append({
            "section_code": entry.get("section_code") or "-",
            "section_path": entry.get("section_path") or entry.get("source_label") or "-",
            "pages": _format_page_span(entry_pages.get(entry.get("section_code", ""))),
        })

    return rows


def get_toc_chunks(config: dict | None = None) -> list[dict[str, str]]:
    return _get_toc_chunks_cached(str(_chunk_lookup_path_from_config(config)))


@lru_cache(maxsize=8)
def _get_toc_text_cached(lookup_path_str: str) -> str:
    rows = _get_toc_chunks_cached(lookup_path_str)
    if not rows:
        return ""

    lines = [
        "## MANUAL TABLE OF CONTENTS",
        "section_code | section_path | pages",
    ]
    for row in rows:
        lines.append(f"{row['section_code']} | {row['section_path']} | {row['pages']}")
    return "\n".join(lines)


def get_toc_text(config: dict | None = None) -> str:
    return _get_toc_text_cached(str(_chunk_lookup_path_from_config(config)))


# Type boosts for reranking — surface procedures and safety info first
TYPE_BOOST = {
    "procedure": 0.25,
    "warning":   0.20,
    "caution":   0.20,
    "figure":    0.15,
    "note":      0.10,
    "table":     0.10,
    "legend":    0.05,
    "paragraph": 0.00,
}

CITATION_RE = re.compile(
    r"\[p(\d+)\s*\|\s*([^\]]+?)\]"
)

# Strips text-citation brackets but keeps figure citations for frontend rendering
# Figure citations like [p5 | fig: fig_id] become clickable links in the UI
CITATION_STRIP_RE = re.compile(r"\[(?![^\[\]]*\bfig:)[^\[\]]*\|[^\[\]]*\]")
CITATION_PREFIX_RE = re.compile(r"^(?:chunk|chunk_id|source(?:_label)?|citation|id)\s*:\s*", re.IGNORECASE)

CONNECTOR_QUERY_RE = re.compile(r"\bconnector\b", re.IGNORECASE)
PINOUT_QUERY_RE = re.compile(
    r"\b(?:pinout|connector identification|pin assignments?|terminals?|cavities)\b",
    re.IGNORECASE,
)
CONNECTOR_FACE_QUERY_RE = re.compile(
    r"\b(?:connector face|connector faces|harness connector faces|terminal end view|end view)\b",
    re.IGNORECASE,
)
CORRESPONDENCE_QUERY_RE = re.compile(
    r"\b(?:which|what|correspond(?:s|ing)?|mapping|match(?:es|ing)?|cross[- ]?reference|equivalent)\b",
    re.IGNORECASE,
)
ECM_QUERY_RE = re.compile(r"\b(?:ecm|engine control module)\b", re.IGNORECASE)

SYSTEM_KEYWORDS = {
    "ac": ["ac", "air conditioning", "heater", "refrigerant", "compressor", "condenser", "evaporator", "r-134a", "r-12"],
    "cooling": ["coolant", "radiator", "thermostat", "water pump", "cooling system", "fan belt", "overflow"],
    "steering": ["steering", "tie rod", "rack", "power steering", "steering wheel", "steering column"],
    "front_suspension": ["front suspension", "strut", "ball joint", "stabilizer", "sway bar", "control arm"],
    "rear_suspension": ["rear suspension", "rear axle", "shock absorber", "leaf spring"],
    "wheels_tires": ["wheel", "tire", "tyre", "lug nut", "rotation", "balance"],
    "brakes": ["brakes", "brake", "brake pad", "brake shoe", "rotor", "caliper", "drum brake", "master cylinder", "bleeding", "parking brake", "brake fluid", "brake line", "brake pedal", "disc brake"],
    "front_drive_axle": ["drive axle", "driveshaft", "cv joint", "axle shaft", "drive shaft", "constant velocity"],
    "engine": ["engine", "motor", "cylinder", "spark", "valves", "g10", "g13", "piston", "crank", "timing belt", "head gasket", "camshaft", "oil pump", "oil pan", "oil seal", "oil pressure"],
    "engine_electrical": ["ignition", "distributor", "coil", "starting", "starter", "charging", "alternator", "timing"],
    "battery": ["battery", "jump start", "jumper cable", "terminal", "jump starting"],
    "emission_controls": ["emission", "egr", "pcv", "catalytic converter", "evap", "canister"],
    "fuel_system": ["fuel", "gas", "tank", "fuel pump", "fuel filter", "fuel line", "carburetor"],
    "fuel_injection": ["fuel injection", "injector", "tbi", "efi", "throttle body", "idle", "oxygen sensor", "check engine"],
    "exhaust": ["exhaust", "muffler", "exhaust pipe", "tailpipe", "exhaust manifold"],
    "automatic_transaxle": ["transmission", "automatic", "transaxle", "shifting", "gear", "fluid", "torque converter"],
    "manual_transaxle": ["manual transmission", "shift linkage", "manual transaxle"],
    "clutch": ["clutch", "clutch disc", "clutch plate", "pressure plate", "release bearing", "throw-out bearing"],
    "electrical": [
        "wiring", "fuse", "relay", "connector", "headlight", "tail light", "horn", "wiper",
        "turn signal", "gauge", "instrument", "cluster", "speedometer", "tachometer", "warning light",
        "cigar lighter", "cigarette lighter", "lighter socket", "power outlet", "12v socket",
        "accessory socket", "usb charger socket",
    ],
    "body": ["body", "door", "window", "seat", "trim", "bumper", "hood", "trunk", "hatch", "weatherstrip", "glass", "underbody", "safety belt"],
    "accessories": ["radio", "antenna", "speaker", "clock"],
    "maintenance": ["maintenance", "schedule", "oil change", "fluid change", "tune-up", "service interval", "engine oil", "oil capacity", "oil type", "oil grade", "oil viscosity", "oil specification", "oil filter", "oil", "chassis", "lubricate", "lubrication", "lube", "grease"],
}

def _detect_system(query: str) -> str | None:
    """Detect the target system from the user query.

    Uses a scoring approach: each system's score is the total length of all
    matching keywords. Longer/more-specific keyword matches beat short ones.
    e.g. "engine oil" (10 chars) in maintenance beats "oil" (3 chars) in engine.
    Short keywords (<=4 chars) require word boundaries to prevent false substring
    matches (e.g. "efi" inside "refrigerant").
    """
    q = query.lower()
    best_sys = None
    best_score = 0
    for sys, keywords in SYSTEM_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if len(kw) <= 4:
                # Short keywords need word boundaries
                if re.search(r"\b" + re.escape(kw) + r"\b", q):
                    score += len(kw)
            else:
                if kw in q:
                    score += len(kw)
        if score > best_score:
            best_score = score
            best_sys = sys
    return best_sys


# ── Section Router ─────────────────────────────────────────────────────────
# Deterministic query-to-section-code router built from chunk corpus profiling.
# Returns ranked section candidates so evidence validation can check retrieval.

@dataclass
class SectionRoute:
    primary_sections: list[str]
    secondary_sections: list[str]
    confidence: str  # "high", "medium", "low"
    matched_terms: list[str]

# Each entry: (regex_pattern, primary_sections, secondary_sections)
# Patterns are tested in order; first high-confidence match wins.
# Terms derived from Group 1/2/3 section profiling.
SECTION_ROUTES: list[tuple[str, list[str], list[str]]] = [
    # ── Group 1: Engine / Fuel / Electrical / Driveability ──

    # 6E2 — Driveability / Emissions Detailed (the "gravity well")
    # Strong signal: ECM, TBI, sensor-voltage combos, trouble codes, diagnostic codes
    (r"\b(?:ecm|electronic control module|engine control module)\b", ["6E2"], ["6E", "6D5", "8A"]),
    (r"\btbi\b|\bthrottle body injection\b", ["6E2"], ["6C"]),
    (r"^(?!.*\b(?:sir|airbag|derm|inflat\w*|restraint)\b).*\b(?:trouble code|diagnostic code)\b", ["6E2"], ["6E"]),
    (r"\b(?:check engine light|mil light)\b", ["6E2"], ["6E"]),
    (r"\bduty cycle\b", ["6E2"], ["6E"]),
    (r"\b(?:cts|coolant temperature sensor)\b.*\b(?:voltage|signal|ecm|ohm|resistance)\b", ["6E2"], ["6E", "6B"]),
    (r"\b(?:voltage|signal|ecm|ohm|resistance)\b.*\b(?:cts|coolant temperature sensor)\b", ["6E2"], ["6E", "6B"]),
    (r"\b(?:tps|throttle position sensor)\b", ["6E2"], ["6E"]),
    (r"\b(?:map sensor|manifold absolute pressure)\b", ["6E2"], ["6E"]),
    (r"\b(?:o2 sensor|oxygen sensor)\b", ["6E2"], ["6E", "6F"]),
    (r"\b(?:iac|idle air control)\b", ["6E2"], ["6E"]),
    (r"\binjector\b.*\b(?:voltage|signal|pulse|resistance)\b", ["6E2"], ["6C"]),
    (r"\b(?:voltage|signal|pulse|resistance)\b.*\binjector\b", ["6E2"], ["6C"]),
    (r"\bsensor\b.*\b(?:voltage|signal|specification|spec|range|output)\b", ["6E2"], ["6E", "8A"]),
    (r"\b(?:voltage|signal|specification|spec|range|output)\b.*\bsensor\b", ["6E2"], ["6E", "8A"]),
    (r"\b(?:key on|eng run|key on eng)\b", ["6E2"], ["6E"]),

    # 6B — Engine Cooling
    (r"\b(?:radiator|thermostat|water pump|coolant overflow|overheating|thermo switch|radiator cap)\b", ["6B"], ["6E2"]),
    (r"\bcoolant\b(?!.*\b(?:sensor|cts|voltage|signal|ecm)\b)", ["6B"], ["6E2"]),

    # 6A1 — Engine Mechanical
    (r"\b(?:camshaft|crankshaft|piston|connecting rod|bearing cap|cylinder head|head gasket|valve lash|timing belt|oil pump)\b", ["6A1"], ["6E2"]),
    (r"\bcompression test\b|\bleak down\b", ["6A1"], ["6E2"]),

    # 6D4 — Ignition
    (r"\b(?:distributor|spark plug|ignition coil|pickup coil|distributor cap|ignitor)\b", ["6D4"], ["6E2", "6D"]),
    (r"\btiming advance\b", ["6D4"], ["6E2"]),

    # 6D1 — Battery
    (r"\bbattery\b(?!.*\b(?:sir|airbag|restraint)\b)", ["6D1"], ["6D2", "6D3"]),

    # 6D2 — Cranking System
    (r"\b(?:starter motor|starter)\b", ["6D2"], ["6D1"]),

    # 6D3 — Charging System
    (r"\balternator\b|\bcharging system\b", ["6D3"], ["6D1"]),

    # 6D — Engine Electrical (general)
    (r"\bengine electrical\b|\bengine wiring\b", ["6D"], ["6D1", "6D2", "6D3", "6D4", "6D5"]),

    # 6D5 — Engine Wiring
    (r"\bengine wiring harness\b|\bengine harness\b", ["6D5"], ["8A", "6D"]),

    # 6 — Engine general (diagnosis tables, general engine)
    (r"\bengine\b.*\b(?:diagnos|symptom|won'?t start|no start|stall|misfire|rough idle|hard start|cranks?)\b", ["6"], ["6E2", "6E"]),
    (r"\b(?:won'?t start|no start|hard start)\b.*\bengine\b", ["6"], ["6E2", "6E"]),

    # 6C — Fuel System (physical)
    (r"\b(?:fuel tank|fuel filter|fuel line|fuel pump)\b", ["6C"], ["6E2"]),

    # 6E — Driveability (broader)
    (r"\b(?:egr|exhaust gas recirculation|pcv valve|emission)\b", ["6E"], ["6E2", "6F"]),

    # 6F — Exhaust
    (r"\b(?:muffler|exhaust pipe|tailpipe|catalytic converter|exhaust manifold)\b", ["6F"], ["6E"]),

    # 8 — Body Electrical Systems (general)
    (r"\b(?:headlight|tail light|turn signal|gauge|instrument cluster|speedometer|tachometer|warning light)\b", ["8"], ["8A"]),

    # 8A — Electrical (body wiring/fuses)
    (r"\b(?:junction block|fuse block|chassis ground|daytime running|fuse panel)\b", ["8A"], ["8", "8B"]),
    (r"\b(?:wiring harness|wiring diagram)\b", ["8A"], ["8B", "6D5"]),

    # 8B — Electrical Wiring
    (r"\b(?:wire color|connector pin|terminal pin|wiring schematic)\b", ["8B"], ["8A"]),

    # ── Group 2: Drivetrain / Suspension / Brakes / Steering ──

    # 3 — Steering/Suspension Diagnosis (general)
    (r"\b(?:steering|suspension)\b.*\b(?:diagnos|noise|vibrat|wander|pull)\b", ["3"], ["3C", "3D", "3B"]),
    (r"\b(?:diagnos|noise|vibrat|wander|pull)\b.*\b(?:steering|suspension)\b", ["3"], ["3C", "3D", "3B"]),

    # 3A — Wheel Alignment
    (r"\b(?:wheel alignment|camber|caster|toe-in|toe in|toe.?out|alignment spec)\b", ["3A"], ["3C", "3D"]),

    # 3B — Manual Rack and Pinion Steering
    (r"\b(?:rack and pinion|tie rod|tie-rod|steering rack|pinion bearing|rack bushing|steering gear|gear case)\b", ["3B"], ["3F", "3C"]),
    (r"\b(?:rack)\b.*\b(?:steering|pinion|bushing)\b", ["3B"], ["3F"]),

    # 3C — Front Suspension
    (r"\bfront\s+(?:suspension|struts?|knuckle|hub|control arm)\b", ["3C"], ["3D", "3"]),
    (r"\bfront\b.*\b(?:struts?|knuckle|ball joint|ball stud|stabilizer|sway bar|control arm|wheel hub|strut bracket|damper)\b", ["3C"], ["3D", "3"]),
    (r"\b(?:strut bracket|strut assembly|front spring)\b", ["3C"], ["3D"]),

    # 3D — Rear Suspension
    (r"\brear\s+(?:suspension|struts?|shocks?|axle|mount)\b", ["3D"], ["3C", "3"]),
    (r"\brear\b.*\b(?:struts?|shock absorbers?|leaf spring|control rod|lower mount)\b", ["3D"], ["3C"]),
    (r"\b(?:control rod|rear strut)\b", ["3D"], ["3C"]),

    # 3E — Wheels and Tires
    (r"\b(?:tire pressure|tire rotation|wheel balance|lug nut|rim|tread|runout|bead|compact spare)\b", ["3E"], []),
    (r"\b(?:tire|tyre)\b.*\b(?:psi|kpa|pressure|balance|rotation)\b", ["3E"], []),

    # 3F — Steering Column (non-SIR)
    (r"^(?!.*\b(?:sir|airbag|inflat\w*|restraint|derm)\b).*\b(?:steering column|steering wheel|steering shaft)\b", ["3F"], ["3F4"]),
    (r"\b(?:horn|dimmer switch|turn signal.*switch|ignition switch|shaft joint)\b", ["3F"], ["3F4"]),

    # 3F4 — SIR / Column Airbag Hardware
    (r"\b(?:steering column|steering wheel)\b.*\b(?:sir|airbag|inflat|restraint|derm)\b", ["3F4"], ["3F", "9J"]),
    (r"\b(?:sir|airbag|inflat|restraint|derm)\b.*\b(?:steering column|steering wheel)\b", ["3F4"], ["3F", "9J"]),
    (r"\b(?:sir coil|inflator module)\b", ["3F4"], ["9J"]),

    # 4D — Drive Axle (bridge section)
    (r"\b(?:cv joint|drive axle|axle shaft|axle boot|constant velocity|tripod joint|joint grease|boot band)\b", ["4D"], ["7B", "7A", "3C"]),
    (r"\b(?:differential.side|wheel.side)\b.*\bjoint\b", ["4D"], ["7B"]),

    # 5 — Brakes (full system)
    (r"\b(?:master cylinder|brake booster|brake shoe|drum brake|disc brake|parking brake|brake pedal|wheel cylinder|brake fluid|reservoir)\b", ["5"], ["5B"]),
    (r"\bbleed.*brake|brake.*bleed\b", ["5"], ["5B"]),

    # 5B — Hydraulic Brake / Caliper Service
    (r"\b(?:caliper|brake pad|brake piston|hydraulic brake|caliper pin|caliper housing|caliper mounting|carrier|flexible brake hose)\b", ["5B"], ["5"]),
    (r"\b(?:pin bolts?)\b.*\bcaliper\b", ["5B"], ["5"]),

    # 7A — Automatic Transaxle Service
    (r"\b(?:automatic trans\w*|auto trans\w*|automatic transaxle)\b|\btorque converter\b", ["7A"], ["7A1"]),
    (r"\b(?:shift solenoid|trans pan|selector lever|select cable|kickdown|neutral safety|shift lock|interlock cable|pressure control cable|transaxle fluid)\b", ["7A"], ["7A1"]),

    # 7A1 — Automatic Transaxle Unit Repair / Internals
    (r"\b(?:planetary|governor|valve body|servo|accumulator|direct clutch|forward clutch|one.?way clutch|lock pawl|line pressure|input shaft|output shaft|rear planetary)\b", ["7A1"], ["7A"]),

    # 7B — Manual Transaxle
    (r"\b(?:manual trans|manual transaxle|synchronizer|shift fork|countershaft|filler level|reverse gear|shift shaft|control lever)\b", ["7B"], ["7C", "4D"]),

    # 7C — Clutch
    (r"\b(?:clutch disc|clutch plate|pressure plate|release bearing|throw.?out bearing|flywheel|clutch pedal|clutch cable|pilot bushing)\b", ["7C"], ["7B"]),
    (r"\bclutch\b(?!.*\b(?:compressor|a/?c|magnetic|one.?way|direct|forward)\b)", ["7C"], ["7B"]),

    # 2 — Body general (section 2 content, structural/paint)
    (r"\b(?:body repair|body panel|paint|underbody coat|undercoat|rustproof|corrosion)\b", ["2"], ["10"]),

    # 9A — Accessories (radio etc.)
    (r"\b(?:radio|antenna|speaker|tape|preset|bass|treble|volume|clock)\b", ["9A"], []),

    # ── Group 3: General / Maintenance / HVAC / Body / SIR ──

    # 9J — SIR / Airbag Diagnostics (the "stealth" section)
    (r"\b(?:derm|arming sensor|discriminating sensor|deployment)\b", ["9J"], ["3F4"]),
    (r"\b(?:sir|airbag|inflatable restraint|supplemental restraint)\b.*\b(?:diagnos|code|resistance|voltage|terminal|connector|module|circuit|sensor)\b", ["9J"], ["3F4"]),
    (r"\b(?:diagnos|code|resistance|voltage|terminal|connector|module|circuit|sensor)\b.*\b(?:sir|airbag|inflatable restraint|supplemental restraint)\b", ["9J"], ["3F4"]),

    # 1A — Heating / Ventilation
    (r"\b(?:blower|heater core|heater|defroster|defrost)\b", ["1A"], ["1B"]),

    # 1B — Air Conditioning
    (r"\b(?:refrigerant|evaporator|condenser|receiver dryer|expansion valve|r-134a|r-12|freon)\b", ["1B"], ["1D"]),
    (r"\b(?:a/?c|air conditioning)\b(?!.*\bcompressor\b.*\b(?:seal|plate|reed|overhaul|rebuild)\b)", ["1B"], ["1D", "1A"]),

    # 1D — AC Compressor Overhaul
    (r"\bcompressor\b.*\b(?:seal|plate|reed|overhaul|rebuild|shaft seal|valve plate)\b", ["1D"], ["1B"]),
    (r"\b(?:seal|plate|reed|overhaul|rebuild|shaft seal|valve plate)\b.*\bcompressor\b", ["1D"], ["1B"]),
    (r"\bmagnetic clutch\b", ["1D"], ["1B"]),

    # 10 — Body Service
    (r"\b(?:door|window|windshield|trim panel|molding|weatherstrip|safety belt|seat belt|deck lid|quarter panel|latch|bumper|fender|hood|trunk|hatch)\b", ["10"], []),

    # 0B — Maintenance
    (r"\b(?:maintenance schedule|service interval|oil change|fluid change|tune-up)\b", ["0B"], []),
    (r"\b(?:miles|months|whichever occurs)\b.*\b(?:schedule|maintenance|service)\b", ["0B"], []),

    # 0A — General Information
    (r"\b(?:fastener|torque specification|vin|vehicle identification)\b", ["0A"], []),
]


def _route_to_sections(query: str) -> SectionRoute:
    """Route a query to candidate section codes using deterministic keyword matching.

    Returns a SectionRoute with primary/secondary section candidates,
    confidence level, and the terms that triggered the match.
    """
    q = query.lower()
    all_primary: list[str] = []
    all_secondary: list[str] = []
    all_matched: list[str] = []

    for pattern, primary, secondary in SECTION_ROUTES:
        m = re.search(pattern, q)
        if m:
            matched_text = m.group(0)
            for s in primary:
                if s not in all_primary:
                    all_primary.append(s)
            for s in secondary:
                if s not in all_secondary and s not in all_primary:
                    all_secondary.append(s)
            all_matched.append(matched_text)

    # Secondary candidates should never duplicate sections that ended up primary.
    all_secondary = [s for s in all_secondary if s not in all_primary]

    if not all_primary:
        return SectionRoute([], [], "low", [])

    # Confidence: high if we have specific section hits, medium if only broad
    if len(all_matched) >= 2:
        confidence = "high"
    elif len(all_primary) == 1:
        confidence = "high"
    else:
        confidence = "medium"

    return SectionRoute(all_primary, all_secondary, confidence, all_matched)


def _validate_and_correct_evidence(
    query: str,
    reranked: list[dict],
    route: SectionRoute,
    index: "RetrievalIndex",
    top_k: int,
    top_n: int,
    engine_variant: str,
) -> list[dict]:
    """Check if reranked evidence matches the router's section candidates.

    If the top evidence is a section mismatch, do a forced section-scoped
    retrieval pass and merge/replace evidence before the LLM sees it.
    """
    if route.confidence == "low" or not route.primary_sections:
        return reranked

    # Check top 3 evidence chunks for section match
    allowed_sections = set(route.primary_sections + route.secondary_sections)
    top_n_check = min(3, len(reranked))
    hits = 0
    for chunk in reranked[:top_n_check]:
        code = (chunk.get("section_code") or "").upper()
        # Match on prefix too (e.g. "6E2" matches allowed "6E2")
        if code in allowed_sections:
            hits += 1
        else:
            # Check prefix match: "6E2" should match if "6E" is allowed
            for allowed in allowed_sections:
                if code.startswith(allowed) or allowed.startswith(code):
                    hits += 1
                    break

    # If at least 1 of top 3 is from an expected section, evidence is OK
    if hits > 0:
        return reranked

    # Evidence mismatch — force section-scoped retrieval
    print(f"  [router] Evidence mismatch detected! Top evidence sections don't match route.")
    print(f"  [router] Expected: {route.primary_sections} + {route.secondary_sections}")
    top_codes = [(c.get("section_code") or "?") for c in reranked[:5]]
    print(f"  [router] Got: {top_codes}")
    print(f"  [router] Matched terms: {route.matched_terms}")

    # Retrieve directly from primary sections
    corrected_chunks = []
    for section_code in route.primary_sections:
        section_results = index.retrieve(
            query,
            top_k=max(8, top_k // 2),
            engine_variant=engine_variant,
        )
        # Filter to target section
        for r in section_results:
            chunk_code = (r["chunk"].get("section_code") or "").upper()
            if chunk_code == section_code.upper() or chunk_code.startswith(section_code.upper()):
                corrected_chunks.append(r["chunk"])

    if not corrected_chunks:
        # Fallback: try fetch_chunks style direct section lookup
        for section_code in route.primary_sections:
            for chunk_data in index.chunks:
                chunk_code = (chunk_data.get("section_code") or "").upper()
                if chunk_code == section_code.upper():
                    corrected_chunks.append(chunk_data)

    if corrected_chunks:
        print(f"  [router] Corrected: found {len(corrected_chunks)} chunks from target sections")
        # Neural rerank the corrected chunks
        ce = _get_cross_encoder()
        pairs = [(query, c.get("text", "")) for c in corrected_chunks]
        if pairs:
            scores = ce.predict(pairs)
            scored = list(zip(corrected_chunks, scores))
            scored.sort(key=lambda x: -x[1])
            corrected_chunks = [c for c, s in scored[:top_n]]

        # Merge: corrected chunks first, then original evidence (deduped)
        seen_ids = {c.get("chunk_id") for c in corrected_chunks}
        merged = list(corrected_chunks)
        for chunk in reranked:
            cid = chunk.get("chunk_id")
            if cid not in seen_ids:
                merged.append(chunk)
                seen_ids.add(cid)
        return merged[:top_n + 4]  # Allow slightly more evidence for correction
    else:
        print(f"  [router] Warning: no chunks found for target sections, keeping original evidence")
        return reranked


QUERY_EXPANSIONS = {
    r"\bvin\b": "vehicle identification number VIN",
    r"\becm\b": "electronic control module ECM",
    r"\btbi\b": "throttle body injection TBI",
    r"\bpcv\b": "positive crankcase ventilation PCV",
    r"\begr\b": "exhaust gas recirculation EGR",
    r"\ba/?c\b": "air conditioning AC",
    r"\brpm\b": "revolutions per minute RPM",
    r"\bvss\b": "vehicle speed sensor VSS",
    r"\bmap\b": "manifold absolute pressure MAP",
    r"\btps\b": "throttle position sensor TPS",
    r"\biat\b": "intake air temperature IAT",
    r"\bcts\b": "coolant temperature sensor CTS",
    r"\bcv\b": "constant velocity CV",
}

# Supplemental search terms for common maintenance queries.
# These add extra retrieval terms to pull in specs/capacities that BM25
# wouldn't otherwise find (e.g., garbled table headers, cross-section torques).
MAINTENANCE_QUERY_SUPPLEMENTS = {
    r"oil\s+change|change.*oil|engine\s+oil": "engine oil filter change engine crankcase capacity viscosity API SG Energy Conserving quarts maintenance schedule",
    r"coolant|antifreeze|flush.*cool": "cooling system capacity drain refill quarts liters",
    r"brake\s+fluid|bleed.*brake": "brake fluid capacity DOT reservoir master cylinder",
    r"transaxle\s+fluid|transmission\s+fluid|trans\s+fluid": "transaxle drain refill capacity quarts",
    r"spark\s+plug": "spark plug gap torque type NGK AC",
    r"tire\s+pressure|tire\s+psi": "tire pressure PSI kPa compact spare",
    r"chassis\s+lubri|lubricat.*chassis|grease.*chassis|chassis.*grease": "chassis grease specification GM 6031M underbody shift linkage parking brake cable guides contact points product lube points fitting",
}

# Component synonym/alias mapping — appends related terms so BM25
# can bridge vocabulary gaps ("belt" → manual says "drive belt", etc.)
COMPONENT_ALIASES = {
    r"\bbelt\b": "drive belt V-belt fan belt",
    r"\bgas\b": "fuel gasoline",
    r"\b(?:cigarette|cigar)\s*lighter\b": "cigarette lighter cigar lighter accessory power outlet 12V socket",
    r"\busb\s+charger\s+socket\b": "cigar lighter power outlet socket 12V accessory outlet",
    r"\bpower\s+outlet\b": "cigar lighter cigarette lighter accessory socket 12V outlet",
    r"\bstarter\b": "starter motor cranking",
    r"\balternator\b": "alternator generator charging",
    r"\bthermostat\b": "thermostat coolant temperature",
    r"\bdistributor\b": "distributor ignition cap rotor",
    r"\bpcv\s+valve\b": "PCV valve crankcase ventilation",
    r"\begr\s+valve\b": "EGR valve exhaust gas recirculation",
    r"(?<!pcv )(?<!egr )\bvalves?\b": "valve intake exhaust",
    r"\bhead\s*gasket\b": "cylinder head gasket",
    r"\bwater\s*pump\b": "water pump coolant pump",
    r"\btiming\s*belt\b": "timing belt cam belt",
    r"\bcv\s*joint\b": "constant velocity joint drive axle boot",
    r"\bmaster\s*cylinder\b": "master cylinder brake hydraulic",
    r"\bthrow.?out\s*bearing\b": "release bearing throw-out bearing clutch",
    r"\bplugs?\b": "spark plug ignition",
    r"\bidle\b": "idle speed fast idle idle air",
    r"\boverheating\b": "overheating coolant temperature radiator thermostat",
    r"\bno\s*start\b": "no start cranking ignition fuel",
    r"\bstalling\b": "stalling idle rough idle engine dies",
    r"\bcarb\b": "carburetor throttle body",
    r"\btranny\b": "transaxle transmission",
    r"\bheater\s*core\b": "heater core heating coolant",
    r"\bpower\s*steering\b": "power steering rack pinion",
    r"\bclutch\b": "clutch disc pressure plate release bearing",
    r"\baxle\b": "drive axle driveshaft CV joint",
    r"\bshock\s+absorbers?|\bshocks\b": "strut damper suspension",
}

DEEP_RESEARCH_HINT_RE = re.compile(
    r"\b(?:all|every|compare|full|complete|deep(?:\s+research|\s+dive)?|"
    r"signal characteristics|voltage ranges?|under various operating conditions|"
    r"pinout|connector pinout|across sections?|detailed)\b",
    re.IGNORECASE,
)

DEEP_RESEARCH_TOPIC_PATTERNS: list[tuple[str, str, str]] = [
    ("CTS", r"\b(?:cts|coolant temperature sensor)\b", "coolant temperature sensor CTS"),
    ("TPS", r"\b(?:tps|throttle position sensor)\b", "throttle position sensor TPS"),
    ("MAP", r"\b(?:map sensor|manifold absolute pressure|(?<!road )\bmap\b)\b", "manifold absolute pressure MAP sensor"),
    ("O2 Sensor", r"\b(?:o2 sensor|oxygen sensor)\b", "oxygen sensor O2"),
    ("IAC", r"\b(?:iac|idle air control)\b", "idle air control IAC"),
    ("Injector", r"\b(?:injector|fuel injector)\b", "fuel injector pulse resistance voltage"),
    ("EGR", r"\b(?:egr|exhaust gas recirculation)\b", "exhaust gas recirculation EGR"),
    ("PCV", r"\b(?:pcv|positive crankcase ventilation)\b", "positive crankcase ventilation PCV valve"),
    ("Distributor", r"\bdistributor\b", "distributor ignition timing pickup coil"),
    ("Ignition Coil", r"\bignition coil\b", "ignition coil voltage resistance"),
    ("Alternator", r"\balternator\b", "alternator charging system output voltage"),
    ("Starter", r"\b(?:starter|starter motor)\b", "starter motor cranking system"),
    ("Radiator", r"\bradiator\b", "radiator cooling system"),
    ("Thermostat", r"\bthermostat\b", "thermostat coolant temperature"),
    ("Water Pump", r"\bwater pump\b", "water pump cooling system"),
    ("Brake Booster", r"\bbrake booster\b", "brake booster vacuum"),
    ("Master Cylinder", r"\bmaster cylinder\b", "master cylinder brake hydraulic"),
    ("Caliper", r"\bcaliper\b", "brake caliper piston"),
    ("Strut", r"\bstrut\b", "suspension strut"),
    ("Tie Rod", r"\btie[\s-]?rod\b", "tie rod steering rack"),
    ("Steering Column", r"\bsteering column\b", "steering column shaft switch"),
    ("ALDL Connector", r"\b(?:aldl|diagnostic connector|monitor coupler)\b", "ALDL connector diagnostic terminals"),
    ("DERM", r"\bderm\b", "DERM diagnostic energy reserve module"),
    ("SIR", r"\b(?:sir|airbag|inflatable restraint|supplemental restraint)\b", "SIR airbag restraint diagnostics"),
]

DEEP_RESEARCH_FOCUS_TERMS: list[tuple[str, str]] = [
    (r"\bvoltage\b", "voltage"),
    (r"\bsignal\b", "signal"),
    (r"\bresistance\b|\bohms?\b", "resistance"),
    (r"\bpinout\b|\bterminals?\b|\bconnector\b", "pinout connector terminals"),
    (r"\bdiagnos(?:is|tic)\b|\bcode\b", "diagnostic code test"),
    (r"\boperating conditions?\b|\bvarious conditions?\b", "operating conditions"),
    (r"\bspec(?:ification)?s?\b|\brange\b", "specification range"),
]


@dataclass
class DeepResearchQuery:
    title: str
    query: str
    target_sections: list[str] = field(default_factory=list)


@dataclass
class DeepResearchPlan:
    forced: bool
    auto_detected: bool
    summary: str
    sub_queries: list[DeepResearchQuery] = field(default_factory=list)


def _is_engine_oil_service_query(query: str) -> bool:
    q_lower = (query or "").lower()
    return (
        any(
            k in q_lower
            for k in [
                "oil change",
                "change the oil",
                "change my oil",
                "change cheri's oil",
                "engine oil",
            ]
        )
        and not any(
            k in q_lower
            for k in [
                "transaxle",
                "transmission",
                "gear oil",
                "manual transaxle",
                "automatic transaxle",
                "atf",
            ]
        )
    )


def _expand_query(query: str) -> str:
    """Expand abbreviations, add component synonyms, and maintenance supplements."""
    expanded = query
    for pattern, replacement in QUERY_EXPANSIONS.items():
        if re.search(pattern, expanded, re.IGNORECASE):
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)

    # Append component synonym terms (don't replace — add alongside)
    for pattern, aliases in COMPONENT_ALIASES.items():
        if re.search(pattern, expanded, re.IGNORECASE):
            expanded = expanded + " " + aliases

    # Append supplemental search terms for common maintenance queries
    for pattern, supplement in MAINTENANCE_QUERY_SUPPLEMENTS.items():
        if re.search(pattern, expanded, re.IGNORECASE):
            expanded = expanded + " " + supplement
            break  # Only one supplement per query

    return expanded


def _route_for_query(search_query: str, raw_query: str) -> SectionRoute:
    route = _route_to_sections(search_query)
    if route.primary_sections:
        return route
    return _route_to_sections(raw_query)


def _section_matches_allowed(section_code: str | None, allowed_sections: list[str]) -> bool:
    code = (section_code or "").upper()
    if not code:
        return False
    for allowed in allowed_sections:
        allowed_upper = allowed.upper()
        if code == allowed_upper or code.startswith(allowed_upper) or allowed_upper.startswith(code):
            return True
    return False


def _filter_results_to_sections(results: list[dict], allowed_sections: list[str]) -> list[dict]:
    if not allowed_sections:
        return results
    filtered = [
        row for row in results
        if _section_matches_allowed(row["chunk"].get("section_code"), allowed_sections)
    ]
    return filtered if filtered else results


def _chunk_has_any_info_type(chunk: dict, *info_types: str) -> bool:
    chunk_types = set(chunk.get("info_types") or [])
    return any(info_type in chunk_types for info_type in info_types)


def _chunk_has_any_entity(chunk: dict, *entities: str) -> bool:
    chunk_entities = set(chunk.get("entities") or [])
    return any(entity in chunk_entities for entity in entities)


def _is_pinout_query(query: str) -> bool:
    return bool(query) and bool(PINOUT_QUERY_RE.search(query)) and bool(
        CONNECTOR_QUERY_RE.search(query) or ECM_QUERY_RE.search(query)
    )


def _is_connector_correspondence_query(query: str) -> bool:
    return bool(query) and bool(
        CONNECTOR_QUERY_RE.search(query)
        and CONNECTOR_FACE_QUERY_RE.search(query)
        and CORRESPONDENCE_QUERY_RE.search(query)
    )


def _is_pinout_candidate(chunk: dict) -> bool:
    chunk_type = chunk.get("type")
    if chunk_type != "table":
        return False
    text = (chunk.get("text") or "").lower()
    if "fuel injection ecm connector identification" in text:
        return True
    return bool(
        _chunk_has_any_info_type(chunk, "pinout", "spec", "wiring")
        and (
            _chunk_has_any_entity(chunk, "ecm", "connector a", "connector b")
            or "ecm connector" in text
        )
        and (
            "pin:" in text
            or "wire color:" in text
            or "circuit:" in text
            or "voltage" in text
        )
    )


def _is_connector_face_candidate(chunk: dict) -> bool:
    chunk_type = chunk.get("type")
    if chunk_type not in ("figure", "table", "paragraph"):
        return False
    text = (chunk.get("text") or "").lower()
    if "harness connector faces" in text:
        return True
    return bool(
        _chunk_has_any_info_type(chunk, "connector_face")
        and (
            chunk_type == "figure"
            or _chunk_has_any_entity(chunk, "connector c1", "connector c2")
        )
    )


def _collect_connector_coverage_candidates(
    *,
    query: str,
    search_query: str,
    index: RetrievalIndex,
    top_k: int,
    engine_variant: str,
) -> dict[str, list[dict]]:
    combined_query = " ".join(part for part in [query, search_query] if part).strip()
    if not combined_query:
        return {}
    is_correspondence_query = _is_connector_correspondence_query(combined_query)
    needs_pinout = _is_pinout_query(combined_query) or _is_connector_correspondence_query(combined_query)
    needs_connector_face = is_correspondence_query or bool(
        CONNECTOR_FACE_QUERY_RE.search(combined_query)
    )
    if not needs_pinout and not needs_connector_face:
        return {}

    candidates: dict[str, list[dict]] = {}
    query_k = max(8, top_k)

    if needs_pinout:
        if is_correspondence_query:
            pinout_query = (
                "fuel injection ecm connector identification "
                "ecm connector a connector b "
                "pinout terminal voltage key on engine run"
            )
        else:
            pinout_query = (
                f"{combined_query} ECM connector identification connector A connector B "
                "pinout terminal voltage key on engine run"
            )
        pinout_results = index.retrieve(pinout_query, top_k=query_k, engine_variant=engine_variant)
        pinout_candidates = [row for row in pinout_results if _is_pinout_candidate(row["chunk"])]
        if pinout_candidates:
            candidates["pinout"] = pinout_candidates[:4]

    if needs_connector_face:
        face_query = (
            f"{combined_query} harness connector faces connector face diagram "
            "terminal end view connector C1 connector C2"
        )
        face_results = index.retrieve(face_query, top_k=query_k, engine_variant=engine_variant)
        face_candidates = [row for row in face_results if _is_connector_face_candidate(row["chunk"])]
        if face_candidates:
            candidates["connector_face"] = face_candidates[:4]

    return candidates


def _merge_candidate_rows(
    results: list[dict],
    extra_rows: list[dict],
    *,
    score_boost: float = 0.0,
    max_extra: int = 8,
) -> list[dict]:
    merged = list(results)
    seen_ids = {row["chunk"]["chunk_id"] for row in results}
    added = 0
    for row in extra_rows:
        if added >= max_extra:
            break
        cid = row["chunk"]["chunk_id"]
        if cid in seen_ids:
            continue
        new_row = dict(row)
        new_row["score"] = row.get("score", 0.0) + score_boost
        merged.append(new_row)
        seen_ids.add(cid)
        added += 1
    return merged


def _reranked_has_coverage(category: str, reranked: list[dict]) -> bool:
    predicate = _is_pinout_candidate if category == "pinout" else _is_connector_face_candidate
    return any(predicate(row["chunk"]) for row in reranked)


def _enforce_metadata_coverage(
    query: str,
    reranked: list[dict],
    coverage_candidates: dict[str, list[dict]],
) -> list[dict]:
    if not coverage_candidates:
        return reranked

    required: list[str] = []
    if _is_connector_correspondence_query(query):
        required = ["pinout", "connector_face"]
    elif _is_pinout_query(query):
        required = ["pinout"]

    if not required:
        return reranked

    seen_ids = {row["chunk"]["chunk_id"] for row in reranked}
    additions: list[dict] = []
    for category in required:
        if _reranked_has_coverage(category, reranked):
            continue
        for candidate in coverage_candidates.get(category, []):
            cid = candidate["chunk"]["chunk_id"]
            if cid in seen_ids:
                continue
            additions.append(candidate)
            seen_ids.add(cid)
            break

    if additions:
        print(f"  [chat] Enforced metadata coverage with {len(additions)} candidate(s)")
    return reranked + additions


def _deep_research_focus_text(query: str) -> str:
    parts: list[str] = []
    for pattern, supplement in DEEP_RESEARCH_FOCUS_TERMS:
        if re.search(pattern, query, re.IGNORECASE):
            parts.append(supplement)
    return " ".join(parts)


def _extract_deep_research_topics(query: str) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    seen: set[str] = set()
    for title, pattern, focused_query in DEEP_RESEARCH_TOPIC_PATTERNS:
        if title in seen:
            continue
        if re.search(pattern, query, re.IGNORECASE):
            topics.append((title, focused_query))
            seen.add(title)
    return topics


def _build_deep_research_plan(
    query: str,
    search_query: str,
    route: SectionRoute,
    *,
    forced: bool,
) -> DeepResearchPlan | None:
    lowered = (query or "").lower()
    topics = _extract_deep_research_topics(query)
    has_broad_research_signal = bool(DEEP_RESEARCH_HINT_RE.search(query))
    has_list_signal = query.count(",") >= 1 or len(re.findall(r"\b(?:and|or)\b", lowered)) >= 2
    very_long_query = len(query.strip()) >= 140
    should_auto = (
        len(topics) >= 2
        or (has_broad_research_signal and len(topics) >= 1)
        or (has_broad_research_signal and has_list_signal)
        or (has_broad_research_signal and very_long_query)
    )
    if not forced and not should_auto:
        return None

    allowed_sections = list(route.primary_sections)
    for section_code in route.secondary_sections:
        if section_code not in allowed_sections:
            allowed_sections.append(section_code)

    focus_text = _deep_research_focus_text(query)
    sub_queries: list[DeepResearchQuery] = []
    if topics:
        for title, focused_query in topics:
            targeted = f"{focused_query} {focus_text}".strip()
            sub_queries.append(
                DeepResearchQuery(
                    title=title,
                    query=targeted,
                    target_sections=allowed_sections,
                )
            )
    else:
        sub_queries.append(
            DeepResearchQuery(
                title="Primary question",
                query=search_query,
                target_sections=allowed_sections,
            )
        )

    section_summary = ", ".join(allowed_sections) if allowed_sections else "broad manual evidence"
    title_summary = ", ".join(sub.title for sub in sub_queries[:6])
    summary = (
        f"Researched {len(sub_queries)} sub-topic"
        f"{'' if len(sub_queries) == 1 else 's'} across {section_summary}: {title_summary}"
    )
    return DeepResearchPlan(
        forced=forced,
        auto_detected=not forced,
        summary=summary,
        sub_queries=sub_queries,
    )


def _deep_research_retrieve(
    plan: DeepResearchPlan,
    *,
    search_query: str,
    index: RetrievalIndex,
    top_k: int,
    top_n: int,
    engine_variant: str,
    progress_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    pool_size = max(top_k * 2, 16)
    per_topic_keep = max(top_n, 6)
    merged_by_id: dict[str, dict] = {}

    for sub_query in plan.sub_queries:
        if progress_cb:
            try:
                progress_cb({"type": "fetching", "label": f"Deep research: {sub_query.title}"})
            except Exception:
                pass

        sub_results = index.retrieve(
            sub_query.query,
            top_k=pool_size,
            engine_variant=engine_variant,
        )
        sub_results = _filter_results_to_sections(sub_results, sub_query.target_sections)
        sub_results = _apply_supplement_authority_to_results(
            sub_results,
            label=f"deep research {sub_query.title}",
        )
        sub_results = _neural_rerank(sub_query.query, sub_results, per_topic_keep + 4)

        for row in sub_results[:per_topic_keep]:
            chunk_id = row["chunk"]["chunk_id"]
            previous = merged_by_id.get(chunk_id)
            if previous is None or row["score"] > previous["score"]:
                merged_by_id[chunk_id] = row

    merged_results = list(merged_by_id.values())
    if not merged_results:
        return []

    merged_results = _apply_supplement_authority_to_results(
        merged_results,
        label="deep research merge",
    )
    return _neural_rerank(search_query, merged_results, max(top_k * 2, top_n * 2 + 6))


_REWRITE_PROMPT = """You are a search query rewriter for a 1990 Geo Metro factory service manual RAG system.

Given a conversation between a car owner and an assistant, rewrite the owner's latest message into a standalone search query that will find the most relevant manual sections.

Rules:
- Output ONLY the rewritten query, nothing else
- Include the specific system/component being discussed (e.g. "AC compressor", "brake pads")
- Capture the user's actual intent — if they've ruled something out, focus on what they need next
- Include relevant technical terms from the conversation
- If the user asks for a diagram, figure, illustration, or picture, include "diagram figure" in the query
- If the user asks for a procedure (how to do something), include action verbs (removal, installation, inspection)
- If the user asks for specifications, include "specification" or "torque"
- Keep it under 30 words
- Fix any spelling errors

Examples:
  Conversation: "My AC isn't cold" → "AC not cooling compressor clutch diagnosis"
  Follow-up: "I already checked the refrigerant pressure" → "AC compressor clutch not engaging electrical diagnosis fuse relay pressure switch wiring"
  Follow-up: "what about the belt?" → "AC compressor drive belt inspection tension squealing"
  Follow-up: "show me a diagram of the compressor" → "AC compressor diagram figure disassembly components"
  Follow-up: "how do I remove it?" → "AC compressor removal procedure disconnect"
"""


def _rewrite_query(query: str, conversation: list[dict], config: dict) -> str:
    """Use LLM to rewrite a follow-up query into a standalone search query."""
    # Build a compact conversation summary for the rewriter
    conv_lines = []
    for m in conversation[-6:]:  # Last 6 messages max
        role = "Owner" if m.get("role") == "user" else "Assistant"
        text = m.get("text", "")[:300]  # Truncate long responses
        conv_lines.append(f"{role}: {text}")
    conv_lines.append(f"Owner: {query}")
    conv_text = "\n".join(conv_lines)

    messages = [{"system": _REWRITE_PROMPT,
                 "messages": [{"role": "user", "content": conv_text}]}]

    try:
        cfg_chat = config.get("chat", {})
        model = cfg_chat.get("model", "gemini-2.5-flash")
        if model.startswith("gemini"):
            api_key = cfg_chat.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
            result = call_gemini(messages, api_key=api_key, model=model)
        else:
            client = anthropic.Anthropic()
            payload = messages[0]
            resp = client.messages.create(
                model=model, max_tokens=100, temperature=0.0,
                system=payload["system"],
                messages=payload["messages"],
            )
            result = resp.content[0].text
        rewritten = result.strip().strip('"').strip("'")
        print(f"  [chat] Query rewrite: \"{query}\" -> \"{rewritten}\"")
        return rewritten
    except Exception as e:
        print(f"  [chat] Query rewrite failed ({e}), using original")
        return query


def _detect_figure_intent(query: str) -> bool:
    """Detect if the user is asking for a diagram, figure, or illustration."""
    q = query.lower()
    fig_keywords = [
        "diagram", "figure", "illustration", "picture", "image",
        "drawing", "schematic", "exploded view", "cutaway",
        "show me", "what does it look like", "what does the",
        "visual", "layout", "where is the",
    ]
    return any(kw in q for kw in fig_keywords)


# Broad system-level terms that dilute figure searches
_SYSTEM_STRIP_TERMS = {
    "air conditioning", "air conditioner", "heating", "hvac ventilation",
    "engine mechanical", "engine cooling", "engine fuel", "engine electrical",
    "emission controls", "manual transaxle", "automatic transaxle",
    "front suspension", "rear suspension", "power steering", "manual steering",
    "drive axle", "maintenance", "lubrication", "electrical wiring",
}

# Words to remove from figure queries (filler/intent words)
_FIGURE_STOP_WORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "or", "is",
    "are", "has", "have", "show", "me", "surely", "manual", "does", "it",
    "look", "like", "where", "what", "how", "can", "you", "find", "get",
    "see", "there", "any", "this", "that", "with", "from", "about",
    "diagram", "figure", "illustration", "picture", "image", "drawing",
}


def _build_figure_query(search_query: str, detected_system: str | None) -> str:
    """Build a component-focused query for figure retrieval.

    Strips broad system terms (e.g., 'air conditioning') that would dilute
    toward system-overview figures, keeping only specific component nouns.
    Appends 'diagram figure' for BM25 matching against figure chunk text.
    """
    q = search_query.lower()

    # Strip broad system terms
    for term in _SYSTEM_STRIP_TERMS:
        q = q.replace(term, " ")

    # Tokenize and remove stop words
    words = q.split()
    component_words = [w for w in words if w.strip() not in _FIGURE_STOP_WORDS and len(w) > 1]

    if not component_words:
        # Fallback: use detected system as component
        component_words = [detected_system or "component"]

    return " ".join(component_words) + " diagram figure"


def chat(query: str,
         conversation: list[dict],
         index: RetrievalIndex,
         config: dict,
         skip_vision: bool = False,
         deep_research: bool = False,
         project_context: str | None = None,
         notes_context: str | None = None,
         vehicle_settings: str | None = None,
         images: list[str] | None = None,
         progress_cb: Callable[[dict], None] | None = None,
         note_callback: Callable[[dict], None] | None = None,
         retrieve_notes: Callable[[dict], tuple[str, bool]] | None = None) -> ChatResponse:
    """Full RAG chat pipeline."""
    cfg_chat = config.get("chat", {})
    top_k    = cfg_chat.get("top_k_retrieve", 20)
    top_n    = cfg_chat.get("top_n_rerank", 8)
    is_engine_oil_service = _is_engine_oil_service_query(query)

    # For follow-up messages, use LLM to rewrite the query into a standalone
    # search query that captures the full conversational intent.
    if conversation:
        rewritten = _rewrite_query(query, conversation, config)
        search_query = _expand_query(rewritten)
    else:
        search_query = _expand_query(query)

    route = _route_for_query(search_query, query)
    deep_research_plan = _build_deep_research_plan(
        query,
        search_query,
        route,
        forced=deep_research,
    )
    response_mode = "deep_research" if deep_research_plan else "normal"
    deep_research_summary = deep_research_plan.summary if deep_research_plan else None

    if deep_research_plan:
        trigger = "forced" if deep_research_plan.forced else "auto"
        print(f"  [deep_research] {trigger}: {deep_research_plan.summary}")

    # Detect figure/diagram intent from both original query and rewritten query
    figure_intent = _detect_figure_intent(query) or _detect_figure_intent(search_query)
    if figure_intent:
        print(f"  [chat] Figure/diagram intent detected")

    # 5.1 Step 1: Keyword pre-filter
    # Use BOTH original query and rewritten query for system detection
    detected_system = _detect_system(search_query)
    if not detected_system:
        detected_system = _detect_system(query)

    # User's car is G10
    engine_variant = "G10"
    coverage_candidates: dict[str, list[dict]] = {}

    if deep_research_plan:
        results = _deep_research_retrieve(
            deep_research_plan,
            search_query=search_query,
            index=index,
            top_k=top_k,
            top_n=top_n,
            engine_variant=engine_variant,
            progress_cb=progress_cb,
        )
    else:
        # 5.1 Step 2: Semantic search within system
        results = index.retrieve(search_query, top_k=top_k, system=detected_system, engine_variant=engine_variant)

        # 5.1 Step 3: Always merge broad (unfiltered) results to catch cross-section content
        if detected_system:
            broad_results = index.retrieve(search_query, top_k=top_k, engine_variant=engine_variant)
            seen_ids = {r["chunk"]["chunk_id"] for r in results}
            added_broad = 0
            for r in broad_results:
                if r["chunk"]["chunk_id"] not in seen_ids:
                    r["chunk"]["is_cross_reference"] = True
                    results.append(r)
                    seen_ids.add(r["chunk"]["chunk_id"])
                    added_broad += 1
            if added_broad:
                print(f"  [chat] Merged {added_broad} cross-section result(s)")
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k * 2]

        if is_engine_oil_service:
            engine_oil_query = "engine oil drain plug torque 35 N·m oil pan refill"
            engine_results = index.retrieve(
                engine_oil_query,
                top_k=max(6, top_k // 2),
                system="engine",
                engine_variant=engine_variant,
            )
            seen_ids = {r["chunk"]["chunk_id"] for r in results}
            added_engine = 0
            for r in engine_results:
                if r["chunk"]["chunk_id"] not in seen_ids:
                    r["chunk"]["is_cross_reference"] = True
                    results.append(r)
                    seen_ids.add(r["chunk"]["chunk_id"])
                    added_engine += 1
            if added_engine:
                print(f"  [chat] Added {added_engine} engine oil service chunk(s)")
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k * 3]

        # 5.1 Step 4: Figure-targeted retrieval when user asks for a diagram
        if figure_intent:
            # Build a component-focused query: strip broad system terms,
            # keep specific component nouns, append "diagram figure"
            fig_query = _build_figure_query(search_query, detected_system)
            print(f"  [chat] Figure query: \"{fig_query}\"")
            fig_results = index.retrieve(fig_query, top_k=top_k, engine_variant=engine_variant)
            # Keep only figure-type chunks from this pass
            fig_results = [r for r in fig_results if r["chunk"].get("type") == "figure"]
            seen_ids = {r["chunk"]["chunk_id"] for r in results}
            added_figs = 0
            for r in fig_results[:8]:
                if r["chunk"]["chunk_id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["chunk"]["chunk_id"])
                    added_figs += 1
            if added_figs:
                print(f"  [chat] Added {added_figs} figure-targeted result(s)")

        coverage_candidates = _collect_connector_coverage_candidates(
            query=query,
            search_query=search_query,
            index=index,
            top_k=top_k,
            engine_variant=engine_variant,
        )
        if coverage_candidates:
            coverage_rows: list[dict] = []
            for rows in coverage_candidates.values():
                coverage_rows.extend(rows)
            before_merge = len(results)
            results = _merge_candidate_rows(results, coverage_rows, score_boost=0.03, max_extra=6)
            added_coverage = len(results) - before_merge
            if added_coverage:
                print(f"  [chat] Added {added_coverage} metadata coverage candidate(s)")

    results = _apply_supplement_authority_to_results(results, label="retrieval merge")

    # 1.5 Neural rerank: cross-encoder scores (query, chunk) pairs for precision.
    # Keep a wider pool (top_k + 10) so borderline cross-section chunks survive
    # for the type-boost reranker to evaluate.
    results = _neural_rerank(search_query, results, top_k + 10)

    # 2. Rerank: type boosts + intent detection + diversity enforcement
    reranked = _rerank(results, top_n, query=query, figure_intent=figure_intent)

    # 2.5 Same-page context: pull in related non-figure chunks from pages
    #     already in the results (e.g. symptom index + diagnostic checklist)
    reranked = _expand_page_context(reranked, results)

    # 2.6 Dependency resolution: pull in referenced figure/flowchart chunks
    reranked = _expand_dependencies(reranked, index)
    reranked = _enforce_metadata_coverage(query, reranked, coverage_candidates)
    reranked = _apply_supplement_authority_to_results(reranked, label="final evidence")

    # 2.7 Section router evidence validation: if top evidence mismatches
    #     the router's predicted sections, force a corrective retrieval pass.
    if route.primary_sections:
        print(f"  [router] Route: {route.primary_sections} (secondary: {route.secondary_sections}), "
              f"confidence={route.confidence}, matched={route.matched_terms}")
        reranked = _validate_and_correct_evidence(
            search_query, reranked, route, index, top_k, top_n, engine_variant,
        )

    # 3. Collect figure images
    figure_evidence = []
    if not skip_vision:
        figure_evidence = _collect_figures(reranked, config)

    # 4. Build prompt
    messages = _build_messages(query, reranked, figure_evidence,
                               conversation, config,
                               project_context=project_context,
                               notes_context=notes_context,
                               vehicle_settings=vehicle_settings,
                               images=images or [])

    # 5. Call Claude
    raw = _call_claude(messages, config, index=index, progress_cb=progress_cb,
                       note_callback=note_callback, retrieve_notes=retrieve_notes)

    # 6. Parse response
    response = _parse_response(raw, reranked, query=query, config=config)
    response.mode = response_mode
    response.deep_research_summary = deep_research_summary

    # 7. Ensure ALL evidence figures are returned to the UI
    #    (not just ones Claude cited with [p### | fig: fig_id] format)
    evidence_fig_ids = {f["figure_id"] for f in figure_evidence}
    existing_refs = set(response.figure_refs)
    for fig_id in evidence_fig_ids:
        if fig_id not in existing_refs:
            response.figure_refs.append(fig_id)

    return response


def load_index(config: dict) -> RetrievalIndex:
    """Load the retrieval index configured for the active vehicle/profile."""
    index_dir = _index_dir_from_config(config)
    return RetrievalIndex(index_dir, config)


# ── Cross-encoder reranker (loaded once) ──────────────────────────────────

_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("  Loading cross-encoder reranker...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _cross_encoder


def _neural_rerank(query: str, results: list[dict], top_n: int) -> list[dict]:
    """Cross-encoder reranking blended with original RRF scores.

    Uses RRF fusion of two rank lists:
      1. Original RRF rank (BM25 + embedding agreement)
      2. Cross-encoder neural rank (query-chunk relevance)

    This prevents garbled OCR text (common in tables) from causing the
    cross-encoder to override strong BM25+embedding matches. Both signals
    contribute to the final ranking.
    """
    if not results:
        return results
    ce = _get_cross_encoder()

    # Build (query, enriched_text) pairs with section context
    pairs = []
    for r in results:
        chunk = r["chunk"]
        section = chunk.get("section_path", "")
        text = chunk.get("text", "")
        enriched = f"{section}: {text}" if section else text
        pairs.append((query, enriched[:500]))

    neural_scores = ce.predict(pairs)

    # Save original RRF rank (results are pre-sorted by RRF score)
    for i, r in enumerate(results):
        r["_rrf_rank"] = i

    # Sort by neural score to get neural rank
    neural_order = sorted(range(len(results)), key=lambda i: neural_scores[i], reverse=True)
    neural_rank = [0] * len(results)
    for rank, idx in enumerate(neural_order):
        neural_rank[idx] = rank

    # RRF fusion of original rank and neural rank
    RRF_K = 60
    for i, r in enumerate(results):
        r["score"] = 1.0 / (RRF_K + r["_rrf_rank"]) + 1.0 / (RRF_K + neural_rank[i])

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]


# ── Internal helpers ──────────────────────────────────────────────────────

def _rerank(results: list[dict], top_n: int, query: str = "",
            figure_intent: bool = False) -> list[dict]:
    """Rerank by retrieval score + type boost + procedure_type match. Return top_n."""
    q_lower = query.lower()
    is_engine_oil_service = _is_engine_oil_service_query(query)

    # Detect specific procedure type intent
    intent = None
    if any(k in q_lower for k in ["install", "installation", "connect", "attach", "assemble"]):
        intent = "installation"
    elif any(k in q_lower for k in ["remove", "removal", "disconnect", "detach", "disassemble"]):
        intent = "removal"
    elif any(k in q_lower for k in ["inspect", "inspection", "check", "test"]):
        intent = "inspection"
    elif any(k in q_lower for k in ["adjust", "adjustment", "calibrate"]):
        intent = "adjustment"

    # Detect general procedural vs informational query
    is_procedural = intent is not None or any(
        k in q_lower for k in [
            "how do i", "how to", "steps to", "procedure for",
            "replace", "change", "fix", "repair", "troubleshoot",
            "diagnose", "bleed", "flush", "drain", "fill",
            "tighten", "loosen", "clean", "reset", "rebuild",
        ]
    )

    # Detect diagnostic/troubleshooting intent — user is asking about
    # causes, symptoms, or diagnosis. Diagnostic TABLE chunks with
    # "CONDITION:" content are the best answer for these queries.
    is_diagnostic = any(
        k in q_lower for k in [
            "cause", "causes", "why does", "why is", "reason",
            "diagnosis", "diagnostic", "symptom", "condition",
            "problem", "issue", "not working", "inoperative",
            "won't", "doesn't", "fails to", "unable to",
        ]
    )

    for r in results:
        chunk = r["chunk"]
        ctype = chunk.get("type", "paragraph")
        chunk_text = chunk.get("text", "")
        chunk_text_lower = chunk_text.lower()
        section_lower = (chunk.get("section_path") or "").lower()

        if figure_intent:
            # Figure/diagram query — heavily boost figure chunks
            if ctype == "figure":
                boost = 0.40
            else:
                boost = {"warning": 0.05, "caution": 0.05, "procedure": 0.05}.get(ctype, 0.0)
        elif is_procedural:
            # Procedural query — boost procedures and safety info
            boost = TYPE_BOOST.get(ctype, 0.0)
            # Extra boost if procedure_type matches specific intent
            ptype = chunk.get("procedure_type")
            if intent and ptype == intent:
                boost += 0.3
        else:
            # Informational query ("where is X?", "what is the spec?")
            # Tables get a meaningful boost — they contain specs, diagnosis,
            # and reference data that directly answer informational queries.
            # Figures get no boost — let relevance alone decide.
            boost = {
                "warning": 0.10,
                "caution": 0.10,
                "table":   0.10,
            }.get(ctype, 0.0)

        # Diagnostic intent: extra boost for table chunks that contain
        # structured CONDITION/CAUSE/CORRECTION diagnostic data.
        if is_diagnostic and ctype == "table" and "CONDITION:" in chunk_text[:300]:
            boost += 0.15

        # Penalize cross-references slightly to prefer local system results.
        # Tables/specs get a lighter penalty — they often contain critical
        # cross-section data (torque values, capacities, part numbers).
        if chunk.get("is_cross_reference"):
            if ctype in ("table",):
                boost -= 0.05
            else:
                boost -= 0.15

        if is_engine_oil_service:
            if (
                "engine oil" in chunk_text_lower
                or "engine crankcase" in chunk_text_lower
                or "viscosity" in chunk_text_lower
                or "oil filter change" in chunk_text_lower
                or "api service sg" in chunk_text_lower
                or "maintenance schedule" in section_lower
            ):
                boost += 0.25
            if (
                "gm goodwrench" in chunk_text_lower
                or "engine oil filter" in chunk_text_lower
                or "engine crankcase" in chunk_text_lower
            ):
                boost += 0.12
            if "maintenance and lubrication" in section_lower or "maintenance schedule" in section_lower:
                boost += 0.15
            if "engine oil viscosity recommendation" in chunk_text_lower:
                boost += 0.20
            if (
                "drain plug" in chunk_text_lower
                and "oil pan" in chunk_text_lower
                and "refill oil" in chunk_text_lower
            ):
                boost += 0.75
            if ctype == "procedure":
                boost -= TYPE_BOOST.get(ctype, 0.0)
            if (
                "transaxle" in chunk_text_lower
                or "gear oil" in chunk_text_lower
                or "oil pressure" in chunk_text_lower
            ):
                boost -= 0.25
            if ctype == "procedure" and (
                "air cleaner" in chunk_text_lower
                or "pcv" in chunk_text_lower
                or "transaxle" in chunk_text_lower
                or "oil pressure" in chunk_text_lower
            ):
                boost -= 0.20
            if ctype == "procedure" and "maintenance and lubrication" not in section_lower:
                boost -= 0.20
            if (
                ctype == "table"
                and "maintenance schedule" in section_lower
                and "engine oil and oil filter change" not in chunk_text_lower
                and "scheduled maintenance" not in chunk_text_lower
                and "engine oil filter" not in chunk_text_lower
                and "engine crankcase" not in chunk_text_lower
            ):
                boost -= 0.12

        r["rerank_score"] = r["score"] + boost

    ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

    # Suppress irrelevant figures: when no figure intent is detected,
    # drop figures whose neural score is well below the best non-figure.
    # This prevents noise like ESD labels showing up for "oil capacity" queries.
    if not figure_intent:
        non_fig_scores = [r["score"] for r in ranked if r["chunk"].get("type") != "figure"]
        if non_fig_scores:
            # Figures must score at least 50% of the top non-figure's neural score
            fig_threshold = max(non_fig_scores) * 0.50
            ranked = [
                r for r in ranked
                if r["chunk"].get("type") != "figure" or r["score"] >= fig_threshold
            ]

    # Enforce type diversity — cap figures to prevent flooding results.
    if figure_intent:
        # For diagram queries, allow more figures (up to 4)
        MAX_FIGURES = 4
    elif is_procedural:
        MAX_FIGURES = 2
    else:
        MAX_FIGURES = 3

    diverse: list[dict] = []
    fig_count = 0
    overflow: list[dict] = []
    for r in ranked:
        if r["chunk"].get("type") == "figure":
            if fig_count < MAX_FIGURES:
                diverse.append(r)
                fig_count += 1
            else:
                overflow.append(r)
        else:
            diverse.append(r)
        if len(diverse) >= top_n:
            break

    # Fill remaining slots with overflow figures if needed
    while len(diverse) < top_n and overflow:
        diverse.append(overflow.pop(0))

    return diverse[:top_n]


def _expand_page_context(reranked: list[dict], all_results: list[dict]) -> list[dict]:
    """Pull in related non-figure chunks from pages already in the reranked results.

    When a table and a procedure are on the same page (e.g. symptom index +
    diagnostic checklist), they should both appear in the evidence. This scans
    the broader retrieval results for same-page chunks that didn't make the
    top-N cut but have decent relevance scores.
    """
    MAX_PAGE_ADDITIONS = 3
    seen_ids = {r["chunk"]["chunk_id"] for r in reranked}
    reranked_pages = {r["chunk"].get("page") for r in reranked if r["chunk"].get("page")}

    # Score threshold: only add chunks scoring at least 60% of the top result
    if not reranked:
        return reranked
    top_score = reranked[0].get("rerank_score", reranked[0]["score"])
    min_score = top_score * 0.6

    additions = []
    for r in all_results:
        if len(additions) >= MAX_PAGE_ADDITIONS:
            break
        chunk = r["chunk"]
        cid = chunk["chunk_id"]
        if cid in seen_ids:
            continue
        if chunk.get("type") == "figure":
            continue  # Figures handled by _expand_dependencies
        page = chunk.get("page")
        if page not in reranked_pages:
            continue
        score = r.get("rerank_score", r["score"])
        if score < min_score:
            continue
        additions.append(r)
        seen_ids.add(cid)

    if additions:
        print(f"  [chat] Added {len(additions)} same-page context chunk(s)")

    return reranked + additions


FIGURE_TEXT_REF_RE = re.compile(r"(?:Figure|Fig\.?)\s+(\d+[A-Z]*\d*[-\d]*)", re.IGNORECASE)
CHART_REF_RE = re.compile(
    r"(?:diagnostic\s+chart|flowchart|wiring\s+diagram|schematic|"
    r"(?:refer\s+to|see)\s+(?:the\s+)?(?:chart|diagram)|"
    r"perform\s+[\"'].+?[\"']\s*check|"
    r"trouble\s*shooting)",
    re.IGNORECASE,
)


def _expand_dependencies(reranked: list[dict], index: RetrievalIndex) -> list[dict]:
    """Pull in referenced figure/flowchart chunks that aren't already retrieved.

    Detects three kinds of references:
      1. figure_refs field (pipeline-assigned figure IDs) — same-page figures
      2. "Figure X" text patterns — cross-page figure chunks
      3. "diagnostic chart" / "diagram" text — same-page and adjacent-page figures
    """
    MAX_DEP_FIGURES = 4
    seen_ids = {r["chunk"]["chunk_id"] for r in reranked}
    additions = []

    for r in reranked:
        if len(additions) >= MAX_DEP_FIGURES:
            break
        chunk = r["chunk"]
        page = chunk.get("page")

        # 1. Check figure_refs field (pipeline-assigned references)
        #    Match on the SPECIFIC figure_id, not just same-page.
        for fig_id in chunk.get("figure_refs", []):
            if len(additions) >= MAX_DEP_FIGURES:
                break
            # Find the chunk whose figure_refs contains this fig_id
            # or whose text mentions the figure
            for cid, candidate in index.lookup.items():
                if cid in seen_ids or candidate.get("type") != "figure":
                    continue
                # Match: chunk has this fig_id in its own figure_refs,
                # or chunk text contains the figure ID string
                cand_figs = candidate.get("figure_refs", [])
                if fig_id in cand_figs or fig_id in candidate.get("text", ""):
                    additions.append({
                        "chunk": candidate,
                        "score": r["score"] * 0.9,
                        "rerank_score": r["rerank_score"] * 0.9,
                    })
                    seen_ids.add(cid)
                    break  # Found the match for this fig_id

        # 2. Scan text for explicit "Figure X" references
        #    Skip figure chunks — they name themselves (e.g. "Figure 3 Oil
        #    Viscosity...") and scanning them matches unrelated same-number figs.
        if len(additions) >= MAX_DEP_FIGURES:
            continue
        text = chunk.get("text", "")
        if chunk.get("type") != "figure":
            for m in FIGURE_TEXT_REF_RE.finditer(text):
                if len(additions) >= MAX_DEP_FIGURES:
                    break
                fig_label = m.group(1)
                for cid, candidate in index.lookup.items():
                    if cid in seen_ids or candidate.get("type") != "figure":
                        continue
                    cand_text = candidate.get("text", "")
                    if f"Figure {fig_label}" in cand_text or f"Figure {fig_label} " in cand_text:
                        additions.append({
                            "chunk": candidate,
                            "score": r["score"] * 0.8,
                            "rerank_score": r["rerank_score"] * 0.8,
                        })
                        seen_ids.add(cid)
                        break

        # 3. Detect "diagnostic chart" / "diagram" references — search
        #    same page and adjacent pages for figure chunks (common in
        #    section 6E2 where chart keys and flowcharts are on facing pages)
        if len(additions) >= MAX_DEP_FIGURES:
            continue
        if page and CHART_REF_RE.search(text):
            nearby_pages = {page - 1, page, page + 1}
            for cid, candidate in index.lookup.items():
                if len(additions) >= MAX_DEP_FIGURES:
                    break
                if (cid not in seen_ids
                        and candidate.get("type") == "figure"
                        and candidate.get("page") in nearby_pages):
                    additions.append({
                        "chunk": candidate,
                        "score": r["score"] * 0.85,
                        "rerank_score": r["rerank_score"] * 0.85,
                    })
                    seen_ids.add(cid)

    # 4. Procedure continuation linking: if a retrieved procedure starts
    #    mid-sequence or has [CONTINUED ON NEXT PAGE], find preceding/following
    #    procedure chunks with the same heading.
    MAX_PROC_EXPANSIONS = 3
    proc_additions = []
    for r in reranked:
        if len(proc_additions) >= MAX_PROC_EXPANSIONS:
            break
        chunk = r["chunk"]
        if chunk.get("type") != "procedure":
            continue
        page = chunk.get("page")
        heading = chunk.get("text", "").split("\n")[0].strip()
        if not heading:
            continue
        start = chunk.get("starting_step") or 1
        has_continuation = "[CONTINUED" in chunk.get("text", "")

        if start > 1 or has_continuation:
            # Find matching procedures on adjacent pages with same heading
            nearby_pages = {page - 1, page, page + 1} if page else set()
            for cid, candidate in index.lookup.items():
                if len(proc_additions) >= MAX_PROC_EXPANSIONS:
                    break
                if cid in seen_ids or candidate.get("type") != "procedure":
                    continue
                if candidate.get("page") not in nearby_pages:
                    continue
                cand_heading = candidate.get("text", "").split("\n")[0].strip()
                if cand_heading == heading:
                    proc_additions.append({
                        "chunk": candidate,
                        "score": r["score"] * 0.95,
                        "rerank_score": r["rerank_score"] * 0.95,
                    })
                    seen_ids.add(cid)

    additions.extend(proc_additions)

    if additions:
        print(f"  [chat] Expanded {len(additions)} dependency chunk(s)")

    return reranked + additions


def _collect_figures(reranked: list[dict], config: dict | None = None) -> list[dict]:
    """Gather figure image data for chunks that reference figures."""
    seen    = set()
    figures = []

    # Load figures lookup
    build_dir = PROJECT_ROOT / "build"
    if config:
        from src.utils import resolve_path
        build_dir = resolve_path(config.get("pipeline", {}).get("build_dir", "build"), PROJECT_ROOT)
    fig_lookup_path = build_dir / "figures.jsonl"
    fig_lookup: dict[str, dict] = {}
    if fig_lookup_path.exists():
        from src.utils import load_jsonl
        for fig in load_jsonl(fig_lookup_path):
            if fig.get("figure_id"):
                fig_lookup[fig["figure_id"]] = fig

    for r in reranked:
        chunk = r["chunk"]
        # Collect figure IDs from figure_refs AND from figure-type chunks themselves
        fig_ids = list(chunk.get("figure_refs", []))
        if chunk.get("type") == "figure":
            # For figure-type chunks, match the specific figure by checking
            # if the chunk's figure_refs or text contains the figure_id
            for fid in fig_lookup:
                if fid in seen:
                    continue
                if fid in fig_ids:
                    continue  # Already from figure_refs
                # Match: figure_id appears in chunk's figure_refs or text
                if fid in chunk.get("figure_refs", []) or fid in chunk.get("text", ""):
                    fig_ids.append(fid)

        for fig_id in fig_ids:
            if fig_id in seen:
                continue
            seen.add(fig_id)
            fig = fig_lookup.get(fig_id)
            if not fig:
                continue
            asset_path = PROJECT_ROOT / fig.get("asset_path", "")
            if not asset_path.exists():
                continue
            try:
                with open(asset_path, "rb") as f:
                    img_data = base64.standard_b64encode(f.read()).decode()
                if not img_data:
                    continue
                figures.append({
                    "figure_id":    fig_id,
                    "page":         fig.get("page"),
                    "caption_text": fig.get("caption_text", ""),
                    "asset_path":   str(asset_path),
                    "b64_data":     img_data,
                    "media_type":   "image/webp",
                })
            except Exception:
                pass

    return figures[:4]  # Claude Vision: max 4 images


def _build_messages(query: str,
                    reranked: list[dict],
                    figures: list[dict],
                    conversation: list[dict],
                    config: dict,
                    project_context: str | None = None,
                    notes_context: str | None = None,
                    vehicle_settings: str | None = None,
                    images: list[str] | None = None) -> list[dict]:
    """Build the Claude API messages list."""
    # Load system prompt
    sys_prompt_path = _system_prompt_path_from_config(config)
    system_prompt   = sys_prompt_path.read_text(encoding="utf-8")

    toc_text = get_toc_text(config)
    if toc_text:
        system_prompt = (
            "You have access to the full manual. Here is the table of contents. "
            "The fetch_chunks tool is available now and should be used for any additional manual lookups. "
            "Never expose manual lookups to the owner, never emit placeholder tags like [CHUNK_REQUEST: ...], "
            "and never narrate that you are requesting chunks. If you need more manual context, call the native tool internally and then answer normally.\n\n"
            f"{toc_text}\n\n"
            f"{system_prompt}"
        )

    # Inject owner-confirmed vehicle settings (overrides defaults in the static prompt)
    if vehicle_settings:
        system_prompt += f"\n\n{vehicle_settings}"

    # Append project context if this chat belongs to a project
    if project_context:
        system_prompt += f"\n\n## ACTIVE PROJECT\n\n{project_context}\n\nAll questions in this conversation relate to the project above. Keep this context in mind when answering — reference the project goals and adapt your guidance accordingly."

    if notes_context:
        system_prompt += (
            "\n\n## NOTEBOOK CONTEXT\n\n"
            f"{notes_context}\n\n"
            "Use these saved notes as prior context for this conversation. "
            "If the owner asks what previous notes say, what has already been ruled out, "
            "or what measurements were saved earlier, use this context and retrieve more notes when needed."
        )

    if _is_connector_correspondence_query(query):
        system_prompt += (
            "\n\n## CORRESPONDENCE RULES\n\n"
            "When the owner asks which connector, table, diagram, or artifact corresponds to another one, "
            "do not infer the mapping from shared pin count, shape, or naming alone. "
            "State a correspondence only if the retrieved manual evidence explicitly links the artifacts. "
            "If the evidence shows both artifacts but does not provide a direct cross-reference, say that clearly and stop there. "
            "Do not list plausible candidates."
        )

    if _is_pinout_query(query):
        system_prompt += (
            "\n\n## CITATION RULES\n\n"
            "When you answer from textual specs or pinout tables, prefer citing the retrieved text chunk directly "
            "with the exact chunk ID format like [p369 | tbl_6e2_p369_1]. "
            "Do not rely on a figure-only citation when the answer content comes from table text."
        )

    # Format evidence blocks — use chunk_id as the label (not numbered)
    # so Claude cites by chunk_id rather than saying "Evidence 3"
    evidence_lines = []
    for i, r in enumerate(reranked, 1):
        chunk = r["chunk"]
        header = (
            f"--- {chunk['chunk_id']} | "
            f"page: {chunk['page']} | "
            f"source_label: {chunk.get('source_label', '')} | "
            f"type: {chunk['type']} ---"
        )
        evidence_lines.append(header)
        evidence_lines.append(chunk.get("text", ""))
        evidence_lines.append("")
    evidence_text = "\n".join(evidence_lines)

    # Build user message content (may include images)
    user_content: list[dict] = [{"type": "text", "text": f"EVIDENCE:\n\n{evidence_text}"}]

    for fig in figures:
        user_content.append({
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": fig["media_type"],
                "data":       fig["b64_data"],
            }
        })
        user_content.append({
            "type": "text",
            "text": (
                f"[Figure: {fig['figure_id']} | "
                f"Page {fig['page']} | "
                f"{fig.get('caption_text', 'No caption')}]"
            )
        })

    # Attach user-uploaded images (from the frontend image picker)
    if images:
        user_content.append({"type": "text", "text": "\nThe owner has attached the following image(s) for you to analyze:"})
        for data_url in images:
            # data_url is like "data:image/jpeg;base64,..."
            if "," in data_url:
                header, b64 = data_url.split(",", 1)
                media_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
            else:
                b64, media_type = data_url, "image/jpeg"
            user_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })

    user_content.append({"type": "text", "text": f"\nQUESTION: {query}"})

    # Build messages list with conversation history
    messages = []
    for turn in conversation:
        text = turn.get("text") or ""
        if not text.strip():
            # API requires non-empty content — use placeholder for image-only messages
            text = "(image)" if turn.get("role") == "user" else "(continued)"
        messages.append({
            "role":    turn["role"],
            "content": text,
        })

    messages.append({"role": "user", "content": user_content})

    return [{"system": system_prompt, "messages": messages}]


def _block_attr(block, name: str, default=None):
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _text_from_blocks(blocks) -> str:
    parts = []
    for block in blocks or []:
        if _block_attr(block, "type") == "text":
            text = _block_attr(block, "text", "")
            if text:
                parts.append(text)
    return "".join(parts).strip()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return _text_from_blocks(content)
    return str(content or "").strip()


def _lookup_intent_without_action(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "[chunk_request:" in lowered:
        return False
    return bool(LOOKUP_INTENT_RE.search(text))


def _provider_for_model(model: str) -> str:
    return "gemini" if (model or "").startswith("gemini") else "claude"


def _get_model_capabilities(model: str) -> ModelCapabilities:
    provider = _provider_for_model(model)
    if provider == "gemini":
        return ModelCapabilities(
            provider=provider,
            supports_tools=True,
            supports_images=True,
            supports_streaming=False,
            max_tool_iterations=FETCH_CHUNKS_MAX_TOOL_ITERATIONS,
        )
    return ModelCapabilities(
        provider=provider,
        supports_tools=True,
        supports_images=True,
        supports_streaming=False,
        max_tool_iterations=FETCH_CHUNKS_MAX_TOOL_ITERATIONS,
    )


def _run_claude_turn(payload: dict,
                     convo: list[dict],
                     *,
                     client,
                     model: str,
                     max_tokens: int,
                     temperature: float) -> AgentTurn:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=payload["system"],
        messages=convo,
        tools=[FETCH_CHUNKS_TOOL, SAVE_NOTE_TOOL, RETRIEVE_NOTES_TOOL],
        tool_choice={"type": "auto"},
    )
    tool_calls = [
        AgentToolCall(
            id=_block_attr(block, "id"),
            name=_block_attr(block, "name"),
            input=_block_attr(block, "input", {}) or {},
        )
        for block in response.content
        if _block_attr(block, "type") == "tool_use"
    ]
    return AgentTurn(
        provider="claude",
        model=model,
        stop_reason=getattr(response, "stop_reason", None),
        text=_text_from_blocks(response.content),
        tool_calls=tool_calls,
        raw_content=response.content,
    )


def _run_gemini_turn(payload: dict,
                     convo: list[dict],
                     *,
                     api_key: str | None,
                     model: str) -> AgentTurn:
    tools = [FETCH_CHUNKS_TOOL, SAVE_NOTE_TOOL, RETRIEVE_NOTES_TOOL]
    result = call_gemini(
        [{"system": payload["system"], "messages": convo}],
        api_key=api_key,
        model=model,
        tools=tools,
    )
    tool_calls = [
        AgentToolCall(
            # Use function name as ID so tool_use_id maps to the function name
            # in tool results — Gemini's functionResponse needs the name, not an ID.
            id=tc["name"],
            name=tc["name"],
            input=tc.get("input", {}),
        )
        for tc in result.get("tool_calls", [])
    ]
    return AgentTurn(
        provider="gemini",
        model=model,
        stop_reason=result.get("stop_reason", "end_turn"),
        text=result.get("text", ""),
        tool_calls=tool_calls,
        raw_content=result,
    )


def _build_model_adapter(config: dict) -> ModelAdapter:
    cfg_chat = config.get("chat", {})
    model = cfg_chat.get("model", "claude-sonnet-4-20250514")
    max_tok = cfg_chat.get("max_tokens", 4096)
    temp = cfg_chat.get("temperature", 0.1)
    capabilities = _get_model_capabilities(model)

    if capabilities.provider == "gemini":
        api_key = cfg_chat.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")

        def _gemini_assistant_message(turn: AgentTurn) -> dict:
            # Store in Claude-compatible format; _convert_messages_to_gemini handles
            # the translation on each API call.
            content = []
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            raw = turn.raw_content or {}
            raw_tool_calls = raw.get("tool_calls", []) if isinstance(raw, dict) else []
            for tc in raw_tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "input": tc.get("input", {}),
                })
            return {"role": "assistant", "content": content}

        return ModelAdapter(
            provider="gemini",
            model=model,
            capabilities=capabilities,
            run_turn=lambda payload, convo: _run_gemini_turn(
                payload,
                convo,
                api_key=api_key,
                model=model,
            ),
            assistant_message=_gemini_assistant_message,
        )

    client = anthropic.Anthropic()
    return ModelAdapter(
        provider="claude",
        model=model,
        capabilities=capabilities,
        run_turn=lambda payload, convo: _run_claude_turn(
            payload,
            convo,
            client=client,
            model=model,
            max_tokens=max_tok,
            temperature=temp,
        ),
        assistant_message=lambda turn: {"role": "assistant", "content": turn.raw_content},
    )


def _manual_lookup_status(progress_cb: Callable[[dict], None] | None) -> None:
    if not progress_cb:
        return
    try:
        progress_cb({"type": "fetching", "label": FETCHING_STATUS_LABEL})
    except Exception:
        pass


def _strip_legacy_chunk_requests(text: str) -> str:
    stripped = LEGACY_CHUNK_REQUEST_RE.sub("", text or "")
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _tool_iteration_count(convo: list[dict]) -> int:
    count = 0
    for message in convo:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        ):
            count += 1
            continue
        if isinstance(content, str) and content.startswith("Internal manual lookup results:"):
            count += 1
    return count


def _resolve_legacy_chunk_request(token: str, index: RetrievalIndex) -> dict[str, object] | None:
    cleaned = token.strip().strip("\"'")
    if not cleaned:
        return None

    lookup = getattr(index, "lookup", {})
    lowered = cleaned.lower()
    if lowered.startswith("toc_") and lowered in lookup:
        toc_chunk = lookup[lowered]
        section_code = toc_chunk.get("section_code")
        if section_code:
            return {"section_code": section_code}
        page = toc_chunk.get("page")
        if isinstance(page, int):
            return {"page": page}

    page_match = re.fullmatch(r"(?:p(?:age)?\.?\s*)?(\d+)", cleaned, re.IGNORECASE)
    if page_match and cleaned.lower().startswith(("p", "page")):
        return {"page": int(page_match.group(1))}

    if " " not in cleaned and re.search(r"\d", cleaned):
        return {"section_code": cleaned.upper()}

    return {"keywords": cleaned}


def _legacy_chunk_request_context(raw: str, index: RetrievalIndex | None) -> str:
    if index is None:
        return ""

    requests: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()

    for payload in LEGACY_CHUNK_REQUEST_RE.findall(raw or ""):
        for token in payload.split(","):
            params = _resolve_legacy_chunk_request(token, index)
            if not params:
                continue
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            requests.append(params)
            if len(requests) >= FETCH_CHUNKS_MAX_LEGACY_REQUESTS:
                break
        if len(requests) >= FETCH_CHUNKS_MAX_LEGACY_REQUESTS:
            break

    if not requests:
        return ""

    chunks = []
    for params in requests:
        chunks.append(format_fetch_chunks_result(fetch_chunks(index, **params)))
    return "\n\n".join(chunks)


def _fallback_answer_from_convo(convo: list[dict], response_text: str) -> str:
    fallback = _strip_legacy_chunk_requests(response_text)
    if fallback:
        return fallback
    for msg in reversed(convo):
        if msg.get("role") != "assistant":
            continue
        earlier = _strip_legacy_chunk_requests(_content_text(msg.get("content")))
        if earlier:
            return earlier
    return ""


def _execute_tool_calls(tool_calls: list[AgentToolCall],
                        *,
                        index: RetrievalIndex | None,
                        note_callback: Callable[[dict], None] | None,
                        retrieve_notes: Callable[[dict], tuple[str, bool]] | None) -> list[dict]:
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call.name
        tool_input = tool_call.input or {}
        if tool_name == "fetch_chunks" and index is not None:
            result = fetch_chunks(index, **tool_input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "tool_name": tool_name,
                "content": format_fetch_chunks_result(result),
                "is_error": bool(result.get("error")),
            })
        elif tool_name == "save_note":
            note_data = {
                "title": tool_input.get("title", "Untitled Note"),
                "content": tool_input.get("content", ""),
                "tags": tool_input.get("tags", []),
                "source": "cheri_doctor",
            }
            if note_callback:
                note_callback(note_data)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "tool_name": tool_name,
                "content": f"Note saved: \"{note_data['title']}\"",
                "is_error": False,
            })
        elif tool_name == "retrieve_notes":
            if retrieve_notes:
                try:
                    content, is_error = retrieve_notes(tool_input)
                except Exception as exc:
                    content, is_error = f"Notes retrieval failed: {exc}", True
            else:
                content, is_error = "Notes retrieval unavailable in this chat.", True
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "tool_name": tool_name,
                "content": content,
                "is_error": is_error,
            })
        else:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "tool_name": tool_name,
                "content": f"Unknown or unavailable tool: {tool_name}",
                "is_error": True,
            })
    return tool_results


def _run_agent_loop(messages: list[dict],
                    config: dict,
                    *,
                    index: RetrievalIndex | None = None,
                    progress_cb: Callable[[dict], None] | None = None,
                    note_callback: Callable[[dict], None] | None = None,
                    retrieve_notes: Callable[[dict], tuple[str, bool]] | None = None) -> str:
    payload = messages[0]
    convo = list(payload["messages"])
    adapter = _build_model_adapter(config)
    caps = adapter.capabilities
    print(
        f"  [agent] provider={adapter.provider} model={adapter.model} "
        f"supports_tools={caps.supports_tools}"
    )

    while True:
        tool_iterations = _tool_iteration_count(convo)
        turn = adapter.run_turn(payload, convo)
        response_text = turn.text
        print(
            f"  [agent] turn provider={turn.provider} stop_reason={turn.stop_reason or '-'} "
            f"tool_calls={len(turn.tool_calls)} tool_iterations={tool_iterations}"
        )

        if not turn.tool_calls:
            legacy_context = _legacy_chunk_request_context(response_text, index)
            if legacy_context and tool_iterations < caps.max_tool_iterations:
                _manual_lookup_status(progress_cb)
                convo.append(adapter.assistant_message(turn))
                convo.append({
                    "role": "user",
                    "content": (
                        "Internal manual lookup results:\n\n"
                        "You previously exposed a visible [CHUNK_REQUEST: ...] placeholder. "
                        "That syntax is internal and must never be shown to the owner. "
                        "I completed those manual lookups for you. Use the results below to answer the owner's "
                        "question directly in one clean response. Do not mention chunk requests, internal tools, "
                        "or loading.\n\n"
                        f"{legacy_context}"
                    ),
                })
                continue

            if (
                caps.supports_tools
                and tool_iterations < caps.max_tool_iterations
                and _lookup_intent_without_action(response_text)
            ):
                print("  [agent] blocked lookup-intent narration without tool use")
                convo.append(adapter.assistant_message(turn))
                convo.append({
                    "role": "user",
                    "content": (
                        "You narrated that you were going to do another manual lookup, but you did not actually "
                        "call a tool. Do not describe future actions to the owner. If you need more manual or note "
                        "context, call the appropriate native tool now. Otherwise answer directly from the current evidence."
                    ),
                })
                continue

            return _strip_legacy_chunk_requests(response_text)

        if tool_iterations >= caps.max_tool_iterations:
            fallback = _fallback_answer_from_convo(convo, response_text)
            if fallback:
                return fallback
            return (
                "I couldn't finish the extra manual lookups within the current limit. "
                "Please ask a narrower follow-up."
            )

        _manual_lookup_status(progress_cb)
        convo.append(adapter.assistant_message(turn))
        tool_results = _execute_tool_calls(
            turn.tool_calls,
            index=index,
            note_callback=note_callback,
            retrieve_notes=retrieve_notes,
        )
        # Strip tool_name from results before appending — Claude API rejects
        # extra fields in tool_result blocks.  Gemini doesn't need it either
        # because we set tool_use_id = function name for Gemini tool calls.
        convo_results = [
            {k: v for k, v in r.items() if k != "tool_name"}
            for r in tool_results
        ]
        convo.append({"role": "user", "content": convo_results})


def _call_claude(messages: list[dict],
                 config: dict,
                 index: RetrievalIndex | None = None,
                 progress_cb: Callable[[dict], None] | None = None,
                 note_callback: Callable[[dict], None] | None = None,
                 retrieve_notes: Callable[[dict], tuple[str, bool]] | None = None) -> str:
    """Run the provider-agnostic agent loop and return raw response text."""
    return _run_agent_loop(
        messages,
        config,
        index=index,
        progress_cb=progress_cb,
        note_callback=note_callback,
        retrieve_notes=retrieve_notes,
    )


@lru_cache(maxsize=8)
def _chunk_fig_lookup_cached(lookup_path_str: str) -> dict[str, str]:
    """Build a comprehensive fig ID resolver → asset-style fig ID.

    Handles three citation styles the LLM may use:
      - chunk_id:     fig_7a_p477_0  → fig_p0477_000
      - source_label: 7A-1           → fig_p0477_000  (via page context)
      - caption ref:  Figure 7A-1    → fig_p0477_000
    """
    lookup_path = Path(lookup_path_str)
    if not lookup_path.exists():
        return {}
    lookup = load_json(lookup_path)
    mapping: dict[str, str] = {}
    if isinstance(lookup, dict):
        for cid, chunk in lookup.items():
            if chunk.get("type") == "figure" and chunk.get("figure_refs"):
                asset_id = chunk["figure_refs"][0]
                # Map by chunk_id
                mapping[cid] = asset_id
                # Map by source_label (e.g. "7A-1") — use page to disambiguate
                sl = chunk.get("source_label", "")
                page = chunk.get("page")
                if sl:
                    mapping[sl] = asset_id  # last-one-wins for ambiguous labels
                    if page:
                        mapping[f"{sl}@{page}"] = asset_id
                # Map by "Figure X" caption pattern
                text = chunk.get("text", "")
                fig_num_match = re.match(r"(?:Figure|Fig\.?)\s+(\S+)", text)
                if fig_num_match:
                    fig_label = fig_num_match.group(1)
                    mapping[fig_label] = asset_id
                    if page:
                        mapping[f"{fig_label}@{page}"] = asset_id
    return mapping


def _chunk_fig_lookup(config: dict | None = None) -> dict[str, str]:
    return _chunk_fig_lookup_cached(str(_chunk_lookup_path_from_config(config)))


def _parse_response(raw: str,
                    reranked: list[dict],
                    *,
                    query: str = "",
                    config: dict | None = None) -> ChatResponse:
    """Extract citations from the raw response text."""
    chunk_map = {r["chunk"]["chunk_id"]: r["chunk"] for r in reranked}
    text_citation_lookup: dict[str, dict] = {}
    for row in reranked:
        chunk = row["chunk"]
        page = chunk.get("page")
        for value in (chunk.get("chunk_id"), chunk.get("source_label")):
            key = _citation_lookup_key(value)
            if not key:
                continue
            text_citation_lookup.setdefault(key, chunk)
            if page is not None:
                text_citation_lookup.setdefault(f"{key}@{page}", chunk)

    citations   = []
    figure_refs = []

    seen_chunks: set[str] = set()
    for m in CITATION_RE.finditer(raw):
        page_str = m.group(1)
        id_part  = m.group(2).strip()

        fig_match = re.match(r"fig:\s*(\S+)", id_part, re.IGNORECASE)
        if fig_match:
            fig_id = fig_match.group(1)
            # LLM may cite chunk-style IDs (fig_0a_p5_1), source labels (7A-1),
            # or asset-style IDs (fig_p0005_001). Resolve to asset-style.
            resolved = False
            chunk = chunk_map.get(fig_id)
            if chunk and chunk.get("figure_refs"):
                for real_id in chunk["figure_refs"]:
                    if real_id not in figure_refs:
                        figure_refs.append(real_id)
                resolved = True
            if not resolved:
                # Try global lookup (handles source_labels, caption refs, chunk IDs)
                global_map = _chunk_fig_lookup(config)
                for key in [f"{fig_id}@{page_str}", fig_id]:
                    if key in global_map:
                        real_id = global_map[key]
                        if real_id not in figure_refs:
                            figure_refs.append(real_id)
                        resolved = True
                        break
            if not resolved and fig_id not in figure_refs:
                figure_refs.append(fig_id)
        else:
            page = int(page_str)
            resolved_chunk = _resolve_text_citation(id_part, page, text_citation_lookup)
            seen_key = resolved_chunk["chunk_id"] if resolved_chunk else f"{page}:{id_part}"
            if seen_key in seen_chunks:
                continue
            seen_chunks.add(seen_key)
            chunk = resolved_chunk or chunk_map.get(id_part, {})
            chunk_id = chunk.get("chunk_id", id_part)
            citations.append(Citation(
                chunk_id=chunk_id,
                page=page,
                source_label=chunk.get("source_label", ""),
                section_path=chunk.get("section_path", ""),
                figure_ids=[],
            ))

    # Rewrite chunk-style figure IDs to asset-style in the answer text
    # so the frontend's jumpToFig can match them to the figures array.
    def _resolve_fig_citation(m):
        page = m.group(1)
        fig_id = m.group(2).strip()
        chunk = chunk_map.get(fig_id)
        if chunk and chunk.get("figure_refs"):
            real_id = chunk["figure_refs"][0]
            return f"[p{page} | fig: {real_id}]"
        # Fallback: resolve via global figure map (handles chunk IDs, source labels, caption refs)
        global_map = _chunk_fig_lookup(config)
        # Try page-qualified key first (most precise), then bare key
        for key in [f"{fig_id}@{page}", fig_id]:
            if key in global_map:
                return f"[p{page} | fig: {global_map[key]}]"
        return m.group(0)
    raw = re.sub(
        r"\[p(\d+)\s*\|\s*fig:\s*([^\]]+)\]",
        _resolve_fig_citation,
        raw,
    )

    # Strip text citation markers from the displayed answer (keep figure citations)
    clean = CITATION_STRIP_RE.sub("", raw)
    clean = re.sub(r" {2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = clean.strip()

    citations = _backfill_citations(citations, reranked, query=query)
    evidence_used = [r["chunk"]["chunk_id"] for r in reranked]
    if not clean:
        clean = _fallback_answer_from_evidence(query, reranked)

    return ChatResponse(
        answer=clean,
        citations=citations,
        figure_refs=figure_refs,
        evidence_used=evidence_used,
    )


def _citation_lookup_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = str(value).strip().strip("`'\"")
    cleaned = CITATION_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.strip().strip("`'\"")
    cleaned = cleaned.strip("()")
    cleaned = cleaned.rstrip(".,;:")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def _resolve_text_citation(id_part: str, page: int, text_citation_lookup: dict[str, dict]) -> dict | None:
    key = _citation_lookup_key(id_part)
    if not key:
        return None
    for lookup_key in (f"{key}@{page}", key):
        chunk = text_citation_lookup.get(lookup_key)
        if chunk:
            return chunk
    return None


def _backfill_citations(
    citations: list[Citation],
    reranked: list[dict],
    *,
    query: str = "",
    max_citations: int = 4,
) -> list[Citation]:
    valid = [
        citation for citation in citations
        if citation.source_label or citation.section_path
    ]
    if citations and len(valid) == len(citations):
        return citations

    backfilled: list[Citation] = list(valid)
    seen_ids: set[str] = {citation.chunk_id for citation in valid if citation.chunk_id}
    ordered_rows = sorted(
        reranked,
        key=lambda row: (
            row["chunk"].get("type") != "table",
            row["chunk"].get("type") == "figure",
            -row.get("score", 0.0),
        ),
    )
    if _is_pinout_query(query):
        pinout_rows = [row for row in ordered_rows if _is_pinout_candidate(row["chunk"])]
        if pinout_rows:
            ordered_rows = pinout_rows

    for row in ordered_rows:
        if len(backfilled) >= max_citations:
            break
        chunk = row["chunk"]
        if chunk.get("type") == "figure":
            continue
        chunk_id = chunk.get("chunk_id")
        if not chunk_id or chunk_id in seen_ids:
            continue
        page = chunk.get("page")
        if page is None:
            continue
        backfilled.append(Citation(
            chunk_id=chunk_id,
            page=page,
            source_label=chunk.get("source_label", ""),
            section_path=chunk.get("section_path", ""),
            figure_ids=[],
        ))
        seen_ids.add(chunk_id)

    if backfilled:
        added = len(backfilled) - len(valid)
        if added > 0:
            print(f"  [chat] Backfilled {added} citation(s) from evidence")
    return backfilled


def _fallback_answer_from_evidence(query: str, reranked: list[dict]) -> str:
    if _is_pinout_query(query):
        pinout_fallback = _pinout_fallback_answer(reranked)
        if pinout_fallback:
            print("  [chat] Built fallback answer from pinout evidence")
            return pinout_fallback
    return ""


def _pinout_fallback_answer(reranked: list[dict]) -> str:
    connector_rows: dict[str, list[dict[str, str]]] = {}
    source_labels: dict[str, str] = {}
    for row in reranked:
        chunk = row["chunk"]
        if not _is_pinout_candidate(chunk):
            continue
        connector, rows = _parse_pinout_rows(chunk.get("text", ""))
        if not connector or not rows or connector in connector_rows:
            continue
        connector_rows[connector] = rows
        source_labels[connector] = chunk.get("source_label") or ""

    if not connector_rows:
        return ""

    lines: list[str] = []
    intro_parts = []
    for connector in ("A", "B"):
        if source_labels.get(connector):
            intro_parts.append(f"Connector {connector}: {source_labels[connector]}")
    if intro_parts:
        lines.append("ECM connector identification tables found in the manual: " + "; ".join(intro_parts) + ".")
        lines.append("")

    for connector in ("B", "A"):
        rows = connector_rows.get(connector)
        if not rows:
            continue
        lines.append(f"**ECM Connector {connector} Pinout**")
        lines.append("")
        lines.append('| PIN | CIRCUIT | KEY "ON" | ENG. RUN | WIRE COLOR |')
        lines.append("|-----|---------|----------|----------|------------|")
        for entry in rows:
            pin = entry.get("pin", "")
            circuit = entry.get("circuit", "")
            key_on = entry.get("key_on", "")
            eng_run = entry.get("eng_run", "")
            wire_color = entry.get("wire_color", "")
            lines.append(
                f"| {pin or '-'} | {circuit or '-'} | {key_on or '-'} | {eng_run or '-'} | {wire_color or '-'} |"
            )
        lines.append("")

    return "\n".join(lines).strip()


def _parse_pinout_rows(text: str) -> tuple[str, list[dict[str, str]]]:
    if not text:
        return "", []

    rows: list[dict[str, str]] = []
    connector = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "FUEL INJECTION ECM CONNECTOR IDENTIFICATION":
            continue
        fields = {
            "pin": "",
            "circuit": "",
            "key_on": "",
            "eng_run": "",
            "wire_color": "",
        }
        for part in line.split("|"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            key, value = part.split(":", 1)
            normalized = key.strip().lower()
            value = value.strip()
            if normalized == "pin":
                fields["pin"] = value
            elif normalized == "circuit":
                fields["circuit"] = value
            elif normalized in ('voltage key "on"', 'key "on"'):
                fields["key_on"] = value
            elif normalized in ('voltage eng. run', "eng. run"):
                fields["eng_run"] = value
            elif normalized == "wire color":
                fields["wire_color"] = value
        if not fields["pin"] and not fields["circuit"]:
            continue
        if fields["pin"] and not connector:
            connector = fields["pin"][0].upper()
        rows.append(fields)
    return connector, rows
