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
from pathlib import Path


import anthropic
from sentence_transformers import CrossEncoder
from src.gemini_api import call_gemini

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
    "electrical": ["wiring", "fuse", "relay", "connector", "headlight", "tail light", "horn", "wiper", "turn signal", "gauge", "instrument", "cluster", "speedometer", "tachometer", "warning light"],
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
    r"oil\s+change|change.*oil|engine\s+oil": "engine crankcase capacity oil filter drain plug torque viscosity quarts",
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
    r"\bstarter\b": "starter motor cranking",
    r"\balternator\b": "alternator generator charging",
    r"\bthermostat\b": "thermostat coolant temperature",
    r"\bdistributor\b": "distributor ignition cap rotor",
    r"\bvalves?\b": "valve intake exhaust",
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
    "air conditioning", "air conditioner", "heating", "ventilation",
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
         skip_vision: bool = False) -> ChatResponse:
    """Full RAG chat pipeline."""
    cfg_chat = config.get("chat", {})
    top_k    = cfg_chat.get("top_k_retrieve", 20)
    top_n    = cfg_chat.get("top_n_rerank", 8)

    # For follow-up messages, use LLM to rewrite the query into a standalone
    # search query that captures the full conversational intent.
    if conversation:
        rewritten = _rewrite_query(query, conversation, config)
        search_query = _expand_query(rewritten)
    else:
        search_query = _expand_query(query)

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

    # 3. Collect figure images
    figure_evidence = []
    if not skip_vision:
        figure_evidence = _collect_figures(reranked)

    # 4. Build prompt
    messages = _build_messages(query, reranked, figure_evidence,
                               conversation, config)

    # 5. Call Claude
    raw = _call_claude(messages, config)

    # 6. Parse response
    response = _parse_response(raw, reranked)

    # 7. Ensure ALL evidence figures are returned to the UI
    #    (not just ones Claude cited with [p### | fig: fig_id] format)
    evidence_fig_ids = {f["figure_id"] for f in figure_evidence}
    existing_refs = set(response.figure_refs)
    for fig_id in evidence_fig_ids:
        if fig_id not in existing_refs:
            response.figure_refs.append(fig_id)

    return response


def load_index(config: dict) -> RetrievalIndex:
    """Load the retrieval index from tools/rag_index/."""
    index_dir = PROJECT_ROOT / "tools" / "rag_index"
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


def _collect_figures(reranked: list[dict]) -> list[dict]:
    """Gather figure image data for chunks that reference figures."""
    seen    = set()
    figures = []

    # Load figures lookup
    fig_lookup_path = PROJECT_ROOT / "build" / "figures.jsonl"
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
                    config: dict) -> list[dict]:
    """Build the Claude API messages list."""
    # Load system prompt
    sys_prompt_path = PROJECT_ROOT / "configs" / "chat_system_prompt.txt"
    system_prompt   = sys_prompt_path.read_text(encoding="utf-8")

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

    user_content.append({"type": "text", "text": f"\nQUESTION: {query}"})

    # Build messages list with conversation history
    messages = []
    for turn in conversation:
        messages.append({
            "role":    turn["role"],
            "content": turn["text"],
        })

    messages.append({"role": "user", "content": user_content})

    return [{"system": system_prompt, "messages": messages}]


def _call_claude(messages: list[dict], config: dict) -> str:
    """Call Claude API and return raw text response."""
    cfg_chat = config.get("chat", {})
    model    = cfg_chat.get("model", "claude-sonnet-4-20250514")
    max_tok  = cfg_chat.get("max_tokens", 4096)
    temp     = cfg_chat.get("temperature", 0.1)
    # Allow model selection: if model starts with "gemini", use Gemini
    if model.startswith("gemini"):
        api_key = cfg_chat.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
        return call_gemini(messages, api_key=api_key, model=model)
    # Default: Claude
    client   = anthropic.Anthropic()
    payload  = messages[0]
    response = client.messages.create(
        model=model,
        max_tokens=max_tok,
        temperature=temp,
        system=payload["system"],
        messages=payload["messages"],
    )
    return response.content[0].text


def _parse_response(raw: str, reranked: list[dict]) -> ChatResponse:
    """Extract citations from the raw response text."""
    chunk_map = {r["chunk"]["chunk_id"]: r["chunk"] for r in reranked}

    citations   = []
    figure_refs = []

    seen_chunks: set[str] = set()
    for m in CITATION_RE.finditer(raw):
        page_str = m.group(1)
        id_part  = m.group(2).strip()

        fig_match = re.match(r"fig:\s*(\S+)", id_part, re.IGNORECASE)
        if fig_match:
            fig_id = fig_match.group(1)
            # LLM may cite chunk-style IDs (fig_0a_p5_1) instead of asset-style
            # IDs (fig_p0005_001). Resolve via the chunk's figure_refs field.
            chunk = chunk_map.get(fig_id)
            if chunk and chunk.get("figure_refs"):
                for real_id in chunk["figure_refs"]:
                    if real_id not in figure_refs:
                        figure_refs.append(real_id)
            elif fig_id not in figure_refs:
                figure_refs.append(fig_id)
        else:
            chunk_id = id_part
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            chunk = chunk_map.get(chunk_id, {})
            citations.append(Citation(
                chunk_id=chunk_id,
                page=int(page_str),
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

    evidence_used = [r["chunk"]["chunk_id"] for r in reranked]

    return ChatResponse(
        answer=clean,
        citations=citations,
        figure_refs=figure_refs,
        evidence_used=evidence_used,
    )
