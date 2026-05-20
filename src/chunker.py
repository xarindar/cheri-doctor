"""Stage F: Retrieval Corpus Build.

Walks document.json blocks and produces:
- chunks.jsonl: one retrieval chunk per record
- figures.jsonl: one figure metadata record per figure block

Chunking rules (per AGENT.md):
- One chunk per ordered_list (procedures are never split)
- One chunk per caution / warning / note
- Paragraphs merged into ~200-600 token chunks at paragraph boundaries
- Legends as key/value chunks
- Tables as retrieval_text chunks with CSV + image asset refs
- TOC entries as a single toc chunk per page
"""

import csv
import re
from src.models import ChunkRecord


CHARS_PER_TOKEN  = 4
FIGURE_REF_RE    = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)
FIGURE_MENTION_RE = re.compile(r"\b(?:figures?|fig\.?|diagrams?|schematics?|illustrations?|charts?)\b", re.IGNORECASE)
TOC_NOISE_RE    = re.compile(r"\.{4,}|0A-\d|\b0B-\d|\b1[A-Z]-\d")
SECTION_CODE_RE = re.compile(r"\b\d+[A-Z]-\d+\b")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z])|\n+")
# Matches section refs like "6A1-33", "10-1-5", "7A-4", "C1-5", "B-3", "A-18"
INDEX_REF_RE    = re.compile(r"^[A-Z\d][A-Z\d]*-\d[\d-]*$")
# Running page header: "8-30 BODY AND CHASSIS ELECTRICAL" or "6E2-2 DRIVEABILITY"
PAGE_HEADER_RE  = re.compile(r"^\d+[A-Z]*\d*-\d+\s+[A-Z][A-Z &/]+$")
# Vision-style: "ENGINE MECHANICAL 6A1-3", "BODY AND CHASSIS ELECTRICAL 8-17"
PAGE_HEADER_RE2 = re.compile(r"^[A-Z][A-Z &/]{5,}\s+\d+[A-Z]*\d*-\d+\s*$")
# Section header: "SECTION 6E - DRIVEABILITY AND EMISSIONS" or "SECTION 6D5 ENGINE WIRING"
SECTION_HEADER_RE = re.compile(r"^SECTION\s+\d+[A-Z]*\d*\b", re.IGNORECASE)
# Standalone page labels: "9A-4", "6B-12", "10-7-2"
PAGE_LABEL_RE   = re.compile(r"^\d+[A-Z]*\d*-\d+(?:-\d+)?$")
PIN_LABEL_RE    = re.compile(r"\b[ABC]\d{1,2}\b")

INFO_TYPE_PATTERNS = (
    ("connector_face", re.compile(r"\b(?:connector faces?|terminal end view|end view|face layout)\b", re.IGNORECASE)),
    ("pinout", re.compile(r"\b(?:pinout|connector identification|pin:|cavity|terminal)\b", re.IGNORECASE)),
    ("diagram", re.compile(r"\b(?:figure|diagram|schematic|illustration|view)\b", re.IGNORECASE)),
    ("spec", re.compile(r"\b(?:spec(?:ification)?s?|voltage|resistance|torque|pressure|clearance|capacity|dimension)\b", re.IGNORECASE)),
    ("location", re.compile(r"\b(?:location|located|mounting location|component location|placement)\b", re.IGNORECASE)),
    ("diagnostic", re.compile(r"\b(?:diagnos(?:is|tic)|symptom|condition|cause|correction|trouble code|code\b)\b", re.IGNORECASE)),
    ("wiring", re.compile(r"\b(?:wire|wiring|circuit|harness|ground|relay|switch terminal)\b", re.IGNORECASE)),
    ("cross_reference", re.compile(r"\b(?:table of contents|refer to|see section|section \d|cross[ -]?reference|index)\b", re.IGNORECASE)),
)

INFO_TYPE_SYNONYMS = {
    "connector_face": "connector face terminal end view layout",
    "pinout": "pinout pin terminal cavity connector identification",
    "diagram": "figure diagram schematic illustration",
    "spec": "spec specification voltage resistance torque pressure capacity",
    "location": "location located component position mounting",
    "diagnostic": "diagnostic diagnosis symptom condition cause correction",
    "procedure": "procedure steps removal installation inspection adjustment",
    "wiring": "wiring wire circuit harness terminal ground power",
    "cross_reference": "cross reference mapping refer index table of contents",
}

ENTITY_PATTERNS = (
    ("ecm", re.compile(r"\b(?:ecm|engine control module)\b", re.IGNORECASE)),
    ("ecu", re.compile(r"\b(?:ecu|electronic control unit)\b", re.IGNORECASE)),
    ("aldl", re.compile(r"\b(?:aldl|assembly line diagnostic link|diagnosis switch terminal)\b", re.IGNORECASE)),
    ("cts", re.compile(r"\b(?:cts|coolant temperature sensor)\b", re.IGNORECASE)),
    ("tps", re.compile(r"\b(?:tps|throttle position sensor)\b", re.IGNORECASE)),
    ("map", re.compile(r"\b(?:map|manifold absolute pressure)\b", re.IGNORECASE)),
    ("o2", re.compile(r"\b(?:o2|oxygen sensor)\b", re.IGNORECASE)),
    ("mat", re.compile(r"\b(?:mat|manifold air temperature)\b", re.IGNORECASE)),
    ("isc", re.compile(r"\b(?:isc|idle speed control|isc solenoid)\b", re.IGNORECASE)),
    ("egr", re.compile(r"\b(?:egr|exhaust gas recirculation)\b", re.IGNORECASE)),
    ("vss", re.compile(r"\b(?:vss|vehicle speed sensor)\b", re.IGNORECASE)),
    ("sir", re.compile(r"\b(?:sir|airbag|inflatable restraint|supplemental restraint)\b", re.IGNORECASE)),
    ("fuel pump", re.compile(r"\bfuel pump\b", re.IGNORECASE)),
    ("connector a", re.compile(r"\b(?:ecm\s+)?connector\s+a\b", re.IGNORECASE)),
    ("connector b", re.compile(r"\b(?:ecm\s+)?connector\s+b\b", re.IGNORECASE)),
    ("connector c1", re.compile(r"\bconnector\s+c1\b", re.IGNORECASE)),
    ("connector c2", re.compile(r"\bconnector\s+c2\b", re.IGNORECASE)),
)

BODY_STYLE_PATTERNS = (
    ("convertible", re.compile(r"\bconvertible\b", re.IGNORECASE)),
    ("hatchback", re.compile(r"\bhatchback\b", re.IGNORECASE)),
    ("sedan", re.compile(r"\bsedan\b", re.IGNORECASE)),
    ("wagon", re.compile(r"\bwagon\b", re.IGNORECASE)),
    ("3-door", re.compile(r"\b3-door\b", re.IGNORECASE)),
    ("5-door", re.compile(r"\b5-door\b", re.IGNORECASE)),
)

TRIM_VARIANT_PATTERNS = (
    ("lsi", re.compile(r"\blsi\b", re.IGNORECASE)),
    ("xfi", re.compile(r"\bxfi\b", re.IGNORECASE)),
)

SIR_FALSE_RE = re.compile(r"\b(?:without|less)\s+(?:sir|airbag)|\bnon[- ]airbag\b", re.IGNORECASE)

SECTION_SYSTEM_MAP = {
    "0A": "general_info",
    "0B": "maintenance",
    "1A": "ac",               # Heating and Ventilation
    "1B": "ac",               # Air Conditioning
    "1D": "ac",               # AC Compressor Overhaul
    "2":  "body",             # (section 2 content)
    "3":  "steering",         # Steering/Suspension Diagnosis
    "3A": "steering",         # Wheel Alignment
    "3B": "steering",         # Manual Rack and Pinion
    "3C": "front_suspension", # Front Suspension
    "3D": "rear_suspension",  # Rear Suspension
    "3E": "wheels_tires",     # Wheels and Tires
    "3F": "steering",         # Steering Wheel and Column
    "4D": "front_drive_axle", # Drive Axle
    "5":  "brakes",           # Brakes (disc, drum, master cyl, parking)
    "6":  "engine",           # Engine general
    "6A1": "engine",          # Engine Mechanical
    "6B": "cooling",          # Engine Cooling
    "6C": "fuel_system",      # Fuel System
    "6D": "engine_electrical",
    "6D1": "battery",
    "6D2": "engine_electrical",  # Cranking System
    "6D3": "engine_electrical",  # Charging System
    "6D4": "engine_electrical",  # Ignition System
    "6D5": "engine_electrical",  # Engine Wiring
    "6E": "emission_controls",   # Driveability and Emissions
    "6E2": "fuel_injection",
    "6F": "exhaust",
    "7A": "automatic_transaxle",
    "7A1": "automatic_transaxle",  # Unit Repair
    "7B": "manual_transaxle",
    "7C": "clutch",
    "8":  "electrical",        # Body Electrical Systems
    "8A": "electrical",
    "9":  "accessories",       # Radio, etc.
    "9A": "accessories",
    "10":   "body",            # Body Service
    "10-1": "body",
    "10-2": "body",            # Stationary Glass
    "10-3": "body",            # Underbody
    "10-4": "body",            # Front End
    "10-5": "body",            # Doors
    "10-6": "body",            # Body Rear Quarters
    "10-7": "body",            # Body Rear End
    "10-8": "body",            # Roof
    "10-9": "body",            # Seats
    "10-10": "body",           # Safety Belts
    "10-11": "body",           # Body Electrical Systems
}

def _get_system(section_code: str | None, section_path: str) -> str:
    """Map section code to a standardized system name."""
    if section_code:
        # Extract prefix like '1B' from '1B-7'
        prefix = section_code.split('-')[0]
        if prefix in SECTION_SYSTEM_MAP:
            return SECTION_SYSTEM_MAP[prefix]

    # Fallback: map section_path category to known system names
    if section_path:
        path_upper = section_path.upper()
        CATEGORY_MAP = {
            "GENERAL INFORMATION": "general_info",
            "HEATING AND AIR CONDITIONING": "ac",
            "STEERING": "steering",
            "SUSPENSION": "steering",
            "ENGINE": "engine",
            "TRANSAXLE": "automatic_transaxle",
            "BODY SERVICE": "body",
            "ACCESSORIES": "accessories",
        }
        for keyword, system in CATEGORY_MAP.items():
            if keyword in path_upper:
                return system

    return "general_info"

def _get_engine_variant(section_code: str | None, section_path: str) -> str:
    """Determine if content applies to G10, G13, or both."""
    path = section_path.upper()
    if "G13" in path or "1.3L" in path:
        return "G13"
    if "G10" in path or "1.0L" in path:
        return "G10"
    
    # Default for this manual is G10 unless specified
    return "G10"

def build_chunks(document: dict, config: dict, source_doc: str = "main") -> tuple[list[dict], list[dict]]:
    """Produce chunks.jsonl and figures.jsonl records from document.json.

    Returns:
        (chunks, figures) — both as lists of plain dicts (JSON-serializable).
    """
    cfg_chunk = config.get("chunking", {})
    min_tok   = cfg_chunk.get("min_tokens", 200)
    max_tok   = cfg_chunk.get("max_tokens", 600)
    doc_id    = document["doc_id"]

    chunks:  list[dict] = []
    figures: list[dict] = []
    last_heading: str = ""  # Track most recent heading across pages for context
    last_table_condition: str = ""  # Track last condition from diagnostic tables across pages

    for page in document["pages"]:
        pn           = page["page_num"]
        section_code = page.get("section_code")
        source_label = page.get("source_label")
        section_path = page.get("section_path", "")

        # Priority 8: Chapter 10 subsection promotion
        # Promote source_label (e.g. "10-5-1") to section_code if it's a bare "10"
        if section_code == "10" and source_label and source_label.startswith("10-"):
            section_code = source_label
            if section_path in ("", "10"):
                section_path = source_label

        last_caption: str = "" # Track most recent caption ON THIS PAGE

        # Counter for deterministic chunk IDs per page
        seq_counter: dict[str, int] = {}

        # Pre-collect captions and map them to the figure they describe
        caption_map: dict[str, str] = {
            b["parent_figure_id"]: (b.get("text") or "")
            for b in page["blocks"]
            if b.get("type") == "caption" and b.get("parent_figure_id")
        }

        # Pre-collect real figure IDs on this page for cross-referencing
        page_figure_ids: list[str] = [
            b["figure_id"] for b in page["blocks"]
            if b.get("type") == "figure" and b.get("figure_id")
        ]

        # Build figure-number-to-ID map: "Figure 22" in text -> fig_p0045_001
        # Parse the number from each figure's caption (e.g. "Figure 22 Evaporator...")
        fig_num_to_id: dict[str, str] = {}
        for b in page["blocks"]:
            if b.get("type") == "figure" and b.get("figure_id"):
                cap = caption_map.get(b["figure_id"], b.get("caption_text") or "")
                m = FIGURE_REF_RE.search(cap)
                if m:
                    fig_num_to_id[m.group(1)] = b["figure_id"]

        # Build block_id → figure_id map for Gemini-extracted pages.
        # Gemini sets associated_figure_ids using block_ids (e.g. "block_1"),
        # not figure_ids. We resolve them here so procedure chunks can link
        # to the correct figures.
        block_id_to_fig_id: dict[str, str] = {
            b["block_id"]: b["figure_id"]
            for b in page["blocks"]
            if b.get("type") == "figure" and b.get("figure_id") and b.get("block_id")
        }

        # Collect paragraph blocks for batch chunking
        para_buffer: list[dict] = []
        # Buffer for merging adjacent small tables into one chunk
        table_buffer: list[dict] = []

        for block in page["blocks"]:
            btype = block.get("type")
            text  = block.get("text") or ""

            # Normalize Vision block types to chunker types.
            # Vision uses "procedure"/"inspection" (chunker: "ordered_list"),
            # "header" (chunker: "heading"), "notice"/"important" (chunker: "caution").
            if btype in ("procedure", "inspection"):
                btype = "ordered_list"
            elif btype == "header":
                btype = "heading"
            elif btype in ("notice", "important"):
                btype = "caution"

            # Phase 2.2: Single-step "procedures" are really checklist items
            # (e.g. "Check boots for breakage"). Convert to paragraph so they
            # merge with adjacent text and inherit heading context, instead of
            # becoming decontextualized orphan chunks.
            if btype == "ordered_list":
                steps = block.get("steps") or []
                if len(steps) < 2 and not block.get("continues_to_next_page") and not block.get("continues_from_previous_page"):
                    list_text = " ".join(str(s) for s in steps).strip()
                    if list_text:
                        block = {**block, "text": list_text, "type": "paragraph"}
                        btype = "paragraph"

            # Symptom/checklist misclassification: ordered_lists where most
            # "steps" are very short (< 8 words) and title contains SYMPTOM
            # or DIAGNOSTIC are actually checklists, not procedures.
            if btype == "ordered_list":
                steps = block.get("steps") or []
                block_title = (block.get("title") or block.get("text") or "").upper()
                if steps and len(steps) >= 2:
                    short_steps = sum(1 for s in steps if len(str(s).split()) < 8)
                    is_symptom_list = (
                        short_steps > len(steps) * 0.7
                        and re.search(r"\b(SYMPTOM|DIAGNOSTIC|CHECKLIST)\b", block_title)
                    )
                    if is_symptom_list:
                        btype = "paragraph"

            # Drop TOC-like noise paragraphs
            if btype == "paragraph":
                if TOC_NOISE_RE.search(text):
                    cleaned = _clean_toc_noise(text)
                    if not cleaned or len(cleaned.split()) < 5:
                        continue
                    block = {**block, "text": cleaned}
                    text = cleaned

            # Flush table buffer on any non-table block
            if btype != "table" and table_buffer:
                merged = _merge_table_blocks(table_buffer)
                ctx = last_heading
                if not ctx or len(ctx) < 10:
                    ctx = last_caption or source_label or ""
                if ctx and re.match(r"^Fig(?:ure)?[\s.\d]", ctx, re.IGNORECASE):
                    ctx = source_label or ""
                new_chunks, cond = _chunk_table(merged, doc_id, pn, section_code,
                                         source_label, section_path, seq_counter,
                                         title=ctx, prev_condition=last_table_condition)
                chunks.extend(new_chunks)
                if cond:
                    last_table_condition = cond
                table_buffer = []

            # Flush paragraph buffer on any non-paragraph block
            if btype != "paragraph" and para_buffer:
                new_chunks = _chunk_paragraphs(
                    para_buffer, doc_id, pn, section_code,
                    source_label, section_path, seq_counter,
                    min_tok, max_tok, page_figure_ids, fig_num_to_id
                )
                chunks.extend(new_chunks)
                para_buffer = []

            if btype == "ordered_list":
                # Context: prefer the block's own text (procedure name) when it
                # differs from the steps, then last heading, then fallback.
                block_title = (block.get("title") or "").strip()
                block_text = (block.get("text") or "").strip()
                steps = block.get("steps") or []

                # If block.text looks like a procedure name (short, distinct
                # from the first step), treat it as the title and update
                # last_heading so subsequent blocks get the right context.
                if (not block_title
                    and block_text
                    and len(block_text) < 80
                    and steps
                    and block_text != str(steps[0]).strip()):
                    block_title = block_text

                if block_title:
                    ctx = block_title
                    last_heading = block_title
                else:
                    ctx = last_heading
                    # Only apply short-title fallback when using inherited heading,
                    # not the block's own title (e.g. "Adjust" is valid).
                    if not ctx or len(ctx) < 10:
                        ctx = last_caption or ""
                # Never use figure captions as procedure titles
                if ctx and re.match(r"^Fig(?:ure)?[\s.\d]", ctx, re.IGNORECASE):
                    ctx = source_label or ""

                chunk = _chunk_procedure(block, doc_id, pn, section_code,
                                         source_label, section_path, seq_counter,
                                         page_figure_ids, fig_num_to_id, title=ctx,
                                         block_id_to_fig_id=block_id_to_fig_id,
                                         caption_map=caption_map)
                if chunk:
                    chunks.append(chunk)

            elif btype in ("caution", "warning", "note"):
                chunk = _chunk_advisory(block, doc_id, pn, section_code,
                                        source_label, section_path, seq_counter)
                chunks.append(chunk)

            elif btype == "legend":
                chunk = _chunk_legend(block, doc_id, pn, section_code,
                                      source_label, section_path, seq_counter)
                chunks.append(chunk)

            elif btype == "table":
                rows = block.get("rows") or []
                # Small tables (< 10 rows) get buffered for merging with
                # adjacent small tables. Large tables or tables with their
                # own title are chunked immediately.
                has_own_title = bool((block.get("title") or "").strip())
                if len(rows) < 10 and not has_own_title:
                    table_buffer.append(block)
                else:
                    # Flush any buffered small tables first
                    if table_buffer:
                        merged = _merge_table_blocks(table_buffer)
                        ctx = last_heading
                        if not ctx or len(ctx) < 10:
                            ctx = last_caption or source_label or ""
                        if ctx and re.match(r"^Fig(?:ure)?[\s.\d]", ctx, re.IGNORECASE):
                            ctx = source_label or ""
                        new_chunks, cond = _chunk_table(merged, doc_id, pn, section_code,
                                                 source_label, section_path, seq_counter,
                                                 title=ctx, prev_condition=last_table_condition)
                        chunks.extend(new_chunks)
                        if cond:
                            last_table_condition = cond
                        table_buffer = []

                    # Chunk the large table
                    ctx = last_heading
                    if not ctx or len(ctx) < 10:
                        ctx = last_caption or source_label or ""
                    if ctx and re.match(r"^Fig(?:ure)?[\s.\d]", ctx, re.IGNORECASE):
                        ctx = source_label or ""
                    new_chunks, cond = _chunk_table(block, doc_id, pn, section_code,
                                         source_label, section_path, seq_counter,
                                         title=ctx, prev_condition=last_table_condition)
                    chunks.extend(new_chunks)
                    if cond:
                        last_table_condition = cond



            elif btype == "figure":
                # Export to figures.jsonl
                fig = _build_figure_record(block, doc_id, pn,
                                           section_code, source_label, section_path,
                                           caption_map,
                                           fallback_caption=last_heading or last_caption or source_label or "")
                figures.append(fig)

                # Update page context
                cap = fig.get("caption_text") or ""
                if cap:
                    last_caption = cap

                # Create a searchable chunk for any figure with caption or vision desc
                # Use actual caption (from caption_map or block), NOT the fallback
                # heading assigned in _build_figure_record — fallbacks like
                # "DISASSEMBLY" are page headers, not figure-specific content.
                fig_id      = block.get("figure_id", "")
                cap         = caption_map.get(fig_id, block.get("caption_text") or "") or ""
                vision_desc = block.get("vision_description") or ""
                legend      = block.get("legend_items")
                legend_text = ""
                if isinstance(legend, dict):
                    legend_text = "\n".join(f"{k}: {v}" for k, v in legend.items() if v)
                elif isinstance(legend, list):
                    legend_text = "\n".join(
                        f"{it.get('key', '')}: {it.get('value', '')}"
                        for it in legend if isinstance(it, dict)
                    )

                fig_text = "\n".join(
                    p for p in [cap, vision_desc, legend_text] if p.strip()
                )
                if fig_text.strip():
                    chunks.append(_make_chunk(
                        chunk_id=_gen_id("fig", section_code, pn, seq_counter),
                        doc_id=doc_id, page=pn,
                        block_ids=[f"p{pn}_{block['block_id']}"],
                        bbox=block.get("bbox"),
                        ctype="figure",
                        section_code=section_code,
                        source_label=source_label,
                        section_path=section_path,
                        text=fig_text,
                        figure_refs=[fig_id] if fig_id else [],
                        asset_refs=[block["asset_path"]] if block.get("asset_path") else [],
                    ))

            elif btype == "paragraph":
                para_buffer.append(block)

            elif btype == "heading":
                # Short headings go into the next paragraph chunk as context;
                # long standalone headings get their own small chunk
                text = (block.get("text") or "").strip()
                if text:
                    # Don't use page-level headers (e.g. "10-7-2 BODY REAR END",
                    # "6D3-8 CHARGING SYSTEM") or running section headers
                    # ("SECTION 6E - DRIVEABILITY...") as context titles.
                    if (not re.match(r"^[\dA-Z]+-[\dA-Z]+-?\d*\s", text)
                            and not SECTION_HEADER_RE.match(text)
                            and not PAGE_HEADER_RE.match(text)
                            and not PAGE_HEADER_RE2.match(text)):
                        last_heading = text  # Carry forward for procedure title context
                if len(text) > 60:
                    chunk = _make_chunk(
                        chunk_id=_gen_id("para", section_code, pn, seq_counter),
                        doc_id=doc_id, page=pn,
                        block_ids=[f"p{pn}_{block['block_id']}"],
                        bbox=block.get("bbox"),
                        ctype="paragraph",
                        section_code=section_code,
                        source_label=source_label,
                        section_path=section_path,
                        text=text,
                    )
                    chunks.append(chunk)

            elif btype == "toc_entry":
                para_buffer.append(block)

        # Flush any remaining buffered tables
        if table_buffer:
            merged = _merge_table_blocks(table_buffer)
            ctx = last_heading
            if not ctx or len(ctx) < 10:
                ctx = last_caption or source_label or ""
            if ctx and re.match(r"^Fig(?:ure)?[\s.\d]", ctx, re.IGNORECASE):
                ctx = source_label or ""
            new_chunks, cond = _chunk_table(merged, doc_id, pn, section_code,
                                     source_label, section_path, seq_counter,
                                     title=ctx, prev_condition=last_table_condition)
            chunks.extend(new_chunks)
            if cond:
                last_table_condition = cond

        # Flush any remaining paragraphs
        if para_buffer:
            new_chunks = _chunk_paragraphs(
                para_buffer, doc_id, pn, section_code,
                source_label, section_path, seq_counter,
                min_tok, max_tok, page_figure_ids, fig_num_to_id
            )
            chunks.extend(new_chunks)

    # Stitch same-page procedure continuations
    chunks = _stitch_procedures(chunks)

    # Filter out noise chunks with too little text content
    MIN_CHUNK_TEXT = 10  # chars — reject single-word and empty chunks
    before = len(chunks)
    chunks = [c for c in chunks if len((c.get("text") or "").strip()) >= MIN_CHUNK_TEXT]
    dropped = before - len(chunks)
    if dropped:
        print(f"    Dropped {dropped} chunks with < {MIN_CHUNK_TEXT} chars of text")

    # Stamp every chunk with the source document identifier.
    # Prefix supplement chunk IDs with "sup_" so they never collide with
    # main-manual IDs when both are loaded into the same index lookup.
    for c in chunks:
        c["source_doc"] = source_doc
        if source_doc == "supplement" and not c["chunk_id"].startswith("sup_"):
            c["chunk_id"] = "sup_" + c["chunk_id"]

    _attach_page_artifact_links(chunks, figures)
    return chunks, figures


_CONTINUATION_STRIP_RE = re.compile(
    r"\s*\((?:continued|cont\.?)\)\s*$", re.IGNORECASE
)
_FIGURE_SUFFIX_RE = re.compile(
    r"\s*\(Figure\s+[A-Z0-9-]+\)\s*$", re.IGNORECASE
)


def _normalize_proc_heading(heading: str) -> str:
    """Normalize a procedure heading for comparison by stripping
    trailing '(continued)', '(Cont.)', '(Figure C2-23)', etc."""
    h = _CONTINUATION_STRIP_RE.sub("", heading)
    h = re.sub(r"\s*(?:continued|cont\.?)\s*$", "", h, flags=re.IGNORECASE)
    h = _FIGURE_SUFFIX_RE.sub("", h)
    return h.strip()


def _stitch_procedures(chunks: list[dict]) -> list[dict]:
    """Merge consecutive procedure chunks when they share a heading and
    have continuous step numbering.

    Handles:
      - Same-page continuations (steps 1-4 + steps 5-8 on same page)
      - Cross-page continuations (steps 1-6 on page N, steps 7-10 on page N+1)
      - Heading normalization: '(continued)', '(Cont.)', '(Figure X)' stripped

    Leaves alone:
      - Sub-procedures with different headings (1D compressor overhaul sub-steps)
      - Non-adjacent pages
    """
    if not chunks:
        return chunks

    result = []
    stitched = 0

    for chunk in chunks:
        if (chunk.get("type") != "procedure"
                or (chunk.get("starting_step") or 1) <= 1
                or not result):
            result.append(chunk)
            continue

        # Walk backwards to find the most recent procedure chunk
        prev = None
        for r in reversed(result):
            if r.get("type") == "procedure":
                prev = r
                break

        if prev is None:
            result.append(chunk)
            continue

        # Must be same page or adjacent page
        prev_page = prev.get("page")
        curr_page = chunk.get("page")
        if curr_page not in (prev_page, prev_page + 1) if prev_page is not None else True:
            result.append(chunk)
            continue

        # Compare normalized headings
        prev_heading = _normalize_proc_heading(prev["text"].split("\n")[0].strip())
        curr_heading = _normalize_proc_heading(chunk["text"].split("\n")[0].strip())

        if prev_heading != curr_heading or not prev_heading:
            result.append(chunk)
            continue

        # Check step continuity: curr must start near where prev left off.
        # Allow a gap of up to 3 steps (OCR may skip a number) but reject
        # cases where numbering restarts or goes backwards.
        prev_steps = prev.get("steps") or []
        curr_start = chunk.get("starting_step") or 1

        # Parse actual last step number from prev's last step string,
        # not from starting_step + len (which drifts when there are gaps)
        prev_last = 0
        if prev_steps:
            last_step_str = str(prev_steps[-1]).strip()
            m = re.match(r"^(\d+)[\.\)]", last_step_str)
            if m:
                prev_last = int(m.group(1))
            else:
                # Fallback to computed value
                prev_last = (prev.get("starting_step") or 1) + len(prev_steps) - 1

        gap = curr_start - prev_last
        if gap < 0 or gap > 3:
            result.append(chunk)
            continue

        # Trial merge: build merged text, then validate before committing
        curr_steps = chunk.get("steps") or []
        curr_text_lines = chunk["text"].split("\n")
        step_lines = [l for l in curr_text_lines[1:] if l.strip()]

        trial_text = prev["text"].rstrip()
        if step_lines:
            trial_text += "\n" + "\n".join(step_lines)
        trial_text = trial_text.replace("[CONTINUED ON NEXT PAGE]", "")
        trial_text = trial_text.replace("[CONTINUED FROM PREVIOUS PAGE]", "")
        trial_text = trial_text.strip()

        # Post-merge validation: check step numbers don't regress.
        # Only match line-initial step numbers (not numbers inside text).
        merged_step_nums = []
        for line in trial_text.split("\n"):
            m = re.match(r"^(\d+)\.\s", line.strip())
            if m:
                merged_step_nums.append(int(m.group(1)))
        reject = False
        for si in range(1, len(merged_step_nums)):
            if merged_step_nums[si] < merged_step_nums[si - 1]:
                reject = True
                break
        if reject:
            result.append(chunk)
            continue

        # Commit the merge
        prev["steps"] = (prev.get("steps") or []) + curr_steps
        prev["text"] = trial_text

        # Merge block_ids (deduplicate page-scoped IDs)
        for bid in chunk.get("block_ids") or []:
            if bid not in prev.get("block_ids", []):
                prev.setdefault("block_ids", []).append(bid)

        # Merge figure_refs (deduplicate)
        for fref in chunk.get("figure_refs", []):
            if fref not in (prev.get("figure_refs") or []):
                prev.setdefault("figure_refs", []).append(fref)

        for field in ("info_types", "entities", "body_styles", "trim_variants", "asset_refs"):
            existing = prev.setdefault(field, [])
            for value in chunk.get(field, []) or []:
                if value not in existing:
                    existing.append(value)
        if prev.get("sir_equipped") is None and chunk.get("sir_equipped") is not None:
            prev["sir_equipped"] = chunk["sir_equipped"]

        # Update token count
        prev["token_count"] = len(prev["text"]) // CHARS_PER_TOKEN

        stitched += 1

    if stitched:
        print(f"    Stitched {stitched} procedure continuation(s) into parent chunks")

    return result


# ── Chunk builders ─────────────────────────────────────────────────────────

def _chunk_procedure(block, doc_id, pn, section_code, source_label,
                     section_path, seq_counter,
                     page_figure_ids: list | None = None,
                     fig_num_to_id: dict | None = None,
                     title: str = "",
                     block_id_to_fig_id: dict | None = None,
                     caption_map: dict | None = None) -> dict | None:
    steps = block.get("steps") or []
    if not steps:
        return None  # A procedure with zero steps is just a header, not a chunk
    if len(steps) < 2 and not block.get("continues_to_next_page") and not block.get("continues_from_previous_page"):
        # Phase 2.4: Prevent orphaned fragments/header-only chunks
        return None

    # Use block's own title (vision pages) or the last heading seen (OCR pages)
    resolved_title = (block.get("title") or "").strip() or title
    
    # Keyword heuristic for OCR-path procedures lacking a procedure_type
    proc_type = block.get("procedure_type")
    if not proc_type:
        text_for_type = (resolved_title + " " + " ".join(str(s) for s in steps)).lower()
        if any(w in text_for_type for w in ["removal", "remove", "detaching"]):
            proc_type = "removal"
        elif any(w in text_for_type for w in ["installation", "install", "attaching"]):
            proc_type = "installation"
        elif any(w in text_for_type for w in ["inspection", "inspect", "check"]):
            proc_type = "inspection"
        elif any(w in text_for_type for w in ["adjustment", "adjust", "setting"]):
            proc_type = "adjustment"
        elif "disassembly" in text_for_type:
            proc_type = "disassembly"
        elif "assembly" in text_for_type:
            proc_type = "assembly"

    text_parts = []
    if resolved_title:
        text_parts.append(resolved_title)

    starting_step = 1
    _numbered_re = re.compile(r"^\d+[\.\)]")

    # Detect starting step from the FIRST numbered step (not just steps[0]).
    # Steps may have unnumbered preconditions before the real numbered steps.
    first_numbered_idx = None
    for idx, s in enumerate(steps):
        m = re.match(r"^(\d+)[\.\)]", str(s).strip())
        if m:
            first_numbered_idx = idx
            if int(m.group(1)) > 1:
                starting_step = int(m.group(1))
            break

    # Check whether steps are a mix of numbered and unnumbered.
    # If so, unnumbered items before the first numbered step are preconditions.
    has_any_numbered = first_numbered_idx is not None

    # Add continuation marker if explicitly flagged but no numbered start detected
    if block.get("continues_from_previous_page") and starting_step == 1:
        text_parts.append("[CONTINUED FROM PREVIOUS PAGE]")

    # Preserve original step numbering or add it starting from starting_step.
    # Unnumbered items before the first numbered step become bullet points
    # to avoid colliding with existing step numbers (e.g. "Clean: ..." + "1. Wiper arm").
    for i, s in enumerate(steps):
        s_str = str(s).strip()
        if _numbered_re.match(s_str):
            # If it already has a number like "1. 5. CHECK", clean it
            s_str = re.sub(r"^\d+[\.\)]\s+(\d+[\.\)])", r"\1", s_str)
            text_parts.append(s_str)
        elif has_any_numbered and i < first_numbered_idx:
            # Unnumbered precondition before first numbered step
            text_parts.append(f"- {s_str}")
        else:
            text_parts.append(f"{starting_step + i}. {s_str}")

    if block.get("continues_to_next_page"):
        text_parts.append("[CONTINUED ON NEXT PAGE]")

    text = "\n".join(text_parts)
    
    # Resolve figure refs from two sources:
    # 1. Text-based: "Figure 3" mentions in step text
    # 2. Gemini associated_figure_ids: block IDs Gemini linked to this procedure
    text_fig_refs = _resolve_fig_refs(text, page_figure_ids or [], fig_num_to_id)
    assoc_ids = block.get("associated_figure_ids") or []
    assoc_fig_refs = [
        block_id_to_fig_id[bid] for bid in assoc_ids
        if block_id_to_fig_id and bid in block_id_to_fig_id
    ]
    # Merge, preserving order, deduplicated
    seen: set[str] = set()
    merged_fig_refs: list[str] = []
    for fid in (assoc_fig_refs + text_fig_refs):
        if fid not in seen:
            seen.add(fid)
            merged_fig_refs.append(fid)

    # Inline compact figure captions for explicitly resolved refs.
    # Only appended when the caption is substantive (>15 chars) to avoid
    # cluttering chunks with fallback headings like "DISASSEMBLY".
    if caption_map and merged_fig_refs:
        inline_caps: list[str] = []
        for fig_id in merged_fig_refs:
            cap = (caption_map.get(fig_id) or "").strip()
            if len(cap) > 15:
                m = FIGURE_REF_RE.search(cap)
                label = f"Figure {m.group(1)}" if m else fig_id
                inline_caps.append(f"[{label}: {cap}]")
        if inline_caps:
            text = text + "\n" + "\n".join(inline_caps)

    return _make_chunk(
        chunk_id=_gen_id("proc", section_code, pn, seq_counter),
        doc_id=doc_id, page=pn,
        block_ids=[f"p{pn}_{block['block_id']}"],
        bbox=block.get("bbox"),
        ctype="procedure",
        procedure_type=proc_type,
        section_code=section_code,
        source_label=source_label,
        section_path=section_path,
        text=text,
        steps=steps,
        starting_step=starting_step,
        figure_refs=merged_fig_refs,
    )


def _chunk_advisory(block, doc_id, pn, section_code, source_label,
                    section_path, seq_counter) -> dict:
    btype = block["type"]  # caution / warning / note
    text  = (block.get("text") or "").strip()
    prefix_map = {"caution": "CAUTION", "warning": "WARNING", "note": "NOTE",
                  "notice": "NOTICE", "important": "IMPORTANT"}
    prefix = prefix_map.get(btype, btype.upper())
    # Don't double-prefix if OCR/vision already included the prefix word
    if text.upper().startswith(prefix + ":"):
        rendered = text
    elif text.upper().startswith(prefix):
        rendered = text
    else:
        rendered = f"{prefix}: {text}"
    return _make_chunk(
        chunk_id=_gen_id(btype[:4], section_code, pn, seq_counter),
        doc_id=doc_id, page=pn,
        block_ids=[f"p{pn}_{block['block_id']}"],
        bbox=block.get("bbox"),
        ctype=btype,
        section_code=section_code,
        source_label=source_label,
        section_path=section_path,
        text=rendered,
    )


def _chunk_legend(block, doc_id, pn, section_code, source_label,
                  section_path, seq_counter) -> dict:
    items = block.get("items") or block.get("legend_items") or []
    if isinstance(items, dict):
        # Convert dict {"1": "BOLT", "2": "NUT"} to list [{"key": "1", "value": "BOLT"}, ...]
        kv = [{"key": k, "value": v} for k, v in items.items()]
    elif isinstance(items, list):
        if items and isinstance(items[0], str):
            # Convert list ["1: BOLT", "2: NUT"] to list [{"key": "1", "value": "BOLT"}, ...]
            kv = []
            for it in items:
                if ":" in it:
                    k, v = it.split(":", 1)
                    kv.append({"key": k.strip(), "value": v.strip()})
                else:
                    kv.append({"key": "", "value": it.strip()})
        else:
            kv = [{"key": it.get("key", ""), "value": it.get("value", "")} for it in items]
    else:
        kv = []
        
    text  = "\n".join(f"{it['key']}: {it['value']}" for it in kv if it['key'] or it['value'])
    fig_id = block.get("parent_figure_id") or block.get("figure_id")
    return _make_chunk(
        chunk_id=_gen_id("lgnd", section_code, pn, seq_counter),
        doc_id=doc_id, page=pn,
        block_ids=[f"p{pn}_{block['block_id']}"],
        bbox=block.get("bbox"),
        ctype="legend",
        section_code=section_code,
        source_label=source_label,
        section_path=section_path,
        text=text,
        kv=kv,
        figure_refs=[fig_id] if fig_id else [],
    )


def _chunk_table(block, doc_id, pn, section_code, source_label,
                 section_path, seq_counter, title: str = "",
                 prev_condition: str = "") -> tuple[list[dict], str]:
    """Create chunks from a table block, grouping by condition for diagnostic tables.

    Returns:
        (chunks, last_condition) — last_condition is the last condition value
        seen in the table's first column, used to carry forward across pages.
    """
    rows = block.get("rows")
    # Normalize dict rows (Gemini sometimes returns [{col: val, ...}]) to lists
    if rows and isinstance(rows[0], dict):
        headers = list(rows[0].keys())
        rows = [headers] + [list(r.values()) for r in rows]
    # Ensure all row cells are strings
    if rows:
        rows = [[str(cell) if cell is not None else "" for cell in r] for r in rows]
    # Use block's own title (vision pages) or last heading seen
    table_title = (block.get("title") or "").strip() or title
    table_type = _classify_table_type(table_title, rows)
    if not rows:
        text = block.get("retrieval_text", "")
        if not text:
            return [], ""
        if table_title:
            text = f"{table_title}\n{text}"
        chunk_id = _gen_id("tbl", section_code, pn, seq_counter)
        return [_make_chunk(
            chunk_id=chunk_id, doc_id=doc_id, page=pn,
            block_ids=[f"p{pn}_{block['block_id']}"], bbox=block.get("bbox"),
            ctype="table", section_code=section_code,
            source_label=source_label, section_path=section_path,
            text=text, asset_refs=[block["asset_path"]] if block.get("asset_path") else [],
            table_type=table_type,
        )], ""

    # Cross-reference index tables: render as flat "Topic: section-ref" list
    if table_type == "index":
        lines = [
            f"{r[0]}: {r[1]}" for r in rows
            if len(r) >= 2
            and r[1].strip()               # skip letter-divider rows (empty col 2)
            and len(r[0].strip()) > 1      # skip single-letter section breaks
        ]
        if not lines:
            return [], ""
        # Split into chunks of ~15 entries for retrieval precision
        chunk_size = 15
        result = []
        for start in range(0, len(lines), chunk_size):
            text = "\n".join(lines[start:start + chunk_size])
            result.append(_make_chunk(
                chunk_id=_gen_id("tbl", section_code, pn, seq_counter),
                doc_id=doc_id, page=pn,
                block_ids=[f"p{pn}_{block['block_id']}"], bbox=block.get("bbox"),
                ctype="table", section_code=section_code,
                source_label=source_label, section_path=section_path,
                text=text,
                asset_refs=[block["asset_path"]] if block.get("asset_path") else [],
                table_type=table_type,
            ))
        return result, ""

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 and _looks_like_header(rows[0]) else rows
    if not data_rows:
        # Phase 2.4: Header-only chunk prevention
        return [], ""

    # Detect diagnostic tables BEFORE fill-down so we see the original
    # sparse first-column values (criterion 2 needs unfilled data).
    is_diagnostic = _is_diagnostic_table(header, data_rows)

    # Fill-down: carry forward the last non-empty value in column 1.
    # Diagnostic tables use "fill-down" formatting where the condition
    # only appears in the first row of a group; subsequent rows have
    # empty col 1 but still belong to the same condition.
    last_condition = ""
    for row in data_rows:
        if row[0].strip():
            last_condition = row[0].strip()
        else:
            row[0] = last_condition

    # Orphan row fix: if the first rows still have empty condition after
    # fill-down (table continues from previous page), inherit from
    # prev_condition or fall back to table title / "(Continued)".
    if data_rows and not data_rows[0][0].strip():
        fallback = prev_condition or table_title or "(Continued)"
        for row in data_rows:
            if row[0].strip():
                break
            row[0] = fallback

    # Small/medium spec tables (< 25 data rows): keep as ONE chunk
    # UNLESS the table is diagnostic (should be split by condition).
    # Splitting measurement tables or continuity matrices by condition
    # produces decontextualized fragments like "NO.: 1 - LENGTH: 1 242".
    if len(data_rows) < 25 and not is_diagnostic:
        lines = []
        for row in data_rows:
            parts = []
            for i, cell in enumerate(row):
                col_name = header[i] if i < len(header) else f"Col{i}"
                if cell.strip():
                    parts.append(f"{col_name}: {cell}")
            if parts:
                lines.append(" | ".join(parts))
        text = "\n".join(lines)
        if table_title:
            text = f"{table_title}\n{text}"
        if text.strip():
            # Track last condition for cross-page carry-forward
            tbl_last_cond = data_rows[-1][0].strip() if data_rows else ""
            return [_make_chunk(
                chunk_id=_gen_id("tbl", section_code, pn, seq_counter),
                doc_id=doc_id, page=pn,
                block_ids=[f"p{pn}_{block['block_id']}"], bbox=block.get("bbox"),
                ctype="table", section_code=section_code,
                source_label=source_label, section_path=section_path,
                text=text,
                asset_refs=[block["asset_path"]] if block.get("asset_path") else [],
                table_type=table_type,
            )], tbl_last_cond

    chunks = []

    # Group rows by the value in the first column (the "condition")
    from itertools import groupby
    groups = groupby(data_rows, key=lambda r: r[0])

    # Build set of header column names for noise detection
    header_names = {h.strip().upper() for h in header if h.strip()}

    for condition, group_rows in groups:
        group_rows = list(group_rows)
        if not condition and len(group_rows) == 1 and not any(group_rows[0][1:]):
            continue  # Skip empty rows

        # Skip header-as-data noise: OCR sometimes picks up repeated header
        # rows at page breaks, producing groups like "CONDITION: CONDITION"
        if condition.strip().upper() in header_names:
            continue

        # Render text for this specific condition group
        group_text = _render_table_group_text(header, group_rows, condition)
        # Prepend parent table title so each group chunk is self-contained
        # (e.g. "ENGINE PERFORMANCE DIAGNOSIS" before "Symptom: Engine Stalls")
        if table_title:
            group_text = f"{table_title}\n{group_text}"

        chunk_id = _gen_id("tbl", section_code, pn, seq_counter)
        chunk = _make_chunk(
            chunk_id=chunk_id,
            doc_id=doc_id, page=pn,
            block_ids=[f"p{pn}_{block['block_id']}"],
            bbox=block.get("bbox"),
            ctype="table",
            section_code=section_code,
            source_label=source_label,
            section_path=section_path,
            text=group_text,
            asset_refs=[block["asset_path"]] if block.get("asset_path") else [],
            table_type=table_type,
        )
        chunks.append(chunk)

    # Return last condition seen for cross-page carry-forward
    tbl_last_cond = data_rows[-1][0].strip() if data_rows else ""
    return chunks, tbl_last_cond


def _merge_table_blocks(blocks: list[dict]) -> dict:
    """Merge multiple small table blocks into one combined table block.

    Used when the extraction splits a single logical table into multiple
    small blocks (e.g., VIN decoder table split into 8 fragments).
    """
    if len(blocks) == 1:
        return blocks[0]

    all_rows = []
    block_ids = []
    bbox = None
    title = ""
    retrieval_texts = []

    for b in blocks:
        rows = b.get("rows") or []
        # Normalize dict rows
        if rows and isinstance(rows[0], dict):
            headers = list(rows[0].keys())
            rows = [headers] + [list(r.values()) for r in rows]
        if rows:
            # Ensure all cells are strings
            rows = [[str(cell) if cell is not None else "" for cell in r] for r in rows]
            all_rows.extend(rows)

        if b.get("block_id"):
            block_ids.append(b["block_id"])
        if not title and b.get("title"):
            title = b["title"]
        if b.get("retrieval_text"):
            retrieval_texts.append(b["retrieval_text"])

    merged = {
        "type": "table",
        "block_id": block_ids[0] if block_ids else "",
        "rows": all_rows if all_rows else None,
        "title": title,
        "bbox": bbox,
        "retrieval_text": "\n".join(retrieval_texts) if retrieval_texts else "",
        "asset_path": blocks[0].get("asset_path"),
    }
    return merged


def _looks_like_header(row: list[str]) -> bool:
    """Return True if a row looks like a header (no purely numeric cells)."""
    return row and not any(
        re.match(r"^\d+\.?\d*$", cell.strip()) for cell in row if cell.strip()
    )

def _is_index_table(rows: list) -> bool:
    """Return True if this is a cross-reference index (col 2 = section refs like '6A1-33').

    These tables should be rendered as flat topic-list chunks, not grouped by condition.
    """
    if not rows or len(rows[0]) < 2:
        return False
    hits = sum(
        1 for r in rows
        if len(r) >= 2 and INDEX_REF_RE.match(r[1].strip())
    )
    return hits >= len(rows) * 0.7


_PINOUT_TABLE_RE = re.compile(
    r"\b(?:pin|terminal|cavity|circuit|wire color|connector)\b",
    re.IGNORECASE,
)
_SPEC_TABLE_RE = re.compile(
    r"\b(?:voltage|resistance|torque|pressure|clearance|spec(?:ification)?|"
    r"capacity|n[·.]?m|ft-?lb|lb\.?-?ft|psi|kpa)\b",
    re.IGNORECASE,
)


def _classify_table_type(title: str, rows: list[list[str]] | None) -> str | None:
    """Classify high-value table types for retrieval and reranking."""
    if not rows:
        return None
    if _is_index_table(rows):
        return "index"

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 and _looks_like_header(rows[0]) else rows
    if _is_diagnostic_table(header, data_rows):
        return "diagnostic"

    header_text = " ".join(cell.strip() for cell in header if str(cell).strip())
    title_text = (title or "").strip()
    combined = " ".join(part for part in (title_text, header_text) if part)
    if _PINOUT_TABLE_RE.search(combined):
        return "pinout"
    if _SPEC_TABLE_RE.search(combined):
        return "spec"
    return None

def _is_diagnostic_table(header: list[str], data_rows: list[list[str]]) -> bool:
    """Return True if this table is a diagnostic/troubleshooting table that should be split by condition.

    Detection criteria:
    1. Header contains diagnostic-style columns (CONDITION/SYMPTOM + CAUSE + CORRECTION), OR
    2. Multiple distinct non-empty values in the first column (multiple condition groups present)
    """
    # Criterion 1: diagnostic header pattern
    if header:
        h_upper = [h.upper().strip() for h in header]
        has_condition = any(
            k in h for h in h_upper
            for k in ("CONDITION", "SYMPTOM", "COMPLAINT", "PROBLEM")
        )
        has_cause = any(
            k in h for h in h_upper
            for k in ("CAUSE", "SOURCE", "REASON")
        )
        has_correction = any(
            k in h for h in h_upper
            for k in ("CORRECTION", "CURE", "REMEDY", "REPAIR", "ACTION")
        )
        if has_condition and has_cause and has_correction:
            return True

    # Criterion 2: sparse fill-down pattern in the first column.
    # Diagnostic tables often only populate the condition in the first row
    # of each group, leaving continuation rows blank until chunk-time fill-down.
    if len(data_rows) < 4:
        return False

    first_col = [row[0].strip() for row in data_rows if row]
    non_empty_conditions = [value for value in first_col if value]
    distinct_conditions = {value for value in non_empty_conditions}

    if len(distinct_conditions) < 2:
        return False

    # Ordinary spec tables usually populate column 0 on every row. We only
    # want the sparse group-label pattern used by troubleshooting tables.
    if not any(not value for value in first_col):
        return False

    non_empty_ratio = len(non_empty_conditions) / len(first_col)
    if non_empty_ratio > 0.6:
        return False

    # Condition labels are typically short symptom phrases, not long prose.
    short_condition_count = sum(
        1 for value in non_empty_conditions
        if len(re.findall(r"\S+", value)) <= 5
    )
    if short_condition_count < len(non_empty_conditions) * 0.8:
        return False

    # Require actual payload outside the first column so sparse index-like
    # tables do not get treated as condition/cause/correction groups.
    payload_rows = sum(
        1 for row in data_rows
        if any(cell.strip() for cell in row[1:])
    )
    if payload_rows < max(3, len(data_rows) // 2):
        return False

    return True


def _render_table_group_text(header: list[str], group_rows: list[list[str]], condition: str) -> str:
    """Render a human-readable text block for a group of table rows under a single condition."""
    lines = []
    condition_header = header[0] if header else "Condition"
    if condition:
        lines.append(f"{condition_header}: {condition}")
    
    cause_header = header[1] if len(header) > 1 else "Cause"
    correction_header = header[2] if len(header) > 2 else "Correction"

    for row in group_rows:
        cause = row[1] if len(row) > 1 else ""
        correction = row[2] if len(row) > 2 else ""

        if cause or correction:
            parts = []
            if cause:
                parts.append(f"  - {cause_header}: {cause}")
            if correction:
                parts.append(f"  - {correction_header}: {correction}")
            lines.append("\n".join(parts))

    text = "\n".join(lines)
    # For large single-condition groups (>600 tokens), repeat the condition
    # name at the end so BM25 matching has a stronger signal.
    if condition and len(text) > 600 * CHARS_PER_TOKEN:
        text += f"\n{condition_header}: {condition}"
    return text


def _strip_page_headers(text: str) -> str:
    """Remove running page headers like '8-30 BODY AND CHASSIS ELECTRICAL' from chunk text."""
    lines = text.split("\n")
    cleaned = [ln for ln in lines
              if not PAGE_HEADER_RE.match(ln.strip())
              and not PAGE_HEADER_RE2.match(ln.strip())
              and not PAGE_LABEL_RE.match(ln.strip())
              and not SECTION_HEADER_RE.match(ln.strip())]
    return "\n".join(cleaned)


def _clean_toc_noise(text: str) -> str:
    """Strip dot-leader TOC artifacts and section codes from merged paragraphs."""
    if not text:
        return text
    t = re.sub(r"\.{3,}", " ", text)
    t = SECTION_CODE_RE.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _split_oversized_paragraph_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    parts: list[str] = []
    remaining = text
    min_split = max(min_chars, max_chars // 2)

    while len(remaining) > max_chars:
        search_window = remaining[:max_chars + 1]
        split_at = None
        for match in SENTENCE_BOUNDARY_RE.finditer(search_window):
            if match.start() >= min_split:
                split_at = match.start()

        if split_at is None:
            split_at = search_window.rfind(" ", min_split, max_chars + 1)
        if split_at is None or split_at < min_split:
            split_at = max_chars

        piece = remaining[:split_at].strip()
        if not piece:
            split_at = max_chars
            piece = remaining[:split_at].strip()
        parts.append(piece)
        remaining = remaining[split_at:].strip()

    if remaining:
        if parts and len(remaining) < max(min_chars // 2, 80):
            merged = f"{parts[-1]} {remaining}".strip()
            if len(merged) <= max_chars + max(min_chars // 2, 40):
                parts[-1] = merged
            else:
                parts.append(remaining)
        else:
            parts.append(remaining)
    return parts


def _chunk_paragraphs(para_blocks: list[dict], doc_id, pn, section_code,
                      source_label, section_path, seq_counter,
                      min_tok: int, max_tok: int,
                      page_figure_ids: list | None = None,
                      fig_num_to_id: dict | None = None) -> list[dict]:
    """Merge adjacent paragraph blocks into token-bounded chunks."""
    max_chars = max_tok * CHARS_PER_TOKEN
    min_chars = min_tok * CHARS_PER_TOKEN

    chunks   = []
    buf_text = []
    buf_ids  = []
    buf_bbox = None

    def flush():
        text = "\n\n".join(buf_text).strip()
        if not text:
            return
        for part in _split_oversized_paragraph_text(text, max_chars, min_chars):
            chunk = _make_chunk(
                chunk_id=_gen_id("para", section_code, pn, seq_counter),
                doc_id=doc_id, page=pn,
                block_ids=list(buf_ids),
                bbox=buf_bbox,
                ctype="paragraph",
                section_code=section_code,
                source_label=source_label,
                section_path=section_path,
                text=part,
                figure_refs=_resolve_fig_refs(part, page_figure_ids or [], fig_num_to_id),
            )
            chunks.append(chunk)

    for block in para_blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue

        # If adding this block would exceed max, flush first
        current_len = sum(len(t) for t in buf_text)
        if buf_text and current_len + len(text) > max_chars:
            flush()
            buf_text = []
            buf_ids  = []
            buf_bbox = None

        buf_text.append(text)
        buf_ids.append(f"p{pn}_{block['block_id']}")
        if buf_bbox is None:
            buf_bbox = block.get("bbox")

    flush()
    return chunks


def _build_figure_record(block: dict, doc_id, pn, section_code,
                         source_label, section_path,
                         caption_map: dict[str, str],
                         fallback_caption: str = "") -> dict:
    system = section_path.split(' > ')[0] if section_path and ' > ' in section_path else (section_path or "Unknown")
    figure_id = block.get("figure_id")
    # Use the pre-built map to get the caption text, ensuring consistency.
    caption = caption_map.get(figure_id, block.get("caption_text"))
    # Fallback chain for figures without captions
    if not caption:
        # Try first sentence of vision description
        vd = (block.get("vision_description") or "").strip()
        if vd:
            first_sent = re.split(r'[.\n]', vd)[0].strip()
            if len(first_sent) > 10:
                caption = first_sent
    if not caption and fallback_caption:
        caption = fallback_caption
    return {
        "figure_id":          figure_id,
        "doc_id":             doc_id,
        "page":               pn,
        "block_id":           block["block_id"],
        "bbox":               block.get("bbox"),
        "asset_path":         block.get("asset_path"),
        "caption_text":       caption,
        "figure_number":      block.get("figure_number"),
        "legend_items":       block.get("legend_items"),
        "vision_description": block.get("vision_description"),
        "system":             system,
        "section_code":       section_code,
        "source_label":       source_label,
        "section_path":       section_path,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_chunk(*, chunk_id, doc_id, page, block_ids, bbox, ctype,
                section_code, source_label, section_path, text,
                procedure_type=None, system=None, engine_variant=None,
                steps=None, kv=None, figure_refs=None, asset_refs=None,
                starting_step=1, info_types=None, entities=None,
                body_styles=None, trim_variants=None, sir_equipped=None,
                table_type=None) -> dict:
    if not system:
        system = _get_system(section_code, section_path)
    if not engine_variant:
        engine_variant = _get_engine_variant(section_code, section_path)

    # Strip running page headers (e.g. "8-30 BODY AND CHASSIS ELECTRICAL")
    text = _strip_page_headers(text)

    return _enrich_chunk_metadata({
        "chunk_id":       chunk_id,
        "doc_id":         doc_id,
        "page":           page,
        "block_ids":      block_ids,
        "bbox":           bbox,
        "type":           ctype,
        "procedure_type": procedure_type,
        "system":         system,
        "engine_variant": engine_variant,
        "section_code":   section_code,
        "source_label":   source_label,
        "section_path":   section_path,
        "text":           text,
        "steps":          steps,
        "starting_step":  starting_step,
        "kv":             kv,
        "figure_refs":    figure_refs or [],
        "asset_refs":     asset_refs or [],
        "info_types":     info_types,
        "entities":       entities,
        "body_styles":    body_styles,
        "trim_variants":  trim_variants,
        "sir_equipped":   sir_equipped,
        "table_type":     table_type,
        "token_count":    len(text) // CHARS_PER_TOKEN,
    })


def _enrich_chunk_metadata(chunk: dict) -> dict:
    text = chunk.get("text") or ""
    context = " ".join(
        part for part in (
            chunk.get("section_path", ""),
            chunk.get("source_label") or "",
            text,
        )
        if part
    )
    info_types = chunk.get("info_types")
    if info_types is None:
        info_types = _infer_info_types(chunk.get("type", ""), context)
    entities = chunk.get("entities")
    if entities is None:
        entities = _extract_entities(context, info_types)
    body_styles = chunk.get("body_styles")
    if body_styles is None:
        body_styles = _infer_matches(context, BODY_STYLE_PATTERNS)
    trim_variants = chunk.get("trim_variants")
    if trim_variants is None:
        trim_variants = _infer_matches(context, TRIM_VARIANT_PATTERNS)
    if chunk.get("sir_equipped") is None:
        chunk["sir_equipped"] = _infer_sir_equipped(context)

    chunk["info_types"] = info_types
    chunk["entities"] = entities
    chunk["body_styles"] = body_styles
    chunk["trim_variants"] = trim_variants
    chunk.setdefault("steps", None)
    chunk.setdefault("kv", None)
    chunk.setdefault("figure_refs", [])
    chunk.setdefault("asset_refs", [])
    chunk.setdefault("same_page_chunk_ids", [])
    chunk.setdefault("same_page_figure_ids", [])
    chunk.setdefault("related_figure_ids", [])
    chunk.setdefault("related_table_ids", [])
    chunk.setdefault("table_type", None)
    chunk.setdefault("source_doc", "main")
    if "token_count" not in chunk:
        chunk["token_count"] = len(text) // CHARS_PER_TOKEN
    return chunk


def _infer_matches(text: str, patterns: tuple[tuple[str, re.Pattern], ...]) -> list[str]:
    return [value for value, pattern in patterns if pattern.search(text)]


def _infer_info_types(ctype: str, text: str) -> list[str]:
    info_types: list[str] = []
    if ctype == "procedure":
        info_types.append("procedure")
    elif ctype == "figure":
        info_types.append("diagram")
    elif ctype in ("toc_entry", "toc_category"):
        info_types.append("cross_reference")

    for info_type, pattern in INFO_TYPE_PATTERNS:
        if pattern.search(text) and info_type not in info_types:
            info_types.append(info_type)
    if ctype == "table" and "spec" not in info_types and re.search(r"\b(?:key|eng\.?\s+run|voltage|resistance|torque)\b", text, re.IGNORECASE):
        info_types.append("spec")
    return info_types


def _extract_entities(text: str, info_types: list[str]) -> list[str]:
    entities: list[str] = []
    for entity, pattern in ENTITY_PATTERNS:
        if pattern.search(text):
            entities.append(entity)

    if any(info in info_types for info in ("pinout", "connector_face", "wiring")):
        for pin in PIN_LABEL_RE.findall(text):
            normalized = pin.lower()
            if normalized not in entities:
                entities.append(normalized)
            if len(entities) >= 20:
                break
    return entities


def _infer_sir_equipped(text: str) -> bool | None:
    if SIR_FALSE_RE.search(text):
        return False
    if re.search(r"\b(?:sir|airbag|inflatable restraint|supplemental restraint)\b", text, re.IGNORECASE):
        return True
    return None


def _attach_page_artifact_links(chunks: list[dict], figures: list[dict]) -> None:
    figures_by_page: dict[int, list[str]] = {}
    for fig in figures:
        page = fig.get("page")
        figure_id = fig.get("figure_id")
        if page is None or not figure_id:
            continue
        figures_by_page.setdefault(page, [])
        if figure_id not in figures_by_page[page]:
            figures_by_page[page].append(figure_id)

    chunks_by_page: dict[int, list[dict]] = {}
    for chunk in chunks:
        page = chunk.get("page")
        if page is None:
            chunk.setdefault("same_page_chunk_ids", [])
            chunk.setdefault("same_page_figure_ids", [])
            chunk.setdefault("related_figure_ids", [])
            chunk.setdefault("related_table_ids", [])
            continue
        chunks_by_page.setdefault(page, []).append(chunk)

    for page, page_chunks in chunks_by_page.items():
        page_chunk_ids = [chunk["chunk_id"] for chunk in page_chunks]
        table_ids = [chunk["chunk_id"] for chunk in page_chunks if chunk.get("type") == "table"]
        page_figure_ids = figures_by_page.get(page, [])
        for chunk in page_chunks:
            chunk["same_page_chunk_ids"] = [cid for cid in page_chunk_ids if cid != chunk["chunk_id"]]
            own_fig_refs = list(chunk.get("figure_refs") or [])
            chunk["same_page_figure_ids"] = [fid for fid in page_figure_ids if fid not in own_fig_refs]

            related_figures = list(own_fig_refs)
            if chunk.get("type") == "table" or any(
                info in (chunk.get("info_types") or [])
                for info in ("pinout", "connector_face", "wiring", "spec")
            ):
                for figure_id in page_figure_ids:
                    if figure_id not in related_figures:
                        related_figures.append(figure_id)
            chunk["related_figure_ids"] = related_figures

            related_tables: list[str] = []
            if chunk.get("type") == "figure" or "diagram" in (chunk.get("info_types") or []):
                related_tables = [table_id for table_id in table_ids if table_id != chunk["chunk_id"]]
            chunk["related_table_ids"] = related_tables


def _gen_id(prefix: str, section_code: str | None, page: int,
            counters: dict) -> str:
    """Generate a deterministic, human-readable chunk ID."""
    sec = (section_code or "xx").lower().replace("-", "")
    key = f"{prefix}_{sec}"
    n   = counters.get(key, 0)
    counters[key] = n + 1
    return f"{prefix}_{sec}_p{page}_{n}"


def _find_fig_refs(text: str) -> list[str]:
    """Extract figure references like 'Figure 3' from text."""
    return [f"fig_{m.group(1)}" for m in FIGURE_REF_RE.finditer(text)]


def _resolve_fig_refs(text: str, page_figure_ids: list[str],
                      fig_num_to_id: dict[str, str] | None = None) -> list[str]:
    """Return real figure IDs referenced by text.

    Uses precise matching: "Figure 22" in text -> looks up figure number 22
    in fig_num_to_id map (built from figure captions on the same page).
    Falls back to all page figures only if text mentions figures generically
    (e.g. "see diagram") but no specific number can be matched.
    """
    if not page_figure_ids:
        return []

    # Try precise matching first: "Figure 22" -> specific figure ID
    matched = []
    if fig_num_to_id:
        for m in FIGURE_REF_RE.finditer(text):
            fig_num = m.group(1)
            fig_id = fig_num_to_id.get(fig_num)
            if fig_id and fig_id not in matched:
                matched.append(fig_id)

    if matched:
        return matched

    # Fallback: text mentions figure/diagram generically but no number parsed.
    # Only attach all page figures if there's a single figure on the page
    # (avoids dumping 5 unrelated figures when text says "see diagram").
    if FIGURE_MENTION_RE.search(text):
        if len(page_figure_ids) == 1:
            return list(page_figure_ids)
        # Multiple figures on page but no specific number — don't guess
        return []

    return []


# ── TOC Markdown Parser ─────────────────────────────────────────────────────

# Matches "Section Name — CODE" or "Section Name - CODE"
_TOC_ENTRY_RE = re.compile(r"^(.+?)\s*[—\-]\s*([0-9]+[A-Z0-9\-]*)$")


def build_toc_chunks(toc_md_path, doc_id: str = "geo_metro_1990") -> list[dict]:
    """Parse a manually-written TABLE OF CONTENTS markdown file into RAG chunks.

    The file format is:
        CATEGORY HEADING
        Section Name — SectionCode
        ...

    Returns a list of chunk dicts compatible with chunks.jsonl format.
    Produces:
      - One chunk per top-level category (GENERAL INFORMATION, ENGINE, etc.)
        containing all its subsections, for broad topic lookups.
      - One chunk per individual section entry, for specific section lookups.
    """
    from pathlib import Path
    path = Path(toc_md_path)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[dict] = []
    counters: dict[str, int] = {}
    current_category: str | None = None
    category_entries: list[str] = []

    def flush_category(cat: str, entries: list[str]) -> None:
        if not cat or not entries:
            return
        text = cat + "\n" + "\n".join(entries)
        chunk_id = "toc_cat_" + re.sub(r"\W+", "_", cat.lower()).strip("_")
        chunks.append(_enrich_chunk_metadata({
            "chunk_id":     chunk_id,
            "doc_id":       doc_id,
            "page":         None,       # TOC has no single page
            "block_ids":    [],
            "bbox":         None,
            "type":         "toc_category",
            "system":       _get_system(None, cat),
            "engine_variant": "both",
            "section_code": None,
            "source_label": cat,
            "section_path": cat,
            "text":         text,
            "steps":        None,
            "kv":           None,
            "figure_refs":  [],
            "asset_refs":   [],
            "token_count":  len(text) // CHARS_PER_TOKEN,
        }))

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Try to parse as "Name — Code" entry
        m = _TOC_ENTRY_RE.match(line)
        is_uppercase_category = False
        
        if m:
            section_name = m.group(1).strip()
            # If name is all caps (and long enough), treat as category header
            # e.g. "STEERING, SUSPENSION, WHEELS AND TIRES — 3"
            if section_name.isupper() and len(section_name) > 4:
                is_uppercase_category = True
            
        if m and not is_uppercase_category:
            section_name = m.group(1).strip()
            section_code = m.group(2).strip()
            entry_text   = f"{section_name} — {section_code}"
            category_entries.append(entry_text)

            # Individual section chunk
            chunk_id = f"toc_{section_code.lower().replace('-', '')}"
            n = counters.get(chunk_id, 0)
            counters[chunk_id] = n + 1
            if n > 0:
                chunk_id = f"{chunk_id}_{n}"

            full_text = (
                f"Table of Contents entry: {section_name} (Section {section_code})\n"
                f"Category: {current_category or 'General'}"
            )
            toc_section_path = f"{current_category} > {section_name}" if current_category else section_name
            chunks.append(_enrich_chunk_metadata({
                "chunk_id":     chunk_id,
                "doc_id":       doc_id,
                "page":         None,
                "block_ids":    [],
                "bbox":         None,
                "type":         "toc_entry",
                "system":       _get_system(section_code, toc_section_path),
                "engine_variant": "both",
                "section_code": section_code,
                "source_label": section_code,
                "section_path": toc_section_path,
                "text":         full_text,
                "steps":        None,
                "kv":           None,
                "figure_refs":  [],
                "asset_refs":   [],
                "token_count":  len(full_text) // CHARS_PER_TOKEN,
            }))
        else:
            # It's a category heading (all-caps or no "—" separator)
            flush_category(current_category, category_entries)
            current_category  = line
            category_entries  = []

    flush_category(current_category, category_entries)
    return chunks


# ── Manual CSV Table Parser ─────────────────────────────────────────────────

def build_csv_table_chunks(table_spec: dict, doc_id: str = "geo_metro_1990",
                           project_root=None) -> list[dict]:
    """Convert a hand-curated CSV table into RAG-ready chunks.

    Each row becomes its own chunk so retrieval is fine-grained.
    A summary chunk containing all rows is also produced for broad queries.

    table_spec keys:
      path, section_code, source_label, section_path, page
    """
    from pathlib import Path
    raw_path = table_spec.get("path", "")
    if project_root:
        path = Path(project_root) / raw_path
    else:
        path = Path(raw_path)

    if not path.exists():
        return []

    section_code = table_spec.get("section_code", "")
    source_label = table_spec.get("source_label", path.stem)
    section_path = table_spec.get("section_path", source_label)
    page         = table_spec.get("page")

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        return []

    chunks: list[dict] = []
    all_lines: list[str] = [f"Scheduled Maintenance — {source_label}"]

    for row in rows:
        parts: list[str] = []
        service = row.get("Service", "").strip()
        if not service:
            continue

        item_num      = row.get("Item", "").strip()
        schedule_type = row.get("Schedule_Type", "").strip()  # "I", "II", or absent

        # Build label — include schedule prefix when multiple schedules present
        if schedule_type:
            label = f"Schedule {schedule_type}, Item {item_num}: {service}" if item_num else f"Schedule {schedule_type}: {service}"
        else:
            label = f"Item {item_num}: {service}" if item_num else service

        # Build interval description
        intervals = []
        if row.get("Interval_Miles", "").strip():
            intervals.append(f"{row['Interval_Miles'].rstrip('0').rstrip('.')} miles")
        if row.get("Interval_Kilometers", "").strip():
            intervals.append(f"{row['Interval_Kilometers'].rstrip('0').rstrip('.')} km")
        if row.get("Interval_Months", "").strip():
            intervals.append(f"{row['Interval_Months'].rstrip('0').rstrip('.')} months")
        if row.get("Whichever_Occurs_First", "").strip() not in ("", "True", "False", "false", "true"):
            intervals.append(row["Whichever_Occurs_First"].strip())

        interval_str = " / ".join(intervals) if intervals else "see notes"
        whichever = " (whichever occurs first)" if row.get("Whichever_Occurs_First", "").strip() == "True" else ""

        parts.append(f"{label} — every {interval_str}{whichever}.")

        if row.get("Emission_Service", "").strip() == "True":
            parts.append("Emission control service item.")
        note = row.get("Reference_Note", "").strip()
        if note:
            parts.append(f"Note: {note}")

        row_text = " ".join(parts)
        all_lines.append(row_text)

        # Chunk ID — include schedule type to avoid collisions across schedules
        sched_suffix = f"_{schedule_type.lower()}" if schedule_type else ""
        chunk_id = f"maint_{section_code.lower()}{sched_suffix}_{item_num or len(chunks)}"
        chunks.append(_enrich_chunk_metadata({
            "chunk_id":     chunk_id,
            "doc_id":       doc_id,
            "page":         page,
            "block_ids":    [],
            "bbox":         None,
            "type":         "table",
            "system":       _get_system(section_code, section_path),
            "engine_variant": "both",
            "section_code": section_code,
            "source_label": source_label,
            "section_path": section_path,
            "text":         row_text,
            "steps":        None,
            "kv":           None,
            "figure_refs":  [],
            "asset_refs":   [str(path)],
            "token_count":  len(row_text) // CHARS_PER_TOKEN,
        }))

    # Summary chunk — all rows together for broad queries like "what's the maintenance schedule?"
    summary_text = "\n".join(all_lines)
    chunks.insert(0, _enrich_chunk_metadata({
        "chunk_id":     f"maint_{section_code.lower()}_summary",
        "doc_id":       doc_id,
        "page":         page,
        "block_ids":    [],
        "bbox":         None,
        "type":         "table",
        "system":       _get_system(section_code, section_path),
        "engine_variant": "both",
        "section_code": section_code,
        "source_label": source_label,
        "section_path": section_path,
        "text":         summary_text,
        "steps":        None,
        "kv":           None,
        "figure_refs":  [],
        "asset_refs":   [str(path)],
        "token_count":  len(summary_text) // CHARS_PER_TOKEN,
    }))

    return chunks
