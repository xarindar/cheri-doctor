"""
RAG-optimized output assembly module.

Processes extracted page data into multiple output formats:
- pages/       — individual page .md files (current format, improved)
- sections/    — all pages merged by section code
- chunks/      — RAG-sized chunks (~500-800 tokens)
- tables/      — all tables extracted separately
- diagrams/    — diagram descriptions separately
- full_manual.md — complete concatenated output
- index.json   — structured index for programmatic access
- index.md     — human-readable index (enhanced)
"""

import json
import re
from pathlib import Path

# Known section names for the 1990 Geo Metro Manual
SECTION_NAMES = {
    "0A": "General Information",
    "0B": "Maintenance and Lubrication",
    "1A": "Heater and Ventilation",
    "1B": "Air Conditioner",
    "2A": "Suspension Front",
    "2B": "Suspension Rear",
    "3A": "Steering",
    "3B": "Power Steering",
    "4A": "Brakes",
    "5A": "Body and Sheet Metal",
    "5B": "Frame and Underbody",
    "6A": "Engine Mechanical",
    "6B": "Engine Cooling",
    "6C": "Engine Fuel",
    "6D": "Engine Electrical",
    "6E": "Emission Controls",
    "7A": "Transaxle Manual",
    "7B": "Transaxle Automatic",
    "8A": "Electrical",
    "8B": "Electrical Wiring",
}

# Approximate token count (rough: 1 token ≈ 4 chars)
TARGET_CHUNK_TOKENS = 600
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * 4  # ~2400 chars


# ── Topic Extraction ─────────────────────────────────────────────────────

def _extract_topics(text: str) -> list[str]:
    """Extract key topics from page text for metadata."""
    topics = []
    # Look for common automotive topics
    topic_patterns = [
        (r'\b(?:oil\s+change|engine\s+oil)\b', "oil change"),
        (r'\bchassis\s+lubrication\b', "chassis lubrication"),
        (r'\btire\s+rotation\b', "tire rotation"),
        (r'\bbrake\s+(?:pad|shoe|fluid|system|inspection)\b', "brakes"),
        (r'\bcoolant\b', "coolant"),
        (r'\brefrigerant\b', "refrigerant"),
        (r'\bcompressor\b', "compressor"),
        (r'\bevaporator\b', "evaporator"),
        (r'\bcondenser\b', "condenser"),
        (r'\btorque\s+specification', "torque specifications"),
        (r'\bwiring\s+(?:diagram|schematic)\b', "wiring diagram"),
        (r'\bdiagnos(?:is|tic)\b', "diagnosis"),
        (r'\btroubleshooting\b', "troubleshooting"),
        (r'\bspecification\b', "specifications"),
        (r'\bvacuum\s+hose\b', "vacuum hoses"),
        (r'\bbelt\s+(?:tension|replacement|routing)\b', "drive belts"),
        (r'\bfilter\s+(?:replacement|change)\b', "filter replacement"),
        (r'\bspark\s+plug\b', "spark plugs"),
        (r'\btiming\b', "timing"),
        (r'\bvalve\s+(?:adjustment|clearance)\b', "valve adjustment"),
        (r'\balignment\b', "alignment"),
        (r'\bsuspension\b', "suspension"),
        (r'\bsteering\b', "steering"),
        (r'\btransaxle\b', "transaxle"),
        (r'\bclutch\b', "clutch"),
        (r'\bignition\b', "ignition"),
        (r'\bfuel\s+(?:system|pump|filter|injection)\b', "fuel system"),
        (r'\bcooling\s+system\b', "cooling system"),
        (r'\bexhaust\b', "exhaust"),
        (r'\bemission\b', "emissions"),
        (r'\belectrical\b', "electrical"),
        (r'\bbattery\b', "battery"),
    ]

    text_lower = text.lower()
    for pattern, topic in topic_patterns:
        if re.search(pattern, text_lower):
            if topic not in topics:
                topics.append(topic)

    return topics[:5]  # Limit to top 5


def _extract_figure_refs(text: str) -> list[str]:
    """Extract figure references from text."""
    refs = re.findall(r'(?:Fig(?:ure)?\.?\s*\d+|FIGURE\s+\d+)', text, re.IGNORECASE)
    return list(dict.fromkeys(refs))  # Deduplicate preserving order


def _extract_cross_refs(text: str) -> list[str]:
    """Extract cross-references to other sections."""
    refs = re.findall(r'(?:Section|section)\s+(\d+[A-Z]\d?)', text)
    return list(dict.fromkeys(f"Section {r}" for r in refs))


# ── Chunking ─────────────────────────────────────────────────────────────

def _chunk_text(text: str, section_code: str, section_name: str,
                page_num: int) -> list[dict]:
    """Split text into RAG-sized chunks with metadata.

    Tries to split at paragraph boundaries. Each chunk gets context metadata.
    """
    if not text or not text.strip():
        return []

    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 > TARGET_CHUNK_CHARS and current_chunk:
            # Save current chunk
            chunks.append({
                "text": current_chunk.strip(),
                "section_code": section_code,
                "section_name": section_name,
                "page_num": page_num,
                "chunk_idx": chunk_idx,
            })
            current_chunk = ""
            chunk_idx += 1

        if current_chunk:
            current_chunk += "\n\n"
        current_chunk += para

    # Save remaining text
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "section_code": section_code,
            "section_name": section_name,
            "page_num": page_num,
            "chunk_idx": chunk_idx,
        })

    return chunks


# ── Output Assembly ──────────────────────────────────────────────────────

def assemble_output(pages: list[dict], out_dir: Path, pdf_name: str):
    """Assemble all extracted page data into RAG-optimized output structure.

    Args:
        pages: List of page result dicts from extract.py processing.
            Each dict should have: page_num, type, section, section_code,
            text, table_md, diagram_description, files.
        out_dir: Base output directory (e.g., output/manual/).
        pdf_name: Name of the source PDF.
    """
    # Create output subdirectories
    pages_dir = out_dir / "pages"
    sections_dir = out_dir / "sections"
    chunks_dir = out_dir / "chunks"
    tables_dir = out_dir / "tables"
    diagrams_dir = out_dir / "diagrams"

    for d in [pages_dir, sections_dir, chunks_dir, tables_dir, diagrams_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Sort pages by page number
    pages_sorted = sorted(pages, key=lambda p: p["page_num"])

    # ── Per-page output (pages/) ─────────────────────────────────────
    all_chunks = []
    all_tables = []
    all_diagrams = []
    section_pages: dict[str, list[dict]] = {}  # section_code → [page_data]
    index_entries = []

    for page in pages_sorted:
        page_num = page["page_num"]
        page_label = f"page_{page_num:04d}"
        page_type = page.get("type", "text")
        section = page.get("section", "")
        section_code = page.get("section_code", "")
        section_page = page.get("section_page", "")
        text = page.get("text", "")
        table_md = page.get("table_md", "")
        diagram_desc = page.get("diagram_description", "")
        files = page.get("files", [])

        # Extract metadata
        topics = _extract_topics(text)
        figure_refs = _extract_figure_refs(text)
        cross_refs = _extract_cross_refs(text)
        section_name = SECTION_NAMES.get(section_code, section or "")

        # Write enhanced page .md file
        md_path = pages_dir / f"{page_label}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            # Enhanced YAML front matter
            f.write("---\n")
            f.write(f"page: {page_num}\n")
            f.write(f"type: {page_type}\n")
            if section_name:
                f.write(f'section: "{section_name}"\n')
            if section_code:
                f.write(f'section_code: "{section_code}"\n')
            if section_page:
                f.write(f'section_page: "{section_page}"\n')
            if topics:
                f.write(f'topics: {json.dumps(topics)}\n')
            if figure_refs:
                f.write(f'figures_referenced: {json.dumps(figure_refs)}\n')
            if cross_refs:
                f.write(f'cross_references: {json.dumps(cross_refs)}\n')
            f.write("---\n\n")

            # Header
            if section_name:
                f.write(f"# Page {page_num} — {section_name} (Section {section_code})\n\n")
            else:
                f.write(f"# Page {page_num}\n\n")

            # Content based on page type
            if page_type == "table" and table_md:
                f.write(f"*Table page — see [{page_label}_diagram.png]({page_label}_diagram.png) for original layout*\n\n")
                f.write(f"## Reconstructed Table\n\n{table_md}\n")
                if text.strip():
                    f.write(f"\n## Additional Text\n\n{text.strip()}\n")
            elif page_type == "diagram":
                f.write(f"*Diagram page — see [{page_label}_diagram.png]({page_label}_diagram.png)*\n\n")
                if diagram_desc:
                    f.write(f"## AI Description\n\n{diagram_desc}\n")
                if text.strip():
                    f.write(f"\n## OCR Text (labels/captions)\n\n{text.strip()}\n")
            elif page_type == "mixed":
                f.write(f"*Mixed page — text and diagram*\n\n")
                f.write(f"![Diagram]({page_label}_diagram.png)\n\n")
                if diagram_desc:
                    f.write(f"## AI Description\n\n{diagram_desc}\n\n")
                f.write(f"## Text Content\n\n{text.strip()}\n")
            elif page_type == "toc":
                f.write(f"## Table of Contents\n\n{text.strip()}\n")
            else:
                f.write(f"{text.strip()}\n")

        # ── Collect tables ───────────────────────────────────────────
        if table_md:
            table_entry = {
                "page_num": page_num,
                "section_code": section_code,
                "section_name": section_name,
                "markdown": table_md,
            }
            all_tables.append(table_entry)

            # Write individual table file
            table_slug = f"table_{page_num:04d}"
            if section_code:
                table_slug += f"_{section_code.lower()}"
            table_path = tables_dir / f"{table_slug}.md"
            with open(table_path, "w", encoding="utf-8") as f:
                f.write(f"---\npage: {page_num}\nsection_code: \"{section_code}\"\n")
                f.write(f'section: "{section_name}"\n---\n\n')
                f.write(f"# Table from Page {page_num}")
                if section_name:
                    f.write(f" — {section_name}")
                f.write(f"\n\n{table_md}\n")

        # ── Collect diagram descriptions ─────────────────────────────
        if diagram_desc:
            diag_entry = {
                "page_num": page_num,
                "section_code": section_code,
                "section_name": section_name,
                "description": diagram_desc,
            }
            all_diagrams.append(diag_entry)

            diag_slug = f"diagram_{page_num:04d}"
            if section_code:
                diag_slug += f"_{section_code.lower()}"
            diag_path = diagrams_dir / f"{diag_slug}.md"
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write(f"---\npage: {page_num}\nsection_code: \"{section_code}\"\n")
                f.write(f'section: "{section_name}"\n---\n\n')
                f.write(f"# Diagram from Page {page_num}")
                if section_name:
                    f.write(f" — {section_name}")
                f.write(f"\n\n{diagram_desc}\n")

        # ── Generate chunks ──────────────────────────────────────────
        chunk_text = text
        if table_md:
            chunk_text += "\n\n" + table_md
        if diagram_desc:
            chunk_text += "\n\n" + diagram_desc

        page_chunks = _chunk_text(chunk_text, section_code, section_name, page_num)
        all_chunks.extend(page_chunks)

        # ── Collect by section ───────────────────────────────────────
        if section_code:
            if section_code not in section_pages:
                section_pages[section_code] = []
            section_pages[section_code].append(page)

        # ── Index entry ──────────────────────────────────────────────
        index_entries.append({
            "page_num": page_num,
            "type": page_type,
            "section": section_name,
            "section_code": section_code,
            "section_page": section_page,
            "text_length": len(text.strip()),
            "topics": topics,
            "figures": figure_refs,
            "cross_references": cross_refs,
            "has_table": bool(table_md),
            "has_diagram_description": bool(diagram_desc),
            "files": files,
        })

    # ── Section files (sections/) ────────────────────────────────────
    for code, sec_pages in sorted(section_pages.items()):
        section_name = SECTION_NAMES.get(code, "")
        slug = f"{code.lower()}_{section_name.lower().replace(' ', '_')}" if section_name else code.lower()
        sec_path = sections_dir / f"{slug}.md"

        with open(sec_path, "w", encoding="utf-8") as f:
            f.write(f"# Section {code} — {section_name or '(untitled)'}\n\n")
            f.write(f"Pages: {sec_pages[0]['page_num']}–{sec_pages[-1]['page_num']}\n\n")
            f.write("---\n\n")

            for page in sorted(sec_pages, key=lambda p: p["page_num"]):
                f.write(f"## Page {page['page_num']}\n\n")
                text = page.get("text", "").strip()
                if text:
                    f.write(f"{text}\n\n")
                table_md = page.get("table_md", "")
                if table_md:
                    f.write(f"### Table\n\n{table_md}\n\n")
                diagram_desc = page.get("diagram_description", "")
                if diagram_desc:
                    f.write(f"### Diagram Description\n\n{diagram_desc}\n\n")
                f.write("---\n\n")

    # ── Chunk files (chunks/) ────────────────────────────────────────
    for i, chunk in enumerate(all_chunks):
        chunk_slug = f"chunk_{i + 1:04d}"
        if chunk["section_code"]:
            chunk_slug += f"_{chunk['section_code'].lower()}"
        chunk_path = chunks_dir / f"{chunk_slug}.md"

        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"chunk: {i + 1}\n")
            f.write(f"page: {chunk['page_num']}\n")
            if chunk["section_code"]:
                f.write(f'section_code: "{chunk["section_code"]}"\n')
            if chunk["section_name"]:
                f.write(f'section: "{chunk["section_name"]}"\n')
            f.write("---\n\n")
            f.write(chunk["text"])
            f.write("\n")

    # ── Full manual (full_manual.md) ─────────────────────────────────
    full_path = out_dir / "full_manual.md"
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_name}\n\n")
        f.write(f"Total pages: {len(pages_sorted)}\n\n")
        f.write("---\n\n")

        for page in pages_sorted:
            page_num = page["page_num"]
            section_code = page.get("section_code", "")
            section_name = SECTION_NAMES.get(section_code, page.get("section", ""))
            text = page.get("text", "").strip()
            table_md = page.get("table_md", "")
            diagram_desc = page.get("diagram_description", "")

            if section_name:
                f.write(f"## Page {page_num} — {section_name} ({section_code})\n\n")
            else:
                f.write(f"## Page {page_num}\n\n")

            if text:
                f.write(f"{text}\n\n")
            if table_md:
                f.write(f"{table_md}\n\n")
            if diagram_desc:
                f.write(f"### Diagram Description\n\n{diagram_desc}\n\n")

            f.write("---\n\n")

    # ── Index JSON (index.json) ──────────────────────────────────────
    index_json = {
        "pdf_name": pdf_name,
        "total_pages": len(pages_sorted),
        "sections": {code: name for code, name in SECTION_NAMES.items()
                     if code in section_pages},
        "page_type_counts": {},
        "pages": index_entries,
        "total_tables": len(all_tables),
        "total_diagrams": len(all_diagrams),
        "total_chunks": len(all_chunks),
    }

    # Count page types
    for entry in index_entries:
        t = entry["type"]
        index_json["page_type_counts"][t] = index_json["page_type_counts"].get(t, 0) + 1

    json_path = out_dir / "index.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index_json, f, indent=2)

    # ── Index MD (index.md) ──────────────────────────────────────────
    _write_index_md(out_dir, pages_sorted, pdf_name, index_json)

    print(f"  Assembly complete:")
    print(f"    Pages:    {len(pages_sorted)} in pages/")
    print(f"    Sections: {len(section_pages)} in sections/")
    print(f"    Chunks:   {len(all_chunks)} in chunks/")
    print(f"    Tables:   {len(all_tables)} in tables/")
    print(f"    Diagrams: {len(all_diagrams)} in diagrams/")


def _write_index_md(out_dir: Path, pages: list[dict], pdf_name: str, index_json: dict):
    """Write enhanced human-readable index."""
    idx_path = out_dir / "index.md"
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_name} — Extraction Index\n\n")
        f.write(f"Total pages: {len(pages)}\n\n")

        # Summary counts
        counts = index_json["page_type_counts"]
        f.write("## Page Type Summary\n\n")
        f.write("| Type | Count |\n|------|-------|\n")
        for t, c in sorted(counts.items()):
            f.write(f"| {t} | {c} |\n")
        f.write("\n")

        # Output structure
        f.write("## Output Structure\n\n")
        f.write("| Directory | Contents |\n|-----------|----------|\n")
        f.write(f"| pages/ | {len(pages)} individual page files |\n")
        f.write(f"| sections/ | {len(index_json['sections'])} section files |\n")
        f.write(f"| chunks/ | {index_json['total_chunks']} RAG-sized chunks |\n")
        f.write(f"| tables/ | {index_json['total_tables']} extracted tables |\n")
        f.write(f"| diagrams/ | {index_json['total_diagrams']} diagram descriptions |\n")
        f.write(f"| full_manual.md | Complete concatenated output |\n")
        f.write(f"| index.json | Structured index for programmatic access |\n")
        f.write("\n")

        # Section summary
        sections = index_json.get("sections", {})
        if sections:
            f.write("## Sections\n\n")
            f.write("| Code | Section Name |\n|------|-------------|\n")
            for code, name in sorted(sections.items()):
                f.write(f"| {code} | {name} |\n")
            f.write("\n---\n\n")

        # Page listing
        f.write("## All Pages\n\n")
        f.write("| Page | Section | Type | Text Length | Topics |\n")
        f.write("|------|---------|------|-------------|--------|\n")
        for entry in index_json["pages"]:
            section = entry.get("section_code", "") or ""
            topics = ", ".join(entry.get("topics", [])[:3])
            f.write(f"| {entry['page_num']} | {section} | {entry['type']} "
                    f"| {entry['text_length']} | {topics} |\n")
