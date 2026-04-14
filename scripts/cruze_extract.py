#!/usr/bin/env python3
"""Cruze Manual Extraction Script.

Extracts text and figures from the 2013 Chevrolet Cruze service manual PDF
using PyMuPDF native text extraction (no OCR needed — the PDF has embedded text).

Outputs:
  build_cruze/document.json   — structured page data with text blocks + figure refs
  build_cruze/assets/         — extracted figure images (PNG)
  build_cruze/toc.json        — table of contents from PDF metadata

Usage:
  python scripts/cruze_extract.py [--pages START-END]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "cruze" / "cruze-manual.pdf"
BUILD_DIR = PROJECT_ROOT / "build_cruze"
ASSETS_DIR = BUILD_DIR / "assets"

# ── Patterns ────────────────────────────────────────────────────────────────

# Figure caption: "Fig. 123: Some Description"
FIG_CAPTION_RE = re.compile(r"^Fig\.\s*(\d+)\s*:\s*(.+)", re.IGNORECASE)
# Numbered procedure step: "1. Do something" or "12. Do something"
STEP_RE = re.compile(r"^(\d{1,3})\.\s+\S")
# WARNING/CAUTION/NOTE labels
WARNING_RE = re.compile(r"^(WARNING|CAUTION|NOTE)\s*:", re.IGNORECASE)
# ALL-CAPS heading (at least 4 chars, may include spaces/slashes)
HEADING_RE = re.compile(r"^[A-Z][A-Z /&\-]{3,}$")
# Page footer: "2013 Chevrolet Cruze"
FOOTER_RE = re.compile(r"^\s*2013\s+Chevrolet\s+Cruze\s*$", re.IGNORECASE)
# Section path footer: "2013 CATEGORY Subsection - Variant - Cruze"
SECTION_PATH_RE = re.compile(
    r"^2013\s+([A-Z][A-Z &/]+?)\s+(.+?)\s*-\s*Cruze\s*$", re.IGNORECASE
)
# Courtesy line (skip)
COURTESY_RE = re.compile(r"^Courtesy of\s+", re.IGNORECASE)
# Bullet character used in the PDF
BULLET_CHAR = "\uf0a1"


def classify_line(line: str) -> str:
    """Classify a text line by type."""
    stripped = line.strip()
    if not stripped:
        return "blank"
    if FOOTER_RE.match(stripped):
        return "footer"
    if SECTION_PATH_RE.match(stripped):
        return "section_path"
    if COURTESY_RE.match(stripped):
        return "courtesy"
    if FIG_CAPTION_RE.match(stripped):
        return "figure_caption"
    if WARNING_RE.match(stripped):
        return "warning"
    if STEP_RE.match(stripped):
        return "step"
    if stripped.startswith(BULLET_CHAR):
        return "bullet"
    if HEADING_RE.match(stripped) and len(stripped) >= 4:
        return "heading"
    return "text"


def extract_section_path(lines: list[str]) -> str | None:
    """Extract section path from page footer lines."""
    for line in reversed(lines):
        m = SECTION_PATH_RE.match(line.strip())
        if m:
            category = m.group(1).strip()
            subsection = m.group(2).strip()
            return f"{category} > {subsection}"
    return None


def extract_figures_from_page(doc: fitz.Document, page_idx: int,
                              page_text_lines: list[str]) -> list[dict]:
    """Extract embedded images and pair with figure captions."""
    page = doc[page_idx]
    images = page.get_images()
    figures = []

    # Collect all figure captions on this page
    captions = {}
    for line in page_text_lines:
        m = FIG_CAPTION_RE.match(line.strip())
        if m:
            fig_num = int(m.group(1))
            caption = m.group(2).strip()
            captions[fig_num] = caption

    # Filter to substantial, unique images (PDFs often embed duplicates)
    seen_xrefs = set()
    substantial_images = []
    for img_info in images:
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            img_data = doc.extract_image(xref)
        except Exception:
            continue
        if not img_data or len(img_data.get("image", b"")) < 500:
            continue
        w = img_data.get("width", 0)
        h = img_data.get("height", 0)
        if w < 50 or h < 50:
            continue
        substantial_images.append((img_data, w, h))

    for img_idx, (img_data, w, h) in enumerate(substantial_images):
        ext = img_data.get("ext", "png")
        fig_id = f"cruze_p{page_idx + 1}_img{img_idx}"

        # Try to match with a caption
        matched_caption = None
        matched_fig_num = None
        caption_keys = sorted(captions.keys())
        if len(caption_keys) == 1:
            matched_fig_num = caption_keys[0]
            matched_caption = captions[matched_fig_num]
            fig_id = f"cruze_p{page_idx + 1}_fig{matched_fig_num}"
        elif img_idx < len(caption_keys):
            matched_fig_num = caption_keys[img_idx]
            matched_caption = captions[matched_fig_num]
            fig_id = f"cruze_p{page_idx + 1}_fig{matched_fig_num}"

        # Save image (keep largest version if fig_id collides)
        img_path = ASSETS_DIR / f"{fig_id}.{ext}"
        if not img_path.exists() or len(img_data["image"]) > img_path.stat().st_size:
            img_path.write_bytes(img_data["image"])

        figures.append({
            "fig_id": fig_id,
            "fig_num": matched_fig_num,
            "caption": matched_caption,
            "asset_path": str(img_path.relative_to(PROJECT_ROOT)),
            "width": w,
            "height": h,
            "page": page_idx + 1,
        })

    # Deduplicate by fig_id — keep the largest image version
    seen = {}
    for fig in figures:
        fid = fig["fig_id"]
        if fid not in seen or (fig["width"] * fig["height"]) > (seen[fid]["width"] * seen[fid]["height"]):
            seen[fid] = fig
    return list(seen.values())


def build_text_blocks(lines: list[str]) -> list[dict]:
    """Parse page text lines into structured blocks."""
    blocks = []
    current_block = {"type": "text", "lines": []}

    def flush():
        if current_block["lines"]:
            text = "\n".join(current_block["lines"]).strip()
            if text:
                blocks.append({
                    "type": current_block["type"],
                    "text": text,
                })
        current_block["lines"] = []
        current_block["type"] = "text"

    for line in lines:
        cls = classify_line(line)
        stripped = line.strip()

        if cls in ("footer", "section_path", "courtesy", "blank"):
            continue

        if cls == "heading":
            flush()
            blocks.append({"type": "heading", "text": stripped})
            continue

        if cls == "figure_caption":
            flush()
            m = FIG_CAPTION_RE.match(stripped)
            blocks.append({
                "type": "figure_caption",
                "text": stripped,
                "fig_num": int(m.group(1)),
                "caption": m.group(2).strip(),
            })
            continue

        if cls == "warning":
            flush()
            current_block["type"] = "warning"
            current_block["lines"].append(stripped)
            continue

        if cls == "step":
            # If we're already in a procedure, continue; otherwise start one
            if current_block["type"] != "procedure":
                flush()
                current_block["type"] = "procedure"
            current_block["lines"].append(stripped)
            continue

        if cls == "bullet":
            # Bullets belong to the current block (often procedure sub-conditions)
            clean = stripped.lstrip(BULLET_CHAR).strip()
            current_block["lines"].append(f"  - {clean}")
            continue

        # Regular text
        if current_block["type"] == "procedure" and not STEP_RE.match(stripped):
            # Continuation of a procedure step
            current_block["lines"].append(stripped)
        elif current_block["type"] == "warning":
            current_block["lines"].append(stripped)
        else:
            if current_block["type"] != "text":
                flush()
                current_block["type"] = "text"
            current_block["lines"].append(stripped)

    flush()
    return blocks


def extract_toc(doc: fitz.Document) -> list[dict]:
    """Extract PDF table of contents."""
    raw_toc = doc.get_toc()
    toc = []
    for level, title, page_num in raw_toc:
        toc.append({
            "level": level,
            "title": title.strip(),
            "page": page_num,
        })
    return toc


def run_extraction(page_range: tuple[int, int] | None = None):
    """Main extraction pipeline."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening {PDF_PATH}...")
    doc = fitz.open(str(PDF_PATH))
    total_pages = doc.page_count
    print(f"Total pages: {total_pages}")

    # ── Extract TOC ─────────────────────────────────────────────────
    toc = extract_toc(doc)
    toc_path = BUILD_DIR / "toc.json"
    with open(toc_path, "w", encoding="utf-8") as f:
        json.dump(toc, f, indent=2, ensure_ascii=False)
    print(f"TOC saved: {len(toc)} entries → {toc_path}")

    # ── Build TOC page→section lookup ───────────────────────────────
    # For each page, find the most recent TOC entry at the deepest level
    toc_sections = {}  # page_num → section title path
    for entry in toc:
        toc_sections[entry["page"]] = entry["title"]

    # ── Process pages ───────────────────────────────────────────────
    start = (page_range[0] - 1) if page_range else 0
    end = page_range[1] if page_range else total_pages

    pages_data = []
    total_figures = 0
    total_blocks = 0

    for page_idx in range(start, end):
        page = doc[page_idx]
        page_num = page_idx + 1
        raw_text = page.get_text()
        lines = raw_text.split("\n")

        # Extract section path from footer
        section_path = extract_section_path(lines)

        # Build structured text blocks
        text_blocks = build_text_blocks(lines)
        total_blocks += len(text_blocks)

        # Extract figures
        figures = extract_figures_from_page(doc, page_idx, lines)
        total_figures += len(figures)

        pages_data.append({
            "page_num": page_num,
            "section_path": section_path,
            "text_blocks": text_blocks,
            "figures": figures,
            "char_count": len(raw_text),
        })

        if page_num % 500 == 0:
            print(f"  Processed {page_num}/{end} pages "
                  f"({total_blocks} blocks, {total_figures} figures)...")

    doc.close()

    # ── Save document.json ──────────────────────────────────────────
    document = {
        "doc_id": "chevy_cruze_2013",
        "source_pdf": str(PDF_PATH.relative_to(PROJECT_ROOT)),
        "page_count": total_pages,
        "pages_extracted": len(pages_data),
        "total_blocks": total_blocks,
        "total_figures": total_figures,
        "pages": pages_data,
    }

    doc_path = BUILD_DIR / "document.json"
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    print(f"\nExtraction complete:")
    print(f"  Pages:   {len(pages_data)}")
    print(f"  Blocks:  {total_blocks}")
    print(f"  Figures: {total_figures}")
    print(f"  Output:  {doc_path}")
    print(f"  Assets:  {ASSETS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Cruze service manual")
    parser.add_argument("--pages", type=str, default=None,
                        help="Page range, e.g. 1-100")
    args = parser.parse_args()

    page_range = None
    if args.pages:
        parts = args.pages.split("-")
        page_range = (int(parts[0]), int(parts[1]))

    run_extraction(page_range)
