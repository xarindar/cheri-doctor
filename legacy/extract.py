"""
Metro Manual PDF Extraction Pipeline — V2
Extracts text (via multi-engine OCR), tables, and diagrams from scanned PDF manuals.

Pipeline:
  PDF Page → PyMuPDF render (300 DPI)
  → Preprocessing (CLAHE + Sauvola binarization + deskew + border removal)
  → Layout Analysis (Surya: text/table/figure/header regions)
  → Per-region processing:
      TEXT regions   → Multi-engine OCR (EasyOCR + Tesseract consensus)
      TABLE regions  → img2table structure + per-cell OCR
      FIGURE regions → Save PNG + Claude Vision description
  → Post-OCR correction (SymSpell + automotive dictionary + dehyphenation)
  → RAG-optimized output assembly
"""

import json
import os
import re
import sys
import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pathlib import Path

# V2 modules
from layout_analysis import analyze_layout, SURYA_AVAILABLE
from ocr_engine import ocr_region, ocr_page_regions, EASYOCR_AVAILABLE
from table_extract_v2 import extract_table
from table_extract import preprocess, deskew, remove_scanner_borders
from text_correction import correct_text
from vision_describe import describe_diagram, is_vision_available, load_description_cache, save_description_cache
from assembler import assemble_output

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Metro Project")
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DPI = 300

PDFS = {
    "manual": BASE_DIR / "1990 Geo Metro Manual.pdf",
    "vert_sup": BASE_DIR / "Vert Sup.pdf",
}

OUTPUT_DIR = BASE_DIR / "output"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Common English words for garbage detection
COMMON_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for',
    'is', 'it', 'be', 'as', 'are', 'was', 'has', 'had', 'not', 'but', 'from',
    'with', 'this', 'that', 'all', 'can', 'will', 'may', 'if', 'do', 'no',
    'see', 'use', 'set', 'get', 'new', 'one', 'two', 'also', 'each', 'when',
    'then', 'than', 'into', 'only', 'after', 'before', 'should', 'must',
    'check', 'replace', 'remove', 'install', 'inspect', 'adjust', 'connect',
    'disconnect', 'figure', 'section', 'table', 'page', 'part', 'system',
    'engine', 'brake', 'oil', 'air', 'water', 'pump', 'belt', 'valve',
    'bolt', 'nut', 'screw', 'torque', 'front', 'rear', 'left', 'right',
    'upper', 'lower', 'vehicle', 'motor', 'electrical', 'control', 'switch',
    'cable', 'wire', 'hose', 'case', 'panel', 'assembly', 'service',
    'maintenance', 'general', 'information', 'heating', 'ventilation',
    'conditioning', 'steering', 'suspension', 'transaxle', 'clutch',
    'fuel', 'cooling', 'exhaust', 'emission', 'ignition', 'battery',
    'diagnosis', 'specification', 'procedure', 'description', 'operation',
    'every', 'miles', 'months', 'pressure', 'temperature', 'level',
    'fluid', 'filter', 'change', 'rotation', 'inspection', 'wheel', 'tire',
    'metric', 'fasteners', 'chassis', 'lubrication', 'compressor', 'mount',
    'evaporator', 'thermistor', 'magnetic', 'condenser', 'receiver',
    'refrigerant', 'replacement', 'identification', 'disconnecting',
    'specifications', 'specification', 'performance', 'circuit', 'wiring',
    'connector', 'alternator', 'distributor', 'carburetor', 'cylinder',
    'crankshaft', 'camshaft', 'piston', 'gasket', 'bearing', 'seal',
    'manual', 'automatic', 'hydraulic', 'vacuum', 'relay', 'fuse', 'lamp',
    'light', 'speed', 'sensor', 'solenoid', 'regulator', 'resistor',
    'capacitor', 'diode', 'terminal', 'ground', 'power', 'voltage',
    'current', 'signal', 'output', 'input', 'high', 'low', 'open', 'closed',
    'test', 'measure', 'reading', 'normal', 'abnormal', 'damage', 'wear',
    'leak', 'crack', 'corrosion', 'rust', 'clean', 'dry', 'wet',
    'tighten', 'loosen', 'apply', 'coat', 'fill', 'drain', 'flush',
    'blower', 'heater', 'core', 'radiator', 'thermostat', 'fan',
    'spring', 'shock', 'strut', 'arm', 'bar', 'link', 'rod', 'shaft',
    'gear', 'ratio', 'drive', 'axle', 'joint', 'boot', 'cover',
    'body', 'door', 'window', 'glass', 'mirror', 'bumper', 'fender',
    'hood', 'trunk', 'roof', 'floor', 'seat', 'dash', 'instrument',
}


def clean_ocr_text(text: str) -> str:
    """Remove garbage lines produced by rotated sidebar text in scanned pages."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        words = re.findall(r'[a-zA-Z]{3,}', stripped)
        if len(words) < 2:
            cleaned.append(line)
            continue
        recognized = sum(1 for w in words if w.lower() in COMMON_WORDS)
        unrecognized = [w for w in words if w.lower() not in COMMON_WORDS]
        recognized_ratio = recognized / len(words)
        if recognized == 0 and len(words) >= 3:
            continue
        if recognized_ratio < 0.2 and len(words) >= 2:
            garbled = sum(1 for w in unrecognized
                         if (re.search(r'[a-z][A-Z]', w) or len(w) > 12))
            if garbled >= 2 or (garbled >= 1 and len(words) <= 3):
                continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# Pattern for section-page codes like "0A-3", "6B1-12", "8A-25"
SECTION_CODE_RE = re.compile(r'\b([O0-9]+[A-Z]\d*)-(\d+)\b')


def detect_section(text: str) -> dict:
    """Extract section info from the running header."""
    if not text:
        return {"section": None, "section_code": None, "section_page": None}
    
    lines = text.strip().splitlines()
    if not lines:
        return {"section": None, "section_code": None, "section_page": None}

    candidates = lines[:3] + lines[-3:]
    for line in candidates:
        line = line.strip()
        match = SECTION_CODE_RE.search(line)
        if match:
            section_code = match.group(1).replace('O', '0')
            section_page = match.group(2)
            section_name = SECTION_CODE_RE.sub('', line).strip()
            section_name = re.sub(r'^[\s\-—]+|[\s\-—]+$', '', section_name)
            return {
                "section": section_name if section_name else None,
                "section_code": section_code,
                "section_page": section_page,
            }

    return {"section": None, "section_code": None, "section_page": None}


def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"completed_pages": []}


def save_progress(progress_file: Path, progress: dict):
    with open(progress_file, "w") as f:
        json.dump(progress, f)


# ── V2 Page Processing ──────────────────────────────────────────────────

def _detect_page_columns(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Detect text columns on the full page image using vertical projection profile.

    Analyzes the full preprocessed page to find content bounds and column gutters,
    bypassing Surya's unreliable region detection for these scanned pages.

    Returns list of (x1, y1, x2, y2) bounding boxes for each detected column,
    or a single bbox for the full content area if single-column.
    """
    img_np = np.array(img)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    h, w = gray.shape

    # The preprocessed image is already binary (0=text, 255=background).
    # Invert so text pixels = 255 for summing.
    text_pixels = (255 - gray).astype(float)

    # ── Find vertical content bounds (top/bottom) via horizontal projection ──
    row_sums = np.sum(text_pixels, axis=1) / 255.0  # count of text pixels per row
    row_threshold = w * 0.005  # row must have >0.5% of width filled
    content_rows = np.where(row_sums > row_threshold)[0]
    if len(content_rows) < 10:
        return [(0, 0, w, h)]

    y_top = max(0, int(content_rows[0]) - 20)
    y_bot = min(h, int(content_rows[-1]) + 20)

    # ── Find horizontal content bounds (left/right) via vertical projection ──
    content_area = text_pixels[y_top:y_bot, :]
    col_sums = np.sum(content_area, axis=0) / 255.0  # count of text pixels per column

    col_threshold = (y_bot - y_top) * 0.005
    content_cols = np.where(col_sums > col_threshold)[0]
    if len(content_cols) < 10:
        return [(0, y_top, w, y_bot)]

    x_left = max(0, int(content_cols[0]) - 20)
    x_right = min(w, int(content_cols[-1]) + 20)

    content_w = x_right - x_left
    if content_w < 400:  # too narrow for two columns
        return [(x_left, y_top, x_right, y_bot)]

    # ── Search for column gutter in the middle 30%-70% of content ──
    local_sums = col_sums[x_left:x_right].copy()
    search_start_rel = int(content_w * 0.30)
    search_end_rel = int(content_w * 0.70)
    search_zone = local_sums[search_start_rel:search_end_rel]

    if len(search_zone) < 20:
        return [(x_left, y_top, x_right, y_bot)]

    # Smooth to avoid noise spikes
    kernel_size = min(41, max(3, len(search_zone) // 4))
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(search_zone, np.ones(kernel_size) / kernel_size, mode='same')

    min_idx = int(np.argmin(smoothed))
    min_val = float(smoothed[min_idx])

    # Average text density in left and right content areas (outside search zone)
    left_density = float(np.mean(local_sums[:search_start_rel])) if search_start_rel > 0 else 0
    right_density = float(np.mean(local_sums[search_end_rel:])) if search_end_rel < len(local_sums) else 0
    avg_density = (left_density + right_density) / 2 if (left_density + right_density) > 0 else 1

    # Gutter: minimum must be much less dense than surrounding text
    if avg_density > 0 and min_val < avg_density * 0.20:
        gutter_center = x_left + search_start_rel + min_idx

        # Expand to find full gutter width (contiguous low-density band)
        gap_threshold = avg_density * 0.25
        gutter_left = gutter_center
        gutter_right = gutter_center
        while gutter_left > x_left and col_sums[gutter_left - 1] < gap_threshold:
            gutter_left -= 1
        while gutter_right < x_right - 1 and col_sums[gutter_right + 1] < gap_threshold:
            gutter_right += 1

        gutter_width = gutter_right - gutter_left
        if gutter_width >= 10:  # at least 10 pixels wide
            left_col = (x_left, y_top, gutter_left, y_bot)
            right_col = (gutter_right, y_top, x_right, y_bot)
            return [left_col, right_col]

    # No gutter found → single column
    return [(x_left, y_top, x_right, y_bot)]


def _filter_overlapping_regions(regions: list) -> list:
    """Remove regions that substantially overlap with other regions.

    Surya sometimes detects the same content area multiple times with
    slightly different bounding boxes. Keep the highest-confidence region
    when two regions overlap by more than 50%.
    """
    if len(regions) <= 1:
        return regions

    # Sort by confidence descending so we keep the best ones
    sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
    kept = []

    for candidate in sorted_regions:
        is_duplicate = False
        for existing in kept:
            # Calculate intersection
            ix1 = max(candidate.x1, existing.x1)
            iy1 = max(candidate.y1, existing.y1)
            ix2 = min(candidate.x2, existing.x2)
            iy2 = min(candidate.y2, existing.y2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                smaller_area = min(candidate.area, existing.area)
                if smaller_area > 0 and intersection / smaller_area > 0.50:
                    is_duplicate = True
                    break

        if not is_duplicate:
            kept.append(candidate)

    return kept


def _deduplicate_text(text: str) -> str:
    """Remove duplicate paragraphs/blocks from OCR output.

    When overlapping regions are OCR'd, the same text block appears
    multiple times. This detects and removes repeated blocks.
    """
    if not text:
        return text

    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) <= 1:
        return text

    # Remove exact duplicate paragraphs (preserving order of first occurrence)
    seen = set()
    unique = []
    for para in paragraphs:
        # Normalize whitespace for comparison
        normalized = ' '.join(para.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(para)
        elif not normalized:
            unique.append(para)

    result = '\n\n'.join(unique)

    # Also check for repeated lines within a single block
    lines = result.splitlines()
    if len(lines) > 5:
        # Check if the second half is a repeat of the first half
        mid = len(lines) // 2
        first_half = ' '.join(lines[:mid].copy())
        second_half = ' '.join(lines[mid:mid + len(lines[:mid])].copy())
        # Use a similarity check — if >80% of words match, it's a duplicate
        words1 = set(first_half.lower().split())
        words2 = set(second_half.lower().split())
        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            if overlap > 0.80:
                result = '\n'.join(lines[:mid])

    return result


def preprocess_image(img: Image.Image) -> Image.Image:
    """Apply enhanced preprocessing to page image for better OCR."""
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    bw = preprocess(img_bgr, for_ocr=True)
    bw = deskew(bw)
    bw = remove_scanner_borders(bw)

    # Convert back to RGB PIL Image
    rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def process_page_v2(doc, page_num: int, out_dir: Path,
                    skip_vision: bool = False,
                    use_consensus: bool = True) -> dict:
    """Process a single page using the V2 pipeline.

    Strategy:
    - Use Surya for page TYPE classification (text / diagram / table / mixed / toc)
    - For text/toc pages: bypass Surya regions entirely and do full-page
      column detection + per-column OCR (Surya's region detection is unreliable
      on these scanned pages — it often detects only part of the content area)
    - For diagram pages: save image + optional vision AI + sparse OCR
    - For table pages: img2table extraction + fallback OCR
    - For mixed pages: column OCR for text + Surya regions for figure extraction
    """
    page = doc[page_num]
    page_label = f"page_{page_num + 1:04d}"

    # ── Step 1: Render ───────────────────────────────────────────────
    zoom = DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # ── Step 2: Preprocess ───────────────────────────────────────────
    img_preprocessed = preprocess_image(img)

    # ── Step 3: Layout analysis (for page type only) ─────────────────
    layout = analyze_layout(img)  # Use original image for layout detection
    page_type = layout.classify_page()
    layout.regions = _filter_overlapping_regions(layout.regions)

    # ── Step 4: Process based on page type ───────────────────────────
    text_parts = []
    table_md_parts = []
    diagram_description = ""
    word_data = {}
    files = []

    if page_type in ("text", "toc", "mixed"):
        # ── TEXT PAGES: full-page column detection + per-column OCR ──
        # This bypasses Surya's unreliable region detection and instead
        # finds content bounds and column gutters on the full page image.
        columns = _detect_page_columns(img_preprocessed)

        for col_bbox in columns:
            x1, y1, x2, y2 = col_bbox
            col_img = img_preprocessed.crop((x1, y1, x2, y2))
            result = ocr_region(col_img, "Text", use_consensus)
            if result["text"].strip():
                text_parts.append(result["text"].strip())
            word_data.update(result.get("word_data", {}))

        # For mixed pages, also extract figure/table regions from Surya
        if page_type == "mixed":
            for region in layout.regions:
                if region.is_figure:
                    figure_img = region.crop_image(img, padding=10)
                    fig_path = out_dir / "pages" / f"{page_label}_diagram.png"
                    fig_path.parent.mkdir(parents=True, exist_ok=True)
                    figure_img.save(str(fig_path), "PNG")
                    files.append(f"{page_label}_diagram.png")
                    if not skip_vision and is_vision_available():
                        desc = describe_diagram(figure_img, cache_key=page_label)
                        if desc:
                            diagram_description = desc

                elif region.is_table:
                    region_img = region.crop_image(img)
                    table_result = extract_table(region_img)
                    if table_result:
                        table_md_parts.append(table_result["markdown"])

    elif page_type == "diagram":
        # ── DIAGRAM PAGES: save image + vision AI + sparse OCR ──
        img_path = out_dir / "pages" / f"{page_label}_diagram.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(img_path), "PNG")
        files.append(f"{page_label}_diagram.png")

        if not skip_vision and is_vision_available():
            desc = describe_diagram(img, cache_key=page_label)
            if desc:
                diagram_description = desc

        # OCR any visible text labels (sparse text mode)
        result = ocr_region(img_preprocessed, "Figure", use_consensus)
        if result["text"].strip():
            text_parts.append(result["text"].strip())
        word_data.update(result.get("word_data", {}))

    elif page_type == "table":
        # ── TABLE PAGES: img2table extraction + OCR fallback ──
        table_result = extract_table(img)
        if table_result:
            table_md_parts.append(table_result["markdown"])

        # Also OCR for any non-table text
        columns = _detect_page_columns(img_preprocessed)
        for col_bbox in columns:
            x1, y1, x2, y2 = col_bbox
            col_img = img_preprocessed.crop((x1, y1, x2, y2))
            result = ocr_region(col_img, "Text", use_consensus)
            if result["text"].strip():
                text_parts.append(result["text"].strip())
            word_data.update(result.get("word_data", {}))

        # Save image for reference
        img_path = out_dir / "pages" / f"{page_label}_diagram.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(img_path), "PNG")
        files.append(f"{page_label}_diagram.png")

    # ── Fallback: if nothing was extracted, try full-page PSM 3 OCR ──
    if not text_parts and not table_md_parts:
        config = '--oem 1 --psm 3 -c preserve_interword_spaces=1'
        fallback_text = pytesseract.image_to_string(img_preprocessed, config=config)
        if fallback_text.strip():
            text_parts.append(fallback_text.strip())

    # ── Step 5: Combine, clean, and correct text ─────────────────────
    raw_text = "\n\n".join(text_parts)
    raw_text = _deduplicate_text(raw_text)

    section_info = detect_section(raw_text)
    text = clean_ocr_text(raw_text)

    is_toc = page_type == "toc"
    text = correct_text(text, confidence_data=word_data, is_toc=is_toc)

    table_md = "\n\n".join(table_md_parts) if table_md_parts else ""

    # Save image for mixed pages that didn't get a figure extracted
    if page_type == "mixed" and not files:
        img_path = out_dir / "pages" / f"{page_label}_diagram.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(img_path), "PNG")
        files.append(f"{page_label}_diagram.png")

    # ── Build result dict ────────────────────────────────────────────
    result = {
        "page_num": page_num + 1,
        "label": page_label,
        "type": page_type,
        "text_length": len(text.strip()),
        "section": section_info["section"],
        "section_code": section_info["section_code"],
        "section_page": section_info["section_page"],
        "text": text,
        "table_md": table_md,
        "diagram_description": diagram_description,
        "files": files,
    }

    return result


def process_pdf(key: str, pdf_path: Path,
                page_range: tuple[int, int] | None = None,
                skip_vision: bool = False,
                reprocess: bool = False,
                use_consensus: bool = True):
    """Process a full PDF (or a page range for testing)."""
    out_dir = OUTPUT_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)

    progress_file = out_dir / "progress.json"
    vision_cache_file = out_dir / "vision_cache.json"

    if reprocess:
        progress = {"completed_pages": []}
    else:
        progress = load_progress(progress_file)

    # Load vision description cache
    load_description_cache(vision_cache_file)

    doc = fitz.open(str(pdf_path))
    total = doc.page_count

    start = (page_range[0] - 1) if page_range else 0
    end = min(page_range[1], total) if page_range else total

    # Print pipeline status
    print(f"\n{'='*60}")
    print(f"Metro Manual Extraction Pipeline V2")
    print(f"{'='*60}")
    print(f"  PDF:        {pdf_path.name}")
    print(f"  Pages:      {start + 1} to {end} (of {total} total)")
    print(f"  Output:     {out_dir}")
    print(f"  Layout:     {'Surya' if SURYA_AVAILABLE else 'Fallback (full-page)'}")
    print(f"  OCR:        {'EasyOCR + Tesseract consensus' if EASYOCR_AVAILABLE and use_consensus else 'Tesseract only'}")
    print(f"  Vision AI:  {'Enabled' if not skip_vision and is_vision_available() else 'Disabled'}")
    print(f"  Reprocess:  {'Yes' if reprocess else 'No (skip completed)'}")
    print(f"{'='*60}\n")

    all_pages = progress.get("pages", [])

    for i in range(start, end):
        page_num_1based = i + 1
        if not reprocess and page_num_1based in progress["completed_pages"]:
            print(f"  Skipping page {page_num_1based} (already processed)")
            continue

        print(f"  Processing page {page_num_1based} of {total}...", end=" ", flush=True)

        try:
            result = process_page_v2(doc, i, out_dir,
                                     skip_vision=skip_vision,
                                     use_consensus=use_consensus)
        except Exception as e:
            err_msg = str(e).encode('ascii', errors='replace').decode('ascii')
            print(f"[ERROR: {err_msg}]")
            continue

        sec = f", {result['section_code']}" if result.get('section_code') else ""
        tbl = ", table" if result.get('table_md') else ""
        vis = ", +vision" if result.get('diagram_description') else ""
        print(f"[{result['type']}, {result['text_length']} chars{sec}{tbl}{vis}]")

        # Remove text/table_md/diagram_description from progress (keep it small)
        progress_entry = {k: v for k, v in result.items()
                         if k not in ('text', 'table_md', 'diagram_description')}

        # Update or replace in all_pages
        all_pages = [p for p in all_pages if p["page_num"] != page_num_1based]
        all_pages.append(result)

        progress_pages = [p for p in progress.get("pages", [])
                         if p["page_num"] != page_num_1based]
        progress_pages.append(progress_entry)

        if page_num_1based not in progress["completed_pages"]:
            progress["completed_pages"].append(page_num_1based)
        progress["pages"] = progress_pages
        save_progress(progress_file, progress)

        # Save vision cache periodically
        save_description_cache(vision_cache_file)

    doc.close()

    # ── Assembly ─────────────────────────────────────────────────────
    all_pages_sorted = sorted(all_pages, key=lambda p: p["page_num"])

    print(f"\nAssembling output...")
    assemble_output(all_pages_sorted, out_dir, pdf_path.stem)

    # Save final vision cache
    save_description_cache(vision_cache_file)

    print(f"\nDone! Output written to {out_dir}")


def main():
    page_range = None
    keys_to_process = list(PDFS.keys())
    skip_vision = False
    reprocess = False
    use_consensus = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--pages" and i + 1 < len(args):
            parts = args[i + 1].split("-")
            page_range = (int(parts[0]), int(parts[1]))
            i += 2
        elif args[i] == "--only" and i + 1 < len(args):
            keys_to_process = [args[i + 1]]
            i += 2
        elif args[i] == "--skip-vision":
            skip_vision = True
            i += 1
        elif args[i] == "--reprocess":
            reprocess = True
            i += 1
        elif args[i] == "--tesseract-only":
            use_consensus = False
            i += 1
        else:
            i += 1

    for key in keys_to_process:
        pdf_path = PDFS[key]
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping.")
            continue
        process_pdf(key, pdf_path, page_range,
                    skip_vision=skip_vision,
                    reprocess=reprocess,
                    use_consensus=use_consensus)


if __name__ == "__main__":
    main()
