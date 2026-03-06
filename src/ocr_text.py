"""Stage D.1: Text Region OCR + Block Classification.

For each text column crop:
1. Upscale 2x (or 3x for small text) with Lanczos
2. Run consensus OCR (EasyOCR + Tesseract)
3. Clean and correct output
4. Classify into typed blocks: heading, paragraph, ordered_list,
   caution, warning, note
"""

import re
import sys
import numpy as np
from PIL import Image

# Legacy OCR + correction modules
from ocr_engine import ocr_region, EASYOCR_AVAILABLE
from text_correction import correct_text

# Reuse helpers from legacy extract.py
import extract as _extract

# ── Block type detection patterns ──────────────────────────────────────────

# Require a number followed by a period or paren to be stricter
ORDERED_LIST_RE = re.compile(
    r"^(?:\d{1,2}[\.\)]\s+|Step\s+\d+[\.\:\s]\s*|\(\d+\)\s+)",
    re.IGNORECASE
)
PROCEDURE_HEADER_RE = re.compile(
    r"^\s*(REMOVAL|INSTALLATION|INSPECTION|DISASSEMBLY|REASSEMBLY|ADJUSTMENT|TESTING|DISCONNECT|CONNECT)\b",
    re.IGNORECASE
)
CAUTION_RE  = re.compile(r"^\s*(?:CAUTION|PRECAUTION)\s*[:\-\u2014]?\s*", re.IGNORECASE)
WARNING_RE  = re.compile(r"^\s*WARNING\s*[:\-\u2014]?\s*", re.IGNORECASE)
NOTE_RE     = re.compile(r"^\s*(?:NOTE|NOTICE|IMPORTANT)\s*[:\-\u2014]?\s*", re.IGNORECASE)
HEADING_RE  = re.compile(r"^[A-Z][A-Z0-9\s\-/&,]{3,}$")

# Characters virtually absent from clean technical manual text but common in
# OCR output of diagrams/charts (viscosity charts, wiring diagrams, etc.)
# U+2014 (em dash —) is the most frequent OCR artifact from diagram line/box borders.
_GARBAGE_CHAR_RE  = re.compile('[\\\\£¥€{}~^|\u2014]')
_GARBLE_THRESHOLD = 0.03   # garbage chars / alpha chars ratio
_GARBLE_MIN_ALPHA = 15     # don't judge very short strings
_GARBLE_MIN_COUNT = 3      # need at least this many garbage chars (prevents false positives
                           # from single OCR'd parenthesis like {2} → real "(2)")


def ocr_text_region(col_img: Image.Image, region_label: str,
                    config: dict) -> dict:
    """Upscale a column crop, run OCR, return raw result dict."""
    cfg_ocr = config.get("ocr", {})
    use_consensus = cfg_ocr.get("use_consensus", True)

    # Upscale for better OCR accuracy
    upscaled = _upscale_crop(col_img, cfg_ocr)

    # Run consensus OCR
    result = ocr_region(upscaled, region_label, use_consensus)
    return result


# PHASE 4.1 — Domain Vocabulary Validator
AUTOMOTIVE_CORRECTIONS = {
    "runabout": "runout",
    "freeplay": "free play",
    "preload": "pre-load",
    "tir": "T.I.R.",
    "atf": "ATF",
    "cv": "CV",
}

def _apply_domain_corrections(text: str) -> str:
    """Apply specific automotive domain corrections to the text."""
    for wrong, right in AUTOMOTIVE_CORRECTIONS.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + wrong + r'\b', right, text, flags=re.IGNORECASE)
    return text

def clean_and_correct(raw_text: str, word_data: dict,
                       is_toc: bool, config: dict) -> str:
    """Chain: deduplicate → clean garbage → filter noise → spell correct → domain correct.

    Preserves paragraph structure (blank-line separators) throughout.
    spell_correct_text() uses text.split() which destroys newlines, so we
    process each paragraph independently and rejoin with double newlines.
    """
    text = _extract._deduplicate_text(raw_text)
    text = _extract.clean_ocr_text(text)
    
    # Filter specific, known OCR noise patterns that are not handled by general cleaning
    text = re.sub(r'sd\s*\|\s*—_AMe\s*9S\s*5424-6E', '', text, flags=re.IGNORECASE)
    
    # PHASE 4.2 — OCR Artifact Filter
    # Strip part codes and garbled label fragments
    text = re.sub(r'\b[A-Z]{2}\d{4}-\d[A-Z]-[A-Z]-[A-Z]{2}\b', '', text)
    text = re.sub(r'[a-z]{1,3}\s*\|\s*[—–]\w+\s*\d+\w+', '', text)

    # Process paragraphs separately to preserve blank-line separators.
    # correct_text() → spell_correct_text() uses text.split() which collapses newlines.
    paragraphs = text.split("\n\n")
    corrected_paras = []
    for para in paragraphs:
        if para.strip():
            corrected = correct_text(para, confidence_data=word_data, is_toc=is_toc)
            corrected = _apply_domain_corrections(corrected)
            corrected_paras.append(corrected)
        else:
            corrected_paras.append("")
    return "\n\n".join(corrected_paras)


def classify_blocks(text: str) -> list:
    """Split corrected OCR text into typed TextBlock objects.

    Parses the text line by line, grouping lines into blocks based on
    detected patterns (CAUTION, WARNING, NOTE, ordered lists, headings).
    Falls back to paragraph for everything else.

    Tesseract often produces single-newline output with no blank line separators.
    We pre-process by inserting blank lines before recognized block starters so
    that _collect_block() can split on paragraph boundaries correctly.
    """
    from src.models import TextBlock

    if not text or not text.strip():
        return []

    # Pre-insert blank lines before block starters that Tesseract doesn't separate
    text = _insert_paragraph_breaks(text)
    lines = text.splitlines()
    blocks: list[TextBlock] = []

    last_procedure_type: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip blank lines between blocks
        if not stripped:
            i += 1
            continue

        # ── CAUTION ───────────────────────────────────────────────
        if CAUTION_RE.match(stripped):
            body, i = _collect_block(lines, i)
            text_clean = CAUTION_RE.sub("", body, count=1).strip()
            blocks.append(TextBlock(type="caution", text=text_clean))
            continue

        # ── WARNING ───────────────────────────────────────────────
        if WARNING_RE.match(stripped):
            body, i = _collect_block(lines, i)
            text_clean = WARNING_RE.sub("", body, count=1).strip()
            blocks.append(TextBlock(type="warning", text=text_clean))
            continue

        # ── NOTE ──────────────────────────────────────────────────
        if NOTE_RE.match(stripped):
            body, i = _collect_block(lines, i)
            text_clean = NOTE_RE.sub("", body, count=1).strip()
            blocks.append(TextBlock(type="note", text=text_clean))
            continue

        # ── ORDERED LIST ──────────────────────────────────────────
        if ORDERED_LIST_RE.match(stripped):
            steps, i = _collect_ordered_list(lines, i)
            if steps:
                full_text = "\n".join(steps)
                blocks.append(TextBlock(
                    type="ordered_list", 
                    text=full_text, 
                    steps=steps,
                    procedure_type=last_procedure_type
                ))
                continue

        # ── HEADING ───────────────────────────────────────────────
        level = _detect_heading_level(stripped)
        if level is not None:
            # Peek at next line — if also all-caps, merge as multi-line heading
            heading_lines = [stripped]
            j = i + 1
            while j < len(lines) and len(heading_lines) < 3:
                next_stripped = lines[j].strip()
                if not next_stripped:
                    break
                if _detect_heading_level(next_stripped) is not None:
                    heading_lines.append(next_stripped)
                    j += 1
                else:
                    break
            heading_text = " ".join(heading_lines)
            
            # Detect procedure_type from heading
            m = PROCEDURE_HEADER_RE.search(heading_text)
            if m:
                ptype = m.group(1).lower()
                if "removal" in ptype or "disconnect" in ptype:
                    last_procedure_type = "removal"
                elif "installation" in ptype or "connect" in ptype:
                    last_procedure_type = "installation"
                elif "inspection" in ptype or "testing" in ptype:
                    last_procedure_type = "inspection"
                elif "adjustment" in ptype:
                    last_procedure_type = "adjustment"

            # Level-1 section-code headings (e.g. "1A-6 HEATER AND VENTILATION")
            # sometimes absorb adjacent diagram/table content from OCR. Truncate at
            # the first pipe char (common OCR artifact from table borders) and cap at
            # 100 chars so stray content doesn't pollute the heading.
            if level == 1 and "|" in heading_text:
                heading_text = heading_text.split("|")[0].strip()
            if len(heading_text) > 100:
                heading_text = heading_text[:100].rsplit(" ", 1)[0].strip()
            ratio, count = _garble_ratio(heading_text)
            if ratio > _GARBLE_THRESHOLD and count >= _GARBLE_MIN_COUNT:
                print(f"      [garble] Skipping garbled heading block "
                      f"(ratio={ratio:.0%}, n={count}): {heading_text[:60]!r}")
                i = j
                continue
            blocks.append(TextBlock(type="heading", text=heading_text, level=level))
            i = j
            continue

        # ── PARAGRAPH (default) ───────────────────────────────────
        body, i = _collect_block(lines, i)
        if body.strip():
            ratio, count = _garble_ratio(body)
            if ratio > _GARBLE_THRESHOLD and count >= _GARBLE_MIN_COUNT:
                print(f"      [garble] Skipping garbled paragraph block "
                      f"(ratio={ratio:.0%}, n={count}): {body[:60]!r}")
                continue
            blocks.append(TextBlock(type="paragraph", text=body.strip()))

    return blocks


# ── Internal helpers ──────────────────────────────────────────────────────

def _upscale_crop(img: Image.Image, cfg_ocr: dict) -> Image.Image:
    """Upscale image for OCR. 2x by default, 3x for very small text."""
    factor         = cfg_ocr.get("upscale_factor", 2)
    tiny_threshold = cfg_ocr.get("upscale_tiny_threshold", 20)

    # Estimate line height via horizontal projection
    line_height = _estimate_line_height(img)
    if line_height < tiny_threshold:
        factor = 3

    w, h = img.size
    return img.resize((w * factor, h * factor), Image.LANCZOS)


def _estimate_line_height(img: Image.Image) -> float:
    """Estimate text line height in pixels via horizontal projection."""
    import cv2
    img_np = np.array(img)
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    text   = (255 - gray).astype(float)
    row_sums = np.sum(text, axis=1) / 255.0

    # Count transitions from low→high density as line starts
    threshold = img.width * 0.05
    in_line   = False
    starts    = []
    for y, v in enumerate(row_sums):
        if not in_line and v > threshold:
            starts.append(y)
            in_line = True
        elif in_line and v <= threshold:
            in_line = False

    if len(starts) < 2:
        return 20.0
    gaps = [starts[k + 1] - starts[k] for k in range(len(starts) - 1)]
    return float(np.median(gaps))


def _garble_ratio(text: str) -> tuple[float, int]:
    """Return (ratio, count) of garbage chars relative to alphabetic chars.

    A block is garbled when BOTH conditions hold:
      ratio > _GARBLE_THRESHOLD  (e.g. >3% of alpha chars are garbage)
      count >= _GARBLE_MIN_COUNT (at least 3 garbage chars absolutely)

    The minimum count prevents false positives from a single OCR'd parenthesis
    like '{2}' (only 2 garbage chars) being mistaken for diagram noise.
    """
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < _GARBLE_MIN_ALPHA:
        return 0.0, 0
    count = len(_GARBAGE_CHAR_RE.findall(text))
    return count / alpha, count


def _detect_heading_level(text: str) -> int | None:
    """Return heading level (1, 2, 3) or None if not a heading."""
    if not text or len(text) < 4:
        return None

    # Section code prefix like "1B-7" → level 1
    if re.match(r"^[0-9]+[A-Z]\d*-\d+", text):
        return 1
        
    # Procedure headers (REMOVAL, etc.) → level 3
    if PROCEDURE_HEADER_RE.match(text):
        return 3

    # All-caps → level 2
    if HEADING_RE.match(text) and len(text) <= 80:
        return 2

    # Title-case short line that looks like a section title → level 3
    words = text.split()
    if (2 <= len(words) <= 8
            and all(w[0].isupper() for w in words if w and w[0].isalpha())
            and not text.endswith(".")):
        return 3

    return None


def _collect_block(lines: list[str], start: int) -> tuple[str, int]:
    """Collect lines into a paragraph until a blank line or type change."""
    collected = []
    i = start
    while i < len(lines):
        line    = lines[i]
        stripped = line.strip()

        if not stripped:
            # Peek ahead to see if the block is truly over.
            # It's over if we hit 2+ blank lines or a new typed block.
            is_continuation = False
            for j in range(i + 1, min(i + 3, len(lines))):
                peek_stripped = lines[j].strip()
                if not peek_stripped: # another blank line
                    continue
                # If it's a new typed block, it's not a continuation
                if not (CAUTION_RE.match(peek_stripped) or
                    WARNING_RE.match(peek_stripped) or
                    NOTE_RE.match(peek_stripped) or
                    ORDERED_LIST_RE.match(peek_stripped) or
                    _detect_heading_level(peek_stripped) is not None):
                    is_continuation = True
                break # Found a non-blank line, so we can decide.
            
            if not is_continuation:
                i += 1 # consume the current blank line
                break
            
            # It's a continuation, so just consume the blank line and keep going
            i += 1
            continue

        # Stop if a new typed block starts on a non-blank line
        if i > start and (
            CAUTION_RE.match(stripped) or
            WARNING_RE.match(stripped) or
            NOTE_RE.match(stripped) or
            ORDERED_LIST_RE.match(stripped) or
            _detect_heading_level(stripped) is not None
        ):
            break

        collected.append(stripped)
        i += 1

    return " ".join(collected), i


def _insert_paragraph_breaks(text: str) -> str:
    """Insert blank lines before typed block starters that Tesseract omits.

    Tesseract PSM 6 produces single-newline line breaks within and between
    paragraphs. We detect line-level transitions to block types and insert
    a blank line before them so classify_blocks() can split correctly.

    Also inserts a break when a short line (end of paragraph) is followed by
    a fresh capitalized sentence (start of new paragraph).
    """
    lines = text.splitlines()
    if len(lines) <= 1:
        return text

    result: list[str] = [lines[0]]

    for i in range(1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        prev = result[-1].strip() if result else ""

        # Always ensure blank line before typed block starters
        is_typed = (
            CAUTION_RE.match(stripped) or
            WARNING_RE.match(stripped) or
            NOTE_RE.match(stripped) or
            ORDERED_LIST_RE.match(stripped) or
            _detect_heading_level(stripped) is not None
        )

        if is_typed and prev:
            # Insert blank line separator before this block
            result.append("")
            result.append(line)
            continue

        # Heuristic paragraph break: short previous line + new sentence start
        # A "short" line ends a paragraph when the next line starts a new thought
        if (prev and len(prev) < 60
                and stripped and stripped[0].isupper()
                and prev[-1] in ".!?:)"
                and not ORDERED_LIST_RE.match(stripped)):
            result.append("")

        result.append(line)

    return "\n".join(result)


def _collect_ordered_list(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect numbered list items, joining wrapped continuation lines."""
    steps    = []
    current  = []
    i        = start

    while i < len(lines):
        line     = lines[i]
        stripped = line.strip()

        if not stripped:
            if current:
                steps.append(" ".join(current))
                current = []
            
            # Peek ahead to see if list continues
            is_list_continuation = False
            for j in range(i + 1, min(i + 3, len(lines))):
                peek_stripped = lines[j].strip()
                if not peek_stripped:
                    continue
                if ORDERED_LIST_RE.match(peek_stripped):
                    is_list_continuation = True
                break
            
            if not is_list_continuation:
                i += 1
                break
            
            i += 1
            continue

        if ORDERED_LIST_RE.match(stripped):
            if current:
                steps.append(" ".join(current))
            # Strip the numbering prefix
            text = ORDERED_LIST_RE.sub("", stripped, count=1).strip()
            current = [text]
        elif current:
            # Continuation of previous step (wrapped line)
            current.append(stripped)
        else:
            # This line is not a list item, and we are not in a step. End of list.
            break

        i += 1

    if current:
        steps.append(" ".join(current))

    return steps, i
