"""
Multi-engine OCR module with consensus logic.

Runs EasyOCR (GPU, primary) + Tesseract (CPU, secondary) on text regions
and uses consensus to pick the best result. Supports region-type-specific
Tesseract PSM modes and reading order reconstruction.
"""

import os
import re
import sys
import numpy as np
import pytesseract
from PIL import Image

# Fix Windows console encoding for EasyOCR's progress bars
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# EasyOCR — may not be available
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Common English words for quality scoring
_COMMON_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for',
    'is', 'it', 'be', 'as', 'are', 'was', 'has', 'had', 'not', 'but', 'from',
    'with', 'this', 'that', 'all', 'can', 'will', 'may', 'if', 'do', 'no',
    'see', 'use', 'set', 'get', 'new', 'one', 'two', 'also', 'each', 'when',
    'then', 'than', 'into', 'only', 'after', 'before', 'should', 'must',
    'check', 'replace', 'remove', 'install', 'inspect', 'adjust', 'connect',
    'figure', 'section', 'table', 'page', 'part', 'system', 'engine',
    'oil', 'air', 'water', 'pump', 'belt', 'valve', 'bolt', 'torque',
    'vehicle', 'motor', 'electrical', 'control', 'switch', 'cable', 'wire',
    'assembly', 'service', 'maintenance', 'general', 'information',
    'heating', 'ventilation', 'conditioning', 'steering', 'suspension',
    'transaxle', 'clutch', 'fuel', 'cooling', 'exhaust', 'emission',
    'battery', 'diagnosis', 'specification', 'procedure', 'description',
    'compressor', 'evaporator', 'condenser', 'refrigerant', 'pressure',
    'temperature', 'fluid', 'filter', 'brake', 'sensor', 'solenoid',
}

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ── Singleton EasyOCR reader ─────────────────────────────────────────────

_easyocr_reader: "easyocr.Reader | None" = None


def _get_easyocr() -> "easyocr.Reader":
    """Lazily initialize EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=True)
    return _easyocr_reader


# ── PSM Mode Selection ───────────────────────────────────────────────────

# Tesseract Page Segmentation Modes for different region types
PSM_MODES = {
    "Text": 6,           # Assume a single uniform block of text
    "ListItem": 6,       # Same as text
    "Caption": 7,        # Treat as a single text line
    "SectionHeader": 7,  # Single line header
    "PageHeader": 7,     # Single line
    "PageFooter": 7,     # Single line
    "Table": 6,          # Block of text (cells processed separately)
    "Figure": 11,        # Sparse text (labels scattered on diagram)
    "Picture": 11,       # Same as figure
    "default": 6,        # Default: single block
}


def _get_psm(region_label: str) -> int:
    """Get appropriate Tesseract PSM mode for a region type."""
    return PSM_MODES.get(region_label, PSM_MODES["default"])


# ── Quality Scoring ──────────────────────────────────────────────────────

def _word_quality_score(text: str) -> float:
    """Score OCR output quality based on dictionary word ratio and patterns.

    Returns a score from 0.0 (garbage) to 1.0 (perfect).
    """
    if not text or not text.strip():
        return 0.0

    words = re.findall(r'[a-zA-Z]{3,}', text)
    if not words:
        return 0.5  # No alphabetic words to judge (could be numbers/symbols)

    # Dictionary word ratio
    dict_count = sum(1 for w in words if w.lower() in _COMMON_WORDS)
    dict_ratio = dict_count / len(words)

    # Penalize garbage patterns
    penalties = 0.0

    # Internal case mixing (e.g., "soueUAajUIeW")
    mixed_case = sum(1 for w in words if re.search(r'[a-z][A-Z]', w))
    penalties += (mixed_case / len(words)) * 0.3

    # Very long words that aren't compound (likely garbled)
    long_words = sum(1 for w in words if len(w) > 15)
    penalties += (long_words / len(words)) * 0.2

    # Excessive non-alphanumeric characters
    non_alpha = sum(1 for c in text if not c.isalnum() and c not in ' .,;:!?()-\n\t"\'')
    if len(text) > 0:
        noise_ratio = non_alpha / len(text)
        if noise_ratio > 0.15:
            penalties += 0.2

    score = dict_ratio * 0.7 + 0.3 - penalties
    return max(0.0, min(1.0, score))


# ── OCR Engines ──────────────────────────────────────────────────────────

def ocr_tesseract(img: Image.Image | np.ndarray, region_label: str = "Text") -> dict:
    """Run Tesseract OCR on an image region.

    Returns dict with 'text', 'confidence', 'word_data' keys.
    """
    psm = _get_psm(region_label)
    config = f'--oem 1 --psm {psm} -c preserve_interword_spaces=1'

    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    else:
        pil_img = img

    # Get text
    text = pytesseract.image_to_string(pil_img, config=config)

    # Get per-word confidence data
    try:
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=config)
        confidences = []
        word_data = {}
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip('-').isdigit() else -1
            if word and conf >= 0:
                confidences.append(conf)
                word_data[word] = conf
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
    except Exception:
        avg_conf = 0
        word_data = {}

    return {
        "text": text.strip(),
        "confidence": avg_conf,
        "word_data": word_data,
        "engine": "tesseract",
    }


def ocr_easyocr(img: Image.Image | np.ndarray) -> dict:
    """Run EasyOCR on an image region.

    Returns dict with 'text', 'confidence', 'word_data' keys.
    """
    if not EASYOCR_AVAILABLE:
        return {"text": "", "confidence": 0, "word_data": {}, "engine": "easyocr"}

    reader = _get_easyocr()

    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img

    try:
        results = reader.readtext(img_np, detail=1)
    except Exception as e:
        return {"text": "", "confidence": 0, "word_data": {}, "engine": "easyocr"}

    if not results:
        return {"text": "", "confidence": 0, "word_data": {}, "engine": "easyocr"}

    # Sort results by vertical position (top-to-bottom) then horizontal (left-to-right)
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    # Group into lines by Y coordinate proximity
    lines = []
    current_line = []
    prev_y = None
    for bbox, text, conf in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        if prev_y is not None and abs(y_center - prev_y) > 20:
            if current_line:
                lines.append(current_line)
            current_line = []
        current_line.append((bbox, text, conf))
        prev_y = y_center
    if current_line:
        lines.append(current_line)

    # Sort each line left-to-right and build text
    text_lines = []
    confidences = []
    word_data = {}
    for line in lines:
        line.sort(key=lambda r: r[0][0][0])
        line_text = " ".join(r[1] for r in line)
        text_lines.append(line_text)
        for _, word_text, conf in line:
            confidences.append(conf * 100)  # EasyOCR uses 0-1 scale
            for w in word_text.split():
                word_data[w] = int(conf * 100)

    full_text = "\n".join(text_lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return {
        "text": full_text,
        "confidence": avg_conf,
        "word_data": word_data,
        "engine": "easyocr",
    }


# ── Consensus Logic ─────────────────────────────────────────────────────

def _consensus(tess_result: dict, easy_result: dict) -> dict:
    """Pick the best OCR result using consensus logic.

    When both engines produce text:
    - If they agree closely → use the one with higher confidence
    - If they disagree → prefer the one with more dictionary-valid words
    """
    tess_text = tess_result["text"]
    easy_text = easy_result["text"]

    # If one is empty, use the other
    if not tess_text.strip() and not easy_text.strip():
        return tess_result
    if not tess_text.strip():
        return easy_result
    if not easy_text.strip():
        return tess_result

    # Score both outputs
    tess_quality = _word_quality_score(tess_text)
    easy_quality = _word_quality_score(easy_text)

    # Combine quality score with engine confidence
    tess_score = tess_quality * 0.6 + (tess_result["confidence"] / 100) * 0.4
    easy_score = easy_quality * 0.6 + (easy_result["confidence"] / 100) * 0.4

    if easy_score > tess_score:
        winner = easy_result
    else:
        winner = tess_result

    # Merge word confidence data from both engines
    merged_word_data = {}
    merged_word_data.update(tess_result["word_data"])
    for word, conf in easy_result["word_data"].items():
        if word in merged_word_data:
            merged_word_data[word] = max(merged_word_data[word], conf)
        else:
            merged_word_data[word] = conf

    winner["word_data"] = merged_word_data
    return winner


# ── Main API ─────────────────────────────────────────────────────────────

def ocr_region(img: Image.Image | np.ndarray,
               region_label: str = "Text",
               use_consensus: bool = True) -> dict:
    """OCR a single image region using multi-engine consensus.

    Args:
        img: Image of the region (PIL or numpy).
        region_label: Surya region label for PSM mode selection.
        use_consensus: If True, run both engines and pick best result.

    Returns:
        Dict with 'text', 'confidence', 'word_data', 'engine' keys.
    """
    tess_result = ocr_tesseract(img, region_label)

    if use_consensus and EASYOCR_AVAILABLE:
        easy_result = ocr_easyocr(img)
        return _consensus(tess_result, easy_result)

    return tess_result


def ocr_page_regions(img: Image.Image, regions: list, use_consensus: bool = True) -> list[dict]:
    """OCR multiple regions from a page image in reading order.

    Args:
        img: Full page PIL Image.
        regions: List of Region objects (from layout_analysis).
        use_consensus: Whether to use multi-engine consensus.

    Returns:
        List of dicts with 'text', 'confidence', 'region', 'word_data' keys.
    """
    results = []
    for region in regions:
        # Crop the region from the page
        region_img = region.crop_image(img)

        # OCR the region
        ocr_result = ocr_region(region_img, region.label, use_consensus)
        ocr_result["region"] = region

        results.append(ocr_result)

    return results
