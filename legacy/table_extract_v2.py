"""
Advanced table extraction module using img2table + fallback to OpenCV pipeline.

Pipeline:
1. img2table (primary) — proper cell boundary detection including merged cells
2. Per-cell OCR for fine-grained results
3. Validation to reject non-table grids (numbered lists, flowing text)
4. Fallback chain: img2table → OpenCV pipeline (table_extract.py) → plain OCR
"""

import os
import tempfile
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Add Tesseract to PATH for img2table
os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + os.pathsep + os.environ.get('PATH', '')

try:
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
except ImportError:
    IMG2TABLE_AVAILABLE = False

# Import original table_extract as fallback
from table_extract import detect_and_extract_table as _legacy_extract

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ── img2table extraction ────────────────────────────────────────────────

def _img2table_extract(img: Image.Image) -> dict | None:
    """Extract tables using img2table library.

    Returns dict with 'markdown' key or None if no tables found.
    """
    if not IMG2TABLE_AVAILABLE:
        return None

    # img2table needs a file path
    tmp_path = os.path.join(tempfile.gettempdir(), '_metro_table_tmp.png')
    img.save(tmp_path, "PNG")

    try:
        tess_ocr = TesseractOCR(lang='eng')
        doc = Img2TableImage(src=tmp_path)
        tables = doc.extract_tables(
            ocr=tess_ocr,
            implicit_rows=True,
            implicit_columns=True,
            borderless_tables=False,
            min_confidence=50,
        )
    except Exception as e:
        print(f"  [table_v2] img2table error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not tables:
        return None

    md_parts = []
    for table in tables:
        df = table.df
        if df is None or df.empty:
            continue

        # Validate: reject tables that look like flowing text
        if not _validate_table_df(df):
            continue

        # Convert DataFrame to markdown
        md = _df_to_markdown(df)
        if md:
            md_parts.append(md)

    if not md_parts:
        return None

    return {
        "markdown": "\n\n".join(md_parts),
        "is_table": True,
        "method": "img2table",
        "table_count": len(md_parts),
    }


def _validate_table_df(df) -> bool:
    """Validate that a DataFrame represents a real table, not flowing text."""
    rows, cols = df.shape

    if rows < 2 or cols < 2:
        return False

    # Check cell content: tables have short, self-contained cells
    word_counts = []
    filled = 0
    total = 0
    lc_starts = 0

    for _, row in df.iterrows():
        for cell in row:
            total += 1
            text = str(cell).strip() if cell is not None else ""
            if text:
                filled += 1
                word_counts.append(len(text.split()))
                if text and text[0].islower():
                    lc_starts += 1

    if not word_counts:
        return False

    word_counts.sort()
    median_words = word_counts[len(word_counts) // 2]
    fill_ratio = filled / total if total else 0

    # Flowing text: high word count per cell
    if median_words > 5:
        return False

    # Dense text: most cells filled with multiple words → prose
    if fill_ratio > 0.70 and median_words > 3:
        return False

    # Mid-sentence fragments (lowercase starts)
    if filled > 5 and lc_starts / filled > 0.45:
        return False

    return True


def _df_to_markdown(df) -> str:
    """Convert a pandas DataFrame to a markdown table."""
    if df.empty:
        return ""

    cols = list(df.columns)
    rows = df.values.tolist()

    # Determine header: use first row if columns are just indices
    if all(isinstance(c, int) or (isinstance(c, str) and c.startswith("0")) for c in cols):
        # No meaningful column names — use first row as header
        if rows:
            header = [str(c).strip() if c else " " for c in rows[0]]
            data_rows = rows[1:]
        else:
            return ""
    else:
        header = [str(c).strip() if c else " " for c in cols]
        data_rows = rows

    num_cols = len(header)
    sep = ["---"] * num_cols

    lines = []
    lines.append("| " + " | ".join(h if h else " " for h in header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    for row in data_rows:
        cells = []
        for c in row:
            text = str(c).strip() if c is not None else ""
            text = text.replace("\n", " ").replace("|", "/")
            cells.append(text if text else " ")
        if any(c.strip() for c in cells):
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ── Per-cell OCR (for regions Surya identified as table) ─────────────────

def ocr_table_cells(img: Image.Image, cell_bboxes: list[tuple]) -> list[list[str]]:
    """OCR individual table cells for better accuracy.

    Args:
        img: Full table region image.
        cell_bboxes: List of (row, col, x1, y1, x2, y2) tuples.

    Returns:
        2D grid of cell text strings.
    """
    if not cell_bboxes:
        return []

    # Determine grid dimensions
    max_row = max(c[0] for c in cell_bboxes) + 1
    max_col = max(c[1] for c in cell_bboxes) + 1
    grid = [["" for _ in range(max_col)] for _ in range(max_row)]

    img_np = np.array(img)

    for row, col, x1, y1, x2, y2 in cell_bboxes:
        # Crop cell with small padding
        pad = 3
        cy1 = max(0, int(y1) - pad)
        cy2 = min(img_np.shape[0], int(y2) + pad)
        cx1 = max(0, int(x1) - pad)
        cx2 = min(img_np.shape[1], int(x2) + pad)

        cell_img = img_np[cy1:cy2, cx1:cx2]
        if cell_img.size == 0:
            continue

        # OCR with PSM 7 (single text line)
        try:
            cell_pil = Image.fromarray(cell_img)
            text = pytesseract.image_to_string(
                cell_pil, config='--oem 1 --psm 7'
            ).strip()
            grid[row][col] = text
        except Exception:
            pass

    return grid


# ── Main Entry Point ─────────────────────────────────────────────────────

def extract_table(img: Image.Image, use_fallback: bool = True) -> dict | None:
    """Extract tables from an image using img2table with legacy fallback.

    Args:
        img: PIL Image (RGB) of the table region.
        use_fallback: If True, fall back to OpenCV pipeline on failure.

    Returns:
        Dict with 'markdown', 'is_table', 'method' keys, or None.
    """
    # Try img2table first
    result = _img2table_extract(img)
    if result:
        return result

    # Fallback to legacy OpenCV pipeline
    if use_fallback:
        legacy = _legacy_extract(img)
        if legacy:
            legacy["method"] = "opencv_legacy"
            return legacy

    return None
