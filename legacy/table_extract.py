"""
OpenCV + Tesseract table extraction pipeline for scanned manual pages.

Pipeline:
1. Preprocess: grayscale -> CLAHE -> Sauvola binarize -> deskew -> border removal
2. Region split: detect vertical divider, crop left/right
3. OCR with bounding boxes: image_to_data (TSV) per region
4. Reconstruct grid: cluster words into rows/cols by center coordinates
5. Render clean markdown tables
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

TESS_CONFIG_BLOCK = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'


# ── 1. Preprocessing ──────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray, for_ocr: bool = False) -> np.ndarray:
    """Enhanced preprocessing: grayscale -> CLAHE -> Sauvola binarize.

    Args:
        img_bgr: Input image in BGR format.
        for_ocr: If True, applies more aggressive enhancement for OCR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement — adapts to uneven scan illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Light denoise (preserve edges better than bilateral)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Sauvola-style adaptive binarization — uses local window so faded text
    # regions aren't lost (unlike global Otsu which picks one threshold)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=12
    )

    # Ensure text-is-black, background-is-white (Tesseract expects this)
    if np.mean(bw) < 127:
        bw = 255 - bw

    return bw


def remove_scanner_borders(bw: np.ndarray, margin_pct: float = 0.03) -> np.ndarray:
    """Detect and crop black scanner edges and ruler marks.

    Scans often have black borders or ruler marks on edges that generate
    OCR garbage. This detects large dark regions at the page edges and
    crops them out.
    """
    h, w = bw.shape
    margin_x = int(w * margin_pct)
    margin_y = int(h * margin_pct)

    # Check each edge for solid dark bands
    top_mean = np.mean(bw[:margin_y, :])
    bot_mean = np.mean(bw[h - margin_y:, :])
    left_mean = np.mean(bw[:, :margin_x])
    right_mean = np.mean(bw[:, w - margin_x:])

    y1, y2, x1, x2 = 0, h, 0, w
    dark_thresh = 80  # mostly black

    if top_mean < dark_thresh:
        # Find where the dark band ends
        for row in range(margin_y, min(h // 4, margin_y * 5)):
            if np.mean(bw[row, :]) > 200:
                y1 = row
                break
    if bot_mean < dark_thresh:
        for row in range(h - margin_y, max(h * 3 // 4, h - margin_y * 5), -1):
            if np.mean(bw[row, :]) > 200:
                y2 = row
                break
    if left_mean < dark_thresh:
        for col in range(margin_x, min(w // 4, margin_x * 5)):
            if np.mean(bw[:, col]) > 200:
                x1 = col
                break
    if right_mean < dark_thresh:
        for col in range(w - margin_x, max(w * 3 // 4, w - margin_x * 5), -1):
            if np.mean(bw[:, col]) > 200:
                x2 = col
                break

    if y1 > 0 or y2 < h or x1 > 0 or x2 < w:
        return bw[y1:y2, x1:x2]
    return bw


def deskew(bw: np.ndarray) -> np.ndarray:
    """Straighten slightly rotated scans using Hough line transform."""
    inv = 255 - bw
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 1000:
        return bw

    # Use Hough line transform for more robust angle detection
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=bw.shape[1] // 6, maxLineGap=10)

    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (within 10 degrees)
            if abs(angle) < 10:
                angles.append(angle)

        if angles:
            # Use median angle to be robust against outliers
            angle = float(np.median(angles))
        else:
            # Fallback to minAreaRect
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
    else:
        # Fallback to minAreaRect
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

    if abs(angle) < 0.3:  # don't bother for tiny angles
        return bw
    h, w = bw.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(bw, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ── 2. Region splitting ──────────────────────────────────────────────────

def split_by_vertical_rule(bw: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect vertical divider lines and split the page into regions.

    Returns list of (x, y, w, h) crop rectangles.
    """
    h, w = bw.shape
    inv = 255 - bw

    # Emphasize vertical lines: kernel height = 1/3 of page
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, h // 3))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [(0, 0, w, h)]

    # Find dividers: tall vertical lines not near the page edges
    edge_margin = int(w * 0.12)
    dividers = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        mid = x + cw // 2
        # Must be tall (>40% of page) and away from edges
        if ch > h * 0.4 and edge_margin < mid < w - edge_margin:
            dividers.append(mid)

    if not dividers:
        return [(0, 0, w, h)]

    # Deduplicate close dividers
    dividers.sort()
    unique = [dividers[0]]
    for d in dividers[1:]:
        if d - unique[-1] > 40:
            unique.append(d)

    # Build regions from dividers
    pad = 10
    regions = []
    left = 0
    for d in unique:
        if d - pad - left > 50:
            regions.append((left, 0, max(1, d - pad - left), h))
        left = d + pad
    if w - left > 50:
        regions.append((left, 0, w - left, h))

    return regions


# ── 3. OCR with bounding boxes ───────────────────────────────────────────

def ocr_words(bw_roi: np.ndarray) -> list[dict]:
    """Run Tesseract on a binary ROI and return word bounding boxes."""
    data = pytesseract.image_to_data(
        bw_roi, output_type=Output.DICT, config=TESS_CONFIG_BLOCK
    )
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip('-').isdigit() else -1
        if txt and conf >= 40:
            words.append({
                "text": txt,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "conf": conf,
            })
    return words


# ── 4. Grid reconstruction ───────────────────────────────────────────────

def cluster_1d(vals: list[int], thresh: int) -> list[int]:
    """Cluster 1D values that are within `thresh` of each other.
    Returns list of cluster centers (sorted).

    Compares each new value against the running cluster *center* rather
    than the last element to prevent chain-drift (where many small steps
    merge values that are far apart in aggregate).
    """
    vals = sorted(vals)
    clusters: list[list[int]] = []
    for v in vals:
        if not clusters:
            clusters.append([v])
        else:
            center = sum(clusters[-1]) / len(clusters[-1])
            if abs(v - center) > thresh:
                clusters.append([v])
            else:
                clusters[-1].append(v)
    return [int(sum(c) / len(c)) for c in clusters]


def words_to_grid(words: list[dict], region_width: int = 0) -> list[list[str]]:
    """Cluster words into a row/col grid by their center coordinates.

    Args:
        words: OCR word dicts with x, y, w, h, text, conf keys.
        region_width: width of the ROI in pixels (for edge filtering).
    """
    if not words:
        return []

    # ── Step 1: Filter edge noise (rotated sidebar text near region edges) ──
    if region_width > 0:
        margin = max(20, int(region_width * 0.05))
        words = [w for w in words
                 if not ((w["x"] + w["w"] // 2 < margin
                          or w["x"] + w["w"] // 2 > region_width - margin)
                         and len(w["text"]) <= 2)]
    if not words:
        return []

    ys = [w["y"] + w["h"] // 2 for w in words]
    xs = [w["x"] + w["w"] // 2 for w in words]

    # ── Step 2: Row clustering (scales with median text height) ─────────
    med_h = int(np.median([w["h"] for w in words]))
    row_centers = cluster_1d(ys, thresh=max(8, med_h))

    # ── Step 3: Adaptive column threshold ───────────────────────────────
    # Use region_width / 12 as baseline (~130 px for a 1600-px half-page).
    # This keeps genuinely separate columns apart while merging words that
    # belong to the same column but are slightly offset (e.g. parenthetical
    # units next to the word they modify).
    col_thresh = max(50, region_width // 10) if region_width > 0 else 50
    col_centers = cluster_1d(xs, thresh=col_thresh)

    # ── Step 4: Assign words to grid cells ──────────────────────────────
    def nearest(v, centers):
        return min(range(len(centers)), key=lambda i: abs(v - centers[i]))

    grid = [[[] for _ in col_centers] for _ in row_centers]

    for w in words:
        r = nearest(w["y"] + w["h"] // 2, row_centers)
        c = nearest(w["x"] + w["w"] // 2, col_centers)
        grid[r][c].append(w)

    # ── Step 5: Convert word lists to strings ───────────────────────────
    out = []
    for row in grid:
        cells = []
        for cell_words in row:
            cell_words_sorted = sorted(cell_words, key=lambda w: w["x"])
            cells.append(" ".join(w["text"] for w in cell_words_sorted).strip())
        if any(cells):
            out.append(cells)

    if not out:
        return []

    # ── Step 6: Remove fully empty columns ──────────────────────────────
    num_cols = len(out[0])
    keep = [any(out[r][c] for r in range(len(out))) for c in range(num_cols)]
    out = [[row[c] for c in range(num_cols) if keep[c]] for row in out]

    if not out or not out[0]:
        return []

    # ── Step 7: Remove garbage columns (>70 % single-char non-digit) ───
    if len(out[0]) > 2:
        garbage_cols = set()
        for c in range(len(out[0])):
            cells = [row[c].strip() for row in out if row[c].strip()]
            if not cells:
                garbage_cols.add(c)
                continue
            garbage_count = sum(
                1 for cell in cells
                if len(cell) <= 2 and not cell.replace('.', '').replace('-', '').isdigit()
            )
            if len(cells) >= 3 and garbage_count > len(cells) * 0.7:
                garbage_cols.add(c)
        remaining = len(out[0]) - len(garbage_cols)
        if garbage_cols and remaining >= 2:
            out = [[row[c] for c in range(len(out[0])) if c not in garbage_cols]
                   for row in out]

    # ── Step 8: Merge sparse complementary columns ──────────────────────
    # Two adjacent columns where < 15 % of rows have BOTH filled are likely
    # fragments of one logical column.  Merge them together.
    if len(out[0]) > 2:
        changed = True
        while changed and len(out[0]) > 2:
            changed = False
            for c in range(len(out[0]) - 1):
                col_a = [row[c].strip() for row in out]
                col_b = [row[c + 1].strip() for row in out]
                both = sum(1 for a, b in zip(col_a, col_b) if a and b)
                either = sum(1 for a, b in zip(col_a, col_b) if a or b)
                if either == 0:
                    for row in out:
                        del row[c + 1]
                    changed = True
                    break
                overlap = both / either
                if overlap < 0.20:
                    for row in out:
                        a, b = row[c].strip(), row[c + 1].strip()
                        row[c] = (a + " " + b).strip() if a and b else (a or b)
                        del row[c + 1]
                    changed = True
                    break

    # ── Step 9: Merge parenthetical suffix columns ───────────────────
    # A column whose content is mostly short parenthetical text (e.g.
    # "(mm)", "(kPa)", "(kg)") is really part of the preceding column.
    if out and len(out[0]) > 2:
        for c in range(len(out[0]) - 1, 0, -1):
            cells = [row[c].strip() for row in out if row[c].strip()]
            if len(cells) < 2:
                continue
            short_or_paren = sum(
                1 for cell in cells
                if cell.startswith("(") or len(cell) <= 4
            )
            if short_or_paren > len(cells) * 0.6:
                for row in out:
                    a, b = row[c - 1].strip(), row[c].strip()
                    row[c - 1] = (a + " " + b).strip() if a and b else (a or b)
                    del row[c]

    # ── Step 10: Trim trailing garbage rows ─────────────────────────
    # Remove rows from the bottom where all non-empty cells are very
    # short (OCR noise from rotated sidebar text or graphical elements).
    while out:
        row = out[-1]
        non_empty = [c.strip() for c in row if c.strip()]
        if not non_empty:
            out.pop()
        elif len(non_empty) <= 1 and all(len(c) <= 3 for c in non_empty):
            out.pop()
        else:
            break

    return out


# ── 5. Markdown rendering ────────────────────────────────────────────────

def to_markdown(rows: list[list[str]]) -> str:
    """Convert a grid of strings to a markdown table."""
    if not rows:
        return ""

    cols = max(len(r) for r in rows)
    norm = [r + [""] * (cols - len(r)) for r in rows]

    # Use first row as header if it looks like one (has text in most cells)
    first_filled = sum(1 for c in norm[0] if c.strip())
    if first_filled >= cols * 0.4:
        header = norm[0]
        data_rows = norm[1:]
    else:
        header = [f"Col {i+1}" for i in range(cols)]
        data_rows = norm

    sep = ["---"] * cols

    lines = []
    lines.append("| " + " | ".join(h if h else " " for h in header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in data_rows:
        cleaned = [c.replace("\n", " ").replace("|", "/").strip() for c in r]
        if any(cleaned):
            lines.append("| " + " | ".join(c if c else " " for c in cleaned) + " |")
    return "\n".join(lines)


# ── 6. Detection: is this page a table? ──────────────────────────────────

def _has_table_structure(bw: np.ndarray) -> bool:
    """Quick check: does the page have enough line structure to be a table?

    Looks for horizontal rules and/or vertical dividers that indicate
    tabular content rather than flowing text.
    """
    h, w = bw.shape
    inv = 255 - bw

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    h_cnts, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_count = sum(1 for c in h_cnts if cv2.boundingRect(c)[2] > w * 0.15)

    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)
    v_cnts, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_count = sum(1 for c in v_cnts if cv2.boundingRect(c)[3] > h * 0.15)

    # A single tall vertical line is just a column divider (two-column text),
    # NOT table evidence.  Require either multiple vertical lines or many
    # horizontal rules to confirm actual table structure.
    return (h_count >= 2 and v_count >= 2) or h_count >= 5 or v_count >= 3


def _is_table_grid(grid: list[list[str]]) -> bool:
    """Reject grids that look like flowing text rather than tabular data.

    Two-column text pages get split into regions and OCR'd, producing grids
    where cells contain broken sentence fragments.  Real tables have short,
    self-contained cell content and many empty cells.
    """
    if not grid or len(grid) < 3:
        return False

    ncols = len(grid[0])

    # Collect word counts for every non-empty cell
    word_counts = []
    total_cells = 0
    filled_cells = 0
    for row in grid:
        for cell in row:
            total_cells += 1
            text = cell.strip()
            if text:
                filled_cells += 1
                word_counts.append(len(text.split()))

    if not word_counts:
        return False

    word_counts.sort()
    median_words = word_counts[len(word_counts) // 2]
    fill_ratio = filled_cells / total_cells if total_cells else 0

    # Flowing text: high median word count per cell (sentences split across cells)
    if median_words > 5:
        return False

    # Dense text: most cells filled AND cells have multiple words → prose, not table
    if fill_ratio > 0.70 and median_words > 3:
        return False

    # Flowing text check: cells that start with a lowercase letter are
    # mid-sentence fragments ("of system due the", "and may cause").
    # Tables have cells starting with uppercase or digits ("Inch", "25.4").
    texts = [cell.strip() for row in grid for cell in row if cell.strip()]
    if len(texts) > 5:
        lc_starts = sum(1 for t in texts if t[0].islower())
        if lc_starts > len(texts) * 0.45:
            return False

    return True


# ── 7. End-to-end entry point ─────────────────────────────────────────────

def detect_and_extract_table(pil_img: Image.Image) -> dict | None:
    """Main entry point: detect table structure and extract as markdown.

    Takes a PIL Image (RGB), returns dict with 'markdown' key or None.
    """
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    bw = preprocess(img_bgr)
    bw = deskew(bw)
    bw = remove_scanner_borders(bw)

    # Quick structural check — skip expensive OCR if no table lines found
    if not _has_table_structure(bw):
        return None

    # Split into regions at vertical dividers
    regions = split_by_vertical_rule(bw)

    md_parts = []
    for (x, y, rw, rh) in regions:
        roi = bw[y:y + rh, x:x + rw]
        words = ocr_words(roi)
        grid = words_to_grid(words, region_width=rw)
        if grid and len(grid) >= 2 and len(grid[0]) >= 2 and _is_table_grid(grid):
            md_parts.append(to_markdown(grid))

    if not md_parts:
        return None

    return {
        "markdown": "\n\n".join(md_parts),
        "is_table": True,
        "regions": len(regions),
    }
