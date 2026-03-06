"""Stage C: Layout Segmentation.

Runs Surya layout detection on preprocessed page crops, then applies:
- Overlapping region deduplication
- Column detection via vertical projection profile
- Page type classification (text / diagram / table / mixed / toc)
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from src.models import PreprocessResult

# Pull in Surya wrapper from legacy
from layout_analysis import analyze_layout, Region, SURYA_AVAILABLE


def analyze_page_layout(pre: PreprocessResult, config: dict) -> dict:
    """Run layout analysis on a preprocessed page.

    Returns a dict with:
      page_type, regions, columns, has_multiple_columns
    """
    cfg_col = config.get("column_detection", {})
    cfg_layout = config.get("layout", {})

    # Load preprocessed image
    pre_img = Image.open(pre.preprocessed_path).convert("RGB")

    # Run Surya on the preprocessed image
    layout = analyze_layout(pre_img, min_confidence=cfg_layout.get("min_confidence", 0.3))
    page_type = layout.classify_page()

    # Deduplicate overlapping regions
    regions = _merge_overlapping(
        layout.regions,
        threshold=cfg_layout.get("overlap_iou_threshold", 0.50)
    )

    # Surya often misses figures and tables in scanned technical documents.
    # Fall back to box detection: every table and diagram in this manual
    # is enclosed in a printed border. Detect those borders, then distinguish
    # tables (internal grid lines) from diagrams (no grid).
    cv_boxes = _detect_boxes_cv(pre_img, regions, config)
    if cv_boxes:
        regions = regions + cv_boxes

    # Reclassify page type now that CV figures may have been added
    from layout_analysis import PageLayout
    updated_layout = PageLayout(regions, pre_img.width, pre_img.height)
    page_type = updated_layout.classify_page()

    # Detect columns on the full preprocessed image
    columns = detect_columns(pre_img, cfg_col)

    # Complexity classification for routing
    is_complex, reasons = _classify_complexity(regions, columns, pre_img)

    return {
        "page_type": page_type,
        "regions": regions,
        "columns": columns,
        "has_multiple_columns": len(columns) > 1,
        "is_complex": is_complex,
        "complexity_reasons": reasons,
    }


def _classify_complexity(regions: list, columns: list, img: Image.Image) -> tuple[bool, list[str]]:
    """Determine if a page is COMPLEX based on layout features."""
    reasons = []
    
    # 1. More than one image/figure region detected
    fig_count = sum(1 for r in regions if r.is_figure)
    if fig_count > 1:
        reasons.append(f"multiple_figures({fig_count})")
    
    # 2. Two-column layout detected
    if len(columns) >= 2:
        reasons.append(f"multi_column({len(columns)})")
    
    # 3. Table structure present
    if any(r.is_table for r in regions):
        reasons.append("table_detected")
    
    # 4. Check for special keywords (requires quick OCR or is handled by stage D)
    # Since we don't have OCR yet, we'll rely on regions and columns for now, 
    # or add a fast OCR pass if needed. But stage D will re-route if needed.
    # For now, let's keep it based on visual layout.
    
    # 5. Numbered list AND image region (often complex procedures)
    # We'll detect this during extraction if needed, but let's see if we can 
    # spot "List" regions if Surya detected them.
    if any(r.label == "List" for r in regions) and fig_count > 0:
        reasons.append("list_and_figure")

    is_complex = len(reasons) > 0
    return is_complex, reasons


def _detect_boxes_cv(img: Image.Image, existing_regions: list, config: dict) -> list:
    """Detect bordered rectangular regions (tables and diagrams) via line detection.

    Every table and diagram in this manual is enclosed in a printed border.
    We detect those borders using morphological line detection, then distinguish:
      - Table: box contains internal grid lines (rows/columns)
      - Picture: box has no internal grid (it's a diagram)

    Much more reliable than blob analysis for scanned technical documents.
    """
    img_gray = np.array(img.convert("L"))
    h, w = img_gray.shape

    _, binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Minimum line length: 5% of the smaller page dimension
    min_side = max(30, int(min(h, w) * 0.05))

    # Detect long horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_side, 1))
    h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect long vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_side))
    v_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Combine and dilate to close gaps at corners (scanning artifacts, slight skew)
    grid = cv2.add(h_lines, v_lines)
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    grid_dilated = cv2.dilate(grid, d_kernel, iterations=5)

    contours, _ = cv2.findContours(grid_dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    existing_bboxes = [(int(r.x1), int(r.y1), int(r.x2), int(r.y2))
                       for r in existing_regions]

    found = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x2, y2 = x + bw, y + bh

        # Must be large enough to be a meaningful box
        if bw * bh < h * w * 0.02:
            continue
        if bw < min_side or bh < min_side:
            continue

        # Skip if substantially covered by an existing Surya region
        covered = False
        for ex1, ey1, ex2, ey2 in existing_bboxes:
            ox1, oy1 = max(x, ex1), max(y, ey1)
            ox2, oy2 = min(x2, ex2), min(y2, ey2)
            if ox1 < ox2 and oy1 < oy2:
                if (ox2 - ox1) * (oy2 - oy1) / (bw * bh) > 0.5:
                    covered = True
                    break
        if covered:
            continue

        # Label all detected boxes as Picture — Stage D will try the table
        # extractor first and route to table or figure based on the result.
        print(f"      [box] ({x},{y})-({x2},{y2}) {bw}×{bh}px → box detected")

        found.append(Region(
            label="Picture",
            bbox=[float(x), float(y), float(x2), float(y2)],
            confidence=0.5,
        ))

    return found


def detect_columns(img: Image.Image, config: dict) -> list[tuple[int, int, int, int]]:
    """Detect text columns using vertical projection profile.

    Analyzes the full page to find content bounds and column gutters.
    Skips the top header_skip_frac and bottom footer_skip_frac of content to
    avoid running headers/footers that span both columns from masking the gutter.
    Returns list of (x1, y1, x2, y2) bounding boxes for each column.
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    h, w = gray.shape

    # Already binary (0=text, 255=background). Invert for counting.
    text_pixels = (255 - gray).astype(float)

    search_band    = config.get("search_band", [0.30, 0.70])
    min_gutter_px  = config.get("min_gutter_width_px", 10)
    density_ratio  = config.get("density_ratio", 0.33)
    smooth_frac    = config.get("smoothing_fraction", 0.25)
    header_skip    = config.get("header_skip_frac", 0.10)
    footer_skip    = config.get("footer_skip_frac", 0.03)

    # ── Vertical content bounds via horizontal projection ──
    row_sums = np.sum(text_pixels, axis=1) / 255.0
    content_rows = np.where(row_sums > w * 0.005)[0]
    if len(content_rows) < 10:
        return [(0, 0, w, h)]

    y_top = max(0, int(content_rows[0]) - 20)
    y_bot = min(h, int(content_rows[-1]) + 20)

    # Skip top header_skip_frac and bottom footer_skip_frac of content
    # to prevent full-width running headers from masking the column gutter
    content_height = y_bot - y_top
    y_analysis_top = y_top + int(content_height * header_skip)
    y_analysis_bot = y_bot - int(content_height * footer_skip)

    # ── Horizontal content bounds via vertical projection ──
    content_area = text_pixels[y_analysis_top:y_analysis_bot, :]
    col_sums = np.sum(content_area, axis=0) / 255.0
    content_cols = np.where(col_sums > (y_bot - y_top) * 0.005)[0]
    if len(content_cols) < 10:
        return [(0, y_top, w, y_bot)]

    x_left  = max(0, int(content_cols[0]) - 20)
    x_right = min(w, int(content_cols[-1]) + 20)
    content_w = x_right - x_left
    if content_w < 400:
        return [(x_left, y_top, x_right, y_bot)]

    # ── Search for gutter in middle band ──
    local_sums = col_sums[x_left:x_right]
    s0 = int(content_w * search_band[0])
    s1 = int(content_w * search_band[1])
    zone = local_sums[s0:s1]
    if len(zone) < 20:
        return [(x_left, y_top, x_right, y_bot)]

    kernel = max(3, min(41, int(len(zone) * smooth_frac)))
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(zone, np.ones(kernel) / kernel, mode="same")

    min_idx = int(np.argmin(smoothed))
    min_val = float(smoothed[min_idx])

    left_density  = float(np.mean(local_sums[:s0])) if s0 > 0 else 0
    right_density = float(np.mean(local_sums[s1:])) if s1 < len(local_sums) else 0
    avg_density   = (left_density + right_density) / 2 or 1.0

    if min_val < avg_density * density_ratio:
        gc = x_left + s0 + min_idx
        # Expand gutter to all positions below the midpoint between min and threshold.
        # Use min_val + (density_ratio * avg_density - min_val) * 0.6 to ensure we
        # include the gutter center even when it's close to the threshold.
        gap_thresh = min_val + (avg_density * density_ratio - min_val) * 0.6
        gap_thresh = max(gap_thresh, min_val * 1.5)
        gl, gr = gc, gc
        while gl > x_left and col_sums[gl - 1] < gap_thresh:
            gl -= 1
        while gr < x_right - 1 and col_sums[gr + 1] < gap_thresh:
            gr += 1
        if (gr - gl) >= min_gutter_px:
            return [(x_left, y_top, gl, y_bot), (gr, y_top, x_right, y_bot)]
        # Fallback: even if width < min_gutter_px, split at gutter center if gutter detected
        return [(x_left, y_top, gc, y_bot), (gc, y_top, x_right, y_bot)]

    return [(x_left, y_top, x_right, y_bot)]


def _merge_overlapping(regions: list, threshold: float = 0.50) -> list:
    """Remove regions that substantially overlap with higher-confidence regions."""
    if len(regions) <= 1:
        return regions

    sorted_r = sorted(regions, key=lambda r: r.confidence, reverse=True)
    kept = []

    for candidate in sorted_r:
        duplicate = False
        for existing in kept:
            ix1 = max(candidate.x1, existing.x1)
            iy1 = max(candidate.y1, existing.y1)
            ix2 = min(candidate.x2, existing.x2)
            iy2 = min(candidate.y2, existing.y2)
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(candidate.area, existing.area)
                if smaller > 0 and intersection / smaller > threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(candidate)

    return kept
