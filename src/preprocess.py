"""Stage B: Page Image Preprocessing.

For each rasterized page:
1. Compute content bounding box (crop margins/scanner artifacts)
2. Deskew (straighten rotated scans)
3. Normalize (CLAHE contrast + adaptive threshold + denoise)
4. Remove scanner borders

Outputs preprocessed images to build/pages_pre/.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from src.models import PreprocessResult
from src.utils import resolve_path

# Import proven preprocessing functions from legacy pipeline
from table_extract import preprocess as legacy_preprocess
from table_extract import deskew as legacy_deskew
from table_extract import remove_scanner_borders as legacy_remove_borders


def preprocess_page(page_meta: dict, config: dict,
                    project_root: Path) -> PreprocessResult:
    """Full preprocessing pipeline for one rasterized page.

    Args:
        page_meta: Dict with page_num, path, width, height.
        config: Pipeline config dict.
        project_root: Project root directory.

    Returns:
        PreprocessResult with preprocessed image path and metadata.
    """
    build_dir = resolve_path(config["pipeline"]["build_dir"], project_root)
    pre_dir = build_dir / "pages_pre"
    pre_dir.mkdir(parents=True, exist_ok=True)

    page_num = page_meta["page_num"]
    img = Image.open(page_meta["path"])
    img_np = np.array(img)
    original_size = (img_np.shape[1], img_np.shape[0])  # (width, height)

    # Convert to BGR for OpenCV functions
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Step 1: Normalize (CLAHE + adaptive threshold + denoise)
    bw = legacy_preprocess(img_bgr, for_ocr=True)

    # Step 2: Deskew
    bw, skew_angle = _deskew_with_angle(bw)

    # Step 3: Remove scanner borders
    bw = legacy_remove_borders(bw)

    # Step 4: Compute content bounding box
    cfg_crop = config.get("preprocess", {}).get("content_crop", {})
    content_bbox = compute_content_bbox(bw, cfg_crop)

    # Save preprocessed image
    out_path = pre_dir / f"page_{page_num:04d}.png"
    cv2.imwrite(str(out_path), bw)

    return PreprocessResult(
        page_num=page_num,
        preprocessed_path=str(out_path),
        content_bbox=content_bbox,
        skew_angle=skew_angle,
        original_size=original_size,
    )


def compute_content_bbox(bw: np.ndarray,
                         config: dict | None = None) -> tuple[int, int, int, int]:
    """Find the bounding box of actual content using projection profiles.

    Analyzes horizontal and vertical ink density to find where text/content
    starts and ends, ignoring empty margins.

    Args:
        bw: Binary image (0=text, 255=background).
        config: Optional crop config with thresholds.

    Returns:
        (x1, y1, x2, y2) bounding box of content area.
    """
    if config is None:
        config = {}

    h, w = bw.shape[:2]
    margin = config.get("margin_px", 40)
    row_thresh_pct = config.get("row_threshold_pct", 0.005)
    col_thresh_pct = config.get("col_threshold_pct", 0.005)

    # Invert: text pixels become 255
    text_pixels = (255 - bw).astype(float)

    # Horizontal projection → find top/bottom content bounds
    row_sums = np.sum(text_pixels, axis=1) / 255.0
    row_threshold = w * row_thresh_pct
    content_rows = np.where(row_sums > row_threshold)[0]

    if len(content_rows) < 5:
        return (0, 0, w, h)

    y1 = max(0, int(content_rows[0]) - margin)
    y2 = min(h, int(content_rows[-1]) + margin)

    # Vertical projection → find left/right content bounds
    content_area = text_pixels[y1:y2, :]
    col_sums = np.sum(content_area, axis=0) / 255.0
    col_threshold = (y2 - y1) * col_thresh_pct
    content_cols = np.where(col_sums > col_threshold)[0]

    if len(content_cols) < 5:
        return (0, y1, w, y2)

    x1 = max(0, int(content_cols[0]) - margin)
    x2 = min(w, int(content_cols[-1]) + margin)

    return (x1, y1, x2, y2)


def _deskew_with_angle(bw: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew image and return both the corrected image and the skew angle.

    Wraps the legacy deskew function but also captures the angle.
    """
    # Detect skew angle using Hough lines (same logic as legacy)
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    h, w = bw.shape[:2]
    min_line_len = w // 6

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=min_line_len, maxLineGap=10
    )

    if lines is None:
        return bw, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return bw, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return bw, median_angle

    # Rotate to correct skew
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(
        bw, rot_mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return corrected, median_angle
