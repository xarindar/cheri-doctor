"""
Layout analysis module using Surya for region detection.

Identifies bounding boxes for: Text, Table, Figure/Picture, Caption,
SectionHeader, PageHeader, PageFooter, ListItem, TableOfContents.

Provides page classification based on dominant region types and
reading-order sorted region list for per-region processing.
"""

import numpy as np
from PIL import Image

# Surya imports — may fail on some Python versions
try:
    from surya.layout import LayoutPredictor, FoundationPredictor
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False

# ── Singleton predictor ──────────────────────────────────────────────────

_foundation: "FoundationPredictor | None" = None
_layout_predictor: "LayoutPredictor | None" = None


def _get_predictor() -> "LayoutPredictor":
    """Lazily initialize the Surya layout predictor (downloads model on first use)."""
    global _foundation, _layout_predictor
    if _layout_predictor is None:
        _foundation = FoundationPredictor()
        _layout_predictor = LayoutPredictor(_foundation)
    return _layout_predictor


# ── Region types ─────────────────────────────────────────────────────────

# Labels that indicate text content (should be OCR'd)
TEXT_LABELS = {"Text", "ListItem", "Caption", "SectionHeader", "Code",
               "PageHeader", "PageFooter", "Footnote"}

# Labels that indicate tabular content
TABLE_LABELS = {"Table", "Form"}

# Labels that indicate figures/diagrams
FIGURE_LABELS = {"Figure", "Picture"}

# Labels for TOC regions
TOC_LABELS = {"TableOfContents"}


class Region:
    """A detected page region with label, bounding box, and confidence."""

    def __init__(self, label: str, bbox: list[float], confidence: float,
                 polygon: list[list[float]] | None = None):
        self.label = label
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.polygon = polygon

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def is_text(self) -> bool:
        return self.label in TEXT_LABELS

    @property
    def is_table(self) -> bool:
        return self.label in TABLE_LABELS

    @property
    def is_figure(self) -> bool:
        return self.label in FIGURE_LABELS

    @property
    def is_toc(self) -> bool:
        return self.label in TOC_LABELS

    def crop_image(self, img: Image.Image, padding: int = 5) -> Image.Image:
        """Crop this region from the page image with optional padding."""
        w, h = img.size
        x1 = max(0, int(self.x1) - padding)
        y1 = max(0, int(self.y1) - padding)
        x2 = min(w, int(self.x2) + padding)
        y2 = min(h, int(self.y2) + padding)
        return img.crop((x1, y1, x2, y2))

    def __repr__(self):
        return (f"Region({self.label}, bbox=[{self.x1:.0f},{self.y1:.0f},"
                f"{self.x2:.0f},{self.y2:.0f}], conf={self.confidence:.2f})")


class PageLayout:
    """Layout analysis result for a single page."""

    def __init__(self, regions: list[Region], page_width: int, page_height: int):
        self.regions = regions
        self.page_width = page_width
        self.page_height = page_height

    @property
    def text_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_text]

    @property
    def table_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_table]

    @property
    def figure_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_figure]

    @property
    def toc_regions(self) -> list[Region]:
        return [r for r in self.regions if r.is_toc]

    def reading_order(self) -> list[Region]:
        """Sort regions in reading order (top-to-bottom, left-to-right).

        Detects two-column layouts by checking for side-by-side text regions
        and processes left column fully before right column.
        """
        if not self.regions:
            return []

        # Detect columns: group regions by horizontal position
        mid_x = self.page_width / 2
        left_regions = []
        right_regions = []
        full_width = []

        for r in self.regions:
            region_width_ratio = r.width / self.page_width
            if region_width_ratio > 0.6:
                # Wide region spans most of the page
                full_width.append(r)
            elif r.center_x < mid_x:
                left_regions.append(r)
            else:
                right_regions.append(r)

        # Sort each group top-to-bottom
        by_y = lambda r: r.y1
        left_regions.sort(key=by_y)
        right_regions.sort(key=by_y)
        full_width.sort(key=by_y)

        # If we have significant content in both columns, process left then right
        if left_regions and right_regions:
            ordered = []
            # Interleave full-width regions at correct vertical positions
            all_positioned = (
                [(r, 'full') for r in full_width] +
                [(r, 'left') for r in left_regions] +
                [(r, 'right') for r in right_regions]
            )
            all_positioned.sort(key=lambda x: x[0].y1)

            # Group into vertical bands: full-width items break the column flow
            current_left = []
            current_right = []
            for r, side in all_positioned:
                if side == 'full':
                    # Flush current columns (left first)
                    ordered.extend(sorted(current_left, key=by_y))
                    ordered.extend(sorted(current_right, key=by_y))
                    current_left = []
                    current_right = []
                    ordered.append(r)
                elif side == 'left':
                    current_left.append(r)
                else:
                    current_right.append(r)

            # Flush remaining
            ordered.extend(sorted(current_left, key=by_y))
            ordered.extend(sorted(current_right, key=by_y))
            return ordered
        else:
            # Single column: just sort by y position
            return sorted(self.regions, key=by_y)

    def classify_page(self) -> str:
        """Classify page type based on dominant region types.

        Returns: 'text', 'table', 'diagram', 'mixed', 'toc'
        """
        if not self.regions:
            return "diagram"  # No detectable regions → likely full-page image

        total_area = sum(r.area for r in self.regions)
        if total_area == 0:
            return "diagram"

        text_area = sum(r.area for r in self.text_regions)
        table_area = sum(r.area for r in self.table_regions)
        figure_area = sum(r.area for r in self.figure_regions)
        toc_area = sum(r.area for r in self.toc_regions)

        text_pct = text_area / total_area
        table_pct = table_area / total_area
        figure_pct = figure_area / total_area
        toc_pct = toc_area / total_area

        if toc_pct > 0.5:
            return "toc"
        if table_pct > 0.4:
            return "table"
        if figure_pct > 0.5 and text_pct < 0.3:
            return "diagram"
        if text_pct > 0.6:
            return "text"
        if figure_pct > 0.2 and text_pct > 0.2:
            return "mixed"
        if table_pct > 0.2 and text_pct > 0.2:
            return "mixed"

        # Default: classify by the dominant type
        dominant = max([("text", text_pct), ("table", table_pct),
                        ("diagram", figure_pct)], key=lambda x: x[1])
        return dominant[0]

    def has_multiple_columns(self) -> bool:
        """Detect if page has a two-column text layout."""
        text_regions = self.text_regions
        if len(text_regions) < 2:
            return False

        mid_x = self.page_width / 2
        left = [r for r in text_regions if r.center_x < mid_x and r.width < self.page_width * 0.6]
        right = [r for r in text_regions if r.center_x >= mid_x and r.width < self.page_width * 0.6]

        return len(left) >= 1 and len(right) >= 1


# ── Main API ─────────────────────────────────────────────────────────────

def analyze_layout(img: Image.Image, min_confidence: float = 0.3) -> PageLayout:
    """Analyze page layout using Surya.

    Args:
        img: PIL Image of the page (RGB).
        min_confidence: Minimum confidence threshold for regions.

    Returns:
        PageLayout with detected regions.
    """
    if not SURYA_AVAILABLE:
        return _fallback_layout(img)

    try:
        predictor = _get_predictor()
        results = predictor([img])
    except Exception as e:
        print(f"  [layout] Surya failed: {e}, using fallback")
        return _fallback_layout(img)

    if not results:
        return _fallback_layout(img)

    result = results[0]
    regions = []
    for box in result.bboxes:
        if box.confidence >= min_confidence:
            regions.append(Region(
                label=box.label,
                bbox=box.bbox,
                confidence=box.confidence,
                polygon=box.polygon,
            ))

    return PageLayout(regions, img.width, img.height)


def _fallback_layout(img: Image.Image) -> PageLayout:
    """Fallback layout when Surya is unavailable: treat entire page as one text region."""
    w, h = img.size
    margin = 20
    regions = [Region(
        label="Text",
        bbox=[margin, margin, w - margin, h - margin],
        confidence=0.5,
    )]
    return PageLayout(regions, w, h)
