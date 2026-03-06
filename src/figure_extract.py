"""Stage D.2: Figure Extraction and Linking.

For each figure/diagram region:
- Crops and exports a WebP image to build/assets/
- Finds the caption ("Figure N ...") in nearby text blocks
- Finds the legend (numbered items near the figure)
- Optionally generates a Claude Vision description
"""

import re
from pathlib import Path
from PIL import Image

from src.models import FigureResult
from src.utils import resolve_path

# Vision description from legacy pipeline
from vision_describe import describe_diagram, is_vision_available

CAPTION_RE = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)\s*(.*)", re.IGNORECASE)
LEGEND_ITEM_RE = re.compile(r"^\s*(\d+)\s*[\.\)\-\u2014]\s*(.+)", re.MULTILINE)


def extract_figure(region,
                   page_img: Image.Image,
                   existing_blocks: list[dict],
                   page_num: int,
                   fig_idx: int,
                   config: dict,
                   project_root: Path,
                   skip_vision: bool = True) -> FigureResult:
    """Extract a figure region: crop, export, find caption/legend, describe.

    Args:
        region:          Surya Region object.
        page_img:        Full-resolution original page PIL image.
        existing_blocks: Already-extracted text blocks from this page (for caption/legend search).
        page_num:        1-based page number.
        fig_idx:         Index of this figure on the page.
        config:          Pipeline config.
        project_root:    Project root Path.
        skip_vision:     If True, skip Claude Vision API call.
    """
    build_dir  = resolve_path(config["pipeline"]["build_dir"], project_root)
    assets_dir = build_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cfg_fig = config.get("figure_extraction", {})
    webp_quality     = cfg_fig.get("webp_quality", 85)
    caption_distance = cfg_fig.get("caption_search_distance_px", 100)
    legend_distance  = cfg_fig.get("legend_search_distance_px", 300)
    legend_min_items = cfg_fig.get("legend_min_items", 3)

    figure_id  = f"fig_p{page_num:04d}_{fig_idx:03d}"
    asset_name = f"page_{page_num:04d}_fig_{fig_idx:03d}.webp"
    asset_path = assets_dir / asset_name

    # ── Crop and export ──────────────────────────────────────────
    w, h = page_img.size
    x1 = max(0, int(region.x1) - 5)
    y1 = max(0, int(region.y1) - 5)
    x2 = min(w, int(region.x2) + 5)
    y2 = min(h, int(region.y2) + 5)

    crop = page_img.crop((x1, y1, x2, y2))
    crop.save(str(asset_path), "WEBP", quality=webp_quality)

    # ── Find caption ─────────────────────────────────────────────
    # Search for a text block near the figure that matches the caption pattern.
    # Most common is below, but sometimes it's above.
    caption_text   = None
    figure_number  = None
    caption_blk_id = None
    
    pattern_candidates  = []  # blocks with "Figure N" / "Fig. N"
    fallback_candidates = []  # nearest text block below (no pattern required)

    for block in existing_blocks:
        if block.get("type") not in ("paragraph", "caption", "heading"):
            continue

        bx1, by1, bx2, by2 = block.get("bbox", [0, 0, 0, 0])

        # Check below (preferred) and above (fallback)
        is_below = by1 >= region.y2 and (by1 - region.y2) <= caption_distance
        is_above = region.y1 >= by2 and (region.y1 - by2) <= caption_distance

        if not (is_below or is_above):
            continue

        text = (block.get("text") or "").strip()
        if not text:
            continue

        distance = (by1 - region.y2) if is_below else (region.y1 - by2)
        m = CAPTION_RE.search(text)
        if m:
            pattern_candidates.append({"distance": distance, "block": block, "match": m})
        elif is_below:
            # Only use blocks below as proximity fallback (above is too noisy)
            fallback_candidates.append({"distance": distance, "block": block, "match": None})

    # Prefer explicit "Figure N" matches; fall back to nearest text below
    best_list = pattern_candidates or fallback_candidates
    if best_list:
        best_candidate = sorted(best_list, key=lambda c: c["distance"])[0]
        block = best_candidate["block"]
        m = best_candidate["match"]

        figure_number  = int(m.group(1)) if m else None
        caption_text   = block.get("text", "").strip()
        caption_blk_id = block.get("block_id")

        block["type"] = "caption"
        block["parent_figure_id"] = figure_id
        if figure_number is not None:
            block["figure_number"] = figure_number


    # ── Find legend ──────────────────────────────────────────────
    legend_items   = None
    legend_blk_id  = None

    for block in existing_blocks:
        if block.get("block_id") == caption_blk_id:
            continue
        if block.get("type") not in ("paragraph", "ordered_list"):
            continue
        bx1, by1, bx2, by2 = block.get("bbox", [0, 0, 0, 0])
        # Legend is below figure (or caption), within legend_distance px
        vertical_ok   = by1 >= region.y2 and (by1 - region.y2) <= legend_distance
        # Or to the right of the figure within the same vertical band
        horizontal_ok = (bx1 >= region.x2 and abs(by1 - region.y1) < region.height * 1.5)

        if vertical_ok or horizontal_ok:
            text = block.get("text", "")
            matches = LEGEND_ITEM_RE.findall(text)
            if len(matches) >= legend_min_items:
                legend_items  = [{"key": k.strip(), "value": v.strip()} for k, v in matches]
                legend_blk_id = block.get("block_id")
                block["type"]              = "legend"
                block["parent_figure_id"]  = figure_id
                break

    # ── Vision description ────────────────────────────────────────
    vision_desc = None
    if not skip_vision and is_vision_available():
        cache_key = figure_id
        vision_desc = describe_diagram(crop, cache_key=cache_key)

    return FigureResult(
        figure_id=figure_id,
        asset_path=str(asset_path.relative_to(project_root)).replace("\\", "/"),
        bbox=(x1, y1, x2, y2),
        caption_text=caption_text,
        figure_number=figure_number,
        legend_items=legend_items,
        vision_description=vision_desc,
        caption_block_id=caption_blk_id,
        legend_block_id=legend_blk_id,
    )
