#!/usr/bin/env python3
"""Rebuild figure assets using CV box detection.

Scans all pages that have figures (from figures.jsonl), detects bordered
regions via OpenCV, crops individual diagrams, and saves as WebP assets.
Replaces full-page screenshots with properly cropped individual figures.

Also updates figures.jsonl with corrected bboxes.

Usage:
    python tools/rebuild_figure_assets.py                  # all pages
    python tools/rebuild_figure_assets.py --pages 337-340  # page range
    python tools/rebuild_figure_assets.py --dry-run        # preview only
"""
import sys, os, argparse, json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "legacy"))
os.chdir(PROJECT_ROOT)

from PIL import Image
from src.layout import _detect_boxes_cv, _remove_container_boxes


def parse_page_range(s: str) -> set[int]:
    pages = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return pages


def main():
    parser = argparse.ArgumentParser(description="Rebuild figure assets with CV detection")
    parser.add_argument("--pages", help="Page range, e.g. 337-340 or 100,200,337")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    build_dir = PROJECT_ROOT / "build"
    assets_dir = build_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load figures.jsonl — the authoritative source of figure data
    figures_path = build_dir / "figures.jsonl"
    figures = []
    with open(figures_path) as f:
        for line in f:
            line = line.strip()
            if line:
                figures.append(json.loads(line))

    # Load preprocess results for image paths
    with open(build_dir / "preprocess_results.json") as f:
        pre_results = json.load(f)

    page_filter = parse_page_range(args.pages) if args.pages else None

    # Group figures by page, sorted by figure index within each page
    by_page = defaultdict(list)
    for fig in figures:
        pn = fig["page"]
        if page_filter and pn not in page_filter:
            continue
        by_page[pn].append(fig)

    # Sort each page's figures by their existing index (from figure_id)
    for pn in by_page:
        by_page[pn].sort(key=lambda f: f["figure_id"])

    stats = {"pages": 0, "cropped": 0, "fullpage": 0, "cv_pages": 0}

    # CV detection cache (one per page image)
    cv_cache: dict[int, list] = {}

    for pn in sorted(by_page.keys()):
        figs = by_page[pn]
        pre_key = str(pn)
        if pre_key not in pre_results:
            continue
        pre_path = pre_results[pre_key].get("preprocessed_path")
        if not pre_path or not Path(pre_path).exists():
            continue

        stats["pages"] += 1

        # Check if any figure on this page has a full-page bbox
        pre_img = None
        img_w, img_h = 0, 0

        def get_pre_img():
            nonlocal pre_img, img_w, img_h
            if pre_img is None:
                pre_img = Image.open(pre_path).convert("RGB")
                img_w, img_h = pre_img.size
            return pre_img

        # Determine if figures need re-cropping
        needs_recrop = False
        for fig in figs:
            bbox = fig.get("bbox", [])
            if len(bbox) == 4:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w > 2500 and h > 3500:
                    needs_recrop = True
                    break

        if not needs_recrop and not args.dry_run:
            # Figures already have reasonable bboxes, skip
            continue

        # Run CV detection
        img = get_pre_img()
        if pn not in cv_cache:
            cv_boxes = _detect_boxes_cv(img, [], {})
            if cv_boxes:
                cv_boxes = _remove_container_boxes(cv_boxes, threshold=0.50)
                cv_boxes.sort(key=lambda r: r.y1)
            cv_cache[pn] = cv_boxes or []

        fig_regions = cv_cache[pn]
        if fig_regions:
            stats["cv_pages"] += 1

        for fig_idx, fig in enumerate(figs):
            asset_path_str = fig.get("asset_path", "")
            asset_name = Path(asset_path_str).name if asset_path_str else f"page_{pn:04d}_fig_{fig_idx:03d}.webp"
            asset_path = assets_dir / asset_name

            if fig_idx < len(fig_regions):
                fr = fig_regions[fig_idx]
                crop_box = (int(fr.x1), int(fr.y1), int(fr.x2), int(fr.y2))

                if args.dry_run:
                    print(f"  [crop] p{pn} {fig['figure_id']}: {crop_box}")
                else:
                    fig_crop = img.crop(crop_box)
                    fig_crop.save(str(asset_path), "WEBP", quality=85)
                    fig["bbox"] = list(crop_box)
                    print(f"  [crop] p{pn} {fig['figure_id']}: {crop_box} → {asset_name}")
                stats["cropped"] += 1
            else:
                if args.dry_run:
                    print(f"  [full] p{pn} {fig['figure_id']}: no matching CV region (idx {fig_idx} >= {len(fig_regions)})")
                stats["fullpage"] += 1

    # Write updated figures.jsonl
    if not args.dry_run and stats["cropped"] > 0:
        with open(figures_path, "w") as f:
            for fig in figures:
                f.write(json.dumps(fig) + "\n")
        print(f"\n  Updated figures.jsonl with {stats['cropped']} new bboxes.")

    print(f"\nDone: {stats['pages']} pages, "
          f"{stats['cropped']} cropped, "
          f"{stats['fullpage']} no-match fallback, "
          f"{stats['cv_pages']} pages with CV regions.")


if __name__ == "__main__":
    main()
