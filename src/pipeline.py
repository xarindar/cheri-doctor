"""Main pipeline orchestrator for the Metro Manual extraction system.

Coordinates all stages:
  A) Ingest & Rasterize
  B) Page Preprocessing
  C) Layout Segmentation
  D) Region-Specific Extraction (text, tables, figures)
  E) Structure Reconstruction
  F) Retrieval Corpus Build
  G) Index Build
  H) Chat (separate entry point)

Usage:
  python -m src.pipeline                          # full run
  python -m src.pipeline --pages 1-30             # page range
  python -m src.pipeline --skip-vision            # no vision AI
  python -m src.pipeline --stages A,B             # specific stages
  python -m src.pipeline --reprocess              # redo all pages
"""

import sys
import json
import argparse
import re
from pathlib import Path

# Load .env before any legacy modules check os.environ for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.utils import load_config, save_json, load_json, now_iso, Timer, resolve_path
from src.models import Manifest

TOC_NOISE_RE = re.compile(r"\.{3,}|\b\d+[A-Z]-\d+\b")

def _clean_toc_noise(text: str) -> str:
    """Strip dot leaders and section codes from TOC-like runs."""
    if not text:
        return text
    t = re.sub(r"\.{3,}", " ", text)
    t = re.sub(r"\b\d+[A-Z]-\d+\b", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _strip_dot_leaders(text: str) -> str:
    """Normalize any long dot-leader runs to a single space."""
    if not text:
        return text
    return re.sub(r"\.{3,}", " ", text)

def _strip_leading_toc_chunk(text: str) -> str:
    """If a paragraph starts with a TOC-like run, keep content from first numbered step onward."""
    if not text:
        return text
    m = re.search(r"\b\d{1,2}\.\s", text)
    if m and m.start() > 20:
        return text[m.start():].strip()
    return text


PROJECT_ROOT = Path(__file__).resolve().parent.parent


import time

def check_pause(build_dir: Path):
    """Check for a 'PAUSE' file in build_dir. If found, sleep until removed."""
    pause_file = build_dir / "PAUSE"
    if pause_file.exists():
        print(f"\n[PAUSED] Found {pause_file}. Delete this file to resume...")
        while pause_file.exists():
            time.sleep(2)
        print("[RESUMED] Continuing pipeline...\n")


def run_pipeline(config_path: str = "configs/default.yaml",
                 page_range: tuple[int, int] | None = None,
                 stages: set[str] | None = None,
                 skip_vision: bool = False,
                 reprocess: bool = False):
    """Run the extraction pipeline."""

    config = load_config(PROJECT_ROOT / config_path)
    build_dir = resolve_path(config["pipeline"]["build_dir"], PROJECT_ROOT)
    build_dir.mkdir(parents=True, exist_ok=True)

    all_stages = {"A", "B", "C", "D", "E", "F", "G"}
    if stages is None:
        stages = all_stages

    print("=" * 60)
    print("Metro Manual - Chat With The Manual Pipeline")
    print("=" * 60)
    print(f"  Config:     {config_path}")
    print(f"  Build dir:  {build_dir}")
    print(f"  Stages:     {', '.join(sorted(stages))}")
    if page_range:
        print(f"  Pages:      {page_range[0]}-{page_range[1]}")
    print(f"  Vision AI:  {'Disabled' if skip_vision else 'Enabled'}")
    print("=" * 60)

    document = {"doc_id": config["pipeline"]["doc_id"], "pages": []}

    # ── Stage A: Ingest & Rasterize ──────────────────────────────
    if "A" in stages:
        manifest_path = build_dir / "manifest.json"
        metas_path = build_dir / "page_metas.json"
        if manifest_path.exists() and metas_path.exists() and not reprocess:
            print("\n[Stage A] Skipping rasterization (outputs exist).")
            page_metas = load_json(metas_path)
            manifest_data = load_json(manifest_path)
            manifest = Manifest(**{k: v for k, v in manifest_data.items()
                                   if k in Manifest.__dataclass_fields__})
        else:
            print("\n[Stage A] Rasterizing PDF...")
            from src.ingest import rasterize_pdf
            with Timer("Rasterize"):
                manifest, page_metas = rasterize_pdf(config, PROJECT_ROOT)
            # Save page_metas for subsequent stages
            save_json(page_metas, build_dir / "page_metas.json")
    else:
        page_metas = load_json(build_dir / "page_metas.json")
        manifest_data = load_json(build_dir / "manifest.json")
        manifest = Manifest(**{k: v for k, v in manifest_data.items()
                               if k in Manifest.__dataclass_fields__})

    # Apply page range filter
    if page_range:
        page_metas = [p for p in page_metas
                      if page_range[0] <= p["page_num"] <= page_range[1]]
        print(f"\n  Filtered to {len(page_metas)} pages")

    # Skip front-matter pages (cover, foreword, comment form, etc.)
    front_matter = config.get("pipeline", {}).get("front_matter_pages", 0)
    if front_matter > 0 and not page_range:
        before = len(page_metas)
        page_metas = [p for p in page_metas if p["page_num"] > front_matter]
        print(f"\n  Skipping front-matter pages 1-{front_matter} "
              f"({before - len(page_metas)} pages skipped)")

    # Skip specific pages (e.g. metric conversion tables, boilerplate)
    skip_pages = set(config.get("pipeline", {}).get("skip_pages", []))
    if skip_pages:
        before = len(page_metas)
        page_metas = [p for p in page_metas if p["page_num"] not in skip_pages]
        skipped = before - len(page_metas)
        if skipped:
            print(f"  Skipping pages {sorted(skip_pages)}: {skipped} page(s) excluded")

    # ── Stage B: Preprocess ──────────────────────────────────────
    if "B" in stages:
        pre_results_path = build_dir / "preprocess_results.json"
        if pre_results_path.exists() and not reprocess:
            print("\n[Stage B] Skipping preprocessing (outputs exist).")
            pre_data = load_json(pre_results_path)
        else:
            print("\n[Stage B] Preprocessing pages...")
            from src.preprocess import preprocess_page
            preprocess_results_tmp = {}
            with Timer("Preprocess"):
                for pm in page_metas:
                    result = preprocess_page(pm, config, PROJECT_ROOT)
                    preprocess_results_tmp[pm["page_num"]] = result
                    if pm["page_num"] % 50 == 0 or pm == page_metas[-1]:
                        print(f"    Preprocessed {pm['page_num']}/{page_metas[-1]['page_num']}")
            # Save preprocess results
            pre_data = {str(k): {
                "page_num": v.page_num,
                "preprocessed_path": v.preprocessed_path,
                "content_bbox": list(v.content_bbox),
                "skew_angle": v.skew_angle,
                "original_size": list(v.original_size),
            } for k, v in preprocess_results_tmp.items()}
            save_json(pre_data, build_dir / "preprocess_results.json")
        
        from src.models import PreprocessResult
        preprocess_results = {}
        for k, v in pre_data.items():
            preprocess_results[int(k)] = PreprocessResult(
                page_num=v["page_num"],
                preprocessed_path=v["preprocessed_path"],
                content_bbox=tuple(v["content_bbox"]),
                skew_angle=v["skew_angle"],
                original_size=tuple(v["original_size"]),
            )
    else:
        pre_data = load_json(build_dir / "preprocess_results.json")
        from src.models import PreprocessResult
        preprocess_results = {}
        for k, v in pre_data.items():
            preprocess_results[int(k)] = PreprocessResult(
                page_num=v["page_num"],
                preprocessed_path=v["preprocessed_path"],
                content_bbox=tuple(v["content_bbox"]),
                skew_angle=v["skew_angle"],
                original_size=tuple(v["original_size"]),
            )

    # ── Stage C: Layout Segmentation ─────────────────────────────
    if "C" in stages:
        layout_results_path = build_dir / "layout_results.json"
        from layout_analysis import Region
        
        if layout_results_path.exists() and not reprocess:
            print(f"\n[Stage C] Loading existing layout results from {layout_results_path.name}...")
            layout_data = load_json(layout_results_path)
            layout_results = {}
            for k, v in layout_data.items():
                regions = [Region(label=r["label"], bbox=r["bbox"],
                                  confidence=r["confidence"])
                           for r in v["regions"]]
                layout_results[int(k)] = {
                    "page_type": v["page_type"],
                    "columns": v["columns"],
                    "has_multiple_columns": v["has_multiple_columns"],
                    "is_complex": v.get("is_complex", False),
                    "complexity_reasons": v.get("complexity_reasons", []),
                    "regions": regions,
                }
            print(f"    Loaded {len(layout_results)} pages.")
        else:
            layout_results = {}
            layout_data = {}

        pages_to_process = [pm for pm in page_metas if pm["page_num"] not in layout_results]
        
        if not pages_to_process:
            print("\n[Stage C] All requested pages already have layout results. Skipping.")
        else:
            print(f"\n[Stage C] Analyzing layout for {len(pages_to_process)} pages...")
            from src.layout import analyze_page_layout
            with Timer("Layout"):
                for i, pm in enumerate(pages_to_process):
                    pn = pm["page_num"]
                    pre = preprocess_results[pn]
                    lr = analyze_page_layout(pre, config)
                    layout_results[pn] = lr
                    
                    # Update layout_data for saving
                    layout_data[str(pn)] = {
                        "page_type": lr["page_type"],
                        "columns": lr["columns"],
                        "has_multiple_columns": lr["has_multiple_columns"],
                        "is_complex": lr.get("is_complex", False),
                        "complexity_reasons": lr.get("complexity_reasons", []),
                        "regions": [
                            {"label": r.label, "bbox": r.bbox,
                             "confidence": r.confidence}
                            for r in lr["regions"]
                        ],
                    }

                    if (i + 1) % 50 == 0 or pm == pages_to_process[-1]:
                        print(f"    Layout {pn}/{pages_to_process[-1]['page_num']} ({i+1}/{len(pages_to_process)})")
                        # Periodic save
                        save_json(layout_data, layout_results_path)
            
            # Final save
            save_json(layout_data, layout_results_path)
    else:
        layout_data = load_json(build_dir / "layout_results.json")
        layout_results = {}
        from layout_analysis import Region
        for k, v in layout_data.items():
            regions = [Region(label=r["label"], bbox=r["bbox"],
                              confidence=r["confidence"])
                       for r in v["regions"]]
            layout_results[int(k)] = {
                "page_type": v["page_type"],
                "columns": v["columns"],
                "has_multiple_columns": v["has_multiple_columns"],
                "is_complex": v.get("is_complex", False),
                "complexity_reasons": v.get("complexity_reasons", []),
                "regions": regions,
            }

    # ── Stage D: Region-Specific Extraction ──────────────────────
    if "D" in stages:
        print("\n[Stage D] Extracting content from regions...")
        from src.ocr_text import ocr_text_region, classify_blocks, clean_and_correct
        from src.figure_extract import extract_figure
        from src.table_extract import extract_table_structured
        from src.vision_classify import (
            classify_region,
            extract_table_rows,
            is_vision_classifier_available,
        )
        from vision_describe import load_description_cache, save_description_cache
        from PIL import Image as PILImage

        # Cache for describe_diagram (legacy vision_describe module)
        vision_cache_file = build_dir / "vision_cache.json"
        load_description_cache(vision_cache_file)
        if vision_cache_file.exists():
            print(f"  [vision] Loaded description cache from {vision_cache_file.name}")

        # Cache for classify_region results keyed by figure_id.
        # classify_region is the primary vision call (1 API call per figure).
        classify_cache_file = build_dir / "vision_classify_cache.json"
        _classify_cache: dict = {}
        if classify_cache_file.exists():
            _classify_cache = load_json(classify_cache_file)
            print(f"  [vision] Loaded classify cache: {len(_classify_cache)} figures")

        # Cache for extract_full_page_vision results keyed by page_num.
        full_page_cache_file = build_dir / "full_page_vision_cache.json"
        _full_page_cache: dict = {}
        if full_page_cache_file.exists():
            _full_page_cache = load_json(full_page_cache_file)
            print(f"  [vision] Loaded full-page cache: {len(_full_page_cache)} pages")

        document_path = build_dir / "document.json"
        if document_path.exists() and not reprocess:
            document = load_json(document_path)
            # Ensure we don't duplicate pages if resuming
            existing_pns = {p["page_num"] for p in document["pages"]}
            page_metas = [pm for pm in page_metas if pm["page_num"] not in existing_pns]
            print(f"  [Stage D] Resuming from document.json ({len(document['pages'])} pages already extracted)")
        else:
            document = {"doc_id": config["pipeline"]["doc_id"], "pages": []}

        vision_cls_available = (not skip_vision) and is_vision_classifier_available()
        if not skip_vision and not vision_cls_available:
            print("  [vision] Box classifier unavailable (missing anthropic/openai package or API key); boxes will be treated as figures.")

        with Timer("Extraction"):
            from src.vision_extract import extract_full_page_vision
            
            for i, pm in enumerate(page_metas):
                pn = pm["page_num"]
                pre = preprocess_results[pn]
                lr = layout_results[pn]

                page_img = PILImage.open(pm["path"])
                pre_img = PILImage.open(pre.preprocessed_path)

                page_data = {
                    "page_num": pn,
                    "page_label": f"page_{pn:04d}",
                    "page_type": lr["page_type"],
                    "width_px": pm["width"],
                    "height_px": pm["height"],
                    "is_complex": lr.get("is_complex", False),
                    "preprocess": {
                        "content_bbox": list(pre.content_bbox),
                        "skew_angle": pre.skew_angle,
                        "columns_detected": len(lr["columns"]),
                        "column_bboxes": lr["columns"],
                    },
                    "blocks": [],
                    "relations": [],
                }

                # ROUTING: Vision (full-page) vs OCR (per-region)
                # Use vision path when:
                #   1. Page is already in the full-page cache (pre-extracted), OR
                #   2. Page is complex/mixed/text and vision API is available
                _page_type = lr.get("page_type", "")
                _cache_key = str(pn)
                _has_cache = _cache_key in _full_page_cache
                _use_vision = (
                    _has_cache
                    or lr.get("is_complex", False)
                    or _page_type == "mixed"
                    or _page_type == "text"
                    or _page_type == "diagram"
                )
                if not skip_vision and _use_vision and (_has_cache or is_vision_classifier_available()):
                    v_res = None
                    if _cache_key in _full_page_cache:
                        print(f"    [vision-full] p{pn:04d} (using cached result)")
                        v_res = _full_page_cache[_cache_key]
                    else:
                        reasons = lr.get('complexity_reasons') or [lr.get('page_type', 'mixed')]
                        print(f"    [vision-full] p{pn:04d} ({', '.join(reasons)})")
                        v_res = extract_full_page_vision(pre_img, pn, model=config["vision"].get("full_page_model", "claude-3-5-sonnet-20241022"))
                        if v_res:
                            _full_page_cache[_cache_key] = v_res

                    if v_res:
                        page_data["page_label"] = v_res.get("page_label", page_data["page_label"])
                        v_fig_idx = 0  # Counter for figure IDs on this vision page
                        for j, vb in enumerate(v_res.get("blocks", [])):
                            b_id = vb.get("block_id", f"p{pn}_b{j:03d}")
                            block_dict = {
                                "block_id": b_id,
                                "type": vb.get("type", "paragraph"),
                                "order_index": j,
                                "text": vb.get("text") or vb.get("title") or None,
                                "steps": vb.get("steps"),
                                "rows": vb.get("rows"),
                                "caption_text": vb.get("caption"),
                                "legend_items": vb.get("legend"),
                                "procedure_type": vb.get("procedure_type"),
                                "continues_from_previous_page": vb.get("continues_from_previous_page", False),
                                "continues_to_next_page": vb.get("continues_to_next_page", False),
                                "associated_figure_ids": vb.get("associated_figure_ids", []),
                                "bbox": list(pre.content_bbox), # Full page bbox as fallback
                            }
                            # If it's a table, we should still try to export an asset
                            if block_dict["type"] == "table" and block_dict["rows"]:
                                # TODO: try to find bbox for the table if vision didn't provide it
                                pass

                            # Vision figure blocks: assign figure_id and export
                            # a full-page WebP crop so the image can be served.
                            if block_dict["type"] == "figure":
                                fig_id = f"fig_p{pn:04d}_{v_fig_idx:03d}"
                                asset_name = f"page_{pn:04d}_fig_{v_fig_idx:03d}.webp"
                                assets_dir = build_dir / "assets"
                                assets_dir.mkdir(parents=True, exist_ok=True)
                                asset_path = assets_dir / asset_name
                                if not asset_path.exists():
                                    # Export the full page as the figure image
                                    page_img.save(str(asset_path), "WEBP", quality=85)
                                block_dict["figure_id"] = fig_id
                                block_dict["asset_path"] = str(asset_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                                v_fig_idx += 1

                            page_data["blocks"].append(block_dict)
                        document["pages"].append(page_data)
                        print(f"      Extracted {len(page_data['blocks'])} blocks via vision")
                        
                        # Periodic cache save for vision
                        if (i + 1) % 5 == 0 or pm == page_metas[-1]:
                            save_json(_full_page_cache, full_page_cache_file)
                            save_json(_classify_cache, classify_cache_file)
                            save_description_cache(vision_cache_file)
                        continue
                    else:
                        print(f"      Vision extraction failed for p{pn}, falling back to OCR")

                # SIMPLE Page (or vision fallback)
                block_idx = 0
                fig_idx = 0
                tbl_idx = 0

                # D.1: Text - column detect -> upscale -> OCR -> classify blocks
                if lr["page_type"] in ("text", "toc", "mixed"):
                    for col_bbox in lr["columns"]:
                        x1, y1, x2, y2 = col_bbox
                        col_img = pre_img.crop((x1, y1, x2, y2))
                        ocr_result = ocr_text_region(col_img, "Text", config)
                        corrected = clean_and_correct(
                            ocr_result["text"], ocr_result.get("word_data", {}),
                            is_toc=(lr["page_type"] == "toc"), config=config
                        )
                        corrected = _strip_dot_leaders(corrected)
                        blocks = classify_blocks(corrected)
                        # Normalize level-1 section-code headings to canonical name.
                        # OCR merges the page header with adjacent content into one
                        # long line; use config section_names as the authoritative name.
                        _sec_names = config.get("structure", {}).get("section_names", {})
                        if _sec_names:
                            _norm = []
                            for b in blocks:
                                if b.type == "heading" and b.level == 1:
                                    _cm = re.match(r'^([0-9]+[A-Z]\d*)-([0-9A-Z]+)', b.text)
                                    if _cm:
                                        _sec = _cm.group(1)
                                        if _sec in _sec_names:
                                            _full_code = _cm.group(0)
                                            _name = _sec_names[_sec].upper()
                                            _new_text = f"{_full_code} {_name}"
                                            b = b._replace(text=_new_text) if hasattr(b, "_replace") else type(b)(**{**b.__dict__, "text": _new_text})
                                _norm.append(b)
                            blocks = _norm
                        for b in blocks:
                            if b.type == "paragraph":
                                cleaned = _clean_toc_noise(b.text)
                                cleaned = _strip_leading_toc_chunk(cleaned)
                                if not cleaned or len(cleaned.split()) < 5:
                                    continue
                                b = b._replace(text=cleaned) if hasattr(b, "_replace") else type(b)(**{**b.__dict__, "text": cleaned})
                            block_dict = {
                                "block_id": f"p{pn}_b{block_idx:03d}",
                                "type": b.type,
                                "bbox": col_bbox,
                                "order_index": block_idx,
                                "text": b.text,
                            }
                            if b.level is not None:
                                block_dict["level"] = b.level
                                block_dict["heading_text"] = b.text
                            if b.steps is not None:
                                block_dict["steps"] = b.steps
                            page_data["blocks"].append(block_dict)
                            block_idx += 1

                # D.2 + D.3: Boxes detected by CV are labelled "Picture".
                # Vision-first: classify; if table -> vision rows -> CSV (no local OCR). Else figure.
                # Surya-detected Table regions are handled in D.3 below.
                for region in lr["regions"]:
                    if region.is_figure:
                        vision_cls = None
                        vision_rows = None

                        if not skip_vision and config["vision"].get("classify_boxes", True) and vision_cls_available:
                            rx1, ry1, rx2, ry2 = map(int, region.bbox)
                            region_crop = pre_img.crop((rx1, ry1, rx2, ry2))
                            # Use cached result if available (avoids re-paying for vision on re-runs)
                            _cache_key = f"fig_p{pn:04d}_{fig_idx:03d}"
                            if _cache_key in _classify_cache:
                                vision_cls = _classify_cache[_cache_key]
                                print(f"      [vision] p{pn} box -> (cached) type={vision_cls.get('type')}")
                            else:
                                vision_cls = classify_region(region_crop, model=config["vision"].get("classify_model"))
                                if vision_cls:
                                    _classify_cache[_cache_key] = vision_cls
                                    print(f"      [vision] p{pn} box -> type={vision_cls.get('type')} rows={len(vision_cls.get('rows') or [])} legend={len(vision_cls.get('legend') or [])}")
                                else:
                                    print(f"      [vision] p{pn} box -> classifier returned None")
                            if vision_cls and vision_cls.get("type") == "table":
                                # Vision-only table transcription (ChatGPT/OpenAI). No python OCR.
                                vision_rows = extract_table_rows(
                                    region_crop,
                                    model=config["vision"].get("table_model", "gpt-4o"),
                                )
                                print(f"      [vision] p{pn} table rows via vision: {len(vision_rows or [])}")

                        tbl_result = None
                        if vision_cls and vision_cls.get("type") == "table" and vision_rows:
                            tbl_result = extract_table_structured(
                                region, pre_img, pn, tbl_idx, config, PROJECT_ROOT, rows_override=vision_rows, skip_ml=True
                            )
                        elif vision_cls and vision_cls.get("type") == "table" and not vision_rows:
                            print(f"      [vision] p{pn} table flagged but vision rows missing; skipping box.")
                        rows_shape = (
                            f"{len(tbl_result.rows)}x{len(tbl_result.rows[0])}"
                            if tbl_result and tbl_result.rows else "none"
                        )
                        print(f"      [tbl?] p{pn} -> {rows_shape}")
                        is_valid_table = (
                            tbl_result
                            and tbl_result.rows
                            and len(tbl_result.rows) >= 2
                            and len(tbl_result.rows[0]) >= 2
                        )
                        if is_valid_table:
                            tbl_block = {
                                "block_id": f"p{pn}_b{block_idx:03d}",
                                "type": "table",
                                "bbox": list(region.bbox),
                                "order_index": block_idx,
                                "table_id": f"tbl_p{pn}_{tbl_idx:03d}",
                                "rows": tbl_result.rows,
                                "csv_path": tbl_result.csv_path,
                                "asset_path": tbl_result.asset_path,
                                "retrieval_text": tbl_result.retrieval_text,
                                "method": tbl_result.method,
                            }
                            page_data["blocks"].append(tbl_block)
                            block_idx += 1
                            tbl_idx += 1
                        else:
                            # If vision already said diagram/legend, reuse description/legend and skip vision call in figure_extract
                            skip_fig_vision = bool(vision_cls)
                            fig_result = extract_figure(
                                region, pre_img, page_data["blocks"],
                                pn, fig_idx, config, PROJECT_ROOT,
                                skip_vision=skip_fig_vision or skip_vision,
                            )
                            fig_block = {
                                "block_id": f"p{pn}_b{block_idx:03d}",
                                "type": "figure",
                                "bbox": list(region.bbox),
                                "order_index": block_idx,
                                "figure_id": fig_result.figure_id,
                                "asset_path": fig_result.asset_path,
                            }
                            if fig_result.caption_text:
                                fig_block["caption_text"] = fig_result.caption_text
                                fig_block["figure_number"] = fig_result.figure_number
                            legend_items = fig_result.legend_items
                            if vision_cls and vision_cls.get("legend"):
                                legend_items = legend_items or []
                                legend_items.extend({"key": "", "value": l} for l in vision_cls["legend"])
                            if legend_items:
                                fig_block["legend_items"] = legend_items
                            vision_desc = fig_result.vision_description
                            if vision_cls and vision_cls.get("description"):
                                vision_desc = vision_desc or vision_cls["description"]
                            if vision_desc:
                                fig_block["vision_description"] = vision_desc
                            page_data["blocks"].append(fig_block)
                            block_idx += 1
                            fig_idx += 1

                # D.3: Surya-detected Table regions (fallback for clean PDFs)
                for region in lr["regions"]:
                    if region.is_table:
                        tbl_result = extract_table_structured(
                            region, page_img, pn, tbl_idx, config, PROJECT_ROOT
                        )
                        if tbl_result:
                            tbl_block = {
                                "block_id": f"p{pn}_b{block_idx:03d}",
                                "type": "table",
                                "bbox": list(region.bbox),
                                "order_index": block_idx,
                                "table_id": f"tbl_p{pn}_{tbl_idx:03d}",
                                "rows": tbl_result.rows,
                                "csv_path": tbl_result.csv_path,
                                "asset_path": tbl_result.asset_path,
                                "retrieval_text": tbl_result.retrieval_text,
                            }
                            page_data["blocks"].append(tbl_block)
                            block_idx += 1
                            tbl_idx += 1

                # Fallback: if no blocks extracted, try full-page OCR
                if not page_data["blocks"]:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = config["ocr"]["tesseract_cmd"]
                    fallback_text = pytesseract.image_to_string(
                        pre_img, config="--oem 1 --psm 3"
                    )
                    if fallback_text.strip():
                        page_data["blocks"].append({
                            "block_id": f"p{pn}_b000",
                            "type": "paragraph",
                            "bbox": list(pre.content_bbox),
                            "order_index": 0,
                            "text": fallback_text.strip(),
                        })

                document["pages"].append(page_data)

                # Periodic cache save for vision (Simple pages too, in case of box classification)
                if (i + 1) % 5 == 0 or pm == page_metas[-1]:
                    save_json(_full_page_cache, full_page_cache_file)
                    save_json(_classify_cache, classify_cache_file)
                    save_description_cache(vision_cache_file)

                # Per-page summary: show region label breakdown and block types
                from collections import Counter
                region_counts = Counter(r.label for r in lr["regions"])
                region_summary = ", ".join(f"{lbl}x{n}" for lbl, n in sorted(region_counts.items())) or "none"
                block_counts = Counter(b["type"] for b in page_data["blocks"])
                block_summary = ", ".join(f"{t}x{n}" for t, n in sorted(block_counts.items())) or "none"
                figs_found = sum(1 for b in page_data["blocks"] if b["type"] == "figure")
                print(f"    p{pn:04d} [{lr['page_type']}] "
                      f"regions=({region_summary}) "
                      f"blocks=({block_summary})"
                      + (f" *** {figs_found} figure(s)" if figs_found else ""))


        save_description_cache(vision_cache_file)
        save_json(_classify_cache, classify_cache_file)
        print(f"  [vision] Classify cache saved: {len(_classify_cache)} figures ({classify_cache_file.name})")
        save_json(_full_page_cache, full_page_cache_file)
        print(f"  [vision] Full-page cache saved: {len(_full_page_cache)} pages ({full_page_cache_file.name})")

    # ── Phase 1.3: Page Continuation Stitching ──────────────────
    if "D" in stages:
        print("\n[Stage D] Stitching multi-page blocks...")
        _stitch_blocks(document["pages"])

        save_json(document, build_dir / "document.json")
        print(f"  document.json saved ({len(document['pages'])} pages)")

    # ── Stage E: Structure Reconstruction ────────────────────────
    if "E" in stages:
        print("\n[Stage E] Building heading hierarchy...")
        from src.structure import build_structure
        if "D" not in stages:
            document = load_json(build_dir / "document.json")
        with Timer("Structure"):
            build_structure(document["pages"], config)
        save_json(document, build_dir / "document.json")
        print("  document.json updated with section_path")

    # ── Stage F: Build Retrieval Corpus ──────────────────────────
    if "F" in stages:
        print("\n[Stage F] Building retrieval chunks...")
        from src.chunker import build_chunks, build_toc_chunks, build_csv_table_chunks
        if "D" not in stages and "E" not in stages:
            document = load_json(build_dir / "document.json")
        with Timer("Chunking"):
            chunks, figures = build_chunks(document, config)

        # Inject manually-written TOC as additional chunks
        toc_md = config.get("pipeline", {}).get("toc_markdown")
        if toc_md:
            toc_path = resolve_path(toc_md, PROJECT_ROOT)
            toc_chunks = build_toc_chunks(
                toc_path,
                doc_id=config.get("pipeline", {}).get("doc_id", "geo_metro_1990")
            )
            if toc_chunks:
                chunks = toc_chunks + chunks  # TOC first for stable ordering
                print(f"  Added {len(toc_chunks)} TOC chunks from {toc_path.name}")

        # Inject hand-curated CSV tables
        manual_tables = config.get("pipeline", {}).get("manual_tables", [])
        csv_count = 0
        for tbl_spec in manual_tables:
            csv_chunks = build_csv_table_chunks(
                tbl_spec,
                doc_id=config.get("pipeline", {}).get("doc_id", "geo_metro_1990"),
                project_root=PROJECT_ROOT,
            )
            if csv_chunks:
                chunks = chunks + csv_chunks
                csv_count += len(csv_chunks)
                print(f"  Added {len(csv_chunks)} chunks from {tbl_spec.get('source_label', tbl_spec.get('path'))}")

        from src.utils import save_jsonl
        save_jsonl(chunks, build_dir / "chunks.jsonl")
        save_jsonl(figures, build_dir / "figures.jsonl")
        print(f"  {len(chunks)} chunks total, {len(figures)} figures")

    # ── Stage G: Build Indices ───────────────────────────────────
    if "G" in stages:
        print("\n[Stage G] Building search indices...")
        from src.index_build import build_indices
        index_dir = resolve_path("tools/rag_index", PROJECT_ROOT)
        with Timer("Indexing"):
            build_indices(build_dir / "chunks.jsonl", config, index_dir)
        print(f"  Indices saved to {index_dir}")

    # Update manifest
    manifest_path = build_dir / "manifest.json"
    if manifest_path.exists():
        manifest_data = load_json(manifest_path)
    else:
        manifest_data = {}
    manifest_data["timestamps"] = manifest_data.get("timestamps", {})
    manifest_data["timestamps"]["pipeline_end"] = now_iso()
    save_json(manifest_data, manifest_path)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


def _stitch_blocks(pages: list[dict]):
    """Merge blocks that continue across page boundaries.

    Checks ALL blocks with continues_to_next_page (not just the last block),
    since figures/headers can appear after a continuing procedure on the same page.
    """
    for i in range(len(pages) - 1):
        curr_page = pages[i]
        next_page = pages[i + 1]

        if not curr_page["blocks"] or not next_page["blocks"]:
            continue

        # Process CONT_TO blocks in reverse order: last block on the page
        # is closest to the page boundary and should match first.
        for src_block in reversed(curr_page["blocks"]):
            if not src_block.get("continues_to_next_page"):
                continue

            # Find matching continuation block on the next page.
            # Prefer same-type match, but allow type mismatch (vision
            # sometimes types a procedure continuation as "paragraph").
            cont_idx = -1
            fallback_idx = -1
            for idx, b in enumerate(next_page["blocks"]):
                if not b.get("continues_from_previous_page"):
                    continue
                if b["type"] == src_block["type"]:
                    cont_idx = idx
                    break
                if fallback_idx == -1:
                    fallback_idx = idx
            if cont_idx == -1:
                cont_idx = fallback_idx

            if cont_idx == -1:
                continue

            cont_block = next_page["blocks"][cont_idx]
            print(f"      Stitching {src_block['type']} from p{curr_page['page_num']} to p{next_page['page_num']}")

            # Merge text
            if src_block.get("text") and cont_block.get("text"):
                src_block["text"] = src_block["text"].strip() + "\n" + cont_block["text"].strip()
            elif cont_block.get("text"):
                src_block["text"] = cont_block["text"]

            # Merge steps
            if src_block.get("steps") and cont_block.get("steps"):
                src_block["steps"].extend(cont_block["steps"])

            # Merge rows for tables
            if src_block.get("rows") and cont_block.get("rows"):
                src_block["rows"].extend(cont_block["rows"])

            # Update continuation flag
            src_block["continues_to_next_page"] = cont_block.get("continues_to_next_page", False)
            src_block["merged_across_pages"] = True

            # Remove the merged block from the next page
            next_page["blocks"].pop(cont_idx)

            # Re-index remaining blocks on next page
            for idx, b in enumerate(next_page["blocks"]):
                b["order_index"] = idx

    # Cleanup: clear unmatched continues_to_next_page flags.
    # These are false positives from the vision model — the text ends
    # with a complete sentence but the model incorrectly flagged it.
    cleared = 0
    for page in pages:
        for block in page["blocks"]:
            if block.get("continues_to_next_page") and not block.get("merged_across_pages"):
                block["continues_to_next_page"] = False
                cleared += 1
    if cleared:
        print(f"      Cleared {cleared} unmatched continues_to_next_page flags")


def main():
    parser = argparse.ArgumentParser(description="Metro Manual Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pages", help="Page range, e.g. 1-30")
    parser.add_argument("--stages", help="Comma-separated stages, e.g. A,B,C")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    args = parser.parse_args()

    page_range = None
    if args.pages:
        parts = args.pages.split("-")
        if len(parts) == 1:
            page_range = (int(parts[0]), int(parts[0]))
        else:
            page_range = (int(parts[0]), int(parts[1]))

    stages = None
    if args.stages:
        stages = set(args.stages.upper().split(","))

    run_pipeline(
        config_path=args.config,
        page_range=page_range,
        stages=stages,
        skip_vision=args.skip_vision,
        reprocess=args.reprocess,
    )


if __name__ == "__main__":
    main()
