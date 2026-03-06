"""Stage A: PDF Rasterization.

Renders each page of a scanned PDF at the configured DPI to individual PNG files.
Creates the initial manifest.json with document metadata.
"""

import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

from src.models import Manifest
from src.utils import (
    file_hash, config_hash, generate_run_id, now_iso,
    save_json, to_dict, resolve_path, Timer,
)


def rasterize_pdf(config: dict, project_root: Path) -> tuple[Manifest, list[dict]]:
    """Render all pages of the source PDF at configured DPI.

    Args:
        config: Pipeline configuration dict.
        project_root: Project root directory.

    Returns:
        (manifest, page_metas) where page_metas is a list of
        {"page_num": int, "path": str, "width": int, "height": int}.
    """
    pdf_path = resolve_path(config["pipeline"]["source_pdf"], project_root)
    build_dir = resolve_path(config["pipeline"]["build_dir"], project_root)
    dpi = config["rasterize"]["dpi"]
    pages_dir = build_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = Manifest(
        doc_id=config["pipeline"]["doc_id"],
        source_pdf=str(pdf_path.name),
        dpi=dpi,
        run_id=generate_run_id(),
        config_hash=config_hash(config),
    )

    # Hash source PDF (can be slow for large files)
    print(f"  Hashing source PDF ({pdf_path.name})...")
    manifest.source_hash = file_hash(pdf_path)

    doc = fitz.open(str(pdf_path))
    manifest.page_count = doc.page_count
    manifest.timestamps["rasterize_start"] = now_iso()

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    page_metas = []

    # Count how many pages need rendering
    to_render = [i for i in range(doc.page_count)
                 if not (pages_dir / f"page_{i + 1:04d}.png").exists()]
    if to_render:
        print(f"  Rasterizing {len(to_render)}/{doc.page_count} pages at {dpi} DPI"
              f" (skipping {doc.page_count - len(to_render)} existing)...")
    else:
        print(f"  All {doc.page_count} pages already rasterized, skipping.")

    for i in range(doc.page_count):
        out_path = pages_dir / f"page_{i + 1:04d}.png"

        if out_path.exists():
            # Read dimensions from existing file without re-rendering
            with Image.open(out_path) as img:
                w, h = img.size
        else:
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(str(out_path), "PNG")
            w, h = pix.width, pix.height

            if (i + 1) % 50 == 0 or (i + 1) == doc.page_count:
                print(f"    {i + 1}/{doc.page_count}")

        page_metas.append({
            "page_num": i + 1,
            "path": str(out_path),
            "width": w,
            "height": h,
        })

    doc.close()
    manifest.timestamps["rasterize_end"] = now_iso()

    # Save manifest
    manifest_path = build_dir / "manifest.json"
    save_json(to_dict(manifest), manifest_path)
    print(f"  Manifest saved to {manifest_path}")

    return manifest, page_metas
