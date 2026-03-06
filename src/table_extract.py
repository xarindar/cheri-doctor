"""Stage D.3: Structured Table Extraction.

Extracts tables from detected table regions and produces:
- rows[][] for document.json
- CSV file in build/tables/
- WebP image crop in build/tables/
- retrieval_text for BM25/embedding indexing
"""

import csv
import re
from pathlib import Path
from PIL import Image

from src.models import TableResult
from src.utils import resolve_path

# Import proven extractors from legacy pipeline
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "legacy"))
os.environ["PATH"] = r"C:\Program Files\Tesseract-OCR" + os.pathsep + os.environ.get("PATH", "")

from table_extract_v2 import extract_table as _img2table_extract
from table_extract import detect_and_extract_table as _opencv_extract


def extract_table_structured(region,
                              page_img: Image.Image,
                              page_num: int,
                              tbl_idx: int,
                              config: dict,
                              project_root: Path,
                              rows_override: list[list[str]] | None = None,
                              skip_ml: bool = False) -> TableResult | None:
    """Full table extraction pipeline for one table region.

    Args:
        region:       Surya Region with bbox.
        page_img:     Full original page image.
        page_num:     1-based page number.
        tbl_idx:      Index of this table on the page.
        config:       Pipeline config.
        project_root: Project root Path.

    Returns:
        TableResult or None if extraction failed.
    """
    build_dir  = resolve_path(config["pipeline"]["build_dir"], project_root)
    tables_dir = build_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cfg_tbl = config.get("table_extraction", {})
    webp_q  = config.get("figure_extraction", {}).get("webp_quality", 85)
    debug   = bool(cfg_tbl.get("debug", False))

    # Crop table region from original (full-colour) image
    w, h = page_img.size
    x1 = max(0, int(region.x1))
    y1 = max(0, int(region.y1))
    x2 = min(w, int(region.x2))
    y2 = min(h, int(region.y2))
    crop = page_img.crop((x1, y1, x2, y2))

    debug_dir = tables_dir / "debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _debug_write(
            debug_dir, page_num, tbl_idx,
            f"bbox=({x1},{y1})-({x2},{y2}) page_size=({w}x{h}) crop_size=({x2-x1}x{y2-y1})"
        )
        # Preserve exact crop used for extraction
        crop_png = debug_dir / f"page_{page_num:04d}_table_{tbl_idx:03d}_crop.png"
        crop.save(str(crop_png), "PNG")

    rows    = None
    method  = None

    # ── Override from Vision ──────────────────────────────────────
    if rows_override:
        rows   = rows_override
        method = "vision"

    # ── Primary: img2table ────────────────────────────────────────
    if rows is None and not skip_ml:
        try:
            result = _img2table_extract(crop)
            if debug:
                if result is None:
                    _debug_write(debug_dir, page_num, tbl_idx, "img2table: result=None")
                else:
                    md = result.get("markdown")
                    md_len = len(md) if md else 0
                    _debug_write(debug_dir, page_num, tbl_idx, f"img2table: markdown_len={md_len}")
                    if md:
                        (_debug_markdown_path(debug_dir, page_num, tbl_idx, "img2table")
                            .write_text(md, encoding="utf-8"))
            if result and result.get("markdown"):
                rows   = _markdown_to_rows(result["markdown"])
                method = "img2table"
        except Exception as e:
            if debug:
                _debug_write(debug_dir, page_num, tbl_idx, f"img2table: exception={type(e).__name__}: {e}")

    # ── Fallback: OpenCV pipeline ─────────────────────────────────
    if rows is None and not skip_ml:
        try:
            import numpy as np, cv2
            crop_np  = np.array(crop)
            crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
            md = _opencv_extract(crop_bgr)
            if debug:
                md_len = len(md) if md else 0
                _debug_write(debug_dir, page_num, tbl_idx, f"opencv: markdown_len={md_len}")
                if md:
                    (_debug_markdown_path(debug_dir, page_num, tbl_idx, "opencv")
                        .write_text(md, encoding="utf-8"))
            if md:
                rows   = _markdown_to_rows(md)
                method = "opencv"
        except Exception as e:
            if debug:
                _debug_write(debug_dir, page_num, tbl_idx, f"opencv: exception={type(e).__name__}: {e}")

    rows = _normalize_rows(rows or [], cfg_tbl)
    if not rows or len(rows) < cfg_tbl.get("min_rows", 2):
        if debug:
            _debug_write(debug_dir, page_num, tbl_idx, f"fail: rows={0 if not rows else len(rows)} min_rows={cfg_tbl.get('min_rows', 2)}")
        return None

    # Check column count
    max_cols = max(len(r) for r in rows) if rows else 0
    if max_cols < cfg_tbl.get("min_cols", 2):
        if debug:
            _debug_write(debug_dir, page_num, tbl_idx, f"fail: max_cols={max_cols} min_cols={cfg_tbl.get('min_cols', 2)}")
        return None

    # ── Export CSV ────────────────────────────────────────────────
    csv_name = f"page_{page_num:04d}_table_{tbl_idx:03d}.csv"
    csv_path = tables_dir / csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # ── Export WebP crop ──────────────────────────────────────────
    webp_name  = f"page_{page_num:04d}_table_{tbl_idx:03d}.webp"
    asset_path = tables_dir / webp_name
    crop.save(str(asset_path), "WEBP", quality=webp_q)

    # ── Build retrieval text ──────────────────────────────────────
    retrieval_text = _render_retrieval_text(rows)

    return TableResult(
        rows=rows,
        csv_path=str(csv_path.relative_to(project_root)).replace("\\", "/"),
        asset_path=str(asset_path.relative_to(project_root)).replace("\\", "/"),
        retrieval_text=retrieval_text,
        method=method,
    )


def _debug_write(debug_dir: Path, page_num: int, tbl_idx: int, line: str) -> None:
    log_path = debug_dir / f"page_{page_num:04d}_table_{tbl_idx:03d}.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _debug_markdown_path(debug_dir: Path, page_num: int, tbl_idx: int, method: str) -> Path:
    return debug_dir / f"page_{page_num:04d}_table_{tbl_idx:03d}_{method}.md"


def _markdown_to_rows(markdown: str) -> list[list[str]]:
    """Convert a markdown table string to rows[][], tolerant of wrapped/misaligned lines."""
    rows: list[list[str]] = []

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue

        has_pipe = "|" in line
        leading_pipe = line.startswith("|")
        trailing_pipe = line.endswith("|")

        if has_pipe and not leading_pipe:
            line = "| " + line
            leading_pipe = True
        if has_pipe and not trailing_pipe:
            line = line + " |"

        if not has_pipe:
            # Continuation with no pipes: append to first cell of previous row.
            if rows:
                rows[-1][0] = (rows[-1][0] + " " + line).strip()
            continue

        if line.startswith("|---") or line.startswith("| --") or set(line) <= set("|-: "):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]

        # If the original line lacked a leading pipe, treat as continuation.
        if not leading_pipe and rows:
            last = rows[-1]
            max_len = max(len(last), len(cells))
            if len(last) < max_len:
                last += [""] * (max_len - len(last))
            if len(cells) < max_len:
                cells += [""] * (max_len - len(cells))
            for i, c in enumerate(cells):
                if c:
                    last[i] = (last[i] + " " + c).strip() if last[i] else c
            rows[-1] = last
        else:
            rows.append(cells)

    return rows


def _render_retrieval_text(rows: list[list[str]]) -> str:
    """Build a human-readable text representation for embedding/BM25.

    Uses the first row as headers if it looks like a header row.
    """
    if not rows:
        return ""

    lines = []
    headers = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else rows

    # Check if first row looks like headers (no purely numeric cells)
    is_header = headers and not any(
        re.match(r"^\d+\.?\d*$", h.strip()) for h in headers if h.strip()
    )

    if is_header:
        lines.append(" | ".join(headers))
        for row in data_rows:
            parts = []
            for i, cell in enumerate(row):
                header = headers[i] if i < len(headers) else f"Col{i}"
                if cell.strip():
                    parts.append(f"{header}: {cell}")
            if parts:
                lines.append(", ".join(parts))
    else:
        for idx, row in enumerate(rows):
            lines.append(f"Row {idx + 1}: " + " | ".join(row))

    return "\n".join(lines)


def _normalize_rows(rows: list[list[str]], cfg_tbl: dict) -> list[list[str]]:
    """Clean OCR table rows, fill down merged cells, and merge wrapped lines."""
    if not rows:
        return rows

    # Normalize cell values and row widths
    max_cols = max(len(r) for r in rows) if rows else 0
    cleaned = []
    for r in rows:
        rr = [(c.strip() if isinstance(c, str) else "") for c in r]
        rr = [c.replace("©", "•").replace("®", "•") for c in rr]
        rr = [("" if c.lower() == "nan" else c) for c in rr]
        if len(rr) < max_cols:
            rr += [""] * (max_cols - len(rr))
        cleaned.append(rr)

    # Fill down merged cells in the first column (for diagnostic tables)
    if max_cols > 1 and cfg_tbl.get("fill_down_first_column", True):
        last_condition = ""
        # Start from row 1 to preserve header, if any
        start_row = 1 if len(rows) > 1 and _looks_like_header(rows[0]) else 0
        for i in range(start_row, len(cleaned)):
            # A row is a continuation if the first col is empty but a subsequent col is not
            is_continuation = not cleaned[i][0].strip() and any(c.strip() for c in cleaned[i][1:])
            if is_continuation:
                cleaned[i][0] = last_condition
            else:
                last_condition = cleaned[i][0]

    cleaned = _enforce_symptom_cause_cure(cleaned, cfg_tbl)

    merged: list[list[str]] = []
    for r in cleaned:
        if not merged:
            merged.append(r)
            continue
        # Never merge into header row (keep row 0 intact)
        if len(merged) == 1 and _looks_like_header(merged[0]):
            merged.append(r)
            continue

        prev = merged[-1]
        row_has_nonempty = any(c for c in r)
        if not row_has_nonempty:
            continue

        # If first col empty, treat as continuation for any non-empty columns.
        if not r[0]:
            for i, c in enumerate(r):
                if c:
                    prev[i] = (prev[i] + " " + c).strip() if prev[i] else c
            continue

        # If only first col has text, and prev has other columns, merge into prev col0.
        if r[0] and not any(c for c in r[1:]) and any(prev[i] for i in range(1, len(prev))):
            prev[0] = (prev[0] + " " + r[0]).strip()
            continue

        merged.append(r)

    return merged


def _enforce_symptom_cause_cure(rows: list[list[str]], cfg_tbl: dict) -> list[list[str]]:
    """If table looks like symptom/cause/cure, normalize to 3 cols and add header."""
    if not rows:
        return rows

    max_cols = max(len(r) for r in rows)
    norm_rows = []
    for r in rows:
        if max_cols <= 3:
            norm_rows.append(r + [""] * (3 - len(r)))
        else:
            merged = r[:2] + [" ".join([c for c in r[2:] if c]).strip()]
            norm_rows.append((merged + [""] * (3 - len(merged)))[:3])

    header_cells = [c.lower() for c in norm_rows[0]]
    has_symptom = any("symptom" in c for c in header_cells)
    has_cause = any("cause" in c for c in header_cells)
    has_cure = any("cure" in c for c in header_cells)

    looks_like_table = cfg_tbl.get("infer_symptom_cause_cure_header", False) and _looks_like_troubleshooting(norm_rows)

    if has_symptom and has_cause and has_cure:
        norm_rows[0] = ["SYMPTOM", "CAUSE", "CURE"]
        return norm_rows

    if looks_like_table:
        return [["SYMPTOM", "CAUSE", "CURE"]] + norm_rows

    return norm_rows


def _maybe_infer_symptom_cause_cure_header(rows: list[list[str]]) -> list[list[str]]:
    """Heuristic: if table looks like a troubleshooting matrix, insert header."""
    if not rows:
        return rows

    max_cols = max(len(r) for r in rows)
    if max_cols != 3 or len(rows) < 3:
        return rows

    def _has_bullet(s: str) -> bool:
        return "•" in s

    bullet_rows = 0
    symptom_like = 0
    for r in rows[:10]:
        c0, c1, c2 = (r + ["", "", ""])[:3]
        if _has_bullet(c1) or _has_bullet(c2):
            bullet_rows += 1
        if len(c0) >= 12 and " " in c0:
            symptom_like += 1

    if bullet_rows >= 2 and symptom_like >= 2:
        return [["SYMPTOM", "CAUSE", "CURE"]] + rows

    return rows


def _looks_like_header(row: list[str]) -> bool:
    """Return True if a row looks like a header (no purely numeric cells)."""
    return row and not any(
        re.match(r"^\d+\.?\d*$", cell.strip()) for cell in row if cell.strip()
    )


def _looks_like_troubleshooting(rows: list[list[str]]) -> bool:
    if not rows or max(len(r) for r in rows) < 3:
        return False
    def has_bullet(s: str) -> bool:
        return "•" in s or "°" in s or "©" in s or "®" in s
    bullet_rows = 0
    symptom_like = 0
    for r in rows[:10]:
        c0, c1, c2 = (r + ["", "", ""])[:3]
        if has_bullet(c1) or has_bullet(c2):
            bullet_rows += 1
        if len(c0) >= 12 and " " in c0:
            symptom_like += 1
    return bullet_rows >= 2 and symptom_like >= 2
