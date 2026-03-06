"""Re-extract diagnostic flowchart pages via Gemini with a flowchart-aware prompt.

These pages contain decision-tree flowcharts that Gemini previously misclassified
as fragmented procedure blocks. This script uses a specialized prompt that tells
Gemini to output them as structured figure blocks with key:value decision paths.

Usage:
    python tools/gemini_flowchart_fix.py                # extract all suspected flowchart pages
    python tools/gemini_flowchart_fix.py --dry-run      # preview which pages
    python tools/gemini_flowchart_fix.py --pages 387,395
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_json, save_json, load_config

MAX_IMAGE_DIM = 1600
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF = 5.0

# Flowchart-specific prompt that produces structured figure output
FLOWCHART_PROMPT = """You are extracting content from a page of an automotive service manual.

This page contains a DIAGNOSTIC FLOWCHART (decision tree). It has numbered steps
in boxes with arrows showing decision paths (OK/NOT OK, YES/NO, code results, etc.).

Return a JSON object with this exact structure:
{
  "page": <PAGE_NUM>,
  "section_id": <string>,
  "section_title": <string>,
  "page_label": <string>,
  "reading_order": ["block_0", "block_1", ...],
  "blocks": [...]
}

Block rules for this flowchart page:

1. The chart TITLE/HEADER (e.g. "CODE 14 ...", "CHART A-7 ...") goes in ONE "header" block.

2. Any NOTE before the flowchart (e.g. "NOTE: IF CODE 22 IS PRESENT...") is a "note" block.

3. The ENTIRE FLOWCHART must be ONE "figure" block. Do NOT split each step into a
   separate procedure block. Encode the decision tree as structured text using this format:
   - "step_N:" for each numbered step box, with ALL bullet points from that box
   - "step_N_result:" for each branch outcome (use descriptive suffixes like _ok, _not_ok,
     _yes, _no, _code_14, _code_15, _high, _low, etc.)
   - For branches that lead to another step, write "Proceed to Step N"
   - For terminal results (diagnosis boxes), write the full diagnosis text

   Example text for a figure block:
   "step_1: IGNITION OFF. DISCONNECT SENSOR. CHECK VOLTAGE...\\nstep_1_ok: Proceed to Step 2\\nstep_1_not_ok: FAULTY WIRING\\nstep_2: CHECK RESISTANCE...\\nstep_2_ok: INTERMITTENT FAULT\\nstep_2_not_ok: REPLACE SENSOR"

4. Any DIAGNOSTIC AID table stays as a separate "table" block with "rows".

5. "CLEAR CODES AND CONFIRM..." at the bottom is a "notice" block.

6. Any "Circuit Description" or "Test Description" paragraphs are "paragraph" blocks.

Important:
- The flowchart figure block title should describe what it diagnoses.
- Preserve ALL text from every box in the flowchart.
- Include the branch condition labels (OK, NOT OK, YES, NO, CODE XX, voltage thresholds, etc.)
- Keep exact measurements, part numbers, and specifications.
"""


def prepare_image(img_path: Path) -> str:
    img = Image.open(img_path)
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_gemini(b64_image: str, page_num: int, model: str, api_key: str) -> dict | None:
    model_name = model.removeprefix("models/")
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model_name}:generateContent")

    prompt = FLOWCHART_PROMPT.replace("<PAGE_NUM>", str(page_num))

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": api_key},
                json=payload,
                timeout=120,
            )

            if resp.status_code in (429, 503):
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"      Rate limited ({resp.status_code}), waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            if not resp.ok:
                print(f"      API error {resp.status_code}: {resp.text[:200]}")
                return None

            out = resp.json()
            candidates = out.get("candidates", [])
            if not candidates:
                reason = out.get("promptFeedback", {}).get("blockReason", "unknown")
                print(f"      No candidates (blockReason: {reason})")
                return None

            candidate = candidates[0]
            if "content" not in candidate:
                print(f"      Response blocked (finishReason: {candidate.get('finishReason')})")
                return None

            text = candidate["content"]["parts"][0]["text"]
            return _parse_json(text)

        except requests.exceptions.Timeout:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            print(f"      Timeout, retrying in {wait:.0f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"      Error attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF)

    return None


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            # Try repairing truncated JSON
            repaired = _repair_json(snippet)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError as e:
                    print(f"      JSON parse failed after repair: {e}")
    return None


def _repair_json(text: str) -> str | None:
    stripped = text.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1]
    open_braces = open_brackets = 0
    in_string = escape = False
    for ch in stripped:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1
    if open_braces < 0 or open_brackets < 0:
        return None
    return stripped + "]" * open_brackets + "}" * open_braces


def find_flowchart_pages(cache: dict) -> list[int]:
    """Find pages that look like flowcharts but are fragmented as procedures."""
    suspects = []
    for pn_str, entry in cache.items():
        sid = entry.get("section_id", "")
        if not sid.startswith("6E"):
            continue
        blocks = entry.get("blocks", [])
        types = [b.get("type") for b in blocks]

        # Already fixed (has figure blocks)
        if "figure" in types:
            continue

        has_proc = "procedure" in types
        has_clear_codes = any("CLEAR CODES" in (b.get("text") or "") for b in blocks)
        proc_count = types.count("procedure")

        if has_proc and has_clear_codes and proc_count >= 2:
            suspects.append(int(pn_str))

    return sorted(suspects)


def validate_result(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    blocks = result.get("blocks", [])
    if not isinstance(blocks, list) or len(blocks) == 0:
        return False
    types = [b.get("type") for b in blocks]
    # Must have at least one figure block (the flowchart)
    if "figure" not in types:
        print("      WARNING: No figure block in result - Gemini may have ignored flowchart instruction")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-extract flowchart pages via Gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated page numbers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    # API key
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    build_dir = PROJECT_ROOT / config["pipeline"]["build_dir"]
    cache_path = build_dir / "full_page_vision_cache.json"
    pages_pre_dir = build_dir / "pages_pre"

    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache loaded: {len(cache)} pages")

    if args.pages:
        target = sorted(int(p.strip()) for p in args.pages.split(","))
    else:
        target = find_flowchart_pages(cache)

    print(f"Flowchart pages to re-extract: {len(target)}")
    print(f"Pages: {target}")

    if args.dry_run or not target:
        return

    print(f"Model: {args.model}")
    print()

    failed = []
    t0 = time.time()

    for i, pn in enumerate(target):
        img_path = pages_pre_dir / f"page_{pn:04d}.png"
        if not img_path.exists():
            print(f"  [{i+1}/{len(target)}] p{pn:04d} - SKIP (no image)")
            failed.append(pn)
            continue

        print(f"  [{i+1}/{len(target)}] p{pn:04d} ...", end=" ", flush=True)

        b64 = prepare_image(img_path)
        result = call_gemini(b64, pn, args.model, api_key)

        if result and validate_result(result):
            blocks = result.get("blocks", [])
            fig_count = sum(1 for b in blocks if b.get("type") == "figure")
            cache[str(pn)] = result
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            print(f"OK ({len(blocks)} blocks, {fig_count} figures)")
        else:
            print("FAILED")
            failed.append(pn)

        if i < len(target) - 1:
            time.sleep(args.delay)

    elapsed = time.time() - t0
    print()
    print("=" * 50)
    print(f"Done in {elapsed:.1f}s")
    print(f"Succeeded: {len(target) - len(failed)}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
    else:
        print("All flowchart pages fixed!")


if __name__ == "__main__":
    main()
