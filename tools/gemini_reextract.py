"""Re-extract pages missing from full_page_vision_cache using Gemini Vision API.

Populates build/full_page_vision_cache.json for pages that currently fall back
to OCR extraction, so the pipeline can use structured vision output for all pages.

Usage:
    python tools/gemini_reextract.py                    # extract all missing pages
    python tools/gemini_reextract.py --pages 18,42,100  # specific pages only
    python tools/gemini_reextract.py --dry-run           # show what would be done
    python tools/gemini_reextract.py --delay 2.0         # seconds between requests
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

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision_extract import FULL_PAGE_PROMPT
from src.utils import load_json, save_json, load_config

# ── Constants ────────────────────────────────────────────────────────────────

MAX_IMAGE_DIM = 1600
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_DELAY = 1.0
MAX_RETRIES = 3
INITIAL_BACKOFF = 5.0  # seconds, doubles on each retry


def load_cache(cache_path: Path) -> dict:
    """Load existing full-page vision cache."""
    if cache_path.exists():
        return load_json(cache_path)
    return {}


def find_missing_pages(cache: dict, pages_pre_dir: Path,
                       config: dict) -> list[int]:
    """Return sorted list of page numbers that have preprocessed images but no cache entry."""
    # Determine which pages exist as preprocessed images
    available = set()
    for f in pages_pre_dir.iterdir():
        if f.suffix == ".png" and f.stem.startswith("page_"):
            try:
                pn = int(f.stem.split("_")[1])
                available.add(pn)
            except (IndexError, ValueError):
                continue

    # Exclude front-matter and skip_pages from config
    front_matter = config.get("pipeline", {}).get("front_matter_pages", 0)
    skip_pages = set(config.get("pipeline", {}).get("skip_pages", []))

    eligible = {pn for pn in available
                if pn > front_matter and pn not in skip_pages}

    cached = {int(k) for k in cache.keys()}
    missing = sorted(eligible - cached)
    return missing


def prepare_image(img_path: Path) -> str:
    """Load image, resize to max dim, convert to RGB, return JPEG base64."""
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


def call_gemini_vision(b64_image: str, page_num: int, model: str,
                       api_key: str) -> dict | None:
    """Send image + prompt to Gemini Vision API and return parsed JSON."""
    model_name = model.removeprefix("models/")
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model_name}:generateContent")

    prompt = FULL_PAGE_PROMPT.replace("<int>", str(page_num))

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_image,
                    }
                },
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

            # Check for blocked / empty responses
            candidates = out.get("candidates", [])
            if not candidates:
                reason = out.get("promptFeedback", {}).get("blockReason", "unknown")
                print(f"      No candidates returned (blockReason: {reason})")
                return None

            candidate = candidates[0]
            finish = candidate.get("finishReason", "")

            # Safety-blocked responses have no content
            if "content" not in candidate:
                print(f"      Response blocked (finishReason: {finish})")
                return None

            text = candidate["content"]["parts"][0]["text"]
            return _parse_json(text)

        except requests.exceptions.Timeout:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            print(f"      Timeout, retrying in {wait:.0f}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"      Error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF)
                continue
            return None

    return None


def _parse_json(text: str) -> dict | None:
    """Find and parse JSON from LLM response, with repair for truncated output."""
    # Try clean parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    snippet = text[start:end + 1]

    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    # Repair: truncated JSON — close open arrays/objects
    repaired = _repair_json(snippet)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            print(f"      Failed to parse JSON even after repair: {e}")

    return None


def _repair_json(text: str) -> str | None:
    """Attempt to close truncated JSON by balancing brackets."""
    # Remove any trailing comma before we close
    stripped = text.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1]

    # Count unmatched brackets/braces (ignoring those inside strings)
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False

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
        return None  # more closes than opens — can't fix

    # Append missing closers
    closing = "]" * open_brackets + "}" * open_braces
    if closing:
        return stripped + closing
    return stripped


def validate_result(result: dict, page_num: int) -> bool:
    """Basic validation of the extracted page structure."""
    if not isinstance(result, dict):
        return False
    if "blocks" not in result:
        return False
    if not isinstance(result["blocks"], list):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract OCR pages via Gemini Vision API")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated page numbers to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show missing pages without processing")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})")
    parser.add_argument("--claude-fallback", action="store_true",
                        help="Use Claude Vision for pages that Gemini blocks")
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        # Try loading from .env
        try:
            from dotenv import load_dotenv
            load_dotenv(PROJECT_ROOT / ".env")
            api_key = os.environ.get("GEMINI_API_KEY")
        except ImportError:
            pass
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set. Export it or add to .env")
            sys.exit(1)

    # Load config
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    build_dir = PROJECT_ROOT / config["pipeline"]["build_dir"]
    cache_path = build_dir / "full_page_vision_cache.json"
    pages_pre_dir = build_dir / "pages_pre"

    if not pages_pre_dir.exists():
        print(f"ERROR: Preprocessed pages not found at {pages_pre_dir}")
        sys.exit(1)

    # Load cache and find missing pages
    cache = load_cache(cache_path)
    print(f"Cache loaded: {len(cache)} pages already cached")

    missing = find_missing_pages(cache, pages_pre_dir, config)
    print(f"Missing pages: {len(missing)}")

    # Filter to requested pages if specified
    if args.pages:
        requested = {int(p.strip()) for p in args.pages.split(",")}
        target = sorted(requested & set(missing))
        skipped = requested - set(missing)
        if skipped:
            print(f"Already cached (skipping): {sorted(skipped)}")
    else:
        target = missing

    if not target:
        print("Nothing to do - all pages are cached!")
        return

    print(f"Pages to extract: {len(target)}")
    if args.dry_run:
        print(f"Page numbers: {target}")
        return

    print(f"Model: {args.model}")
    print(f"Delay: {args.delay}s between requests")
    print()

    # Process pages
    failed = []
    t0 = time.time()

    for i, pn in enumerate(target):
        img_path = pages_pre_dir / f"page_{pn:04d}.png"
        if not img_path.exists():
            print(f"  [{i+1}/{len(target)}] p{pn:04d} - SKIP (image not found)")
            failed.append(pn)
            continue

        print(f"  [{i+1}/{len(target)}] p{pn:04d} ...", end=" ", flush=True)

        b64 = prepare_image(img_path)
        result = call_gemini_vision(b64, pn, args.model, api_key)

        if result and validate_result(result, pn):
            n_blocks = len(result.get("blocks", []))
            cache[str(pn)] = result
            # Save after every page for crash safety
            save_json(cache, cache_path)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            print(f"OK ({n_blocks} blocks) [{rate:.1f} pages/min]")
        else:
            print("FAILED")
            failed.append(pn)

        # Delay between requests (skip after last)
        if i < len(target) - 1:
            time.sleep(args.delay)

    # Summary
    elapsed = time.time() - t0
    print()
    print("=" * 50)
    print(f"Done in {elapsed:.1f}s")
    print(f"Cache now has {len(cache)} pages")
    print(f"Succeeded: {len(target) - len(failed)}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")
    else:
        print("All pages extracted successfully!")
    print()
    print("Next step: re-run pipeline stages D-G:")
    print('  python -m src.pipeline --stages "D,E,F,G" --reprocess')


if __name__ == "__main__":
    main()
