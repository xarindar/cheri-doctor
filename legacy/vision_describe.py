"""
AI Vision module for diagram description.

Sends diagram images to Claude Vision API (or GPT-4o fallback) with
automotive-context prompts to generate rich, searchable descriptions
of diagrams, exploded views, wiring schematics, etc.
"""

import base64
import io
import json
import os
import time
from pathlib import Path
from PIL import Image

# ── API Client Setup ─────────────────────────────────────────────────────

ANTHROPIC_AVAILABLE = False
OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

# ── Configuration ────────────────────────────────────────────────────────

MAX_IMAGE_DIM = 1568  # Max dimension for API submission (cost control)
RATE_LIMIT_DELAY = 1.0  # Seconds between API calls
MAX_RETRIES = 3

# Track last API call time for rate limiting
_last_call_time = 0.0

# Track already-described diagrams to avoid re-processing
_described_cache: dict[str, str] = {}

DIAGRAM_PROMPT = """You are analyzing a page from a 1990 Geo Metro / Suzuki Swift service manual.
Describe this diagram in detail for a text-based AI assistant that cannot see images.

Include:
1. **Diagram type** (exploded view, wiring schematic, system overview, cutaway, flowchart, etc.)
2. **All labeled components** with their reference numbers (e.g., "1. Compressor", "2. Condenser")
3. **Spatial relationships** (what connects to what, flow direction, relative positions)
4. **All visible text** (labels, notes, specifications, callout numbers)
5. **Assembly/disassembly sequences** if shown (numbered steps, arrows indicating order)
6. **Measurements or specifications** if visible

Format your response as structured markdown with clear headings.
Be thorough — this description will be the ONLY way to search and reference this diagram's content."""


# ── Image Preparation ────────────────────────────────────────────────────

def _prepare_image(img: Image.Image) -> tuple[str, str]:
    """Resize image if needed and encode as base64.

    Returns (base64_data, media_type).
    """
    # Downscale if larger than MAX_IMAGE_DIM
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Encode as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    return b64, "image/png"


def _rate_limit():
    """Enforce rate limiting between API calls."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_call_time = time.time()


# ── Claude Vision ────────────────────────────────────────────────────────

def _describe_with_claude(img: Image.Image) -> str | None:
    """Send image to Claude Vision API for description."""
    if not ANTHROPIC_AVAILABLE:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    b64_data, media_type = _prepare_image(img)
    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        _rate_limit()
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": DIAGRAM_PROMPT,
                        },
                    ],
                }],
            )
            return response.content[0].text
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = RATE_LIMIT_DELAY * (attempt + 2)
                print(f"  [vision] Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            print(f"  [vision] Claude API error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return None

    return None


# ── GPT-4o Fallback ─────────────────────────────────────────────────────

def _describe_with_openai(img: Image.Image) -> str | None:
    """Fallback: send image to GPT-4o for description."""
    if not OPENAI_AVAILABLE:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    b64_data, media_type = _prepare_image(img)
    client = openai.OpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        _rate_limit()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": DIAGRAM_PROMPT,
                        },
                    ],
                }],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = RATE_LIMIT_DELAY * (attempt + 2)
                print(f"  [vision] Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            print(f"  [vision] OpenAI API error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            return None

    return None


# ── Main API ─────────────────────────────────────────────────────────────

def describe_diagram(img: Image.Image, cache_key: str | None = None) -> str | None:
    """Generate a text description of a diagram using AI vision.

    Args:
        img: PIL Image of the diagram region.
        cache_key: Optional key to cache/deduplicate descriptions (e.g., page label).

    Returns:
        Markdown description string, or None if no API is available.
    """
    # Check cache
    if cache_key and cache_key in _described_cache:
        return _described_cache[cache_key]

    # Try Claude first, then GPT-4o
    description = _describe_with_claude(img)
    if description is None:
        description = _describe_with_openai(img)

    if description and cache_key:
        _described_cache[cache_key] = description

    return description


def is_vision_available() -> bool:
    """Check if any vision API is configured and available."""
    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        return True
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        return True
    return False


def load_description_cache(cache_file: Path) -> None:
    """Load previously generated descriptions from a JSON cache file."""
    global _described_cache
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            _described_cache = json.load(f)


def save_description_cache(cache_file: Path) -> None:
    """Save generated descriptions to a JSON cache file."""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(_described_cache, f, indent=2)
