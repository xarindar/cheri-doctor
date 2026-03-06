"""Vision-based full-page extraction using Claude Sonnet 3.5/3.7."""

import os
import json
import base64
import io
import time
from typing import Any, Optional
from PIL import Image
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Constants
MAX_IMAGE_DIM = 1600  # Higher for full-page extraction
RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 3

FULL_PAGE_PROMPT = """You are extracting content from a page of an automotive service manual.

Return a JSON object with this exact structure:
{
 "page": <int>,
 "section_id": <string>, // e.g. "3D", "6A1", "1Q"
 "section_title": <string>, // e.g. "REAR SUSPENSION"
 "page_label": <string>, // e.g. "3D-12" from page header
 "reading_order": ["block_0", "block_1", ...],
 "blocks": [
 {
 "block_id": "block_0",
 "type": "procedure"|"table"|"figure"|"caution"|"warning"|"notice"|"important"|"header"|"note"|"paragraph",
 "title": <string or null>,
 "procedure_type": "removal"|"installation"|"inspection"|"adjustment"|null,
 "text": <string or null>, // for paragraphs, cautions, etc.
 "steps": [<string>, ...], // for procedures only
 "rows": [[<string>, ...], ...], // for tables only
 "caption": <string or null>,// for figures only
 "legend": { // for figures only
 "1": <string>,
 "2": <string>
 },
 "associated_figure_ids": [<string>, ...],
 "continues_from_previous_page": <bool>,
 "continues_to_next_page": <bool>
 }
 ]
}

Rules:
- Never merge two separate procedures into one block.
- Never split one procedure across two blocks if it's fully on this page.
- Preserve exact step numbering from the source.
- If a step number restarts at 1 mid-page, that is a new procedure block.
- CAUTION/WARNING/NOTICE/Important blocks are always their own block type.
- Associate each figure with the procedure block it illustrates.
- If a sentence ends without a period at the bottom of the page, set continues_to_next_page: true.
- Identify procedure_type from context headers (Remove or Disconnect / Install or Connect / Inspect / Adjust).
- Capture the section_id (e.g., 3D) and page_label (e.g., 3D-12) from the top/outer corners of the page.
"""

def is_vision_available() -> bool:
    return ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))

def extract_full_page_vision(img: Image.Image, page_num: int, model: str = "claude-sonnet-4-6") -> Optional[dict[str, Any]]:
    """Send full page image to Claude Vision for structured extraction."""
    if not is_vision_available():
        print("  [vision] Anthropic API key or package missing.")
        return None

    # Prepare image
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    prompt = FULL_PAGE_PROMPT.replace("<int>", str(page_num))

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            }
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = resp.content[0].text
            return _parse_json(text)
        except Exception as e:
            print(f"  [vision] Attempt {attempt+1} failed: {e}")
            if "429" in str(e):
                time.sleep(RATE_LIMIT_DELAY * (attempt + 2))
                continue
            time.sleep(2)
    
    return None

def _parse_json(text: str) -> Optional[dict[str, Any]]:
    """Find and parse JSON in LLM response."""
    try:
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        
        json_str = text[start:end+1]
        return json.loads(json_str)
    except Exception as e:
        print(f"  [vision] Failed to parse JSON: {e}")
        return None
