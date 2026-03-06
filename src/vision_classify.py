"""Vision-based region classification (table vs. diagram) and table extraction helper."""

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any
from PIL import Image

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

MAX_IMAGE_DIM = 1200
RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 2
KEY_LOCATIONS = [
    Path(__file__).resolve().parent.parent / "build" / "API KEY.txt",
    Path(__file__).resolve().parent.parent / "build" / "ANTHROPIC_API_KEY.txt",
]

# Load API keys from known fallback files if env vars are missing
def _ensure_keys_loaded() -> None:
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        return
    for p in KEY_LOCATIONS:
        if p.exists():
            try:
                key = p.read_text(encoding="utf-8").strip()
                if key.startswith("sk-") and "ANTHROPIC_API_KEY" not in os.environ:
                    os.environ["ANTHROPIC_API_KEY"] = key
                if key.startswith("sk-") and "OPENAI_API_KEY" not in os.environ:
                    # do not auto-set OpenAI with Anthropic key
                    pass
            except OSError:
                continue

_ensure_keys_loaded()

CLASSIFY_PROMPT = """You are analyzing one boxed region from a scanned automotive service manual.

Decide if the region is a:
- "table" (grid with rows/columns of text)
- "diagram_with_key" (drawing plus a legend box listing numbered labels)
- "diagram" (drawing without a legend)
- "other" (anything else)

If it is a table: return rows as an array-of-arrays of strings in reading order. Include the header row if present. Keep bullets as literal characters like "•".
If it is a diagram_with_key: return legend as an array of strings like "1: compressor".
If it is a diagram: provide a 1-2 sentence description of what it shows.

Return ONLY JSON in this schema:
{
  "type": "table" | "diagram_with_key" | "diagram" | "other",
  "rows": [["..."]],        // required when type=table else []
  "legend": ["1: ..."],     // required when type=diagram_with_key else []
  "description": "..."      // required when type=diagram else ""
}
If uncertain, use type "other".
"""

TABLE_PROMPT = """You are transcribing a TABLE from a scanned automotive service manual image.

Return ONLY JSON with this exact schema:
{
  "rows": [
    ["col1_header", "col2_header", "col3_header"],
    ["row1col1", "row1col2", "row1col3"]
  ]
}

Rules:
- Preserve row order and column alignment.
- Keep bullet characters like "•" intact.
- Do not drop blank cells; use "" for empty.
- Do not add commentary or extra keys.
"""

def is_vision_classifier_available() -> bool:
    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        return True
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        return True
    return False


def classify_region(img: Image.Image, model: str | None = None) -> dict[str, Any] | None:
    """Classify a region and optionally return table rows/legend/description."""
    if not is_vision_classifier_available():
        return None

    b64, media_type = _prepare_image(img)
    payload = _build_payload(b64, media_type)

    text = _call_claude(payload, model=model) or _call_openai(payload)
    if not text:
        return None

    parsed = _parse_json_response(text)
    return parsed


def extract_table_rows(img: Image.Image, model: str | None = None) -> list[list[str]] | None:
    """Use vision model to transcribe a table region into rows (Claude first, OpenAI fallback)."""
    if not is_vision_classifier_available():
        return None

    b64, media_type = _prepare_image(img)
    payload = _build_payload(b64, media_type, text=TABLE_PROMPT)

    text = _call_claude(payload, model=model) or _call_openai(payload)
    if not text:
        return None

    parsed = _parse_json_response(text, expect_rows_only=True)
    if parsed and parsed.get("rows"):
        return parsed["rows"]
    return None


def _prepare_image(img: Image.Image) -> tuple[str, str]:
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def _rate_limit():
    global _LAST_CALL
    now = time.time()
    elapsed = now - _LAST_CALL
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _LAST_CALL = time.time()


_LAST_CALL = 0.0


def _build_payload(b64: str, media_type: str, text: str = CLASSIFY_PROMPT) -> dict:
    return {
        "image": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
        "text": text,
    }


def _call_claude(payload: dict, model: str | None = None) -> str | None:
    if not (ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY")):
        return None
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    chosen_model = model or "claude-sonnet-4-6"

    for attempt in range(MAX_RETRIES):
        _rate_limit()
        try:
            resp = client.messages.create(
                model=chosen_model,
                max_tokens=700,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": payload["image"]},
                        {"type": "text", "text": payload["text"]},
                    ],
                }],
            )
            return resp.content[0].text if resp and resp.content else None
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(RATE_LIMIT_DELAY * (attempt + 2))
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5)
                continue
            return None
    return None


def _call_openai(payload: dict) -> str | None:
    if not (OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY")):
        return None
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for attempt in range(MAX_RETRIES):
        _rate_limit()
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=700,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{payload['image']['media_type']};base64,{payload['image']['data']}",
                            },
                        },
                        {"type": "text", "text": payload["text"]},
                    ],
                }],
            )
            return resp.choices[0].message.content if resp and resp.choices else None
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(RATE_LIMIT_DELAY * (attempt + 2))
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5)
                continue
            return None
    return None


def _parse_json_response(text: str, expect_rows_only: bool = False) -> dict[str, Any] | None:
    """Extract first JSON object from text and coerce fields."""
    if not text:
        return None

    # Strip code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # After stripping, remove possible language tags
        if text.startswith("json"):
            text = text[4:].strip()

    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    rows = data.get("rows") or []
    clean_rows: list[list[str]] = []
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, list):
                clean_rows.append([str(c) if c is not None else "" for c in r])

    if expect_rows_only:
        return {"rows": clean_rows}

    dtype = data.get("type")
    if dtype not in ("table", "diagram_with_key", "diagram", "other"):
        return None

    legend = data.get("legend") or []
    desc = data.get("description") or ""

    clean_legend = [str(l) for l in legend] if isinstance(legend, list) else []

    return {
        "type": dtype,
        "rows": clean_rows,
        "legend": clean_legend,
        "description": str(desc) if desc else "",
    }
