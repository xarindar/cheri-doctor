import requests
import os

def call_gemini(messages, api_key=None, model=None):
    """Call Google Gemini API and return the response text.

    Parameters
    ----------
    messages : list
        Single-element list: [{"system": str, "messages": [...]}]
        Each message has {"role": "user"|"assistant", "content": str|list}.
        Content list items can be:
          - {"type": "text", "text": "..."}
          - {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    api_key : str, optional
    model : str, optional
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key not set")
    if not model:
        raise RuntimeError("Gemini model not set")
    model = model.removeprefix("models/")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    payload = messages[0]
    system_prompt = payload.get("system", "")
    conv_messages = payload.get("messages", [])

    # Build Gemini contents array from conversation messages
    contents = []
    for m in conv_messages:
        role = "user" if m["role"] == "user" else "model"
        content = m.get("content")
        parts = []

        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image":
                    source = item.get("source", {})
                    parts.append({
                        "inline_data": {
                            "mime_type": source.get("media_type", "image/webp"),
                            "data": source.get("data", ""),
                        }
                    })
        else:
            parts.append({"text": str(content)})

        if parts:
            contents.append({"role": role, "parts": parts})

    data = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }

    # Add system instruction if present
    if system_prompt:
        data["system_instruction"] = {"parts": [{"text": system_prompt}]}

    resp = requests.post(
        url, headers={"Content-Type": "application/json"},
        params={"key": api_key}, json=data, timeout=120,
    )
    if not resp.ok:
        print(f"GEMINI ERROR {resp.status_code}: {resp.text[:2000]}")
    resp.raise_for_status()
    out = resp.json()
    return out["candidates"][0]["content"]["parts"][0]["text"]
