"""Gemini API client with function-calling support."""
import os
import requests


# ---------------------------------------------------------------------------
# Tool schema translation: Claude → Gemini
# ---------------------------------------------------------------------------

def _claude_tool_to_gemini(tool: dict) -> dict:
    """Convert a Claude tool definition to a Gemini functionDeclaration."""
    schema = dict(tool.get("input_schema", {}))
    # Gemini does not support additionalProperties in function schemas
    schema.pop("additionalProperties", None)
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": schema,
    }


def claude_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """Convert a list of Claude tool definitions to Gemini tools format."""
    if not tools:
        return []
    return [{"functionDeclarations": [_claude_tool_to_gemini(t) for t in tools]}]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_gemini_response(response_json: dict) -> dict:
    """Parse a Gemini generateContent response into a normalized dict.

    Returns:
        {
            "text": str,              # concatenated text parts
            "tool_calls": list[dict], # [{name, id, input}, ...]
            "stop_reason": str,       # "tool_use" if function calls present, else "end_turn"
            "raw_parts": list,        # original parts list for assistant_message reconstruction
        }
    """
    candidate = response_json.get("candidates", [{}])[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])

    text_parts = []
    tool_calls = []

    for i, part in enumerate(parts):
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "name": fc.get("name", ""),
                "id": f"gemini_tool_{i}",
                "input": fc.get("args", {}),
            })

    finish_reason = candidate.get("finishReason", "STOP")
    stop_reason = "tool_use" if tool_calls else (
        "end_turn" if finish_reason in ("STOP", "MAX_TOKENS") else finish_reason
    )

    return {
        "text": "".join(text_parts).strip(),
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
        "raw_parts": parts,
    }


# ---------------------------------------------------------------------------
# Message format translation
# ---------------------------------------------------------------------------

def _convert_messages_to_gemini(conv_messages: list[dict]) -> list[dict]:
    """Convert Claude-format messages to Gemini contents array."""
    contents = []
    for m in conv_messages:
        role = "user" if m["role"] == "user" else "model"
        content = m.get("content")
        parts = []

        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item_type == "image":
                        source = item.get("source", {})
                        parts.append({
                            "inline_data": {
                                "mime_type": source.get("media_type", "image/webp"),
                                "data": source.get("data", ""),
                            }
                        })
                    elif item_type == "tool_use":
                        parts.append({
                            "functionCall": {
                                "name": item.get("name", ""),
                                "args": item.get("input", {}),
                            }
                        })
                    elif item_type == "tool_result":
                        # Prefer tool_name (the actual function name) over
                        # tool_use_id (a synthetic call ID) for Gemini's
                        # functionResponse which requires the function name.
                        name = item.get("tool_name") or item.get("tool_use_id", "")
                        parts.append({
                            "functionResponse": {
                                "name": name,
                                "response": {"content": item.get("content", "")},
                            }
                        })
        else:
            parts.append({"text": str(content or "")})

        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def build_gemini_assistant_parts(text: str, tool_calls: list[dict]) -> list[dict]:
    """Build Gemini-format parts for reconstructing an assistant turn in the conversation."""
    parts = []
    if text:
        parts.append({"text": text})
    for tc in tool_calls:
        parts.append({
            "functionCall": {
                "name": tc.get("name", ""),
                "args": tc.get("input", {}),
            }
        })
    return parts


def build_tool_result_parts(tool_results: list[dict]) -> list[dict]:
    """Convert tool_result dicts to Gemini functionResponse parts."""
    parts = []
    for result in tool_results:
        # Gemini needs the function name, not the tool_use_id
        name = result.get("tool_name") or result.get("tool_use_id", "")
        parts.append({
            "functionResponse": {
                "name": name,
                "response": {"content": result.get("content", "")},
            }
        })
    return parts


# ---------------------------------------------------------------------------
# Main API call
# ---------------------------------------------------------------------------

def call_gemini(messages, api_key=None, model=None, tools=None):
    """Call Google Gemini API with optional function-calling support.

    Parameters
    ----------
    messages : list
        Single-element list: [{"system": str, "messages": [...]}]
    api_key : str, optional
    model : str, optional
    tools : list[dict], optional
        Claude-format tool definitions. Converted to Gemini format internally.

    Returns
    -------
    dict
        Parsed response with keys: text, tool_calls, stop_reason, raw_parts
        (When tools is None or empty, returns legacy text-only format for backwards compat)
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
    contents = _convert_messages_to_gemini(conv_messages)

    data = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }

    if system_prompt:
        data["system_instruction"] = {"parts": [{"text": system_prompt}]}

    # Add tool definitions if provided
    gemini_tools = claude_tools_to_gemini(tools) if tools else []
    if gemini_tools:
        data["tools"] = gemini_tools

    resp = requests.post(
        url, headers={"Content-Type": "application/json"},
        params={"key": api_key}, json=data, timeout=120,
    )
    if not resp.ok:
        print(f"GEMINI ERROR {resp.status_code}: {resp.text[:2000]}")
    resp.raise_for_status()
    out = resp.json()

    # If tools were provided, return the full parsed response
    if tools:
        return parse_gemini_response(out)

    # Legacy text-only return for backwards compatibility
    return out["candidates"][0]["content"]["parts"][0]["text"]
