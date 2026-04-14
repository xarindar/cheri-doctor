"""Chat backend web server.

FastAPI app serving:
  GET  /              — Chat UI (HTML)
  POST /api/chat      — Chat endpoint
  GET  /api/figures/{figure_id}  — Serve figure images

Usage:
  python -m tools.chat_backend.serve
  uvicorn tools.chat_backend.serve:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import inspect
import sys
import json
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils import load_config
from src.chat import chat, load_index
from src.utils import load_jsonl

app = FastAPI(title="Metro Manual Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass(frozen=True)
class VehicleProfile:
    vehicle_id: str
    label: str
    config_path: Path
    prompt_path: Path
    index_dir: Path
    figure_sources: tuple[Path, ...]
    chunk_sources: tuple[Path, ...]
    page_image_dir: Path | None = None
    shopping_vehicle: dict[str, object] | None = None


@dataclass
class VehicleRuntime:
    profile: VehicleProfile
    config: dict
    index: object | None = None
    fig_lookup: dict[str, dict] = field(default_factory=dict)
    chunk_fig_map: dict[str, str] = field(default_factory=dict)
    chunk_text_lookup: dict[str, dict] = field(default_factory=dict)


DEFAULT_VEHICLE_ID = "metro"
VEHICLE_ALIASES = {
    "cheri": "metro",
    "geo": "metro",
    "geo-metro": "metro",
    "geo_metro": "metro",
    "cruze": "cruze",
    "chevy-cruze": "cruze",
    "chevy_cruze": "cruze",
    "chevrolet-cruze": "cruze",
    "chevrolet_cruze": "cruze",
}
VEHICLE_PROFILES: dict[str, VehicleProfile] = {
    "metro": VehicleProfile(
        vehicle_id="metro",
        label="1990 Geo Metro",
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
        prompt_path=PROJECT_ROOT / "configs" / "chat_system_prompt.txt",
        index_dir=PROJECT_ROOT / "tools" / "rag_index",
        figure_sources=(
            PROJECT_ROOT / "build" / "figures.jsonl",
            PROJECT_ROOT / "build_supplement" / "figures.jsonl",
        ),
        chunk_sources=(
            PROJECT_ROOT / "build" / "chunks.jsonl",
            PROJECT_ROOT / "build_supplement" / "chunks.jsonl",
        ),
        page_image_dir=PROJECT_ROOT / "build" / "pages",
        shopping_vehicle={"year": 1990, "make": "Geo", "model": "Metro", "engine": "993cc L3"},
    ),
    "cruze": VehicleProfile(
        vehicle_id="cruze",
        label="Chevrolet Cruze",
        config_path=PROJECT_ROOT / "configs" / "cruze.yaml",
        prompt_path=PROJECT_ROOT / "configs" / "cruze_chat_system_prompt.txt",
        index_dir=PROJECT_ROOT / "tools" / "cruze_index",
        figure_sources=(PROJECT_ROOT / "build_cruze" / "figures.jsonl",),
        chunk_sources=(PROJECT_ROOT / "build_cruze" / "chunks.jsonl",),
        page_image_dir=PROJECT_ROOT / "build_cruze" / "pages",
        shopping_vehicle={"year": 2014, "make": "Chevrolet", "model": "Cruze"},
    ),
}
_vehicle_runtimes: dict[str, VehicleRuntime] = {}


def _project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _normalize_vehicle_id(vehicle: str | None) -> str:
    key = (vehicle or DEFAULT_VEHICLE_ID).strip().lower()
    key = VEHICLE_ALIASES.get(key, key)
    if key not in VEHICLE_PROFILES:
        raise HTTPException(400, f"Unknown vehicle: {vehicle}")
    return key


def _index_ready(index_dir: Path) -> bool:
    required = ("bm25_index.pkl", "embeddings.npy", "chunk_ids.json", "chunk_lookup.json")
    return all((index_dir / name).exists() for name in required)


def _vehicle_config(profile: VehicleProfile) -> dict:
    config = load_config(profile.config_path)
    config = dict(config)
    config["chat"] = dict(config.get("chat", {}))
    config["chat"]["system_prompt"] = _project_relative(profile.prompt_path)
    config["chat"]["index_dir"] = _project_relative(profile.index_dir)
    config["_vehicle_id"] = profile.vehicle_id
    config["_vehicle_label"] = profile.label
    config["_system_prompt_path"] = _project_relative(profile.prompt_path)
    config["_index_dir"] = _project_relative(profile.index_dir)
    config["_chunk_lookup_path"] = _project_relative(profile.index_dir / "chunk_lookup.json")
    return config


def _load_vehicle_runtime(vehicle: str | None, *, require_index: bool = False) -> VehicleRuntime:
    vehicle_id = _normalize_vehicle_id(vehicle)
    cached = _vehicle_runtimes.get(vehicle_id)
    if cached is not None:
        if require_index and cached.index is None:
            if not _index_ready(cached.profile.index_dir):
                raise HTTPException(503, f"{cached.profile.label} index is not available yet")
            cached.index = load_index(cached.config)
        return cached

    profile = VEHICLE_PROFILES[vehicle_id]
    runtime = VehicleRuntime(profile=profile, config=_vehicle_config(profile))

    if _index_ready(profile.index_dir):
        runtime.index = load_index(runtime.config)
    elif require_index:
        raise HTTPException(503, f"{profile.label} index is not available yet")

    for fig_path in profile.figure_sources:
        if not fig_path.exists():
            continue
        for fig in load_jsonl(fig_path):
            if fig.get("figure_id"):
                runtime.fig_lookup[fig["figure_id"]] = fig

    import re as _startup_re

    for chunk_path in profile.chunk_sources:
        if not chunk_path.exists():
            continue
        for chunk in load_jsonl(chunk_path):
            cid = chunk.get("chunk_id")
            if cid:
                runtime.chunk_text_lookup[cid] = {
                    "type": chunk.get("type", "text"),
                    "text": chunk.get("text", ""),
                    "section_path": chunk.get("section_path", ""),
                }
            if chunk.get("type") == "figure" and chunk.get("figure_refs"):
                asset_id = chunk["figure_refs"][0]
                runtime.chunk_fig_map[chunk["chunk_id"]] = asset_id
                source_label = chunk.get("source_label", "")
                if source_label:
                    runtime.chunk_fig_map[source_label] = asset_id
                cap = _startup_re.match(r"(?:Figure|Fig\.?)\s+(\S+)", chunk.get("text", ""))
                if cap:
                    runtime.chunk_fig_map[cap.group(1)] = asset_id

    _vehicle_runtimes[vehicle_id] = runtime
    return runtime

# ── Authentication ────────────────────────────────────────────────────────
_PASSWORD_HASH = hashlib.sha256(b"maytoe").hexdigest()

def _load_auth_secret() -> str:
    """Load a stable auth secret.

    Priority:
    1) AUTH_SECRET env var (explicit deployment configuration)
    2) persisted secret file in data/ (survives process restarts)
    3) generate + persist a new secret
    """
    env_secret = os.environ.get("AUTH_SECRET")
    if env_secret:
        return env_secret
    secret_file = PROJECT_ROOT / "data" / "auth_secret.txt"
    try:
        if secret_file.exists():
            val = secret_file.read_text(encoding="utf-8").strip()
            if val:
                return val
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        val = secrets.token_hex(32)
        secret_file.write_text(val, encoding="utf-8")
        return val
    except Exception:
        # Last-resort fallback keeps server running even if disk write fails.
        return secrets.token_hex(32)

_AUTH_SECRET = _load_auth_secret()
_COOKIE_NAME = "cheri_session"
_SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days

_PUBLIC_PATHS = {"/login", "/assets/logowhite.svg", "/favicon.ico", "/ebay/notifications"}

def _sign_token(timestamp: int) -> str:
    msg = f"{timestamp}".encode()
    sig = hmac.new(_AUTH_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return f"{timestamp}.{sig}"

def _verify_token(token: str) -> bool:
    try:
        ts_str, sig = token.split(".", 1)
        ts = int(ts_str)
        if time.time() - ts > _SESSION_MAX_AGE:
            return False
        expected = hmac.new(_AUTH_SECRET.encode(), ts_str.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_PATHS:
            return await call_next(request)
        token = request.cookies.get(_COOKIE_NAME)
        if not token or not _verify_token(token):
            if path.startswith("/api/"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return RedirectResponse("/login")
        return await call_next(request)

app.add_middleware(AuthMiddleware)

_LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cheri Doctor — Login</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; }
    .login-card { background: #222; border-radius: 14px; padding: 40px 36px; width: 340px;
                  box-shadow: 0 8px 32px rgba(0,0,0,0.4); text-align: center; }
    .login-card img { width: 80px; margin-bottom: 16px; }
    .login-card h1 { font-size: 1.2rem; margin-bottom: 24px; font-weight: 600; }
    .login-card input { width: 100%; padding: 12px 14px; border-radius: 8px; border: 1px solid #444;
                        background: #2a2a3e; color: #e0e0e0; font-size: 0.95rem; outline: none; }
    .login-card input:focus { border-color: #3b82f6; }
    .login-card button { width: 100%; margin-top: 14px; padding: 12px; border: none; border-radius: 8px;
                         background: #3b82f6; color: white; font-size: 0.95rem; font-weight: 600;
                         cursor: pointer; transition: background 0.15s; }
    .login-card button:hover { background: #2563eb; }
    .error { color: #ef4444; font-size: 0.82rem; margin-top: 10px; min-height: 1.2em; }
  </style>
</head>
<body>
  <div class="login-card">
    <img src="/assets/logowhite.svg" alt="Cheri Doctor">
    <h1>Cheri Doctor</h1>
    <form id="login-form">
      <input type="password" id="pw" placeholder="Password" autocomplete="current-password" autofocus>
      <button type="submit">Sign In</button>
      <div class="error" id="err"></div>
    </form>
  </div>
  <script>
    document.getElementById('login-form').onsubmit = async (e) => {
      e.preventDefault();
      const pw = document.getElementById('pw').value;
      const res = await fetch('/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({password: pw}),
      });
      if (res.ok) { window.location.href = '/'; }
      else { document.getElementById('err').textContent = 'Incorrect password'; }
    };
  </script>
</body>
</html>
"""

class LoginRequest(BaseModel):
    password: str

# ── eBay Marketplace Account Deletion compliance endpoint ──────────────────
# Required by eBay to obtain production API keys.
# Docs: https://developer.ebay.com/marketplace-account-deletion
#
# Add to .env:
#   EBAY_VERIFICATION_TOKEN=<secret you create, paste into eBay dev console>
#   EBAY_ENDPOINT_URL=https://cheridoc.xarindar.com/ebay/notifications
#
_EBAY_VERIFICATION_TOKEN = os.environ.get("EBAY_VERIFICATION_TOKEN", "")
_EBAY_ENDPOINT_URL       = os.environ.get("EBAY_ENDPOINT_URL", "https://cheridoc.xarindar.com/ebay/notifications")

@app.get("/ebay/notifications")
async def ebay_challenge(challenge_code: str = ""):
    """eBay ownership verification — responds with sha256(challengeCode + verificationToken + endpointUrl)."""
    if not challenge_code:
        return JSONResponse({"error": "missing challenge_code"}, status_code=400)
    if not _EBAY_VERIFICATION_TOKEN:
        return JSONResponse({"error": "EBAY_VERIFICATION_TOKEN not set in .env"}, status_code=500)
    import hashlib as _hl
    digest = _hl.sha256((challenge_code + _EBAY_VERIFICATION_TOKEN + _EBAY_ENDPOINT_URL).encode()).hexdigest()
    return JSONResponse({"challengeResponse": digest})

@app.post("/ebay/notifications")
async def ebay_notification(request: Request):
    """Acknowledge eBay marketplace account deletion notifications."""
    body = await request.body()
    print(f"[eBay notification] {body.decode('utf-8', errors='replace')[:500]}")
    return Response(status_code=200)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(_LOGIN_HTML)

@app.post("/login")
async def login(req: LoginRequest):
    if hashlib.sha256(req.password.encode()).hexdigest() == _PASSWORD_HASH:
        token = _sign_token(int(time.time()))
        resp = JSONResponse({"ok": True})
        resp.set_cookie(_COOKIE_NAME, token, max_age=_SESSION_MAX_AGE, httponly=True, samesite="lax")
        return resp
    return JSONResponse({"error": "invalid"}, status_code=401)

# ── SQLite storage ────────────────────────────────────────────────────────
DB_PATH = PROJECT_ROOT / "data" / "cheri.db"

def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id      TEXT PRIMARY KEY,
                created INTEGER NOT NULL,
                updated INTEGER NOT NULL,
                data    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                id      TEXT PRIMARY KEY,
                created INTEGER NOT NULL,
                data    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id         TEXT PRIMARY KEY,
                project_id TEXT,
                session_id TEXT,
                created    INTEGER NOT NULL,
                updated    INTEGER NOT NULL,
                title      TEXT NOT NULL DEFAULT '',
                content    TEXT NOT NULL DEFAULT '',
                tags       TEXT NOT NULL DEFAULT '[]',
                source     TEXT NOT NULL DEFAULT 'user'
            );
        """)

_init_db()

# ── Session / Project API ─────────────────────────────────────────────────

@app.get("/api/sessions")
async def get_sessions():
    with _get_db() as conn:
        rows = conn.execute("SELECT data FROM sessions ORDER BY updated DESC").fetchall()
    result = {}
    for row in rows:
        sess = json.loads(row["data"])
        result[sess["id"]] = sess
    return JSONResponse(result)

@app.put("/api/sessions/{session_id}")
async def upsert_session(session_id: str, request: Request):
    data = await request.json()
    data["id"] = session_id
    created = data.get("created", int(time.time() * 1000))
    updated = data.get("updated", int(time.time() * 1000))
    with _get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, created, updated, data) VALUES (?, ?, ?, ?)",
            (session_id, created, updated, json.dumps(data))
        )
    return JSONResponse({"ok": True})

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with _get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return JSONResponse({"ok": True})

@app.get("/api/projects")
async def get_projects():
    with _get_db() as conn:
        rows = conn.execute("SELECT data FROM projects ORDER BY created DESC").fetchall()
    result = {}
    for row in rows:
        proj = json.loads(row["data"])
        result[proj["id"]] = proj
    return JSONResponse(result)

@app.put("/api/projects/{project_id}")
async def upsert_project(project_id: str, request: Request):
    data = await request.json()
    data["id"] = project_id
    created = data.get("created", int(time.time() * 1000))
    with _get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO projects (id, created, data) VALUES (?, ?, ?)",
            (project_id, created, json.dumps(data))
        )
    return JSONResponse({"ok": True})

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    with _get_db() as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return JSONResponse({"ok": True})

# ── Notes API ─────────────────────────────────────────────────────────────

@app.get("/api/notes")
async def get_notes(project_id: str | None = None, session_id: str | None = None):
    """List notes for the current project or chat scope."""
    return JSONResponse(_load_notes_for_scope(project_id=project_id, session_id=session_id)[0])

@app.post("/api/notes")
async def create_note(request: Request):
    """Create a new note."""
    data = await request.json()
    project_id = _clean_scope_id(data.get("project_id"))
    session_id = _clean_scope_id(data.get("session_id"))
    if not project_id and not session_id:
        raise HTTPException(400, "Notes must belong to a chat or project.")
    note_id = data.get("id") or f"note_{int(time.time()*1000)}_{secrets.token_hex(4)}"
    now = int(time.time() * 1000)
    with _get_db() as conn:
        conn.execute(
            "INSERT INTO notes (id, project_id, session_id, created, updated, title, content, tags, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                project_id,
                session_id,
                now,
                now,
                data.get("title", ""),
                data.get("content", ""),
                json.dumps(data.get("tags", [])),
                data.get("source", "user"),
            )
        )
    return JSONResponse({"ok": True, "id": note_id})

@app.put("/api/notes/{note_id}")
async def update_note(note_id: str, request: Request):
    """Update an existing note."""
    data = await request.json()
    now = int(time.time() * 1000)
    with _get_db() as conn:
        conn.execute(
            "UPDATE notes SET title = ?, content = ?, tags = ?, updated = ? WHERE id = ?",
            (
                data.get("title", ""),
                data.get("content", ""),
                json.dumps(data.get("tags", [])),
                now,
                note_id,
            )
        )
    return JSONResponse({"ok": True})

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    with _get_db() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return JSONResponse({"ok": True})


_NOTES_CONTEXT_LIMIT = 4
_NOTES_CONTEXT_CHARS = 220
_NOTES_TOOL_LIMIT = 6
_NOTES_TOOL_CHARS = 320


def _row_to_note(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "session_id": row["session_id"],
        "created": row["created"],
        "updated": row["updated"],
        "title": row["title"],
        "content": row["content"],
        "tags": json.loads(row["tags"]),
        "source": row["source"],
    }


def _clean_scope_id(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def _resolve_notes_scope(
    project_id: str | None,
    session_id: str | None,
    requested_scope: str = "auto",
) -> tuple[str | None, str]:
    project_id = _clean_scope_id(project_id)
    session_id = _clean_scope_id(session_id)
    scope = (requested_scope or "auto").strip().lower()

    if scope == "project":
        return project_id, "project"
    if scope == "chat":
        return session_id, "chat"
    if project_id:
        return project_id, "project"
    if session_id:
        return session_id, "chat"
    return None, "none"


def _load_notes_for_scope(
    project_id: str | None = None,
    session_id: str | None = None,
    requested_scope: str = "auto",
) -> tuple[list[dict], str]:
    scope_id, scope_kind = _resolve_notes_scope(project_id, session_id, requested_scope)
    if not scope_id or scope_kind == "none":
        return [], scope_kind

    with _get_db() as conn:
        if scope_kind == "project":
            rows = conn.execute(
                "SELECT * FROM notes WHERE project_id = ? ORDER BY updated DESC",
                (scope_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notes WHERE session_id = ? ORDER BY updated DESC",
                (scope_id,),
            ).fetchall()
    return [_row_to_note(row) for row in rows], scope_kind


def _note_scope_label(scope_kind: str) -> str:
    if scope_kind == "project":
        return "project notebook"
    if scope_kind == "chat":
        return "chat notebook"
    return "notebook"


def _truncate_note_text(text: str | None, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _note_timestamp_label(note: dict) -> str:
    raw = note.get("updated") or note.get("created")
    if not isinstance(raw, int) or raw <= 0:
        return "unknown date"
    return time.strftime("%Y-%m-%d", time.gmtime(raw / 1000))


def _format_note_entry(note: dict, content_limit: int) -> str:
    title = (note.get("title") or "Untitled").strip()
    content = _truncate_note_text(note.get("content"), content_limit)
    tags = [str(tag).strip() for tag in (note.get("tags") or []) if str(tag).strip()]
    tag_text = f" [tags: {', '.join(tags)}]" if tags else ""
    return f"- {title} ({_note_timestamp_label(note)}){tag_text}\n  {content}"


def _build_notes_context(project_id: str | None, session_id: str | None) -> str | None:
    notes, scope_kind = _load_notes_for_scope(project_id=project_id, session_id=session_id)
    if not notes:
        return None

    scope_label = _note_scope_label(scope_kind)
    lines = [f"Saved notes from the current {scope_label}:"]
    for note in notes[:_NOTES_CONTEXT_LIMIT]:
        lines.append(_format_note_entry(note, _NOTES_CONTEXT_CHARS))
    remaining = len(notes) - _NOTES_CONTEXT_LIMIT
    if remaining > 0:
        lines.append(f"...and {remaining} more saved note(s) in this {scope_label}.")
    return "\n".join(lines)


def _filter_notes(notes: list[dict], keywords: str | None, tags: list[str] | None) -> list[dict]:
    tokens = re.findall(r"[a-z0-9]+", (keywords or "").lower())
    required_tags = {str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()}

    filtered: list[dict] = []
    for note in notes:
        note_tags = {str(tag).strip().lower() for tag in (note.get("tags") or []) if str(tag).strip()}
        haystack = " ".join(
            [
                str(note.get("title") or ""),
                str(note.get("content") or ""),
                " ".join(note_tags),
            ]
        ).lower()
        if tokens and not all(token in haystack for token in tokens):
            continue
        if required_tags and not required_tags.issubset(note_tags):
            continue
        filtered.append(note)
    return filtered


def _retrieve_notes_result(
    project_id: str | None,
    session_id: str | None,
    tool_input: dict,
) -> tuple[str, bool]:
    requested_scope = (tool_input.get("scope") or "auto").strip().lower()
    notes, scope_kind = _load_notes_for_scope(
        project_id=project_id,
        session_id=session_id,
        requested_scope=requested_scope,
    )
    scope_label = _note_scope_label(scope_kind)

    if scope_kind == "none":
        return "No chat or project notebook is active in this request.", True
    if requested_scope == "project" and not _clean_scope_id(project_id):
        return "No project notebook is available in this chat.", True
    if requested_scope == "chat" and not _clean_scope_id(session_id):
        return "No chat notebook is available in this request.", True
    if not notes:
        return f"No saved notes are available in the current {scope_label}.", False

    limit = tool_input.get("limit")
    if not isinstance(limit, int):
        limit = _NOTES_TOOL_LIMIT
    limit = max(1, min(limit, _NOTES_TOOL_LIMIT))

    filtered = _filter_notes(
        notes,
        keywords=tool_input.get("keywords"),
        tags=tool_input.get("tags"),
    )
    if not filtered:
        return f"No saved notes matched that search in the current {scope_label}.", False

    lines = [f"Saved notes from the current {scope_label}:"]
    for note in filtered[:limit]:
        lines.append(_format_note_entry(note, _NOTES_TOOL_CHARS))
    remaining = len(filtered) - limit
    if remaining > 0:
        lines.append(f"...and {remaining} more matching note(s).")
    return "\n".join(lines), False

# ── Serve static resources ────────────────────────────────────────────────

@app.get("/static/style.css")
async def serve_css():
    css_path = PROJECT_ROOT / "frontend" / "style.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    return HTMLResponse("/* CSS file not found */", status_code=404)

@app.get("/static/app.js")
async def serve_js():
    js_path = PROJECT_ROOT / "frontend" / "app.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    return HTMLResponse("// JS file not found", status_code=404)

@app.get("/assets/{file_path:path}")
async def serve_assets(file_path: str):
    import re
    if not re.match(r'^[a-zA-Z0-9_\-./]+$', file_path):
        return HTMLResponse("Invalid path", status_code=400)
    asset = PROJECT_ROOT / "frontend" / "assets" / file_path
    resolved = asset.resolve()
    allowed = (PROJECT_ROOT / "frontend" / "assets").resolve()
    if not str(resolved).startswith(str(allowed)) or not resolved.is_file():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(str(resolved))

# ── Vehicle settings ─────────────────────────────────────────────────────
SETTINGS_FILE = PROJECT_ROOT / "configs" / "vehicle_settings.json"
SERVICE_HISTORY_CSV = PROJECT_ROOT / "data" / "CheriServiceHistory.csv"


def _format_service_history_csv() -> str:
    """Format the full receipt-based service history CSV for the system prompt."""
    if not SERVICE_HISTORY_CSV.exists():
        return ""
    import csv
    import re as _re2

    lines = [
        "**Cheri's Full Service History** (from previous owner receipts — use to estimate "
        "component ages and weight diagnostic likelihood; recently replaced parts are less "
        "likely to be the failure; old/unknown-history parts are higher suspects):",
    ]
    with SERVICE_HISTORY_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date  = (row.get("Date") or "").strip()
            odo   = (row.get("Odometer") or "").strip()
            cat   = (row.get("Service Category") or "").strip()
            desc  = (row.get("Description") or "").strip()
            notes = (row.get("Notes / Source (PDF Page)") or "").strip()

            # Build the date/mileage prefix
            prefix_parts = []
            if date:
                prefix_parts.append(date)
            odo_clean = odo.replace(",", "").strip()
            if odo_clean and odo_clean.lower() not in ("unknown", ""):
                try:
                    prefix_parts.append(f"{int(odo_clean):,} mi")
                except ValueError:
                    pass

            prefix = " | ".join(prefix_parts) if prefix_parts else "Unknown date"

            # Truncate description to keep the prompt compact
            short_desc = desc[:130] + ("…" if len(desc) > 130 else "")

            line = f"  - {prefix} — **{cat}**: {short_desc}"

            # Append tech notes that are diagnostically relevant (strip trailing page refs)
            note_clean = _re2.split(r'\.\s*Pg\s*\d', notes)[0].strip()
            if any(kw in note_clean.lower() for kw in ("⚠️", "tech note", "oil leak", "wear bar", "split", "play", "recommend")):
                line += f"  ⚠️ {note_clean}"

            lines.append(line)

    return "\n".join(lines)

_DEFAULT_SETTINGS = {
    "vin": None,
    "maintenance_schedule": "I",
    "oil_viscosity": "5W-30",
    "odometer": None,
    "tire_size": None,
    "driving_profile": "severe",
    "service_history": {
        "oil_change":    {"mileage": None},
        "spark_plugs":   {"mileage": None},
        "trans_fluid":   {"mileage": None},
        "coolant_flush": {"mileage": None},
        "brake_fluid":   {"mileage": None},
    },
    "modifications": None,
}

def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(_DEFAULT_SETTINGS)

def _save_settings(data: dict):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

# Trim-specific feature overrides — these lock in which manual variant applies
# for systems where the manual covers multiple configurations.
TRIM_RULES: dict[str, list[str]] = {
    "LSi": [
        "**Rule — headlights**: Cheri is the **LSi** trim. LSi uses composite (aero-style) "
        "headlights with replaceable **9004 halogen bulbs** — NOT sealed beams. "
        "Sealed beam headlight procedures and specs in the manual apply to the base "
        "trim only. Always answer headlight questions for the composite/bulb configuration.",
    ],
}

SCHED_LABELS = {
    "I":  "Schedule I — Severe Service",
    "II": "Schedule II — Normal Service",
}
SCHED_INTERVALS = {
    # key: (Schedule I miles, Schedule II miles)
    "oil_change":    (3_000,  7_500),
    "spark_plugs":   (15_000, 30_000),
    "trans_fluid":   (15_000, 30_000),
    "coolant_flush": (30_000, 30_000),
    "brake_fluid":   (24_000, 24_000),
}
SVC_NAMES = {
    "oil_change":    "Oil change",
    "spark_plugs":   "Spark plugs",
    "trans_fluid":   "Transmission fluid",
    "coolant_flush": "Coolant flush",
    "brake_fluid":   "Brake fluid",
}
VISC_LABELS = {
    "5W-30":  "SAE 5W-30",
    "10W-30": "SAE 10W-30",
    "20W-50": "SAE 20W-50",
    "chart":  "per temperature chart (show all options when asked)",
}
PROFILE_LABELS = {
    "severe": "Severe — mostly short city trips, stop-and-go, dusty/dirty",
    "mixed":  "Mixed — city and highway driving",
    "normal": "Normal — mostly highway, mild conditions",
}


def _build_settings_context(settings: dict) -> str:
    """Build a system-prompt block that locks in the owner's confirmed vehicle settings."""
    vin     = settings.get("vin") or ""
    trim    = settings.get("trim") or ""
    sched   = settings.get("maintenance_schedule", "I")
    visc    = settings.get("oil_viscosity", "5W-30")
    odo     = settings.get("odometer")
    tire    = settings.get("tire_size") or ""
    profile = settings.get("driving_profile", "severe")
    svc     = settings.get("service_history") or {}
    mods    = settings.get("modifications") or ""

    lines = [
        "## VEHICLE CONFIGURATION (OWNER-CONFIRMED SETTINGS)",
        "",
        "These values have been confirmed by the owner. Apply them to every answer "
        "without re-listing alternatives unless the owner explicitly asks.",
        "",
    ]

    # Identity
    if vin:
        lines.append(f"**VIN**: {vin}")
    if trim:
        lines.append(f"**Trim Level**: {trim}")
    if odo:
        lines.append(f"**Current Odometer**: ~{int(odo):,} miles")
    if tire:
        lines.append(f"**Tire Size**: {tire}")
    lines.append(f"**Maintenance Schedule**: {SCHED_LABELS.get(sched, sched)}")
    lines.append(f"**Oil Viscosity**: {VISC_LABELS.get(visc, visc)}")
    lines.append(f"**Driving Profile**: {PROFILE_LABELS.get(profile, profile)}")

    # Service history with overdue calculations
    sched_idx = 0 if sched == "I" else 1
    svc_lines = []
    for key, name in SVC_NAMES.items():
        entry    = svc.get(key) or {}
        last_mi  = entry.get("mileage")
        interval = SCHED_INTERVALS[key][sched_idx]
        if last_mi is not None and odo:
            next_due    = int(last_mi) + interval
            overdue_by  = int(odo) - next_due
            if overdue_by > 0:
                status = f"⚠️ OVERDUE by {overdue_by:,} mi (last at {int(last_mi):,} mi, due at {next_due:,} mi)"
            else:
                status = f"OK — last at {int(last_mi):,} mi, next due at {next_due:,} mi ({-overdue_by:,} mi remaining)"
        elif last_mi is not None:
            status = f"last done at {int(last_mi):,} mi"
        else:
            status = "unknown — treat as potentially overdue"
        svc_lines.append(f"  - {name}: {status}")

    if svc_lines:
        lines.append("")
        lines.append("**Service History** (use these to determine what is overdue):")
        lines.extend(svc_lines)

    # Modifications — array of {category, title, description}; legacy plain string also accepted
    mod_list = mods if isinstance(mods, list) else (
        [{"category": "note", "title": "General Note", "description": mods}] if mods else []
    )
    if mod_list:
        CAT_LABEL = {
            "retrofit":   "RETROFIT",
            "conversion": "CONVERSION",
            "repair":     "REPAIR",
            "note":       "NOTE",
        }
        CAT_CONTEXT = {
            "retrofit":   "Factory manual specs for this system no longer apply — use retrofit/updated specs instead.",
            "conversion": "This system has been converted; factory specs for it do not apply.",
            "repair":     "This component was serviced or replaced.",
            "note":       "",
        }
        lines.append("")
        lines.append("**Known Modifications & Notes** (adjust which specs apply accordingly):")
        for mod in mod_list:
            cat   = mod.get("category", "note")
            title = mod.get("title", "")
            desc  = mod.get("description", "")
            label = CAT_LABEL.get(cat, "NOTE")
            ctx   = CAT_CONTEXT.get(cat, "")
            entry = f"- [{label}] **{title}**: {desc}"
            if ctx:
                entry += f" — {ctx}"
            lines.append(entry)

    # Full receipt-based service history
    svc_history = _format_service_history_csv()
    if svc_history:
        lines.append("")
        lines.append(svc_history)

    # Behaviour rules
    lines.append("")
    if visc != "chart":
        lines.append(
            f"**Rule — oil viscosity**: Always recommend **{VISC_LABELS.get(visc, visc)}**. "
            "Do NOT list multiple grades or show the temperature chart unless explicitly asked."
        )
    lines.append(
        f"**Rule — maintenance intervals**: Use **{SCHED_LABELS.get(sched, sched)}** intervals only. "
        "Do NOT present both schedules side-by-side."
    )
    if tire:
        lines.append(
            f"**Rule — tire pressure**: Use specs for **{tire}** only. "
            "Do not list pressures for other sizes."
        )
    for rule in TRIM_RULES.get(trim, []):
        lines.append(rule)

    return "\n".join(lines)


# ── Load resources at startup ─────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    runtime = _load_vehicle_runtime(DEFAULT_VEHICLE_ID, require_index=True)
    chunk_total = len(runtime.index.chunk_ids) if runtime.index is not None else 0
    print(f"Default vehicle ready: {runtime.profile.label} ({chunk_total} chunks)")


# ── Web fetch / search / RockAuto utilities ───────────────────────────────
import re as _re
import requests as _requests
from bs4 import BeautifulSoup as _BS

_URL_RE    = _re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
_SEARCH_RE = _re.compile(
    r'\b(search\s+(for|online|the\s+web)?|look\s+up\s+online|find\s+online|google\s+(this|for)?)\b',
    _re.IGNORECASE,
)
_RA_SEARCH_RE = _re.compile(
    r'\b(?:rock\s*auto|roc\s*auto|rockauto|rocauto|amazon|auto\s*zone|o\'?\s*reill(?:y|ey))\b',
    _re.IGNORECASE,
)

# Map natural-language source mentions to ShopHop source keys
_SOURCE_NAMES = {
    "amazon":    "amazon",
    "ebay":      "ebay",
    "e-bay":     "ebay",
    "rockauto":  "rockauto",
    "rock auto": "rockauto",
    "rocauto":   "rockauto",
    "roc auto":  "rockauto",
    "rokauto":   "rockauto",
    "autozone":  "autozone",
    "auto zone": "autozone",
    "oreilly":   "oreilly",
    "o'reilly":  "oreilly",
    "oreily":    "oreilly",
    "o'reily":   "oreilly",
    "oriley":    "oreilly",
    "o'riley":   "oreilly",
}

def _extract_sources(query: str) -> list[str] | None:
    """Extract specific source names from the query.
    Returns a list of ShopHop source keys, or None to search all."""
    q = query.lower()
    found = []
    for name, key in _SOURCE_NAMES.items():
        if name in q and key not in found:
            found.append(key)
    return found if found else None


def _last_user_query_with_source(conversation: list[dict] | None) -> str:
    for msg in reversed(conversation or []):
        if msg.get("role") != "user":
            continue
        text = (msg.get("text") or "").strip()
        if text and _extract_sources(text):
            return text
    return ""


def _resolve_shop_sources(query: str, conversation: list[dict] | None) -> list[str] | None:
    """Resolve shopping sources for the current turn.

    Priority:
    1) Sources explicitly named in the current query.
    2) For contextual follow-ups, inherit the last user-specified source.
    3) Otherwise None (search all sources).
    """
    explicit = _extract_sources(query)
    if explicit:
        return explicit
    prior = _last_user_query_with_source(conversation)
    if prior:
        return _extract_sources(prior)
    return None


def _normalize_shop_sources(sources: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for source in sources or []:
        key = (source or "").strip().lower()
        if key in _SHOP_PROGRESS_SOURCE_LABELS and key not in normalized:
            normalized.append(key)
    return normalized


_GENERIC_SHOP_PART_HINTS = {
    "part",
    "parts",
    "item",
    "items",
    "product",
    "products",
    "auto part",
    "auto parts",
    "car part",
    "car parts",
}

_SHOP_WATCHDOG_DEFAULT_SOURCES = ["amazon", "ebay", "rockauto"]
_SHOP_REFINEMENT_KEYWORDS = [
    "cigar lighter assembly",
    "cigarette lighter assembly",
    "cigar lighter",
    "cigarette lighter",
    "usb charger",
    "usb charger socket",
    "power outlet",
    "socket",
    "flush mount",
    "flush-mount",
    "12v",
    "g202",
]


_CONVERSATIONAL_PREFIX_RE = _re.compile(
    r'^(?:yeah|yes|sure|ok|okay|no|hey|please|now|can you|could you|'
    r'go ahead(?:\s+and)?|i want you to|i need you to|'
    r'look(?:\s+(?:on|at|for|up))?|search(?:\s+(?:on|for))?|find(?:\s+(?:me|on))?|'
    r'check(?:\s+(?:on|for))?|try(?:\s+(?:to find|looking))?)\b[\s,:-]*',
    _re.IGNORECASE,
)


def _strip_conversational_prefix(text: str) -> str:
    cleaned = (text or "").strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _CONVERSATIONAL_PREFIX_RE.sub("", cleaned).strip()
        cleaned = _re.sub(r"^(?:for|to|me)\b[\s,:-]*", "", cleaned, flags=_re.IGNORECASE).strip()
    return cleaned


def _sanitize_shop_part_hint(value: str | None) -> str:
    hint = _strip_conversational_prefix((value or "").strip())
    if not hint:
        return ""
    normalized = _re.sub(r"[^a-z0-9]+", " ", hint.lower()).strip()
    if normalized in _GENERIC_SHOP_PART_HINTS:
        return ""
    return hint


def _last_assistant_text(conversation: list[dict] | None) -> str:
    return next(
        (m.get("text", "") for m in reversed(conversation or []) if m.get("role") == "assistant"),
        "",
    )


def _extract_offered_sources(text: str) -> list[str]:
    if not text or not _SHOP_OFFER_RE.search(text):
        return []
    return _normalize_shop_sources(_extract_sources(text))


def _shop_session_key(
    query: str,
    conversation: list[dict] | None,
    project_context: str | None = None,
) -> str:
    """Build a stable key for a chat's shopping state."""
    anchor = ""
    for msg in conversation or []:
        if msg.get("role") == "user":
            anchor = (msg.get("text") or "").strip()
            if anchor:
                break
    if not anchor:
        anchor = (query or "").strip()
    seed = json.dumps(
        {
            "anchor": anchor[:240].lower(),
            "project": (project_context or "")[:120],
        },
        sort_keys=True,
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def _get_shop_session(
    query: str,
    conversation: list[dict] | None,
    project_context: str | None = None,
) -> "ShopSession":
    key = _shop_session_key(query, conversation, project_context)
    session = _shop_sessions.get(key)
    if session is None:
        session = ShopSession()
        _shop_sessions[key] = session
    return session


def _is_affirmative_shop_followup(
    query: str,
    conversation: list[dict] | None,
    shop_session: "ShopSession",
) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    is_affirmative = bool(_AFFIRMATIVE_RE.search(q) or _AFFIRM_SHOP_FOLLOWUP_RE.search(q))
    if not is_affirmative:
        return False
    if shop_session.pending_plan:
        return True
    last_asst = _last_assistant_text(conversation)
    return bool(last_asst and _SHOP_OFFER_RE.search(last_asst))


def _extract_watchdog_search_phrase(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(?:search(?:ing)?(?:\s+(?:for|on))?|look(?:ing)?\s+for|targeted\s+search\s+for)\s+([^\n:!?]+)",
        r"(?:run|doing)\s+(?:a\s+)?(?:proper\s+)?(?:targeted\s+)?search\s+for\s+([^\n:!?]+)",
    ]
    candidate = ""
    for pat in patterns:
        matches = _re.findall(pat, text, flags=_re.IGNORECASE)
        if matches:
            candidate = matches[-1]
    candidate = (candidate or "").strip()
    if not candidate:
        return ""
    
    # Clean up greedy trailing text
    candidate = _re.split(r'\s+(?:that|which|to|it|and|for|on|with|in)\s+', candidate, flags=_re.IGNORECASE)[0]
    candidate = _re.sub(r"\bstand\s+by\b.*$", "", candidate, flags=_re.IGNORECASE).strip()
    candidate = _re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=_re.IGNORECASE).strip()
    candidate = candidate.strip(" -,:;.")
    
    # Filter out generic filler phrases
    if any(filler in candidate.lower() for filler in ["is something", "is one that", "are those"]):
        return ""
        
    return _sanitize_shop_part_hint(candidate)


def _extract_shop_refinement_terms(query: str) -> list[str]:
    q = (query or "").lower()
    if not q:
        return []
    found: list[str] = []
    for kw in _SHOP_REFINEMENT_KEYWORDS:
        if kw in q and kw not in found:
            found.append(kw)
    # OEM/part number-ish tokens.
    for m in _re.findall(r"\b(?:oem|p\/n|pn|part\s*#?)\s*[:#-]?\s*([a-z0-9-]{3,})\b", q, flags=_re.IGNORECASE):
        token = m.strip()
        if token and token not in found:
            found.append(token)
    # Dimensions like 20.63mm / 0.81 in.
    for m in _re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|in|inch|inches)\b", q, flags=_re.IGNORECASE):
        token = _re.sub(r"\s+", "", m.lower())
        if token and token not in found:
            found.append(token)
    return found[:6]


def _should_auto_refine_shop_query(query: str) -> bool:
    q = (query or "").lower()
    if not q:
        return False
    if _re.search(r"\b(?:oem|p\/n|pn|part\s*#)\b", q):
        return True
    if _re.search(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|in|inch|inches)\b", q):
        return True
    return any(kw in q for kw in _SHOP_REFINEMENT_KEYWORDS)


def _build_watchdog_search_terms(
    assistant_text: str,
    user_query: str,
    base_part: str,
) -> str:
    phrase = _extract_watchdog_search_phrase(assistant_text)
    base = _sanitize_shop_part_hint(base_part)
    if not phrase:
        phrase = base
    if not phrase:
        return ""
    if _should_auto_refine_shop_query(user_query):
        refinements = _extract_shop_refinement_terms(user_query)
        if refinements or base:
            merged: list[str] = [phrase]
            lower_merged = phrase.lower()
            if base and base.lower() not in lower_merged:
                merged.append(base)
                lower_merged += " " + base.lower()
            for ref in refinements:
                normalized_ref = ref.strip().lower()
                if normalized_ref and normalized_ref not in lower_merged:
                    merged.append(ref)
                    lower_merged += " " + normalized_ref
            phrase = " ".join(merged)
    return _re.sub(r"\s+", " ", phrase).strip()


def _infer_watchdog_sources(text: str) -> list[str]:
    sources = _normalize_shop_sources(_extract_sources(text))
    return sources or list(_SHOP_WATCHDOG_DEFAULT_SOURCES)


_SHOP_INTENT_RE = _re.compile(
    r'\b('
    # explicit "look something up / search" requests
    r'look\s+up|'
    r'search\s+(?:for|on)\s+|'
    r'find\s+(?:me\s+)?(?:a|an|the|one|some|those|these)\s+|'
    r'check\s+(?:amazon|ebay|rock\s*auto|roc\s*auto|autozone|o\'?\s*reill(?:y|ey))\s+for\s+|'
    r'source\s+(?:a|an|the)\s+|'
    r'price\s+check|compare\s+prices|'
    # direct buy/order language only (not generic "should I get")
    r'buy|purchase|order\s+online|shop\s+for|for\s+sale\s+online|'
    r'where\s+(?:can\s+I|do\s+I)\s+(?:buy|order|find)'
    r')',
    _re.IGNORECASE,
)
_AFFIRMATIVE_RE = _re.compile(
    r'^\s*(?:yes|yeah|yep|yup|sure|please|go ahead|do it|please do|ok|okay|'
    r'sounds good|let\'?s (?:do it|go)|why not|absolutely|of course|definitely|'
    r'go for it|do that|pull it up|look it up|check it out|check that)\b.*$',
    _re.IGNORECASE,
)
def _extract_part_from_bold(conversation: list[dict]) -> str:
    """Extract a part name from **bold** terms in the most recent assistant message.
    Used as a fallback when direct part name extraction fails (e.g. interior trim pieces)."""
    last_asst = next(
        (m.get("text", "") for m in reversed(conversation) if m.get("role") == "assistant"),
        "",
    )
    # Find all **term** patterns
    bolds = _re.findall(r'\*\*([^*]{3,50})\*\*', last_asst)
    # Prefer terms mentioned later in the message (most specific/recent)
    for term in reversed(bolds):
        words = term.strip().split()
        # 1–5 words, skip spec-like terms (voltages, measurements, etc.)
        if 1 <= len(words) <= 5 and not _re.search(r'\d+[\.\-]\d+|\bV\b|\bpsi\b|\bft\b|\blb\b|\bin\b', term):
            return term.strip().lower()
    return ""


async def _extract_part_from_description(query: str,
                                        conversation: list[dict],
                                        runtime: VehicleRuntime) -> str:
    """Use Haiku + RAG to identify the auto part being described in natural language.
    Called only when regex extraction fails — i.e. the user described the part
    conceptually rather than naming it directly.

    RAG retrieval is done first so Haiku has the same manual context the main
    LLM uses — this is what lets it translate 'little door on the glove box
    under the radio' into 'console box door' rather than a generic guess.
    """
    import anthropic as _ant
    recent = "\n".join(
        f"{m['role']}: {(m.get('text') or '')[:300]}"
        for m in (conversation or [])[-4:]
    )

    # Retrieve manual chunks to give Haiku the exact part names the service
    # manual uses. Use the last assistant message as the retrieval query when
    # available — it's already RAG-informed and names parts correctly, unlike
    # the user's vague description which may mention unrelated landmarks
    # ("under the radio", "near the glove box") that skew retrieval.
    last_asst_text = next(
        (m.get("text", "") for m in reversed(conversation or []) if m.get("role") == "assistant"),
        "",
    )
    retrieval_query = (last_asst_text[:400] + " " + query).strip() if last_asst_text else query
    manual_context = _rag_context_for_part(retrieval_query, runtime)
    context_block = ""
    if manual_context:
        context_block = (
            f"\nRELEVANT SERVICE MANUAL EXCERPTS:\n"
            f'"""\n{manual_context}\n"""\n'
        )

    prompt = (
        f"A user wants to search for an auto part to buy for a {runtime.profile.label}.\n"
        f"Recent conversation:\n{recent}\n"
        f"User's latest message: {query}\n"
        f"{context_block}\n"
        f"What should we search for? Extract the specific product or part they want.\n"
        f"- If they mention a brand name or product name, INCLUDE it (e.g. 'Sylvania SilverStar 9004')\n"
        f"- If they mention a part number, INCLUDE it (e.g. '9004ST.BP2')\n"
        f"- If they describe a generic part, name it precisely (e.g. 'console box door')\n"
        f"- Use the manual excerpts above for correct terminology when relevant\n"
        f"- Ignore store names (Amazon, eBay, etc.) — just the product/part\n"
        f"Reply with ONLY the search term (1-6 words) — nothing else.\n"
        f"Examples: 'Sylvania SilverStar 9004'  'console box door'  'oil filter'  "
        f"'NGK spark plug BKR5E'  'alternator'  'brake pads'"
    )
    client = _ant.AsyncAnthropic()
    resp = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=30,
        messages=[{"role": "user", "content": prompt}],
    )
    # Keep original casing for brand names (SilverStar, NGK, etc.)
    return resp.content[0].text.strip()


_FETCH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def _fetch_url(url: str, max_chars: int = 4000) -> str:
    resp = _requests.get(url, timeout=12, headers=_FETCH_HEADERS)
    resp.raise_for_status()
    soup = _BS(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text[:max_chars]

def _web_search(query: str, max_results: int = 5) -> str:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No results found."
    return "\n\n".join(
        f"**{r.get('title','')}**\n{r.get('href','')}\n{r.get('body','')}"
        for r in results
    )

_SHOPHOP_URL = os.environ.get("SHOPHOP_URL", "http://localhost:8001/hunt/auto")

def _build_vehicle_payload(profile: VehicleProfile) -> dict:
    """Build the ShopHop vehicle payload for the active vehicle profile."""
    if profile.vehicle_id != DEFAULT_VEHICLE_ID:
        return dict(profile.shopping_vehicle or {})

    settings = _load_settings()
    vin = settings.get("vin", "")
    # Engine derived from VIN position 8 where possible; fall back to G10 default.
    engine = "993cc L3"  # G10 — matches VIN JG1MR3362LK769576
    payload = dict(profile.shopping_vehicle or {})
    payload["engine"] = engine
    if vin:
        payload["vin"] = vin
    return payload


def _rag_context_for_part(part_name: str, runtime: VehicleRuntime | None = None) -> str | None:
    """Retrieve the top 3 manual chunks for a part and return them as a text block.

    This gives ShopHop's Haiku expander real OEM part numbers, cross-references,
    and manual terminology — making its query expansion far more targeted than
    generic layperson guessing.
    """
    runtime = runtime or _load_vehicle_runtime(DEFAULT_VEHICLE_ID)
    if runtime.index is None:
        return None
    try:
        chunks = runtime.index.retrieve(part_name, top_k=3)
        if not chunks:
            return None
        parts = []
        for entry in chunks:
            # retrieve() returns list of {'chunk': {...}, 'score': float}
            c    = entry.get("chunk", entry)
            text = c.get("text", "").strip()
            if text:
                src = c.get("source_doc", "")
                pg  = c.get("page", c.get("page_num", ""))
                parts.append(f"[p{pg} {src}]\n{text[:500]}")
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None


_SHOP_PROGRESS_SOURCE_LABELS = {
    "amazon": "Amazon",
    "rockauto": "RockAuto",
    "ebay": "eBay",
    "autozone": "AutoZone",
    "oreilly": "O'Reilly",
}
_DEFAULT_SHOP_PROGRESS_ORDER = ["amazon", "rockauto", "ebay", "autozone", "oreilly"]


def _pretty_shop_source(source: str) -> str:
    return _SHOP_PROGRESS_SOURCE_LABELS.get(source.lower(), source.title())


@dataclass
class ShopSession:
    last_part_name: str = ""
    last_queries_used: list[str] = field(default_factory=list)
    last_sources_searched: list[str] = field(default_factory=list)
    last_sources_offered: list[str] = field(default_factory=list)
    last_results: list[dict] = field(default_factory=list)
    pending_plan: dict | None = None
    last_manual_context: str | None = None


@dataclass
class PartsSearchResult:
    llm_text: str
    sources: list[str]
    items: list[dict]
    queries_used: list[str] = field(default_factory=list)
    sources_searched: list[str] = field(default_factory=list)
    source_outcomes: dict[str, dict] = field(default_factory=dict)
    manual_context: str | None = None


_shop_sessions: dict[str, ShopSession] = {}


_SHOP_QUERY_STOP_WORDS = {
    "is",
    "a",
    "an",
    "the",
    "that",
    "this",
    "these",
    "those",
    "it",
    "for",
    "of",
    "and",
    "or",
    "with",
    "something",
    "some",
    "thing",
    "stuff",
    "one",
    "get",
    "me",
    "look",
    "looking",
    "search",
    "searching",
    "find",
    "finding",
    "check",
    "checking",
    "buy",
    "purchase",
    "order",
    "shop",
    "shopping",
    "now",
    "yeah",
    "yep",
    "yup",
    "sure",
    "please",
    "go",
    "ahead",
    "on",
    "up",
    "there",
    "here",
    "amazon",
    "ebay",
    "rockauto",
    "autozone",
    "oreilly",
    "let",
    "can",
    "could",
    "would",
    "should",
    "will",
}
_SHOP_QUERY_GENERIC_TERMS = {
    "thing",
    "things",
    "something",
    "stuff",
    "item",
    "items",
    "part",
    "parts",
}
_SHOP_QUERY_KNOWN_JUNK_PHRASES = {
    "is something that",
    "is one that",
    "are those",
    "something that",
    "one that",
}
_SHOP_QUERY_FALLBACK_SKIP_WORDS = _SHOP_QUERY_STOP_WORDS | {
    "look",
    "search",
    "find",
    "check",
    "buy",
    "purchase",
    "order",
    "shop",
    "need",
    "needs",
    "want",
    "wants",
    "replace",
    "replacement",
    "please",
    "yeah",
    "yep",
    "sure",
    "now",
    "can",
    "could",
    "would",
    "should",
    "will",
    "let",
    "go",
    "ahead",
    "there",
    "here",
    "amazon",
    "ebay",
    "rockauto",
    "autozone",
    "oreilly",
    "oreily",
    "oriley",
    "reilly",
}
_SHOP_QUERY_USER_SCAFFOLD_WORDS = {
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "in",
    "stock",
    "available",
    "availability",
    "there",
    "this",
    "cheri",
    "cheris",
    "cheri's",
}


def _shop_query_real_tokens(text: str) -> list[str]:
    tokens = _re.findall(r"[a-z0-9]+", (text or "").lower())
    return [tok for tok in tokens if tok not in _SHOP_QUERY_STOP_WORDS and len(tok) > 1]


def _is_valid_shop_search_terms(terms: str) -> bool:
    cleaned = _sanitize_shop_part_hint(terms)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if any(lowered == phrase or lowered.startswith(f"{phrase} ") for phrase in _SHOP_QUERY_KNOWN_JUNK_PHRASES):
        return False

    real_tokens = _shop_query_real_tokens(cleaned)
    if len(real_tokens) < 3:
        return False
    if not any(len(tok) >= 4 for tok in real_tokens):
        return False
    if all(tok in _SHOP_QUERY_GENERIC_TERMS for tok in real_tokens):
        return False
    return True


def _fallback_shop_query_from_user_text(user_query: str) -> str:
    phrase = _sanitize_shop_part_hint(_extract_watchdog_search_phrase(user_query))
    if phrase:
        phrase_tokens = _shop_query_real_tokens(phrase)
        if (len(phrase_tokens) >= 2 and any(len(tok) >= 4 for tok in phrase_tokens)) or any(
            len(tok) >= 5 for tok in phrase_tokens
        ):
            return phrase

    fallback_tokens: list[str] = []
    for tok in _re.findall(r"[a-z0-9]+", (user_query or "").lower()):
        if tok in _SHOP_QUERY_FALLBACK_SKIP_WORDS or tok in _SHOP_QUERY_USER_SCAFFOLD_WORDS:
            continue
        if len(tok) < 3 and not any(ch.isdigit() for ch in tok):
            continue
        fallback_tokens.append(tok)

    if len(fallback_tokens) >= 2 and any(len(tok) >= 4 for tok in fallback_tokens):
        return " ".join(fallback_tokens[:6])
    if len(fallback_tokens) == 1:
        token = fallback_tokens[0]
        if len(token) >= 5 and token not in _SHOP_QUERY_GENERIC_TERMS:
            return token
    return ""


def _validate_shop_search_terms(
    terms: str,
    shop_session: "ShopSession | None",
    user_query: str,
) -> str | None:
    cleaned = _sanitize_shop_part_hint(terms)
    if _is_valid_shop_search_terms(cleaned):
        return cleaned

    original = cleaned or (terms or "").strip()
    session_part = _sanitize_shop_part_hint((shop_session.last_part_name if shop_session else ""))
    if session_part:
        print(f"[shop-query-rejected] original: {original!r}, fallback: session.last_part_name={session_part!r}")
        return session_part

    user_fallback = _sanitize_shop_part_hint(_fallback_shop_query_from_user_text(user_query))
    if user_fallback:
        print(f"[shop-query-rejected] original: {original!r}, fallback: user_query={user_fallback!r}")
        return user_fallback

    print(f"[shop-query-rejected] original: {original!r}, no fallback available - skipping search")
    return None


async def _parts_search(
    part_name: str,
    runtime: VehicleRuntime,
    sources: list[str] | None = None,
    manual_context: str | None = None,
    progress_cb: Callable[[dict], Awaitable[None]] | None = None,
) -> PartsSearchResult:
    """Call ShopHop for parts.
    If sources is given, only search those specific ShopHop sources."""
    import httpx as _httpx

    # Pull manual context before calling ShopHop so the query expander
    # has OEM part numbers and cross-references to work with.
    if manual_context is None:
        manual_context = _rag_context_for_part(part_name, runtime)

    requested_sources = _normalize_shop_sources(sources)

    try:
        payload = {
            "part": part_name,
            "vehicle": _build_vehicle_payload(runtime.profile),
            "max_results": 20,
        }
        if requested_sources:
            payload["sources"] = requested_sources
        if manual_context:
            payload["context"] = manual_context

        if progress_cb:
            await progress_cb({"type": "shopping", "label": "Calling ShopHop"})
            # When the user requested specific sources (e.g. RockAuto), show
            # per-source progress immediately so status is visible even if the
            # upstream call is slow or returns empty.
            if requested_sources:
                for src_name in requested_sources:
                    await progress_cb({
                        "type": "shopping",
                        "label": f"Searching {_pretty_shop_source(src_name)}",
                    })

        async with _httpx.AsyncClient() as hc:
            r = await hc.post(
                _SHOPHOP_URL,
                json=payload,
                timeout=90,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return PartsSearchResult(
            llm_text=f"[Parts search unavailable: {exc}]",
            sources=[],
            items=[],
            source_outcomes={"request": {"status": "error", "reason": str(exc)}},
            manual_context=manual_context,
        )

    items            = data.get("items", [])
    sources_searched = _normalize_shop_sources(data.get("sources_searched", []))
    queries_used     = data.get("queries_used", [])
    source_outcomes  = data.get("source_outcomes", {})
    source_errors    = data.get("source_errors", {})

    # Backward compatibility path for older ShopHop responses.
    if not isinstance(source_outcomes, dict):
        source_outcomes = {}
    for src_name, err in (source_errors or {}).items():
        source_outcomes.setdefault(
            src_name,
            {"status": "error", "reason": str(err)},
        )

    item_counts: dict[str, int] = {}
    for item in items:
        src_name = (item.get("source") or "").lower()
        if src_name:
            item_counts[src_name] = item_counts.get(src_name, 0) + 1

    # Guarantee deterministic status for every source touched by the request.
    expected_sources = requested_sources or sources_searched
    for src_name in expected_sources:
        outcome = source_outcomes.get(src_name)
        count = item_counts.get(src_name, 0)
        if not isinstance(outcome, dict):
            source_outcomes[src_name] = {
                "status": "ok" if count > 0 else "no_results",
                "count": count,
            }
            continue
        outcome.setdefault("count", count)
        status = (outcome.get("status") or "").lower()
        if status == "ok" and count == 0:
            outcome["status"] = "no_results"
        elif not status:
            outcome["status"] = "ok" if count > 0 else "no_results"
        source_outcomes[src_name] = outcome

    if progress_cb:
        progress_sources = sources_searched if not requested_sources else []
        for src_name in progress_sources:
            await progress_cb({
                "type": "shopping",
                "label": f"Searching {_pretty_shop_source(src_name)}",
            })
        await progress_cb({"type": "shopping", "label": "Wrapping Up"})

    source_refs = [f"parts search ({', '.join(sources_searched)}): {part_name}"]

    if not items:
        unavailable_sources = [
            src_name
            for src_name, outcome in (source_outcomes or {}).items()
            if isinstance(outcome, dict)
            and (outcome.get("status") or "").lower() in {"blocked", "error"}
        ]
        llm_text = (
            f"[PARTS SEARCH: no results for '{part_name}' across "
            f"{', '.join(sources_searched) or 'all sources'}]"
        )
        if unavailable_sources:
            llm_text += (
                "\n[Search unavailable on: "
                + ", ".join(sorted(set(unavailable_sources)))
                + "]"
            )
        if source_outcomes:
            llm_text += f"\n[Source outcomes: {source_outcomes}]"
        return PartsSearchResult(
            llm_text=llm_text,
            sources=source_refs,
            items=[],
            queries_used=queries_used,
            sources_searched=sources_searched,
            source_outcomes=source_outcomes,
            manual_context=manual_context,
        )

    # Build a compact LLM summary — sidebar shows the full cards
    lines = [
        f"[PARTS SEARCH — {len(items)} results for '{part_name}' "
        f"from {', '.join(sources_searched)}]"
    ]
    for it in items[:10]:
        line = f"- [{it.get('source','').upper()}] {it.get('title','')}"
        if it.get("price"):
            line += f" — {it['price']}"
        if it.get("item_url"):
            line += f" [View]({it['item_url']})"
        lines.append(line)
    if len(items) > 10:
        lines.append(f"...and {len(items) - 10} more shown in the Parts sidebar.")
    if source_outcomes:
        lines.append(f"[Source outcomes: {source_outcomes}]")
    lines.append("[END PARTS SEARCH]")
    llm_text = "\n".join(lines)

    return PartsSearchResult(
        llm_text=llm_text,
        sources=source_refs,
        items=items,
        queries_used=queries_used,
        sources_searched=sources_searched,
        source_outcomes=source_outcomes,
        manual_context=manual_context,
    )



async def _identify_part_for_search(query: str,
                                    conversation: list[dict] | None,
                                    runtime: VehicleRuntime) -> str:
    """Single source of truth for part name extraction before a shopping search.

    Priority:
    1. Bold term from the last assistant message — it's already RAG-informed
       from the previous turn, so it names things correctly (e.g. 'console box door').
    2. Haiku + RAG — give Haiku the manual chunks for this query so it can
       identify the part from what the service manual actually calls it.
    """
    if conversation:
        bold = _extract_part_from_bold(conversation)
        if bold:
            return bold
    try:
        return await _extract_part_from_description(query, conversation or [], runtime)
    except Exception:
        return ""


async def _enrich_query(query: str, conversation: list[dict] | None = None) -> tuple[str, list[str], list[dict]]:
    """Enrich the query with fetched URLs and web search results.
    Parts search is now handled by the LLM via [SHOP_SEARCH] tags.
    Returns (enriched_query, sources, shopping_items).
    """
    extra, sources, shopping_items = [], [], []
    urls = _URL_RE.findall(query)
    # Do not run generic web search enrichment for parts-buy intent; those
    # queries should stay clean so the LLM can emit [SHOP_SEARCH: ...].
    is_shop_intent = _is_shop_query(query, conversation)

    for url in urls[:3]:
        try:
            content = _fetch_url(url)
            extra.append(f"[WEB PAGE CONTENT FROM {url}]\n{content}\n[END WEB PAGE CONTENT]")
            sources.append(url)
        except Exception as exc:
            extra.append(f"[Could not fetch {url}: {exc}]")

    if not urls and not is_shop_intent:
        if _SEARCH_RE.search(query):
            search_q = _SEARCH_RE.sub('', query).strip(' .,?!')
            if search_q:
                try:
                    results = _web_search(search_q)
                    extra.append(f"[WEB SEARCH RESULTS FOR: {search_q}]\n{results}\n[END WEB SEARCH RESULTS]")
                    sources.append(f"web search: {search_q}")
                except Exception as exc:
                    extra.append(f"[Web search failed: {exc}]")

    enriched = "\n\n".join(extra) + "\n\n" + query if extra else query
    return enriched, sources, shopping_items


# ── API models ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
  query: str
  conversation: list[dict] = []
  model: str | None = None
  vehicle: str | None = None
  project_context: str | None = None
  images: list[str] = []  # base64 data URLs from the frontend
  shop_mode_hint: bool = False
  shop_part_hint: str | None = None
  tech_mode_hint: bool = False
  deep_research: bool = False
  session_id: str | None = None
  project_id: str | None = None


class TitleRequest(BaseModel):
  message: str


class TTSRequest(BaseModel):
  text: str
  voice: str | None = None


class WorksheetRequest(BaseModel):
  prompt: str  # e.g. "alternator diagnostic tests"
  vehicle: str | None = None


class ChatAPIResponse(BaseModel):
    answer: str
    citations: list[dict]
    figure_refs: list[str]
    figures: list[dict]  # {figure_id, page, caption_text, url}
    mode: str = "normal"
    deep_research_summary: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────


_DEFAULT_TTS_VOICE = "en-US-Studio-Q"
_MAX_TTS_TEXT_CHARS = 5000


def _tts_api_key() -> str:
    return (
        os.environ.get("GOOGLE_TTS_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()


def _tts_language_code(voice_name: str) -> str:
    parts = [part for part in (voice_name or "").split("-") if part]
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return "en-US"


# Named voices (Enceladus, Puck, Charon, Kore, etc.) use the Gemini TTS model
_NAMED_VOICES = {
    "Enceladus", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus",
    "Pegasus", "Perseus", "Schedar", "Sulafat", "Umbriel", "Zephyr",
    "Achernar", "Autonoe", "Callirrhoe", "Desdema", "Elara", "Gacrux",
}


def _is_named_voice(voice_name: str) -> bool:
    return voice_name.split("-")[-1] in _NAMED_VOICES or voice_name in _NAMED_VOICES


def _synthesize_google_tts(text: str, voice: str | None = None) -> bytes:
    api_key = _tts_api_key()
    if not api_key:
        raise HTTPException(500, "Google TTS API key is not configured")

    clean_text = (text or "").strip()
    if not clean_text:
        raise HTTPException(400, "Text is required")
    if len(clean_text) > _MAX_TTS_TEXT_CHARS:
        raise HTTPException(400, f"Text exceeds {_MAX_TTS_TEXT_CHARS} characters")

    voice_name = (voice or _DEFAULT_TTS_VOICE).strip() or _DEFAULT_TTS_VOICE

    if _is_named_voice(voice_name):
        # Gemini TTS model voices — use v1beta1 with modelName and prompt
        payload = {
            "input": {
                "text": clean_text,
                "prompt": _DEFAULT_TTS_PROMPT,
            },
            "voice": {
                "languageCode": "en-us",
                "modelName": _DEFAULT_TTS_MODEL,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "pitch": 0,
                "speakingRate": 1,
            },
        }
        api_version = "v1beta1"
    else:
        # Standard Cloud TTS voices (Neural2, Studio, etc.)
        payload = {
            "input": {"text": clean_text},
            "voice": {
                "languageCode": _tts_language_code(voice_name),
                "name": voice_name,
            },
            "audioConfig": {"audioEncoding": "MP3"},
        }
        api_version = "v1"

    request = urllib.request.Request(
        url=(
            f"https://texttospeech.googleapis.com/{api_version}/text:synthesize?"
            + urllib.parse.urlencode({"key": api_key})
        ),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(502, f"Google TTS request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(502, f"Google TTS request failed: {exc.reason}") from exc

    audio_content = response_payload.get("audioContent")
    if not audio_content:
        raise HTTPException(502, "Google TTS response did not include audio content")
    try:
        return base64.b64decode(audio_content)
    except Exception as exc:
        raise HTTPException(502, "Google TTS response contained invalid audio data") from exc


@app.get("/api/settings")
async def get_settings():
    return JSONResponse(_load_settings())


@app.put("/api/settings")
async def put_settings(request: Request):
    data = await request.json()
    allowed = {
        "vin", "maintenance_schedule", "oil_viscosity", "odometer",
        "tire_size", "driving_profile", "service_history", "modifications",
    }
    cleaned = {k: v for k, v in data.items() if k in allowed}
    _save_settings(cleaned)
    return JSONResponse({"ok": True})


@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    voice_name = (req.voice or _DEFAULT_TTS_VOICE).strip() or _DEFAULT_TTS_VOICE
    audio_bytes = await asyncio.to_thread(_synthesize_google_tts, req.text, req.voice)
    media_type = "audio/wav" if _is_named_voice(voice_name) else "audio/mpeg"
    return Response(content=audio_bytes, media_type=media_type)


@app.post("/api/generate-title")
async def api_generate_title(req: TitleRequest) -> JSONResponse:
  """Generate a short chat title from the first user message using Gemini."""
  import os
  from src.gemini_api import call_gemini
  try:
    msg = [{"system": "You generate short titles for chat conversations. Given the user's message, create a concise 3-6 word title that captures what they're asking about. Return ONLY the title text. No quotes, no punctuation at the end. Examples: 'Oil Change Procedure', 'Brake Pad Replacement', 'Engine Won't Start', 'AC Blowing Warm Air'.", "messages": [{"role": "user", "content": req.message}]}]
    title = call_gemini(msg, model="gemini-2.5-flash")
    title = title.strip().strip('"').strip("'").rstrip(".")
    if len(title) > 50:
      title = title[:50].rsplit(" ", 1)[0]
    return JSONResponse({"title": title})
  except Exception as e:
    print(f"Title generation failed: {e}")
    fallback = req.message[:42] + ("…" if len(req.message) > 42 else "")
    return JSONResponse({"title": fallback})


_SHOP_SEARCH_RE = _re.compile(
    r'\[SHOP[-_\s]*SEARCH\s*[:\-]?\s*(.+?)(?:\s*\|\s*([a-z,\s]+))?\s*\]',
    _re.IGNORECASE,
)
_SHOP_PROMISE_RE = _re.compile(
    r'\b(?:let\s+me\s+(?:search|look|fire\s+off))|'
    r'let\s+me\s+do\s+(?:a\s+)?(?:proper\s+)?(?:targeted\s+)?search|'
    r'on\s+it(?:\s*[—-]\s*|\s*,?\s*)?(?:i\'?m\s+)?search(?:ing)?\b|'
    r'searching\s+(?:amazon|ebay|rock\s*auto|roc\s*auto|autozone|o\'?\s*reill(?:y|ey)|for)\b|'
    r'while\s+that\s+runs|stand\s+by|'
    r'i\'?ll\s+(?:search|look)|'
    r'i\s+will\s+(?:search|look)|'
    r'going\s+to\s+(?:search|look)\b',
    _re.IGNORECASE,
)


_SHOP_FOLLOWUP_RE = _re.compile(
    r'\b(?:options?|ones?|crummy|junk|better|best|cheaper|price|prices|'
    r'availability|available|in\s+stock|worth|quality|fit(?:ment)?|'
    r'which\s+one|what\s+about|how\s+about|do\s+they\s+have)\b',
    _re.IGNORECASE,
)
_SHOP_CONTEXT_RE = _re.compile(
    r'\b(?:amazon|rock\s*auto|roc\s*auto|ebay|autozone|o\'?\s*reill(?:y|ey)|'
    r'parts\s+search|view\s+on|fitment|twin\s+pack|silverstar|9004)\b',
    _re.IGNORECASE,
)
_SHOP_OFFER_RE = _re.compile(
    r'\b(?:want\s+me\s+to\s+(?:search|look)|let\s+me\s+search|'
    r'should\s+I\s+(?:search|look|check)|want\s+me\s+to\s+(?:pull|check))\b',
    _re.IGNORECASE,
)
_AFFIRM_SHOP_FOLLOWUP_RE = _re.compile(
    r'^\s*(?:yes|yeah|yep|yup|sure|ok|okay|please|do\s+it|go\s+ahead|'
    r'why\s+not|absolutely|of\s+course)?\s*'
    r'(?:find|search|look(?:\s+for)?|check|pull|take\s+a\s+look)?\s*'
    r'(?:me\s+)?(?:it|that|those|these|there|thoes)?\s*[.!]?\s*$',
    _re.IGNORECASE,
)


def _is_shop_query(query: str, conversation: list[dict] | None = None) -> bool:
    q = query or ""
    if _SHOP_INTENT_RE.search(q) or _RA_SEARCH_RE.search(q) or _extract_sources(q):
        return True
    last_asst = _last_assistant_text(conversation)
    if last_asst and _SHOP_OFFER_RE.search(last_asst):
        if _AFFIRMATIVE_RE.search(q) or _AFFIRM_SHOP_FOLLOWUP_RE.search(q):
            return True
    if not _SHOP_FOLLOWUP_RE.search(q):
        return False
    if last_asst and _SHOP_CONTEXT_RE.search(last_asst):
        return True
    return bool(last_asst and _SHOP_OFFER_RE.search(last_asst) and _AFFIRM_SHOP_FOLLOWUP_RE.search(q))


def _last_shop_user_query(conversation: list[dict] | None) -> str:
    for msg in reversed(conversation or []):
        if msg.get("role") != "user":
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if _SHOP_INTENT_RE.search(text) or _RA_SEARCH_RE.search(text) or _extract_sources(text):
            return text
    return ""


def _to_chat_payload(response,
                     web_sources: list[str],
                     shopping_results: list[dict],
                     shopping_part_name: str | None,
                     *,
                     runtime: VehicleRuntime,
                     vehicle_id: str,
                     saved_notes: list[dict] | None = None) -> dict:
    figures_out = []
    print(f"  [figures] figure_refs from response: {response.figure_refs}")
    # Also check what figure citations are in the answer text
    import re as _fig_re
    fig_cites_in_answer = _fig_re.findall(r'\[p(\d+)\s*\|\s*fig:\s*([^\]]+)\]', response.answer, _fig_re.IGNORECASE)
    print(f"  [figures] fig citations in answer text: {fig_cites_in_answer}")
    for fig_id in response.figure_refs:
        # Try direct lookup first (asset-style ID like fig_p0477_000)
        fig = runtime.fig_lookup.get(fig_id)
        # Fallback: resolve chunk-style ID (fig_7a_p477_0) → asset-style
        resolved_id = fig_id
        if not fig and fig_id in runtime.chunk_fig_map:
            resolved_id = runtime.chunk_fig_map[fig_id]
            fig = runtime.fig_lookup.get(resolved_id)
        if fig:
            fig_url = f"/api/figures/{resolved_id}"
            if vehicle_id != DEFAULT_VEHICLE_ID:
                fig_url += f"?vehicle={vehicle_id}"
            figures_out.append({
                "figure_id":    resolved_id,
                "page":         fig.get("page"),
                "caption_text": fig.get("caption_text", ""),
                "url":          fig_url,
            })

    enriched_citations = []
    for c in response.citations:
        cite = {
            "chunk_id":     c.chunk_id,
            "page":         c.page,
            "source_label": c.source_label,
            "section_path": c.section_path,
        }
        # Enrich with type and text preview from chunk lookup
        chunk_info = runtime.chunk_text_lookup.get(c.chunk_id)
        if chunk_info:
            cite["type"] = chunk_info["type"]
            text = chunk_info["text"] or ""
            cite["text_preview"] = text[:250] + ("…" if len(text) > 250 else "")
            if chunk_info["section_path"]:
                cite["section_path"] = chunk_info["section_path"]
        else:
            cite["type"] = "text"
            cite["text_preview"] = ""
        enriched_citations.append(cite)

    return {
        "answer":      response.answer,
        "citations":   enriched_citations,
        "figure_refs":         response.figure_refs,
        "figures":             figures_out,
        "mode":                response.mode,
        "deep_research_summary": response.deep_research_summary,
        "web_sources":         web_sources,
        "shopping_results":    shopping_results,
        "shopping_part_name":  shopping_part_name,
        "saved_notes":         saved_notes or [],
    }


def _schedule_progress_callback(
    progress_cb: Callable[[dict], Awaitable[None]] | None,
    data: dict,
) -> None:
    if not progress_cb:
        return
    result = progress_cb(data)
    if inspect.isawaitable(result):
        asyncio.create_task(result)


async def _run_chat_request(
    req: ChatRequest,
    progress_cb: Callable[[dict], Awaitable[None]] | None = None,
) -> dict:
    vehicle_id = _normalize_vehicle_id(req.vehicle)
    runtime = _load_vehicle_runtime(vehicle_id, require_index=True)
    if runtime.index is None:
        raise HTTPException(503, f"{runtime.profile.label} index not loaded yet")

    # Allow model override from frontend
    chat_config = dict(runtime.config)
    if req.model:
        chat_config = dict(runtime.config)
        chat_config = chat_config.copy()
        chat_config["chat"] = dict(chat_config.get("chat", {}))
        chat_config["chat"]["model"] = req.model

    vehicle_settings = ""
    if vehicle_id == DEFAULT_VEHICLE_ID:
        vehicle_settings = _build_settings_context(_load_settings())

    # Enrich query (URL fetch, web search) — parts search is now handled by the
    # LLM via [SHOP_SEARCH] tags so it can use its full context to decide what
    # to search for.
    enriched_query, web_sources, shopping_results = await _enrich_query(req.query, req.conversation)

    # LLM-first mode: always let Cheri Doctor respond first, then execute
    # [SHOP_SEARCH] tags it emits (or watchdog-generated tags if missing).
    shop_session = _get_shop_session(req.query, req.conversation, req.project_context)
    if not shop_session.last_sources_offered:
        offered_from_context = _extract_offered_sources(_last_assistant_text(req.conversation))
        if offered_from_context:
            shop_session.last_sources_offered = offered_from_context

    chat_progress_cb = None
    if progress_cb:
        loop = asyncio.get_running_loop()

        def chat_progress_cb(data: dict) -> None:
            loop.call_soon_threadsafe(_schedule_progress_callback, progress_cb, data)

    if progress_cb:
        await progress_cb({"type": "thinking", "label": "Thinking"})

    notes_context = _build_notes_context(req.project_id, req.session_id)

    def retrieve_notes(tool_input: dict) -> tuple[str, bool]:
        return _retrieve_notes_result(req.project_id, req.session_id, tool_input)

    # Note-saving callback: when the LLM calls save_note, persist to DB
    saved_notes: list[dict] = []

    def note_callback(note_data: dict) -> None:
        if not _clean_scope_id(req.project_id) and not _clean_scope_id(req.session_id):
            print(f"  [note] Skipped unscoped note: \"{note_data.get('title')}\"")
            return
        note_id = f"note_{int(time.time()*1000)}_{secrets.token_hex(4)}"
        now_ms = int(time.time() * 1000)
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO notes (id, project_id, session_id, created, updated, title, content, tags, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    note_id,
                    req.project_id,
                    req.session_id,
                    now_ms,
                    now_ms,
                    note_data.get("title", ""),
                    note_data.get("content", ""),
                    json.dumps(note_data.get("tags", [])),
                    note_data.get("source", "cheri_doctor"),
                )
            )
        saved_notes.append({"id": note_id, **note_data})
        print(f"  [note] Saved: \"{note_data.get('title')}\" (id={note_id})")

    response = await asyncio.to_thread(
        chat,
        query=enriched_query,
        conversation=req.conversation,
        index=runtime.index,
        config=chat_config,
        skip_vision=False,
        deep_research=req.deep_research,
        project_context=req.project_context,
        notes_context=notes_context,
        vehicle_settings=vehicle_settings,
        images=req.images,
        progress_cb=chat_progress_cb,
        note_callback=note_callback,
        retrieve_notes=retrieve_notes,
    )

    shopping_part_name: str | None = None

    # Watchdog safety-net: if Cheri promises a search but misses [SHOP_SEARCH],
    # auto-inject a tag and execute it so the turn doesn't dead-end.
    if (not _SHOP_SEARCH_RE.search(response.answer or "")) and _SHOP_PROMISE_RE.search(response.answer or ""):
        watchdog_base_part = _sanitize_shop_part_hint(shop_session.last_part_name)
        if not watchdog_base_part:
            hinted_part = _sanitize_shop_part_hint(req.shop_part_hint)
            if hinted_part and _is_valid_shop_search_terms(hinted_part):
                watchdog_base_part = hinted_part
            elif hinted_part:
                print(
                    f"[shop-query-rejected] original: {hinted_part!r}, "
                    "fallback: ignored shop_part_hint for watchdog base"
                )
        watchdog_terms = _build_watchdog_search_terms(
            assistant_text=response.answer or "",
            user_query=req.query,
            base_part=watchdog_base_part,
        )
        if watchdog_terms:
            watchdog_sources = _infer_watchdog_sources(response.answer or "")
            auto_tag = f"[SHOP_SEARCH: {watchdog_terms} | {', '.join(watchdog_sources)}]"
            print(f"[shop-watchdog] Auto-injected tag: {auto_tag}")
            from src.models import ChatResponse
            response = ChatResponse(
                answer=(response.answer or "").rstrip() + "\n\n" + auto_tag,
                citations=response.citations,
                figure_refs=response.figure_refs,
                mode=response.mode,
                deep_research_summary=response.deep_research_summary,
            )

    # Parse all [SHOP_SEARCH] tags in the response and execute each once.
    shop_matches = list(_SHOP_SEARCH_RE.finditer(response.answer or ""))
    if shop_matches:
        search_specs: list[tuple[str, list[str] | None]] = []
        seen_specs: set[tuple[str, tuple[str, ...]]] = set()
        for match in shop_matches:
            terms = _validate_shop_search_terms(
                terms=match.group(1),
                shop_session=shop_session,
                user_query=req.query,
            )
            if not terms:
                continue
            source_str = match.group(2)
            parsed_sources = (
                _normalize_shop_sources([s.strip() for s in source_str.split(",") if s.strip()])
                if source_str else None
            )
            spec_key = (terms.lower(), tuple(parsed_sources or []))
            if spec_key in seen_specs:
                continue
            seen_specs.add(spec_key)
            search_specs.append((terms, parsed_sources))

        if not search_specs:
            from src.models import ChatResponse
            response = ChatResponse(
                answer=_SHOP_SEARCH_RE.sub('', response.answer).strip(),
                citations=response.citations,
                figure_refs=response.figure_refs,
                mode=response.mode,
                deep_research_summary=response.deep_research_summary,
            )

        if search_specs:
            if progress_cb:
                await progress_cb({"type": "shopping", "label": "Running parts search"})

            all_results: list[PartsSearchResult] = []
            all_items: list[dict] = []
            all_llm_blocks: list[str] = []
            all_queries: list[str] = []
            all_sources_searched: list[str] = []
            merged_outcomes: dict[str, dict] = {}

            try:
                for search_terms, shop_sources in search_specs:
                    search_result = await _parts_search(
                        search_terms,
                        runtime,
                        sources=shop_sources,
                        progress_cb=progress_cb,
                    )
                    all_results.append(search_result)
                    all_llm_blocks.append(search_result.llm_text)
                    web_sources.extend(search_result.sources)
                    all_items.extend(search_result.items)
                    all_queries.extend(search_result.queries_used)
                    for src in search_result.sources_searched:
                        if src not in all_sources_searched:
                            all_sources_searched.append(src)
                    for src, outcome in (search_result.source_outcomes or {}).items():
                        if isinstance(outcome, dict):
                            merged_outcomes[src] = outcome
                        elif src not in merged_outcomes:
                            merged_outcomes[src] = {"status": "error", "reason": str(outcome)}

                # Deduplicate merged results across multiple tags/queries.
                deduped_items: list[dict] = []
                seen_item_keys: set[tuple[str, str, str]] = set()
                for item in all_items:
                    key = (
                        (item.get("item_url") or "").strip().lower(),
                        (item.get("source") or "").strip().lower(),
                        (item.get("title") or "").strip().lower(),
                    )
                    if key in seen_item_keys:
                        continue
                    seen_item_keys.add(key)
                    deduped_items.append(item)
                shopping_results = deduped_items

                shopping_part_name = search_specs[0][0]
                shop_session.last_part_name = shopping_part_name
                shop_session.last_queries_used = all_queries
                shop_session.last_sources_searched = all_sources_searched
                shop_session.last_results = list(shopping_results)
                if all_results and all_results[-1].manual_context:
                    shop_session.last_manual_context = all_results[-1].manual_context

                # Strip [SHOP_SEARCH] tags from the assistant text before the
                # narration pass so the owner sees natural language.
                pre_search_text = _SHOP_SEARCH_RE.sub('', response.answer).strip()

                followup_conversation = list(req.conversation or [])
                followup_conversation.append({"role": "user", "text": req.query})
                if pre_search_text:
                    followup_conversation.append({"role": "assistant", "text": pre_search_text})

                if progress_cb:
                    await progress_cb({"type": "shopping", "label": "Summarizing results"})

                followup_query = (
                    "\n\n".join(all_llm_blocks)
                    + "\n\nNow summarize the parts search results above for the owner. "
                      "Highlight the best matches, prices, and any fitment concerns."
                )

                response = await asyncio.to_thread(
                    chat,
                    query=followup_query,
                    conversation=followup_conversation,
                    index=runtime.index,
                    config=chat_config,
                    skip_vision=True,
                    project_context=req.project_context,
                    notes_context=notes_context,
                    vehicle_settings=vehicle_settings,
                    progress_cb=chat_progress_cb,
                    retrieve_notes=retrieve_notes,
                )

                offered_sources = _extract_offered_sources(response.answer)
                if offered_sources:
                    shop_session.last_sources_offered = offered_sources

                if shop_session.last_sources_offered:
                    available_sources = [
                        src for src in shop_session.last_sources_offered
                        if src not in all_sources_searched
                    ]
                    unavailable_sources = [
                        src
                        for src, outcome in merged_outcomes.items()
                        if isinstance(outcome, dict)
                        and (outcome.get("status") or "").lower() in {"blocked", "error", "not_implemented"}
                    ]
                    if available_sources:
                        shop_session.pending_plan = {
                            "part": shopping_part_name,
                            "completed": all_sources_searched,
                            "available": available_sources,
                            "unavailable": unavailable_sources,
                        }
                    else:
                        shop_session.pending_plan = None
                else:
                    shop_session.pending_plan = None

                # Prepend the pre-search text so the owner sees full narrative.
                if pre_search_text:
                    from src.models import ChatResponse
                    response = ChatResponse(
                        answer=pre_search_text + "\n\n" + response.answer,
                        citations=response.citations,
                        figure_refs=response.figure_refs,
                        mode=response.mode,
                        deep_research_summary=response.deep_research_summary,
                    )
            except Exception as exc:
                from src.models import ChatResponse
                response = ChatResponse(
                    answer=_SHOP_SEARCH_RE.sub('', response.answer).strip()
                        + f"\n\n*Parts search failed: {exc}*",
                    citations=response.citations,
                    figure_refs=response.figure_refs,
                    mode=response.mode,
                    deep_research_summary=response.deep_research_summary,
                )

    return _to_chat_payload(
        response,
        web_sources,
        shopping_results,
        shopping_part_name,
        runtime=runtime,
        vehicle_id=vehicle_id,
        saved_notes=saved_notes,
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> JSONResponse:
  payload = await _run_chat_request(req)
  return JSONResponse(payload)


@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
  queue: asyncio.Queue[tuple[str | None, dict]] = asyncio.Queue()

  async def _emit_progress(data: dict):
    await queue.put(("status", data))

  async def _worker():
    try:
      payload = await _run_chat_request(req, progress_cb=_emit_progress)
      await queue.put(("final", payload))
    except Exception as exc:
      await queue.put(("error", {"message": str(exc)}))
    finally:
      await queue.put((None, {}))

  asyncio.create_task(_worker())

  async def _event_stream():
    while True:
      event, data = await queue.get()
      if event is None:
        break
      yield _sse_event(event, data)

  return StreamingResponse(
    _event_stream(),
    media_type="text/event-stream",
    headers={
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  )


@app.post("/api/generate-worksheet")
async def api_generate_worksheet(req: WorksheetRequest):
  """Generate a PDF diagnostic worksheet from a user prompt."""
  runtime = _load_vehicle_runtime(req.vehicle, require_index=True)
  if runtime.index is None:
    raise HTTPException(503, f"{runtime.profile.label} index not loaded yet")
  try:
    from tools.worksheet_generator import generate_worksheet_pdf
    pdf_bytes = generate_worksheet_pdf(req.prompt, runtime.index, runtime.config)
    
    # Ensure we have proper bytes object
    if isinstance(pdf_bytes, bytearray):
      pdf_bytes = bytes(pdf_bytes)
    elif not isinstance(pdf_bytes, bytes):
      pdf_bytes = bytes(pdf_bytes, 'latin-1') if isinstance(pdf_bytes, str) else bytes(pdf_bytes)
    
    # Sanitize filename from prompt
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in req.prompt[:50]).strip() or "worksheet"
    filename = f"{safe_name}.pdf"
    return Response(
      content=pdf_bytes,
      media_type="application/pdf",
      headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
  except Exception as e:
    raise HTTPException(500, str(e))


@app.get("/api/figures/{figure_id}")
async def serve_figure(figure_id: str, vehicle: str | None = None):
    vehicle_id = _normalize_vehicle_id(vehicle)
    runtime = _load_vehicle_runtime(vehicle_id)
    resolved_id = runtime.chunk_fig_map.get(figure_id, figure_id)
    fig = runtime.fig_lookup.get(resolved_id)
    if not fig:
        raise HTTPException(404, f"Figure {figure_id} not found")
    asset_path = PROJECT_ROOT / fig.get("asset_path", "")
    if not asset_path.exists():
        raise HTTPException(404, "Image file not found")
    media_type = "image/webp" if asset_path.suffix == ".webp" else "image/png"
    return FileResponse(str(asset_path), media_type=media_type)


@app.get("/api/manual/page/{page_num}")
async def serve_manual_page(page_num: int, vehicle: str | None = None):
    """Serve a rasterized manual page image."""
    runtime = _load_vehicle_runtime(vehicle)
    if runtime.profile.page_image_dir is None:
        raise HTTPException(404, f"Page images are not available for {runtime.profile.label}")
    page_path = runtime.profile.page_image_dir / f"page_{page_num:04d}.png"
    if not page_path.exists():
        raise HTTPException(404, f"Page {page_num} not found")
    return FileResponse(str(page_path), media_type="image/png")

@app.get("/api/manual/toc")
async def get_manual_toc(vehicle: str | None = None):
    """Return the table of contents with section codes and page numbers."""
    runtime = _load_vehicle_runtime(vehicle)
    section_names = runtime.config.get("structure", {}).get("section_names", {})
    # Build section → first page mapping from chunk data
    section_pages: dict[str, int] = {}
    for cid, info in runtime.chunk_text_lookup.items():
        sp = info.get("section_path", "")
        # Extract section code from chunk_id (e.g., proc_6a1_p241_0 → 6a1 → 6A1)
        parts = cid.split("_")
        if len(parts) >= 3:
            try:
                page = int(parts[-2].lstrip("p"))
                # Try to find section code from section_path or chunk_id
                for code in section_names:
                    if code.lower() in cid.lower() or sp.startswith(section_names[code]):
                        if code not in section_pages or page < section_pages[code]:
                            section_pages[code] = page
            except (ValueError, IndexError):
                pass

    toc = []
    for code, name in section_names.items():
        toc.append({
            "code": code,
            "name": name,
            "start_page": section_pages.get(code, 0),
        })
    return JSONResponse(toc)

@app.get("/api/manual/info")
async def get_manual_info(vehicle: str | None = None):
    """Return manual metadata (total pages, etc.)."""
    runtime = _load_vehicle_runtime(vehicle)
    pages_dir = runtime.profile.page_image_dir
    total = len(list(pages_dir.glob("page_*.png"))) if pages_dir and pages_dir.exists() else 0
    return JSONResponse({"total_pages": total, "title": f"{runtime.profile.label} Service Manual"})

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    frontend = PROJECT_ROOT / "frontend" / "index.html"
    if frontend.exists():
        return HTMLResponse(frontend.read_text(encoding="utf-8"))
    return HTMLResponse(_CHAT_UI_HTML)


# ── Chat UI HTML ──────────────────────────────────────────────────────────

_CHAT_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>1990 Geo Metro — Service Manual Chat</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f0f0f;
      color: #e0e0e0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: #1a1a1a;
      border-bottom: 1px solid #333;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    header h1 { font-size: 1.1rem; font-weight: 600; color: #fff; }
    header span { font-size: 0.8rem; color: #888; }
    .badge {
      background: #2a5a2a;
      color: #6fcf6f;
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 20px;
      font-weight: 600;
    }

    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .message { max-width: 820px; width: 100%; }
    .message.user { align-self: flex-end; }
    .message.assistant { align-self: flex-start; }

    .bubble {
      padding: 14px 18px;
      border-radius: 12px;
      line-height: 1.6;
      font-size: 0.93rem;
    }
    .user .bubble {
      background: #1e3a5f;
      color: #ddeeff;
      border-bottom-right-radius: 4px;
    }
    .assistant .bubble {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      color: #e0e0e0;
      border-bottom-left-radius: 4px;
    }

    /* Markdown-ish rendering */
    .bubble h1, .bubble h2, .bubble h3 {
      margin: 12px 0 6px;
      color: #fff;
    }
    .bubble p { margin: 6px 0; }
    .bubble ol, .bubble ul { padding-left: 20px; margin: 8px 0; }
    .bubble li { margin: 4px 0; }
    .bubble strong { color: #fff; }
    .bubble code {
      background: #252525;
      padding: 1px 5px;
      border-radius: 3px;
      font-family: monospace;
      font-size: 0.88em;
    }
    .bubble pre {
      background: #252525;
      padding: 10px;
      border-radius: 6px;
      overflow-x: auto;
      margin: 8px 0;
    }
    .bubble blockquote {
      border-left: 3px solid #555;
      padding-left: 12px;
      color: #aaa;
      margin: 8px 0;
    }

    /* Citations */
    .citations {
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .citation-tag {
      background: #1e2a1e;
      border: 1px solid #2a4a2a;
      color: #6fcf6f;
      font-size: 0.72rem;
      padding: 3px 8px;
      border-radius: 4px;
      font-family: monospace;
      cursor: default;
    }
    .citation-tag:hover { background: #253525; }

    /* Figures */
    .figures {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }
    .figure-card {
      border: 1px solid #333;
      border-radius: 8px;
      overflow: hidden;
      max-width: 280px;
      background: #1a1a1a;
    }
    .figure-card img {
      width: 100%;
      display: block;
      cursor: pointer;
    }
    .figure-caption {
      padding: 6px 10px;
      font-size: 0.78rem;
      color: #999;
    }

    /* Lightbox */
    #lightbox {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.92);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }
    #lightbox.active { display: flex; }
    #lightbox img { max-width: 90vw; max-height: 90vh; border-radius: 4px; }

    /* Input area */
    #input-area {
      border-top: 1px solid #222;
      padding: 16px 24px;
      background: #0f0f0f;
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }
    #query {
      flex: 1;
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 8px;
      padding: 12px 14px;
      color: #e0e0e0;
      font-size: 0.93rem;
      resize: none;
      min-height: 46px;
      max-height: 160px;
      outline: none;
      font-family: inherit;
      line-height: 1.5;
    }
    #query:focus { border-color: #555; }
    #send-btn {
      background: #1e4a8a;
      border: none;
      color: #fff;
      padding: 12px 20px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 600;
      white-space: nowrap;
      height: 46px;
    }
    #send-btn:hover { background: #2255a0; }
    #send-btn:disabled { background: #333; color: #666; cursor: default; }

    .typing {
      color: #666;
      font-style: italic;
      font-size: 0.88rem;
      padding: 10px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
  </style>
</head>
<body>

<header>
  <h1>1990 Geo Metro — Service Manual</h1>
  <span>Factory manual Q&amp;A</span>
  <div class="badge">RAG + Claude Vision</div>
</header>

<div id="messages">
  <div class="message assistant">
    <div class="bubble">
      <p>Hi! I'm your factory service manual assistant for the <strong>1990 Geo Metro / Suzuki Swift</strong>.</p>
      <p>Ask me anything — troubleshooting, specifications, procedures, torque values, wiring, or part locations. I'll answer directly from the manual and cite every source.</p>
      <p><em>Example: "How do I replace the filter drier on the AC system?"</em></p>
    </div>
  </div>
</div>

<div id="lightbox">
  <img id="lightbox-img" src="" alt="">
</div>

<div id="input-area">
  <textarea id="query" placeholder="Ask a question about the Geo Metro..." rows="1"></textarea>
  <button id="send-btn">Send</button>
</div>

<script>
  const messagesEl = document.getElementById("messages");
  const queryEl    = document.getElementById("query");
  const sendBtn    = document.getElementById("send-btn");
  const lightbox   = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");

  let conversation = [];

  // Auto-resize textarea
  queryEl.addEventListener("input", () => {
    queryEl.style.height = "auto";
    queryEl.style.height = Math.min(queryEl.scrollHeight, 160) + "px";
  });

  // Send on Enter (Shift+Enter for newline)
  queryEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener("click", sendMessage);

  lightbox.addEventListener("click", () => lightbox.classList.remove("active"));

  function showLightbox(url) {
    lightboxImg.src = url;
    lightbox.classList.add("active");
  }

  function appendMessage(role, content) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.innerHTML = `<div class="bubble">${content}</div>`;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function renderMarkdown(text) {
    // Minimal markdown → HTML (no external lib dependency)
    return text
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/^## (.+)$/gm, "<h2>$1</h2>")
      .replace(/^# (.+)$/gm, "<h1>$1</h1>")
      .replace(/^\d+\. (.+)$/gm, (_, t) => `<li>${t}</li>`)
      .replace(/^[-*] (.+)$/gm, (_, t) => `<li>${t}</li>`)
      .replace(/(<li>.*<\/li>\n?)+/g, m => `<ol>${m}</ol>`)
      .replace(/\n\n+/g, "</p><p>")
      .replace(/\n/g, "<br>")
      .replace(/^(?!<[hol])(.+)$/, "<p>$1</p>");
  }

  function buildCitations(citations) {
    if (!citations.length) return "";
    const tags = citations.map(c =>
      `<span class="citation-tag" title="${c.section_path}">p${c.page} ${c.source_label || ""} | ${c.chunk_id}</span>`
    ).join("");
    return `<div class="citations">${tags}</div>`;
  }

  function buildFigures(figures) {
    if (!figures.length) return "";
    const cards = figures.map(f => `
      <div class="figure-card">
        <img src="${f.url}" alt="${f.caption_text || ""}"
             onclick="showLightbox('${f.url}')">
        <div class="figure-caption">${f.caption_text || "Figure"} — p${f.page}</div>
      </div>
    `).join("");
    return `<div class="figures">${cards}</div>`;
  }

  async function sendMessage() {
    const q = queryEl.value.trim();
    if (!q) return;

    queryEl.value = "";
    queryEl.style.height = "auto";
    sendBtn.disabled = true;

    appendMessage("user", renderMarkdown(q));
    conversation.push({ role: "user", text: q });

    const typingDiv = appendMessage("assistant", '<span class="typing">Searching manual...</span>');

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, conversation }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();

      const answerHTML = renderMarkdown(data.answer);
      const citeHTML   = buildCitations(data.citations || []);
      const figHTML    = buildFigures(data.figures || []);

      typingDiv.querySelector(".bubble").innerHTML = answerHTML + citeHTML + figHTML;

      conversation.push({ role: "assistant", text: data.answer });
    } catch (err) {
      typingDiv.querySelector(".bubble").innerHTML =
        `<span style="color:#e05a5a">Error: ${err.message}</span>`;
    }

    sendBtn.disabled = false;
    queryEl.focus();
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
