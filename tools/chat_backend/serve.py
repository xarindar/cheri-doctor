"""Chat backend web server.

FastAPI app serving:
  GET  /              — Chat UI (HTML)
  POST /api/chat      — Chat endpoint
  GET  /api/figures/{figure_id}  — Serve figure images

Usage:
  python -m tools.chat_backend.serve
  uvicorn tools.chat_backend.serve:app --host 0.0.0.0 --port 8000
"""

import sys
import json
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, RedirectResponse
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

# ── Authentication ────────────────────────────────────────────────────────
_PASSWORD_HASH = hashlib.sha256(b"maytoe").hexdigest()
_AUTH_SECRET = os.environ.get("AUTH_SECRET", secrets.token_hex(32))
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
config     = load_config(PROJECT_ROOT / "configs" / "default.yaml")
index      = None
fig_lookup: dict[str, dict] = {}


@app.on_event("startup")
async def startup():
    global index, fig_lookup
    print("Loading retrieval index...")
    index = load_index(config)
    print(f"Index loaded: {len(index.chunk_ids)} chunks")

    fig_sources = [
        PROJECT_ROOT / "build" / "figures.jsonl",
        PROJECT_ROOT / "build_supplement" / "figures.jsonl",
    ]
    for fig_path in fig_sources:
        if fig_path.exists():
            before = len(fig_lookup)
            for fig in load_jsonl(fig_path):
                if fig.get("figure_id"):
                    fig_lookup[fig["figure_id"]] = fig
            print(f"  Loaded {len(fig_lookup) - before} figures from {fig_path.parent.name}/figures.jsonl")
    print(f"Figures loaded: {len(fig_lookup)} total")


# ── Web fetch / search / RockAuto utilities ───────────────────────────────
import re as _re
import requests as _requests
from bs4 import BeautifulSoup as _BS

_URL_RE    = _re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
_RA_URL_RE = _re.compile(r'https?://(?:www\.)?rockauto\.com(/en/catalog/[^\s<>"{}|\\^`\[\]?#]+)')
_SEARCH_RE = _re.compile(
    r'\b(search\s+(for|online|the\s+web)?|look\s+up\s+online|find\s+online|google\s+(this|for)?)\b',
    _re.IGNORECASE,
)
_RA_SEARCH_RE = _re.compile(r'\brock\s*auto\b', _re.IGNORECASE)
_BUY_INTENT_RE = _re.compile(
    r'\b('
    # explicit purchase intent
    r'buy|purchase|order\s+online|shop\s+for|'
    # "find me a/the/an X", "find a X", "find the X", "can you find X"
    r'(?:can\s+you\s+)?find\s+(?:me\s+)?(?:a|an|the|one)(\s|$)|'
    # "find it/one online" or "find X to buy"
    r'find\s+(?:it\s+|one\s+|a\s+|an\s+)?(?:online|to\s+buy|for\s+(?:me\s+to\s+buy|purchase))|'
    # "look up the X", "look for a X"
    r'look\s+(?:up|for)\s+(?:a|an|the|one)(\s|$)|'
    # "search for a X"
    r'search\s+for\s+(?:a|an|the)(\s|$)|'
    # "source a X"
    r'source\s+(?:a|an|the)\s|'
    r'looking\s+to\s+buy|want\s+to\s+buy|'
    r'where\s+(?:can\s+I|do\s+I)\s+(?:buy|get|order|find)|'
    r'get\s+(?:one|it)\s+online|for\s+sale\s+online'
    r')',
    _re.IGNORECASE,
)
_AFFIRMATIVE_RE = _re.compile(
    r'^\s*(yes|yeah|yep|yup|sure|please|go ahead|do it|please do|ok|okay|'
    r'sounds good|let\'?s (do it|go)|why not|absolutely|of course|definitely|'
    r'go for it|do that|pull it up|look it up|check it out|check that)\s*[.!]?\s*$',
    _re.IGNORECASE,
)
# Common automotive parts for extracting search context from conversation
_AUTO_PARTS = [
    "battery", "alternator", "starter", "spark plug", "spark plugs",
    "ignition", "timing belt", "serpentine belt", "drive belt", "belt",
    "brake pad", "brake rotor", "caliper", "brake", "rotor",
    "oil filter", "air filter", "fuel filter", "cabin filter", "filter",
    "water pump", "fuel pump", "pump",
    "radiator", "thermostat", "coolant hose", "hose",
    "shock absorber", "strut", "coil spring", "control arm", "ball joint",
    "cv axle", "axle shaft", "axle", "wheel bearing", "bearing", "seal",
    "clutch", "transmission", "transaxle",
    "distributor cap", "distributor", "ignition wire", "plug wire",
    "oxygen sensor", "o2 sensor", "sensor",
    "tie rod", "wheel", "tire",
]

def _extract_part_from_conversation(conversation: list[dict]) -> str:
    """Scan recent conversation messages for an automotive part name."""
    for msg in reversed(conversation[-8:]):
        text = (msg.get("text") or "").lower()
        for part in _AUTO_PARTS:
            if part in text:
                return part
    return ""


def _extract_part_from_bold(conversation: list[dict]) -> str:
    """Extract a part name from **bold** terms in the most recent assistant message.
    Used as a fallback when _AUTO_PARTS matching fails (e.g. interior trim pieces)."""
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


async def _extract_part_from_description(query: str, conversation: list[dict]) -> str:
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
    manual_context = _rag_context_for_part(retrieval_query)
    context_block = ""
    if manual_context:
        context_block = (
            f"\nRELEVANT SERVICE MANUAL EXCERPTS:\n"
            f'"""\n{manual_context}\n"""\n'
        )

    prompt = (
        f"A user is describing an auto part they want to find/buy for a 1990 Geo Metro.\n"
        f"Recent conversation:\n{recent}\n"
        f"User's latest message: {query}\n"
        f"{context_block}\n"
        f"What auto part are they describing? Use the manual excerpts above to "
        f"identify the exact part. Reply with ONLY a short part name "
        f"(2-5 words) suitable for an eBay search — nothing else, no explanation.\n"
        f"Examples of good replies: 'console box door'  'console box lid'  "
        f"'console compartment cover'  'oil filter'  'alternator'"
    )
    client = _ant.AsyncAnthropic()
    resp = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().lower()


_FETCH_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Cheri's fixed vehicle identifiers
_RA_MAKE    = 'GEO'
_RA_YEAR    = 1990
_RA_MODEL   = 'METRO'
_RA_CARCODE = '1430189'  # 993cc L3

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

_SHOPHOP_URL    = "http://localhost:8001/hunt/auto"
_CHERI_VEHICLE  = {"year": 1990, "make": "Geo", "model": "Metro", "engine": "993cc L3"}


def _rag_context_for_part(part_name: str) -> str | None:
    """Retrieve the top 3 manual chunks for a part and return them as a text block.

    This gives ShopHop's Haiku expander real OEM part numbers, cross-references,
    and manual terminology — making its query expansion far more targeted than
    generic layperson guessing.
    """
    if index is None:
        return None
    try:
        chunks = index.retrieve(part_name, top_k=3)
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


async def _parts_search(part_name: str) -> tuple[str, list[str], list[dict]]:
    """Call ShopHop for parts; return (llm_text, sources, shopping_items)."""
    import httpx as _httpx

    # Pull manual context before calling ShopHop so the query expander
    # has OEM part numbers and cross-references to work with.
    manual_context = _rag_context_for_part(part_name)

    try:
        payload = {
            "part": part_name,
            "vehicle": _CHERI_VEHICLE,
            "max_results": 20,
        }
        if manual_context:
            payload["context"] = manual_context

        async with _httpx.AsyncClient() as hc:
            r = await hc.post(
                _SHOPHOP_URL,
                json=payload,
                timeout=35,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return f"[Parts search unavailable: {exc}]", [], []

    items            = data.get("items", [])
    sources_searched = data.get("sources_searched", [])
    queries_used     = data.get("queries_used", [])
    source_errors    = data.get("source_errors", {})

    sources = [f"parts search ({', '.join(sources_searched)}): {part_name}"]

    if not items:
        llm_text = (
            f"[PARTS SEARCH: no results for '{part_name}' across "
            f"{', '.join(sources_searched) or 'all sources'}]"
        )
        return llm_text, sources, []

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
    if source_errors:
        lines.append(f"[Search errors: {source_errors}]")
    lines.append("[END PARTS SEARCH]")
    llm_text = "\n".join(lines)

    return llm_text, sources, items


def _extract_ra_part_name(query: str, conversation: list[dict] | None = None) -> str:
    """Extract the auto part name from a query mentioning RockAuto."""
    # Scan the query directly for a known part name first (most reliable)
    q_lower = query.lower()
    for part in _AUTO_PARTS:
        if part in q_lower:
            return part
    # Fall back to stripping trigger phrases and using whatever's left
    q = _RA_SEARCH_RE.sub('', query).strip()
    q = _re.sub(
        r'^(and\s+)?(see\s+if\s+(you\s+)?(can\s+)?|try\s+to\s+|please\s+|can\s+you\s+)?'
        r'(find|look\s+(?:up|for)|search\s+for|locate|source)\s+',
        '', q, flags=_re.IGNORECASE,
    ).strip()
    # strip leading articles/pronouns left after verb removal
    q = _re.sub(r'^(for\s+|a\s+|an\s+|the\s+|some\s+|me\s+|one\s+)', '', q, flags=_re.IGNORECASE).strip()
    q = q.strip(' .,?!')
    if q and len(q.split()) <= 5:
        return q
    # Last resort: scan conversation context
    if conversation:
        return _extract_part_from_conversation(conversation)
    return ""

_RA_CAT_MAP_UNUSED: dict[str, str] = {  # kept for reference, now in shophop/sources/rockauto.py
    # electrical
    "battery":          "electrical",
    "alternator":       "electrical",
    "generator":        "electrical",
    "starter":          "electrical",
    "fuse":             "electrical",
    "horn":             "electrical",
    "flasher":          "electrical",
    "voltage regulator":"electrical",
    # belt & drive
    "belt":             "belt+drive",
    "tensioner":        "belt+drive",
    "idler":            "belt+drive",
    # brakes
    "brake":            "brake+%26+wheel+hub",
    "rotor":            "brake+%26+wheel+hub",
    "caliper":          "brake+%26+wheel+hub",
    "wheel hub":        "brake+%26+wheel+hub",
    # cooling
    "coolant":          "cooling+system",
    "antifreeze":       "cooling+system",
    "radiator":         "cooling+system",
    "thermostat":       "cooling+system",
    "water pump":       "cooling+system",
    "hose":             "cooling+system",
    "cooling fan":      "cooling+system",
    # drivetrain
    "cv axle":          "drivetrain",
    "axle shaft":       "drivetrain",
    "cv joint":         "drivetrain",
    "differential":     "drivetrain",
    # engine
    "timing belt":      "engine",
    "oil filter":       "engine",
    "oil pan":          "engine",
    "valve cover":      "engine",
    "gasket":           "engine",
    "motor mount":      "engine",
    "oil pump":         "engine",
    "piston":           "engine",
    "camshaft seal":    "engine",
    "crankshaft seal":  "engine",
    # ignition
    "spark plug":       "ignition",
    "distributor":      "ignition",
    "plug wire":        "ignition",
    "ignition coil":    "ignition",
    "ignition wire":    "ignition",
    "tune":             "ignition",
    # fuel & air
    "fuel filter":      "fuel+%26+air",
    "fuel pump":        "fuel+%26+air",
    "air filter":       "fuel+%26+air",
    "carburetor":       "fuel+%26+air",
    "throttle":         "fuel+%26+air",
    # exhaust
    "exhaust":          "exhaust+%26+emission",
    "muffler":          "exhaust+%26+emission",
    "oxygen sensor":    "exhaust+%26+emission",
    "o2 sensor":        "exhaust+%26+emission",
    "catalytic":        "exhaust+%26+emission",
    # steering
    "tie rod":          "steering",
    "rack":             "steering",
    # suspension
    "shock":            "suspension",
    "strut":            "suspension",
    "control arm":      "suspension",
    "ball joint":       "suspension",
    "sway bar":         "suspension",
    "spring":           "suspension",
    # transmission
    "transmission":     "transmission-automatic",
    "trans fluid":      "transmission-automatic",
    "torque converter": "transmission-automatic",
    "clutch":           "transmission-manual",
    "flywheel":         "transmission-manual",
    # heat & AC
    "ac":               "heat+%26+air+conditioning",
    "heater":           "heat+%26+air+conditioning",
    "blower":           "heat+%26+air+conditioning",
    "compressor":       "heat+%26+air+conditioning",
    # interior
    "wiper":            "wiper+%26+washer",
    "washer":           "wiper+%26+washer",
}



async def _identify_part_for_search(query: str, conversation: list[dict] | None) -> str:
    """Single source of truth for part name extraction before a shopping search.

    Priority:
    1. Bold term from the last assistant message — it's already RAG-informed
       from the previous turn, so it names things correctly (e.g. 'console box door').
    2. Haiku + RAG — give Haiku the manual chunks for this query so it can
       identify the part from what the service manual actually calls it.

    Intentionally skips the old regex/_AUTO_PARTS chain — it was too brittle
    (returned 'console' instead of 'console box door', 'hose' instead of
    'coolant hose', etc.) and provided no real benefit over Haiku+RAG.
    """
    if conversation:
        bold = _extract_part_from_bold(conversation)
        if bold:
            return bold
    try:
        return await _extract_part_from_description(query, conversation or [])
    except Exception:
        return ""


async def _enrich_query(query: str, conversation: list[dict] | None = None) -> tuple[str, list[str], list[dict]]:
    """Enrich the query with fetched URLs, web search, and parts search results.
    Everything runs BEFORE the LLM so the LLM can see and reference the results.
    Returns (enriched_query, sources, shopping_items).
    """
    extra, sources, shopping_items = [], [], []
    urls = _URL_RE.findall(query)

    for url in urls[:3]:
        ra_match = _RA_URL_RE.match(url)
        if ra_match:
            try:
                content = await _rockauto_from_url(ra_match.group(1))
                extra.append(f"[ROCKAUTO PARTS LISTING]\n{content}\n[END ROCKAUTO]")
                sources.append(f"RockAuto: {url}")
            except Exception as exc:
                extra.append(f"[RockAuto fetch failed: {exc}]")
        else:
            try:
                content = _fetch_url(url)
                extra.append(f"[WEB PAGE CONTENT FROM {url}]\n{content}\n[END WEB PAGE CONTENT]")
                sources.append(url)
            except Exception as exc:
                extra.append(f"[Could not fetch {url}: {exc}]")

    if not urls:
        if _RA_SEARCH_RE.search(query) or _BUY_INTENT_RE.search(query):
            part_q = await _identify_part_for_search(query, conversation)
            if part_q:
                try:
                    llm_text, src, items = await _parts_search(part_q)
                    extra.append(llm_text)
                    sources.extend(src)
                    shopping_items = items
                except Exception as exc:
                    extra.append(f"[Parts search failed: {exc}]")

        elif _SEARCH_RE.search(query):
            search_q = _SEARCH_RE.sub('', query).strip(' .,?!')
            if search_q:
                try:
                    results = _web_search(search_q)
                    extra.append(f"[WEB SEARCH RESULTS FOR: {search_q}]\n{results}\n[END WEB SEARCH RESULTS]")
                    sources.append(f"web search: {search_q}")
                except Exception as exc:
                    extra.append(f"[Web search failed: {exc}]")

        elif _AFFIRMATIVE_RE.match(query) and conversation:
            last_asst = next(
                (m.get("text", "") for m in reversed(conversation) if m.get("role") == "assistant"),
                "",
            )
            if any(kw in last_asst.lower() for kw in ("rockauto", "ebay", "parts", "find", "buy", "source")):
                part_q = await _identify_part_for_search(query, conversation)
                if part_q:
                    try:
                        llm_text, src, items = await _parts_search(part_q)
                        extra.append(llm_text)
                        sources.extend(src)
                        shopping_items = items
                    except Exception as exc:
                        extra.append(f"[Parts search failed: {exc}]")

    enriched = "\n\n".join(extra) + "\n\n" + query if extra else query
    return enriched, sources, shopping_items

    return sources, shopping_items


# ── API models ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
  query: str
  conversation: list[dict] = []
  model: str | None = None
  project_context: str | None = None
  images: list[str] = []  # base64 data URLs from the frontend


class TitleRequest(BaseModel):
  message: str


class WorksheetRequest(BaseModel):
  prompt: str  # e.g. "alternator diagnostic tests"


class ChatAPIResponse(BaseModel):
    answer: str
    citations: list[dict]
    figure_refs: list[str]
    figures: list[dict]  # {figure_id, page, caption_text, url}


# ── Routes ────────────────────────────────────────────────────────────────


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


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> JSONResponse:
  if index is None:
    raise HTTPException(503, "Index not loaded yet")

  # Allow model override from frontend
  chat_config = dict(config)
  if req.model:
    chat_config = dict(config)
    chat_config = chat_config.copy()
    chat_config["chat"] = dict(chat_config.get("chat", {}))
    chat_config["chat"]["model"] = req.model

  vehicle_settings = _build_settings_context(_load_settings())

  # Enrich query (URL fetch, web search, parts search) then run LLM.
  # Parts search runs first so the LLM can see and reference the results.
  enriched_query, web_sources, shopping_results = await _enrich_query(req.query, req.conversation)

  response = chat(
    query=enriched_query,
    conversation=req.conversation,
    index=index,
    config=chat_config,
    skip_vision=False,
    project_context=req.project_context,
    vehicle_settings=vehicle_settings,
    images=req.images,
  )

  # Build figure metadata for the UI
  figures_out = []
  for fig_id in response.figure_refs:
    fig = fig_lookup.get(fig_id, {})
    if fig:
      figures_out.append({
        "figure_id":    fig_id,
        "page":         fig.get("page"),
        "caption_text": fig.get("caption_text", ""),
        "url":          f"/api/figures/{fig_id}",
      })

  return JSONResponse({
    "answer":      response.answer,
    "citations":   [
      {
        "chunk_id":     c.chunk_id,
        "page":         c.page,
        "source_label": c.source_label,
        "section_path": c.section_path,
      }
      for c in response.citations
    ],
    "figure_refs":       response.figure_refs,
    "figures":           figures_out,
    "web_sources":       web_sources,
    "shopping_results":  shopping_results,
  })


@app.post("/api/generate-worksheet")
async def api_generate_worksheet(req: WorksheetRequest):
  """Generate a PDF diagnostic worksheet from a user prompt."""
  if index is None:
    raise HTTPException(503, "Index not loaded yet")
  try:
    from tools.worksheet_generator import generate_worksheet_pdf
    pdf_bytes = generate_worksheet_pdf(req.prompt, index, config)
    
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
async def serve_figure(figure_id: str):
    fig = fig_lookup.get(figure_id)
    if not fig:
        raise HTTPException(404, f"Figure {figure_id} not found")
    asset_path = PROJECT_ROOT / fig.get("asset_path", "")
    if not asset_path.exists():
        raise HTTPException(404, "Image file not found")
    media_type = "image/webp" if asset_path.suffix == ".webp" else "image/png"
    return FileResponse(str(asset_path), media_type=media_type)


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
