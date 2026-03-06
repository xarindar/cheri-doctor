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
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

    fig_path = PROJECT_ROOT / "build" / "figures.jsonl"
    if fig_path.exists():
        for fig in load_jsonl(fig_path):
            if fig.get("figure_id"):
                fig_lookup[fig["figure_id"]] = fig
    print(f"Figures loaded: {len(fig_lookup)}")


# ── API models ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
  query: str
  conversation: list[dict] = []
  model: str | None = None


class WorksheetRequest(BaseModel):
  prompt: str  # e.g. "alternator diagnostic tests"


class ChatAPIResponse(BaseModel):
    answer: str
    citations: list[dict]
    figure_refs: list[str]
    figures: list[dict]  # {figure_id, page, caption_text, url}


# ── Routes ────────────────────────────────────────────────────────────────


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

  response = chat(
    query=req.query,
    conversation=req.conversation,
    index=index,
    config=chat_config,
    skip_vision=False,
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
    "figure_refs": response.figure_refs,
    "figures":     figures_out,
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
