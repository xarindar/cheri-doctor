"""Generate diagnostic/test worksheets as PDFs.

Uses RAG to pull relevant manual content, then an LLM to produce
structured worksheet steps with result fields. Renders to PDF via fpdf2.

Usage:
  from tools.worksheet_generator import generate_worksheet_pdf
  pdf_bytes = generate_worksheet_pdf("alternator diagnostic tests", index, config)
"""

import json
import os
import re
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file explicitly
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.gemini_api import call_gemini
from src.utils import load_config

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None


WORKSHEET_SYSTEM_PROMPT = """You are Cheri Doctor — an expert mechanic for a 1990 Geo Metro (G10 3-cylinder, automatic transaxle). You are generating a **diagnostic worksheet** that the owner will print, take to the garage, perform tests, fill in results, and bring back for further troubleshooting.

Use the EVIDENCE from the factory service manual. Every test step and spec must come from the evidence when available. Fill gaps with your mechanic expertise.

Output ONLY valid JSON in this exact format (no markdown, no code fence, no extra text):
{
  "title": "Short worksheet title (e.g. Alternator Diagnostic Worksheet)",
  "subtitle": "1990 Geo Metro · G10 3-cyl",
  "intro": "Optional 1-2 sentence intro for the owner.",
  "steps": [
    {
      "number": 1,
      "instruction": "Clear step-by-step instruction. Include specs from the manual when relevant (voltage, resistance, etc.).",
      "result_field": "What to record (e.g. Voltage reading, Belt condition)"
    }
  ]
}

Rules:
- 5–15 steps. Each step: one clear instruction + one result field to fill in.
- Result fields should be short labels (e.g. "Voltage (V)", "Resistance (Ω)", "Pass/Fail").
- Include safety notes (CAUTION/WARNING) from the manual when relevant.
- Cite specs exactly from the evidence (torque values, clearances, voltages).
- Output ONLY the JSON object, nothing else."""


def _retrieve_evidence(prompt: str, index, config: dict) -> str:
    """Retrieve relevant chunks and format as evidence text."""
    search_query = prompt.lower()
    if "worksheet" in search_query or "generate" in search_query:
        # Strip meta words for better retrieval
        for w in ["worksheet", "generate", "me", "a", "for", "tests", "diagnostic"]:
            search_query = search_query.replace(w, " ")
        search_query = " ".join(search_query.split())

    results = index.retrieve(search_query or "diagnostic test", top_k=15, engine_variant="G10")
    evidence_lines = []
    for r in results[:12]:
        chunk = r["chunk"]
        header = (
            f"--- {chunk['chunk_id']} | page: {chunk['page']} | "
            f"type: {chunk['type']} ---"
        )
        evidence_lines.append(header)
        evidence_lines.append(chunk.get("text", ""))
        evidence_lines.append("")
    return "\n".join(evidence_lines)


def _call_worksheet_llm(prompt: str, evidence: str, config: dict) -> dict:
    """Call Gemini to generate structured worksheet JSON."""
    cfg = config.get("chat", {})
    model = cfg.get("model", "gemini-2.5-flash")
    api_key = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or config")

    user_content = (
        f"EVIDENCE FROM MANUAL:\n\n{evidence}\n\n"
        f"Generate a diagnostic worksheet for: {prompt}"
    )
    messages = [{
        "system": WORKSHEET_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }]
    
    try:
        raw = call_gemini(messages, api_key=api_key, model=model)
        # Debug: print raw response to help troubleshoot
        print(f"DEBUG: Raw LLM response: {raw[:200]}...")

        # Extract JSON (handle markdown code block if LLM wraps it)
        raw = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in raw text
            start = raw.find("{")
            if start >= 0:
                depth = 0
                for i, c in enumerate(raw[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(raw[start : i + 1])
            raise ValueError(f"Could not parse worksheet JSON from LLM response. Raw response: {raw[:500]}")
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")


def _render_pdf(data: dict) -> bytes:
    """Render worksheet data to PDF bytes using fpdf2."""
    if FPDF is None:
        raise RuntimeError("fpdf2 not installed. Run: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    w = pdf.epw  # effective page width (full width minus margins)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(w, 10, data.get("title", "Diagnostic Worksheet"), align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(w, 6, data.get("subtitle", "1990 Geo Metro"), align="C")
    pdf.ln(4)

    intro = data.get("intro", "").strip()
    if intro:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(w, 6, intro)
        pdf.ln(6)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    steps = data.get("steps", [])
    for s in steps:
        num = s.get("number", 0)
        instruction = s.get("instruction", "")
        result_field = s.get("result_field", "Result")

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(w, 7, f"Step {num}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(w, 6, instruction)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(40, 6, f"{result_field}:")
        pdf.set_draw_color(180, 180, 180)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, "", border="B", ln=True)
        pdf.ln(4)

    pdf.ln(6)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(w, 5, "Generated by Cheri Doctor · Geo Metro Manual Chat · Bring this sheet back for further troubleshooting.")

    # Ensure we return bytes, not bytearray
    output = pdf.output()
    if isinstance(output, bytearray):
        return bytes(output)
    return output


def generate_worksheet_pdf(prompt: str, index, config: dict) -> bytes:
    """Generate a PDF worksheet from a user prompt.

    Args:
        prompt: User request (e.g. "alternator diagnostic tests")
        index: RetrievalIndex from chat.load_index()
        config: Loaded config dict

    Returns:
        PDF file as bytes
    """
    evidence = _retrieve_evidence(prompt, index, config)
    data = _call_worksheet_llm(prompt, evidence, config)
    return _render_pdf(data)
