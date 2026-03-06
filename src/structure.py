"""Stage E: Structure Reconstruction.

Walks all pages and blocks in document.json to:
- Detect heading levels from typography cues
- Build a running section stack (chapter > section > subsection)
- Assign section_path and source_label to every block
- Extract page-level metadata (topics, figure refs, cross-refs)
"""

import re
from typing import Any

SECTION_CODE_RE = re.compile(r"\b([0-9]{1,2}[A-Z]\d{0,2})-(?:[A-Z]\d*-)?([0-9]+)\b")
# Broader pattern to extract a section label from page_label
# Matches: "7B-11", "6E2-A-60", "6E2-C1-2", "10-8-2", "5-18"
# Word boundaries prevent matching inside document numbers (e.g. "7701-6E")
PAGE_LABEL_RE = re.compile(r"\b((?:\d{1,2}[A-Z]\d{0,2}|\d{1,2})(?:-[A-Z]\d*)?-\d+(?:-\d+)*)\b")

# Reject publication dates mis-parsed as section labels (e.g. "5-10-89" = May 10, 1989)
_DATE_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{2})$")

def _is_date_label(label: str) -> bool:
    """Return True if label looks like a MM-DD-YY date (YY >= 70)."""
    m = _DATE_RE.match(label)
    return bool(m) and int(m.group(3)) >= 70


_LEGACY_SECTION_CODE_RE = re.compile(r'\b([O0-9]+[A-Z]\d*)-(\d+)\b')


def _detect_section(text: str) -> dict:
    """Extract section code from block text (first/last 3 lines).

    Inlined from legacy/extract.py to avoid symspellpy import chain.
    """
    if not text:
        return {"section_code": None, "section_page": None}
    lines = text.strip().splitlines()
    if not lines:
        return {"section_code": None, "section_page": None}
    for line in (lines[:3] + lines[-3:]):
        m = _LEGACY_SECTION_CODE_RE.search(line.strip())
        if m:
            return {
                "section_code": m.group(1).replace('O', '0'),
                "section_page": m.group(2),
            }
    return {"section_code": None, "section_page": None}


def build_structure(pages: list[dict], config: dict):
    """Assign section_path and source_label to all blocks across all pages.

    Modifies pages list in-place.
    """
    import assembler as _assembler

    section_names = config.get("structure", {}).get("section_names", {})
    stack = SectionStack(section_names)

    for page in pages:
        page_num = page["page_num"]

        # Try to extract section info from page header/footer blocks first
        for block in page["blocks"]:
            if block.get("type") in ("heading", "paragraph"):
                info = _detect_section(block.get("text") or "")
                if info["section_code"]:
                    stack.update_from_section_code(
                        info["section_code"],
                        info["section_page"],
                        page_num,
                    )
                    break

        # Assign section_path to every block
        for block in page["blocks"]:
            block["section_code"]  = stack.current_code()
            block["source_label"]  = stack.current_label()
            block["section_path"]  = stack.current_path()

            # Push headings onto the stack
            if block["type"] == "heading":
                stack.push(block.get("text") or "", block.get("level", 2), page_num)

        # Page-level section info — prefer page_label (per-page) over stack label (per-section)
        page["section_code"]  = stack.current_code()
        page["section_path"]  = stack.current_path()

        page_label = page.get("page_label", "")
        pl_match = PAGE_LABEL_RE.search(page_label) if page_label else None
        if pl_match and _is_date_label(pl_match.group(1)):
            pl_match = None  # Reject publication dates (e.g. "5-10-89")
        if pl_match:
            extracted_label = pl_match.group(1)

            # Check if label has a proper section code (letter component,
            # e.g. "7C-3", "4D-1", "6E2-A-60").  SECTION_CODE_RE requires a
            # letter after digits, so "5-18" and "10-2-8" won't match.
            sec_match = SECTION_CODE_RE.search(extracted_label)

            if sec_match:
                # Proper section-page label — trust it and update the stack.
                # This handles section transitions (e.g. 3E -> 4D).
                stack.update_from_section_code(
                    sec_match.group(1), sec_match.group(2), page_num
                )
                page["section_code"] = stack.current_code()
                page["section_path"] = stack.current_path()
                page["source_label"] = extracted_label
                for block in page["blocks"]:
                    block["section_code"] = stack.current_code()
                    block["section_path"] = stack.current_path()
                    block["source_label"] = extracted_label
            else:
                # Numeric-only label (no letter, e.g. "5-18", "10-2-8").
                # Parse prefix as the section code.
                label_prefix = extracted_label.split('-')[0]
                label_page = extracted_label.split('-', 1)[1] if '-' in extracted_label else None
                stack_code = stack.current_code()

                if stack_code is None or label_prefix != stack_code:
                    # Section transition (e.g. "4D" -> "5") — update the stack
                    stack.update_from_section_code(
                        label_prefix, label_page, page_num
                    )
                    page["section_code"] = stack.current_code()
                    page["section_path"] = stack.current_path()

                page["source_label"] = extracted_label
                for block in page["blocks"]:
                    block["section_code"] = stack.current_code()
                    block["source_label"] = extracted_label
                    block["section_path"] = stack.current_path()
        else:
            page["source_label"] = stack.current_label()

        # Page-level metadata extraction
        all_text = " ".join(
            b.get("text") or "" for b in page["blocks"] if b.get("text")
        )
        page["topics"]           = _assembler._extract_topics(all_text)
        page["figures_referenced"] = _assembler._extract_figure_refs(all_text)
        page["cross_references"] = _assembler._extract_cross_refs(all_text)


class SectionStack:
    """Tracks the current heading hierarchy across pages."""

    def __init__(self, section_names: dict[str, str]):
        self.section_names = section_names
        self._code: str | None = None
        self._page: str | None = None
        self._subsection: str | None = None

    def update_from_section_code(self, code: str, page: str | None, page_num: int):
        """Update current section when a section code is detected."""
        self._code = code
        self._page = page
        self._subsection = None  # Reset subsection on new section

    def push(self, heading_text: str, level: int, page_num: int):
        """Push a heading onto the stack."""
        # Extract section code from heading text if present
        m = SECTION_CODE_RE.search(heading_text)
        if m:
            self._code = m.group(1)
            self._page = m.group(2)
            self._subsection = None
        elif level <= 2:
            # Level 2 heading resets subsection
            self._subsection = heading_text.strip()
        elif level == 3:
            self._subsection = heading_text.strip()

    def current_code(self) -> str | None:
        return self._code

    def current_label(self) -> str | None:
        if self._code and self._page:
            return f"{self._code}-{self._page}"
        return self._code

    def current_path(self) -> str:
        parts = []
        if self._code:
            name = self.section_names.get(self._code, self._code)
            parts.append(name)
        if self._subsection:
            parts.append(self._subsection)
        return " > ".join(parts) if parts else "Unknown"
