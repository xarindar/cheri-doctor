"""Shared data models for the Metro Manual pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Manifest:
    """Pipeline run manifest — saved as build/manifest.json."""
    doc_id: str
    source_pdf: str
    source_hash: str = ""
    dpi: int = 350
    page_count: int = 0
    run_id: str = ""
    config_hash: str = ""
    timestamps: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


@dataclass
class PreprocessResult:
    """Result of preprocessing a single page."""
    page_num: int
    preprocessed_path: str
    content_bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    skew_angle: float
    original_size: tuple[int, int]  # width, height


@dataclass
class TextBlock:
    """A classified block of text from OCR output."""
    type: str       # heading, paragraph, ordered_list, caution, warning, note
    text: str
    level: int | None = None         # for headings (1, 2, 3)
    steps: list[str] | None = None   # for ordered_list
    procedure_type: str | None = None  # removal, installation, inspection, adjustment


@dataclass
class TableResult:
    """Result of structured table extraction."""
    rows: list[list[str]]
    csv_path: str
    asset_path: str
    retrieval_text: str
    method: str  # "img2table" or "opencv"


@dataclass
class FigureResult:
    """Result of figure extraction and linking."""
    figure_id: str
    asset_path: str
    bbox: tuple[int, int, int, int]
    caption_text: str | None = None
    figure_number: int | None = None
    legend_items: list[dict] | None = None  # [{"key": "1", "value": "..."}]
    vision_description: str | None = None
    caption_block_id: str | None = None
    legend_block_id: str | None = None


@dataclass
class ChunkRecord:
    """A single retrieval chunk for RAG."""
    chunk_id: str
    doc_id: str
    page: int
    block_ids: list[str]
    bbox: tuple[int, int, int, int] | None
    type: str  # procedure, warning, caution, note, paragraph, legend, table, toc
    section_code: str | None
    source_label: str | None
    section_path: str
    text: str
    procedure_type: str | None = None  # removal, installation, inspection, adjustment
    system: str | None = None          # ac, steering, brakes, etc.
    engine_variant: str | None = None  # G10, G13, both
    steps: list[str] | None = None
    kv: list[dict] | None = None
    figure_refs: list[str] = field(default_factory=list)
    asset_refs: list[str] = field(default_factory=list)
    info_types: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    body_styles: list[str] = field(default_factory=list)
    trim_variants: list[str] = field(default_factory=list)
    sir_equipped: bool | None = None
    same_page_chunk_ids: list[str] = field(default_factory=list)
    same_page_figure_ids: list[str] = field(default_factory=list)
    related_figure_ids: list[str] = field(default_factory=list)
    related_table_ids: list[str] = field(default_factory=list)
    table_type: str | None = None
    token_count: int = 0
    source_doc: str = "main"           # "main" or "supplement" — supplement wins on conflict


@dataclass
class Citation:
    """A citation in a chat response."""
    chunk_id: str
    page: int
    source_label: str
    section_path: str
    figure_ids: list[str] = field(default_factory=list)

    def format(self) -> str:
        fig = f" | fig: {', '.join(self.figure_ids)}" if self.figure_ids else ""
        return f"[p{self.page} {self.section_path} {self.source_label} | chunk: {self.chunk_id}{fig}]"


@dataclass
class ChatResponse:
    """Structured response from the chat backend."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    figure_refs: list[str] = field(default_factory=list)
    next_question: str | None = None
    evidence_used: list[str] = field(default_factory=list)
    mode: str = "normal"  # "normal" or "deep_research"
    deep_research_summary: str | None = None
