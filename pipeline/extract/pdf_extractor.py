"""
pipeline/extract/pdf_extractor.py

Replaces pdfplumber with Docling for layout-aware extraction.

Docling gives us:
  - Automatic section header detection via layout analysis
  - True table extraction (TableFormer model) — no bbox-crop hacks needed
  - Cleaner prose blocks that never duplicate table text
  - Page provenance on every item

For concalls, we still apply the regex-based speaker-turn parser on top of
Docling's text output, because Docling has no concept of Q&A dialogue structure.

Dependencies:
    pip install docling
    # docling will pull docling-core, docling-ibm-models, etc.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import MAX_PDF_SIZE_MB
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Data models  (unchanged interface)
# ─────────────────────────────────────────────
@dataclass
class PageBlock:
    page_num:     int
    block_type:   str           # prose | table | section_header | speaker_turn
    text:         str
    section:      Optional[str] = None
    speaker:      Optional[str] = None
    speaker_role: Optional[str] = None
    table_data:   Optional[List[List]] = None


@dataclass
class ExtractedDocument:
    file_path:   str
    doc_type:    str
    total_pages: int
    blocks: List[PageBlock] = field(default_factory=list)


# ─────────────────────────────────────────────
# Docling converter (module-level singleton)
# ─────────────────────────────────────────────
_converter = None


def _get_converter():
    global _converter
    if _converter is not None:
        return _converter

    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr             = False   # native PDFs — skip OCR for speed
    pipeline_options.do_table_structure = True    # TableFormer for accurate tables

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: pipeline_options,
        }
    )
    log.info("Docling DocumentConverter initialised")
    return _converter


# ─────────────────────────────────────────────
# Speaker detection (concalls) — unchanged logic
# ─────────────────────────────────────────────
SPEAKER_PATTERN = re.compile(
    r"^([A-Z][A-Za-z\s\.\-]{2,50}?)(?:\s*[:\-–]\s*|\n)",
    re.MULTILINE,
)

MGMT_KEYWORDS = [
    "ceo", "cfo", "coo", "cmd", "managing director", "chief executive",
    "chief financial", "chairman", "director", "president", "vice president",
    "head of", "moderator", "operator", "coordinator",
]
ANALYST_KEYWORDS = [
    "analyst", "research", "securities", "capital", "bank", "asset",
    "fund", "investment", "equity", "management",
]


def _detect_speaker_role(speaker: str) -> str:
    low = speaker.lower()
    if any(k in low for k in ["moderator", "operator", "coordinator"]):
        return "moderator"
    if any(k in low for k in MGMT_KEYWORDS):
        return "management"
    if any(k in low for k in ANALYST_KEYWORDS):
        return "analyst"
    return "unknown"


def _extract_speaker_turns(page_text: str, page_num: int) -> List[PageBlock]:
    blocks   = []
    segments = SPEAKER_PATTERN.split(page_text)

    i = 0
    while i < len(segments):
        seg = segments[i].strip()
        if not seg:
            i += 1
            continue
        if i + 1 < len(segments) and len(seg) < 60 and "\n" not in seg:
            speaker = seg
            body    = segments[i + 1].strip() if i + 1 < len(segments) else ""
            if body:
                blocks.append(PageBlock(
                    page_num     = page_num,
                    block_type   = "speaker_turn",
                    text         = body,
                    speaker      = speaker,
                    speaker_role = _detect_speaker_role(speaker),
                ))
            i += 2
        else:
            if seg:
                blocks.append(PageBlock(
                    page_num   = page_num,
                    block_type = "prose",
                    text       = seg,
                ))
            i += 1

    return blocks or [PageBlock(page_num=page_num, block_type="prose", text=page_text)]


# ─────────────────────────────────────────────
# Docling item label constants
# ─────────────────────────────────────────────
def _get_labels():
    """Import DocItemLabel — compatible with docling-core ≥ 2.x."""
    try:
        from docling_core.types.doc import DocItemLabel
        return DocItemLabel
    except ImportError:
        # Older docling bundles it differently
        from docling.datamodel.document import DocItemLabel  # type: ignore
        return DocItemLabel


# ─────────────────────────────────────────────
# Annual report extractor  (Docling)
# ─────────────────────────────────────────────
def extract_annual_report(pdf_path: Path) -> ExtractedDocument:
    doc_out = ExtractedDocument(
        file_path   = str(pdf_path),
        doc_type    = "annual_report",
        total_pages = 0,
    )

    L = _get_labels()
    converter = _get_converter()

    log.info(f"  Extracting annual report with Docling: {pdf_path.name}")
    result = converter.convert(source=str(pdf_path))
    dl_doc = result.document

    # Total pages from document metadata
    doc_out.total_pages = getattr(dl_doc, "num_pages", 0) or _count_pages(dl_doc)

    current_section = "General"

    for item, _level in dl_doc.iterate_items():
        label    = getattr(item, "label", None)
        text     = getattr(item, "text", "") or ""
        text     = text.strip()
        page_num = _page_of(item)

        if not text and label != L.TABLE:
            continue

        # ── Section headers ───────────────────────────────────────────────
        if label == L.SECTION_HEADER:
            current_section = text
            doc_out.blocks.append(PageBlock(
                page_num   = page_num,
                block_type = "section_header",
                text       = text,
                section    = text,
            ))

        # ── Tables ────────────────────────────────────────────────────────
        elif label == L.TABLE:
            table_data, table_text = _extract_table(item)
            if table_text.strip():
                prefix = f"[Section: {current_section}] [Table]\n"
                doc_out.blocks.append(PageBlock(
                    page_num   = page_num,
                    block_type = "table",
                    text       = prefix + table_text,
                    section    = current_section,
                    table_data = table_data,
                ))

        # ── Prose / list items / captions ─────────────────────────────────
        elif label in (
            L.TEXT, L.PARAGRAPH, L.LIST_ITEM,
            L.CAPTION, L.FOOTNOTE,
        ):
            doc_out.blocks.append(PageBlock(
                page_num   = page_num,
                block_type = "prose",
                text       = text,
                section    = current_section,
            ))

        # Everything else (pictures, formulas, etc.) → skip

    log.info(f"  → {len(doc_out.blocks)} blocks | {doc_out.total_pages} pages")
    return doc_out


# ─────────────────────────────────────────────
# Concall extractor  (Docling text → speaker turns)
# ─────────────────────────────────────────────
def extract_concall(pdf_path: Path) -> ExtractedDocument:
    doc_out = ExtractedDocument(
        file_path   = str(pdf_path),
        doc_type    = "concall",
        total_pages = 0,
    )

    L = _get_labels()
    converter = _get_converter()

    log.info(f"  Extracting concall with Docling: {pdf_path.name}")
    result = converter.convert(source=str(pdf_path))
    dl_doc = result.document

    doc_out.total_pages = getattr(dl_doc, "num_pages", 0) or _count_pages(dl_doc)

    # Collect all text blocks grouped by page, then run speaker-turn parser
    page_texts: dict[int, list[str]] = {}

    for item, _level in dl_doc.iterate_items():
        label = getattr(item, "label", None)
        text  = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if label in (L.TEXT, L.PARAGRAPH, L.LIST_ITEM, L.SECTION_HEADER):
            page_num = _page_of(item)
            page_texts.setdefault(page_num, []).append(text)

    for page_num in sorted(page_texts):
        page_text = "\n".join(page_texts[page_num])
        turns     = _extract_speaker_turns(page_text, page_num)
        doc_out.blocks.extend(turns)

    log.info(f"  → {len(doc_out.blocks)} speaker blocks | {doc_out.total_pages} pages")
    return doc_out


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _page_of(item) -> int:
    """Get page number (1-based) from a Docling item's provenance."""
    prov = getattr(item, "prov", None)
    if prov:
        try:
            return prov[0].page_no
        except (IndexError, AttributeError):
            pass
    return 0


def _count_pages(dl_doc) -> int:
    """Fallback: count unique page numbers seen in items."""
    pages = set()
    for item, _ in dl_doc.iterate_items():
        pages.add(_page_of(item))
    return max(pages) if pages else 0


def _extract_table(item) -> tuple[List[List], str]:
    """
    Convert a Docling TableItem to (table_data, text).
    Tries dataframe export first, falls back to markdown string.
    """
    table_data = []
    table_text = ""

    try:
        df = item.export_to_dataframe()
        # Convert to list-of-lists for compatibility with existing chunker
        table_data = [list(df.columns)] + df.values.tolist()
        rows = []
        for row in table_data:
            rows.append(" | ".join(str(c or "").strip() for c in row))
        table_text = "\n".join(rows)
    except Exception:
        # Fallback: Docling markdown export
        try:
            table_text = item.export_to_markdown() or ""
        except Exception:
            table_text = getattr(item, "text", "") or ""

    return table_data, table_text


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def extract_pdf(pdf_path: Path, doc_type: str) -> Optional[ExtractedDocument]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        log.error(f"File not found: {pdf_path}")
        return None

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        log.warning(f"Skipping {pdf_path.name}: {size_mb:.1f} MB exceeds limit")
        return None

    try:
        if doc_type == "annual_report":
            return extract_annual_report(pdf_path)
        elif doc_type == "concall":
            return extract_concall(pdf_path)
        else:
            log.error(f"Unknown doc_type: {doc_type}")
            return None
    except Exception as e:
        log.exception(f"Extraction failed for {pdf_path.name}: {e}")
        return None