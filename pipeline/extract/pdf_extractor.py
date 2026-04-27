"""
pipeline/extract/pdf_extractor.py

FIXES applied in this version:
  [FIX 4] extract_annual_report — prose block no longer duplicates table text.
           pdfplumber bounding boxes are collected for every detected table,
           then the page is cropped to exclude those regions before calling
           extract_text(). This means prose chunks contain ONLY non-table text,
           eliminating the near-duplicate embeddings that were diluting retrieval.
  [FIX 5] extract_annual_report — section header lines are removed from the
           prose block text (they were appearing both as a section_header block
           AND as part of the following prose block).
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pdfplumber

from config.settings import MAX_PDF_SIZE_MB
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────
@dataclass
class PageBlock:
    """Atomic block of content from one PDF page."""
    page_num: int
    block_type: str          # prose | table | section_header | speaker_turn
    text: str
    section: Optional[str] = None
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    table_data: Optional[List[List]] = None


@dataclass
class ExtractedDocument:
    file_path: str
    doc_type: str
    total_pages: int
    blocks: List[PageBlock] = field(default_factory=list)


# ─────────────────────────────────────────────
# Section header detection (annual reports)
# ─────────────────────────────────────────────
SECTION_HEADER_PATTERNS = [
    r"^(?:CHAPTER|SECTION|PART)\s+[IVXLCDM\d]+",
    r"^\d+\.\s+[A-Z][A-Za-z\s]{5,50}$",
    r"^[A-Z][A-Z\s]{8,60}$",
    r"^(?:Notes? to|Standalone|Consolidated)\s+",
    r"^(?:Directors|Management|Auditors?|Board)\s+",
    r"^(?:Statement of|Balance Sheet|Profit and Loss|Cash Flow)",
]
SECTION_PATTERNS = [re.compile(p, re.MULTILINE) for p in SECTION_HEADER_PATTERNS]


def _is_section_header(text: str) -> bool:
    text = text.strip()
    if len(text) > 120 or len(text) < 4:
        return False
    return any(p.match(text) for p in SECTION_PATTERNS)


# ─────────────────────────────────────────────
# Speaker detection (concalls)
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
    """Split concall page text into individual speaker turn blocks."""
    blocks = []
    segments = SPEAKER_PATTERN.split(page_text)

    i = 0
    while i < len(segments):
        seg = segments[i].strip()
        if not seg:
            i += 1
            continue

        if i + 1 < len(segments) and len(seg) < 60 and "\n" not in seg:
            speaker = seg
            body = segments[i + 1].strip() if i + 1 < len(segments) else ""
            if body:
                blocks.append(PageBlock(
                    page_num=page_num,
                    block_type="speaker_turn",
                    text=body,
                    speaker=speaker,
                    speaker_role=_detect_speaker_role(speaker),
                ))
            i += 2
        else:
            if seg:
                blocks.append(PageBlock(
                    page_num=page_num,
                    block_type="prose",
                    text=seg,
                ))
            i += 1

    return blocks or [PageBlock(page_num=page_num, block_type="prose", text=page_text)]


# ─────────────────────────────────────────────
# Table extraction helpers
# ─────────────────────────────────────────────
def _table_to_text(table: List[List]) -> str:
    """Convert pdfplumber table rows to readable text."""
    lines = []
    for row in table:
        cells = [str(c or "").strip() for c in row]
        line = " | ".join(cells)
        if line.strip(" |"):
            lines.append(line)
    return "\n".join(lines)


def _get_table_bboxes(page) -> List[Tuple[float, float, float, float]]:
    """
    Return bounding boxes (x0, top, x1, bottom) for all tables on a page.
    Used to crop them out before extracting prose text.  [FIX 4]
    """
    bboxes = []
    for table_obj in page.find_tables():
        try:
            bbox = table_obj.bbox   # (x0, top, x1, bottom)
            bboxes.append(bbox)
        except Exception:
            pass
    return bboxes


def _extract_prose_excluding_tables(page, table_bboxes) -> str:
    """
    Extract text from a page while cropping out table regions.
    Falls back to full page.extract_text() if cropping fails.  [FIX 4]
    """
    if not table_bboxes:
        return page.extract_text() or ""

    try:
        # Crop away each table bbox and collect remaining text
        remaining = page
        text_parts = []

        # Sort bboxes top-to-bottom so we can extract prose strips between them
        sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])  # sort by 'top'

        page_top    = page.bbox[1]
        page_bottom = page.bbox[3]
        prev_bottom = page_top

        for (x0, top, x1, bottom) in sorted_bboxes:
            # Strip above this table
            if top > prev_bottom + 2:
                strip = page.crop((page.bbox[0], prev_bottom, page.bbox[2], top))
                t = strip.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
            prev_bottom = bottom

        # Strip below last table
        if prev_bottom < page_bottom - 2:
            strip = page.crop((page.bbox[0], prev_bottom, page.bbox[2], page_bottom))
            t = strip.extract_text() or ""
            if t.strip():
                text_parts.append(t)

        return "\n".join(text_parts)

    except Exception as e:
        log.debug(f"  Table-crop fallback triggered: {e}")
        return page.extract_text() or ""


# ─────────────────────────────────────────────
# Annual report extractor  [FIX 4 + FIX 5]
# ─────────────────────────────────────────────
def extract_annual_report(pdf_path: Path) -> ExtractedDocument:
    doc = ExtractedDocument(
        file_path=str(pdf_path),
        doc_type="annual_report",
        total_pages=0,
    )

    current_section = "General"

    with pdfplumber.open(str(pdf_path)) as pdf:
        doc.total_pages = len(pdf.pages)
        log.info(f"  Extracting annual report: {pdf_path.name} ({doc.total_pages} pages)")

        for page in pdf.pages:
            page_num = page.page_number

            # ── Step 1: detect & record table bounding boxes ─────────────
            table_bboxes = _get_table_bboxes(page)      # [FIX 4]

            # ── Step 2: extract and store table blocks ────────────────────
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                table_text = _table_to_text(table)
                if len(table_text.strip()) > 10:
                    doc.blocks.append(PageBlock(
                        page_num=page_num,
                        block_type="table",
                        text=table_text,
                        section=current_section,
                        table_data=table,
                    ))

            # ── Step 3: extract prose text EXCLUDING table regions ────────
            # [FIX 4] Previously used page.extract_text() which included ALL
            # text on the page — duplicating every table cell as prose too.
            prose_text = _extract_prose_excluding_tables(page, table_bboxes)

            if not prose_text.strip():
                continue

            # ── Step 4: detect section headers, update state ──────────────
            section_lines = set()
            for line in prose_text.split("\n"):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                if _is_section_header(line_stripped):
                    current_section = line_stripped
                    section_lines.add(line_stripped)    # [FIX 5] track for removal
                    doc.blocks.append(PageBlock(
                        page_num=page_num,
                        block_type="section_header",
                        text=line_stripped,
                        section=line_stripped,
                    ))

            # ── Step 5: build prose block without section header lines ────
            # [FIX 5] Remove lines that were already emitted as section_header
            # blocks to avoid them appearing twice in the prose chunk.
            if section_lines:
                filtered_lines = [
                    ln for ln in prose_text.split("\n")
                    if ln.strip() not in section_lines
                ]
                prose_text = "\n".join(filtered_lines)

            if prose_text.strip():
                doc.blocks.append(PageBlock(
                    page_num=page_num,
                    block_type="prose",
                    text=prose_text,
                    section=current_section,
                ))

    log.info(f"  → {len(doc.blocks)} blocks extracted")
    return doc


# ─────────────────────────────────────────────
# Concall extractor (unchanged)
# ─────────────────────────────────────────────
def extract_concall(pdf_path: Path) -> ExtractedDocument:
    doc = ExtractedDocument(
        file_path=str(pdf_path),
        doc_type="concall",
        total_pages=0,
    )

    with pdfplumber.open(str(pdf_path)) as pdf:
        doc.total_pages = len(pdf.pages)
        log.info(f"  Extracting concall: {pdf_path.name} ({doc.total_pages} pages)")

        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""
            if not text.strip():
                continue

            turns = _extract_speaker_turns(text, page_num)
            doc.blocks.extend(turns)

    log.info(f"  → {len(doc.blocks)} speaker blocks extracted")
    return doc


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def extract_pdf(pdf_path: Path, doc_type: str) -> Optional[ExtractedDocument]:
    """
    Extract content from a PDF.
    doc_type: 'annual_report' or 'concall'
    Returns None on failure.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        log.error(f"File not found: {pdf_path}")
        return None

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        log.warning(f"Skipping {pdf_path.name}: {size_mb:.1f}MB exceeds limit")
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