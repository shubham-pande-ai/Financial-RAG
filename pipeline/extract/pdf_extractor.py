"""
pipeline/extract/pdf_extractor.py

Extracts structured content from PDFs.
- Annual reports: detects prose vs financial tables, captures section headers
- Concall transcripts: detects speaker turns, Q&A blocks, tags speaker role
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
    section: Optional[str] = None     # nearest section header
    speaker: Optional[str] = None     # concall only
    speaker_role: Optional[str] = None  # management | analyst | moderator
    table_data: Optional[List[List]] = None  # raw table rows


@dataclass
class ExtractedDocument:
    file_path: str
    doc_type: str             # annual_report | concall
    total_pages: int
    blocks: List[PageBlock] = field(default_factory=list)


# ─────────────────────────────────────────────
# Section header detection (annual reports)
# ─────────────────────────────────────────────
SECTION_HEADER_PATTERNS = [
    r"^(?:CHAPTER|SECTION|PART)\s+[IVXLCDM\d]+",
    r"^\d+\.\s+[A-Z][A-Za-z\s]{5,50}$",
    r"^[A-Z][A-Z\s]{8,60}$",           # ALL CAPS lines
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

    # segments = [pre_text, speaker1, text1, speaker2, text2, ...]
    i = 0
    while i < len(segments):
        seg = segments[i].strip()
        if not seg:
            i += 1
            continue

        # check if this looks like a speaker name
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


# ─────────────────────────────────────────────
# Annual report extractor
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

            # Extract tables first
            tables = page.extract_tables()
            used_bbox_regions = []

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

            # Extract remaining text (prose)
            text = page.extract_text() or ""
            if not text.strip():
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if _is_section_header(line):
                    current_section = line
                    doc.blocks.append(PageBlock(
                        page_num=page_num,
                        block_type="section_header",
                        text=line,
                        section=line,
                    ))

            # Prose block for whole page text
            doc.blocks.append(PageBlock(
                page_num=page_num,
                block_type="prose",
                text=text,
                section=current_section,
            ))

    log.info(f"  → {len(doc.blocks)} blocks extracted")
    return doc


# ─────────────────────────────────────────────
# Concall extractor
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