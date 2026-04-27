"""
pipeline/loader/chunker.py

Chunking strategies:
  Annual reports → 512-token sliding window for prose, row-group for tables
  Concalls       → speaker-turn aware, never split mid Q&A, 1024 token max

Each chunk carries rich metadata that goes into ChromaDB.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import ANNUAL_REPORT, CONCALL, MIN_CHUNK_WORDS
from pipeline.extract.pdf_extractor import ExtractedDocument, PageBlock
from pipeline.extract.text_cleaner import clean_text, is_garbage_text
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str            # UUID — used as ChromaDB document id
    doc_type: str
    text: str                # cleaned chunk text
    chunk_index: int
    chunk_type: str          # prose | table | speaker_turn
    section: Optional[str]
    speaker: Optional[str]
    speaker_role: Optional[str]
    page_start: int
    page_end: int
    word_count: int
    # metadata fields that go into ChromaDB metadata dict
    symbol: str
    year: Optional[int]
    title: str


def _make_id() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
# Token-based split helper (approximate)
# We approximate tokens as words × 1.3
# ─────────────────────────────────────────────
def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split text into overlapping windows by approximate token count."""
    words = text.split()
    max_words = int(max_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)

    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += max_words - overlap_words

    return chunks


# ─────────────────────────────────────────────
# Annual Report Chunker
# ─────────────────────────────────────────────
def chunk_annual_report(
    doc: ExtractedDocument,
    symbol: str,
    year: Optional[int],
    title: str,
) -> List[Chunk]:
    cfg = ANNUAL_REPORT
    chunks: List[Chunk] = []
    idx = 0

    # Group consecutive blocks
    current_section = "General"
    prose_buffer = []
    buffer_pages = []

    def flush_prose():
        nonlocal idx
        if not prose_buffer:
            return
        combined = " ".join(prose_buffer)
        cleaned = clean_text(combined, aggressive=True)
        if is_garbage_text(cleaned, MIN_CHUNK_WORDS):
            prose_buffer.clear()
            buffer_pages.clear()
            return

        # Inject section header into every prose chunk
        if cfg["inject_section_header"] and current_section:
            prefixed = f"[Section: {current_section}]\n{cleaned}"
        else:
            prefixed = cleaned

        splits = _split_by_tokens(prefixed, cfg["chunk_size"], cfg["chunk_overlap"])
        for split in splits:
            if is_garbage_text(split, MIN_CHUNK_WORDS):
                continue
            chunks.append(Chunk(
                chunk_id=_make_id(),
                doc_type="annual_report",
                text=split,
                chunk_index=idx,
                chunk_type="prose",
                section=current_section,
                speaker=None,
                speaker_role=None,
                page_start=buffer_pages[0] if buffer_pages else 0,
                page_end=buffer_pages[-1] if buffer_pages else 0,
                word_count=len(split.split()),
                symbol=symbol,
                year=year,
                title=title,
            ))
            idx += 1

        prose_buffer.clear()
        buffer_pages.clear()

    for block in doc.blocks:
        if block.block_type == "section_header":
            flush_prose()
            current_section = block.text
            continue

        if block.block_type == "table":
            flush_prose()
            # Table chunking: group rows
            rows = block.table_data or []
            row_group = cfg["table_row_group"]
            header_row = rows[0] if rows else []

            for start in range(0, max(1, len(rows) - 1), row_group):
                group = rows[start: start + row_group]
                # Prepend header to every group for context
                if start > 0 and header_row:
                    group = [header_row] + group
                table_text = "\n".join(
                    " | ".join(str(c or "").strip() for c in row) for row in group
                )
                table_text = clean_text(table_text)
                if is_garbage_text(table_text, 5):
                    continue

                prefix = f"[Section: {current_section}] [Table]\n"
                chunks.append(Chunk(
                    chunk_id=_make_id(),
                    doc_type="annual_report",
                    text=prefix + table_text,
                    chunk_index=idx,
                    chunk_type="table",
                    section=current_section,
                    speaker=None,
                    speaker_role=None,
                    page_start=block.page_num,
                    page_end=block.page_num,
                    word_count=len(table_text.split()),
                    symbol=symbol,
                    year=year,
                    title=title,
                ))
                idx += 1
            continue

        if block.block_type == "prose":
            prose_buffer.append(block.text)
            buffer_pages.append(block.page_num)
            # Flush if buffer is getting large
            if _approx_tokens(" ".join(prose_buffer)) > cfg["chunk_size"] * 2:
                flush_prose()

    flush_prose()  # final flush

    log.info(f"  → {len(chunks)} chunks from annual report")
    return chunks


# ─────────────────────────────────────────────
# Concall Chunker
# ─────────────────────────────────────────────
def chunk_concall(
    doc: ExtractedDocument,
    symbol: str,
    year: Optional[int],
    title: str,
) -> List[Chunk]:
    cfg = CONCALL
    chunks: List[Chunk] = []
    idx = 0

    # Group speaker turns into chunks without splitting Q&A exchanges
    # Strategy: accumulate turns until token limit, then flush
    turn_buffer: List[PageBlock] = []
    buffer_tokens = 0

    def flush_turns():
        nonlocal idx
        if not turn_buffer:
            return

        # Build text: "SPEAKER_NAME (role): text\n\n"
        parts = []
        for t in turn_buffer:
            if t.speaker:
                role_tag = f" [{t.speaker_role}]" if t.speaker_role else ""
                parts.append(f"{t.speaker}{role_tag}:\n{clean_text(t.text)}")
            else:
                parts.append(clean_text(t.text))

        combined = "\n\n".join(parts)
        if is_garbage_text(combined, MIN_CHUNK_WORDS):
            turn_buffer.clear()
            return

        # Speaker context prefix
        speakers = list({t.speaker for t in turn_buffer if t.speaker})
        speaker_tag = ", ".join(speakers[:3]) if speakers else "Unknown"

        prefix = f"[Concall: {symbol} FY{year or '?'}] [Speakers: {speaker_tag}]\n"
        final_text = prefix + combined

        chunks.append(Chunk(
            chunk_id=_make_id(),
            doc_type="concall",
            text=final_text,
            chunk_index=idx,
            chunk_type="speaker_turn",
            section=None,
            speaker=speaker_tag,
            speaker_role=turn_buffer[0].speaker_role if turn_buffer else None,
            page_start=turn_buffer[0].page_num,
            page_end=turn_buffer[-1].page_num,
            word_count=len(final_text.split()),
            symbol=symbol,
            year=year,
            title=title,
        ))
        idx += 1
        turn_buffer.clear()

    for block in doc.blocks:
        block_tokens = _approx_tokens(block.text)

        # Detect Q&A boundary — flush before analyst question
        is_analyst_question = (
            block.speaker_role == "analyst"
            and any(
                q in block.text.lower()[:100]
                for q in ["?", "question", "could you", "can you", "what is", "how do"]
            )
        )

        # Flush if: token limit exceeded OR new analyst question starts
        if buffer_tokens + block_tokens > cfg["chunk_size"] or (
            is_analyst_question and buffer_tokens > 0
        ):
            flush_turns()
            buffer_tokens = 0

        turn_buffer.append(block)
        buffer_tokens += block_tokens

    flush_turns()

    log.info(f"  → {len(chunks)} chunks from concall")
    return chunks


# ─────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────
def chunk_document(
    doc: ExtractedDocument,
    symbol: str,
    year: Optional[int],
    title: str,
) -> List[Chunk]:
    if doc.doc_type == "annual_report":
        return chunk_annual_report(doc, symbol, year, title)
    elif doc.doc_type == "concall":
        return chunk_concall(doc, symbol, year, title)
    else:
        raise ValueError(f"Unknown doc_type: {doc.doc_type}")