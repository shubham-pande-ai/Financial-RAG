#!/usr/bin/env python3
"""
ingest.py
CLI to run the full ingestion pipeline:
  PDF files (from screener_downloader.py) → extract → chunk → embed → ChromaDB + SQLite

Usage:
    python ingest.py --symbol RELIANCE
    python ingest.py --symbol RELIANCE --type annual
    python ingest.py --symbol RELIANCE --type concall
    python ingest.py --all           # ingest everything in screener_docs/
    python ingest.py --stats         # show DB stats
"""

import argparse
import sys
import time
from pathlib import Path
from config.settings import SCREENER_DOCS_DIR, LOG_DIR
from db.database import (
    init_db,
    upsert_document,
    mark_document_ingested,
    mark_document_failed,
    is_already_ingested,
    log_ingestion,
    get_stats,
    insert_chunk,
)
from pipeline.extract import extract_pdf
from pipeline.loader import chunk_document, load_chunks_to_chroma
from utils.logger import get_logger

log = get_logger(__name__, LOG_DIR)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def resolve_doc_type(folder_name: str) -> str:
    """Map screener_downloader output folder name to doc_type."""
    if "annual" in folder_name.lower():
        return "annual_report"
    if "concall" in folder_name.lower():
        return "concall"
    return None


def extract_year_from_path(path: Path) -> int:
    """screener_downloader saves to symbol/doc_type/year/filename.pdf"""
    parts = path.parts
    for part in reversed(parts):
        if part.isdigit() and 2000 <= int(part) <= 2035:
            return int(part)
    return None


def find_pdfs(symbol: str, doc_type_filter: str = None) -> list:
    """Find all PDF files for a symbol under screener_docs/."""
    symbol_dir = SCREENER_DOCS_DIR / symbol.upper()
    if not symbol_dir.exists():
        log.error(f"No data directory for {symbol}: {symbol_dir}")
        log.info("Run: python screener_downloader.py {symbol}")
        return []

    pdfs = []
    for pdf_path in symbol_dir.rglob("*.pdf"):
        # Infer doc_type from parent folder
        doc_type = None
        for part in pdf_path.parts:
            dt = resolve_doc_type(part)
            if dt:
                doc_type = dt
                break

        if not doc_type:
            continue
        if doc_type_filter and doc_type != doc_type_filter:
            continue

        year = extract_year_from_path(pdf_path)
        pdfs.append({
            "path": pdf_path,
            "doc_type": doc_type,
            "year": year,
            "symbol": symbol.upper(),
            "title": pdf_path.stem,
        })

    return pdfs


# ─────────────────────────────────────────────
# Core ingestion function (one PDF)
# ─────────────────────────────────────────────
def ingest_pdf(pdf_info: dict, force: bool = False) -> bool:
    path: Path = pdf_info["path"]
    doc_type: str = pdf_info["doc_type"]
    symbol: str = pdf_info["symbol"]
    year: int = pdf_info["year"]
    title: str = pdf_info["title"]

    log.info(f"\n{'='*60}")
    log.info(f"Ingesting: {path.name}")
    log.info(f"  type={doc_type} | symbol={symbol} | year={year}")

    # Skip if already done
    if not force and is_already_ingested(str(path)):
        log.info(f"  ⏭ Already ingested, skipping (use --force to re-ingest)")
        return True

    file_size_kb = path.stat().st_size // 1024

    # Register in SQLite
    doc_id = upsert_document(
        symbol=symbol,
        doc_type=doc_type,
        year=year,
        title=title,
        file_path=str(path),
        file_size_kb=file_size_kb,
    )

    t0 = time.time()

    try:
        # Step 1: Extract
        log.info("[1/3] Extracting PDF...")
        extracted = extract_pdf(path, doc_type)
        if not extracted or not extracted.blocks:
            raise ValueError("Extraction returned no content")

        log.info(f"  → {len(extracted.blocks)} blocks | {extracted.total_pages} pages")

        # Step 2: Chunk
        log.info("[2/3] Chunking...")
        chunks = chunk_document(extracted, symbol, year, title)
        if not chunks:
            raise ValueError("Chunker returned no chunks")

        log.info(f"  → {len(chunks)} chunks")

        # Step 3: Embed + load to ChromaDB
        log.info("[3/3] Embedding + loading to ChromaDB...")
        collection_name = load_chunks_to_chroma(chunks, doc_type)

        # Step 4: Record chunks in SQLite
        for chunk in chunks:
            insert_chunk(
                doc_id=doc_id,
                chroma_id=chunk.chunk_id,
                collection=collection_name,
                chunk_index=chunk.chunk_index,
                chunk_type=chunk.chunk_type,
                section=chunk.section,
                speaker=chunk.speaker,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                word_count=chunk.word_count,
            )

        duration = time.time() - t0
        mark_document_ingested(doc_id, len(chunks), extracted.total_pages)
        log_ingestion(
            symbol=symbol,
            doc_type=doc_type,
            file_path=str(path),
            status="success",
            chunks_created=len(chunks),
            duration_sec=duration,
        )

        log.info(f"  ✓ Done in {duration:.1f}s | {len(chunks)} chunks stored")
        return True

    except Exception as e:
        duration = time.time() - t0
        log.exception(f"  ✗ Failed: {e}")
        mark_document_failed(doc_id)
        log_ingestion(
            symbol=symbol,
            doc_type=doc_type,
            file_path=str(path),
            status="failed",
            message=str(e),
            duration_sec=duration,
        )
        return False


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Financial RAG — Ingestion Pipeline")
    ap.add_argument("--symbol", "-s", help="Company symbol (e.g. RELIANCE)")
    ap.add_argument(
        "--type", "-t",
        choices=["annual", "concall"],
        help="Filter by document type",
    )
    ap.add_argument("--all", action="store_true", help="Ingest all symbols in screener_docs/")
    ap.add_argument("--force", action="store_true", help="Re-ingest already ingested docs")
    ap.add_argument("--stats", action="store_true", help="Show database stats and exit")
    args = ap.parse_args()

    # Init DB
    init_db()

    # Stats mode
    if args.stats:
        stats = get_stats()
        print("\n── Financial RAG DB Stats ──────────────────")
        print(f"  Companies  : {stats['companies']}")
        print(f"  Documents  : {stats['documents_ingested']}")
        print(f"  Chunks     : {stats['total_chunks']}")
        for dtype, n in stats.get("by_type", {}).items():
            print(f"  {dtype:<20}: {n} docs")
        print("────────────────────────────────────────────\n")
        return

    # Resolve symbols to process
    if args.all:
        if not SCREENER_DOCS_DIR.exists():
            log.error(f"screener_docs/ not found at {SCREENER_DOCS_DIR}")
            sys.exit(1)
        symbols = [d.name for d in SCREENER_DOCS_DIR.iterdir() if d.is_dir()]
        log.info(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        ap.print_help()
        sys.exit(1)

    # Map type arg
    doc_type_filter = None
    if args.type == "annual":
        doc_type_filter = "annual_report"
    elif args.type == "concall":
        doc_type_filter = "concall"

    # Run ingestion
    total_success = 0
    total_fail = 0

    for symbol in symbols:
        log.info(f"\n{'#'*60}")
        log.info(f"Processing symbol: {symbol}")

        pdfs = find_pdfs(symbol, doc_type_filter)
        if not pdfs:
            log.warning(f"No PDFs found for {symbol}")
            continue

        log.info(f"Found {len(pdfs)} PDF(s)")
        for pdf_info in pdfs:
            if ingest_pdf(pdf_info, force=args.force):
                total_success += 1
            else:
                total_fail += 1

    print(f"\n{'='*50}")
    print(f"Ingestion complete: {total_success} success | {total_fail} failed")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()