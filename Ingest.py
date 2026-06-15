#!/usr/bin/env python3
"""
ingest.py
CLI to run the full ingestion pipeline:
  MinIO PDFs (annual reports / concalls)
    → Docling extract → chunk → embed → Qdrant + MySQL

PDFs live in MinIO under:
  {BUCKET}/{SYMBOL}/{doc_type}/{year}/{filename}.pdf
  e.g.  quant-docs/RELIANCE/annual_report/2024/reliance_ar_2024.pdf

Usage:
    python ingest.py --symbol RELIANCE
    python ingest.py --symbol RELIANCE --type annual
    python ingest.py --symbol RELIANCE --type concall
    python ingest.py --all           # ingest every object in the bucket
    python ingest.py --stats         # show DB stats
    python ingest.py --list          # list all MinIO keys available
"""

import argparse
import sys
import time
import tempfile
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from config.settings import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_SECURE, MINIO_BUCKET, INGEST_TMP_DIR, LOG_DIR,
)
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
from pipeline.loader import chunk_document, load_chunks_to_qdrant
from utils.logger import get_logger

log = get_logger(__name__, LOG_DIR)


# ─────────────────────────────────────────────
# MinIO client (module-level singleton)
# ─────────────────────────────────────────────
def _minio_client() -> Minio:
    return Minio(
        endpoint   = MINIO_ENDPOINT,
        access_key = MINIO_ACCESS_KEY,
        secret_key = MINIO_SECRET_KEY,
        secure     = MINIO_SECURE,
    )


# ─────────────────────────────────────────────
# Key parsing helpers
# Key layout: {SYMBOL}/{doc_type}/{year}/{filename}.pdf
# ─────────────────────────────────────────────
def _parse_minio_key(key: str) -> dict | None:
    """
    Parse a MinIO object key into (symbol, doc_type, year, title).
    Returns None if the key doesn't match the expected layout or isn't a PDF.
    """
    if not key.lower().endswith(".pdf"):
        return None

    parts = key.split("/")
    # Minimum: SYMBOL/doc_type/year/file.pdf  → 4 parts
    if len(parts) < 4:
        return None

    symbol   = parts[0].upper()
    raw_type = parts[1].lower()
    raw_year = parts[2]
    filename = parts[-1]

    # Normalise doc_type
    if "annual" in raw_type:
        doc_type = "annual_report"
    elif "concall" in raw_type:
        doc_type = "concall"
    else:
        return None

    # Year
    year = int(raw_year) if raw_year.isdigit() and 2000 <= int(raw_year) <= 2035 else None

    return {
        "minio_key": key,
        "symbol":    symbol,
        "doc_type":  doc_type,
        "year":      year,
        "title":     Path(filename).stem,
    }


# ─────────────────────────────────────────────
# PDF discovery from MinIO
# ─────────────────────────────────────────────
def list_minio_pdfs(
    symbol:          str  = None,
    doc_type_filter: str  = None,
    client:          Minio = None,
) -> list[dict]:
    """List all matching PDF objects in MinIO bucket."""
    if client is None:
        client = _minio_client()

    prefix = ""
    if symbol:
        prefix = f"{symbol.upper()}/"
        if doc_type_filter:
            # map normalised type back to folder name
            folder = "annual_report" if doc_type_filter == "annual_report" else "concall"
            prefix += f"{folder}/"

    try:
        objects = client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
    except S3Error as e:
        log.error(f"MinIO list failed: {e}")
        return []

    pdfs = []
    for obj in objects:
        parsed = _parse_minio_key(obj.object_name)
        if parsed is None:
            continue
        if doc_type_filter and parsed["doc_type"] != doc_type_filter:
            continue
        parsed["size_bytes"] = obj.size or 0
        pdfs.append(parsed)

    return pdfs


# ─────────────────────────────────────────────
# Download one PDF from MinIO to a temp file
# ─────────────────────────────────────────────
def _download_pdf(minio_key: str, client: Minio) -> Path:
    """Download object to INGEST_TMP_DIR and return local Path."""
    INGEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    local_path = INGEST_TMP_DIR / Path(minio_key).name
    client.fget_object(MINIO_BUCKET, minio_key, str(local_path))
    return local_path


# ─────────────────────────────────────────────
# Core ingestion function (one PDF)
# ─────────────────────────────────────────────
def ingest_pdf(pdf_info: dict, force: bool = False, client: Minio = None) -> bool:
    minio_key: str = pdf_info["minio_key"]
    doc_type:  str = pdf_info["doc_type"]
    symbol:    str = pdf_info["symbol"]
    year:      int = pdf_info.get("year")
    title:     str = pdf_info["title"]

    log.info(f"\n{'='*60}")
    log.info(f"Ingesting: {minio_key}")
    log.info(f"  type={doc_type} | symbol={symbol} | year={year}")

    if not force and is_already_ingested(minio_key):
        log.info("  ⏭ Already ingested, skipping (use --force to re-ingest)")
        return True

    file_size_kb = pdf_info.get("size_bytes", 0) // 1024

    doc_id = upsert_document(
        symbol      = symbol,
        doc_type    = doc_type,
        year        = year,
        title       = title,
        minio_key   = minio_key,
        file_size_kb = file_size_kb,
    )

    if client is None:
        client = _minio_client()

    local_path = None
    t0 = time.time()

    try:
        # Step 1: Download from MinIO
        log.info("[1/4] Downloading from MinIO...")
        local_path = _download_pdf(minio_key, client)
        log.info(f"  → {local_path} ({file_size_kb} KB)")

        # Step 2: Extract (Docling)
        log.info("[2/4] Extracting with Docling...")
        extracted = extract_pdf(local_path, doc_type)
        if not extracted or not extracted.blocks:
            raise ValueError("Extraction returned no content")
        log.info(f"  → {len(extracted.blocks)} blocks | {extracted.total_pages} pages")

        # Step 3: Chunk
        log.info("[3/4] Chunking...")
        chunks = chunk_document(extracted, symbol, year, title)
        if not chunks:
            raise ValueError("Chunker returned no chunks")
        log.info(f"  → {len(chunks)} chunks")

        # Step 4: Embed + upsert to Qdrant
        log.info("[4/4] Embedding + loading to Qdrant...")
        collection_name = load_chunks_to_qdrant(chunks, doc_type)

        # Record chunks in MySQL
        for chunk in chunks:
            insert_chunk(
                doc_id      = doc_id,
                qdrant_id   = chunk.chunk_id,
                collection  = collection_name,
                chunk_index = chunk.chunk_index,
                chunk_type  = chunk.chunk_type,
                section     = chunk.section,
                speaker     = chunk.speaker,
                page_start  = chunk.page_start,
                page_end    = chunk.page_end,
                word_count  = chunk.word_count,
            )

        duration = time.time() - t0
        mark_document_ingested(doc_id, len(chunks), extracted.total_pages)
        log_ingestion(
            symbol         = symbol,
            doc_type       = doc_type,
            minio_key      = minio_key,
            status         = "success",
            chunks_created = len(chunks),
            duration_sec   = duration,
        )
        log.info(f"  ✓ Done in {duration:.1f}s | {len(chunks)} chunks stored in Qdrant")
        return True

    except Exception as e:
        duration = time.time() - t0
        log.exception(f"  ✗ Failed: {e}")
        mark_document_failed(doc_id)
        log_ingestion(
            symbol    = symbol,
            doc_type  = doc_type,
            minio_key = minio_key,
            status    = "failed",
            message   = str(e),
            duration_sec = duration,
        )
        return False

    finally:
        # Always clean up the temp file
        if local_path and local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Financial RAG — Ingestion Pipeline (MinIO → Qdrant)")
    ap.add_argument("--symbol", "-s",  help="Company symbol (e.g. RELIANCE)")
    ap.add_argument(
        "--type", "-t",
        choices=["annual", "concall"],
        help="Filter by document type",
    )
    ap.add_argument("--all",   action="store_true", help="Ingest all objects in MinIO bucket")
    ap.add_argument("--force", action="store_true", help="Re-ingest already ingested docs")
    ap.add_argument("--stats", action="store_true", help="Show database stats and exit")
    ap.add_argument("--list",  action="store_true", help="List available MinIO objects and exit")
    args = ap.parse_args()

    init_db()

    if args.stats:
        stats = get_stats()
        print("\n── Financial RAG DB Stats (MySQL) ──────────────────")
        print(f"  Companies  : {stats['companies']}")
        print(f"  Documents  : {stats['documents_ingested']}")
        print(f"  Chunks     : {stats['total_chunks']}")
        for dtype, n in stats.get("by_type", {}).items():
            print(f"  {dtype:<22}: {n} docs")
        print("────────────────────────────────────────────────────\n")
        return

    client = _minio_client()

    if args.list:
        pdfs = list_minio_pdfs(client=client)
        print(f"\n── MinIO bucket '{MINIO_BUCKET}' — {len(pdfs)} PDF(s) ──")
        for p in pdfs:
            ingested = "✓" if is_already_ingested(p["minio_key"]) else "·"
            print(f"  [{ingested}] {p['minio_key']}")
        print()
        return

    # Resolve doc_type filter
    doc_type_filter = None
    if args.type == "annual":
        doc_type_filter = "annual_report"
    elif args.type == "concall":
        doc_type_filter = "concall"

    # Resolve symbols
    if args.all:
        pdfs = list_minio_pdfs(doc_type_filter=doc_type_filter, client=client)
        log.info(f"Found {len(pdfs)} PDF(s) in bucket")
    elif args.symbol:
        pdfs = list_minio_pdfs(
            symbol=args.symbol.upper(),
            doc_type_filter=doc_type_filter,
            client=client,
        )
        if not pdfs:
            log.error(
                f"No PDFs found for {args.symbol.upper()} in "
                f"MinIO bucket '{MINIO_BUCKET}' "
                f"(prefix: {args.symbol.upper()}/)"
            )
            sys.exit(1)
    else:
        ap.print_help()
        sys.exit(1)

    total_success = 0
    total_fail    = 0

    for pdf_info in pdfs:
        if ingest_pdf(pdf_info, force=args.force, client=client):
            total_success += 1
        else:
            total_fail += 1

    print(f"\n{'='*50}")
    print(f"Ingestion complete: {total_success} success | {total_fail} failed")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()