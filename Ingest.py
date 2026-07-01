#!/usr/bin/env python3
"""
ingest.py
CLI to run the full ingestion pipeline:
  MinIO PDFs (annual reports / concalls)
    → Docling extract → chunk → embed → Qdrant + MySQL

PDFs live in MinIO across two doc-type-specific buckets (no shared bucket,
no year subfolder — year is parsed from the filename):

  annual-reports/{symbol_lower}/{filename}.pdf
  concall-transcripts/{symbol_lower}/{filename}.pdf

  e.g.  annual-reports/hal/2024_Financial_Year_2024_from_bse.pdf
        concall-transcripts/hal/2024_May_2024_Transcript.pdf

The stored `minio_key` in MySQL is the full "{bucket}/{key}" path, matching
the `object_path` convention already used by the Quant Copilot pdf_documents
table, so it stays globally unique across both buckets.

Usage:
    python ingest.py --symbol HAL
    python ingest.py --symbol HAL --type annual
    python ingest.py --symbol HAL --type concall
    python ingest.py --all           # ingest every object in both buckets
    python ingest.py --all --year 2020   # ingest only PDFs from 2020
    python ingest.py --stats         # show DB stats
    python ingest.py --list          # list all MinIO keys available
"""

import argparse
import re
import sys
import time
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from config.settings import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_SECURE, INGEST_TMP_DIR, LOG_DIR,
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
# Doc-type → bucket mapping (real layout, not a single shared bucket)
# ─────────────────────────────────────────────
DOC_TYPE_BUCKETS = {
    "annual_report": "annual-reports",
    "concall":       "concall-transcripts",
}

_YEAR_RE = re.compile(r"^(\d{4})")


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
# Key layout within a doc-type bucket: {symbol_lower}/{filename}.pdf
# ─────────────────────────────────────────────
def _parse_minio_key(key: str, doc_type: str, bucket: str) -> dict | None:
    """
    Parse a MinIO object key (within a doc_type-specific bucket) into
    (symbol, doc_type, year, title). Returns None if the key doesn't match
    the expected layout or isn't a PDF.

    Year is parsed from the leading 4 digits of the filename since there's
    no separate year folder in the real layout.
    """
    if not key.lower().endswith(".pdf"):
        return None

    parts = key.split("/")
    # Minimum: symbol/file.pdf → 2 parts
    if len(parts) < 2:
        return None

    symbol   = parts[0].upper()
    filename = parts[-1]

    m = _YEAR_RE.match(filename)
    year = int(m.group(1)) if m else None

    return {
        "minio_key": f"{bucket}/{key}",   # full identifier — stored in MySQL, globally unique
        "bucket":    bucket,
        "key":       key,                 # raw key — used for MinIO API calls
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
    year:            int  = None,
    client:          Minio = None,
) -> list[dict]:
    """List all matching PDF objects across the doc-type-specific MinIO buckets."""
    if client is None:
        client = _minio_client()

    doc_types = [doc_type_filter] if doc_type_filter else list(DOC_TYPE_BUCKETS.keys())
    prefix = f"{symbol.lower()}/" if symbol else ""

    pdfs = []
    for doc_type in doc_types:
        bucket = DOC_TYPE_BUCKETS[doc_type]
        try:
            objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        except S3Error as e:
            log.error(f"MinIO list failed for bucket '{bucket}': {e}")
            continue

        for obj in objects:
            parsed = _parse_minio_key(obj.object_name, doc_type, bucket)
            if parsed is None:
                continue
            # Apply year filter if provided
            if year is not None and parsed.get("year") != year:
                continue
            parsed["size_bytes"] = obj.size or 0
            pdfs.append(parsed)

    return pdfs


# ─────────────────────────────────────────────
# Download one PDF from MinIO to a temp file
# ─────────────────────────────────────────────
def _download_pdf(bucket: str, key: str, client: Minio) -> Path:
    """Download object to INGEST_TMP_DIR and return local Path."""
    INGEST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    local_path = INGEST_TMP_DIR / Path(key).name
    client.fget_object(bucket, key, str(local_path))
    return local_path


# ─────────────────────────────────────────────
# Core ingestion function (one PDF)
# ─────────────────────────────────────────────
def ingest_pdf(pdf_info: dict, force: bool = False, client: Minio = None) -> bool:
    bucket:    str = pdf_info["bucket"]
    key:       str = pdf_info["key"]
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
        local_path = _download_pdf(bucket, key, client)
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
    ap.add_argument("--symbol", "-s",  help="Company symbol (e.g. HAL)")
    ap.add_argument(
        "--type", "-t",
        choices=["annual", "concall"],
        help="Filter by document type",
    )
    ap.add_argument("--year", "-y", type=int, help="Filter by year (e.g. 2020)")
    ap.add_argument("--all",   action="store_true", help="Ingest all objects across both buckets")
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
        pdfs = list_minio_pdfs(client=client, year=args.year)
        print(f"\n── MinIO objects — buckets {list(DOC_TYPE_BUCKETS.values())} — {len(pdfs)} PDF(s) ──")
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
        pdfs = list_minio_pdfs(doc_type_filter=doc_type_filter, year=args.year, client=client)
        log.info(f"Found {len(pdfs)} PDF(s) across bucket(s)")
    elif args.symbol:
        pdfs = list_minio_pdfs(
            symbol=args.symbol.upper(),
            doc_type_filter=doc_type_filter,
            year=args.year,
            client=client,
        )
        if not pdfs:
            buckets = (
                [DOC_TYPE_BUCKETS[doc_type_filter]] if doc_type_filter
                else list(DOC_TYPE_BUCKETS.values())
            )
            year_msg = f" with year {args.year}" if args.year else ""
            log.error(
                f"No PDFs found for {args.symbol.upper()} in bucket(s) {buckets} "
                f"(prefix: {args.symbol.lower()}/){year_msg}"
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