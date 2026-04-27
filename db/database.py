"""
db/database.py
SQLite metadata store — shared across both RAG pipelines.

Tables:
  companies       — company master
  documents       — one row per PDF file
  chunks          — one row per chunk (links SQLite ↔ ChromaDB)
  ingestion_log   — track what has been processed
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from config.settings import DB_PATH
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS companies (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT UNIQUE NOT NULL,
    name        TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id),
    symbol      TEXT NOT NULL,
    doc_type    TEXT NOT NULL CHECK(doc_type IN ('annual_report', 'concall')),
    year        INTEGER,
    title       TEXT,
    file_path   TEXT UNIQUE,
    file_size_kb INTEGER,
    total_pages INTEGER,
    total_chunks INTEGER DEFAULT 0,
    ingested    INTEGER DEFAULT 0,   -- 0=pending, 1=done, 2=failed
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    chroma_id   TEXT UNIQUE NOT NULL,    -- ChromaDB document id
    collection  TEXT NOT NULL,           -- which Chroma collection
    chunk_index INTEGER NOT NULL,
    chunk_type  TEXT,                    -- prose / table / speaker_turn
    section     TEXT,                    -- section header context
    speaker     TEXT,                    -- concall speaker name/role
    page_start  INTEGER,
    page_end    INTEGER,
    word_count  INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chroma ON chunks(chroma_id);
CREATE INDEX IF NOT EXISTS idx_docs_symbol ON documents(symbol);
CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    doc_type    TEXT,
    file_path   TEXT,
    status      TEXT,
    message     TEXT,
    chunks_created INTEGER DEFAULT 0,
    duration_sec   REAL,
    created_at  TEXT DEFAULT (datetime('now'))
);
"""


# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────
@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    log.info(f"DB initialised at {DB_PATH}")


# ─────────────────────────────────────────────
# Companies
# ─────────────────────────────────────────────
def upsert_company(symbol: str, name: str = None) -> int:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO companies (symbol, name) VALUES (?, ?)",
            (symbol.upper(), name or symbol.upper()),
        )
        row = conn.execute(
            "SELECT id FROM companies WHERE symbol = ?", (symbol.upper(),)
        ).fetchone()
        return row["id"]


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────
def upsert_document(
    symbol: str,
    doc_type: str,
    year: Optional[int],
    title: str,
    file_path: str,
    file_size_kb: int = 0,
) -> int:
    company_id = upsert_company(symbol)
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id, ingested FROM documents WHERE file_path = ?", (file_path,)
        ).fetchone()

        if existing:
            return existing["id"]

        cur = conn.execute(
            """INSERT INTO documents
               (company_id, symbol, doc_type, year, title, file_path, file_size_kb)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (company_id, symbol.upper(), doc_type, year, title, file_path, file_size_kb),
        )
        return cur.lastrowid


def mark_document_ingested(doc_id: int, total_chunks: int, total_pages: int = 0):
    with get_conn() as conn:
        conn.execute(
            """UPDATE documents
               SET ingested=1, total_chunks=?, total_pages=?
               WHERE id=?""",
            (total_chunks, total_pages, doc_id),
        )


def mark_document_failed(doc_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET ingested=2 WHERE id=?", (doc_id,)
        )


def get_pending_documents(symbol: str = None, doc_type: str = None) -> List[Dict]:
    query = "SELECT * FROM documents WHERE ingested=0"
    params = []
    if symbol:
        query += " AND symbol=?"
        params.append(symbol.upper())
    if doc_type:
        query += " AND doc_type=?"
        params.append(doc_type)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def is_already_ingested(file_path: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT ingested FROM documents WHERE file_path=?", (file_path,)
        ).fetchone()
        return row is not None and row["ingested"] == 1


# ─────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────
def insert_chunk(
    doc_id: int,
    chroma_id: str,
    collection: str,
    chunk_index: int,
    chunk_type: str = "prose",
    section: str = None,
    speaker: str = None,
    page_start: int = None,
    page_end: int = None,
    word_count: int = 0,
):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO chunks
               (doc_id, chroma_id, collection, chunk_index, chunk_type,
                section, speaker, page_start, page_end, word_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, chroma_id, collection, chunk_index, chunk_type,
             section, speaker, page_start, page_end, word_count),
        )


def get_chunks_for_doc(doc_id: int) -> List[Dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Ingestion log
# ─────────────────────────────────────────────
def log_ingestion(
    symbol: str,
    doc_type: str,
    file_path: str,
    status: str,
    message: str = "",
    chunks_created: int = 0,
    duration_sec: float = 0.0,
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ingestion_log
               (symbol, doc_type, file_path, status, message, chunks_created, duration_sec)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, doc_type, file_path, status, message, chunks_created, duration_sec),
        )


# ─────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────
def get_stats() -> Dict[str, Any]:
    with get_conn() as conn:
        companies = conn.execute("SELECT COUNT(*) as n FROM companies").fetchone()["n"]
        docs = conn.execute("SELECT COUNT(*) as n FROM documents WHERE ingested=1").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        by_type = conn.execute(
            "SELECT doc_type, COUNT(*) as n FROM documents WHERE ingested=1 GROUP BY doc_type"
        ).fetchall()
        return {
            "companies": companies,
            "documents_ingested": docs,
            "total_chunks": chunks,
            "by_type": {r["doc_type"]: r["n"] for r in by_type},
        }