"""
db/database.py
MySQL metadata store — shared across both RAG pipelines.

Tables:
  rag_companies      — company master
  rag_documents      — one row per PDF file (stored in MinIO)
  rag_chunks         — one row per chunk (links MySQL ↔ Qdrant)
  rag_ingestion_log  — track what has been processed

Prefixed with `rag_` to coexist safely with the ETL schema (sm_*, price_*, etc.)
in the same `ai_hedge_fund` database.

Dependencies:
    pip install mysql-connector-python
"""

import mysql.connector
from mysql.connector import pooling, Error as MySQLError
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from config.settings import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_POOL_SIZE
)
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Schema  (MySQL DDL)
# ─────────────────────────────────────────────
_SCHEMA_STMTS = [
    """
    CREATE TABLE IF NOT EXISTS rag_companies (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        symbol     VARCHAR(20) UNIQUE NOT NULL,
        name       VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_documents (
        id           INT AUTO_INCREMENT PRIMARY KEY,
        company_id   INT NOT NULL,
        symbol       VARCHAR(20) NOT NULL,
        doc_type     VARCHAR(30) NOT NULL,
        year         SMALLINT,
        title        VARCHAR(500),
        minio_key    VARCHAR(1000) UNIQUE,
        file_size_kb INT DEFAULT 0,
        total_pages  INT DEFAULT 0,
        total_chunks INT DEFAULT 0,
        ingested     TINYINT DEFAULT 0,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_rag_doc_company FOREIGN KEY (company_id)
            REFERENCES rag_companies(id),
        CONSTRAINT chk_doc_type CHECK (doc_type IN ('annual_report', 'concall')),
        CONSTRAINT chk_ingested CHECK (ingested IN (0, 1, 2)),
        INDEX idx_rag_doc_symbol (symbol),
        INDEX idx_rag_doc_type   (doc_type),
        INDEX idx_rag_doc_ingested (ingested)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_chunks (
        id           INT AUTO_INCREMENT PRIMARY KEY,
        doc_id       INT NOT NULL,
        qdrant_id    VARCHAR(36) UNIQUE NOT NULL,
        collection   VARCHAR(100) NOT NULL,
        chunk_index  INT NOT NULL,
        chunk_type   VARCHAR(30),
        section      VARCHAR(500),
        speaker      VARCHAR(200),
        page_start   INT,
        page_end     INT,
        word_count   INT DEFAULT 0,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_rag_chunk_doc FOREIGN KEY (doc_id)
            REFERENCES rag_documents(id),
        INDEX idx_rag_chunk_doc    (doc_id),
        INDEX idx_rag_chunk_qdrant (qdrant_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS rag_ingestion_log (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        symbol         VARCHAR(20) NOT NULL,
        doc_type       VARCHAR(30),
        minio_key      VARCHAR(1000),
        status         VARCHAR(20),
        message        TEXT,
        chunks_created INT DEFAULT 0,
        duration_sec   FLOAT,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
]


# ─────────────────────────────────────────────
# Connection pool (module-level singleton)
# ─────────────────────────────────────────────
_pool: Optional[pooling.MySQLConnectionPool] = None


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="finrag",
            pool_size=DB_POOL_SIZE,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            charset="utf8mb4",
            autocommit=False,
            connection_timeout=10,
        )
        log.info(f"MySQL pool created — {DB_HOST}:{DB_PORT}/{DB_NAME} (size={DB_POOL_SIZE})")
    return _pool


@contextmanager
def get_conn():
    pool = _get_pool()
    conn = pool.get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create RAG metadata tables if they don't exist."""
    with get_conn() as conn:
        cursor = conn.cursor()
        for stmt in _SCHEMA_STMTS:
            cursor.execute(stmt)
        cursor.close()
    log.info("RAG metadata tables initialised in MySQL")


# ─────────────────────────────────────────────
# Companies
# ─────────────────────────────────────────────
def upsert_company(symbol: str, name: str = None) -> int:
    sym = symbol.upper()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT IGNORE INTO rag_companies (symbol, name) VALUES (%s, %s)",
            (sym, name or sym),
        )
        cur.execute("SELECT id FROM rag_companies WHERE symbol = %s", (sym,))
        row = cur.fetchone()
        cur.close()
        return row[0]


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────
def upsert_document(
    symbol:      str,
    doc_type:    str,
    year:        Optional[int],
    title:       str,
    minio_key:   str,
    file_size_kb: int = 0,
) -> int:
    company_id = upsert_company(symbol)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM rag_documents WHERE minio_key = %s",
            (minio_key,),
        )
        existing = cur.fetchone()
        if existing:
            cur.close()
            return existing[0]

        cur.execute(
            """INSERT INTO rag_documents
               (company_id, symbol, doc_type, year, title, minio_key, file_size_kb)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (company_id, symbol.upper(), doc_type, year, title, minio_key, file_size_kb),
        )
        doc_id = cur.lastrowid
        cur.close()
        return doc_id


def mark_document_ingested(doc_id: int, total_chunks: int, total_pages: int = 0):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """UPDATE rag_documents
               SET ingested=1, total_chunks=%s, total_pages=%s
               WHERE id=%s""",
            (total_chunks, total_pages, doc_id),
        )
        cur.close()


def mark_document_failed(doc_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE rag_documents SET ingested=2 WHERE id=%s", (doc_id,)
        )
        cur.close()


def get_pending_documents(symbol: str = None, doc_type: str = None) -> List[Dict]:
    query  = "SELECT * FROM rag_documents WHERE ingested=0"
    params = []
    if symbol:
        query += " AND symbol=%s"
        params.append(symbol.upper())
    if doc_type:
        query += " AND doc_type=%s"
        params.append(doc_type)
    with get_conn() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return rows


def is_already_ingested(minio_key: str) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT ingested FROM rag_documents WHERE minio_key=%s", (minio_key,)
        )
        row = cur.fetchone()
        cur.close()
        return row is not None and row[0] == 1


# ─────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────
def insert_chunk(
    doc_id:      int,
    qdrant_id:   str,
    collection:  str,
    chunk_index: int,
    chunk_type:  str  = "prose",
    section:     str  = None,
    speaker:     str  = None,
    page_start:  int  = None,
    page_end:    int  = None,
    word_count:  int  = 0,
):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT IGNORE INTO rag_chunks
               (doc_id, qdrant_id, collection, chunk_index, chunk_type,
                section, speaker, page_start, page_end, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (doc_id, qdrant_id, collection, chunk_index, chunk_type,
             section, speaker, page_start, page_end, word_count),
        )
        cur.close()


def get_chunks_for_doc(doc_id: int) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM rag_chunks WHERE doc_id=%s ORDER BY chunk_index",
            (doc_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows


# ─────────────────────────────────────────────
# Ingestion log
# ─────────────────────────────────────────────
def log_ingestion(
    symbol:        str,
    doc_type:      str,
    minio_key:     str,
    status:        str,
    message:       str   = "",
    chunks_created: int  = 0,
    duration_sec:  float = 0.0,
):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO rag_ingestion_log
               (symbol, doc_type, minio_key, status, message, chunks_created, duration_sec)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (symbol, doc_type, minio_key, status, message, chunks_created, duration_sec),
        )
        cur.close()


# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────
def get_stats() -> Dict[str, Any]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rag_companies")
        companies = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM rag_documents WHERE ingested=1")
        docs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM rag_chunks")
        chunks = cur.fetchone()[0]

        cur.execute(
            "SELECT doc_type, COUNT(*) FROM rag_documents "
            "WHERE ingested=1 GROUP BY doc_type"
        )
        by_type = {row[0]: row[1] for row in cur.fetchall()}
        cur.close()

        return {
            "companies":          companies,
            "documents_ingested": docs,
            "total_chunks":       chunks,
            "by_type":            by_type,
        }