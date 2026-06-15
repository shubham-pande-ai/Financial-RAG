"""
config/settings.py
Central configuration for the Financial RAG system.

Migration notes (v2):
  - SQLite  → MySQL 8.0   (metadata store)
  - MinIO               (PDF source — replaces screener_docs/ local dir)
  - ChromaDB → Qdrant   (vector store)
  - pdfplumber → Docling (PDF extraction)

EMBEDDING_MODEL is still FinLang/finance-embeddings-investopedia (768-dim).
If you change it, drop + recreate both Qdrant collections and re-ingest.
"""

import os
from pathlib import Path


# ─────────────────────────────────────────────
# Load .env file automatically
# ─────────────────────────────────────────────
def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


# ─────────────────────────────────────────────
# Paths (local only — no more SQLite, no more screener_docs)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR  = BASE_DIR / "logs"


# ─────────────────────────────────────────────
# MySQL  (metadata store: companies, documents, chunks, ingestion_log)
# ─────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = os.getenv("DB_NAME",     "ai_hedge_fund")
DB_USER     = os.getenv("DB_USER",     "quant_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "quant_password")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))


# ─────────────────────────────────────────────
# MinIO  (PDF source — annual reports + concalls)
# ─────────────────────────────────────────────
# Bucket key layout (mirrors old screener_docs/ tree):
#   {SYMBOL}/{doc_type}/{year}/{filename}.pdf
#   e.g.  RELIANCE/annual_report/2024/reliance_ar_2024.pdf
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE     = os.getenv("MINIO_SECURE",     "false").lower() == "true"
MINIO_BUCKET     = os.getenv("MINIO_BUCKET",     "quant-docs")


# ─────────────────────────────────────────────
# Qdrant  (vector store — replaces ChromaDB)
# ─────────────────────────────────────────────
QDRANT_HOST             = os.getenv("QDRANT_HOST",     "localhost")
QDRANT_PORT             = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY          = os.getenv("QDRANT_API_KEY",  "")   # empty = no auth (local)
QDRANT_ANNUAL_COLLECTION  = "annual_reports"
QDRANT_CONCALL_COLLECTION = "concalls"


# ─────────────────────────────────────────────
# Groq
# ─────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "gemma2-9b-it"
GROQ_MAX_TOKENS     = 1024
GROQ_TEMPERATURE    = 0.1


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
# FinLang/finance-embeddings-investopedia → 768-dim vectors.
# ⚠  Changing EMBEDDING_MODEL requires dropping Qdrant collections + re-ingesting.
EMBEDDING_MODEL      = "FinLang/finance-embeddings-investopedia"
EMBEDDING_DIM        = 768
EMBEDDING_BATCH_SIZE = 64


# ─────────────────────────────────────────────
# Re-ranker
# ─────────────────────────────────────────────
RERANKER_MODEL    = "rerank-2"                    # Voyage Rerank-2.5
RERANKER_FALLBACK = "BAAI/bge-reranker-v2-m3"
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
ANNUAL_REPORT = {
    "chunk_size":            512,
    "chunk_overlap":          64,
    "table_row_group":         8,
    "inject_section_header": True,
}

CONCALL = {
    "chunk_size":              1024,
    "chunk_overlap":            128,
    "respect_speaker_turns":   True,
    "inject_speaker_label":    True,
}


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
ANNUAL_RETRIEVAL = {
    "top_k_vector": 30,
    "top_k_bm25":   10,
    "top_k_rerank": 15,
    "search_type":  "hybrid",
}

CONCALL_RETRIEVAL = {
    "top_k_vector": 25,
    "top_k_rerank":  8,
    "search_type":  "hybrid",
}

COMBINED_ANNUAL_CHUNKS  = 12
COMBINED_CONCALL_CHUNKS =  6


# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────
MIN_CHUNK_WORDS  = 20
MAX_PDF_SIZE_MB  = 100

# Temp dir for MinIO downloads during ingestion (cleaned up after each PDF)
import tempfile
INGEST_TMP_DIR = Path(tempfile.gettempdir()) / "finrag_ingest"


# ─────────────────────────────────────────────
# Structured financial DB  (Quant Copilot MySQL — same DB_* vars above)
# Tables: sm_*, price_*, fundamentals_*, etc. populated by the ETL repo.
# metric_engine.py queries this directly.
# ─────────────────────────────────────────────
# No separate path needed — it's the same MySQL instance.
# FINANCE_DB_PATH (old SQLite ref) is removed.