"""
config/settings.py
Central configuration for the Financial RAG system.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Load .env file automatically (works on Windows + Linux + Mac)
# Supports both:
#   GROQ_API_KEY=gsk_...          (standard dotenv format)
#   export GROQ_API_KEY="gsk_..." (bash export format)
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
            # strip leading 'export ' if present
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:   # don't overwrite real env vars
                os.environ[key] = value

_load_dotenv()

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"
DB_PATH = BASE_DIR / "financial_rag.db"
LOG_DIR = BASE_DIR / "logs"

# Where screener_downloader.py saves PDFs
SCREENER_DOCS_DIR = BASE_DIR / "screener_docs"

# ─────────────────────────────────────────────
# Groq (free LLM)
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"   # best free model on Groq
GROQ_FALLBACK_MODEL = "gemma2-9b-it"     # fallback if rate limited
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.1                   # low = more factual

# ─────────────────────────────────────────────
# Embeddings (free, local, CPU-friendly)
# ─────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # 384 dim, ~80ms/chunk on CPU
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 64               # tune down if RAM is tight

# ─────────────────────────────────────────────
# Re-ranker (free, local, CPU-friendly)
# ─────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # L-6 = faster on CPU

# ─────────────────────────────────────────────
# ChromaDB collections
# ─────────────────────────────────────────────
CHROMA_ANNUAL_COLLECTION = "annual_reports"
CHROMA_CONCALL_COLLECTION = "concalls"

# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
ANNUAL_REPORT = {
    "chunk_size": 512,          # tokens (approx words × 1.3)
    "chunk_overlap": 64,
    "table_row_group": 8,       # rows per table chunk
    "inject_section_header": True,
}

CONCALL = {
    "chunk_size": 1024,         # bigger — full Q&A exchanges
    "chunk_overlap": 128,
    "respect_speaker_turns": True,
    "inject_speaker_label": True,
}

# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
ANNUAL_RETRIEVAL = {
    "top_k_vector": 20,
    "top_k_bm25": 10,
    "top_k_rerank": 5,          # final chunks sent to LLM
    "search_type": "hybrid",    # vector + BM25
}

CONCALL_RETRIEVAL = {
    "top_k_vector": 10,
    "top_k_rerank": 4,
    "search_type": "semantic",  # pure vector
}

# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────
MIN_CHUNK_WORDS = 20            # discard tiny chunks
MAX_PDF_SIZE_MB = 100           # skip files above this