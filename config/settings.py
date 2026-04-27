"""
config/settings.py
Central configuration for the Financial RAG system.

FIXES applied in this version:
  [FIX 3] COMBINED_ANNUAL_CHUNKS raised 6 → 12
          COMBINED_CONCALL_CHUNKS raised 3 →  6
          A 3-year × 3-metric query needs at least 9 unique data chunks
          to fully answer — the old limits caused structural answer gaps.
  [FIX 3] ANNUAL_RETRIEVAL top_k_rerank raised 8 → 15
          CONCALL_RETRIEVAL top_k_rerank raised 4 →  8
          More candidates survive reranking before the LLM sees them.
  [FIX 3] CONCALL_RETRIEVAL top_k_vector raised 15 → 25
          Matches annual pipeline scale for multi-year queries.
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
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent
DATA_DIR          = BASE_DIR / "data"
CHROMA_DIR        = BASE_DIR / "chroma_store"
DB_PATH           = BASE_DIR / "financial_rag.db"
LOG_DIR           = BASE_DIR / "logs"
SCREENER_DOCS_DIR = BASE_DIR / "screener_docs"

# ─────────────────────────────────────────────
# Groq
# ─────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "gemma2-9b-it"
GROQ_MAX_TOKENS    = 1024
GROQ_TEMPERATURE   = 0.1

# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
EMBEDDING_DIM        = 384
EMBEDDING_BATCH_SIZE = 64

# ─────────────────────────────────────────────
# Re-ranker
# ─────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─────────────────────────────────────────────
# ChromaDB collections
# ─────────────────────────────────────────────
CHROMA_ANNUAL_COLLECTION  = "annual_reports"
CHROMA_CONCALL_COLLECTION = "concalls"

# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
ANNUAL_REPORT = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "table_row_group": 8,
    "inject_section_header": True,
}

CONCALL = {
    "chunk_size": 1024,
    "chunk_overlap": 128,
    "respect_speaker_turns": True,
    "inject_speaker_label": True,
}

# ─────────────────────────────────────────────
# Retrieval  [FIX 3]
# ─────────────────────────────────────────────
ANNUAL_RETRIEVAL = {
    "top_k_vector": 30,
    "top_k_bm25":   10,
    "top_k_rerank": 15,     # was 8 — more candidates survive to LLM
    "search_type":  "hybrid",
}

CONCALL_RETRIEVAL = {
    "top_k_vector": 25,     # was 15 — matches annual scale for multi-year
    "top_k_rerank":  8,     # was 4
    "search_type":  "semantic",
}

# "both" mode chunk budget  [FIX 3]
# Rule of thumb: N_years × N_metrics × 1.5 safety margin
# For 3-year queries: 3 × 3 × 1.5 ≈ 13 → use 12 annual + 6 concall
COMBINED_ANNUAL_CHUNKS  = 12    # was 6
COMBINED_CONCALL_CHUNKS =  6    # was 3

# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────
MIN_CHUNK_WORDS = 20
MAX_PDF_SIZE_MB = 100