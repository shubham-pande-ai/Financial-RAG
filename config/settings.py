"""
config/settings.py
Central configuration for the Financial RAG system.

FIXES applied in this version:
  [FIX 3] COMBINED_ANNUAL_CHUNKS raised 6 → 12
          COMBINED_CONCALL_CHUNKS raised 3 →  6
  [FIX 3] ANNUAL_RETRIEVAL top_k_rerank raised 8 → 15
          CONCALL_RETRIEVAL top_k_rerank raised 4 →  8
  [FIX 3] CONCALL_RETRIEVAL top_k_vector raised 15 → 25

  ══════════════════════════════════════════════════════════════════════
  [FIX ROOT-CAUSE] ChromaDB dimension mismatch  384 vs 768
  ══════════════════════════════════════════════════════════════════════
  ERROR:
    chromadb.errors.InvalidArgumentError:
      Collection expecting embedding with dimension of 384, got 768

  ROOT CAUSE:
    chroma_store/ was created with all-MiniLM-L6-v2 (384-dim).
    EMBEDDING_MODEL was later changed to
    FinLang/finance-embeddings-investopedia (768-dim) but the store
    was never deleted — ChromaDB locked the dimension to 384 on first
    write and now rejects 768-dim query vectors at runtime.

  ONE-TIME FIX — run these commands, then re-ingest:
    Windows : rmdir /s /q chroma_store
    Linux   : rm -rf chroma_store/
    Then    : python ingest.py --symbol ADANIPORTS  (repeat per symbol)
  ══════════════════════════════════════════════════════════════════════

  [FIX RERANKER] CONCALL_RETRIEVAL search_type changed "semantic" → "hybrid"
                 Concall now uses BM25 + vector fusion (same as annual).
                 Keyword-heavy queries (e.g. "capex FY24") benefit from BM25.

  [FIX RERANKER-FALLBACK] Qwen3-Reranker-8B added as local fallback when
                 VOYAGE_API_KEY is absent or Voyage quota is exhausted.
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
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "gemma2-9b-it"
GROQ_MAX_TOKENS     = 1024
GROQ_TEMPERATURE    = 0.1

# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
# FinLang/finance-embeddings-investopedia → 768-dim vectors.
#
# ⚠  If you see "Collection expecting dimension 384, got 768":
#    → Delete chroma_store/ and re-run ingest.py for every symbol.
#    Changing EMBEDDING_MODEL always requires full re-ingestion.
EMBEDDING_MODEL      = "FinLang/finance-embeddings-investopedia"
EMBEDDING_DIM        = 768    # must match model output; used for validation
EMBEDDING_BATCH_SIZE = 64

# ─────────────────────────────────────────────
# Re-ranker
# ─────────────────────────────────────────────
# Primary  : Voyage Rerank-2.5 — finance/SEC domain-tuned, 32k ctx.
#            Requires VOYAGE_API_KEY → https://dash.voyageai.com/
#
# Fallback : Qwen/Qwen3-Reranker-8B (local HuggingFace) — free, 32k ctx.
#            Auto-activated when VOYAGE_API_KEY is absent or Voyage returns
#            a rate-limit / quota error (HTTP 429/402).
#            Install: pip install transformers torch accelerate
#            RAM  : ~16 GB fp32 | ~8 GB fp16 | ~4 GB int8 (with bitsandbytes)
#            Speed: ~5-20s/batch on CPU; GPU recommended for production.
RERANKER_MODEL    = "rerank-2"                # Voyage Rerank-2.5
RERANKER_FALLBACK = "Qwen/Qwen3-Reranker-8B" # local HF fallback
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")

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
    "top_k_rerank": 15,       # was 8
    "search_type":  "hybrid",
}

CONCALL_RETRIEVAL = {
    "top_k_vector": 25,       # was 15
    "top_k_rerank":  8,       # was 4
    "search_type":  "hybrid", # FIX: was "semantic" — now uses BM25 fusion too
}

# "both" mode chunk budget  [FIX 3]
COMBINED_ANNUAL_CHUNKS  = 12    # was 6
COMBINED_CONCALL_CHUNKS =  6    # was 3

# ─────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────
MIN_CHUNK_WORDS = 20
MAX_PDF_SIZE_MB = 100