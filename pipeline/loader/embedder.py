"""
pipeline/loader/embedder.py

Wraps sentence-transformers for embedding chunks.
Model: all-MiniLM-L6-v2  (384 dim, CPU-friendly, ~80ms/chunk)
Free — no API key, runs fully local.
"""

from typing import List
import numpy as np

from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from utils.logger import get_logger

log = get_logger(__name__)

_model = None  # lazy load


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {EMBEDDING_MODEL} (first time only)")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model loaded")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings.
    Returns list of float vectors (384 dim each).
    """
    if not texts:
        return []

    model = _get_model()
    batch_size = EMBEDDING_BATCH_SIZE

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        log.debug(f"  Embedding batch {i//batch_size + 1}: {len(batch)} texts")
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine via dot product
        )
        all_embeddings.extend(embeddings.tolist())

    return all_embeddings


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]