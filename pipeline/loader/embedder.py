"""
pipeline/loader/embedder.py

Singleton embedding model — loaded ONCE per process, never reloaded.
PID guard ensures subprocesses reload safely.

build_embedding_text is imported from chunker to avoid circular imports.
Re-exported here so qdrant_loader can do a single import from embedder.
"""

import os
from typing import List

from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from utils.logger import get_logger

log = get_logger(__name__)

_model      = None
_loaded_pid = None


def _get_model():
    global _model, _loaded_pid
    current_pid = os.getpid()

    if _model is not None and _loaded_pid == current_pid:
        return _model

    log.info(f"Loading embedding model: {EMBEDDING_MODEL} (PID {current_pid})")
    from sentence_transformers import SentenceTransformer
    _model      = SentenceTransformer(EMBEDDING_MODEL)
    _loaded_pid = current_pid
    log.info("Embedding model ready")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    model      = _get_model()
    all_embs   = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        embs  = model.encode(
            batch,
            batch_size          = EMBEDDING_BATCH_SIZE,
            show_progress_bar   = False,
            convert_to_numpy    = True,
            normalize_embeddings = True,
        )
        all_embs.extend(embs.tolist())

    return all_embs


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


# Re-export so qdrant_loader only needs: from pipeline.loader.embedder import ...
from pipeline.loader.chunker import build_embedding_text  # noqa: E402, F401