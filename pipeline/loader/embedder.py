"""
pipeline/loader/embedder.py

Singleton embedding model — loaded ONCE per process, never reloaded.
Fix: module-level _model with explicit process-level guard using os.getpid().
"""

import os
from typing import List

from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from utils.logger import get_logger

log = get_logger(__name__)

# (model_instance, pid_it_was_loaded_in)
# If PID changes (new subprocess), reload. Otherwise reuse.
_model       = None
_loaded_pid  = None


def _get_model():
    global _model, _loaded_pid
    current_pid = os.getpid()

    if _model is not None and _loaded_pid == current_pid:
        return _model  # already loaded in this process — skip

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
    batch_size = EMBEDDING_BATCH_SIZE
    all_embs   = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embs  = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embs.extend(embs.tolist())

    return all_embs


def embed_query(query: str) -> List[float]:
    """Embed a single query — same model, no reload."""
    return embed_texts([query])[0]