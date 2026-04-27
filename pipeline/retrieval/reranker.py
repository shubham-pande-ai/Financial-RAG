"""
pipeline/retrieval/reranker.py

Cross-encoder re-ranking using ms-marco-MiniLM-L-6-v2.
Free, local, CPU-friendly (~200ms for 20 candidates).
Significantly boosts retrieval precision on long financial docs.
"""

from typing import List, Optional

from config.settings import RERANKER_MODEL, ANNUAL_RETRIEVAL, CONCALL_RETRIEVAL
from pipeline.retrieval.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

_reranker = None  # lazy load


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        log.info(f"Loading re-ranker: {RERANKER_MODEL} (first time only)")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        log.info("Re-ranker loaded")
    return _reranker


def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    doc_type: str,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Re-rank retrieved chunks using cross-encoder.
    Returns top_k chunks sorted by re-rank score (descending).
    """
    if not chunks:
        return []

    if top_k is None:
        top_k = (
            ANNUAL_RETRIEVAL["top_k_rerank"]
            if doc_type == "annual_report"
            else CONCALL_RETRIEVAL["top_k_rerank"]
        )

    reranker = _get_reranker()

    # Cross-encoder takes (query, passage) pairs
    pairs = [(query, chunk.text[:512]) for chunk in chunks]  # truncate to 512 tokens

    log.debug(f"  Re-ranking {len(pairs)} candidates")
    scores = reranker.predict(pairs)

    # Attach re-rank scores and sort
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    top = reranked[:top_k]

    log.info(f"  Re-ranked: {len(chunks)} → top {len(top)}")
    return top