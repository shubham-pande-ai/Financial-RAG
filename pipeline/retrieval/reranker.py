"""
pipeline/retrieval/reranker.py

Score fix: ms-marco-MiniLM outputs large negative logits for financial text
(-15 to -20 range). sigmoid(-15) = 0.000 — useless for ranking.

Solution: softmax normalization across the batch.
softmax converts any range of logits into a proper 0-1 probability
distribution that preserves relative ordering.

Example:
  raw logits:  [-18.2, -16.5, -19.1, -15.3]
  softmax:     [0.041,  0.213,  0.015,  0.731]
  → clear winner, meaningful scores
"""

import math
from typing import List, Optional

from config.settings import RERANKER_MODEL, ANNUAL_RETRIEVAL, CONCALL_RETRIEVAL
from pipeline.retrieval.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

_reranker     = None
_reranker_pid = None


def _get_reranker():
    import os
    global _reranker, _reranker_pid
    pid = os.getpid()
    if _reranker is not None and _reranker_pid == pid:
        return _reranker
    from sentence_transformers import CrossEncoder
    log.info(f"Loading re-ranker: {RERANKER_MODEL} (PID {pid})")
    _reranker     = CrossEncoder(RERANKER_MODEL, max_length=512)
    _reranker_pid = pid
    log.info("Re-ranker loaded")
    return _reranker


def _softmax(logits: List[float]) -> List[float]:
    """
    Numerically stable softmax.
    Handles very negative logits correctly — sigmoid cannot.
    """
    max_l = max(logits)                          # stability: shift by max
    exps  = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    doc_type: str,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Re-rank chunks using cross-encoder.
    Scores are softmax-normalised → always 0-1, sum to 1 across batch.
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
    pairs    = [(query, chunk.text[:512]) for chunk in chunks]

    raw_scores      = reranker.predict(pairs).tolist()
    normed_scores   = _softmax(raw_scores)

    for chunk, score in zip(chunks, normed_scores):
        chunk.score = score

    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    top      = reranked[:top_k]

    log.info(
        f"  Re-ranked: {len(chunks)} -> top {len(top)} | "
        f"score range [{top[-1].score:.4f} - {top[0].score:.4f}]"
    )
    return top


def rerank_separate(
    query: str,
    annual_chunks: List[RetrievedChunk],
    concall_chunks: List[RetrievedChunk],
    annual_top_k: int,
    concall_top_k: int,
) -> List[RetrievedChunk]:
    """
    Rerank each collection independently so concall prose
    cannot displace annual report financial data chunks.
    Annual chunks come first in the merged output.
    """
    top_annual  = rerank(query, annual_chunks,  "annual_report", top_k=annual_top_k)
    top_concall = rerank(query, concall_chunks, "concall",       top_k=concall_top_k)

    merged = top_annual + top_concall
    log.info(
        f"  Merged: {len(top_annual)} annual + {len(top_concall)} concall "
        f"= {len(merged)} total chunks to LLM"
    )
    return merged