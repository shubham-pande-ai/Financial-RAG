"""
pipeline/retrieval/retriever.py

Retrieval strategies:
  Annual reports → Hybrid (vector + BM25 keyword fusion)
  Concalls       → Semantic (pure vector with metadata filters)

BM25 is done in-memory on the candidate set from ChromaDB — no extra DB needed.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import math
import re

from config.settings import ANNUAL_RETRIEVAL, CONCALL_RETRIEVAL
from pipeline.loader.embedder import embed_query
from pipeline.loader.chroma_loader import query_collection
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float           # fused / final score
    vector_score: float
    bm25_score: float
    metadata: Dict[str, Any]


# ─────────────────────────────────────────────
# BM25 implementation (in-memory, no dependency)
# ─────────────────────────────────────────────
class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenized = [self._tokenize(d) for d in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.N, 1)
        self._build_idf()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9₹%]+\b", text.lower())

    def _build_idf(self):
        from collections import Counter
        df = Counter()
        for doc in self.tokenized:
            df.update(set(doc))
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        query_terms = self._tokenize(query)
        doc = self.tokenized[doc_idx]
        doc_len = len(doc)
        from collections import Counter
        term_freq = Counter(doc)

        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = term_freq.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            score += self.idf[term] * numerator / denominator
        return score

    def get_scores(self, query: str) -> List[float]:
        return [self.score(query, i) for i in range(self.N)]


# ─────────────────────────────────────────────
# Score normalisation
# ─────────────────────────────────────────────
def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ─────────────────────────────────────────────
# Build ChromaDB where filter
# ─────────────────────────────────────────────
def _build_where(
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[tuple] = None,
) -> Optional[dict]:
    conditions = []

    if symbol:
        conditions.append({"symbol": {"$eq": symbol.upper()}})

    if year:
        conditions.append({"year": {"$eq": year}})
    elif year_range:
        start, end = year_range
        conditions.append({"year": {"$gte": start}})
        conditions.append({"year": {"$lte": end}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ─────────────────────────────────────────────
# Annual Report: Hybrid retrieval
# ─────────────────────────────────────────────
def retrieve_annual(
    query: str,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[tuple] = None,
) -> List[RetrievedChunk]:
    cfg = ANNUAL_RETRIEVAL
    query_vec = embed_query(query)
    where = _build_where(symbol, year, year_range)

    # Step 1: Vector search
    result = query_collection(
        doc_type="annual_report",
        query_embedding=query_vec,
        top_k=cfg["top_k_vector"],
        where=where,
    )

    if not result["ids"] or not result["ids"][0]:
        log.warning("No vector results for annual report query")
        return []

    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    # ChromaDB cosine distance → similarity (1 - distance)
    vector_scores = [1 - d for d in distances]

    # Step 2: BM25 on the candidate set
    bm25 = BM25(docs)
    bm25_scores = bm25.get_scores(query)

    # Step 3: Normalise + fuse (equal weight)
    norm_vec = _minmax_normalize(vector_scores)
    norm_bm25 = _minmax_normalize(bm25_scores)
    fused = [(v + b) / 2 for v, b in zip(norm_vec, norm_bm25)]

    # Step 4: Sort by fused score
    ranked = sorted(
        zip(ids, docs, metas, fused, vector_scores, bm25_scores),
        key=lambda x: x[3],
        reverse=True,
    )

    results = []
    for chunk_id, text, meta, score, vscore, bscore in ranked:
        results.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=score,
            vector_score=vscore,
            bm25_score=bscore,
            metadata=meta,
        ))

    log.info(f"  Annual retrieval: {len(results)} candidates (hybrid)")
    return results


# ─────────────────────────────────────────────
# Concall: Semantic retrieval
# ─────────────────────────────────────────────
def retrieve_concall(
    query: str,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[tuple] = None,
    speaker_role: Optional[str] = None,
) -> List[RetrievedChunk]:
    cfg = CONCALL_RETRIEVAL
    query_vec = embed_query(query)

    where_conditions = []
    if symbol:
        where_conditions.append({"symbol": {"$eq": symbol.upper()}})
    if year:
        where_conditions.append({"year": {"$eq": year}})
    elif year_range:
        where_conditions.append({"year": {"$gte": year_range[0]}})
        where_conditions.append({"year": {"$lte": year_range[1]}})
    if speaker_role:
        where_conditions.append({"speaker_role": {"$eq": speaker_role}})

    where = None
    if len(where_conditions) == 1:
        where = where_conditions[0]
    elif len(where_conditions) > 1:
        where = {"$and": where_conditions}

    result = query_collection(
        doc_type="concall",
        query_embedding=query_vec,
        top_k=cfg["top_k_vector"],
        where=where,
    )

    if not result["ids"] or not result["ids"][0]:
        log.warning("No vector results for concall query")
        return []

    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    vector_scores = [1 - d for d in distances]

    results = []
    for chunk_id, text, meta, vscore in zip(ids, docs, metas, vector_scores):
        results.append(RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=vscore,
            vector_score=vscore,
            bm25_score=0.0,
            metadata=meta,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    log.info(f"  Concall retrieval: {len(results)} candidates (semantic)")
    return results


# ─────────────────────────────────────────────
# Unified entry
# ─────────────────────────────────────────────
def retrieve(
    query: str,
    doc_type: str,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[tuple] = None,
    speaker_role: Optional[str] = None,
) -> List[RetrievedChunk]:
    if doc_type == "annual_report":
        return retrieve_annual(query, symbol, year, year_range)
    elif doc_type == "concall":
        return retrieve_concall(query, symbol, year, year_range, speaker_role)
    else:
        # Cross-collection: run both and merge
        annual = retrieve_annual(query, symbol, year, year_range)
        concall = retrieve_concall(query, symbol, year, year_range)
        merged = annual + concall
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged