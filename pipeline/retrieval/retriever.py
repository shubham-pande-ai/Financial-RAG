"""
pipeline/retrieval/retriever.py

Key fixes in this version:
  - Hard year filtering via ChromaDB $in operator (not just range)
  - Recency boost: newer FY scores higher even at equal semantic similarity
  - Fallback: if year-filtered results < MIN_RESULTS, widens to all years
  - No-year queries default to latest 3 FY (2023, 2024, 2025)
"""

import re
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from config.settings import ANNUAL_RETRIEVAL, CONCALL_RETRIEVAL
from pipeline.loader.embedder import embed_query
from pipeline.loader.chroma_loader import query_collection
from utils.logger import get_logger

log = get_logger(__name__)

CURRENT_FY       = 2025          # latest FY in your ingested data
DEFAULT_LOOKBACK = 3             # "recent" queries → last 3 FY by default
MIN_RESULTS      = 5             # fallback threshold


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    vector_score: float
    bm25_score: float
    metadata: Dict[str, Any]


# ─────────────────────────────────────────────
# Year intent parser
# ─────────────────────────────────────────────
def parse_year_intent(query: str) -> Optional[List[int]]:
    """
    Returns an explicit list of years to filter on (for $in operator),
    or None if no year intent found.

    Examples:
      "last 3 years"        → [2023, 2024, 2025]
      "last 5 years"        → [2021, 2022, 2023, 2024, 2025]
      "FY2023-25"           → [2023, 2024, 2025]
      "FY24"                → [2024]
      "2023 and 2024"       → [2023, 2024]
      (no year hint)        → [2023, 2024, 2025]  ← DEFAULT to recent
    """
    q = query.lower()

    # "last N years" / "past N years"
    m = re.search(r"(?:last|past|previous|recent)\s+(\d+)\s+years?", q)
    if m:
        n = int(m.group(1))
        return list(range(CURRENT_FY - n + 1, CURRENT_FY + 1))

    # "FY2023-25" or "FY23-25"
    m = re.search(r"fy[\s\-]*(\d{2,4})[\s\-\/–]+(\d{2,4})", q)
    if m:
        y1 = int(m.group(1)); y2 = int(m.group(2))
        y1 = 2000 + y1 if y1 < 100 else y1
        y2 = 2000 + y2 if y2 < 100 else y2
        return list(range(min(y1, y2), max(y1, y2) + 1))

    # "2023 to 2025" / "2023-2025"
    m = re.search(r"(20\d{2})\s*(?:to|\-|–)\s*(20\d{2})", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return list(range(min(y1, y2), max(y1, y2) + 1))

    # "FY24" or "FY2024" — single year
    m = re.search(r"\bfy[\s\-]*(\d{2,4})\b", q)
    if m:
        y = int(m.group(1))
        return [2000 + y if y < 100 else y]

    # "financial year 2024"
    m = re.search(r"(?:financial\s+year|year)\s+(20\d{2})", q)
    if m:
        return [int(m.group(1))]

    # Multiple plain 4-digit years
    years = list({int(y) for y in re.findall(r"\b(20\d{2})\b", q)
                  if 2010 <= int(y) <= CURRENT_FY})
    if years:
        return sorted(years)

    # No year hint → default to recent N years
    log.info(f"  No year hint in query — defaulting to last {DEFAULT_LOOKBACK} FY")
    return list(range(CURRENT_FY - DEFAULT_LOOKBACK + 1, CURRENT_FY + 1))


# ─────────────────────────────────────────────
# Recency boost
# ─────────────────────────────────────────────
def recency_boost(year: int) -> float:
    """
    Small additive boost so FY2025 > FY2015 at equal semantic similarity.
    Range: 0.00 (year=2000) to ~0.025 (year=2025)
    Kept small so it nudges rather than overrides relevance.
    """
    return max(0.0, (year - 2000) * 0.001)


# ─────────────────────────────────────────────
# BM25 (in-memory, zero dependencies)
# ─────────────────────────────────────────────
class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokenized = [self._tok(d) for d in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.N, 1)
        self._build_idf()

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9%]+\b", text.lower())

    def _build_idf(self):
        from collections import Counter
        df = Counter()
        for doc in self.tokenized:
            df.update(set(doc))
        self.idf = {t: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
                    for t, f in df.items()}

    def get_scores(self, query: str) -> List[float]:
        from collections import Counter
        qtoks = self._tok(query)
        scores = []
        for doc in self.tokenized:
            tf = Counter(doc)
            dl = len(doc)
            s = sum(
                self.idf.get(t, 0) *
                tf.get(t, 0) * (self.k1 + 1) /
                (tf.get(t, 0) + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                for t in qtoks
            )
            scores.append(s)
        return scores


def _minmax(scores: List[float]) -> List[float]:
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ─────────────────────────────────────────────
# ChromaDB where filter — uses $in for explicit year list
# ─────────────────────────────────────────────
def _build_where(symbol: Optional[str], years: Optional[List[int]]) -> Optional[dict]:
    conditions = []
    if symbol:
        conditions.append({"symbol": {"$eq": symbol.upper()}})
    if years:
        if len(years) == 1:
            conditions.append({"year": {"$eq": years[0]}})
        else:
            conditions.append({"year": {"$in": years}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


# ─────────────────────────────────────────────
# Core annual retrieval
# ─────────────────────────────────────────────
def _run_annual_query(query: str, symbol: Optional[str],
                      years: Optional[List[int]], top_k: int) -> List[RetrievedChunk]:
    query_vec = embed_query(query)
    where = _build_where(symbol, years)

    result = query_collection(
        doc_type="annual_report",
        query_embedding=query_vec,
        top_k=top_k,
        where=where,
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    ids      = result["ids"][0]
    docs     = result["documents"][0]
    metas    = result["metadatas"][0]
    dists    = result["distances"][0]
    vscore   = [1 - d for d in dists]

    bm25     = BM25(docs)
    bscore   = bm25.get_scores(query)

    nv = _minmax(vscore)
    nb = _minmax(bscore)
    fused = [(v + b) / 2 for v, b in zip(nv, nb)]

    results = []
    for cid, text, meta, fs, vs, bs in zip(ids, docs, metas, fused, vscore, bscore):
        yr = meta.get("year", 0)
        # Recency boost applied AFTER fusion
        final = fs + recency_boost(yr)
        results.append(RetrievedChunk(
            chunk_id=cid, text=text, score=final,
            vector_score=vs, bm25_score=bs, metadata=meta,
        ))

    results.sort(key=lambda c: c.score, reverse=True)
    return results


def retrieve_annual(
    query: str,
    symbol: Optional[str] = None,
    years: Optional[List[int]] = None,
) -> List[RetrievedChunk]:
    cfg = ANNUAL_RETRIEVAL

    results = _run_annual_query(query, symbol, years, cfg["top_k_vector"])

    # Fallback: if filtered results are too few, widen to all years
    if len(results) < MIN_RESULTS and years:
        log.warning(f"  Only {len(results)} results for years={years}, "
                    f"falling back to all years")
        results = _run_annual_query(query, symbol, years=None,
                                    top_k=cfg["top_k_vector"])

    log.info(f"  Annual retrieval: {len(results)} candidates | years={years}")
    return results


# ─────────────────────────────────────────────
# Core concall retrieval
# ─────────────────────────────────────────────
def retrieve_concall(
    query: str,
    symbol: Optional[str] = None,
    years: Optional[List[int]] = None,
    speaker_role: Optional[str] = None,
) -> List[RetrievedChunk]:
    cfg = CONCALL_RETRIEVAL
    query_vec = embed_query(query)

    conditions = []
    if symbol:
        conditions.append({"symbol": {"$eq": symbol.upper()}})
    if years:
        conditions.append({"year": {"$in": years}} if len(years) > 1
                          else {"year": {"$eq": years[0]}})
    if speaker_role:
        conditions.append({"speaker_role": {"$eq": speaker_role}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    result = query_collection(
        doc_type="concall",
        query_embedding=query_vec,
        top_k=cfg["top_k_vector"],
        where=where,
    )

    if not result["ids"] or not result["ids"][0]:
        # fallback — no year filter
        if years:
            log.warning("  Concall: no results with year filter, falling back")
            where_fallback = {"symbol": {"$eq": symbol.upper()}} if symbol else None
            result = query_collection(
                doc_type="concall",
                query_embedding=query_vec,
                top_k=cfg["top_k_vector"],
                where=where_fallback,
            )
        if not result["ids"] or not result["ids"][0]:
            return []

    ids    = result["ids"][0]
    docs   = result["documents"][0]
    metas  = result["metadatas"][0]
    dists  = result["distances"][0]
    vscore = [1 - d for d in dists]

    results = []
    for cid, text, meta, vs in zip(ids, docs, metas, vscore):
        yr = meta.get("year", 0)
        results.append(RetrievedChunk(
            chunk_id=cid, text=text,
            score=vs + recency_boost(yr),
            vector_score=vs, bm25_score=0.0, metadata=meta,
        ))

    results.sort(key=lambda c: c.score, reverse=True)
    log.info(f"  Concall retrieval: {len(results)} candidates | years={years}")
    return results


# ─────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────
def retrieve(
    query: str,
    doc_type: str,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
    speaker_role: Optional[str] = None,
):
    """
    Returns:
      - List[RetrievedChunk]                    for doc_type in (annual_report, concall)
      - Tuple[List[RetrievedChunk], List[...]]  for doc_type == "both"

    Year resolution priority:
      1. Explicit --year CLI flag
      2. Explicit --year-range CLI flag
      3. Parsed from query text
      4. Default: latest 3 FY
    """
    # Resolve to explicit year list
    if year:
        years = [year]
    elif year_range:
        years = list(range(year_range[0], year_range[1] + 1))
    else:
        years = parse_year_intent(query)

    log.info(f"  Resolved year filter: {years}")

    if doc_type == "annual_report":
        return retrieve_annual(query, symbol, years)
    elif doc_type == "concall":
        return retrieve_concall(query, symbol, years, speaker_role)
    else:
        annual  = retrieve_annual(query, symbol, years)
        concall = retrieve_concall(query, symbol, years, speaker_role)
        return annual, concall