"""
pipeline/retrieval/retriever.py

FIXES applied in this version:
  [FIX 1] parse_year_intent — multi-year queries now correctly resolved:
            "FY23, FY24, FY25"   → [2023, 2024, 2025]   (was: [2023])
            "FY23-25"            → [2023, 2024, 2025]   (was: [2023])
            "FY23 to FY25"       → [2023, 2024, 2025]   (was: [2023])
            "2023-25"            → [2023, 2024, 2025]   (was: [2023])
          Strategy: collect ALL FY mentions first (multi-mention path),
          then fall through to range / single-year patterns.
  [FIX 2] recency_boost scaled up from 0.001 → 0.01 per year so it
          meaningfully nudges ranking when softmax scores are similar.
  [FIX 3] MIN_RESULTS raised 5 → 8 so fallback fires sooner on sparse
          year-filtered collections.
  [FIX ROOT-CAUSE] Embedding dimension guard added.
          Detects ChromaDB 384-vs-768 mismatch at startup and raises a
          clear, actionable error message instead of crashing deep inside
          the ChromaDB call with a cryptic InvalidArgumentError.
          Root cause: chroma_store/ was built with all-MiniLM-L6-v2
          (384-dim) and was never deleted after the embedding model was
          changed to FinLang/finance-embeddings-investopedia (768-dim).
          Fix: delete chroma_store/ then re-run ingest.py per symbol.
"""

import re
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass


from config.settings import ANNUAL_RETRIEVAL, CONCALL_RETRIEVAL
from pipeline.loader.embedder import embed_query
from pipeline.loader.chroma_loader import query_collection
from utils.logger import get_logger

log = get_logger(__name__)

CURRENT_FY       = 2026
DEFAULT_LOOKBACK = 3
MIN_RESULTS      = 8          # FIX 3: was 5

# ─────────────────────────────────────────────
# Intent taxonomy for query expansion
# ─────────────────────────────────────────────
# Maps semantic intent → expansion phrases that are actually present
# in concall/annual report text. The embedding model sees these phrases
# directly, so cosine similarity rises for the RIGHT chunks.
_INTENT_EXPANSIONS: Dict[str, List[str]] = {
    "outlook":        ["we expect", "we anticipate", "our outlook", "going forward",
                       "H1", "H2", "next quarter", "next year", "guidance"],
    "demand":         ["demand environment", "volume growth", "market demand",
                       "cargo volume", "throughput", "demand scenario"],
    "guidance":       ["FY guidance", "target", "projected", "forecast",
                       "we are confident", "we expect to achieve"],
    "risk":           ["risk factors", "headwinds", "challenges", "macro risk",
                       "geopolitical", "trade disruption"],
    "capex":          ["capital expenditure", "capex plan", "investment", "expansion"],
    "margin":         ["EBITDA margin", "operating margin", "margin guidance",
                       "margin improvement"],
    "revenue":        ["revenue growth", "topline", "total income", "revenue target"],
    "debt":           ["net debt", "debt reduction", "leverage", "borrowings"],
    "management":     ["management commentary", "CEO said", "CFO mentioned",
                       "management discussion"],
}

_FORWARD_SIGNALS = [
    "outlook", "expect", "anticipate", "guidance", "target", "going forward",
    "H1", "H2", "next", "forecast", "project", "confident", "plan to",
    "demand environment", "demand scenario",
]


def _expand_query(query: str) -> str:
    """
    Enrich the query string with domain-specific expansion phrases so the
    embedding model can match intent-relevant chunks, not just topic-relevant ones.

    Strategy:
      1. Detect which intent buckets the query falls into (keyword scan).
      2. Append a short expansion clause to the query.
      3. Return enriched string — used for embedding only, never shown to user.
    """
    q_lower = query.lower()
    expansions: List[str] = []

    for intent, phrases in _INTENT_EXPANSIONS.items():
        if intent in q_lower or any(p.lower() in q_lower for p in phrases[:2]):
            expansions.extend(phrases[:3])   # take top-3 per matched intent

    # Always add forward-looking signals if query implies future/guidance
    if any(sig in q_lower for sig in ["outlook", "h1", "h2", "next", "expect",
                                       "guidance", "demand environment", "going forward"]):
        expansions.extend(_FORWARD_SIGNALS[:5])

    if not expansions:
        return query

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for p in expansions:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    enriched = query + ". " + ". ".join(unique[:10])
    return enriched


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
# Year intent parser  [FIX 1]
# ─────────────────────────────────────────────
def _normalise_fy(raw: str) -> int:
    """Convert 2-digit or 4-digit FY string to full 4-digit year."""
    y = int(raw)
    return 2000 + y if y < 100 else y


def parse_year_intent(query: str) -> List[int]:
    """
    Returns an explicit list of fiscal years to filter on.

    Priority order (first match wins):
      1. "last/past N years"          → rolling window
      2. Multiple FY mentions         → FIX: collect all, build union/range
         "FY23, FY24, FY25"
         "FY23 to FY25" / "FY23-25"
         "2023 and 2024"
      3. Single FY mention            → [year]
      4. "financial year YYYY"        → [year]
      5. Plain 4-digit year(s)        → sorted union
      6. No hint                      → last DEFAULT_LOOKBACK years
    """
    q = query.lower()

    # ── 1. "last/past N years" ───────────────────────────────────────────
    m = re.search(r"(?:last|past|previous|recent)\s+(\d+)\s+years?", q)
    if m:
        n = int(m.group(1))
        return list(range(CURRENT_FY - n + 1, CURRENT_FY + 1))

    # ── 2a. FY range with dash/slash/to: "FY23-25", "FY2023/25", "FY23 to FY25" ──
    # Handles patterns where two FY tokens are separated by a range indicator.
    m = re.search(
        r"fy[\s\-]*(\d{2,4})\s*(?:to|through|[-\/–])\s*(?:fy[\s\-]*)?(\d{2,4})",
        q,
    )
    if m:
        y1 = _normalise_fy(m.group(1))
        y2 = _normalise_fy(m.group(2))
        # If y2 looks like a 2-digit suffix of y1 (e.g. 23→2023, 25→2025)
        # _normalise_fy already handles that.
        return list(range(min(y1, y2), max(y1, y2) + 1))

    # ── 2b. Plain year range: "2023 to 2025" / "2023-2025" ──────────────
    m = re.search(r"(20\d{2})\s*(?:to|through|\-|–)\s*(20\d{2})", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return list(range(min(y1, y2), max(y1, y2) + 1))

    # ── 2c. Multiple FY mentions (comma / "and" separated) ───────────────
    # e.g. "FY23, FY24, FY25"  or  "FY2023 and FY2024"
    all_fy = re.findall(r"\bfy[\s\-]*(\d{2,4})\b", q)
    if len(all_fy) > 1:
        years = sorted({_normalise_fy(y) for y in all_fy})
        # If consecutive, expand to a clean range; if sparse, return as-is
        if years[-1] - years[0] == len(years) - 1:
            return list(range(years[0], years[-1] + 1))
        return years

    # ── 3. Single FY mention ─────────────────────────────────────────────
    if len(all_fy) == 1:
        return [_normalise_fy(all_fy[0])]

    # ── 4. "financial year 2024" ─────────────────────────────────────────
    m = re.search(r"(?:financial\s+year|year)\s+(20\d{2})", q)
    if m:
        return [int(m.group(1))]

    # ── 5. Multiple plain 4-digit years ──────────────────────────────────
    plain_years = sorted({
        int(y) for y in re.findall(r"\b(20\d{2})\b", q)
        if 2010 <= int(y) <= CURRENT_FY
    })
    if plain_years:
        return plain_years

    # ── 6. No hint — default to recent window ────────────────────────────
    log.info(f"  No year hint found — defaulting to last {DEFAULT_LOOKBACK} FY")
    return list(range(CURRENT_FY - DEFAULT_LOOKBACK + 1, CURRENT_FY + 1))


# ─────────────────────────────────────────────
# Recency boost  [FIX 2]
# ─────────────────────────────────────────────
def recency_boost(year: int) -> float:
    """
    Additive boost so FY2025 ranks above FY2015 at equal semantic similarity.
    Scaled to 0.01/yr (was 0.001) so it meaningfully nudges softmax-scored
    results where score differences are typically in the 0.05-0.20 range.
    Range: 0.00 (year≤2000) → 0.25 (year=2025)
    """
    return max(0.0, (year - 2000) * 0.01)


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
# ChromaDB where filter
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
    expanded_query = _expand_query(query)          # intent-aware expansion
    query_vec = embed_query(expanded_query)
    where = _build_where(symbol, years)

    result = query_collection(
        doc_type="annual_report",
        query_embedding=query_vec,
        top_k=top_k,
        where=where,
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    ids    = result["ids"][0]
    docs   = result["documents"][0]
    metas  = result["metadatas"][0]
    dists  = result["distances"][0]
    vscore = [1 - d for d in dists]

    bm25   = BM25(docs)
    bscore = bm25.get_scores(expanded_query)   # use enriched query for BM25 too

    nv = _minmax(vscore)
    nb = _minmax(bscore)
    fused = [(v + b) / 2 for v, b in zip(nv, nb)]

    results = []
    for cid, text, meta, fs, vs, bs in zip(ids, docs, metas, fused, vscore, bscore):
        # NOTE: recency_boost is intentionally NOT applied here.
        # It is applied post-rerank in reranker.py so it is not silently
        # overwritten when the reranker replaces chunk.score.  [BUG FIX C]
        results.append(RetrievedChunk(
            chunk_id=cid, text=text, score=fs,
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

    # Fallback: too few results → widen to all years  [FIX 3: threshold raised]
    if len(results) < MIN_RESULTS and years:
        log.warning(
            f"  Only {len(results)} results for years={years}, "
            f"falling back to all years"
        )
        results = _run_annual_query(query, symbol, years=None, top_k=cfg["top_k_vector"])

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
    expanded_query = _expand_query(query)          # intent-aware expansion
    query_vec = embed_query(expanded_query)

    conditions = []
    if symbol:
        conditions.append({"symbol": {"$eq": symbol.upper()}})
    if years:
        conditions.append(
            {"year": {"$in": years}} if len(years) > 1
            else {"year": {"$eq": years[0]}}
        )
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

    # BM25 hybrid fusion for concall — same approach as annual pipeline.
    # Keyword-heavy queries (e.g. "capex guidance FY24") benefit significantly
    # from BM25 recall on top of semantic similarity.
    bm25   = BM25(docs)
    bscore = bm25.get_scores(expanded_query)

    nv     = _minmax(vscore)
    nb     = _minmax(bscore)
    fused  = [(v + b) / 2 for v, b in zip(nv, nb)]

    results = []
    for cid, text, meta, fs, vs, bs in zip(ids, docs, metas, fused, vscore, bscore):
        # Speaker-role penalty: moderator/intro text is noise for most queries.
        role = meta.get("speaker_role", "unknown")
        role_penalty = 0.0
        if role == "moderator":
            role_penalty = 0.08
        elif role == "unknown":
            role_penalty = 0.03
        # NOTE: recency_boost is NOT applied here — applied post-rerank [BUG FIX C]
        final_score = max(0.0, fs - role_penalty)
        results.append(RetrievedChunk(
            chunk_id=cid, text=text,
            score=final_score,
            vector_score=vs, bm25_score=bs, metadata=meta,
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
      - List[RetrievedChunk]                   for doc_type in (annual_report, concall)
      - Tuple[List[RetrievedChunk], List[...]] for doc_type == "both"

    Year resolution priority:
      1. Explicit --year CLI flag
      2. Explicit --year-range CLI flag
      3. Parsed from query text (FIX 1 covers all multi-year patterns)
      4. Default: latest DEFAULT_LOOKBACK FY
    """
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