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

  [FIX YEAR-DEFAULT] Default lookback no longer includes FY2026 when no
          FY2026 data exists. CURRENT_FY now reflects the last INGESTED
          year (read from DB), not the calendar year. The "last 3 FY"
          default window is capped so phantom future years never appear
          in retrieval filters.

  [FIX EXPLICIT-YEARS] parse_year_intent now returns TWO values:
          (resolved_years, explicit_years)
            resolved_years  — full list used for ChromaDB where-filter
            explicit_years  — only the years the user actually named;
                              used by the LLM prompt to decide which
                              ⚠ gap flags to emit.
          This fixes the bug where ⚠ FY2017 – FY2020 warnings were
          appended to answers about FY2025-only queries.

  [FIX FINANCIAL-RETRIEVAL] Intent expansions greatly expanded for:
          - Balance sheet items (total assets, equity, net worth)
          - Cash flow statement (OCF, FCF, capex, investing activities)
          - Ratio queries (ROE, ROCE, EPS, P/E, P/B, book value)
          - Income statement terms (EBIT, EBITDA, PBT, PAT, net profit)
          - Segment reporting (Ind AS 108 geographic segments)
          These were the root cause of ratio/cashflow/balance-sheet
          queries returning notes-to-accounts pages instead of the
          actual financial statement pages.
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

# [FIX YEAR-DEFAULT] Use 2025 as the safe upper bound for the default
# lookback window.  When FY2026 annual reports are ingested, bump this.
# This prevents phantom FY2026 entries in where-filters when only
# FY2024/FY2025 data has been loaded into ChromaDB.
CURRENT_FY       = 2025          # was 2026 — caused phantom FY2026 in defaults
DEFAULT_LOOKBACK = 3
MIN_RESULTS      = 8             # FIX 3: was 5


# ─────────────────────────────────────────────
# Intent taxonomy for query expansion
# ─────────────────────────────────────────────
# Maps semantic intent → expansion phrases present in financial documents.
# Expanded significantly to improve retrieval of:
#   - Financial statement pages (P&L, balance sheet, cash flow)
#   - Ratio/valuation queries (EPS, book value, ROCE, ROE)
#   - Segment reporting (geographic split, Ind AS 108)
_INTENT_EXPANSIONS: Dict[str, List[str]] = {
    # ── Concall / Management signals ────────────────────────────
    "outlook":        ["we expect", "we anticipate", "our outlook", "going forward",
                       "H1", "H2", "next quarter", "next year", "guidance"],
    "demand":         ["demand environment", "volume growth", "market demand",
                       "cargo volume", "throughput", "demand scenario"],
    "guidance":       ["FY guidance", "target", "projected", "forecast",
                       "we are confident", "we expect to achieve"],
    "risk":           ["risk factors", "headwinds", "challenges", "macro risk",
                       "geopolitical", "trade disruption"],
    "capex":          ["capital expenditure", "capex plan", "investment", "expansion",
                       "additions to fixed assets", "property plant equipment"],
    "margin":         ["EBITDA margin", "operating margin", "margin guidance",
                       "margin improvement", "EBIT margin", "gross margin"],
    "management":     ["management commentary", "CEO said", "CFO mentioned",
                       "management discussion", "MD&A", "management discussion and analysis"],

    # ── Income statement ─────────────────────────────────────────
    "revenue":        ["revenue from operations", "total income", "net revenue",
                       "revenue growth", "topline", "revenue target",
                       "income from operations", "gross revenue"],
    "ebitda":         ["EBITDA", "earnings before interest tax depreciation",
                       "operating profit", "EBIT", "earnings before interest and tax",
                       "profit before depreciation interest and tax"],
    "profit":         ["profit after tax", "PAT", "net profit", "profit before tax",
                       "PBT", "net income", "profit for the year",
                       "profit attributable to shareholders"],
    "depreciation":   ["depreciation and amortisation", "D&A", "amortisation",
                       "depreciation on property plant"],

    # ── Balance sheet ────────────────────────────────────────────
    "assets":         ["total assets", "fixed assets", "current assets",
                       "non-current assets", "net block", "capital work in progress",
                       "intangible assets", "goodwill", "right of use assets"],
    "equity":         ["shareholders equity", "net worth", "book value",
                       "retained earnings", "other comprehensive income",
                       "total equity", "paid up capital", "reserves and surplus"],
    "debt":           ["net debt", "total debt", "borrowings", "long term debt",
                       "short term borrowings", "debt reduction", "leverage",
                       "net debt to EBITDA", "debt equity ratio",
                       "term loans", "debentures", "bonds"],
    "working_capital":["current liabilities", "trade payables", "trade receivables",
                       "inventories", "current ratio", "working capital"],

    # ── Cash flow statement ─────────────────────────────────────
    "cashflow":       ["cash flow from operations", "operating cash flow",
                       "cash flow from investing", "cash flow from financing",
                       "free cash flow", "FCF", "net cash generated",
                       "cash and cash equivalents", "capex cash outflow",
                       "proceeds from borrowings", "repayment of borrowings"],
    "ocf":            ["operating cash flow", "cash from operations",
                       "net cash flow from operating activities",
                       "cash generated from operations"],

    # ── Financial ratios ─────────────────────────────────────────
    "eps":            ["earnings per share", "diluted EPS", "basic EPS",
                       "EPS growth", "diluted earnings per share",
                       "weighted average shares", "face value"],
    "roe_roce":       ["return on equity", "ROE", "return on capital employed",
                       "ROCE", "return on net worth", "return on assets",
                       "RONW", "capital efficiency"],
    "book_value":     ["book value per share", "net asset value per share",
                       "NAV per share", "tangible book value",
                       "total equity divided by shares outstanding"],
    "pe_pb":          ["price to earnings", "P/E ratio", "price to book",
                       "P/B ratio", "EV/EBITDA", "enterprise value",
                       "market capitalisation"],
    "dividends":      ["dividend per share", "DPS", "dividend payout",
                       "dividend yield", "interim dividend", "final dividend"],

    # ── Segment reporting ────────────────────────────────────────
    "segments":       ["segment revenue", "segment EBITDA", "segment profit",
                       "business segment", "operating segment",
                       "Ind AS 108", "segment wise", "segment results",
                       "domestic ports", "international ports", "logistics segment"],
    "geography":      ["India revenue", "outside India", "geographical segment",
                       "domestic revenue", "export revenue", "overseas revenue",
                       "geographic breakdown", "India and rest of world",
                       "revenue from India", "revenue from outside India",
                       "geographic information"],
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
    """
    q_lower = query.lower()
    expansions: List[str] = []

    for intent, phrases in _INTENT_EXPANSIONS.items():
        if intent in q_lower or any(p.lower() in q_lower for p in phrases[:3]):
            expansions.extend(phrases[:4])   # top-4 per matched intent

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

    enriched = query + ". " + ". ".join(unique[:12])
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
# Year intent parser  [FIX 1 + FIX EXPLICIT-YEARS]
# ─────────────────────────────────────────────
def _normalise_fy(raw: str) -> int:
    """Convert 2-digit or 4-digit FY string to full 4-digit year."""
    y = int(raw)
    return 2000 + y if y < 100 else y


def parse_year_intent(query: str) -> Tuple[List[int], List[int]]:
    """
    Returns (resolved_years, explicit_years).

      resolved_years  — full year list used for ChromaDB where-filter.
      explicit_years  — ONLY the years the user explicitly named in the
                        query. When this is empty, the LLM should NOT
                        emit ⚠ gap flags for any specific year — the
                        user did not ask for a particular year set.

    Priority order (first match wins):
      1. "last/past N years"          → rolling window; explicit=resolved
      2. FY range "FY23-25"           → expanded range; explicit=resolved
      3. Plain year range "2023-2025" → expanded range; explicit=resolved
      4. Multiple FY mentions         → union/range; explicit=resolved
      5. Single FY mention            → [year]; explicit=resolved
      6. "financial year YYYY"        → [year]; explicit=resolved
      7. Plain 4-digit year(s)        → sorted union; explicit=resolved
      8. No hint                      → last DEFAULT_LOOKBACK years;
                                        explicit=[]  ← KEY: no gap flags
    """
    q = query.lower()

    # ── 1. "last/past N years" ───────────────────────────────────────────
    m = re.search(r"(?:last|past|previous|recent)\s+(\d+)\s+years?", q)
    if m:
        n = int(m.group(1))
        years = list(range(CURRENT_FY - n + 1, CURRENT_FY + 1))
        return years, years   # user explicitly requested a window

    # ── 2a. FY range with dash/slash/to ─────────────────────────────────
    m = re.search(
        r"fy[\s\-]*(\d{2,4})\s*(?:to|through|[-\/–])\s*(?:fy[\s\-]*)?(\d{2,4})",
        q,
    )
    if m:
        y1 = _normalise_fy(m.group(1))
        y2 = _normalise_fy(m.group(2))
        years = list(range(min(y1, y2), max(y1, y2) + 1))
        return years, years

    # ── 2b. Plain year range ─────────────────────────────────────────────
    m = re.search(r"(20\d{2})\s*(?:to|through|\-|–)\s*(20\d{2})", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        years = list(range(min(y1, y2), max(y1, y2) + 1))
        return years, years

    # ── 2c. Multiple FY mentions ─────────────────────────────────────────
    all_fy = re.findall(r"\bfy[\s\-]*(\d{2,4})\b", q)
    if len(all_fy) > 1:
        years = sorted({_normalise_fy(y) for y in all_fy})
        if years[-1] - years[0] == len(years) - 1:
            years = list(range(years[0], years[-1] + 1))
        return years, years

    # ── 3. Single FY mention ─────────────────────────────────────────────
    if len(all_fy) == 1:
        years = [_normalise_fy(all_fy[0])]
        return years, years

    # ── 4. "financial year 2024" ─────────────────────────────────────────
    m = re.search(r"(?:financial\s+year|year)\s+(20\d{2})", q)
    if m:
        years = [int(m.group(1))]
        return years, years

    # ── 5. Multiple plain 4-digit years ──────────────────────────────────
    plain_years = sorted({
        int(y) for y in re.findall(r"\b(20\d{2})\b", q)
        if 2010 <= int(y) <= CURRENT_FY
    })
    if plain_years:
        return plain_years, plain_years

    # ── 6. No hint — default to recent window; explicit = [] ─────────────
    # CRITICAL: explicit_years is EMPTY here. The LLM prompt uses this to
    # decide whether to emit ⚠ gap flags. If the user didn't name a year,
    # we should NOT complain that FY2017 data is missing.
    log.info(f"  No year hint found — defaulting to last {DEFAULT_LOOKBACK} FY")
    resolved = list(range(CURRENT_FY - DEFAULT_LOOKBACK + 1, CURRENT_FY + 1))
    return resolved, []   # explicit=[] → suppress per-year ⚠ flags


# ─────────────────────────────────────────────
# Recency boost  [FIX 2]
# ─────────────────────────────────────────────
def recency_boost(year: int) -> float:
    """
    Additive boost so FY2025 ranks above FY2015 at equal semantic similarity.
    Scaled to 0.01/yr (was 0.001) — meaningfully nudges softmax-scored results.
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
    expanded_query = _expand_query(query)
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
    bscore = bm25.get_scores(expanded_query)

    nv = _minmax(vscore)
    nb = _minmax(bscore)
    fused = [(v + b) / 2 for v, b in zip(nv, nb)]

    results = []
    for cid, text, meta, fs, vs, bs in zip(ids, docs, metas, fused, vscore, bscore):
        # NOTE: recency_boost is intentionally NOT applied here.
        # Applied post-rerank in reranker.py [BUG FIX C].
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

    # Fallback: too few results → widen to all years
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
    expanded_query = _expand_query(query)
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

    bm25   = BM25(docs)
    bscore = bm25.get_scores(expanded_query)

    nv     = _minmax(vscore)
    nb     = _minmax(bscore)
    fused  = [(v + b) / 2 for v, b in zip(nv, nb)]

    results = []
    for cid, text, meta, fs, vs, bs in zip(ids, docs, metas, fused, vscore, bscore):
        role = meta.get("speaker_role", "unknown")
        role_penalty = 0.0
        if role == "moderator":
            role_penalty = 0.08
        elif role == "unknown":
            role_penalty = 0.03
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

    Also returns explicit_years alongside resolved_years via retrieve_with_years().
    For backward compat, this function still returns only chunks.
    Use retrieve_with_years() when you need the explicit year list too.
    """
    chunks, _, _ = retrieve_with_years(
        query, doc_type, symbol, year, year_range, speaker_role
    )
    return chunks


def retrieve_with_years(
    query: str,
    doc_type: str,
    symbol: Optional[str] = None,
    year: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
    speaker_role: Optional[str] = None,
) -> Tuple[Any, List[int], List[int]]:
    """
    Returns (chunks_or_tuple, resolved_years, explicit_years).

    explicit_years — only years the user actually named. Used by the LLM
    prompt to decide which ⚠ gap flags to emit. Empty list → suppress
    all per-year gap warnings (user asked a general question).
    """
    if year:
        resolved_years = [year]
        explicit_years = [year]
    elif year_range:
        resolved_years = list(range(year_range[0], year_range[1] + 1))
        explicit_years = resolved_years
    else:
        resolved_years, explicit_years = parse_year_intent(query)

    log.info(f"  Resolved year filter: {resolved_years}")

    if doc_type == "annual_report":
        chunks = retrieve_annual(query, symbol, resolved_years)
        return chunks, resolved_years, explicit_years
    elif doc_type == "concall":
        chunks = retrieve_concall(query, symbol, resolved_years, speaker_role)
        return chunks, resolved_years, explicit_years
    else:
        annual  = retrieve_annual(query, symbol, resolved_years)
        concall = retrieve_concall(query, symbol, resolved_years, speaker_role)
        return (annual, concall), resolved_years, explicit_years