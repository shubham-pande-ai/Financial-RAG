"""
fusion/fusion_layer.py

Layer 3 of the Quant CoPilot Intent Decomposition Pipeline.

Receives a BridgeResult (SQL rows + vector chunks) and produces a
FusionResult — a structured, annotated context object ready for the
synthesis prompt builder.

What it does
────────────
1. ORGANISE
   Group SQL rows by sub_type → symbol → year into a clean lookup dict.
   Tag each vector chunk by its signal type (annual report prose vs
   concall management commentary vs concall analyst Q&A).

2. EXTRACT numeric claims from concall chunks
   Regex pulls (value, unit) pairs from management sentences that mention
   the same metric keywords as the SQL atoms (e.g. "EBITDA margin of 22%"
   → {metric: ebitda_margin, value: 22.0, unit: "%"}).

3. CROSS-REFERENCE actuals vs guidance
   For each SQL atom that has both a reported figure and a concall claim:
     - CONFIRM   : claim within ±CONFIRM_THRESHOLD of actual (default 10 %)
     - CONTRADICT: claim diverges by > CONTRADICT_THRESHOLD (default 15 %)
     - FORWARD   : claim is clearly forward-looking (no actual to compare)
     - UNMATCHED : SQL has data but no concall claim found (gap)

4. SURFACE insights
   Contradictions and confirmations are surfaced as FusionInsight objects
   so the synthesis prompt can highlight them without the LLM having to
   re-discover them.

5. BUILD structured context dict
   fusion_result.to_context_dict() returns everything the prompt builder
   needs: metric tables, concall quotes, insight annotations, year range.

Output schema
─────────────
FusionResult
  .metric_table    : List[MetricRow]   — one row per (symbol, sub_type, year, value)
  .concall_claims  : List[ConcallClaim]— numeric claims extracted from chunks
  .insights        : List[FusionInsight]— CONFIRM / CONTRADICT / FORWARD / UNMATCHED
  .annual_chunks   : List[RetrievedChunk] — AR prose chunks (MD&A, risk factors)
  .concall_chunks  : List[RetrievedChunk] — concall chunks (sorted by score)
  .errors          : List[str]          — non-fatal issues encountered
  .to_context_dict()→ dict              — ready for prompt builder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pipeline.retrieval.retriever import RetrievedChunk
from schema_bridge.schema_bridge import BridgeResult, SqlAtomResult, VectorAtomResult
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────
CONFIRM_THRESHOLD     = 0.10   # within 10 %  → CONFIRM
CONTRADICT_THRESHOLD  = 0.15   # beyond 15 %  → CONTRADICT


# ─────────────────────────────────────────────────────────────────────────────
# Metric signal map
# sub_type → (sql_value_column, concall_keyword_list, unit_hint)
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_SIGNALS: Dict[str, Tuple[str, List[str], str]] = {
    "revenue":        ("sales",                ["revenue", "sales", "topline"],          "crore"),
    "net_profit":     ("net_profit",           ["profit", "pat", "net income"],           "crore"),
    "ebitda":         ("ebitda",               ["ebitda", "operating profit"],            "crore"),
    "ebitda_margin":  ("ebitda_margin_pct",    ["ebitda margin", "operating margin"],    "%"),
    "opm":            ("opm_pct",              ["opm", "operating margin"],              "%"),
    "gross_margin":   ("gross_margin_pct",     ["gross margin"],                         "%"),
    "net_margin":     ("net_profit_margin_pct",["net margin", "profit margin"],          "%"),
    "roce":           ("roce_pct",             ["roce", "return on capital"],             "%"),
    "roe":            ("roe_pct",              ["roe", "return on equity"],               "%"),
    "net_debt":       ("net_debt",             ["net debt", "debt"],                     "crore"),
    "capex":          ("capex",                ["capex", "capital expenditure", "capex plan"], "crore"),
    "fcf":            ("free_cash_flow",       ["free cash flow", "fcf"],                "crore"),
    "ocf":            ("cfo",                  ["operating cash flow", "cash from operations"], "crore"),
    "borrowings":     ("borrowings",           ["borrowings", "total debt"],             "crore"),
    "eps":            ("eps_annual",           ["eps", "earnings per share"],            "rs"),
    "revenue_cagr":   ("sales_cagr_3y",        ["revenue cagr", "sales cagr", "revenue growth"], "%"),
    "profit_cagr":    ("profit_cagr_3y",       ["profit cagr", "earnings cagr"],         "%"),
}

# Keywords that signal a forward-looking statement (don't compare to actuals)
_FORWARD_KEYWORDS = re.compile(
    r"\b(we expect|we anticipate|going forward|guidance|target|forecast|"
    r"next year|next quarter|h1|h2|fy2[0-9]|by fy|by march|plan to|"
    r"we are confident|we aim|we intend|projected|aspire)\b",
    re.IGNORECASE,
)

# Regex: extract (number_str, range_high_or_None, unit_or_None)
# Only captures when a unit (%, crore, etc.) is present — avoids bare years
_NUMBER_RE = re.compile(
    r"(?:Rs\.?\s*)?"
    r"(\d[\d,]*(?:\.\d+)?)"                     # primary number
    r"(?:\s*[-–]\s*(\d[\d,]*(?:\.\d+)?))?"      # optional range high
    r"\s*"
    r"(%|crore|cr\.?\b|lakh|billion|million|rs\.?\b|x\b)",
    re.IGNORECASE,
)

# Management roles whose claims carry higher weight
_MGMT_ROLES = {"management", "ceo", "cfo", "md", "cmd"}


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricRow:
    """One reported financial figure from the SQL channel."""
    symbol:    str
    sub_type:  str
    metric:    str           # human label
    year:      Optional[int]
    period:    str           # e.g. "2024-03-31"
    value:     Optional[float]
    value_col: str           # which DB column the value came from
    unit:      str           # "crore" / "%" / "rs" / ""
    raw_row:   Dict[str, Any] = field(default_factory=dict)


class InsightType(str, Enum):
    CONFIRM     = "CONFIRM"      # concall claim aligns with reported figure
    CONTRADICT  = "CONTRADICT"   # concall claim diverges significantly
    FORWARD     = "FORWARD"      # forward-looking claim, no actual to compare
    UNMATCHED   = "UNMATCHED"    # SQL has data but no concall claim found


@dataclass
class ConcallClaim:
    """A numeric claim extracted from a concall chunk."""
    sub_type:     str
    metric:       str
    value_low:    float
    value_high:   Optional[float]   # for ranges like "22-23%"
    unit:         str
    is_forward:   bool
    speaker:      str
    speaker_role: str
    year:         Optional[int]
    source_text:  str               # sentence containing the claim
    chunk_score:  float


@dataclass
class FusionInsight:
    """A cross-referenced finding between SQL actuals and concall claims."""
    insight_type:  InsightType
    sub_type:      str
    metric:        str
    symbol:        str
    year:          Optional[int]
    sql_value:     Optional[float]
    sql_period:    str
    claim:         Optional[ConcallClaim]
    divergence_pct: Optional[float]  # abs((claim - actual) / actual * 100)
    note:          str               # human-readable summary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_year_from_period(period: str) -> Optional[int]:
    """
    '2024-03-31' → 2024  (Indian FY ends in March, year = calendar year)
    '2023-12-31' → 2024  (fallback: if month <= 3 bump year by 0, else +1)
    """
    if not period:
        return None
    try:
        parts = period.split("-")
        y, m = int(parts[0]), int(parts[1])
        # Indian FY: Apr–Mar.  Period end in Jan/Feb/Mar belongs to that FY year.
        return y if m >= 4 else y
    except Exception:
        return None


def _strip_commas(s: str) -> float:
    return float(s.replace(",", ""))


def _extract_numeric_claims(
    chunk: RetrievedChunk,
    sub_type: str,
    keywords: List[str],
    unit_hint: str,
) -> List[ConcallClaim]:
    """
    Scan chunk text sentence-by-sentence.
    For each sentence that contains a keyword AND a number with matching unit,
    emit a ConcallClaim.
    """
    claims: List[ConcallClaim] = []
    meta   = chunk.metadata
    speaker      = meta.get("speaker", "Unknown")
    speaker_role = (meta.get("speaker_role") or "unknown").lower()
    year         = meta.get("year")

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", chunk.text)

    for sent in sentences:
        sent_lower = sent.lower()

        # Must contain at least one metric keyword
        if not any(kw in sent_lower for kw in keywords):
            continue

        # Extract all (value, range_high, unit) with matching unit
        for m in _NUMBER_RE.finditer(sent):
            raw_val  = m.group(1)
            raw_high = m.group(2)
            raw_unit = (m.group(3) or "").lower().rstrip(".")

            # Normalise unit
            if raw_unit in ("%",):
                unit = "%"
            elif raw_unit in ("crore", "cr"):
                unit = "crore"
            elif raw_unit in ("lakh",):
                unit = "lakh"
            elif raw_unit in ("billion",):
                unit = "billion"
            elif raw_unit in ("million",):
                unit = "million"
            elif raw_unit in ("rs", "rs."):
                unit = "rs"
            else:
                unit = raw_unit

            # Only keep if unit matches the metric's expected unit (loose match)
            if unit_hint == "%" and unit != "%":
                continue
            if unit_hint == "crore" and unit not in ("crore", "lakh", "billion"):
                continue

            try:
                val_low  = _strip_commas(raw_val)
                val_high = _strip_commas(raw_high) if raw_high else None
            except ValueError:
                continue

            is_fwd = bool(_FORWARD_KEYWORDS.search(sent))

            claims.append(ConcallClaim(
                sub_type     = sub_type,
                metric       = sub_type,
                value_low    = val_low,
                value_high   = val_high,
                unit         = unit,
                is_forward   = is_fwd,
                speaker      = speaker,
                speaker_role = speaker_role,
                year         = year,
                source_text  = sent.strip()[:300],
                chunk_score  = chunk.score,
            ))

    return claims


def _get_period_col(row: Dict[str, Any]) -> str:
    """Return whichever date column is present in the row."""
    for col in ("period_end", "as_of_date", "annual_end", "date",
                "snapshot_date", "action_date", "effective_date"):
        if col in row and row[col]:
            return str(row[col])
    return ""


def _pct_divergence(actual: float, claim: float) -> float:
    """Absolute percentage divergence between actual and claim."""
    if actual == 0:
        return 0.0
    return abs((claim - actual) / actual)


# ─────────────────────────────────────────────────────────────────────────────
# Core fusion logic
# ─────────────────────────────────────────────────────────────────────────────

class FusionLayer:
    """
    Cross-references SQL actuals with concall management claims.

    Usage:
        fusion = FusionLayer()
        result = fusion.fuse(bridge_result)

        for insight in result.insights:
            print(insight.insight_type, insight.metric, insight.note)

        context = result.to_context_dict()   # → synthesis prompt builder
    """

    def __init__(
        self,
        confirm_threshold:    float = CONFIRM_THRESHOLD,
        contradict_threshold: float = CONTRADICT_THRESHOLD,
    ):
        self.confirm_threshold    = confirm_threshold
        self.contradict_threshold = contradict_threshold

    # ── Main entry point ──────────────────────────────────────────────────────

    def fuse(self, bridge: BridgeResult) -> "FusionResult":
        errors: List[str] = list(bridge.errors)  # carry forward bridge errors

        # ── Step 1: build MetricRow list from SQL results ─────────────────────
        metric_rows = self._build_metric_rows(bridge.sql_results)

        # ── Step 2: split vector results into annual vs concall chunks ─────────
        annual_chunks:  List[RetrievedChunk] = []
        concall_chunks: List[RetrievedChunk] = []
        for vr in bridge.vector_results:
            for chunk in vr.chunks:
                if chunk.metadata.get("doc_type") == "concall":
                    concall_chunks.append(chunk)
                else:
                    annual_chunks.append(chunk)

        concall_chunks.sort(key=lambda c: c.score, reverse=True)
        annual_chunks.sort(key=lambda c: c.score, reverse=True)

        # ── Step 3: extract numeric claims from concall chunks ─────────────────
        concall_claims = self._extract_all_claims(bridge.sql_results, concall_chunks)

        # ── Step 4: cross-reference → insights ───────────────────────────────
        insights = self._cross_reference(metric_rows, concall_claims)

        log.info(
            f"[fusion] {len(metric_rows)} metric rows | "
            f"{len(concall_claims)} concall claims | "
            f"{len(insights)} insights "
            f"({sum(1 for i in insights if i.insight_type == InsightType.CONTRADICT)} contradictions)"
        )

        return FusionResult(
            metric_rows    = metric_rows,
            concall_claims = concall_claims,
            insights       = insights,
            annual_chunks  = annual_chunks,
            concall_chunks = concall_chunks,
            errors         = errors,
        )

    # ── Step 1: MetricRow builder ─────────────────────────────────────────────

    def _build_metric_rows(
        self, sql_results: List[SqlAtomResult]
    ) -> List[MetricRow]:
        rows: List[MetricRow] = []
        for sr in sql_results:
            if sr.error or not sr.rows:
                continue

            atom      = sr.atom
            sub_type  = atom.sub_type
            sig       = _METRIC_SIGNALS.get(sub_type)
            val_col   = sig[0] if sig else (atom.sql_columns[0] if atom.sql_columns else "")
            unit      = sig[2] if sig else ""

            for raw in sr.rows:
                period  = _get_period_col(raw)
                year    = _parse_year_from_period(period)
                value   = raw.get(val_col)
                if value is not None:
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = None

                rows.append(MetricRow(
                    symbol   = str(raw.get("symbol", atom.symbol or "")),
                    sub_type = sub_type,
                    metric   = atom.metric,
                    year     = year,
                    period   = period,
                    value    = value,
                    value_col= val_col,
                    unit     = unit,
                    raw_row  = raw,
                ))
        return rows

    # ── Step 2: claim extractor ───────────────────────────────────────────────

    def _extract_all_claims(
        self,
        sql_results:    List[SqlAtomResult],
        concall_chunks: List[RetrievedChunk],
    ) -> List[ConcallClaim]:
        """
        For each sub_type that has SQL results, scan concall chunks for
        matching numeric claims.
        """
        # Only scan for sub_types that actually have SQL atoms
        active_sub_types = {sr.atom.sub_type for sr in sql_results if not sr.error}

        all_claims: List[ConcallClaim] = []
        for chunk in concall_chunks:
            for sub_type, (val_col, keywords, unit_hint) in _METRIC_SIGNALS.items():
                if sub_type not in active_sub_types:
                    continue
                claims = _extract_numeric_claims(chunk, sub_type, keywords, unit_hint)
                all_claims.extend(claims)

        # Deduplicate: same text + same sub_type → keep highest chunk score
        seen: Dict[str, ConcallClaim] = {}
        for c in all_claims:
            key = f"{c.sub_type}|{c.source_text[:80]}"
            if key not in seen or c.chunk_score > seen[key].chunk_score:
                seen[key] = c
        return list(seen.values())

    # ── Step 3: cross-reference ───────────────────────────────────────────────

    def _cross_reference(
        self,
        metric_rows:    List[MetricRow],
        concall_claims: List[ConcallClaim],
    ) -> List[FusionInsight]:
        insights: List[FusionInsight] = []

        # Index claims by sub_type for fast lookup
        claims_by_sub: Dict[str, List[ConcallClaim]] = {}
        for c in concall_claims:
            claims_by_sub.setdefault(c.sub_type, []).append(c)

        for row in metric_rows:
            sub_type = row.sub_type
            relevant = claims_by_sub.get(sub_type, [])

            if row.value is None:
                continue

            if not relevant:
                # SQL has data but no concall claim extracted
                insights.append(FusionInsight(
                    insight_type  = InsightType.UNMATCHED,
                    sub_type      = sub_type,
                    metric        = row.metric,
                    symbol        = row.symbol,
                    year          = row.year,
                    sql_value     = row.value,
                    sql_period    = row.period,
                    claim         = None,
                    divergence_pct= None,
                    note          = (
                        f"Reported {row.metric} = {row.value:,.1f} {row.unit} "
                        f"({row.period}) — no management commentary found."
                    ),
                ))
                continue

            # Pick the best claim: prefer management role, then highest score
            def _claim_priority(c: ConcallClaim) -> Tuple:
                role_score = 1 if c.speaker_role in _MGMT_ROLES else 0
                return (role_score, c.chunk_score)

            best = max(relevant, key=_claim_priority)

            # Forward-looking claim — no numeric comparison to actuals
            if best.is_forward:
                val_str = (f"{best.value_low}–{best.value_high}"
                           if best.value_high else str(best.value_low))
                insights.append(FusionInsight(
                    insight_type  = InsightType.FORWARD,
                    sub_type      = sub_type,
                    metric        = row.metric,
                    symbol        = row.symbol,
                    year          = row.year,
                    sql_value     = row.value,
                    sql_period    = row.period,
                    claim         = best,
                    divergence_pct= None,
                    note          = (
                        f"Reported {row.metric} = {row.value:,.1f} {row.unit} "
                        f"({row.period}). "
                        f"{best.speaker} guidance: {val_str} {best.unit}. "
                        f"[Forward-looking — {best.source_text[:120]}]"
                    ),
                ))
                continue

            # Historical claim: compare to actual
            claim_val = (best.value_low + best.value_high) / 2 \
                        if best.value_high else best.value_low

            # Unit normalisation: lakh → crore
            if best.unit == "lakh" and row.unit == "crore":
                claim_val /= 100.0

            div = _pct_divergence(row.value, claim_val)

            if div <= self.confirm_threshold:
                itype = InsightType.CONFIRM
                note  = (
                    f"✓ {row.metric} confirmed: reported {row.value:,.1f} {row.unit}, "
                    f"{best.speaker} stated {claim_val:,.1f} {best.unit} "
                    f"(divergence {div*100:.1f}%)."
                )
            elif div >= self.contradict_threshold:
                itype = InsightType.CONTRADICT
                note  = (
                    f"⚠ {row.metric} CONTRADICTION: reported {row.value:,.1f} {row.unit} "
                    f"but {best.speaker} stated {claim_val:,.1f} {best.unit} "
                    f"(divergence {div*100:.1f}% > threshold {self.contradict_threshold*100:.0f}%)."
                )
            else:
                itype = InsightType.CONFIRM   # within tolerance
                note  = (
                    f"~ {row.metric} broadly aligned: reported {row.value:,.1f} {row.unit}, "
                    f"{best.speaker} stated {claim_val:,.1f} {best.unit} "
                    f"(divergence {div*100:.1f}%)."
                )

            insights.append(FusionInsight(
                insight_type  = itype,
                sub_type      = sub_type,
                metric        = row.metric,
                symbol        = row.symbol,
                year          = row.year,
                sql_value     = row.value,
                sql_period    = row.period,
                claim         = best,
                divergence_pct= round(div * 100, 2),
                note          = note,
            ))

        # Also surface forward-looking claims with NO matching SQL row
        matched_sub_types = {r.sub_type for r in metric_rows}
        for sub_type, claims in claims_by_sub.items():
            if sub_type in matched_sub_types:
                continue
            for c in claims:
                if c.is_forward:
                    val_str = (f"{c.value_low}–{c.value_high}"
                               if c.value_high else str(c.value_low))
                    insights.append(FusionInsight(
                        insight_type  = InsightType.FORWARD,
                        sub_type      = sub_type,
                        metric        = sub_type,
                        symbol        = c.year and "" or "",
                        year          = c.year,
                        sql_value     = None,
                        sql_period    = "",
                        claim         = c,
                        divergence_pct= None,
                        note          = (
                            f"{c.speaker} guidance on {sub_type}: "
                            f"{val_str} {c.unit}. "
                            f"[No reported actual available — {c.source_text[:100]}]"
                        ),
                    ))

        return insights


# ─────────────────────────────────────────────────────────────────────────────
# FusionResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    metric_rows:    List[MetricRow]       = field(default_factory=list)
    concall_claims: List[ConcallClaim]    = field(default_factory=list)
    insights:       List[FusionInsight]   = field(default_factory=list)
    annual_chunks:  List[RetrievedChunk]  = field(default_factory=list)
    concall_chunks: List[RetrievedChunk]  = field(default_factory=list)
    errors:         List[str]             = field(default_factory=list)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def contradictions(self) -> List[FusionInsight]:
        return [i for i in self.insights if i.insight_type == InsightType.CONTRADICT]

    def confirmations(self) -> List[FusionInsight]:
        return [i for i in self.insights if i.insight_type == InsightType.CONFIRM]

    def forward_guidance(self) -> List[FusionInsight]:
        return [i for i in self.insights if i.insight_type == InsightType.FORWARD]

    def unmatched(self) -> List[FusionInsight]:
        return [i for i in self.insights if i.insight_type == InsightType.UNMATCHED]

    # ── Context dict for synthesis prompt builder ──────────────────────────────

    def to_context_dict(self) -> Dict[str, Any]:
        """
        Serialise everything the synthesis prompt builder needs.

        Structure:
          {
            "metric_table":    [ {symbol, sub_type, year, value, unit, period}, ... ],
            "contradictions":  [ {metric, symbol, year, sql_value, claim_value, note}, ... ],
            "confirmations":   [ {metric, note}, ... ],
            "forward_guidance":[ {metric, speaker, claim_text, note}, ... ],
            "unmatched":       [ {metric, symbol, year, sql_value, note}, ... ],
            "concall_quotes":  [ {score, speaker, role, year, text}, ... ],
            "annual_excerpts": [ {score, section, year, text}, ... ],
            "errors":          [ str, ... ],
          }
        """
        def _row_dict(r: MetricRow) -> dict:
            return {
                "symbol":   r.symbol,
                "sub_type": r.sub_type,
                "metric":   r.metric,
                "year":     r.year,
                "period":   r.period,
                "value":    r.value,
                "unit":     r.unit,
            }

        def _insight_dict(i: FusionInsight) -> dict:
            d: Dict[str, Any] = {
                "type":       i.insight_type.value,
                "metric":     i.metric,
                "symbol":     i.symbol,
                "year":       i.year,
                "sql_value":  i.sql_value,
                "sql_period": i.sql_period,
                "note":       i.note,
            }
            if i.claim:
                d["claim_value"] = i.claim.value_low
                d["claim_high"]  = i.claim.value_high
                d["claim_unit"]  = i.claim.unit
                d["claim_text"]  = i.claim.source_text
                d["speaker"]     = i.claim.speaker
            if i.divergence_pct is not None:
                d["divergence_pct"] = i.divergence_pct
            return d

        return {
            "metric_table":    [_row_dict(r) for r in self.metric_rows],
            "contradictions":  [_insight_dict(i) for i in self.contradictions()],
            "confirmations":   [_insight_dict(i) for i in self.confirmations()],
            "forward_guidance":[_insight_dict(i) for i in self.forward_guidance()],
            "unmatched":       [_insight_dict(i) for i in self.unmatched()],
            "concall_quotes": [
                {
                    "score":   round(c.score, 4),
                    "speaker": c.metadata.get("speaker", ""),
                    "role":    c.metadata.get("speaker_role", ""),
                    "year":    c.metadata.get("year"),
                    "text":    c.text[:500],
                }
                for c in self.concall_chunks[:10]
            ],
            "annual_excerpts": [
                {
                    "score":   round(c.score, 4),
                    "section": c.metadata.get("section", ""),
                    "year":    c.metadata.get("year"),
                    "text":    c.text[:500],
                }
                for c in self.annual_chunks[:10]
            ],
            "errors": self.errors,
        }