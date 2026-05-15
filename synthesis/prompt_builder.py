"""
synthesis/prompt_builder.py

Layer 5 of the Quant CoPilot Intent Decomposition Pipeline.

Takes a FusionResult (structured context from Layer 4) plus the original
raw vector chunks and assembles a single (system_prompt, user_prompt) pair
ready to pass straight into rag_engine._call_with_retry().

Design goals
────────────
1. FREE-MODEL FRIENDLY
   Prompts are written for sub-10B models (Qwen3-8B, Qwen3-30B, llama-3.3-70b
   on Groq/OpenRouter) — no chain-of-thought forcing, no JSON-output demands,
   clear numbered rules the model can follow in one pass.

2. STRUCTURED CONTEXT FIRST
   SQL metric table is rendered as a compact ASCII table at the top of the
   user prompt so even tiny-context models (Groq ~5.5k tok) see hard numbers
   before prose chunks.  Vector chunks follow in ranked order.

3. CITATION ANCHORS
   Every chunk is tagged [SRC-N] and every SQL row is tagged [SQL-N].
   The system prompt instructs the model to cite using these tags so the
   caller can parse sources without regex.

4. INSIGHT CALLOUTS
   Contradictions, confirmations, and forward-guidance insights from the
   fusion layer are surfaced as ⚠ / ✓ / → callout blocks so the analyst
   LLM notices them without having to re-discover them.

5. TOKEN-BUDGET AWARE
   build() accepts a max_context_chars budget.  SQL table is always kept
   whole.  Vector chunks are trimmed from the bottom (lowest score) if the
   budget is exceeded.  The builder never silently drops SQL data.

6. LEGACY COMPATIBLE
   When FusionResult is None (pure vector-only path) the builder falls back
   to the same prompt format as the existing rag_engine so no code change is
   needed in query.py for the transition period.

Public API
──────────
    from synthesis.prompt_builder import PromptBuilder, BuiltPrompt

    builder = PromptBuilder()
    built   = builder.build(
        query          = "What is ADANIPORTS revenue growth FY23-25?",
        fusion_result  = fusion_result,   # may be None
        chunks         = top_chunks,      # List[RetrievedChunk]
        doc_type       = "both",
        resolved_years = [2023, 2024, 2025],
        explicit_years = [2023, 2024, 2025],
    )

    # Pass straight to rag_engine
    result = _call_with_retry(built.system_prompt, built.user_prompt, entry)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── graceful imports (same pattern as fusion_layer) ──────────────────────────
try:
    from fusion.fusion_layer import FusionResult, InsightType
except ModuleNotFoundError:
    FusionResult  = None   # type: ignore[assignment,misc]
    InsightType   = None   # type: ignore[assignment,misc]

try:
    from pipeline.retrieval.retriever import RetrievedChunk
except ModuleNotFoundError:
    from dataclasses import dataclass as _dc, field as _f
    @_dc
    class RetrievedChunk:                     # type: ignore[no-redef]
        chunk_id: str = ""; text: str = ""; score: float = 0.0
        vector_score: float = 0.0; bm25_score: float = 0.0
        metadata: Dict[str, Any] = _f(default_factory=dict)

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_CHARS_PER_TOKEN   = 3.5          # conservative for dense financial text
_DEFAULT_MAX_CHARS = 80_000       # ~22k tokens — safe for Qwen3-30B free tier
_GROQ_MAX_CHARS    = 18_000       # ~5.1k tokens — Groq free hard cap
_SQL_TABLE_HDR     = "── STRUCTURED FINANCIAL DATA (from SQLite) ──"
_VECTOR_HDR        = "── DOCUMENT EXCERPTS (annual reports / concalls) ──"
_INSIGHTS_HDR      = "── CROSS-CHANNEL INSIGHTS (auto-detected) ──"


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BuiltPrompt:
    system_prompt:   str
    user_prompt:     str
    sql_rows_used:   int           # how many SQL metric rows were included
    chunks_used:     int           # how many vector chunks were included
    insights_used:   int           # how many fusion insights were included
    total_chars:     int           # estimated total prompt size
    was_trimmed:     bool          # True if chunks were dropped for budget


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

# Shared financial rules injected into every system prompt
_FINANCIAL_RULES = """\
FINANCIAL ANALYSIS RULES (follow strictly):
A. EXACT METRIC MATCHING: If the user asks for EBIT but only EBITDA is available,
   flag it: "Note: EBIT not found; showing EBITDA (includes D&A of ₹X cr)."
   Never silently substitute one metric for another.
B. SHOW YOUR MATH: For any growth/YoY/CAGR calculation write the formula and
   numbers inline. Example: Revenue growth FY24→FY25 = (31,079 − 26,711) / 26,711 × 100 = +16.3%
C. CITE EVERY NUMBER: After each figure write [SQL-N] if it came from the
   structured data table, or [SRC-N] if it came from a document excerpt.
D. CURRENCY: State amounts exactly as in source (₹ Crore / Lakh / Million).
   Never convert unless the user asks.
E. NO HALLUCINATION: If a number is not in the provided context, write
   "Not available in provided documents." Never guess or back-calculate.
F. RECENCY FIRST: Lead with the most recent fiscal year available."""

_SYSTEM_PROMPT_FUSION = """\
You are a senior equity research analyst specialising in Indian listed companies \
(BSE/NSE).

You receive three types of pre-processed context:
  [SQL]      Hard numbers directly from a structured financial database.
             These are ground truth — always prefer them over prose.
  [EXCERPTS] Ranked passages from annual reports and concall transcripts.
             Use these for qualitative colour and management commentary.
  [INSIGHTS] Pre-detected contradictions, confirmations, and guidance flags
             surfaced by the cross-referencing layer.
             Highlight these prominently in your answer.

ANSWER FORMAT:
1. If comparing metrics across years → use a Markdown table.
2. If summarising management commentary → use bullet points with speaker names.
3. If a contradiction insight is present → start your answer with a ⚠ callout.
4. Always end with a "Sources" line listing the [SQL-N] / [SRC-N] tags used.

""" + _FINANCIAL_RULES

_SYSTEM_PROMPT_VECTOR_ONLY = """\
You are a senior equity research analyst specialising in Indian listed companies \
(BSE/NSE).

BEFORE WRITING YOUR ANSWER reason step-by-step internally:
  Step 1 — What EXACTLY is being asked? One sentence.
  Step 2 — Which chunks directly answer Step 1? List them.
  Step 3 — Exclude chunks that are only tangentially related.
  Step 4 — Build your answer ONLY from Step 2 chunks.
  Step 5 — For missing data write "Not available in provided documents."

ANSWER FORMAT:
- Multi-year comparisons → Markdown table.
- Single-year or qualitative → bullet points or short paragraphs.
- Cite every number with [SRC-N].

""" + _FINANCIAL_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_value(value: Optional[float], unit: str) -> str:
    """Format a metric value with its unit for display."""
    if value is None:
        return "N/A"
    if unit in ("%",):
        return f"{value:.2f}%"
    if unit in ("x",):
        return f"{value:.2f}x"
    if unit in ("days",):
        return f"{value:.1f} days"
    if unit in ("rs",):
        return f"₹{value:,.2f}"
    if unit in ("crore",):
        return f"₹{value:,.1f} cr"
    return f"{value:,.2f} {unit}".strip()


def _render_sql_table(metric_rows: List[Dict]) -> str:
    """
    Render the list of MetricRow dicts as a compact ASCII table.

    Groups rows by (symbol, sub_type) so multi-year series appears as
    one logical block.  Annotates each row with [SQL-N].

    Returns empty string if metric_rows is empty.
    """
    if not metric_rows:
        return ""

    lines = [_SQL_TABLE_HDR]
    lines.append(f"{'#':<6} {'Symbol':<14} {'Metric':<22} {'FY':<6} {'Period':<12} {'Value'}")
    lines.append("─" * 80)

    for i, row in enumerate(metric_rows, 1):
        sym    = str(row.get("symbol", ""))[:13]
        metric = str(row.get("metric", row.get("sub_type", "")))[:21]
        fy     = str(row.get("year", ""))[:5]
        period = str(row.get("period", ""))[:11]
        value  = _fmt_value(row.get("value"), row.get("unit", ""))
        lines.append(f"[SQL-{i}] {sym:<14} {metric:<22} {fy:<6} {period:<12} {value}")

    lines.append("─" * 80)
    return "\n".join(lines)


def _render_insights(insights: List[Dict]) -> str:
    """
    Render fusion insights (contradictions / confirmations / forward / unmatched)
    as a labelled callout block.

    Returns empty string if no insights.
    """
    if not insights:
        return ""

    contras  = [i for i in insights if i.get("type") == "CONTRADICT"]
    confirms = [i for i in insights if i.get("type") == "CONFIRM"]
    forwards = [i for i in insights if i.get("type") == "FORWARD"]
    unmatch  = [i for i in insights if i.get("type") == "UNMATCHED"]

    lines = [_INSIGHTS_HDR]

    for ins in contras:
        lines.append(f"⚠  CONTRADICTION  [{ins.get('metric','')}]")
        lines.append(f"   {ins.get('note','')}")

    for ins in confirms:
        lines.append(f"✓  CONFIRMED  [{ins.get('metric','')}]")
        lines.append(f"   {ins.get('note','')}")

    for ins in forwards:
        lines.append(f"→  GUIDANCE  [{ins.get('metric','')} | {ins.get('symbol','')}]")
        lines.append(f"   {ins.get('note','')}")

    for ins in unmatch:
        lines.append(f"·  DATA ONLY (no mgmt commentary)  [{ins.get('metric','')}]")
        lines.append(f"   {ins.get('note','')}")

    return "\n".join(lines)


def _render_chunks(chunks: List["RetrievedChunk"], start_index: int = 1) -> str:
    """Render vector chunks as labelled [SRC-N] blocks."""
    if not chunks:
        return ""
    lines = [_VECTOR_HDR]
    for i, chunk in enumerate(chunks, start_index):
        meta    = chunk.metadata
        symbol  = meta.get("symbol", "")
        year    = meta.get("year", "")
        dt      = "AR" if meta.get("doc_type") == "annual_report" else "CC"
        section = (meta.get("section") or meta.get("speaker", ""))[:50]
        page    = meta.get("page_start", "?")
        score   = round(chunk.score, 4)
        header  = f"[SRC-{i}] {symbol} FY{year} [{dt}] | {section} | p.{page} | score={score}"
        lines.append(header)
        lines.append(textwrap.fill(chunk.text[:1200], width=100))
        lines.append("")
    return "\n".join(lines)


def _build_gap_flag_note(
    resolved_years: Optional[List[int]],
    explicit_years: Optional[List[int]],
) -> str:
    if not explicit_years:
        return (
            "\nGAP FLAG INSTRUCTION: The user did NOT specify particular years. "
            "Do NOT emit any ⚠ 'not in retrieved excerpts' flags. "
            "Simply state what is available."
        )
    fy_str = "/".join(f"FY{y}" for y in explicit_years)
    return (
        f"\nGAP FLAG INSTRUCTION: The user explicitly asked about {fy_str}. "
        f"Emit ⚠ gap warnings ONLY for these years if data is missing. "
        f"Do NOT emit ⚠ for any other year."
    )


def _build_intent_note(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["outlook", "guidance", "expect", "h1", "h2",
                                "demand", "going forward", "forecast", "target"]):
        return (
            "\nINTENT: FORWARD-LOOKING query. Prioritise [SRC-N] chunks containing "
            "'we expect', 'going forward', 'H1/H2', 'guidance', 'target'. "
            "Do NOT substitute past performance for forward commentary."
        )
    if any(kw in q for kw in ["yoy", "year on year", "growth", "cagr", "trend"]):
        return (
            "\nINTENT: GROWTH / TREND query. Show a multi-year table with explicit "
            "YoY % calculations. Lead with most recent year."
        )
    return ""


def _build_metric_note(query: str) -> str:
    q = query.lower()
    if "ebit" in q and "ebitda" not in q:
        return (
            "\nMETRIC NOTE: User asked for EBIT. If only EBITDA is available, "
            "state: 'EBIT not found; EBITDA shown. EBIT = EBITDA − D&A (₹X cr if available).'"
        )
    if "ebitda" in q and "ebit" not in q:
        return (
            "\nMETRIC NOTE: User asked for EBITDA. Do not report EBIT as EBITDA "
            "without disclosing the difference."
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles (system_prompt, user_prompt) from FusionResult + raw chunks.

    Usage:
        builder = PromptBuilder()
        built   = builder.build(query, fusion_result, chunks, ...)
        result  = _call_with_retry(built.system_prompt, built.user_prompt, entry)
    """

    def __init__(self, max_context_chars: int = _DEFAULT_MAX_CHARS):
        self.max_context_chars = max_context_chars

    # ── Main entry point ──────────────────────────────────────────────────────

    def build(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        fusion_result:  Optional[Any] = None,   # FusionResult | None
        doc_type:       str = "both",
        resolved_years: Optional[List[int]] = None,
        explicit_years: Optional[List[int]] = None,
    ) -> BuiltPrompt:
        """
        Build the full (system, user) prompt pair.

        Parameters
        ──────────
        query           Raw user question.
        chunks          Reranked RetrievedChunk list (vector channel output).
        fusion_result   FusionResult from the fusion layer — may be None for
                        pure-vector queries (legacy path).
        doc_type        "annual_report" | "concall" | "both"
        resolved_years  All years used for retrieval filtering.
        explicit_years  Only years the user explicitly named — controls gap flags.
        """
        has_fusion = (
            fusion_result is not None
            and FusionResult is not None
            and isinstance(fusion_result, FusionResult)
        )

        if has_fusion:
            return self._build_fusion_prompt(
                query, chunks, fusion_result,
                doc_type, resolved_years, explicit_years,
            )
        else:
            return self._build_vector_only_prompt(
                query, chunks, doc_type, resolved_years, explicit_years,
            )

    # ── Fusion path (SQL + vector + insights) ─────────────────────────────────

    def _build_fusion_prompt(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        fusion_result:  Any,
        doc_type:       str,
        resolved_years: Optional[List[int]],
        explicit_years: Optional[List[int]],
    ) -> BuiltPrompt:

        ctx = fusion_result.to_context_dict()

        # ── 1. SQL table (always kept whole) ──────────────────────────────────
        sql_block     = _render_sql_table(ctx.get("metric_table", []))
        sql_rows_used = len(ctx.get("metric_table", []))

        # ── 2. Insights block ─────────────────────────────────────────────────
        all_insights = (
            ctx.get("contradictions",  []) +
            ctx.get("confirmations",   []) +
            ctx.get("forward_guidance",[]) +
            ctx.get("unmatched",       [])
        )
        insights_block  = _render_insights(all_insights)
        insights_used   = len(all_insights)

        # ── 3. Budget: how much space left for vector chunks? ─────────────────
        system_prompt  = _SYSTEM_PROMPT_FUSION
        notes          = (
            _build_gap_flag_note(resolved_years, explicit_years)
            + _build_intent_note(query)
            + _build_metric_note(query)
        )
        fixed_chars    = (
            len(system_prompt)
            + len(sql_block)
            + len(insights_block)
            + len(query)
            + len(notes)
            + 600          # overhead: headers, separators, question line
        )
        chunk_budget   = max(0, self.max_context_chars - fixed_chars)

        # ── 4. Trim chunks to budget ──────────────────────────────────────────
        safe_chunks, was_trimmed = _trim_chunks(chunks, chunk_budget)

        # ── 5. Chunk block ────────────────────────────────────────────────────
        chunk_block = _render_chunks(safe_chunks, start_index=1)

        # ── 6. Assemble user prompt ───────────────────────────────────────────
        year_note = ""
        if resolved_years:
            year_note = (
                f"\nDATA SEARCHED: Retrieval covered "
                f"FY{'/'.join(str(y) for y in resolved_years)} documents."
            )

        sections = []
        if sql_block:
            sections.append(sql_block)
        if insights_block:
            sections.append(insights_block)
        if chunk_block:
            sections.append(chunk_block)
        if ctx.get("errors"):
            sections.append("── PIPELINE ERRORS ──\n" + "\n".join(ctx["errors"]))

        context_block = "\n\n".join(sections)

        user_prompt = (
            f"CONTEXT:\n{'=' * 70}\n"
            f"{context_block}\n"
            f"{'=' * 70}"
            f"{year_note}{notes}\n\n"
            f"QUESTION: {query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Use [SQL-N] data as ground truth; cite it after every number.\n"
            f"- Use [SRC-N] excerpts for qualitative context and management commentary.\n"
            f"- If ⚠ CONTRADICTION insights are present, lead with them.\n"
            f"- Show YoY % calculations explicitly (formula + numbers).\n"
            f"- Use a Markdown table for multi-year comparisons.\n"
            f"- End with a concise 'Sources used: [SQL-1], [SRC-2], ...' line.\n"
        )

        total_chars = len(system_prompt) + len(user_prompt)
        log.info(
            f"[prompt_builder] fusion path | sql_rows={sql_rows_used} "
            f"chunks={len(safe_chunks)}/{len(chunks)} insights={insights_used} "
            f"chars={total_chars:,} trimmed={was_trimmed}"
        )

        return BuiltPrompt(
            system_prompt  = system_prompt,
            user_prompt    = user_prompt,
            sql_rows_used  = sql_rows_used,
            chunks_used    = len(safe_chunks),
            insights_used  = insights_used,
            total_chars    = total_chars,
            was_trimmed    = was_trimmed,
        )

    # ── Vector-only path (legacy / no SQL data) ───────────────────────────────

    def _build_vector_only_prompt(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        doc_type:       str,
        resolved_years: Optional[List[int]],
        explicit_years: Optional[List[int]],
    ) -> BuiltPrompt:

        system_prompt = _SYSTEM_PROMPT_VECTOR_ONLY
        notes         = (
            _build_gap_flag_note(resolved_years, explicit_years)
            + _build_intent_note(query)
            + _build_metric_note(query)
        )

        fixed_chars  = len(system_prompt) + len(query) + len(notes) + 400
        chunk_budget = max(0, self.max_context_chars - fixed_chars)

        safe_chunks, was_trimmed = _trim_chunks(chunks, chunk_budget)
        chunk_block  = _render_chunks(safe_chunks, start_index=1)

        year_note = ""
        if resolved_years:
            year_note = (
                f"\nDATA SEARCHED: Retrieval covered "
                f"FY{'/'.join(str(y) for y in resolved_years)} documents."
            )

        user_prompt = (
            f"CONTEXT FROM FINANCIAL DOCUMENTS (most relevant first):\n"
            f"{'=' * 70}\n"
            f"{chunk_block}\n"
            f"{'=' * 70}"
            f"{year_note}{notes}\n\n"
            f"QUESTION: {query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer using ONLY the context above.\n"
            f"- Cite every number with [SRC-N].\n"
            f"- Show explicit calculations for any growth/trend figures.\n"
            f"- Use a Markdown table for multi-year comparisons.\n"
            f"- Flag only explicitly-requested missing years (see GAP FLAG above).\n"
        )

        total_chars = len(system_prompt) + len(user_prompt)
        log.info(
            f"[prompt_builder] vector-only path | "
            f"chunks={len(safe_chunks)}/{len(chunks)} "
            f"chars={total_chars:,} trimmed={was_trimmed}"
        )

        return BuiltPrompt(
            system_prompt  = system_prompt,
            user_prompt    = user_prompt,
            sql_rows_used  = 0,
            chunks_used    = len(safe_chunks),
            insights_used  = 0,
            total_chars    = total_chars,
            was_trimmed    = was_trimmed,
        )

    # ── Convenience: adjust budget for a specific provider ────────────────────

    def for_provider(self, model: str) -> "PromptBuilder":
        """
        Return a new PromptBuilder sized for a specific model.

        Usage:
            built = PromptBuilder().for_provider("llama-3.3-70b-versatile").build(...)
        """
        _MODEL_CHARS = {
            # Groq free tier
            "llama-3.3-70b-versatile":          _GROQ_MAX_CHARS,
            "gemma2-9b-it":                     12_000,
            # OpenRouter Qwen free
            "qwen/qwen3-30b-a3b:free":          _DEFAULT_MAX_CHARS,
            "qwen/qwen3-8b:free":               _DEFAULT_MAX_CHARS,
            "qwen/qwen2.5-72b-instruct:free":   _DEFAULT_MAX_CHARS,
            # Gemini
            "google/gemini-2.0-flash-001":      600_000,
            "gemini-2.0-flash":                 600_000,
            # NVIDIA NIM
            "meta/llama-3.3-70b-instruct":      150_000,
        }
        chars = _MODEL_CHARS.get(model, _DEFAULT_MAX_CHARS)
        return PromptBuilder(max_context_chars=chars)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: chunk trimmer
# ─────────────────────────────────────────────────────────────────────────────

def _trim_chunks(
    chunks: List["RetrievedChunk"],
    char_budget: int,
) -> tuple:  # (List[RetrievedChunk], bool was_trimmed)
    """
    Keep top-ranked chunks until char_budget is consumed.
    Returns (kept_chunks, was_trimmed).
    SQL data is never touched here — budget calculation excludes it.
    """
    if char_budget <= 0:
        return [], True

    kept  = []
    used  = 0
    # ~250 chars per chunk for header + separators
    for chunk in chunks:
        cost = len(chunk.text) + 250
        if used + cost > char_budget:
            break
        kept.append(chunk)
        used += cost

    was_trimmed = len(kept) < len(chunks)
    if was_trimmed:
        log.info(
            f"[prompt_builder] trimmed {len(chunks)} → {len(kept)} chunks "
            f"({used:,}/{char_budget:,} chars used)"
        )
    return kept or (chunks[:1] if chunks else []), was_trimmed