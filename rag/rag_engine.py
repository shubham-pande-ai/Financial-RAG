"""
rag/rag_engine.py

FIXES applied in this version:

  [FIX CONTEXT-TRIM] Context trimmer now adapts top_k to provider capacity.
          Groq free tier (~5.5k tok) → up to 6 chunks.
          Qwen/Gemini/NVIDIA (≥30k tok) → all 18 chunks, no trimming.
          The reranker's work is now preserved for large-context providers.
          Recommended default provider: or-qwen30b (free, 131k ctx).

  [FIX EXPLICIT-YEARS] System prompt now receives explicit_years separately
          from resolved_years. ⚠ gap flags are emitted ONLY for years the
          user explicitly named. Queries without a year hint no longer
          generate spurious ⚠ FY2017 – FY2020 warnings.

  [FIX METRIC-MISMATCH] System prompt now instructs the LLM to flag when
          the requested metric (e.g. EBIT) is absent and a related metric
          (e.g. EBITDA) is substituted. Previously this was silent.

  [FIX PROVIDER-DEFAULT] Added --provider CLI flag support + --auto flag.
          Default provider menu is shown only in interactive sessions.
          Scripted use: pass provider_id="or-qwen30b" or auto=True.

  [FIX FINANCIAL-STATEMENTS] System prompts now explicitly instruct the
          LLM to look for: balance sheet, cash flow statement, EPS note,
          ratios note, segment reporting (Ind AS 108). These were being
          missed when the retrieved chunks contained the right pages but
          the LLM was not looking for them by their exact heading names.

  [FIX SOURCES-BUG] sources list now correctly reports safe_chunks (chunks
          actually sent to LLM after trimming), not all reranked chunks.
          Previously sources showed chunks the LLM never saw.

  [FIX I] Root cause of repeated 413 on Groq llama-3.3-70b:
          Groq's FREE tier hard-caps input to ~6k tokens. Budget set to
          5500 tokens. _CHARS_PER_TOKEN tightened 4.0 → 3.5.

  [NEW-QWEN] Qwen models via OpenRouter — FREE, 131k context window.
  [NEW-OLLAMA] Local LLM via Ollama — zero API calls, zero rate limits.
  [NEW-PICKER] Interactive numbered menu + --provider / --auto flags.
"""

import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass

import requests

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
)
from pipeline.retrieval.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_API_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
GEMINI_API_URL     = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ─────────────────────────────────────────────
# Context window budgets (tokens)
# ─────────────────────────────────────────────
_MODEL_CTX: Dict[str, int] = {
    # Groq — free tier hard-capped at ~6k input regardless of model window
    "llama-3.3-70b-versatile":           5_500,
    "gemma2-9b-it":                      3_200,
    "llama3-8b-8192":                    6_000,
    # Google Gemini (direct)
    "gemini-2.0-flash":                200_000,
    "gemini-1.5-flash":                200_000,
    "gemini-1.5-pro":                  200_000,
    # OpenRouter — Qwen (free, 131k context)
    # [FIX CONTEXT-TRIM] These providers can receive ALL chunks — no trim needed.
    # Budget set to 30k (conservative) to avoid OpenRouter free-tier limits.
    "qwen/qwen3-30b-a3b:free":          30_000,
    "qwen/qwen3-8b:free":               30_000,
    "qwen/qwen2.5-72b-instruct:free":   30_000,
    # OpenRouter — Gemini
    "google/gemini-2.0-flash-001":     200_000,
    "anthropic/claude-3-haiku":         50_000,
    # NVIDIA NIM
    "meta/llama-3.3-70b-instruct":      50_000,
    # Ollama default (conservative — actual limit is RAM-dependent)
    "_ollama_default":                 100_000,
}
_DEFAULT_CTX             = 12_000
_CHARS_PER_TOKEN: float  = 3.5        # dense financial text
_GROQ_MAX_PAYLOAD_CHARS  = 130_000
_NO_SYSTEM_ROLE          = {"gemma2-9b-it", "gemma-7b-it"}

# [FIX CONTEXT-TRIM] Providers with ≥30k token context get all chunks.
# Groq free tier (~5.5k) is the only provider that needs aggressive trimming.
_LARGE_CONTEXT_THRESHOLD = 20_000


# ─────────────────────────────────────────────
# Provider entry dataclass
# ─────────────────────────────────────────────
@dataclass
class ProviderEntry:
    id:           str
    label:        str
    provider:     str
    model:        str
    api_key:      str
    api_url:      str
    context_note: str = ""


# ─────────────────────────────────────────────
# Ollama autodiscovery
# ─────────────────────────────────────────────
def _discover_ollama(base_url: str) -> List[ProviderEntry]:
    """Query /api/tags; return one entry per installed model. Never raises."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        entries = []
        for m in resp.json().get("models", []):
            name     = m.get("name", "")
            size_gb  = round(m.get("size", 0) / 1e9, 1)
            safe_id  = name.replace(":", "-").replace("/", "-")
            entries.append(ProviderEntry(
                id           = f"ollama-{safe_id}",
                label        = f"Ollama local — {name}",
                provider     = "ollama",
                model        = name,
                api_key      = "",
                api_url      = f"{base_url}/v1/chat/completions",
                context_note = f"{size_gb} GB, local",
            ))
        return entries
    except Exception:
        return []


# ─────────────────────────────────────────────
# Build provider catalogue
# ─────────────────────────────────────────────
def build_provider_catalogue() -> List[ProviderEntry]:
    cat: List[ProviderEntry] = []

    groq_key   = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    or_key     = os.getenv("OPENROUTER_API_KEY", "")
    nv_key     = os.getenv("NVIDIA_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    if groq_key:
        cat.append(ProviderEntry(
            id="groq-llama", label="Groq — llama-3.3-70b-versatile",
            provider="groq", model=GROQ_MODEL,
            api_key=groq_key, api_url=GROQ_API_URL,
            context_note="~5.5k tok (free cap)",
        ))

    if or_key:
        cat.extend([
            ProviderEntry(
                id="or-qwen30b", label="OpenRouter — Qwen3 30B MoE [FREE]",
                provider="openrouter", model="qwen/qwen3-30b-a3b:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),
            ProviderEntry(
                id="or-qwen72b", label="OpenRouter — Qwen2.5 72B Instruct [FREE]",
                provider="openrouter", model="qwen/qwen2.5-72b-instruct:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),
            ProviderEntry(
                id="or-qwen8b", label="OpenRouter — Qwen3 8B [FREE]",
                provider="openrouter", model="qwen/qwen3-8b:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),
            ProviderEntry(
                id="or-gemini", label="OpenRouter — Gemini 2.0 Flash",
                provider="openrouter", model="google/gemini-2.0-flash-001",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="1M ctx",
            ),
        ])

    if gemini_key:
        cat.append(ProviderEntry(
            id="gemini", label="Google Gemini — gemini-2.0-flash (direct API)",
            provider="gemini", model="gemini-2.0-flash",
            api_key=gemini_key, api_url=GEMINI_API_URL,
            context_note="1M ctx, 15 RPM free",
        ))

    if nv_key:
        cat.append(ProviderEntry(
            id="nvidia", label="NVIDIA NIM — llama-3.3-70b-instruct",
            provider="nvidia", model="meta/llama-3.3-70b-instruct",
            api_key=nv_key, api_url=NVIDIA_API_URL,
            context_note="128k ctx",
        ))

    cat.extend(_discover_ollama(ollama_url))

    if groq_key:
        cat.append(ProviderEntry(
            id="groq-gemma", label="Groq — gemma2-9b-it [last resort]",
            provider="groq", model=GROQ_FALLBACK_MODEL,
            api_key=groq_key, api_url=GROQ_API_URL,
            context_note="3.2k tok, tiny ctx",
        ))

    return cat


# ─────────────────────────────────────────────
# Interactive provider picker
# ─────────────────────────────────────────────
def pick_provider_interactive(catalogue: List[ProviderEntry]) -> ProviderEntry:
    W_LABEL = 46
    W_NOTE  = 18
    border  = "─" * (W_LABEL + W_NOTE + 10)

    print(f"\n┌{border}┐")
    print(f"│  🤖  Select LLM Provider{' ' * (len(border) - 24)}│")
    print(f"├────┬{'─' * W_LABEL}┬{'─' * W_NOTE}┤")
    print(f"│ #  │ {'Provider / Model'.ljust(W_LABEL - 1)}│ {'Context / Notes'.ljust(W_NOTE - 1)}│")
    print(f"├────┼{'─' * W_LABEL}┼{'─' * W_NOTE}┤")
    for i, e in enumerate(catalogue, 1):
        label = e.label[:W_LABEL - 1].ljust(W_LABEL - 1)
        note  = e.context_note[:W_NOTE - 1].ljust(W_NOTE - 1)
        print(f"│ {str(i).ljust(2)} │ {label}│ {note}│")
    print(f"└────┴{'─' * W_LABEL}┴{'─' * W_NOTE}┘")
    print()
    print("  💡 Tip: use --provider or-qwen30b for best results (free, 131k ctx, no trimming)")
    print("         use --auto to skip this menu\n")

    while True:
        raw = input(f"  Enter number [1-{len(catalogue)}] (or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= len(catalogue):
            chosen = catalogue[int(raw) - 1]
            print(f"  ✔  Using: {chosen.label}\n")
            return chosen
        print(f"  ⚠  Enter a number between 1 and {len(catalogue)}")


def get_provider(
    catalogue:   List[ProviderEntry],
    provider_id: Optional[str],
    auto:        bool,
) -> List[ProviderEntry]:
    if not catalogue:
        raise ValueError(
            "No LLM providers available.\n\n"
            "Set at least one in your .env:\n"
            "  GROQ_API_KEY=gsk_...          https://console.groq.com/\n"
            "  OPENROUTER_API_KEY=sk-or-...  https://openrouter.ai/\n"
            "    (Qwen3-30B is FREE on OpenRouter, 131k context — best option)\n"
            "  GEMINI_API_KEY=AIza...        https://aistudio.google.com/app/apikey\n"
            "  NVIDIA_API_KEY=nvapi-...      https://build.nvidia.com/\n"
            "\nOr use Ollama locally (no key needed):\n"
            "  ollama serve\n"
            "  ollama pull qwen2.5:7b\n"
        )

    if provider_id:
        matches = [e for e in catalogue if e.id == provider_id]
        if not matches:
            valid = ", ".join(e.id for e in catalogue)
            raise ValueError(
                f"Provider '{provider_id}' not found.\n"
                f"Available IDs: {valid}"
            )
        return matches

    if auto:
        return catalogue

    chosen = pick_provider_interactive(catalogue)
    return [chosen]


# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────

# [FIX FINANCIAL-STATEMENTS] Instructions added for:
#   - Exact metric matching (EBIT ≠ EBITDA)
#   - Balance sheet / cash flow / ratio page identification
#   - Segment reporting (Ind AS 108 geographic note)
#   - Suppressing gap flags when years weren't explicitly requested

_FINANCIAL_STATEMENT_RULES = """\
FINANCIAL STATEMENT RETRIEVAL RULES (apply to every query):
A. EXACT METRIC MATCHING: If the user asks for EBIT but only EBITDA is in the context,
   flag this explicitly: "Note: EBIT not found; showing EBITDA (includes D&A of ₹X cr).
   To get EBIT: EBITDA − D&A." Never silently substitute one metric for another.
B. BALANCE SHEET: Look for pages titled "Balance Sheet", "Statement of Assets and
   Liabilities", or "Consolidated Balance Sheet". Key line items: total assets,
   shareholders equity / net worth, total borrowings, current assets, current liabilities.
C. CASH FLOW STATEMENT: Look for "Statement of Cash Flows" or "Cash Flow Statement".
   Key items: net cash from operating activities (OCF), capital expenditure (under
   investing activities), free cash flow = OCF − capex, net change in borrowings.
D. EPS / RATIOS NOTE: In Indian annual reports, EPS (basic and diluted) is disclosed
   in "Notes to Financial Statements" under "Earnings Per Share" (Ind AS 33).
   Look for "weighted average number of equity shares" and "face value ₹X per share".
E. GEOGRAPHIC SEGMENT (Ind AS 108): Revenue split India vs Outside India is in
   "Segment Information" or "Notes — Segment Reporting". Look for a table with
   columns "India" and "Outside India" or "Rest of World".
F. SEGMENT EBITDA / EBIT: Per Ind AS 108, segment profit is reported as "segment
   result" which may be EBIT or EBITDA — state which one clearly.
G. RATIOS: ROCE = EBIT / Capital Employed. ROE = PAT / Avg Shareholders Equity.
   Show the numerator and denominator values from context before computing the ratio.\
"""

ANNUAL_SYSTEM_PROMPT = """\
You are a senior equity research analyst at a top-tier investment firm, \
specialising in Indian listed companies (BSE/NSE).

BEFORE WRITING YOUR ANSWER, reason step-by-step internally:
  Step 1 — Re-read the question. What EXACTLY is being asked? Write it in one sentence.
  Step 2 — Scan the context. List ONLY the chunks that directly answer Step 1.
  Step 3 — If a chunk is only tangentially related (e.g. general company intro, \
boilerplate, unrelated financials), EXCLUDE it from your answer entirely.
  Step 4 — Build your answer ONLY from the chunks identified in Step 2.
  Step 5 — Do NOT generate tables, estimates, or calculations for data that is NOT \
present in Step 2 chunks. If data is missing, flag it with ⚠ and stop.

YOUR RULES — follow every one strictly:
1. RECENCY FIRST: Always prefer and prominently feature data from the most recent \
fiscal year available in the context. Never lead with old data.
2. USE ONLY CONTEXT: Do not use prior knowledge. If a number is not in the \
provided excerpts, say explicitly: "Not available in provided documents."
3. SHOW YOUR MATH: For any growth/trend calculation, write the formula and numbers. \
Example: Revenue growth FY24→FY25 = (31,079 − 26,711) / 26,711 × 100 = +16.3%
4. CURRENCY: State amounts exactly as shown in source (Crore / Lakh / Million). \
Never convert unless asked.
5. STRUCTURED OUTPUT: For multi-year comparisons, use a table. For single-year \
analysis, use bullet points. For qualitative topics, use paragraphs.
6. CITE EVERY NUMBER: After each data point write [FY<year>, AR, Page <n>].
7. FLAG GAPS — ONLY FOR EXPLICITLY REQUESTED YEARS: If the user named specific years \
(e.g. "FY23, FY24, FY25") and data for one of those years is missing, write: \
"⚠ FY<year>: data not in retrieved excerpts." \
DO NOT emit gap flags for years the user never mentioned.
8. NO HALLUCINATION: If you are not certain, say so. Never guess a number. \
Never back-calculate or extrapolate missing years from a single growth rate.

""" + _FINANCIAL_STATEMENT_RULES

CONCALL_SYSTEM_PROMPT = """\
You are a senior buy-side equity analyst reviewing earnings call transcripts \
for an Indian listed company.

BEFORE WRITING YOUR ANSWER, reason step-by-step internally:
  Step 1 — Re-read the question. Is it asking for (a) forward-looking guidance/outlook, \
(b) past performance, or (c) specific management commentary? Identify the type.
  Step 2 — Scan the context. Find ONLY chunks where a named management speaker \
(CEO, CFO, MD) directly addresses the question topic. Ignore moderator lines, \
analyst questions, and generic introductions.
  Step 3 — If the context does NOT contain a direct answer, say so clearly. \
Do NOT substitute operational results for guidance.
  Step 4 — Build your answer only from Step 2 chunks.

YOUR RULES:
1. RECENCY FIRST: Lead with the most recent concall available. State the date/quarter.
2. FORWARD-LOOKING PRIORITY: The user often asks about OUTLOOK, GUIDANCE, DEMAND \
ENVIRONMENT, or MANAGEMENT EXPECTATIONS. Search for phrases like \
"we expect", "going forward", "H1/H2 guidance", "demand scenario", "we are confident", \
"target", "capex plan". If found, lead with those. If not found, say so explicitly.
3. QUOTE ACCURATELY: When citing management, use exact words and name the speaker \
and their role. Format: "CFO [Name]: '...'"
4. SEPARATE MGMT vs ANALYST: Clearly distinguish management commentary from \
analyst questions and pushback.
5. KEY THEMES: Extract: guidance, risks mentioned, capex plans, margin commentary, \
volume/revenue targets.
6. FLAG GAPS — ONLY FOR EXPLICITLY REQUESTED YEARS: Only emit ⚠ for years the user \
specifically asked about. Do not flag years that were never part of the question.
7. USE ONLY CONTEXT: Do not use prior knowledge about the company.\
"""

COMBINED_SYSTEM_PROMPT = """\
You are a senior equity research analyst with access to both annual reports \
and earnings call transcripts for an Indian listed company.

BEFORE WRITING YOUR ANSWER, reason step-by-step internally:
  Step 1 — Re-read the question carefully. What specific metric, event, or \
commentary is being asked for?
  Step 2 — Scan ALL provided sources. Tag each as RELEVANT or IRRELEVANT \
to Step 1. Irrelevant chunks (e.g. boilerplate, unrelated sections, moderator \
lines) must be ignored entirely.
  Step 3 — From RELEVANT chunks only: extract facts, numbers, and quotes.
  Step 4 — If a requested data point is absent from RELEVANT chunks, flag it \
with ⚠ ONLY for years the user explicitly named. Do NOT back-calculate, \
interpolate, or extrapolate missing data.
  Step 5 — Write the answer using only what Step 3 produced.

YOUR RULES:
1. RECENCY FIRST: Always lead with the most recent data available. State the FY prominently.
2. CROSS-SOURCE SYNTHESIS: When annual report numbers are confirmed or expanded \
by concall commentary, show both. Label each: [Annual Report] or [Concall].
3. SHOW MATH: For any YOY growth, show: (new − old) / old × 100.
4. TABLES FOR TRENDS: Multi-year comparisons MUST be in a table with columns: \
FY | Metric | Value | YOY%
5. CITE SOURCES: After each data point: [FY<year>, AR/CC, Page <n>].
6. FLAG MISSING DATA — ONLY FOR EXPLICITLY REQUESTED YEARS: Write \
"⚠ FY<year>: not in retrieved excerpts." ONLY for years the user named in the query. \
If the user asked a general question without naming specific years, omit all ⚠ gap flags.
7. NO HALLUCINATION: If a number is not in context, do not estimate or extrapolate.
8. HONEST ABOUT LIMITS: If the context is insufficient, say what IS available \
and what is MISSING — but do not invent placeholders for years never asked about.

""" + _FINANCIAL_STATEMENT_RULES


# ─────────────────────────────────────────────
# Token / size helpers
# ─────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ─────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────
def _build_context(chunks: List[RetrievedChunk]) -> str:
    def sort_key(c):
        yr  = c.metadata.get("year", 0)
        typ = 0 if c.metadata.get("doc_type") == "annual_report" else 1
        return (-yr, typ)

    parts     = []
    separator = "\n\n" + "─" * 60 + "\n\n"
    for i, chunk in enumerate(sorted(chunks, key=sort_key), 1):
        meta      = chunk.metadata
        doc_label = "Annual Report" if meta.get("doc_type") == "annual_report" else "Concall Transcript"
        section   = (meta.get("section") or meta.get("speaker") or "")[:60]
        tag = (
            f"[Source {i} | {meta.get('symbol','')} | {doc_label} | "
            f"FY{meta.get('year','')} | {section} | Page {meta.get('page_start','')}]"
        )
        parts.append(f"{tag}\n{chunk.text.strip()}")
    return separator.join(parts)


def _build_user_prompt(
    query: str,
    context: str,
    doc_type: str,
    resolved_years: Optional[List[int]] = None,
    explicit_years: Optional[List[int]] = None,
) -> str:
    """
    Build the user-turn prompt.

    resolved_years — used for the year-filter note so the LLM knows what
                     data was searched.
    explicit_years — ONLY years the user named; drives the ⚠ gap flag
                     instruction. Empty list → "do not flag missing years".
    """
    year_note = ""
    if resolved_years:
        year_note = (
            f"\n\nDATA SEARCHED: The retrieval system searched FY"
            f"{'/'.join(str(y) for y in resolved_years)} documents."
        )

    # [FIX EXPLICIT-YEARS] Tight gap-flag instruction
    gap_flag_note = ""
    if explicit_years:
        gap_flag_note = (
            f"\n\nGAP FLAG INSTRUCTION: The user explicitly asked about "
            f"FY{'/'.join(str(y) for y in explicit_years)}. "
            f"Emit ⚠ gap warnings ONLY for these years if data is missing. "
            f"Do NOT emit ⚠ for any other year."
        )
    else:
        gap_flag_note = (
            "\n\nGAP FLAG INSTRUCTION: The user did NOT specify particular years. "
            "Do NOT emit any ⚠ 'not in retrieved excerpts' flags. "
            "Simply state what is available and summarise it."
        )

    # Forward-looking intent note
    q_lower = query.lower()
    intent_note = ""
    if any(kw in q_lower for kw in ["outlook", "guidance", "expect", "h1", "h2",
                                      "demand environment", "going forward", "forecast"]):
        intent_note = (
            "\n\nINTENT NOTE: This is a FORWARD-LOOKING query. Prioritise chunks that "
            "contain phrases like 'we expect', 'going forward', 'H1/H2', 'demand environment', "
            "'guidance', 'we are confident', 'target'. Do NOT substitute past performance "
            "data for forward-looking commentary."
        )

    # [FIX METRIC-MISMATCH] Detect EBIT/EBITDA confusion in query
    metric_note = ""
    if "ebit" in q_lower and "ebitda" not in q_lower:
        metric_note = (
            "\n\nMETRIC NOTE: The user asked for EBIT (Earnings Before Interest and Tax). "
            "EBIT = Revenue − COGS − Operating Expenses (excludes D&A from EBITDA). "
            "If only EBITDA is available, state: 'EBIT not available; EBITDA shown instead. "
            "EBIT = EBITDA − Depreciation & Amortisation (D&A = ₹X cr if available).' "
            "Never silently report EBITDA when EBIT was requested."
        )
    elif "ebitda" in q_lower and "ebit" not in q_lower:
        metric_note = (
            "\n\nMETRIC NOTE: The user asked for EBITDA (Earnings Before Interest, Tax, "
            "Depreciation, and Amortisation). EBITDA = EBIT + D&A. "
            "Do not report EBIT as EBITDA without disclosing the difference."
        )

    return (
        f"CONTEXT FROM FINANCIAL DOCUMENTS (most recent first):\n"
        f"{'=' * 60}\n{context}\n{'=' * 60}"
        f"{year_note}{gap_flag_note}{intent_note}{metric_note}\n\n"
        f"QUESTION: {query}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer using ONLY the context above.\n"
        f"- Lead with the most recent year's data.\n"
        f"- Show calculations explicitly for any growth/trend figures.\n"
        f"- Use a table if comparing across multiple years.\n"
        f"- Flag only explicitly-requested missing years (see GAP FLAG INSTRUCTION).\n"
    )


# ─────────────────────────────────────────────
# Context trimmer  [FIX CONTEXT-TRIM]
# ─────────────────────────────────────────────
def _trim_chunks_to_budget(
    chunks:         List[RetrievedChunk],
    system_prompt:  str,
    query:          str,
    doc_type:       str,
    resolved_years: Optional[List[int]],
    explicit_years: Optional[List[int]],
    entry:          ProviderEntry,
) -> List[RetrievedChunk]:
    """
    Drop lowest-ranked chunks until the full prompt fits within:
      (a) the model's token budget, AND
      (b) Groq's hard payload char limit

    [FIX CONTEXT-TRIM] For large-context providers (≥20k token budget),
    this function returns ALL chunks without trimming. Only Groq's free
    tier (5.5k tokens) triggers aggressive trimming.
    """
    model     = entry.model
    provider  = entry.provider
    is_merged = model in _NO_SYSTEM_ROLE

    ctx_budget = _MODEL_CTX.get(model) or (
        _MODEL_CTX["_ollama_default"] if provider == "ollama" else _DEFAULT_CTX
    )

    # [FIX CONTEXT-TRIM] Large-context providers: skip trimming entirely.
    # This ensures the reranker's carefully ordered top-18 all reach the LLM.
    if ctx_budget >= _LARGE_CONTEXT_THRESHOLD:
        log.info(
            f"  Large-context provider ({model}: {ctx_budget:,} tok budget) — "
            f"sending all {len(chunks)} chunks without trimming"
        )
        return chunks

    token_budget     = ctx_budget - GROQ_MAX_TOKENS
    system_tokens    = _estimate_tokens(system_prompt)
    base_prompt      = _build_user_prompt(query, "", doc_type, resolved_years, explicit_years)
    base_tokens      = _estimate_tokens(base_prompt)
    sep_overhead     = 150 if is_merged else 50
    available_tokens = token_budget - system_tokens - base_tokens - sep_overhead

    if available_tokens <= 0:
        log.warning(f"  Budget too tight for {model}, using top 3 chunks only")
        return chunks[:3]

    kept        = []
    used_tokens = 0
    used_chars  = len(system_prompt) + len(base_prompt)

    for chunk in chunks:
        chunk_tokens = _estimate_tokens(chunk.text) + 60
        chunk_chars  = len(chunk.text) + 250

        over_tok   = (used_tokens + chunk_tokens) > available_tokens
        over_chars = (provider == "groq") and \
                     (used_chars + chunk_chars) > _GROQ_MAX_PAYLOAD_CHARS

        if over_tok or over_chars:
            break

        kept.append(chunk)
        used_tokens += chunk_tokens
        used_chars  += chunk_chars

    if len(kept) < len(chunks):
        log.info(
            f"  Trimmed {len(chunks)} → {len(kept)} chunks "
            f"({model}: ~{used_tokens} tok, ~{used_chars} chars)"
        )
    return kept or chunks[:1]


# ─────────────────────────────────────────────
# Gemini native call
# ─────────────────────────────────────────────
def _call_gemini(system_prompt: str, user_prompt: str,
                 model: str, api_key: str) -> dict:
    url  = GEMINI_API_URL.format(model=model)
    resp = requests.post(
        url,
        json={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature":     GROQ_TEMPERATURE,
                "maxOutputTokens": GROQ_MAX_TOKENS,
            },
        },
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response: {data}") from exc
    usage = data.get("usageMetadata", {})
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens":     usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens":      usage.get("totalTokenCount", 0),
        },
    }


# ─────────────────────────────────────────────
# OpenAI-compatible call
# ─────────────────────────────────────────────
def _call_openai_compat(system_prompt: str, user_prompt: str,
                        entry: ProviderEntry) -> dict:
    merge   = entry.model in _NO_SYSTEM_ROLE
    headers = {"Content-Type": "application/json"}
    if entry.api_key:
        headers["Authorization"] = f"Bearer {entry.api_key}"
    if "openrouter" in entry.api_url:
        headers["HTTP-Referer"] = "https://github.com/Import-Saurabh/Financial-Rag"
        headers["X-Title"]      = "Financial RAG"

    messages = (
        [{"role": "user", "content": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
        if merge else
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": user_prompt}]
    )
    resp = requests.post(
        entry.api_url,
        json={
            "model":       entry.model,
            "messages":    messages,
            "max_tokens":  GROQ_MAX_TOKENS,
            "temperature": GROQ_TEMPERATURE,
        },
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Retry wrapper
# ─────────────────────────────────────────────
def _call_with_retry(system_prompt: str, user_prompt: str,
                     entry: ProviderEntry, max_retries: int = 4) -> dict:
    delays   = [5, 15, 30, 60]
    last_exc = None

    for attempt in range(max_retries):
        try:
            if entry.provider == "gemini":
                return _call_gemini(system_prompt, user_prompt,
                                    entry.model, entry.api_key)
            return _call_openai_compat(system_prompt, user_prompt, entry)

        except requests.HTTPError as e:
            status   = e.response.status_code
            last_exc = e
            if status == 429 and attempt < max_retries - 1:
                wait = delays[attempt]
                log.warning(f"  429 on {entry.model} (attempt {attempt+1}/{max_retries}), retry in {wait}s")
                time.sleep(wait)
            else:
                raise

    raise last_exc


# ─────────────────────────────────────────────
# RAGResponse
# ─────────────────────────────────────────────
@dataclass
class RAGResponse:
    answer:      str
    model_used:  str
    chunks_used: int
    sources:     List[dict]
    tokens_used: int
    latency_sec: float


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def generate_answer(
    query:          str,
    chunks:         List[RetrievedChunk],
    doc_type:       str = "annual_report",
    api_key:        Optional[str] = None,          # legacy compat, unused
    resolved_years: Optional[List[int]] = None,    # full year filter used
    explicit_years: Optional[List[int]] = None,    # years user actually named
    years:          Optional[List[int]] = None,    # legacy compat → resolved_years
    provider_id:    Optional[str] = None,
    auto:           bool = False,
) -> RAGResponse:
    """
    resolved_years — years used for ChromaDB retrieval filter.
    explicit_years — ONLY years the user explicitly mentioned in the query.
                     Controls which ⚠ gap flags the LLM emits.
                     Pass [] when user asked a general question.
    years          — legacy alias for resolved_years; kept for back-compat.
    provider_id    — pin to a specific provider (e.g. "or-qwen30b").
    auto           — True = silent waterfall, no interactive menu.
    """
    # Back-compat: if old callers pass years=, treat as resolved_years
    if resolved_years is None and years is not None:
        resolved_years = years
    if explicit_years is None:
        explicit_years = resolved_years or []

    if not chunks:
        return RAGResponse(
            answer=(
                "No relevant documents found for this query.\n"
                "Possible reasons:\n"
                "  • Documents for the requested years not ingested\n"
                "  • Run: python ingest.py --symbol <SYMBOL>\n"
                "  • Try --year-range to broaden the search"
            ),
            model_used="none", chunks_used=0,
            sources=[], tokens_used=0, latency_sec=0.0,
        )

    system    = {"annual_report": ANNUAL_SYSTEM_PROMPT, "concall": CONCALL_SYSTEM_PROMPT}.get(
        doc_type, COMBINED_SYSTEM_PROMPT
    )
    catalogue = build_provider_catalogue()
    entries   = get_provider(catalogue, provider_id, auto)

    t0         = time.time()
    last_error = None

    for entry in entries:
        log.info(f"  → {entry.label}")

        safe_chunks = _trim_chunks_to_budget(
            chunks, system, query, doc_type, resolved_years, explicit_years, entry
        )
        context     = _build_context(safe_chunks)
        user_prompt = _build_user_prompt(
            query, context, doc_type, resolved_years, explicit_years
        )

        try:
            result  = _call_with_retry(system, user_prompt, entry)
            latency = time.time() - t0
            answer  = result["choices"][0]["message"]["content"]
            usage   = result.get("usage", {})

            log.info(
                f"  LLM ✓ [{entry.label}]: "
                f"{usage.get('completion_tokens', 0)} tokens | "
                f"{latency:.1f}s | {len(safe_chunks)}/{len(chunks)} chunks"
            )
            return RAGResponse(
                answer      = answer,
                model_used  = entry.label,
                chunks_used = len(safe_chunks),
                sources     = [
                    {
                        "symbol":   c.metadata.get("symbol"),
                        "year":     c.metadata.get("year"),
                        "doc_type": c.metadata.get("doc_type"),
                        "section":  (c.metadata.get("section") or c.metadata.get("speaker", ""))[:50],
                        "page":     c.metadata.get("page_start"),
                        "score":    round(c.score, 4),
                    }
                    # [FIX SOURCES-BUG] safe_chunks only — not all reranked chunks
                    for c in safe_chunks
                ],
                tokens_used = usage.get("total_tokens", 0),
                latency_sec = round(latency, 2),
            )

        except (requests.HTTPError, Exception) as e:
            status   = getattr(getattr(e, "response", None), "status_code", "ERR")
            is_last  = (entry == entries[-1])
            log.warning(
                f"  {entry.label} → HTTP {status} | "
                f"{'all providers exhausted' if is_last else 'trying next'}"
            )
            last_error = e

    raise RuntimeError(
        f"All selected providers failed. Last error: {last_error}\n\n"
        "Quick fixes:\n"
        "  • Best option: add OPENROUTER_API_KEY and use Qwen3-30B (free, 131k ctx)\n"
        "      https://openrouter.ai/ → pick 'or-qwen30b' from the menu\n"
        "      or pass --provider or-qwen30b on the CLI\n"
        "  • Local Ollama (no key): ollama serve && ollama pull qwen2.5:7b\n"
        "  • Groq free tier: wait ~60s then retry\n"
        "  • Gemini: GEMINI_API_KEY → https://aistudio.google.com/app/apikey\n"
    )