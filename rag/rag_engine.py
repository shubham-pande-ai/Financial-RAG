"""
rag/rag_engine.py

Groq LLM integration with finance-grade prompts.

Key fixes:
  - Structured financial analysis prompts (not generic QA)
  - Explicit rules: prefer latest data, show YOY math, flag missing data
  - Context builder sorts chunks by year DESC so LLM sees latest first
  - Fallback model on rate limit
"""

import os
import time
from typing import List, Optional
from dataclasses import dataclass

import requests

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
)
from pipeline.retrieval.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class RAGResponse:
    answer: str
    model_used: str
    chunks_used: int
    sources: List[dict]
    tokens_used: int
    latency_sec: float


# ─────────────────────────────────────────────
# System prompts — finance-grade, structured
# ─────────────────────────────────────────────

ANNUAL_SYSTEM_PROMPT = """\
You are a senior equity research analyst at a top-tier investment firm, \
specialising in Indian listed companies (BSE/NSE).

YOUR RULES — follow every one strictly:
1. RECENCY FIRST: Always prefer and prominently feature data from the most recent \
fiscal year available in the context. Never lead with old data.
2. USE ONLY CONTEXT: Do not use prior knowledge. If a number is not in the \
provided excerpts, say explicitly: "Not available in provided documents."
3. SHOW YOUR MATH: For any growth/trend calculation, write the formula and numbers. \
Example: Revenue growth FY24 = (19,500 - 16,200) / 16,200 = +20.4%
4. CURRENCY: State amounts exactly as shown in source (Crore / Lakh / Million). \
Never convert unless asked.
5. STRUCTURED OUTPUT: For multi-year comparisons, use a table. For single-year \
analysis, use bullet points. For qualitative topics, use paragraphs.
6. CITE EVERY NUMBER: After each data point write [FY<year>, Page <n>].
7. FLAG GAPS: If data for a specific year is missing from context, write: \
"⚠ FY<year>: data not in retrieved excerpts."
8. NO HALLUCINATION: If you are not certain, say so. Never guess a number.\
"""

CONCALL_SYSTEM_PROMPT = """\
You are a senior buy-side equity analyst reviewing earnings call transcripts \
for an Indian listed company.

YOUR RULES:
1. RECENCY FIRST: Lead with the most recent concall available. State the date/quarter.
2. QUOTE ACCURATELY: When citing management, use exact words and name the speaker \
and their role. Format: "CFO [Name]: '...'" 
3. SEPARATE MGMT vs ANALYST: Clearly distinguish management commentary from \
analyst questions and pushback.
4. KEY THEMES: Extract: guidance, risks mentioned, capex plans, margin commentary, \
volume/revenue targets.
5. FLAG GAPS: If the query topic was not discussed in the transcript, say so clearly.
6. USE ONLY CONTEXT: Do not use prior knowledge about the company.\
"""

COMBINED_SYSTEM_PROMPT = """\
You are a senior equity research analyst with access to both annual reports \
and earnings call transcripts for an Indian listed company.

YOUR RULES:
1. RECENCY FIRST: Always lead with the most recent data available (prefer FY2025 > \
FY2024 > FY2023 > older). State the FY prominently.
2. CROSS-SOURCE SYNTHESIS: When annual report numbers are confirmed or expanded \
by concall commentary, show both. Label each: [Annual Report] or [Concall].
3. SHOW MATH: For any YOY growth, show: (new - old) / old × 100.
4. TABLES FOR TRENDS: Multi-year comparisons MUST be in a table with columns: \
FY | Metric | Value | YOY%
5. CITE SOURCES: After each data point: [FY<year>, AR/CC, Page <n>].
6. FLAG MISSING DATA: "⚠ FY<year>: not in retrieved excerpts." for each gap.
7. NO HALLUCINATION: If a number is not in context, do not estimate or extrapolate.
8. HONEST ABOUT LIMITS: If the context is insufficient to answer fully, \
say what IS available and what is MISSING.\
"""


# ─────────────────────────────────────────────
# Context builder — sorted by year DESC so LLM
# sees latest data first in its context window
# ─────────────────────────────────────────────
def _build_context(chunks: List[RetrievedChunk]) -> str:
    # Sort: most recent year first, annual before concall within same year
    def sort_key(c):
        yr  = c.metadata.get("year", 0)
        typ = 0 if c.metadata.get("doc_type") == "annual_report" else 1
        return (-yr, typ)

    sorted_chunks = sorted(chunks, key=sort_key)

    parts = []
    for i, chunk in enumerate(sorted_chunks, 1):
        meta = chunk.metadata
        doc_label = "Annual Report" if meta.get("doc_type") == "annual_report" else "Concall Transcript"
        section   = (meta.get("section") or meta.get("speaker") or "")[:60]
        tag = (
            f"[Source {i} | {meta.get('symbol','')} | {doc_label} | "
            f"FY{meta.get('year','')} | {section} | Page {meta.get('page_start','')}]"
        )
        parts.append(f"{tag}\n{chunk.text.strip()}")

    return "\n\n{'─'*60}\n\n".join(parts)


def _build_user_prompt(query: str, context: str, doc_type: str,
                        years: Optional[List[int]] = None) -> str:
    year_instruction = ""
    if years:
        year_instruction = (
            f"\n\nIMPORTANT: The user is asking about FY{'/'.join(str(y) for y in years)}. "
            f"Prioritise data from these years. If data for any of these years is missing "
            f"from the context, explicitly flag it with ⚠."
        )

    return f"""\
CONTEXT FROM FINANCIAL DOCUMENTS (most recent first):
{'='*60}
{context}
{'='*60}
{year_instruction}

QUESTION: {query}

INSTRUCTIONS:
- Answer using ONLY the context above.
- Lead with the most recent year's data.
- Show calculations explicitly for any growth/trend figures.
- Use a table if comparing across multiple years.
- Flag any years where data is missing from context.
"""


# ─────────────────────────────────────────────
# Groq API call
# ─────────────────────────────────────────────
def _call_groq(system_prompt: str, user_prompt: str,
               model: str, api_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": GROQ_MAX_TOKENS,
        "temperature": GROQ_TEMPERATURE,
    }
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Main RAG function
# ─────────────────────────────────────────────
def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
    doc_type: str = "annual_report",
    api_key: Optional[str] = None,
    years: Optional[List[int]] = None,
) -> RAGResponse:
    api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set.\n"
            "Add to .env: GROQ_API_KEY=gsk_...\n"
            "Get free key: https://console.groq.com/"
        )

    if not chunks:
        return RAGResponse(
            answer=(
                "No relevant documents found for this query.\n"
                "Possible reasons:\n"
                "  • Documents for the requested years not ingested\n"
                "  • Run: python ingest.py --symbol <SYMBOL>\n"
                "  • Try --year-range to broaden the search"
            ),
            model_used=GROQ_MODEL, chunks_used=0,
            sources=[], tokens_used=0, latency_sec=0.0,
        )

    system = {
        "annual_report": ANNUAL_SYSTEM_PROMPT,
        "concall":       CONCALL_SYSTEM_PROMPT,
    }.get(doc_type, COMBINED_SYSTEM_PROMPT)

    context     = _build_context(chunks)
    user_prompt = _build_user_prompt(query, context, doc_type, years)

    t0 = time.time()
    model_used = GROQ_MODEL
    try:
        result = _call_groq(system, user_prompt, GROQ_MODEL, api_key)
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            log.warning(f"Rate limited on {GROQ_MODEL}, trying {GROQ_FALLBACK_MODEL}")
            time.sleep(2)
            model_used = GROQ_FALLBACK_MODEL
            result = _call_groq(system, user_prompt, GROQ_FALLBACK_MODEL, api_key)
        else:
            raise

    latency = time.time() - t0
    answer  = result["choices"][0]["message"]["content"]
    usage   = result.get("usage", {})

    sources = [
        {
            "symbol":   c.metadata.get("symbol"),
            "year":     c.metadata.get("year"),
            "doc_type": c.metadata.get("doc_type"),
            "section":  (c.metadata.get("section") or c.metadata.get("speaker", ""))[:50],
            "page":     c.metadata.get("page_start"),
            "score":    round(c.score, 4),
        }
        for c in chunks
    ]

    log.info(f"  LLM: {usage.get('completion_tokens',0)} tokens | "
             f"{latency:.1f}s | {model_used}")

    return RAGResponse(
        answer=answer, model_used=model_used, chunks_used=len(chunks),
        sources=sources, tokens_used=usage.get("total_tokens", 0),
        latency_sec=round(latency, 2),
    )