"""
rag/rag_engine.py

Groq LLM integration + prompt builder for equity research.
Groq is FREE (14,400 req/day, no credit card) and very fast
(runs llama-3.3-70b on custom inference chips at ~500 tok/s).

Get your key at: https://console.groq.com/
"""

import os
import time
from typing import List, Optional
from dataclasses import dataclass

import requests

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_TEMPERATURE,
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
    sources: List[dict]   # metadata of chunks used
    tokens_used: int
    latency_sec: float


# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
ANNUAL_SYSTEM_PROMPT = """You are a senior equity research analyst specialising in Indian listed companies. 
You have access to excerpts from annual reports. 
Answer questions accurately using only the provided context.
Format numbers clearly (use Crore/Lakh/Million as given in source).
If the context doesn't contain enough information, say so clearly.
Always cite the section and year when referencing specific data."""

CONCALL_SYSTEM_PROMPT = """You are a senior equity research analyst specialising in Indian listed companies.
You have access to earnings call transcripts.
Summarise management commentary, analyst questions, and key guidance accurately.
Quote speakers directly when relevant. Note the speaker name and their role.
If the context doesn't contain the answer, say so clearly."""

COMBINED_SYSTEM_PROMPT = """You are a senior equity research analyst specialising in Indian listed companies.
You have access to annual report excerpts and earnings call transcripts.
Synthesise information from both sources when relevant.
Clearly indicate whether a point comes from the annual report or a concall.
Format numbers clearly and cite sources."""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    """Format chunks into a readable context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        source_tag = (
            f"[Source {i}: {meta.get('symbol','')} "
            f"{'Annual Report' if meta.get('doc_type') == 'annual_report' else 'Concall'} "
            f"FY{meta.get('year','')} | "
            f"Section: {meta.get('section','') or meta.get('speaker','')} | "
            f"Page: {meta.get('page_start','')}]"
        )
        parts.append(f"{source_tag}\n{chunk.text.strip()}")

    return "\n\n---\n\n".join(parts)


def _build_user_prompt(query: str, context: str, doc_type: str) -> str:
    return f"""Context from financial documents:

{context}

---

Question: {query}

Please provide a detailed, accurate answer based on the context above.
{"Focus on financial metrics, trends, and management commentary." if doc_type == "annual_report" else ""}
{"Capture management guidance, key quotes, and analyst concerns." if doc_type == "concall" else ""}
"""


# ─────────────────────────────────────────────
# Groq API call
# ─────────────────────────────────────────────
def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
) -> RAGResponse:
    """
    Generate an LLM answer from retrieved chunks.
    Uses Groq (free) with fallback to second model on rate limit.
    """
    api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/\n"
            "Then: export GROQ_API_KEY=gsk_..."
        )

    if not chunks:
        return RAGResponse(
            answer="No relevant documents found for this query. Try ingesting more documents or broadening your search.",
            model_used=GROQ_MODEL,
            chunks_used=0,
            sources=[],
            tokens_used=0,
            latency_sec=0.0,
        )

    # Choose system prompt
    if doc_type == "annual_report":
        system = ANNUAL_SYSTEM_PROMPT
    elif doc_type == "concall":
        system = CONCALL_SYSTEM_PROMPT
    else:
        system = COMBINED_SYSTEM_PROMPT

    context = _build_context(chunks)
    user_prompt = _build_user_prompt(query, context, doc_type)

    # Try primary model, fall back to secondary on rate limit
    t0 = time.time()
    model_used = GROQ_MODEL
    try:
        result = _call_groq(system, user_prompt, GROQ_MODEL, api_key)
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            log.warning(f"Rate limited on {GROQ_MODEL}, trying fallback {GROQ_FALLBACK_MODEL}")
            time.sleep(2)
            model_used = GROQ_FALLBACK_MODEL
            result = _call_groq(system, user_prompt, GROQ_FALLBACK_MODEL, api_key)
        else:
            raise

    latency = time.time() - t0
    answer = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    sources = [
        {
            "symbol": c.metadata.get("symbol"),
            "year": c.metadata.get("year"),
            "doc_type": c.metadata.get("doc_type"),
            "section": c.metadata.get("section") or c.metadata.get("speaker"),
            "page": c.metadata.get("page_start"),
            "score": round(c.score, 4),
        }
        for c in chunks
    ]

    log.info(f"  LLM response: {usage.get('completion_tokens',0)} tokens | {latency:.1f}s | {model_used}")

    return RAGResponse(
        answer=answer,
        model_used=model_used,
        chunks_used=len(chunks),
        sources=sources,
        tokens_used=usage.get("total_tokens", 0),
        latency_sec=round(latency, 2),
    )