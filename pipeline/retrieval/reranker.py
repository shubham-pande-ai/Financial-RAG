"""
pipeline/retrieval/reranker.py

Primary  re-ranker : Voyage Rerank-2.5 (finance/SEC domain-tuned, 32k ctx)
Fallback re-ranker : BAAI/bge-reranker-v2-m3 (local cross-encoder, ~1.1 GB fp16)

The fallback activates automatically when:
  - VOYAGE_API_KEY is not set in .env
  - Voyage API returns HTTP 429 (rate limit) or 402 (quota exceeded)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why BAAI/bge-reranker-v2-m3 and NOT Fin-E5 or Qwen3-Reranker-8B
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fin-E5 is fine-tuned on e5-mistral-7b-instruct — a 7B causal LM.
  fp16 weights alone = ~14 GB.  On a 16 GB machine this leaves < 2 GB for
  the OS, Python, ChromaDB, and the 768-dim embedding model.  The process
  will either OOM-kill immediately or grind to a halt on swap.

Qwen3-Reranker-8B is an 8B causal LM.
  fp16 = ~8 GB, fp32 = ~16 GB.  Same problem on 16 GB total RAM.

Both models were designed for GPU server deployment, not a 16 GB laptop CPU.

BAAI/bge-reranker-v2-m3 is a cross-encoder (BERT-style encoder, NOT a LM).
  ~568 M parameters → ~1.1 GB fp16 / ~2.2 GB fp32.
  Leaves 13-14 GB free — fully comfortable on i5-1240P / 16 GB.
  512-token context window matches chunk_size = 512 exactly.
  Top-ranked cross-encoder on MTEB reranking leaderboard.
  Outputs raw logit scores — genuine separation, not softmax-collapsed.
  No extra dependencies beyond sentence-transformers (already installed).
  Install: pip install sentence-transformers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scoring pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Voyage rerank-2 or bge-reranker-v2-m3 → relevance score per chunk
  2. _forward_boost()   → additive bonus/penalty for intent signals
  3. recency_boost()    → additive fiscal-year nudge  [applied POST-rerank]
  4. Zero-score filter  → noise chunks never reach the LLM

[FIX A] Voyage Rerank-2.5 as primary (finance domain-tuned, 32k ctx)
[FIX B] Zero-score chunks filtered before LLM call
[FIX C] recency_boost applied POST-rerank (was silently discarded before)
[FIX D] Single-chunk fast path: skip API call, assign score=1.0
[FIX E] bge-reranker-v2-m3 as local fallback — fits on 16 GB comfortably
"""

import os
import re
from typing import List, Optional

from config.settings import (
    ANNUAL_RETRIEVAL, CONCALL_RETRIEVAL,
    RERANKER_MODEL, RERANKER_FALLBACK,
)
from pipeline.retrieval.retriever import RetrievedChunk, recency_boost
from utils.logger import get_logger

log = get_logger(__name__)

# Minimum relevance score for a chunk to be passed to the LLM.
# Voyage scores are well-calibrated — anything below 0.05 is noise.
# bge-reranker-v2-m3 outputs sigmoid(logit) in [0,1]; same threshold applies.
_MIN_SCORE_THRESHOLD = 0.05

# ─────────────────────────────────────────────
# Forward / noise signal patterns
# ─────────────────────────────────────────────
_FORWARD_SIGNALS = re.compile(
    r"\b(we expect|we anticipate|going forward|our outlook|guidance|H1 FY|H2 FY"
    r"|next quarter|next year|demand environment|we are confident|target|forecast"
    r"|we plan|we intend|capex plan|volume target|revenue target)\b",
    re.IGNORECASE,
)

_NOISE_SIGNALS = re.compile(
    r"\b(ladies and gentlemen|welcome to|thank you for joining"
    r"|good morning|good evening|good afternoon"
    r"|please go ahead|next question|operator|moderator)\b",
    re.IGNORECASE,
)


def _forward_boost(chunk_text: str, query: str) -> float:
    """
    Additive score adjustment:
      + bonus  when query is forward-looking AND chunk has guidance language
      - penalty when chunk contains moderator/intro noise
    """
    q = query.lower()
    is_forward_query = any(kw in q for kw in [
        "outlook", "guidance", "expect", "h1", "h2", "demand environment",
        "going forward", "next", "forecast", "target", "plan",
    ])

    bonus = 0.0
    if is_forward_query:
        n_forward = len(_FORWARD_SIGNALS.findall(chunk_text))
        bonus += min(n_forward * 0.015, 0.06)

    n_noise = len(_NOISE_SIGNALS.findall(chunk_text))
    bonus  -= min(n_noise * 0.02, 0.08)

    return bonus


# ─────────────────────────────────────────────
# Voyage AI client (lazy singleton)
# ─────────────────────────────────────────────
_voyage_client = None


def _get_voyage_client():
    """Returns a Voyage client if VOYAGE_API_KEY is set, else None."""
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client

    api_key = os.getenv("VOYAGE_API_KEY", "")
    if not api_key:
        return None   # no key → caller uses local fallback

    try:
        import voyageai
    except ImportError:
        log.warning(
            "voyageai package not installed. Run: pip install voyageai\n"
            "Falling back to BAAI/bge-reranker-v2-m3."
        )
        return None

    _voyage_client = voyageai.Client(api_key=api_key)
    log.info(f"Voyage AI client initialised (reranker: {RERANKER_MODEL})")
    return _voyage_client


# ─────────────────────────────────────────────
# [FIX E] BAAI/bge-reranker-v2-m3 local fallback
# ─────────────────────────────────────────────
_bge_model = None


def _get_bge_reranker():
    """
    Lazy-load BAAI/bge-reranker-v2-m3 via sentence-transformers.

    Memory footprint:
      fp32 on CPU  : ~2.2 GB  (comfortable on 16 GB)
      fp16 on GPU  : ~1.1 GB
    Context window : 512 tokens  (matches chunk_size in settings)
    Dependencies   : sentence-transformers  (already used by embedder)
    """
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for the local reranker fallback.\n"
            "Run: pip install sentence-transformers"
        )

    model_name = RERANKER_FALLBACK   # "BAAI/bge-reranker-v2-m3"
    log.info(f"Loading local fallback reranker: {model_name}")
    log.info("  ~568M params | ~2.2 GB fp32 | fits on 16 GB RAM")

    # CrossEncoder handles truncation internally at 512 tokens.
    # max_length matches your chunk_size so no financial text is silently cut.
    _bge_model = CrossEncoder(model_name, max_length=512)
    log.info(f"  {model_name} loaded (cross-encoder, CPU-friendly)")
    return _bge_model


def _bge_score_batch(query: str, documents: List[str]) -> List[float]:
    """
    Score each document against the query using bge-reranker-v2-m3.

    CrossEncoder.predict() returns raw logits by default.
    We apply sigmoid to convert to [0, 1] for consistency with Voyage scores
    and with _MIN_SCORE_THRESHOLD = 0.05.

    Sigmoid preserves score ordering so ranking is unaffected.
    """
    import math

    model  = _get_bge_reranker()
    pairs  = [[query, doc] for doc in documents]

    # predict() returns a numpy array of raw logits
    logits = model.predict(pairs, show_progress_bar=False)

    # sigmoid(logit) → probability in (0, 1)
    scores = [1.0 / (1.0 + math.exp(-float(l))) for l in logits]
    return scores


# ─────────────────────────────────────────────
# Core rerank  (Voyage primary → BGE fallback)
# ─────────────────────────────────────────────
def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    doc_type: str,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Re-rank retrieved chunks.
      Primary  : Voyage Rerank-2.5  (API, finance/SEC fine-tuned, 32k ctx)
      Fallback : BAAI/bge-reranker-v2-m3  (local cross-encoder, ~2.2 GB RAM)

    Pipeline:
      1. Voyage or BGE       → relevance score in [0, 1] per chunk
      2. recency_boost       → post-rerank fiscal-year nudge     [FIX C]
      3. _forward_boost      → intent-signal bonus / noise penalty
      4. zero-score filter   → noise chunks never reach the LLM   [FIX B]
      5. return top_k sorted by final score
    """
    if not chunks:
        return []

    if top_k is None:
        top_k = (
            ANNUAL_RETRIEVAL["top_k_rerank"]
            if doc_type == "annual_report"
            else CONCALL_RETRIEVAL["top_k_rerank"]
        )

    # [FIX D] Single-chunk fast path — skip API/model call entirely.
    if len(chunks) == 1:
        c  = chunks[0]
        yr = c.metadata.get("year", 0)
        c.score = 1.0 + recency_boost(yr) + _forward_boost(c.text, query)
        log.info("  Re-ranked: 1 chunk (single-chunk fast path)")
        return chunks

    documents = [chunk.text for chunk in chunks]
    score_map: dict = {}
    backend_used = "none"

    # ── 1. Try Voyage Rerank-2.5 ─────────────────────────────────
    voyage_client = _get_voyage_client()
    if voyage_client is not None:
        try:
            reranking = voyage_client.rerank(
                query=query,
                documents=documents,
                model=RERANKER_MODEL,   # "rerank-2"
                top_k=len(chunks),      # get all scores; we filter ourselves
                truncation=True,        # safe fallback for any oversized chunks
            )
            score_map    = {r.index: r.relevance_score for r in reranking.results}
            backend_used = f"Voyage {RERANKER_MODEL}"
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", "ERR")
            log.warning(
                f"  Voyage rerank failed (HTTP {status}): {e}\n"
                f"  Falling back to {RERANKER_FALLBACK}."
            )

    # ── 2. Fallback: BAAI/bge-reranker-v2-m3 ────────────────────
    if not score_map:
        try:
            raw_scores   = _bge_score_batch(query, documents)
            score_map    = {i: s for i, s in enumerate(raw_scores)}
            backend_used = RERANKER_FALLBACK
        except Exception as e:
            log.error(
                f"  {RERANKER_FALLBACK} failed: {e}\n"
                f"  Using raw hybrid retrieval scores as last resort."
            )
            # Absolute last resort: keep existing hybrid scores unchanged
            score_map    = {i: c.score for i, c in enumerate(chunks)}
            backend_used = "raw-hybrid-score"

    log.info(f"  Re-ranker backend: {backend_used}")

    # ── 3. Apply post-rerank boosts [FIX C] ──────────────────────
    for i, chunk in enumerate(chunks):
        base_score  = score_map.get(i, 0.0)
        yr          = chunk.metadata.get("year", 0)
        # recency_boost is applied HERE (post-rerank) so it is not silently
        # overwritten when the reranker replaces chunk.score.
        chunk.score = max(0.0,
                          base_score
                          + _forward_boost(chunk.text, query)
                          + recency_boost(yr))

    # ── 4. Filter noise-floor chunks [FIX B] ─────────────────────
    meaningful = [c for c in chunks if c.score >= _MIN_SCORE_THRESHOLD]
    if not meaningful:
        # Safety fallback: sparse/off-topic query → keep all rather than nothing
        meaningful = chunks

    reranked   = sorted(meaningful, key=lambda c: c.score, reverse=True)
    top        = reranked[:top_k]
    n_filtered = len(chunks) - len(meaningful)
    note       = f" (filtered {n_filtered} below threshold)" if n_filtered else ""

    log.info(
        f"  Re-ranked: {len(chunks)} → top {len(top)}{note} | "
        f"score range [{top[-1].score:.4f} – {top[0].score:.4f}]"
    )
    return top


# ─────────────────────────────────────────────
# Separate rerank for "both" mode
# ─────────────────────────────────────────────
def rerank_separate(
    query: str,
    annual_chunks: List[RetrievedChunk],
    concall_chunks: List[RetrievedChunk],
    annual_top_k: int,
    concall_top_k: int,
) -> List[RetrievedChunk]:
    """
    Rerank each collection independently so concall prose cannot displace
    annual report financial data chunks.
    The merged list is sorted by final score for a clean LLM context handoff.
    """
    top_annual  = rerank(query, annual_chunks,  "annual_report", top_k=annual_top_k)
    top_concall = rerank(query, concall_chunks, "concall",       top_k=concall_top_k)

    merged = sorted(top_annual + top_concall, key=lambda c: c.score, reverse=True)

    log.info(
        f"  Merged: {len(top_annual)} annual + {len(top_concall)} concall "
        f"= {len(merged)} total chunks to LLM"
    )
    return merged