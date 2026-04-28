"""
pipeline/retrieval/reranker.py

Primary  re-ranker : Voyage Rerank-2.5 (finance/SEC domain-tuned, 32k ctx)
Fallback re-ranker : Qwen/Qwen3-Reranker-8B (local HuggingFace, free, 32k ctx)

Fallback is activated automatically when:
  - VOYAGE_API_KEY is not set in .env
  - Voyage API returns HTTP 429 (rate limit) or 402 (quota exceeded)

Why Voyage Rerank-2.5 is the primary choice:
  - Finance/SEC domain fine-tuning: best precision on annual reports & concalls.
  - 32k token context window: handles full 512-token annual report chunks without
    truncation (cross-encoders cap at 512 tokens, silently dropping tail text).
  - API-based: no GPU/CPU overhead, no model loading delay.
  - Returns calibrated relevance scores in [0,1] with genuine separation.

Why Qwen3-Reranker-8B is the best local fallback:
  - 32k context (same as Voyage) — no truncation.
  - Free, runs on CPU (slow ~5-20s/batch) or GPU.
  - Significantly better than MiniLM/BGE cross-encoders on financial text.
  - HuggingFace model: Qwen/Qwen3-Reranker-8B
  - Install: pip install transformers torch accelerate
  - RAM: ~16 GB fp32 | ~8 GB fp16 | ~4 GB int8 (with bitsandbytes)

Scoring pipeline:
  1. voyageai.rerank() or Qwen3-Reranker-8B → relevance_score in [0, 1]
  2. _forward_boost()   → additive bonus/penalty for intent signals
  3. recency_boost()    → additive boost for more recent fiscal years
     [FIX C] recency_boost applied POST-rerank (was silently discarded before)
  4. Zero-score filter  → noise chunks never reach the LLM

[FIX A] Reranker swapped → Voyage Rerank-2.5 (finance domain-tuned, 32k ctx)
[FIX B] Zero-score chunks filtered before LLM call
[FIX C] recency_boost applied POST-rerank (was silently discarded before)
[FIX D] Single-chunk edge case: skip API call, assign score=1.0 directly
[FIX E] Qwen3-Reranker-8B local fallback when Voyage unavailable/rate-limited
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
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client

    api_key = os.getenv("VOYAGE_API_KEY", "")
    if not api_key:
        return None   # caller will use fallback

    try:
        import voyageai
    except ImportError:
        log.warning(
            "voyageai package not installed. Run: pip install voyageai\n"
            "Falling back to Qwen3-Reranker-8B."
        )
        return None

    _voyage_client = voyageai.Client(api_key=api_key)
    log.info(f"Voyage AI client initialised (reranker: {RERANKER_MODEL})")
    return _voyage_client


# ─────────────────────────────────────────────
# [FIX E] Qwen3-Reranker-8B local fallback
# ─────────────────────────────────────────────
_qwen_model   = None
_qwen_tokenizer = None


def _get_qwen_reranker():
    """
    Lazy-load Qwen3-Reranker-8B.  Returns (tokenizer, model) or raises.

    Qwen3-Reranker-8B uses a task-instruction format:
      <Instruct>: Given a web search query, retrieve relevant passages...
      <query>: {query}
      <document>: {document}

    The model is a causal LM; we take the logit of the "yes" token at the
    last position as the relevance score (standard Qwen3-Reranker recipe).
    """
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is not None:
        return _qwen_tokenizer, _qwen_model

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "Qwen3-Reranker-8B requires: pip install transformers torch accelerate\n"
            "For lower memory usage: pip install bitsandbytes  (enables int8 loading)"
        )

    model_name = RERANKER_FALLBACK
    log.info(f"Loading Qwen3-Reranker-8B from HuggingFace: {model_name}")
    log.info("  First run downloads ~16 GB. Subsequent runs use cache.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    _qwen_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    # Try bitsandbytes int8 first (halves RAM); fall back to fp16/fp32
    try:
        import bitsandbytes  # noqa: F401
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto",
        )
        log.info("  Qwen3-Reranker-8B loaded in int8 (bitsandbytes)")
    except (ImportError, Exception):
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            _qwen_model = _qwen_model.to(device)
        log.info(f"  Qwen3-Reranker-8B loaded in {dtype} on {device}")

    _qwen_model.eval()
    return _qwen_tokenizer, _qwen_model


# Qwen3-Reranker instruction template (from HF model card)
_QWEN_TASK = (
    "Given a financial document chunk and a query about company financials, "
    "determine if the chunk is relevant to answering the query."
)
_QWEN_PREFIX = "<Instruct>: {task}\n<query>: {query}\n<document>: "
_QWEN_SUFFIX = "\n<response>:"   # model predicts "yes" / "no" here


def _qwen_score_batch(query: str, documents: List[str]) -> List[float]:
    """
    Score each document against the query using Qwen3-Reranker-8B.
    Returns a list of float scores in approximately [0, 1].
    """
    import torch

    tokenizer, model = _get_qwen_reranker()
    device = next(model.parameters()).device

    prefix = _QWEN_PREFIX.format(task=_QWEN_TASK, query=query)

    # Token IDs for "yes" and "no" (Qwen tokeniser)
    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no",  add_special_tokens=False)[0]

    scores = []
    for doc in documents:
        text   = prefix + doc + _QWEN_SUFFIX
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=32768,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab_size)

        # Take the last token's logits and compare yes vs no
        last_logits = logits[0, -1, :]      # (vocab_size,)
        yes_logit   = last_logits[yes_id].item()
        no_logit    = last_logits[no_id].item()

        # Softmax over {yes, no} → probability of "yes"
        import math
        exp_yes = math.exp(yes_logit)
        exp_no  = math.exp(no_logit)
        score   = exp_yes / (exp_yes + exp_no)
        scores.append(score)

    return scores


# ─────────────────────────────────────────────
# Core rerank  (Voyage primary → Qwen3 fallback)
# ─────────────────────────────────────────────
def rerank(
    query: str,
    chunks: List[RetrievedChunk],
    doc_type: str,
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Re-rank chunks.
      Primary  : Voyage Rerank-2.5 (API, finance-tuned)
      Fallback : Qwen3-Reranker-8B (local HuggingFace)

    Pipeline:
      1. API/local reranker  → calibrated relevance scores in [0, 1]
      2. recency_boost       → post-rerank fiscal year nudge  [FIX C]
      3. forward_boost       → intent signal bonus/penalty
      4. zero-score filter   → noise never reaches LLM        [FIX B]
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

    # [FIX D] Single-chunk: skip API call — score 1.0 directly.
    if len(chunks) == 1:
        c  = chunks[0]
        yr = c.metadata.get("year", 0)
        c.score = 1.0 + recency_boost(yr) + _forward_boost(c.text, query)
        log.info("  Re-ranked: 1 chunk (single-chunk fast path)")
        return chunks

    documents   = [chunk.text for chunk in chunks]
    score_map   = {}
    used_voyage = False

    # ── Try Voyage first ──────────────────────────────────────────
    voyage_client = _get_voyage_client()
    if voyage_client is not None:
        try:
            reranking = voyage_client.rerank(
                query=query,
                documents=documents,
                model=RERANKER_MODEL,
                top_k=len(chunks),   # get all scores, we filter ourselves
                truncation=True,
            )
            score_map   = {r.index: r.relevance_score for r in reranking.results}
            used_voyage = True
            log.info(f"  Re-ranker: Voyage {RERANKER_MODEL}")
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", "ERR")
            log.warning(
                f"  Voyage rerank failed (HTTP {status}): {e}\n"
                f"  Falling back to Qwen3-Reranker-8B."
            )

    # ── Fallback: Qwen3-Reranker-8B ──────────────────────────────
    if not used_voyage:
        try:
            raw_scores = _qwen_score_batch(query, documents)
            score_map  = {i: s for i, s in enumerate(raw_scores)}
            log.info("  Re-ranker: Qwen3-Reranker-8B (local fallback)")
        except Exception as e:
            log.error(
                f"  Qwen3-Reranker-8B also failed: {e}\n"
                f"  Using raw retrieval scores as fallback."
            )
            # Last-resort: use existing hybrid scores unchanged
            score_map = {i: c.score for i, c in enumerate(chunks)}

    # ── Apply post-rerank boosts [FIX C] ─────────────────────────
    for i, chunk in enumerate(chunks):
        base_score  = score_map.get(i, 0.0)
        yr          = chunk.metadata.get("year", 0)
        fwd_boost   = _forward_boost(chunk.text, query)
        rec_boost   = recency_boost(yr)
        chunk.score = max(0.0, base_score + fwd_boost + rec_boost)

    # ── Filter noise-floor [FIX B] ────────────────────────────────
    meaningful = [c for c in chunks if c.score >= _MIN_SCORE_THRESHOLD]
    if not meaningful:
        meaningful = chunks   # safety: sparse query — keep all

    reranked   = sorted(meaningful, key=lambda c: c.score, reverse=True)
    top        = reranked[:top_k]
    n_filtered = len(chunks) - len(meaningful)
    filter_note = f" (filtered {n_filtered} below threshold)" if n_filtered else ""

    log.info(
        f"  Re-ranked: {len(chunks)} → top {len(top)}{filter_note} | "
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
    Rerank each collection independently so concall prose
    cannot displace annual report financial data chunks.
    The merged output is sorted by final score for clean LLM context ordering.
    """
    top_annual  = rerank(query, annual_chunks,  "annual_report", top_k=annual_top_k)
    top_concall = rerank(query, concall_chunks, "concall",       top_k=concall_top_k)

    merged = sorted(top_annual + top_concall, key=lambda c: c.score, reverse=True)

    log.info(
        f"  Merged: {len(top_annual)} annual + {len(top_concall)} concall "
        f"= {len(merged)} total chunks to LLM"
    )
    return merged