"""
pipeline/retrieval/reranker.py

BUGS FIXED IN THIS VERSION
────────────────────────────

[BUG-2 FIX] export=True re-traced PyTorch → ONNX on EVERY cold start.
    The TracerWarning in your logs was the proof. 'export=True' on
    ORTModelForSequenceClassification triggers full PyTorch→ONNX tracing
    at import time, adding ~60s before the first query.
    FIX: Load from a pre-exported path (RERANKER_INT8_PATH). If path doesn't
    exist, falls back to fp32 CrossEncoder cleanly. Run
    tools/quantize_reranker.py once to produce the INT8 model.

[BUG-3 FIX] Model was NOT INT8 — it was fp32 ONNX.
    'export=True' exports fp32 only. ORTQuantizer was never called.
    Log said "INT8 ONNX reranker loaded" but inference took ~38s per 30 pairs
    (fp32 speed). True INT8 via ORTQuantizer should be 8-15s per 30 pairs.
    FIX: Load only from a path produced by quantize_reranker.py (which calls
    ORTQuantizer with AVX512_VNNI). No in-process quantization.

[BUG-4 FIX] rerank_separate() ran annual then concall sequentially.
    38s + 31s = 69s. They share NO state, so they can run in parallel.
    FIX: ThreadPoolExecutor runs both _rerank_batch calls concurrently.
    New time = max(38, 31) = 38s. Saves ~31s for free.

NOTE ON BUG-1 (new process per query):
    This file cannot fix the "new PID per query" problem — that's an
    architecture issue in how query.py is invoked. See server.py for the
    FastAPI wrapper that keeps the process alive between queries.
    Even with this fix, cold-start costs ~3-5s instead of ~65s.

Hardware note (i5-1240p, 16 GB RAM, Iris Xe)
──────────────────────────────────────────────
  fp32 PyTorch : 30 pairs ~40-70s   (broken old path)
  fp32 ONNX   : 30 pairs ~20-40s   (what the old code actually ran)
  INT8 ONNX   : 30 pairs ~ 8-15s   (this file, after quantize_reranker.py)
  INT8 + parallel: max(annual, concall) ~8-15s instead of sum

SETUP (one-time):
    python tools/quantize_reranker.py
    # add to .env:
    RERANKER_INT8_PATH=./models/bge-reranker-v2-m3-int8

USAGE (unchanged):
    from pipeline.retrieval.reranker import rerank, rerank_separate
    top = rerank(query, candidates, doc_type)
    top = rerank_separate(query, annual_cands, concall_cands)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

try:
    from config.settings import LOG_DIR
    log = get_logger(__name__, LOG_DIR)
except Exception:
    import logging
    log = logging.getLogger(__name__)

try:
    from pipeline.retrieval.retriever import RetrievedChunk
except ModuleNotFoundError:
    from dataclasses import dataclass, field
    from typing import Any, Dict
    @dataclass
    class RetrievedChunk:               # type: ignore[no-redef]
        chunk_id: str = ""; text: str = ""; score: float = 0.0
        vector_score: float = 0.0; bm25_score: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
_RERANKER_MODEL  = os.getenv("RERANKER_MODEL",    "BAAI/bge-reranker-v2-m3")
# Path to pre-quantized INT8 ONNX dir produced by tools/quantize_reranker.py
_INT8_PATH       = os.getenv("RERANKER_INT8_PATH",
                              str(Path("./models/bge-reranker-v2-m3-int8")))
_TOP_K_ANNUAL    = int(os.getenv("RERANKER_TOP_K_ANNUAL",  "12"))
_TOP_K_CONCALL   = int(os.getenv("RERANKER_TOP_K_CONCALL",  "6"))
_SCORE_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD",    "0.0"))


# ─────────────────────────────────────────────────────────────────────────────
# Model singleton — loaded ONCE per process
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: Optional[object] = None
_MODEL_LOCK:  threading.Lock   = threading.Lock()
_MODEL_READY: threading.Event  = threading.Event()


def _load_model_blocking() -> object:
    """
    Load the reranker model. Priority:
      1. Pre-quantized INT8 ONNX at _INT8_PATH  (~3s load, ~8-15s per 30 pairs)
      2. fp32 CrossEncoder fallback             (~10s load, ~40-70s per 30 pairs)

    CRITICAL: No export=True. That triggers PyTorch->ONNX tracing on every
    startup, which is what caused the ~60s TracerWarning delay in your logs.
    """
    int8_path = Path(_INT8_PATH)

    # ── Path 1: pre-quantized INT8 ONNX ──────────────────────────────────────
    # ORTQuantizer saves as "model_quantized.onnx", not "model.onnx"
    _INT8_FILENAME = "model_quantized.onnx"
    if int8_path.exists() and (int8_path / _INT8_FILENAME).exists():
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
            import torch

            log.info(f"[reranker] Loading INT8 ONNX from {int8_path}/{_INT8_FILENAME}")
            tok   = AutoTokenizer.from_pretrained(_RERANKER_MODEL)
            model = ORTModelForSequenceClassification.from_pretrained(
                str(int8_path),
                file_name=_INT8_FILENAME,        # explicit filename required
                provider="CPUExecutionProvider",
                # No export=True — loading pre-built INT8 model from disk
            )
            log.info("[reranker] INT8 ONNX loaded (~8-15s per 30 pairs expected)")

            class _OrtReranker:
                def __init__(self, m, t):
                    self.model, self.tok = m, t

                def predict(self, pairs: List[tuple]) -> List[float]:
                    enc = self.tok(
                        [p[0] for p in pairs],
                        [p[1] for p in pairs],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        out = self.model(**enc)
                    logits = out.logits.squeeze(-1)
                    scores = torch.sigmoid(logits).numpy().tolist()
                    return scores if isinstance(scores, list) else [scores]

            return _OrtReranker(model, tok)

        except Exception as e:
            log.warning(
                f"[reranker] INT8 load failed ({type(e).__name__}: {e})\n"
                f"           Run: python tools/quantize_reranker.py\n"
                f"           Falling back to fp32 CrossEncoder."
            )
    else:
        log.warning(
            f"[reranker] INT8 model not found at {int8_path / 'model_quantized.onnx'}\n"
            f"           Run: python tools/quantize_reranker.py  (one-time, ~5 min)\n"
            f"           Falling back to fp32 CrossEncoder (~40-70s per 30 pairs)."
        )

    # ── Path 2: fp32 CrossEncoder fallback ───────────────────────────────────
    from sentence_transformers.cross_encoder import CrossEncoder
    log.info(f"[reranker] Loading fp32 CrossEncoder: {_RERANKER_MODEL}")
    reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)
    log.info("[reranker] fp32 CrossEncoder loaded (slow — run quantize_reranker.py)")
    return reranker


def _ensure_model() -> object:
    """Return the cached model, loading it if needed (thread-safe)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    with _MODEL_LOCK:
        if _MODEL_CACHE is None:
            _MODEL_CACHE = _load_model_blocking()
            _MODEL_READY.set()
    return _MODEL_CACHE


def _warm_model_background():
    """Fire model loading in a daemon thread so the first query is faster."""
    t = threading.Thread(target=_ensure_model, daemon=True, name="reranker-warmup")
    t.start()


# Start warming immediately on module import
_warm_model_background()


# ─────────────────────────────────────────────────────────────────────────────
# Core reranking
# ─────────────────────────────────────────────────────────────────────────────

def _rerank_batch(
    query:      str,
    candidates: List[RetrievedChunk],
    top_k:      int,
    label:      str = "",
) -> List[RetrievedChunk]:
    """
    Score all candidates in one batched call and return top_k.
    Falls back to original vector scores if model inference fails.
    """
    if not candidates:
        return []

    top_k = min(top_k, len(candidates))

    try:
        import time as _time
        model   = _ensure_model()
        pairs   = [(query, c.text) for c in candidates]
        t0      = _time.perf_counter()
        scores  = model.predict(pairs)
        elapsed = (_time.perf_counter() - t0) * 1000

        scored = [
            RetrievedChunk(
                chunk_id     = c.chunk_id,
                text         = c.text,
                score        = float(sc),
                vector_score = c.vector_score,
                bm25_score   = c.bm25_score,
                metadata     = c.metadata,
            )
            for c, sc in zip(candidates, scores)
        ]

        scored.sort(key=lambda c: c.score, reverse=True)
        top = [c for c in scored if c.score >= _SCORE_THRESHOLD][:top_k]

        tag = f"[{label}] " if label else ""
        if top:
            log.info(
                f"  {tag}Re-ranked {len(candidates)} → top {len(top)} | "
                f"scores [{top[-1].score:.4f}–{top[0].score:.4f}] | {elapsed:.0f}ms"
            )
        else:
            log.info(f"  {tag}Re-ranked {len(candidates)} → 0 above threshold | {elapsed:.0f}ms")
        return top

    except Exception as exc:
        log.warning(f"  Reranker failed ({exc}) — using original vector scores")
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    query:      str,
    candidates: List[RetrievedChunk],
    doc_type:   str = "annual_report",
    top_k:      int = None,
) -> List[RetrievedChunk]:
    """Rerank a flat list of candidates (single doc_type or mixed)."""
    if top_k is None:
        top_k = _TOP_K_CONCALL if doc_type == "concall" else _TOP_K_ANNUAL
    return _rerank_batch(query, candidates, top_k)


def rerank_separate(
    query:              str,
    annual_candidates:  List[RetrievedChunk],
    concall_candidates: List[RetrievedChunk],
    annual_top_k:       int = None,
    concall_top_k:      int = None,
) -> List[RetrievedChunk]:
    """
    Rerank annual and concall candidates in PARALLEL, then merge.

    BUG-4 FIX: Previously sequential (38s + 31s = 69s).
    Now concurrent: max(38s, 31s) = 38s. Saves ~31s for free.
    Annual and concall are independent — no shared state.
    """
    annual_top_k  = annual_top_k  or _TOP_K_ANNUAL
    concall_top_k = concall_top_k or _TOP_K_CONCALL

    top_annual:  List[RetrievedChunk] = []
    top_concall: List[RetrievedChunk] = []

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker") as pool:
        future_annual  = pool.submit(
            _rerank_batch, query, annual_candidates,  annual_top_k,  "annual"
        )
        future_concall = pool.submit(
            _rerank_batch, query, concall_candidates, concall_top_k, "concall"
        )
        for future in as_completed([future_annual, future_concall]):
            if future is future_annual:
                top_annual  = future.result()
            else:
                top_concall = future.result()

    merged = top_annual + top_concall
    log.info(
        f"  Merged: {len(top_annual)} annual + "
        f"{len(top_concall)} concall = {len(merged)} total chunks to LLM"
    )
    return merged