"""
server.py  — FastAPI wrapper around query.py
─────────────────────────────────────────────
Fixes BUG-1: new PID (new process) per query.

YOUR ACTUAL PROBLEM (from logs):
  Every query shows a new PID:  PID 14212, PID 10396, PID 23948...
  Python process dies after query.py exits.
  ALL singletons die with it: embedder, reranker, ChromaDB connection.
  Next query pays full cold-start: ~36s embed + ~9s reranker = 45s before
  a single token is scored.

THE FIX:
  Run this server once. It stays alive. All models load on first request
  and are reused for every subsequent query. From query 2 onward:
    - Embedder: 0s (already loaded)
    - Reranker model: 0s (already loaded)
    - ChromaDB: 0s (already connected)
    - Reranking (INT8 + parallel): ~8-15s
    - LLM: ~2-3s
    - Total: ~10-18s per query

USAGE:
    pip install fastapi uvicorn

    # Terminal 1 — start server once, keep it running
    python server.py

    # Terminal 2 — query it (replace your old query.py calls)
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"query": "ADANIPORTS revenue FY25", "symbol": "ADANIPORTS", "provider": "groq"}'

    # Or use the thin CLI wrapper so your workflow barely changes:
    python query_client.py --symbol ADANIPORTS "revenue FY25"

STARTUP TIME:
    First request after server start: ~45s (models loading)
    Every subsequent request: ~10-18s
    Server restart: only happens when YOU restart it (not between queries)
"""

from __future__ import annotations

import os
import sys
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Make sure project root is on path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.logger import get_logger
    from config.settings import LOG_DIR
    log = get_logger("server", LOG_DIR)
except Exception:
    import logging
    log = logging.getLogger("server")

# ─────────────────────────────────────────────────────────────────────────────
# Warm ALL heavy models at startup so the first query is fast
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm every heavy component before accepting requests."""
    log.info("[server] Warming models...")
    t0 = time.perf_counter()

    # Warm embedder — triggers model load on first call and caches it
    # Uses embed_query (the actual export) not get_embedder which doesn't exist
    try:
        from pipeline.loader.embedder import embed_query
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, embed_query, "warmup")
        log.info("[server] Embedder warm ✓")
    except Exception as e:
        log.warning(f"[server] Embedder warm failed: {e}")

    # Warm reranker (runs in background thread via its own warmup,
    # but we block here to ensure it's done before the first request)
    try:
        from pipeline.retrieval.reranker import _ensure_model
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _ensure_model)
        log.info("[server] Reranker warm ✓")
    except Exception as e:
        log.warning(f"[server] Reranker warm failed: {e}")

    # Warm ChromaDB connection
    try:
        from pipeline.loader.chroma_loader import get_chroma_client
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_chroma_client)
        log.info("[server] ChromaDB warm ✓")
    except Exception as e:
        log.warning(f"[server] ChromaDB warm failed: {e}")

    elapsed = time.perf_counter() - t0
    log.info(f"[server] All models warm in {elapsed:.1f}s — ready for queries")
    yield
    log.info("[server] Shutting down.")


app = FastAPI(title="FinRAG Server", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:    str
    symbol:   Optional[str]  = None
    doc_type: str             = "both"
    year:     Optional[int]  = None
    provider: str             = "auto"   # groq | or-qwen30b | gemini | auto


class QueryResponse(BaseModel):
    answer:        str
    model_used:    str
    chunks_used:   int
    sql_rows:      int
    insights:      int
    latency_sec:   float
    pipeline_mode: str
    sources:       list


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────────────────────────────────────

# Hard timeout for the whole pipeline (retrieval + reranking + LLM).
# Must be shorter than the client's 180s so we return a clean 504 instead
# of the client seeing a raw connection reset.
_QUERY_TIMEOUT_SEC = 160

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    t0 = time.perf_counter()
    log.info(f"[server] Query: {req.query!r} | symbol={req.symbol} | provider={req.provider}")

    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _run_query, req)
        # asyncio.wait_for wraps the future so we can cancel it on timeout
        result = await asyncio.wait_for(future, timeout=_QUERY_TIMEOUT_SEC)
        result["latency_sec"] = round(time.perf_counter() - t0, 2)
        log.info(f"[server] Done in {result['latency_sec']}s")
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        elapsed = round(time.perf_counter() - t0, 1)
        msg = (
            f"Query timed out after {elapsed}s (server limit {_QUERY_TIMEOUT_SEC}s). "
            "Try a faster provider (groq-llama) or --auto."
        )
        log.error(f"[server] {msg}")
        raise HTTPException(status_code=504, detail=msg)
    except Exception as e:
        log.error(f"[server] Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _run_query(req: QueryRequest) -> dict:
    """
    Synchronous query execution — runs in a thread pool so it doesn't
    block the event loop. All heavy objects (embedder, reranker, chroma)
    are already warm from lifespan startup.
    """
    from pipeline.retrieval.retriever import retrieve_with_years
    from pipeline.retrieval.reranker  import rerank_separate, rerank
    from rag.rag_engine               import generate_answer

    # ── Retrieval ─────────────────────────────────────────────────────────────
    raw, resolved_years, explicit_years = retrieve_with_years(
        query    = req.query,
        doc_type = req.doc_type,
        symbol   = req.symbol,
        year     = req.year,
    )

    # ── Reranking ─────────────────────────────────────────────────────────────
    if req.doc_type == "both" and isinstance(raw, tuple):
        annual_chunks, concall_chunks = raw
        chunks = rerank_separate(req.query, annual_chunks, concall_chunks)
    else:
        candidates = raw if not isinstance(raw, tuple) else raw[0]
        chunks = rerank(req.query, candidates, req.doc_type)

    # ── Answer generation ─────────────────────────────────────────────────────
    response = generate_answer(
        query          = req.query,
        chunks         = chunks,
        doc_type       = req.doc_type,
        symbol         = req.symbol,
        resolved_years = resolved_years,
        explicit_years = explicit_years,
        provider_id    = req.provider if req.provider != "auto" else None,
        auto           = req.provider == "auto",
    )

    return {
        "answer":        response.answer,
        "model_used":    response.model_used,
        "chunks_used":   response.chunks_used,
        "sql_rows":      getattr(response, "sql_rows", 0),
        "insights":      getattr(response, "insights", 0),
        "latency_sec":   0,   # filled by caller
        "pipeline_mode": getattr(response, "pipeline_mode", "full"),
        "sources":       getattr(response, "sources", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/providers")
async def providers_endpoint():
    """Return available provider list so query_client can show the menu."""
    try:
        from rag.rag_engine import build_provider_catalogue
        cat = build_provider_catalogue()
        return [{"id": e.id, "label": e.label, "note": e.context_note} for e in cat]
    except Exception as e:
        return []


@app.get("/health")
async def health():
    from pipeline.retrieval.reranker import _MODEL_CACHE, _MODEL_READY
    return {
        "status":          "ok",
        "reranker_warm":   _MODEL_CACHE is not None,
        "reranker_ready":  _MODEL_READY.is_set(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FINRAG_PORT", "8000"))
    log.info(f"[server] Starting FinRAG server on port {port}")
    uvicorn.run(
        "server:app",
        host    = "0.0.0.0",
        port    = port,
        workers = 1,        # single worker: models are process-level singletons
        reload  = False,    # never reload in production — would kill warm models
        log_level = "warning",
    )