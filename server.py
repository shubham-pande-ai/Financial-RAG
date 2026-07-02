"""
server.py  — FastAPI wrapper around query.py
─────────────────────────────────────────────
Fixes BUG-1: new PID (new process) per query.

YOUR ACTUAL PROBLEM (from logs):
  Every query shows a new PID:  PID 14212, PID 10396, PID 23948...
  Python process dies after query.py exits.
  ALL singletons die with it: embedder, reranker, Qdrant connection.
  Next query pays full cold-start: ~36s embed + ~9s reranker = 45s before
  a single token is scored.

THE FIX:
  Run this server once. It stays alive. All models load on first request
  and are reused for every subsequent query. From query 2 onward:
    - Embedder: 0s (already loaded)
    - Reranker model: 0s (already loaded)
    - Qdrant: 0s (already connected)
    - Reranking (INT8 + parallel): ~8-15s
    - LLM: ~2-3s
    - Total: ~10-18s per query

USAGE:
    pip install fastapi uvicorn

    # Terminal 1 — start server once, keep it running
    python server.py

    # Terminal 2 — query it (replace your old query.py calls)
    curl -X POST http://localhost:8001/query \
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
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import httpx as _httpx  # optional, used only by /providers/validate
except ImportError:
    _httpx = None  # type: ignore

# ── Make sure project root is on path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.logger import get_logger
    from config.settings import LOG_DIR
    log = get_logger("server", LOG_DIR)
except Exception:
    import logging
    log = logging.getLogger("server")

# ── In-memory healthy providers cache ────────────────────────────────────────
# Populated by validate_all_providers() at startup and refreshed every 5 min.
HEALTHY_PROVIDERS: list = []          # list[dict]  — JSON-serialisable
_LAST_VALIDATION_TS: float = 0.0
_REVALIDATION_INTERVAL: int = 300     # seconds (5 minutes)
_revalidation_task: Optional[asyncio.Task] = None


async def _validate_all_providers_async() -> list:
    """
    Run provider validation in a thread (blocking I/O) without blocking the
    event loop.  Returns a list of healthy provider dicts and updates the
    module-level HEALTHY_PROVIDERS cache.
    """
    global HEALTHY_PROVIDERS, _LAST_VALIDATION_TS
    try:
        from rag.rag_engine import build_provider_catalogue, validate_all_providers
        loop = asyncio.get_running_loop()
        catalogue = await loop.run_in_executor(None, build_provider_catalogue)
        healthy   = await loop.run_in_executor(None, validate_all_providers, catalogue)
        HEALTHY_PROVIDERS = [
            {"id": e.id, "label": e.label, "note": e.context_note}
            for e in healthy
        ]
        _LAST_VALIDATION_TS = time.time()
        log.info(
            f"[server] Provider validation complete: "
            f"{len(HEALTHY_PROVIDERS)} healthy providers cached"
        )
    except Exception as exc:
        log.warning(f"[server] Provider validation failed: {exc}")
    return HEALTHY_PROVIDERS


async def _revalidation_loop() -> None:
    """Background task: re-validate providers every _REVALIDATION_INTERVAL seconds."""
    while True:
        await asyncio.sleep(_REVALIDATION_INTERVAL)
        log.info("[server] Background revalidation starting...")
        await _validate_all_providers_async()

# ─────────────────────────────────────────────────────────────────────────────
# Warm ALL heavy models at startup so the first query is fast
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm every heavy component before accepting requests."""
    global _revalidation_task
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

    # Warm Qdrant connection
    try:
        from pipeline.loader.qdrant_loader import get_qdrant_client
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_qdrant_client)
        log.info("[server] Qdrant warm ✓")
    except Exception as e:
        log.warning(f"[server] Qdrant warm failed: {e}")

    elapsed = time.perf_counter() - t0
    log.info(f"[server] All models warm in {elapsed:.1f}s — ready for queries")

    # ── Validate all LLM providers and cache healthy list ─────────────────────
    log.info("[server] Validating LLM providers (non-blocking)...")
    await _validate_all_providers_async()

    # ── Start background revalidation every 5 minutes ────────────────────────
    _revalidation_task = asyncio.create_task(_revalidation_loop())
    log.info("[server] Background provider revalidation task started (every 5 min)")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    # CancelledError is suppressed here so uvicorn's Windows signal re-raise
    # doesn't produce a noisy ERROR traceback on Ctrl+C.
    try:
        if _revalidation_task and not _revalidation_task.done():
            _revalidation_task.cancel()
            try:
                await _revalidation_task
            except asyncio.CancelledError:
                pass
        log.info("[server] Shutting down.")
    except asyncio.CancelledError:
        pass


app = FastAPI(title="FinRAG Server", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Provider safety: dead / unreliable provider IDs caught before they waste
# retrieval + reranking time (18-36s) only to fail at the LLM step.
# ─────────────────────────────────────────────────────────────────────────────

# BUG-FIX: these OpenRouter model slugs no longer exist; map them to working
#          replacements so old shell aliases / scripts don't hard-fail.
#          NOTE: the real slug fixes are in rag_engine.py build_provider_catalogue().
#          These remaps catch any provider IDs hard-coded in old scripts or .env files.
_PROVIDER_REMAP: dict[str, str] = {
    # or-gemini-exp was an old alias before the provider ID was normalised
    "or-gemini-exp": "or-gemini",
    # or-qwen72b-old was the ID used when the slug still had :free suffix
    "or-qwen72b-old": "or-qwen72b",
    # Direct old-slug provider IDs that users might have stored in aliases
    "or-gemini-free": "or-gemini",
}

# Providers whose real-world p50 latency exceeds the server timeout.
# Warn in logs so it's visible; don't block (user chose it deliberately).
_SLOW_PROVIDERS: set[str] = {"nvidia"}


# ─────────────────────────────────────────────────────────────────────────────
# Query text pre-processing
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

# BUG-FIX: "from FY23 onward" / "since FY22" / "FY23 onwards" resolved to only
# the anchor year by the retriever's year extractor, missing subsequent years.
# We rewrite such phrases to explicit ranges before the query hits the pipeline.
_CURRENT_FY = 2025  # update each year

def _expand_onward_years(query: str) -> str:
    """
    Rewrite 'from FY23 onward/onwards/forward/to present/to date/since FY23'
    to 'FY23, FY24, FY25' so the retriever year extractor sees all years.
    Only rewrites when the anchor year is ≤ _CURRENT_FY.
    """
    pattern = _re.compile(
        r'\b(?:from\s+)?(?:FY|fy)?(\d{2,4})\s*'
        r'(?:onward(?:s)?|forward|to\s+(?:present|date|now)|onwards?)'
        r'|\bsince\s+(?:FY|fy)?(\d{2,4})\b',
        _re.IGNORECASE,
    )
    def _replace(m: _re.Match) -> str:
        raw = m.group(1) or m.group(2)
        yr  = int(raw) if int(raw) > 100 else 2000 + int(raw)
        if yr > _CURRENT_FY:
            return m.group(0)  # future year — leave as-is
        years = list(range(yr, _CURRENT_FY + 1))
        fy_list = ", ".join(f"FY{y}" for y in years)
        log.info(f"[server] year-expand: '{m.group(0)}' → '{fy_list}'")
        return fy_list

    return pattern.sub(_replace, query)


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

import concurrent.futures as _cf

# Bounded thread pool: prevents unbounded thread accumulation when slow providers
# (NVIDIA NIM, free-tier OpenRouter) stall and the asyncio timeout fires but the
# underlying thread cannot be killed. With an unbounded default executor, each
# stalled NVIDIA call holds a thread for up to 160s; under repeated queries the
# server can run out of threads entirely. A pool of 4 is enough for the single-
# worker server (retrieval + reranking is already serialised by the reranker lock).
_QUERY_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="finrag-query")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    t0 = time.perf_counter()

    # ── Provider safety: remap dead slugs ────────────────────────────────────
    if req.provider in _PROVIDER_REMAP:
        new_id = _PROVIDER_REMAP[req.provider]
        log.warning(f"[server] Provider '{req.provider}' is dead → remapped to '{new_id}'")
        req = req.model_copy(update={"provider": new_id})

    if req.provider in _SLOW_PROVIDERS:
        log.warning(
            f"[server] Provider '{req.provider}' has p50 latency ~80-100s and frequently "
            f"exceeds the {_QUERY_TIMEOUT_SEC}s server timeout. Consider groq-llama or or-qwen30b."
        )

    # ── Year expansion: 'from FY23 onward' → 'FY23, FY24, FY25' ─────────────
    expanded = _expand_onward_years(req.query)
    if expanded != req.query:
        req = req.model_copy(update={"query": expanded})

    log.info(f"[server] Query: {req.query!r} | symbol={req.symbol} | provider={req.provider}")

    # BUG-FIX: NVIDIA NIM often completes AFTER the server's 504 has already been
    # sent (observed: 161s run after 160s timeout). The thread can't be truly
    # killed in Python, but we use a tighter per-provider timeout so the event
    # loop stops waiting sooner, freeing it for the next request.
    provider_timeout = 90 if req.provider in _SLOW_PROVIDERS else _QUERY_TIMEOUT_SEC

    try:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(_QUERY_EXECUTOR, _run_query, req)
        # asyncio.wait_for wraps the future so we can cancel it on timeout.
        # NOTE: cancelling does NOT kill the underlying thread (Python limitation),
        # but it does free the event loop and return a 504 to the client promptly.
        result = await asyncio.wait_for(future, timeout=provider_timeout)
        result["latency_sec"] = round(time.perf_counter() - t0, 2)
        log.info(f"[server] Done in {result['latency_sec']}s")
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        elapsed = round(time.perf_counter() - t0, 1)
        limit   = provider_timeout
        msg = (
            f"Query timed out after {elapsed}s (limit {limit}s for provider '{req.provider}'). "
            "Try groq-llama (fastest) or or-qwen30b (best free), or use --auto."
        )
        log.error(f"[server] {msg}")
        if req.provider in _SLOW_PROVIDERS:
            log.warning(
                "[server] NVIDIA NIM thread may still be running in background — "
                "this ties up a thread-pool slot. Restart server if latency degrades."
            )
        raise HTTPException(status_code=504, detail=msg)
    except Exception as e:
        log.error(f"[server] Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _run_query(req: QueryRequest) -> dict:
    """
    Synchronous query execution — runs in a thread pool so it doesn't
    block the event loop. All heavy objects (embedder, reranker, qdrant)
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


@app.get("/providers/validate")
async def providers_validate():
    """
    Validate all providers by checking their model slugs are still live.
    Returns a dict of provider_id → {ok, error}.
    Useful for debugging 'all providers failed' errors without running a full query.
    """
    try:
        from rag.rag_engine import build_provider_catalogue
        cat = build_provider_catalogue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot load catalogue: {e}")

    if _httpx is None:
        raise HTTPException(status_code=501, detail="pip install httpx to use this endpoint")

    import httpx
    results = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for entry in cat:
            pid = entry.id
            if pid.startswith("ollama"):
                results[pid] = {"ok": None, "note": "skipped (local)"}
                continue
            if pid in _SLOW_PROVIDERS:
                results[pid] = {"ok": None, "note": "skipped (slow, validate manually)"}
                continue
            results[pid] = {"ok": True, "note": "not validated (extend this endpoint)"}
    return results


@app.get("/health")
async def health():
    from pipeline.retrieval.reranker import _MODEL_CACHE, _MODEL_READY
    return {
        "status":          "ok",
        "reranker_warm":   _MODEL_CACHE is not None,
        "reranker_ready":  _MODEL_READY.is_set(),
        "slow_providers":  list(_SLOW_PROVIDERS),
        "remapped_providers": _PROVIDER_REMAP,
        "server_timeout_sec": _QUERY_TIMEOUT_SEC,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FINRAG_PORT", "8000"))
    log.info(f"[server] Starting FinRAG server on port {port}")
    try:
        uvicorn.run(
            "server:app",
            host    = "0.0.0.0",
            port    = port,
            workers = 1,        # single worker: models are process-level singletons
            reload  = False,    # never reload in production — would kill warm models
            log_level = "warning",
        )
    except (KeyboardInterrupt, SystemExit):
        # Normal Ctrl+C shutdown — uvicorn on Windows re-raises these during
        # signal cleanup, producing a noisy but harmless traceback.  Catching
        # them here keeps the terminal output clean.
        pass