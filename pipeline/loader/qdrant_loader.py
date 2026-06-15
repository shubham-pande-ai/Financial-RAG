"""
pipeline/loader/qdrant_loader.py

Replaces chroma_loader.py.  Writes chunks + embeddings into Qdrant.

Two collections:
  annual_reports   — 768-dim cosine
  concalls         — 768-dim cosine

Qdrant point IDs are the chunk UUIDs (stored as string UUID → Qdrant UUID type).
All ChromaDB metadata fields are preserved as Qdrant payload keys so existing
retriever filters work without changes (just swap import + function names).

Dependencies:
    pip install qdrant-client
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    UpdateStatus,
)

from config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_ANNUAL_COLLECTION,
    QDRANT_CONCALL_COLLECTION,
    EMBEDDING_DIM,
)
from pipeline.loader.chunker import Chunk
from pipeline.loader.embedder import embed_texts, build_embedding_text
from utils.logger import get_logger

log = get_logger(__name__)

_client: Optional[QdrantClient] = None


# ─────────────────────────────────────────────
# Client (singleton)
# ─────────────────────────────────────────────
def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        kwargs: dict = {"host": QDRANT_HOST, "port": QDRANT_PORT}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _client = QdrantClient(**kwargs)
        log.info(f"Qdrant client initialised → {QDRANT_HOST}:{QDRANT_PORT}")
    return _client


# ─────────────────────────────────────────────
# Collection bootstrap
# ─────────────────────────────────────────────
def _ensure_collection(name: str) -> None:
    """Create collection if it doesn't exist. Never deletes existing data."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name = name,
            vectors_config  = VectorParams(
                size     = EMBEDDING_DIM,
                distance = Distance.COSINE,
            ),
        )
        log.info(f"  Created Qdrant collection '{name}' (dim={EMBEDDING_DIM}, cosine)")


def get_collection_name(doc_type: str) -> str:
    return (
        QDRANT_ANNUAL_COLLECTION if doc_type == "annual_report"
        else QDRANT_CONCALL_COLLECTION
    )


# ─────────────────────────────────────────────
# Write chunks
# ─────────────────────────────────────────────
def load_chunks_to_qdrant(
    chunks:     List[Chunk],
    doc_type:   str,
    batch_size: int = 64,
) -> str:
    """
    Embed and upsert chunks into Qdrant.
    Returns collection name.

    Embedding text is built via build_embedding_text() which prefixes
    structured context (symbol, doc_type, section, tags, etc.) so the
    vector captures semantic role, not just raw words.
    """
    if not chunks:
        log.warning("No chunks to load")
        return ""

    collection_name = get_collection_name(doc_type)
    _ensure_collection(collection_name)
    client = get_qdrant_client()

    log.info(f"  Loading {len(chunks)} chunks → Qdrant '{collection_name}'")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        # Build rich embedding payloads (section + speaker + tags prefix)
        embed_texts_batch = [build_embedding_text(c) for c in batch]
        embeddings        = embed_texts(embed_texts_batch)

        points = []
        for chunk, emb in zip(batch, embeddings):
            payload = {
                "chunk_id":    chunk.chunk_id,
                "symbol":      chunk.symbol,
                "year":        chunk.year or 0,
                "title":       (chunk.title or "")[:200],
                "doc_type":    chunk.doc_type,
                "chunk_type":  chunk.chunk_type,
                "section":     (chunk.section or "")[:200],
                "speaker":     (chunk.speaker or "")[:100],
                "speaker_role": (chunk.speaker_role or ""),
                "page_start":  chunk.page_start or 0,
                "page_end":    chunk.page_end   or 0,
                "word_count":  chunk.word_count,
                "chunk_index": chunk.chunk_index,
                # Store the display text in payload for retrieval
                "text":        chunk.text,
            }
            points.append(
                PointStruct(
                    id      = chunk.chunk_id,   # UUID string — Qdrant accepts this
                    vector  = emb,
                    payload = payload,
                )
            )

        result = client.upsert(
            collection_name = collection_name,
            points          = points,
            wait            = True,
        )
        if result.status != UpdateStatus.COMPLETED:
            log.warning(f"  Qdrant upsert batch {i // batch_size + 1} status: {result.status}")
        else:
            log.debug(f"  Upserted batch {i // batch_size + 1}: {len(batch)} chunks")

    log.info(f"  ✓ Loaded {len(chunks)} chunks into '{collection_name}'")
    return collection_name


# ─────────────────────────────────────────────
# Query Qdrant
# ─────────────────────────────────────────────
def query_collection(
    doc_type:        str,
    query_embedding: List[float],
    top_k:           int           = 20,
    where:           Optional[dict] = None,   # same filter-dict format as before
) -> list[dict]:
    """
    Search Qdrant with a vector.
    `where` accepts the same key-value dict used with ChromaDB filters, e.g.:
        {"symbol": "RELIANCE", "year": 2024}

    Returns list of dicts with keys: id, score, payload (text, metadata).
    """
    collection_name = get_collection_name(doc_type)
    client          = get_qdrant_client()

    qdrant_filter = _build_filter(where) if where else None

    results = client.search(
        collection_name = collection_name,
        query_vector    = query_embedding,
        limit           = top_k,
        query_filter    = qdrant_filter,
        with_payload    = True,
    )

    return [
        {
            "id":       str(r.id),
            "score":    r.score,
            "payload":  r.payload,
            # Flatten common fields for backward compat with retriever code
            "text":     r.payload.get("text", ""),
            "metadata": {k: v for k, v in r.payload.items() if k != "text"},
        }
        for r in results
    ]


def _build_filter(where: dict) -> Filter:
    """
    Convert a flat {key: value} dict to a Qdrant Filter.
    Supports: str/int equality and int range via {"$gte": x, "$lte": y}.
    """
    conditions = []
    for key, value in where.items():
        if isinstance(value, dict):
            # Range filter: {"year": {"$gte": 2022, "$lte": 2024}}
            gte = value.get("$gte")
            lte = value.get("$lte")
            conditions.append(
                FieldCondition(key=key, range=Range(gte=gte, lte=lte))
            )
        else:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
    return Filter(must=conditions)


# ─────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────
def collection_count(doc_type: str) -> int:
    client = get_qdrant_client()
    name   = get_collection_name(doc_type)
    try:
        info = client.get_collection(name)
        return info.points_count or 0
    except Exception:
        return 0


def list_symbols(doc_type: str) -> List[str]:
    """List unique symbols in a collection via scroll (use sparingly)."""
    client = get_qdrant_client()
    name   = get_collection_name(doc_type)
    symbols: set[str] = set()

    offset = None
    while True:
        records, offset = client.scroll(
            collection_name = name,
            limit           = 100,
            offset          = offset,
            with_payload    = ["symbol"],
        )
        for r in records:
            if r.payload and "symbol" in r.payload:
                symbols.add(r.payload["symbol"])
        if offset is None:
            break

    return sorted(symbols)


def delete_by_symbol(doc_type: str, symbol: str) -> int:
    """
    Delete all points for a given symbol from a collection.
    Useful for re-ingesting a single company without dropping the full collection.
    Returns number of deleted points.
    """
    client = get_qdrant_client()
    name   = get_collection_name(doc_type)

    result = client.delete(
        collection_name = name,
        points_selector = Filter(
            must=[FieldCondition(key="symbol", match=MatchValue(value=symbol.upper()))]
        ),
        wait=True,
    )
    count = collection_count(doc_type)
    log.info(f"  Deleted symbol '{symbol}' from '{name}' — status: {result.status}")
    return count