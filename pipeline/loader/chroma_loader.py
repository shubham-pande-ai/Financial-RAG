"""
pipeline/loader/chroma_loader.py

Writes chunks + embeddings into ChromaDB.
Two separate collections: annual_reports / concalls
Each collection persists to disk at CHROMA_DIR.
"""

from typing import List, Optional
import chromadb
from chromadb.config import Settings

from config.settings import (
    CHROMA_DIR,
    CHROMA_ANNUAL_COLLECTION,
    CHROMA_CONCALL_COLLECTION,
)
from pipeline.loader.chunker import Chunk
from pipeline.loader.embedder import embed_texts
from utils.logger import get_logger

log = get_logger(__name__)

_client: Optional[chromadb.PersistentClient] = None


# ─────────────────────────────────────────────
# Client (singleton)
# ─────────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        log.info(f"ChromaDB client initialised at {CHROMA_DIR}")
    return _client


def get_collection(doc_type: str):
    """Get or create the ChromaDB collection for a doc type."""
    client = get_chroma_client()
    name = (
        CHROMA_ANNUAL_COLLECTION if doc_type == "annual_report"
        else CHROMA_CONCALL_COLLECTION
    )
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    return collection, name


# ─────────────────────────────────────────────
# Write chunks
# ─────────────────────────────────────────────
def load_chunks_to_chroma(
    chunks: List[Chunk],
    doc_type: str,
    batch_size: int = 50,
) -> str:
    """
    Embed and upsert chunks into ChromaDB.
    Returns collection name.
    """
    if not chunks:
        log.warning("No chunks to load")
        return ""

    collection, collection_name = get_collection(doc_type)

    log.info(f"  Loading {len(chunks)} chunks → ChromaDB collection '{collection_name}'")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]

        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]

        # Build metadata dicts (ChromaDB only supports str/int/float/bool)
        metadatas = []
        for c in batch:
            metadatas.append({
                "symbol": c.symbol,
                "year": c.year or 0,
                "title": c.title[:200],
                "doc_type": c.doc_type,
                "chunk_type": c.chunk_type,
                "section": (c.section or "")[:200],
                "speaker": (c.speaker or "")[:100],
                "speaker_role": (c.speaker_role or ""),
                "page_start": c.page_start or 0,
                "page_end": c.page_end or 0,
                "word_count": c.word_count,
                "chunk_index": c.chunk_index,
            })

        embeddings = embed_texts(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.debug(f"  Upserted batch {i//batch_size + 1}: {len(batch)} chunks")

    log.info(f"  ✓ Loaded {len(chunks)} chunks into '{collection_name}'")
    return collection_name


# ─────────────────────────────────────────────
# Query ChromaDB
# ─────────────────────────────────────────────
def query_collection(
    doc_type: str,
    query_embedding: List[float],
    top_k: int = 20,
    where: Optional[dict] = None,
) -> dict:
    """
    Query ChromaDB with a vector.
    Returns chromadb QueryResult dict.
    """
    collection, _ = get_collection(doc_type)

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def collection_count(doc_type: str) -> int:
    collection, _ = get_collection(doc_type)
    return collection.count()


def list_symbols(doc_type: str) -> List[str]:
    """List all unique symbols in a collection."""
    collection, _ = get_collection(doc_type)
    if collection.count() == 0:
        return []
    # Peek at all metadata (can be slow for large collections — use sparingly)
    result = collection.get(include=["metadatas"])
    symbols = list({m.get("symbol", "") for m in result["metadatas"]})
    return sorted(symbols)