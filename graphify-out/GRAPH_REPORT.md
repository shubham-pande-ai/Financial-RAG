# Graph Report - Financial-Rag  (2026-06-15)

## Corpus Check
- 39 files · ~47,247 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 622 nodes · 1664 edges · 65 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS · INFERRED: 790 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]

## God Nodes (most connected - your core abstractions)
1. `RetrievedChunk` - 100 edges
2. `BridgeResult` - 62 edges
3. `VectorAtomResult` - 60 edges
4. `SynthesisPipeline` - 55 edges
5. `AtomicNeed` - 54 edges
6. `PromptBuilder` - 50 edges
7. `NeedType` - 49 edges
8. `TimeHorizon` - 49 edges
9. `FusionResult` - 49 edges
10. `SqlAtomResult` - 47 edges

## Surprising Connections (you probably didn't know these)
- `pipeline/retrieval/reranker.py  BUGS FIXED IN THIS VERSION ──────────────────` --uses--> `RetrievedChunk`  [INFERRED]
  pipeline/retrieval/reranker.py → pipeline\retrieval\retriever.py
- `Load the reranker model. Priority:       1. Pre-quantized INT8 ONNX at _INT8_PA` --uses--> `RetrievedChunk`  [INFERRED]
  pipeline/retrieval/reranker.py → pipeline\retrieval\retriever.py
- `Return the cached model, loading it if needed (thread-safe).` --uses--> `RetrievedChunk`  [INFERRED]
  pipeline/retrieval/reranker.py → pipeline\retrieval\retriever.py
- `Fire model loading in a daemon thread so the first query is faster.` --uses--> `RetrievedChunk`  [INFERRED]
  pipeline/retrieval/reranker.py → pipeline\retrieval\retriever.py
- `Score all candidates in one batched call and return top_k.     Falls back to or` --uses--> `RetrievedChunk`  [INFERRED]
  pipeline/retrieval/reranker.py → pipeline\retrieval\retriever.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (96): AtomicDecomposer, AtomicNeed, _extract_years(), _is_ebitda_multi_year_query(), _llm_decompose(), NeedType, _normalise_fy(), decomposer/atomic_decomposer.py  — patched  Bug fixes applied in this version (+88 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (56): _builder_available(), _infer_symbol(), _minimal_fallback_prompt(), _pipeline_available(), SynthesisPipeline, SynthesisResult, _build_gap_flag_note(), _build_intent_note() (+48 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (49): run_tests(), embed_query(), embed_texts(), _get_model(), pipeline/loader/embedder.py  Singleton embedding model — loaded ONCE per proce, Embed a single query — same model, no reload., RAGResponse, BM25 (+41 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (50): _detect_speaker_role(), extract_annual_report(), extract_concall(), extract_pdf(), _extract_prose_excluding_tables(), _extract_speaker_turns(), ExtractedDocument, _get_table_bboxes() (+42 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (34): BaseModel, _expand_onward_years(), lifespan(), providers_endpoint(), providers_validate(), query_endpoint(), QueryRequest, QueryResponse (+26 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (23): _compute_metrics(), DatasetBuilder, EvalDataset, EvalResults, GoldenSample, load(), _nanmean(), _print_query_row() (+15 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (29): _build_context_legacy(), build_provider_catalogue(), _build_user_prompt_legacy(), _call_anthropic(), _call_gemini(), _call_openai_compat(), _call_with_retry(), _discover_ollama() (+21 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (29): ═══════════════════════════════════════════════════════════════════════════, BUG 1 — Voyage AI: repeated HTTP failure on every query [FIXED], BUG 2 — Reranker model reloads on every query [FIXED, CRITICAL LATENCY], BUG 3 — Embedding model also reloads on every query [MANUAL FIX NEEDED], BUG 4 — Symbol not passed to decomposer → SQL returns wrong/no data [FIXED], BUG 5 — YoY / multi-year queries only parse the first year [FIXED], BUG 6 — EBITDA CAGR: fundamentals only has latest snapshot [FIXED], BUG 7 — Groq context trimming drops most chunks [INFORMATIONAL] (+21 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (27): _extract_numeric_claims(), _strip_commas(), _atom(), _bridge(), _chunk(), _fail(), _pass(), run() (+19 more)

### Community 9 - "Community 9"
Cohesion: 0.16
Nodes (22): get_chunks_for_doc(), get_conn(), get_pending_documents(), get_stats(), init_db(), insert_chunk(), is_already_ingested(), log_ingestion() (+14 more)

### Community 10 - "Community 10"
Cohesion: 0.21
Nodes (21): _build_sql(), _classify(), _execute_sql_atom(), _fy_date_range(), _atom(), _fail(), _pass(), run_all() (+13 more)

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (11): Architecture, code:block1 (Financial-Rag/), code:bash (# 1. Clone and install), Directory Structure, Financial RAG — Equity Research System, Groq Free Tier, High-Level Architecture, Low-Level Pipeline Design (+3 more)

### Community 12 - "Community 12"
Cohesion: 0.46
Nodes (7): classify_doc(), download_pdf(), extract_documents(), extract_year(), fetch_page(), main(), safe_name()

### Community 13 - "Community 13"
Cohesion: 0.47
Nodes (5): fetch_providers(), main(), pick_provider(), query_client.py — drop-in replacement for query.py ────────────────────────────, Try to get live provider list from server; fall back to hardcoded.

### Community 14 - "Community 14"
Cohesion: 0.5
Nodes (1): config/settings.py Central configuration for the Financial RAG system.  FIXES

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (1): utils/logger.py Structured logging with file + console output.

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (1): latency_optimizer.py ───────────────────── Latency analysis and optimization f

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Map screener_downloader output folder name to doc_type.

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): screener_downloader saves to symbol/doc_type/year/filename.pdf

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Find all PDF files for a symbol under screener_docs/.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): latency_optimizer.py ───────────────────── Latency analysis and optimization f

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): query_client.py — drop-in replacement for query.py ────────────────────────────

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Try to get live provider list from server; fall back to hardcoded.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): server.py  — FastAPI wrapper around query.py ──────────────────────────────────

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Pre-warm every heavy component before accepting requests.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Rewrite 'from FY23 onward/onwards/forward/to present/to date/since FY23'     to

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Synchronous query execution — runs in a thread pool so it doesn't     block the

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Return available provider list so query_client can show the menu.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Validate all providers by checking their model slugs are still live.     Return

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): # NOTE: cancelling does NOT kill the underlying thread (Python limitation),

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Simplified trimmer: for large-context providers returns all chunks.     For Gro

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Retry policy (revised):       • 429 (rate-limit)  → back off and retry (max 2 r

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Generate a cited answer using the full synthesis pipeline.      New parameters

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): One atomic information need extracted from the user query.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Populate sql_table and sql_columns from SUBTYPE_TABLE_MAP.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Extract explicitly mentioned fiscal years from a query string.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Extract NSE symbols mentioned in the query (uppercase tokens).

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Apply pattern rules to extract atomic needs. Returns list (may be empty).

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Post-processing: when the query has forward-looking signals AND a QUANTITATIVE

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Call LLM to decompose a query into atoms.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Decompose a user query into a list of AtomicNeed objects.          Strategy:

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Same as decompose() but returns a diagnostic dict:           {             "qu

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Additive score adjustment:       + bonus  when query is forward-looking AND chu

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Returns a Voyage client if VOYAGE_API_KEY is set, else None.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Lazy-load BAAI/bge-reranker-v2-m3 via sentence-transformers.      Memory footp

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Score each document against the query using bge-reranker-v2-m3.      CrossEnco

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Re-rank retrieved chunks.       Primary  : Voyage Rerank-2.5  (API, finance/SEC

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Rerank each collection independently so concall prose cannot displace     annua

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Additive boost so FY2025 ranks above FY2015 at equal semantic similarity.     S

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Returns:       - List[RetrievedChunk]                   for doc_type in (annual

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Returns (chunks_or_tuple, resolved_years, explicit_years).      explicit_years

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): # NOTE: recency_boost is intentionally NOT applied here.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Query /api/tags; return one entry per installed model. Never raises.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Build the user-turn prompt.      resolved_years — used for the year-filter not

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Drop lowest-ranked chunks until the full prompt fits within:       (a) the mode

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): resolved_years — years used for ChromaDB retrieval filter.     explicit_years —

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Indian FY: April of (fy_year-1) → March of fy_year.     e.g. FY2024 = 2023-04-0

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Build a parameterized SELECT for one SQL-backed AtomicNeed.      Returns (sql_

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Run the SQL for one atom, return SqlAtomResult (never raises).

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Run a ChromaDB vector query for one atom, return VectorAtomResult.

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): A COMPARATIVE atom targets multiple symbols.     Expand it into one atom per sy

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Return 'sql', 'vector', or 'both'.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Translates a list of AtomicNeed objects into concrete data-fetch results.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Dispatch all atoms to the appropriate channel(s), running them in         paral

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Helper for the common case: fetch several metrics for one company.          Ex

## Knowledge Gaps
- **130 isolated node(s):** `Map screener_downloader output folder name to doc_type.`, `screener_downloader saves to symbol/doc_type/year/filename.pdf`, `Find all PDF files for a symbol under screener_docs/.`, `latency_optimizer.py ───────────────────── Latency analysis and optimization f`, `query_client.py — drop-in replacement for query.py ────────────────────────────` (+125 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 14`** (4 nodes): `__init__.py`, `_load_dotenv()`, `settings.py`, `config/settings.py Central configuration for the Financial RAG system.  FIXES`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (4 nodes): `__init__.py`, `get_logger()`, `logger.py`, `utils/logger.py Structured logging with file + console output.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (3 nodes): `print_summary()`, `latency_optimizer.py ───────────────────── Latency analysis and optimization f`, `latency_optimizer.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Map screener_downloader output folder name to doc_type.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `screener_downloader saves to symbol/doc_type/year/filename.pdf`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Find all PDF files for a symbol under screener_docs/.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `latency_optimizer.py ───────────────────── Latency analysis and optimization f`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `query_client.py — drop-in replacement for query.py ────────────────────────────`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Try to get live provider list from server; fall back to hardcoded.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `server.py  — FastAPI wrapper around query.py ──────────────────────────────────`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Pre-warm every heavy component before accepting requests.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Rewrite 'from FY23 onward/onwards/forward/to present/to date/since FY23'     to`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Synchronous query execution — runs in a thread pool so it doesn't     block the`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Return available provider list so query_client can show the menu.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Validate all providers by checking their model slugs are still live.     Return`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `# NOTE: cancelling does NOT kill the underlying thread (Python limitation),`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Simplified trimmer: for large-context providers returns all chunks.     For Gro`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Retry policy (revised):       • 429 (rate-limit)  → back off and retry (max 2 r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Generate a cited answer using the full synthesis pipeline.      New parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `One atomic information need extracted from the user query.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Populate sql_table and sql_columns from SUBTYPE_TABLE_MAP.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Extract explicitly mentioned fiscal years from a query string.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Extract NSE symbols mentioned in the query (uppercase tokens).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Apply pattern rules to extract atomic needs. Returns list (may be empty).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Post-processing: when the query has forward-looking signals AND a QUANTITATIVE`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Call LLM to decompose a query into atoms.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Decompose a user query into a list of AtomicNeed objects.          Strategy:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Same as decompose() but returns a diagnostic dict:           {             "qu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Additive score adjustment:       + bonus  when query is forward-looking AND chu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Returns a Voyage client if VOYAGE_API_KEY is set, else None.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Lazy-load BAAI/bge-reranker-v2-m3 via sentence-transformers.      Memory footp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Score each document against the query using bge-reranker-v2-m3.      CrossEnco`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Re-rank retrieved chunks.       Primary  : Voyage Rerank-2.5  (API, finance/SEC`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Rerank each collection independently so concall prose cannot displace     annua`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Additive boost so FY2025 ranks above FY2015 at equal semantic similarity.     S`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Returns:       - List[RetrievedChunk]                   for doc_type in (annual`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Returns (chunks_or_tuple, resolved_years, explicit_years).      explicit_years`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `# NOTE: recency_boost is intentionally NOT applied here.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Query /api/tags; return one entry per installed model. Never raises.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Build the user-turn prompt.      resolved_years — used for the year-filter not`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Drop lowest-ranked chunks until the full prompt fits within:       (a) the mode`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `resolved_years — years used for ChromaDB retrieval filter.     explicit_years —`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Indian FY: April of (fy_year-1) → March of fy_year.     e.g. FY2024 = 2023-04-0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Build a parameterized SELECT for one SQL-backed AtomicNeed.      Returns (sql_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Run the SQL for one atom, return SqlAtomResult (never raises).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Run a ChromaDB vector query for one atom, return VectorAtomResult.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `A COMPARATIVE atom targets multiple symbols.     Expand it into one atom per sy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Return 'sql', 'vector', or 'both'.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Translates a list of AtomicNeed objects into concrete data-fetch results.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Dispatch all atoms to the appropriate channel(s), running them in         paral`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Helper for the common case: fetch several metrics for one company.          Ex`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RetrievedChunk` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `SynthesisPipeline` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 6`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `RAGResponse` connect `Community 2` to `Community 0`, `Community 1`, `Community 6`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 97 inferred relationships involving `RetrievedChunk` (e.g. with `RetrievedChunk` and `SqlAtomResult`) actually correct?**
  _`RetrievedChunk` has 97 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `BridgeResult` (e.g. with `RetrievedChunk` and `SqlAtomResult`) actually correct?**
  _`BridgeResult` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `VectorAtomResult` (e.g. with `RetrievedChunk` and `SqlAtomResult`) actually correct?**
  _`VectorAtomResult` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 45 inferred relationships involving `SynthesisPipeline` (e.g. with `ProviderEntry` and `RAGResponse`) actually correct?**
  _`SynthesisPipeline` has 45 INFERRED edges - model-reasoned connections that need verification._