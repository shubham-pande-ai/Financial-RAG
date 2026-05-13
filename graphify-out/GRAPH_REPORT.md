# Graph Report - FinRag  (2026-05-13)

## Corpus Check
- 28 files · ~27,211 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 263 nodes · 475 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 75 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `aef68b04`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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

## God Nodes (most connected - your core abstractions)
1. `AtomicNeed` - 14 edges
2. `_atom()` - 14 edges
3. `_pass()` - 14 edges
4. `get_conn()` - 13 edges
5. `ingest_pdf()` - 12 edges
6. `_build_sql()` - 12 edges
7. `SchemaBridge` - 12 edges
8. `_rule_based_decompose()` - 10 edges
9. `_run_annual_query()` - 10 edges
10. `retrieve_concall()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `ingest_pdf()` --calls--> `extract_pdf()`  [INFERRED]
  Ingest.py → pipeline/extract/pdf_extractor.py
- `ingest_pdf()` --calls--> `chunk_document()`  [INFERRED]
  Ingest.py → pipeline/loader/chunker.py
- `ingest_pdf()` --calls--> `load_chunks_to_chroma()`  [INFERRED]
  Ingest.py → pipeline/loader/chroma_loader.py
- `run_query()` --calls--> `retrieve()`  [INFERRED]
  query.py → pipeline/retrieval/retriever.py
- `run_query()` --calls--> `parse_year_intent()`  [INFERRED]
  query.py → pipeline/retrieval/retriever.py

## Communities (14 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (36): AtomicDecomposer, AtomicNeed, _extract_symbols(), _extract_years(), _infer_period(), _infer_time_horizon(), _llm_decompose(), NeedType (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (28): _detect_speaker_role(), extract_annual_report(), extract_concall(), extract_pdf(), _extract_prose_excluding_tables(), _extract_speaker_turns(), ExtractedDocument, _get_table_bboxes() (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.18
Nodes (26): _build_sql(), _classify(), _fy_date_range(), Indian FY: April of (fy_year-1) → March of fy_year.     e.g. FY2024 = 2023-04-0, Build a parameterized SELECT for one SQL-backed AtomicNeed.      Returns (sql_, Return 'sql', 'vector', or 'both'., _atom(), _fail() (+18 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (21): embed_query(), Embed a single query — same model, no reload., BM25, _build_where(), _expand_query(), _minmax(), _normalise_fy(), parse_year_intent() (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.16
Nodes (22): get_chunks_for_doc(), get_conn(), get_pending_documents(), get_stats(), init_db(), insert_chunk(), is_already_ingested(), log_ingestion() (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (19): interactive_mode(), main(), print_answer(), run_query(), _bge_score_batch(), _forward_boost(), _get_bge_reranker(), _get_voyage_client() (+11 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (19): _build_context(), build_provider_catalogue(), _build_user_prompt(), _call_gemini(), _call_openai_compat(), _call_with_retry(), _discover_ollama(), _estimate_tokens() (+11 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (14): collection_count(), get_chroma_client(), get_collection(), list_symbols(), load_chunks_to_chroma(), query_collection(), pipeline/loader/chroma_loader.py  Writes chunks + embeddings into ChromaDB. T, Query ChromaDB with a vector.     Returns chromadb QueryResult dict. (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (11): Architecture, code:block1 (Financial-Rag/), code:bash (# 1. Clone and install), Directory Structure, Financial RAG — Equity Research System, Groq Free Tier, High-Level Architecture, Low-Level Pipeline Design (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (9): clean_text(), fix_ligatures(), is_garbage_text(), normalize_numbers(), normalize_whitespace(), pipeline/extract/text_cleaner.py Normalize and clean text extracted from financ, Return True if text is too short or too noisy to be useful., Clean extracted PDF text.     aggressive=True removes more noise (good for pros (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.46
Nodes (7): classify_doc(), download_pdf(), extract_documents(), extract_year(), fetch_page(), main(), safe_name()

## Knowledge Gaps
- **77 isolated node(s):** `Map screener_downloader output folder name to doc_type.`, `screener_downloader saves to symbol/doc_type/year/filename.pdf`, `Find all PDF files for a symbol under screener_docs/.`, `config/settings.py Central configuration for the Financial RAG system.  FIXES`, `db/database.py SQLite metadata store — shared across both RAG pipelines.  Tab` (+72 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RetrievedChunk` connect `Community 3` to `Community 0`, `Community 6`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Why does `_execute_sql_atom()` connect `Community 0` to `Community 1`, `Community 2`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `ingest_pdf()` connect `Community 4` to `Community 1`, `Community 7`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `str` (e.g. with `ingest_pdf()` and `download_pdf()`) actually correct?**
  _`str` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `AtomicNeed` (e.g. with `SqlAtomResult` and `VectorAtomResult`) actually correct?**
  _`AtomicNeed` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Map screener_downloader output folder name to doc_type.`, `screener_downloader saves to symbol/doc_type/year/filename.pdf`, `Find all PDF files for a symbol under screener_docs/.` to the rest of the system?**
  _77 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._