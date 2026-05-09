# Graph Report - FinRag  (2026-05-09)

## Corpus Check
- 22 files · ~20,410 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 179 nodes · 292 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c90edd8c`
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

## God Nodes (most connected - your core abstractions)
1. `get_conn()` - 12 edges
2. `ingest_pdf()` - 11 edges
3. `_run_annual_query()` - 10 edges
4. `generate_answer()` - 10 edges
5. `run_query()` - 9 edges
6. `retrieve_concall()` - 9 edges
7. `extract_annual_report()` - 8 edges
8. `rerank()` - 8 edges
9. `clean_text()` - 7 edges
10. `get_collection()` - 7 edges

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

## Communities (11 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.13
Nodes (23): query_collection(), Query ChromaDB with a vector.     Returns chromadb QueryResult dict., embed_query(), Embed a single query — same model, no reload., BM25, _build_where(), _expand_query(), _minmax() (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (22): get_chunks_for_doc(), get_conn(), get_pending_documents(), get_stats(), init_db(), insert_chunk(), is_already_ingested(), log_ingestion() (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (19): interactive_mode(), main(), print_answer(), run_query(), _bge_score_batch(), _forward_boost(), _get_bge_reranker(), _get_voyage_client() (+11 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (18): clean_text(), fix_ligatures(), is_garbage_text(), normalize_numbers(), normalize_whitespace(), pipeline/extract/text_cleaner.py Normalize and clean text extracted from financ, Return True if text is too short or too noisy to be useful., Clean extracted PDF text.     aggressive=True removes more noise (good for pros (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (19): _build_context(), build_provider_catalogue(), _build_user_prompt(), _call_gemini(), _call_openai_compat(), _call_with_retry(), _discover_ollama(), _estimate_tokens() (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.18
Nodes (18): _detect_speaker_role(), extract_annual_report(), extract_concall(), extract_pdf(), _extract_prose_excluding_tables(), _extract_speaker_turns(), ExtractedDocument, _get_table_bboxes() (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.19
Nodes (12): collection_count(), get_chroma_client(), get_collection(), list_symbols(), load_chunks_to_chroma(), pipeline/loader/chroma_loader.py  Writes chunks + embeddings into ChromaDB. T, List all unique symbols in a collection., Get or create the ChromaDB collection for a doc type. (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (11): Architecture, code:block1 (Financial-Rag/), code:bash (# 1. Clone and install), Directory Structure, Financial RAG — Equity Research System, Groq Free Tier, High-Level Architecture, Low-Level Pipeline Design (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.46
Nodes (7): classify_doc(), download_pdf(), extract_documents(), extract_year(), fetch_page(), main(), safe_name()

## Knowledge Gaps
- **53 isolated node(s):** `Map screener_downloader output folder name to doc_type.`, `screener_downloader saves to symbol/doc_type/year/filename.pdf`, `Find all PDF files for a symbol under screener_docs/.`, `config/settings.py Central configuration for the Financial RAG system.  FIXES`, `db/database.py SQLite metadata store — shared across both RAG pipelines.  Tab` (+48 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ingest_pdf()` connect `Community 1` to `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.244) - this node is a cross-community bridge._
- **Why does `run_query()` connect `Community 2` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Why does `init_db()` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `ingest_pdf()` (e.g. with `is_already_ingested()` and `upsert_document()`) actually correct?**
  _`ingest_pdf()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_run_annual_query()` (e.g. with `embed_query()` and `query_collection()`) actually correct?**
  _`_run_annual_query()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `run_query()` (e.g. with `retrieve()` and `parse_year_intent()`) actually correct?**
  _`run_query()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Map screener_downloader output folder name to doc_type.`, `screener_downloader saves to symbol/doc_type/year/filename.pdf`, `Find all PDF files for a symbol under screener_docs/.` to the rest of the system?**
  _53 weakly-connected nodes found - possible documentation gaps or missing edges._