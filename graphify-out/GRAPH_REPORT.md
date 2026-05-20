# Graph Report - FinRag  (2026-05-20)

## Corpus Check
- 42 files · ~46,746 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 606 nodes · 1137 edges · 20 communities (16 shown, 4 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 221 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `86c47d02`
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
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]

## God Nodes (most connected - your core abstractions)
1. `SynthesisPipeline` - 29 edges
2. `RetrievedChunk` - 27 edges
3. `PromptBuilder` - 25 edges
4. `BridgeResult` - 21 edges
5. `AtomicNeed` - 19 edges
6. `VectorAtomResult` - 19 edges
7. `SynthesisResult` - 19 edges
8. `_pass()` - 19 edges
9. `_pass()` - 18 edges
10. `FusionResult` - 17 edges

## Surprising Connections (you probably didn't know these)
- `ingest_pdf()` --calls--> `extract_pdf()`  [INFERRED]
  Ingest.py → pipeline/extract/pdf_extractor.py
- `ingest_pdf()` --calls--> `chunk_document()`  [INFERRED]
  Ingest.py → pipeline/loader/chunker.py
- `run_query()` --calls--> `retrieve()`  [INFERRED]
  query.py → pipeline/retrieval/retriever.py
- `run_query()` --calls--> `parse_year_intent()`  [INFERRED]
  query.py → pipeline/retrieval/retriever.py
- `run_query()` --calls--> `generate_answer()`  [INFERRED]
  query.py → rag/rag_engine.py

## Communities (20 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (62): AtomicNeed, _llm_decompose(), NeedType, One atomic information need extracted from the user query., Call LLM to decompose a query into atoms., TimeHorizon, Enum, BridgeResult (+54 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (58): _build_gap_flag_note(), _build_intent_note(), _build_metric_note(), _fmt_value(), PromptBuilder, synthesis/prompt_builder.py  Layer 5 of the Quant CoPilot Intent Decomposition, Format a metric value with its unit for display., Render the list of MetricRow dicts as a compact ASCII table.      Groups rows (+50 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (42): _extract_numeric_claims(), Scan chunk text sentence-by-sentence.     For each sentence that contains a key, _strip_commas(), _atom(), _bridge(), _chunk(), _fail(), _pass() (+34 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (39): BaseModel, _expand_onward_years(), lifespan(), providers_endpoint(), providers_validate(), query_endpoint(), QueryRequest, QueryResponse (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (37): _build_sql(), _classify(), _execute_sql_atom(), _expand_comparative(), _fy_date_range(), schema_bridge/schema_bridge.py  Layer 2 of the Quant CoPilot Intent Decomposit, Indian FY: April of (fy_year-1) → March of fy_year.     e.g. FY2024 = 2023-04-0, Indian FY: April of (fy_year-1) → March of fy_year.     e.g. FY2024 = 2023-04-0 (+29 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (36): _detect_speaker_role(), extract_annual_report(), extract_concall(), extract_pdf(), _extract_prose_excluding_tables(), _extract_speaker_turns(), ExtractedDocument, _get_table_bboxes() (+28 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (34): get_chunks_for_doc(), get_conn(), get_pending_documents(), get_stats(), init_db(), insert_chunk(), is_already_ingested(), log_ingestion() (+26 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (23): _compute_metrics(), DatasetBuilder, EvalDataset, EvalResults, GoldenSample, load(), _nanmean(), _print_query_row() (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (30): query_collection(), Query ChromaDB with a vector.     Returns chromadb QueryResult dict., embed_query(), Embed a single query — same model, no reload., BM25, _build_where(), _expand_query(), _minmax() (+22 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (24): AtomicDecomposer, _extract_symbols(), _extract_years(), _infer_period(), _infer_time_horizon(), _is_ebitda_multi_year_query(), _normalise_fy(), decomposer/atomic_decomposer.py  — patched  Bug fixes applied in this version (+16 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (30): interactive_mode(), main(), print_answer(), run_query(), Synchronous query execution — runs in a thread pool so it doesn't     block the, _run_query(), _bge_score_batch(), _ensure_model() (+22 more)

### Community 11 - "Community 11"
Cohesion: 0.09
Nodes (21): cached_generate(), _deserialize(), _exact_key(), FakeResponse, get_cache(), _meta_key(), pipeline/retrieval/semantic_cache.py ───────────────────────────────────── Sma, Deterministic hash key for exact-match lookup. (+13 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (29): ═══════════════════════════════════════════════════════════════════════════, BUG 1 — Voyage AI: repeated HTTP failure on every query [FIXED], BUG 2 — Reranker model reloads on every query [FIXED, CRITICAL LATENCY], BUG 3 — Embedding model also reloads on every query [MANUAL FIX NEEDED], BUG 4 — Symbol not passed to decomposer → SQL returns wrong/no data [FIXED], BUG 5 — YoY / multi-year queries only parse the first year [FIXED], BUG 6 — EBITDA CAGR: fundamentals only has latest snapshot [FIXED], BUG 7 — Groq context trimming drops most chunks [INFORMATIONAL] (+21 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (11): Architecture, code:block1 (Financial-Rag/), code:bash (# 1. Clone and install), Directory Structure, Financial RAG — Equity Research System, Groq Free Tier, High-Level Architecture, Low-Level Pipeline Design (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.46
Nodes (7): classify_doc(), download_pdf(), extract_documents(), extract_year(), fetch_page(), main(), safe_name()

### Community 15 - "Community 15"
Cohesion: 0.47
Nodes (5): fetch_providers(), main(), pick_provider(), query_client.py — drop-in replacement for query.py ────────────────────────────, Try to get live provider list from server; fall back to hardcoded.

## Knowledge Gaps
- **219 isolated node(s):** `Map screener_downloader output folder name to doc_type.`, `screener_downloader saves to symbol/doc_type/year/filename.pdf`, `Find all PDF files for a symbol under screener_docs/.`, `latency_optimizer.py ───────────────────── Latency analysis and optimization f`, `query_client.py — drop-in replacement for query.py ────────────────────────────` (+214 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RetrievedChunk` connect `Community 0` to `Community 8`, `Community 1`, `Community 10`, `Community 3`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `AtomicNeed` connect `Community 0` to `Community 9`, `Community 2`, `Community 4`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `SynthesisPipeline` connect `Community 0` to `Community 9`, `Community 3`, `Community 1`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `SynthesisPipeline` (e.g. with `ProviderEntry` and `RAGResponse`) actually correct?**
  _`SynthesisPipeline` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `RetrievedChunk` (e.g. with `RetrievedChunk` and `SqlAtomResult`) actually correct?**
  _`RetrievedChunk` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `PromptBuilder` (e.g. with `RetrievedChunk` and `SynthesisResult`) actually correct?**
  _`PromptBuilder` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `BridgeResult` (e.g. with `RetrievedChunk` and `SqlAtomResult`) actually correct?**
  _`BridgeResult` has 17 INFERRED edges - model-reasoned connections that need verification._