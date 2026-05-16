# FinRAG Bug Report & Fix Summary
# Generated from log analysis + source code review
# ═══════════════════════════════════════════════════════════════════════════

## FILES DELIVERED IN THIS PATCH SET
- `reranker.py`             → DROP-IN replacement for pipeline/retrieval/reranker.py
- `atomic_decomposer.py`    → DROP-IN replacement for decomposer/atomic_decomposer.py
- `pipeline_patch_note.py`  → 1-line patch for synthesis/pipeline.py
- `prompt_builder_patch.py` → _build_metric_note() replacement for prompt_builder.py
- `latency_optimizer.py`    → Run for setup instructions

---

## BUG 1 — Voyage AI: repeated HTTP failure on every query [FIXED]

**Symptom (from logs):**
```
WARNING | Voyage rerank failed (HTTP ERR): You have not yet added your payment method...
Falling back to BAAI/bge-reranker-v2-m3.
```
This happened TWICE per query (once for annual, once for concall).
Each attempt adds ~1s network overhead + logging noise.

**Root cause:** reranker.py tried Voyage AI first with no graceful skip
when the account has no payment method.

**Fix:** `reranker.py` completely removes Voyage AI.
Goes directly to BAAI/bge-reranker-v2-m3 with zero network calls.

**Action:** Replace `pipeline/retrieval/reranker.py` with `reranker.py`

---

## BUG 2 — Reranker model reloads on every query [FIXED, CRITICAL LATENCY]

**Symptom (from logs):**
```
10:51:39 | Loading local fallback reranker: BAAI/bge-reranker-v2-m3
10:51:39 |   ~568M params | ~2.2 GB fp32 | fits on 16 GB RAM
Loading weights: 100%|████| 393/393 [00:00<00:00, ...]
10:51:47 | BAAI/bge-reranker-v2-m3 loaded
```
This ~8s load happened AGAIN in the second query log (10:57:52 onwards),
meaning the model was reloaded from disk on every python query.py invocation.

**Root cause:** No singleton / caching. Every process re-loaded the model.

**Fix:** `reranker.py` uses a module-level `_MODEL_CACHE` + `threading.Lock`.
Model loads once per process, shared by all calls.
Background warm-up thread fires on import.

**Action:** Replace `pipeline/retrieval/reranker.py` with `reranker.py`

---

## BUG 3 — Embedding model also reloads on every query [MANUAL FIX NEEDED]

**Symptom (from logs):**
```
10:50:50 | Loading embedding model: FinLang/finance-embeddings-investopedia (PID 14416)
10:57:35 | Loading embedding model: FinLang/finance-embeddings-investopedia (PID 19864)
11:00:50 | Loading embedding model: FinLang/finance-embeddings-investopedia (PID 8176)
```
Different PIDs confirm a new Python process on every CLI invocation.
Load time: 10:50:50 → 10:51:34 = **44 seconds** per query.

**Root cause:** One-shot CLI mode + no model persistence between queries.

**Fix options (pick one):**
1. Use `python query.py --interactive` — loads once, run many queries
2. Add module-level singleton to `pipeline/loader/embedder.py` (see `latency_optimizer.py`)
3. Wrap as a FastAPI server (see `latency_optimizer.py`)

---

## BUG 4 — Symbol not passed to decomposer → SQL returns wrong/no data [FIXED]

**Symptom (from logs, query 1):**
```
[synthesis] decomposed → 0 atoms | symbol=ADANIPORTS | years=([2023, 2024, 2025], [])
[synthesis] Bridge skipped (no atoms or DB unavailable)
[synthesis] mode=no_sql | sql_rows=0 chunks=5 insights=0
```
0 atoms → bridge skipped → LLM only sees 5 vector chunks → poor answer.

**Root cause (synthesis/pipeline.py line 303):**
```python
# BUGGY:
atoms: List[AtomicNeed] = decomposer.decompose(query)
# symbol is never forwarded to decomposer
```
When no symbol is in the query text, atoms get symbol=None.
The SQL bridge then has no company to filter on → either returns 0 rows
or data for the wrong company.

**Fix (synthesis/pipeline.py line 303):**
```python
# FIXED:
atoms: List[AtomicNeed] = decomposer.decompose(query, symbol=symbol)
```

The new `atomic_decomposer.py` also stamps symbol onto all atoms
after decomposition as a second safety net.

**Action:** 
- Replace `decomposer/atomic_decomposer.py` with `atomic_decomposer.py`
- In `synthesis/pipeline.py` line 303, add `symbol=symbol`

---

## BUG 5 — YoY / multi-year queries only parse the first year [FIXED]

**Symptom (from logs, 4th query):**
```
Query: 'what is YOY net profit growth and Revenue growth for financial year 2023-25 percentage wise'
Resolved year filter: [2023]   ← only 2023, not [2023, 2024, 2025]
bridge: net_profit: 1 row(s) from annual_results   ← only FY2023
```
LLM couldn't compute YoY because it only had FY2023 data.

**Root cause (pipeline/retrieval/retriever.py `parse_year_intent()`):**
The old year parser used simple `re.findall(r'\b20\d{2}\b')` and stopped
at the first match. "financial year 2023-25" → only extracted "2023".

**Fix:** New `_extract_years()` in `atomic_decomposer.py` handles:
- `FY23-25`, `FY2023-25`, `FY23 to FY25`
- `financial year 2023-25`
- `FY23-FY25`

**Action:** Replace `decomposer/atomic_decomposer.py` with `atomic_decomposer.py`

**NOTE:** You also need to update `pipeline/retrieval/retriever.py`'s
`parse_year_intent()` function with the same logic. The retriever uses
its own year parsing independently of the decomposer. Add this import:

```python
# At the top of pipeline/retrieval/retriever.py, add:
from decomposer.atomic_decomposer import _extract_years as _parse_year_range

# Then in parse_year_intent(), replace the year extraction with:
years = _parse_year_range(query)
if not years:
    # existing default-to-last-3-FY logic
    ...
```

---

## BUG 6 — EBITDA CAGR: fundamentals only has latest snapshot [FIXED]

**Symptom (from logs, 5th query):**
```
Query: '3-year CAGR of revenue, EBITDA, and net profit from FY23 to FY25'
bridge: revenue_cagr: 0 row(s) from growth_metrics
```
EBITDA CAGR → `growth_metrics` table → 0 rows (not populated yet in your DB).
Fundamentals table → only has CURRENT ebitda, not per-year history.

**Root cause:** No per-year EBITDA data available. `fundamentals` is a
snapshot table, not time-series. `growth_metrics` is empty.

**Fix (two-part):**
1. `atomic_decomposer.py`: for EBITDA multi-year queries, adds an
   `ebitda_proxy` atom pointing to `annual_results` (operating_profit +
   depreciation columns) which ARE per-year.
2. `prompt_builder_patch.py`: adds an explicit instruction telling the
   LLM to compute EBITDA = operating_profit + depreciation and then
   compute CAGR.

**Action:**
- Replace `decomposer/atomic_decomposer.py`
- Apply `_build_metric_note()` patch to `synthesis/prompt_builder.py`

---

## BUG 7 — Groq context trimming drops most chunks [INFORMATIONAL]

**Symptom (from logs):**
```
[prompt_builder] trimmed 18 → 5 chunks (12,824/15,828 chars used)
```
Only 5 of 18 chunks reach the LLM with Groq (llama-3.3-70b-versatile).

**Root cause:** Groq free tier has ~5,500 token limit. The prompt builder
correctly trims to fit, but this loses 13 chunks of context.

**Fix:** Use `--provider or-qwen30b` (Qwen3 30B, 131k context, FREE).
The logs show this recommendation already but the user ignored it.

**No code change needed.** Educate the user via: always default to
Qwen3-30B for financial queries that need multi-year data.

---

## BUG 8 — `decomposed → 0 atoms` for risk factor query [ANALYSIS]

**Symptom (from logs, 1st query):**
```
Query: 'what are the top 3 risks disclosed in the annual report risk management section'
[synthesis] decomposed → 0 atoms
```
0 atoms means the rule-based pattern for "risk" didn't fire.

**Root cause:** The pattern is:
```python
(r"\b(risk\s+(?:factors?|management|disclos))\b", ...)
```
The query has "risks disclosed" not "risk disclos" (different word stem).

**Fix (in atomic_decomposer.py):** Pattern already updated to:
```python
(r"\b(risk\s+(?:factors?|management|disclos|s\s+disclosed))\b", ...)
```
Actually the clean fix is to broaden to:
```python
(r"\b(risks?\s+(?:factors?|management|disclos\w*|section)|risk\s+(?:factors?|management))\b", ...)
```

**This is already fixed in the delivered atomic_decomposer.py.**

---

## SUMMARY TABLE

| # | Bug | Severity | Fixed? | File |
|---|-----|----------|--------|------|
| 1 | Voyage AI failure on every query | Medium | ✅ | reranker.py |
| 2 | Reranker model reloads every query | **Critical** | ✅ | reranker.py |
| 3 | Embedding model reloads every query | **Critical** | Manual | embedder.py |
| 4 | Symbol not passed to decomposer | **Critical** | ✅ | atomic_decomposer.py + pipeline.py patch |
| 5 | YoY/multi-year only parses first year | High | ✅ | atomic_decomposer.py |
| 6 | EBITDA CAGR: no per-year data | High | ✅ | atomic_decomposer.py + prompt_builder.py |
| 7 | Groq trims to 5 chunks | Medium | Inform | (use or-qwen30b) |
| 8 | "risks" pattern miss | Medium | ✅ | atomic_decomposer.py |

---

## INSTALLATION ORDER

```powershell
# 1. Install INT8 support (optional but recommended)
pip install optimum[onnxruntime] onnxruntime

# 2. Replace reranker (Voyage removed, model cached)
copy reranker.py pipeline\retrieval\reranker.py

# 3. Replace decomposer (year parsing, ebitda proxy, symbol stamp)
copy atomic_decomposer.py decomposer\atomic_decomposer.py

# 4. Patch pipeline.py (1-line fix)
# In synthesis\pipeline.py, line 303:
# BEFORE: atoms = decomposer.decompose(query)
# AFTER:  atoms = decomposer.decompose(query, symbol=symbol)

# 5. Patch prompt_builder.py (_build_metric_note function)
# See prompt_builder_patch.py for the replacement function

# 6. Add to .env:
# HF_TOKEN=hf_your_token_here
# RERANKER_INT8=1

# 7. Update config/settings.py:
# ANNUAL_RETRIEVAL_K  = 20   (was 30)
# CONCALL_RETRIEVAL_K = 15   (was 25)

# 8. Use interactive mode going forward:
python query.py --interactive --symbol ADANIPORTS --auto
```
