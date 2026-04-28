"""
rag/rag_engine.py

FIXES & ADDITIONS in this version:

  [FIX I] Root cause of repeated 413 on Groq llama-3.3-70b:
          Groq's FREE tier hard-caps input to ~6k tokens regardless of the
          model's advertised 128k window. Previous budget was 26k — way too
          high. Now set to 5500 tokens. Also tightened _CHARS_PER_TOKEN from
          4.0 → 3.5 (financial prose is denser) and payload char ceiling from
          160k → 130k.

  [NEW-QWEN] Qwen models via OpenRouter — FREE, 131k context window.
          These solve the context problem entirely: no trimming needed.
          Slugs: qwen/qwen3-30b-a3b:free, qwen/qwen-2.5-72b-instruct:free,
                 qwen/qwen3-8b:free

  [NEW-OLLAMA] Local LLM via Ollama — zero API calls, zero rate limits.
          Autodiscovers installed models via GET /api/tags.
          Your installed models: llama3.1:latest (4.9 GB), phi3:latest (2.2 GB)
          Uses Ollama's OpenAI-compatible endpoint /v1/chat/completions.

  [NEW-PICKER] User-selectable provider via --provider CLI flag OR interactive
          numbered menu. No more silent auto-waterfall guessing. User sees all
          available providers with context-window notes and picks by number.
          --auto flag restores old waterfall behaviour for scripted use.

  Full provider catalogue (only shows entries with keys/services present):
    ID            Provider      Model                          Notes
    ──────────────────────────────────────────────────────────────────
    groq-llama    Groq          llama-3.3-70b-versatile        5.5k tok free
    or-qwen30b    OpenRouter    qwen/qwen3-30b-a3b:free        131k, FREE
    or-qwen72b    OpenRouter    qwen/qwen-2.5-72b-instruct     131k, FREE
    or-qwen8b     OpenRouter    qwen/qwen3-8b:free             131k, FREE
    or-gemini     OpenRouter    google/gemini-2.0-flash-001    1M ctx
    gemini        Google        gemini-2.0-flash (direct)      1M, 15RPM
    nvidia        NVIDIA NIM    meta/llama-3.3-70b-instruct    128k
    ollama-*      Ollama local  (autodiscovered)               no rate limit
    groq-gemma    Groq          gemma2-9b-it                   3.2k, last resort

  .env keys:
    GROQ_API_KEY=gsk_...          https://console.groq.com/
    GEMINI_API_KEY=AIza...        https://aistudio.google.com/app/apikey
    OPENROUTER_API_KEY=sk-or-...  https://openrouter.ai/
    NVIDIA_API_KEY=nvapi-...      https://build.nvidia.com/
    OLLAMA_BASE_URL=http://localhost:11434   (optional, this is default)
"""

import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass

import requests

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
)
from pipeline.retrieval.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_API_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
GEMINI_API_URL     = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ─────────────────────────────────────────────
# Context window budgets (tokens)
# ─────────────────────────────────────────────
_MODEL_CTX: Dict[str, int] = {
    # Groq — free tier is hard-capped at ~6k input regardless of model window
    "llama-3.3-70b-versatile":           5_500,   # FIX I: was 26_000
    "gemma2-9b-it":                      3_200,   # 8k window, heavy system overhead
    "llama3-8b-8192":                    6_000,
    # Google Gemini (direct)
    "gemini-2.0-flash":                200_000,
    "gemini-1.5-flash":                200_000,
    "gemini-1.5-pro":                  200_000,
    # OpenRouter — Qwen (free, 131k context)
    "qwen/qwen3-30b-a3b:free":          30_000,
    "qwen/qwen3-8b:free":               30_000,
    "qwen/qwen-2.5-72b-instruct:free":  30_000,
    # OpenRouter — Gemini
    "google/gemini-2.0-flash-001":     200_000,
    "anthropic/claude-3-haiku":         50_000,
    # NVIDIA NIM
    "meta/llama-3.3-70b-instruct":      50_000,
    # Ollama default (conservative — actual limit is RAM-dependent)
    "_ollama_default":                 100_000,
}
_DEFAULT_CTX             = 12_000
_CHARS_PER_TOKEN: float  = 3.5        # FIX I: dense financial text
_GROQ_MAX_PAYLOAD_CHARS  = 130_000    # FIX I: tightened from 160k
_NO_SYSTEM_ROLE          = {"gemma2-9b-it", "gemma-7b-it"}


# ─────────────────────────────────────────────
# Provider entry dataclass
# ─────────────────────────────────────────────
@dataclass
class ProviderEntry:
    id:           str   # e.g. "ollama-phi3-latest"
    label:        str   # e.g. "Ollama local — phi3:latest"
    provider:     str   # routing key: groq | gemini | openrouter | nvidia | ollama
    model:        str   # exact model name/slug
    api_key:      str
    api_url:      str
    context_note: str = ""  # shown in picker menu


# ─────────────────────────────────────────────
# Ollama autodiscovery
# ─────────────────────────────────────────────
def _discover_ollama(base_url: str) -> List[ProviderEntry]:
    """Query /api/tags; return one entry per installed model. Never raises."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        entries = []
        for m in resp.json().get("models", []):
            name     = m.get("name", "")
            size_gb  = round(m.get("size", 0) / 1e9, 1)
            safe_id  = name.replace(":", "-").replace("/", "-")
            entries.append(ProviderEntry(
                id           = f"ollama-{safe_id}",
                label        = f"Ollama local — {name}",
                provider     = "ollama",
                model        = name,
                api_key      = "",
                api_url      = f"{base_url}/v1/chat/completions",
                context_note = f"{size_gb} GB, local",
            ))
        return entries
    except Exception:
        return []


# ─────────────────────────────────────────────
# Build provider catalogue
# ─────────────────────────────────────────────
def build_provider_catalogue() -> List[ProviderEntry]:
    """
    Return all available providers in recommended order.
    Entries are included only if their key / service exists.
    """
    cat: List[ProviderEntry] = []

    groq_key   = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    or_key     = os.getenv("OPENROUTER_API_KEY", "")
    nv_key     = os.getenv("NVIDIA_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── 1. Groq primary ───────────────────────────────────────────
    if groq_key:
        cat.append(ProviderEntry(
            id="groq-llama", label="Groq — llama-3.3-70b-versatile",
            provider="groq", model=GROQ_MODEL,
            api_key=groq_key, api_url=GROQ_API_URL,
            context_note="~5.5k tok (free cap)",
        ))

    # ── 2. OpenRouter — Qwen free (large context, zero cost) ──────
    if or_key:
        cat.extend([
            ProviderEntry(
                id="or-qwen30b", label="OpenRouter — Qwen3 30B MoE [FREE]",
                provider="openrouter", model="qwen/qwen3-30b-a3b:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),
            ProviderEntry(
                id="or-qwen72b", label="OpenRouter — Qwen2.5 72B Instruct [FREE]",
                provider="openrouter", model="qwen/qwen-2.5-72b-instruct:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),
            ProviderEntry(
                id="or-qwen8b", label="OpenRouter — Qwen3 8B [FREE]",
                provider="openrouter", model="qwen/qwen3-8b:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),
            ProviderEntry(
                id="or-gemini", label="OpenRouter — Gemini 2.0 Flash",
                provider="openrouter", model="google/gemini-2.0-flash-001",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="1M ctx",
            ),
        ])

    # ── 3. Google Gemini direct ───────────────────────────────────
    if gemini_key:
        cat.append(ProviderEntry(
            id="gemini", label="Google Gemini — gemini-2.0-flash (direct API)",
            provider="gemini", model="gemini-2.0-flash",
            api_key=gemini_key, api_url=GEMINI_API_URL,
            context_note="1M ctx, 15 RPM free",
        ))

    # ── 4. NVIDIA NIM ─────────────────────────────────────────────
    if nv_key:
        cat.append(ProviderEntry(
            id="nvidia", label="NVIDIA NIM — llama-3.3-70b-instruct",
            provider="nvidia", model="meta/llama-3.3-70b-instruct",
            api_key=nv_key, api_url=NVIDIA_API_URL,
            context_note="128k ctx",
        ))

    # ── 5. Ollama local (autodiscovered) ──────────────────────────
    cat.extend(_discover_ollama(ollama_url))

    # ── 6. Groq gemma2 — absolute last resort ─────────────────────
    if groq_key:
        cat.append(ProviderEntry(
            id="groq-gemma", label="Groq — gemma2-9b-it [last resort]",
            provider="groq", model=GROQ_FALLBACK_MODEL,
            api_key=groq_key, api_url=GROQ_API_URL,
            context_note="3.2k tok, tiny ctx",
        ))

    return cat


# ─────────────────────────────────────────────
# Interactive provider picker
# ─────────────────────────────────────────────
def pick_provider_interactive(catalogue: List[ProviderEntry]) -> ProviderEntry:
    """Print a numbered menu and block until the user picks one."""
    W_LABEL = 46
    W_NOTE  = 18
    border  = "─" * (W_LABEL + W_NOTE + 10)

    print(f"\n┌{border}┐")
    print(f"│  🤖  Select LLM Provider{' ' * (len(border) - 24)}│")
    print(f"├────┬{'─' * W_LABEL}┬{'─' * W_NOTE}┤")
    print(f"│ #  │ {'Provider / Model'.ljust(W_LABEL - 1)}│ {'Context / Notes'.ljust(W_NOTE - 1)}│")
    print(f"├────┼{'─' * W_LABEL}┼{'─' * W_NOTE}┤")
    for i, e in enumerate(catalogue, 1):
        label = e.label[:W_LABEL - 1].ljust(W_LABEL - 1)
        note  = e.context_note[:W_NOTE - 1].ljust(W_NOTE - 1)
        print(f"│ {str(i).ljust(2)} │ {label}│ {note}│")
    print(f"└────┴{'─' * W_LABEL}┴{'─' * W_NOTE}┘")
    print()

    while True:
        raw = input(f"  Enter number [1-{len(catalogue)}] (or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= len(catalogue):
            chosen = catalogue[int(raw) - 1]
            print(f"  ✔  Using: {chosen.label}\n")
            return chosen
        print(f"  ⚠  Enter a number between 1 and {len(catalogue)}")


def get_provider(
    catalogue:   List[ProviderEntry],
    provider_id: Optional[str],
    auto:        bool,
) -> List[ProviderEntry]:
    """
    Returns the list of entries to try.
      provider_id set → exactly that one entry
      auto=True       → all entries in order (waterfall, no menu)
      otherwise       → show picker, return the single chosen entry
    """
    if not catalogue:
        raise ValueError(
            "No LLM providers available.\n\n"
            "Set at least one in your .env:\n"
            "  GROQ_API_KEY=gsk_...          https://console.groq.com/\n"
            "  OPENROUTER_API_KEY=sk-or-...  https://openrouter.ai/\n"
            "    (Qwen3-30B is FREE on OpenRouter, 131k context — best option)\n"
            "  GEMINI_API_KEY=AIza...        https://aistudio.google.com/app/apikey\n"
            "  NVIDIA_API_KEY=nvapi-...      https://build.nvidia.com/\n"
            "\nOr use Ollama locally (no key needed):\n"
            "  ollama serve\n"
            "  ollama pull qwen2.5:7b\n"
        )

    if provider_id:
        matches = [e for e in catalogue if e.id == provider_id]
        if not matches:
            valid = ", ".join(e.id for e in catalogue)
            raise ValueError(
                f"Provider '{provider_id}' not found.\n"
                f"Available IDs: {valid}"
            )
        return matches

    if auto:
        return catalogue   # try all in order

    # Interactive
    chosen = pick_provider_interactive(catalogue)
    return [chosen]


# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────
ANNUAL_SYSTEM_PROMPT = """\
You are a senior equity research analyst at a top-tier investment firm, \
specialising in Indian listed companies (BSE/NSE).

YOUR RULES — follow every one strictly:
1. RECENCY FIRST: Always prefer and prominently feature data from the most recent \
fiscal year available in the context. Never lead with old data.
2. USE ONLY CONTEXT: Do not use prior knowledge. If a number is not in the \
provided excerpts, say explicitly: "Not available in provided documents."
3. SHOW YOUR MATH: For any growth/trend calculation, write the formula and numbers. \
Example: Revenue growth FY24 = (19,500 - 16,200) / 16,200 = +20.4%
4. CURRENCY: State amounts exactly as shown in source (Crore / Lakh / Million). \
Never convert unless asked.
5. STRUCTURED OUTPUT: For multi-year comparisons, use a table. For single-year \
analysis, use bullet points. For qualitative topics, use paragraphs.
6. CITE EVERY NUMBER: After each data point write [FY<year>, Page <n>].
7. FLAG GAPS: If data for a specific year is missing from context, write: \
"⚠ FY<year>: data not in retrieved excerpts."
8. NO HALLUCINATION: If you are not certain, say so. Never guess a number.\
"""

CONCALL_SYSTEM_PROMPT = """\
You are a senior buy-side equity analyst reviewing earnings call transcripts \
for an Indian listed company.

YOUR RULES:
1. RECENCY FIRST: Lead with the most recent concall available. State the date/quarter.
2. QUOTE ACCURATELY: When citing management, use exact words and name the speaker \
and their role. Format: "CFO [Name]: '...'"
3. SEPARATE MGMT vs ANALYST: Clearly distinguish management commentary from \
analyst questions and pushback.
4. KEY THEMES: Extract: guidance, risks mentioned, capex plans, margin commentary, \
volume/revenue targets.
5. FLAG GAPS: If the query topic was not discussed in the transcript, say so clearly.
6. USE ONLY CONTEXT: Do not use prior knowledge about the company.\
"""

COMBINED_SYSTEM_PROMPT = """\
You are a senior equity research analyst with access to both annual reports \
and earnings call transcripts for an Indian listed company.

YOUR RULES:
1. RECENCY FIRST: Always lead with the most recent data available (prefer FY2025 > \
FY2024 > FY2023 > older). State the FY prominently.
2. CROSS-SOURCE SYNTHESIS: When annual report numbers are confirmed or expanded \
by concall commentary, show both. Label each: [Annual Report] or [Concall].
3. SHOW MATH: For any YOY growth, show: (new - old) / old × 100.
4. TABLES FOR TRENDS: Multi-year comparisons MUST be in a table with columns: \
FY | Metric | Value | YOY%
5. CITE SOURCES: After each data point: [FY<year>, AR/CC, Page <n>].
6. FLAG MISSING DATA: "⚠ FY<year>: not in retrieved excerpts." for each gap.
7. NO HALLUCINATION: If a number is not in context, do not estimate or extrapolate.
8. HONEST ABOUT LIMITS: If the context is insufficient to answer fully, \
say what IS available and what is MISSING.\
"""


# ─────────────────────────────────────────────
# Token / size helpers
# ─────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ─────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────
def _build_context(chunks: List[RetrievedChunk]) -> str:
    def sort_key(c):
        yr  = c.metadata.get("year", 0)
        typ = 0 if c.metadata.get("doc_type") == "annual_report" else 1
        return (-yr, typ)

    parts     = []
    separator = "\n\n" + "─" * 60 + "\n\n"
    for i, chunk in enumerate(sorted(chunks, key=sort_key), 1):
        meta      = chunk.metadata
        doc_label = "Annual Report" if meta.get("doc_type") == "annual_report" else "Concall Transcript"
        section   = (meta.get("section") or meta.get("speaker") or "")[:60]
        tag = (
            f"[Source {i} | {meta.get('symbol','')} | {doc_label} | "
            f"FY{meta.get('year','')} | {section} | Page {meta.get('page_start','')}]"
        )
        parts.append(f"{tag}\n{chunk.text.strip()}")
    return separator.join(parts)


def _build_user_prompt(query: str, context: str, doc_type: str,
                       years: Optional[List[int]] = None) -> str:
    year_note = ""
    if years:
        year_note = (
            f"\n\nIMPORTANT: The user is asking about FY{'/'.join(str(y) for y in years)}. "
            f"Prioritise data from these years. Flag missing years with ⚠."
        )
    return (
        f"CONTEXT FROM FINANCIAL DOCUMENTS (most recent first):\n"
        f"{'=' * 60}\n{context}\n{'=' * 60}"
        f"{year_note}\n\n"
        f"QUESTION: {query}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer using ONLY the context above.\n"
        f"- Lead with the most recent year's data.\n"
        f"- Show calculations explicitly for any growth/trend figures.\n"
        f"- Use a table if comparing across multiple years.\n"
        f"- Flag any years where data is missing from context.\n"
    )


# ─────────────────────────────────────────────
# Context trimmer
# ─────────────────────────────────────────────
def _trim_chunks_to_budget(
    chunks:        List[RetrievedChunk],
    system_prompt: str,
    query:         str,
    doc_type:      str,
    years:         Optional[List[int]],
    entry:         ProviderEntry,
) -> List[RetrievedChunk]:
    """
    Drop lowest-ranked chunks (from the tail) until the full prompt fits within:
      (a) the model's token budget, AND
      (b) Groq's hard payload char limit
    """
    model     = entry.model
    provider  = entry.provider
    is_merged = model in _NO_SYSTEM_ROLE

    # For Ollama, look up by the default key since model names vary
    ctx_budget   = _MODEL_CTX.get(model) or (
        _MODEL_CTX["_ollama_default"] if provider == "ollama" else _DEFAULT_CTX
    )
    token_budget     = ctx_budget - GROQ_MAX_TOKENS
    system_tokens    = _estimate_tokens(system_prompt)
    base_tokens      = _estimate_tokens(_build_user_prompt(query, "", doc_type, years))
    sep_overhead     = 150 if is_merged else 50
    available_tokens = token_budget - system_tokens - base_tokens - sep_overhead

    if available_tokens <= 0:
        log.warning(f"  Budget too tight for {model}, using top 3 chunks only")
        return chunks[:3]

    kept        = []
    used_tokens = 0
    used_chars  = len(system_prompt) + len(_build_user_prompt(query, "", doc_type, years))

    for chunk in chunks:
        chunk_tokens = _estimate_tokens(chunk.text) + 60
        chunk_chars  = len(chunk.text) + 250

        over_tok   = (used_tokens + chunk_tokens) > available_tokens
        over_chars = (provider == "groq") and \
                     (used_chars + chunk_chars) > _GROQ_MAX_PAYLOAD_CHARS

        if over_tok or over_chars:
            break

        kept.append(chunk)
        used_tokens += chunk_tokens
        used_chars  += chunk_chars

    if len(kept) < len(chunks):
        log.info(
            f"  Trimmed {len(chunks)} → {len(kept)} chunks "
            f"({model}: ~{used_tokens} tok, ~{used_chars} chars)"
        )
    return kept or chunks[:1]


# ─────────────────────────────────────────────
# Gemini native call  (generateContent schema)
# ─────────────────────────────────────────────
def _call_gemini(system_prompt: str, user_prompt: str,
                 model: str, api_key: str) -> dict:
    url  = GEMINI_API_URL.format(model=model)
    resp = requests.post(
        url,
        json={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature":     GROQ_TEMPERATURE,
                "maxOutputTokens": GROQ_MAX_TOKENS,
            },
        },
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response: {data}") from exc
    usage = data.get("usageMetadata", {})
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens":     usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens":      usage.get("totalTokenCount", 0),
        },
    }


# ─────────────────────────────────────────────
# OpenAI-compatible call  (Groq / OpenRouter / NVIDIA / Ollama)
# ─────────────────────────────────────────────
def _call_openai_compat(system_prompt: str, user_prompt: str,
                        entry: ProviderEntry) -> dict:
    merge   = entry.model in _NO_SYSTEM_ROLE
    headers = {"Content-Type": "application/json"}
    if entry.api_key:
        headers["Authorization"] = f"Bearer {entry.api_key}"
    if "openrouter" in entry.api_url:
        headers["HTTP-Referer"] = "https://github.com/Import-Saurabh/Financial-Rag"
        headers["X-Title"]      = "Financial RAG"

    messages = (
        [{"role": "user", "content": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
        if merge else
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": user_prompt}]
    )
    resp = requests.post(
        entry.api_url,
        json={
            "model":       entry.model,
            "messages":    messages,
            "max_tokens":  GROQ_MAX_TOKENS,
            "temperature": GROQ_TEMPERATURE,
        },
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Retry wrapper
# ─────────────────────────────────────────────
def _call_with_retry(system_prompt: str, user_prompt: str,
                     entry: ProviderEntry, max_retries: int = 3) -> dict:
    """Route to correct call fn; retry 429 with back-off; raise on 400/413."""
    delay    = 5
    last_exc = None

    for attempt in range(max_retries):
        try:
            if entry.provider == "gemini":
                return _call_gemini(system_prompt, user_prompt,
                                    entry.model, entry.api_key)
            return _call_openai_compat(system_prompt, user_prompt, entry)

        except requests.HTTPError as e:
            status   = e.response.status_code
            last_exc = e
            if status == 429 and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                log.warning(f"  429 on {entry.model} (attempt {attempt+1}), retry in {wait}s")
                time.sleep(wait)
            else:
                raise   # 400, 413, or retries exhausted — move to next provider

    raise last_exc  # pragma: no cover


# ─────────────────────────────────────────────
# RAGResponse
# ─────────────────────────────────────────────
@dataclass
class RAGResponse:
    answer:      str
    model_used:  str
    chunks_used: int
    sources:     List[dict]
    tokens_used: int
    latency_sec: float


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def generate_answer(
    query:       str,
    chunks:      List[RetrievedChunk],
    doc_type:    str = "annual_report",
    api_key:     Optional[str] = None,   # legacy compat param, unused
    years:       Optional[List[int]] = None,
    provider_id: Optional[str] = None,   # pin to specific provider ID
    auto:        bool = False,           # True = silent waterfall, no menu
) -> RAGResponse:
    """
    provider_id: pin a provider by its short ID (e.g. "ollama-phi3-latest",
                 "or-qwen30b", "groq-llama"). Use --provider on the CLI.
    auto:        waterfall through all providers without showing the menu.
                 Use --auto on the CLI (or for scripted/test runs).
    """
    if not chunks:
        return RAGResponse(
            answer=(
                "No relevant documents found for this query.\n"
                "Possible reasons:\n"
                "  • Documents for the requested years not ingested\n"
                "  • Run: python ingest.py --symbol <SYMBOL>\n"
                "  • Try --year-range to broaden the search"
            ),
            model_used="none", chunks_used=0,
            sources=[], tokens_used=0, latency_sec=0.0,
        )

    system    = {"annual_report": ANNUAL_SYSTEM_PROMPT, "concall": CONCALL_SYSTEM_PROMPT}.get(
        doc_type, COMBINED_SYSTEM_PROMPT
    )
    catalogue = build_provider_catalogue()
    entries   = get_provider(catalogue, provider_id, auto)

    t0         = time.time()
    last_error = None

    for entry in entries:
        log.info(f"  → {entry.label}")

        safe_chunks = _trim_chunks_to_budget(chunks, system, query, doc_type, years, entry)
        context     = _build_context(safe_chunks)
        user_prompt = _build_user_prompt(query, context, doc_type, years)

        try:
            result  = _call_with_retry(system, user_prompt, entry)
            latency = time.time() - t0
            answer  = result["choices"][0]["message"]["content"]
            usage   = result.get("usage", {})

            log.info(
                f"  LLM ✓ [{entry.label}]: "
                f"{usage.get('completion_tokens', 0)} tokens | "
                f"{latency:.1f}s | {len(safe_chunks)}/{len(chunks)} chunks"
            )
            return RAGResponse(
                answer      = answer,
                model_used  = entry.label,
                chunks_used = len(safe_chunks),
                sources     = [
                    {
                        "symbol":   c.metadata.get("symbol"),
                        "year":     c.metadata.get("year"),
                        "doc_type": c.metadata.get("doc_type"),
                        "section":  (c.metadata.get("section") or c.metadata.get("speaker", ""))[:50],
                        "page":     c.metadata.get("page_start"),
                        "score":    round(c.score, 4),
                    }
                    for c in chunks
                ],
                tokens_used = usage.get("total_tokens", 0),
                latency_sec = round(latency, 2),
            )

        except (requests.HTTPError, Exception) as e:
            status   = getattr(getattr(e, "response", None), "status_code", "ERR")
            is_last  = (entry == entries[-1])
            log.warning(
                f"  {entry.label} → HTTP {status} | "
                f"{'all providers exhausted' if is_last else 'trying next'}"
            )
            last_error = e

    raise RuntimeError(
        f"All selected providers failed. Last error: {last_error}\n\n"
        "Quick fixes:\n"
        "  • Best option: add OPENROUTER_API_KEY and use Qwen3-30B (free, 131k ctx)\n"
        "      https://openrouter.ai/ → pick 'or-qwen30b' from the menu\n"
        "  • Local Ollama (no key): ollama serve && ollama pull qwen2.5:7b\n"
        "  • Groq free tier: wait ~60s then retry\n"
        "  • Gemini: GEMINI_API_KEY → https://aistudio.google.com/app/apikey\n"
    )