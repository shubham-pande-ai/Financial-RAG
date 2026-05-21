"""
rag/rag_engine.py  — patched to use SynthesisPipeline (Layer 5)

What changed from the previous version
────────────────────────────────────────
[SYNTHESIS] generate_answer() now creates a SynthesisPipeline and calls
            .run() *after* chunk trimming.  The pipeline attempts:

              1. Atomic decomposer   → typed AtomicNeed list
              2. Schema bridge       → SQL rows + vector chunks (parallel)
              3. Fusion layer        → metric table + contradiction insights
              4. PromptBuilder       → structured (system, user) pair
                                       with SQL table, insight callouts,
                                       and [SRC-N] / [SQL-N] citation anchors

            If any step fails (missing DB, import error, network) the
            pipeline degrades gracefully to the vector-only prompt path
            that was already in place — so all existing queries keep working.

[FREE MODELS] No hard-coded Claude/GPT.  Provider catalogue (Groq, OpenRouter
              Qwen, Gemini, NVIDIA NIM, Ollama) is unchanged.  The prompt
              builder sizes itself to the chosen model automatically.

Everything else (provider picker, retry logic, Gemini native call, context
trimmer, RAGResponse) is exactly as before.
"""

import os
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

import requests

from config.settings import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
)
from pipeline.retrieval.retriever import RetrievedChunk
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Validation result cache  (populated by validate_provider / validate_all)
# ─────────────────────────────────────────────────────────────────────────────
_HEALTHY_PROVIDERS: List["ProviderEntry"] = []
_VALIDATION_TS: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_URL       = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_API_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
GEMINI_API_URL     = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ─────────────────────────────────────────────────────────────────────────────
# Context window budgets (tokens)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CTX: Dict[str, int] = {
    # ── Groq ─────────────────────────────────────────────────────────────────
    "llama-3.3-70b-versatile":                    5_500,
    "llama-3.1-8b-instant":                       6_000,
    "llama3-8b-8192":                             6_000,
    "gemma2-9b-it":                               3_200,   # kept for compat only
    # ── Gemini (direct) ───────────────────────────────────────────────────────
    "gemini-2.0-flash":                         200_000,
    "gemini-2.0-flash-lite":                    200_000,
    "gemini-1.5-flash":                         200_000,
    "gemini-1.5-pro":                           200_000,
    # ── OpenRouter — Qwen3 (current, no :free needed for catalogue) ───────────
    "qwen/qwen3-30b-a3b":                        30_000,
    "qwen/qwen3-30b-a3b:free":                   30_000,   # compat
    "qwen/qwen3-8b":                             30_000,
    "qwen/qwen3-8b:free":                        30_000,   # compat
    "qwen/qwen2.5-72b-instruct":                 30_000,
    # ── OpenRouter — Gemini ────────────────────────────────────────────────────
    "google/gemini-2.0-flash-001":              200_000,   # fixed slug
    "google/gemini-2.5-flash-preview":          200_000,
    # ── OpenRouter — Meta Llama (free) ────────────────────────────────────────
    "meta-llama/llama-3.3-70b-instruct:free":    30_000,
    "meta-llama/llama-3.1-8b-instruct:free":     30_000,
    # ── OpenRouter — Mistral (free) ───────────────────────────────────────────
    "mistralai/mistral-7b-instruct:free":        30_000,
    # ── OpenRouter — DeepSeek (free) ─────────────────────────────────────────
    "deepseek/deepseek-r1:free":                 30_000,
    "deepseek/deepseek-chat-v3-0324:free":       30_000,
    # ── OpenRouter — Claude via OR (paid — needs key with credits) ────────────
    "anthropic/claude-3-haiku":                  50_000,
    "anthropic/claude-3.5-haiku":               100_000,
    # ── NVIDIA NIM ────────────────────────────────────────────────────────────
    "meta/llama-3.3-70b-instruct":               50_000,
    # ── Ollama fallback ───────────────────────────────────────────────────────
    "_ollama_default":                          100_000,
}
_DEFAULT_CTX             = 12_000
_CHARS_PER_TOKEN: float  = 3.5
_GROQ_MAX_PAYLOAD_CHARS  = 130_000
_NO_SYSTEM_ROLE          = {"gemma2-9b-it", "gemma-7b-it"}
_LARGE_CONTEXT_THRESHOLD = 20_000


# ─────────────────────────────────────────────────────────────────────────────
# Provider entry dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ProviderEntry:
    id:           str
    label:        str
    provider:     str
    model:        str
    api_key:      str
    api_url:      str
    context_note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Ollama autodiscovery
# ─────────────────────────────────────────────────────────────────────────────
def _discover_ollama(base_url: str) -> List[ProviderEntry]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        entries = []
        for m in resp.json().get("models", []):
            name    = m.get("name", "")
            size_gb = round(m.get("size", 0) / 1e9, 1)
            safe_id = name.replace(":", "-").replace("/", "-")
            entries.append(ProviderEntry(
                id=f"ollama-{safe_id}", label=f"Ollama local — {name}",
                provider="ollama", model=name, api_key="",
                api_url=f"{base_url}/v1/chat/completions",
                context_note=f"{size_gb} GB, local",
            ))
        return entries
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Provider catalogue
# ─────────────────────────────────────────────────────────────────────────────
def build_provider_catalogue() -> List[ProviderEntry]:
    """
    Build the full catalogue of configured providers.

    Model slug correctness (as of 2025-06):
      - OpenRouter free models use   model_slug:free   OR no suffix (both work
        if the model has a free tier).  We use the explicit :free suffix only
        when required by OR to avoid routing to paid variants.
      - Dead slugs removed:
          google/gemini-2.0-flash-exp:free  → google/gemini-2.0-flash-001
          qwen/qwen2.5-72b-instruct:free   → qwen/qwen2.5-72b-instruct  (no :free)
          gemma2-9b-it (Groq)              → replaced by llama-3.1-8b-instant
    """
    cat: List[ProviderEntry] = []

    groq_key      = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    gemini_key    = os.getenv("GEMINI_API_KEY", "")
    or_key        = os.getenv("OPENROUTER_API_KEY", "")
    nv_key        = os.getenv("NVIDIA_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    ollama_url    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Groq (fast, free tier, limited context) ───────────────────────────────
    if groq_key:
        cat.append(ProviderEntry(
            id="groq-llama", label="Groq — llama-3.3-70b-versatile ★ FASTEST",
            provider="groq", model=GROQ_MODEL,
            api_key=groq_key, api_url=GROQ_API_URL,
            context_note="~5.5k tok (free cap)",
        ))
        cat.append(ProviderEntry(
            id="groq-llama-8b", label="Groq — llama-3.1-8b-instant (fast fallback)",
            provider="groq", model="llama-3.1-8b-instant",
            api_key=groq_key, api_url=GROQ_API_URL,
            context_note="6k tok, fastest",
        ))

    # ── OpenRouter free-tier models (verified working slugs) ─────────────────
        # ── OpenRouter free-tier models (verified working slugs as of 2026-05) ──
    if or_key:
        cat.extend([
            ProviderEntry(
                id="or-gemini", label="OpenRouter — Gemini 2.0 Flash [FREE]",
                provider="openrouter", model="google/gemini-2.0-flash-001",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="1M ctx, FREE",
            ),
            ProviderEntry(
                id="or-llama70b", label="OpenRouter — Llama 3.3 70B Instruct [FREE]",
                provider="openrouter", model="meta-llama/llama-3.3-70b-instruct:free",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="131k ctx, FREE",
            ),

        ])
        # Claude via OpenRouter (paid)
        cat.extend([
            ProviderEntry(
                id="or-claude-haiku", label="OpenRouter — Claude 3.5 Haiku (via OR, paid)",
                provider="openrouter", model="anthropic/claude-3.5-haiku",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="100k ctx, paid",
            ),
            ProviderEntry(
                id="or-claude-haiku-3", label="OpenRouter — Claude 3 Haiku (via OR, paid)",
                provider="openrouter", model="anthropic/claude-3-haiku",
                api_key=or_key, api_url=OPENROUTER_API_URL,
                context_note="50k ctx, paid",
            ),
        ])

    # ── Claude direct API (Anthropic) ─────────────────────────────────────────
    if anthropic_key:
        cat.extend([
            ProviderEntry(
                id="claude-haiku", label="Claude 3.5 Haiku (Anthropic direct)",
                provider="anthropic", model="claude-3-5-haiku-20241022",
                api_key=anthropic_key, api_url="https://api.anthropic.com/v1/messages",
                context_note="200k ctx, fast",
            ),
            ProviderEntry(
                id="claude-sonnet", label="Claude 3.5 Sonnet (Anthropic direct)",
                provider="anthropic", model="claude-3-5-sonnet-20241022",
                api_key=anthropic_key, api_url="https://api.anthropic.com/v1/messages",
                context_note="200k ctx, best",
            ),
        ])

    # ── Gemini direct API ─────────────────────────────────────────────────────
    if gemini_key:
        cat.append(ProviderEntry(
            id="gemini", label="Google Gemini — gemini-2.0-flash (direct API)",
            provider="gemini", model="gemini-2.0-flash",
            api_key=gemini_key, api_url=GEMINI_API_URL,
            context_note="1M ctx, 15 RPM free",
        ))
        cat.append(ProviderEntry(
            id="gemini-lite", label="Google Gemini — gemini-2.0-flash-lite (direct API)",
            provider="gemini", model="gemini-2.0-flash-lite",
            api_key=gemini_key, api_url=GEMINI_API_URL,
            context_note="1M ctx, faster",
        ))

    # ── NVIDIA NIM (slow — warn users) ───────────────────────────────────────
    if nv_key:
        cat.append(ProviderEntry(
            id="nvidia", label="NVIDIA NIM — llama-3.3-70b-instruct ⚠ SLOW (~90s)",
            provider="nvidia", model="meta/llama-3.3-70b-instruct",
            api_key=nv_key, api_url=NVIDIA_API_URL,
            context_note="128k ctx, slow",
        ))

    # ── Local Ollama (autodiscovered) ─────────────────────────────────────────
    cat.extend(_discover_ollama(ollama_url))

    return cat


# ─────────────────────────────────────────────────────────────────────────────
# Claude direct API call  (Anthropic Messages API — not OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
def _call_anthropic(system_prompt: str, user_prompt: str,
                    entry: "ProviderEntry") -> dict:
    """Call Anthropic Messages API directly."""
    resp = requests.post(
        entry.api_url,
        json={
            "model":      entry.model,
            "max_tokens": GROQ_MAX_TOKENS,
            "system":     system_prompt,
            "messages":   [{"role": "user", "content": user_prompt}],
        },
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         entry.api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["content"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Anthropic response: {data}") from exc
    usage = data.get("usage", {})
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens":     usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens":      usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Provider validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_provider(entry: "ProviderEntry") -> Tuple[bool, str]:
    """
    Lightweight health-check for a single provider.

    Strategy per provider type:
      • groq / openrouter / nvidia  → GET /models (cheap list endpoint)
      • gemini (direct)             → GET /v1beta/models?key=...
      • anthropic (direct)          → GET /v1/models
      • ollama                      → GET /api/tags (already used in discovery)

    Returns (is_healthy: bool, error_message: str).
    """
    try:
        if entry.provider == "ollama":
            base = entry.api_url.replace("/v1/chat/completions", "")
            r = requests.get(f"{base}/api/tags", timeout=5)
            r.raise_for_status()
            return True, ""

        if entry.provider == "gemini":
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            r = requests.get(url, params={"key": entry.api_key}, timeout=8)
            r.raise_for_status()
            # Verify the specific model slug exists in the response
            names = [m.get("name", "") for m in r.json().get("models", [])]
            model_path = f"models/{entry.model}"
            if names and not any(entry.model in n for n in names):
                return False, f"Model '{entry.model}' not found in Gemini catalogue"
            return True, ""

        if entry.provider == "anthropic":
            r = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": entry.api_key, "anthropic-version": "2023-06-01"},
                timeout=8,
            )
            if r.status_code == 401:
                return False, "Invalid Anthropic API key (401)"
            r.raise_for_status()
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            if ids and entry.model not in ids:
                return False, f"Model '{entry.model}' not in Anthropic catalogue"
            return True, ""

        # groq / openrouter / nvidia — all expose /models
        if entry.provider == "groq":
            models_url = "https://api.groq.com/openai/v1/models"
        elif entry.provider == "openrouter":
            models_url = "https://openrouter.ai/api/v1/models"
        elif entry.provider == "nvidia":
            models_url = "https://integrate.api.nvidia.com/v1/models"
        else:
            # Unknown provider — attempt a tiny completion as a last resort
            return _validate_via_completion(entry)

        headers = {"Authorization": f"Bearer {entry.api_key}"}
        if entry.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Import-Saurabh/FinancialRag"
            headers["X-Title"]      = "FinancialRAG"

        r = requests.get(models_url, headers=headers, timeout=10)
        if r.status_code in (401, 403):
            return False, f"Auth error ({r.status_code}) — check API key"
        r.raise_for_status()

        # For OpenRouter, verify the specific free model is available
        if entry.provider == "openrouter":
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            if ids and entry.model not in ids:
                return False, f"Model '{entry.model}' not listed on OpenRouter"

        return True, ""

    except requests.exceptions.Timeout:
        return False, "Timed out during validation"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}"
    except requests.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def _validate_via_completion(entry: "ProviderEntry") -> Tuple[bool, str]:
    """Fallback: send a 1-token completion to check the provider is alive."""
    try:
        payload = {
            "model":      entry.model,
            "messages":   [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        headers = {"Authorization": f"Bearer {entry.api_key}", "Content-Type": "application/json"}
        r = requests.post(entry.api_url, json=payload, headers=headers, timeout=15)
        if r.status_code in (401, 403):
            return False, f"Auth error ({r.status_code})"
        if r.status_code == 404:
            return False, f"Model not found (404)"
        r.raise_for_status()
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_all_providers(catalogue: Optional[List["ProviderEntry"]] = None) -> List["ProviderEntry"]:
    """
    Validate every entry in *catalogue* (or build a fresh one if None).
    Updates the module-level _HEALTHY_PROVIDERS cache and returns healthy list.
    Skips NVIDIA NIM (known slow — validated separately).
    """
    global _HEALTHY_PROVIDERS, _VALIDATION_TS

    if catalogue is None:
        catalogue = build_provider_catalogue()

    healthy: List[ProviderEntry] = []
    for entry in catalogue:
        if entry.provider == "nvidia":
            # Keep NVIDIA in catalogue but skip live validation (too slow)
            log.info(f"[validate] {entry.id}: skipped (NVIDIA NIM — validate manually)")
            healthy.append(entry)   # include but warn at query time
            continue

        ok, err = validate_provider(entry)
        if ok:
            log.info(f"[validate] {entry.id}: ✓ healthy")
            healthy.append(entry)
        else:
            log.warning(f"[validate] {entry.id}: ✗ unhealthy — {err}")

    _HEALTHY_PROVIDERS = healthy
    _VALIDATION_TS     = time.time()
    log.info(f"[validate] {len(healthy)}/{len(catalogue)} providers healthy")
    return healthy


def get_healthy_providers() -> List["ProviderEntry"]:
    """Return cached healthy providers; rebuild if cache is empty."""
    if _HEALTHY_PROVIDERS:
        return _HEALTHY_PROVIDERS
    return validate_all_providers()




# ─────────────────────────────────────────────────────────────────────────────
# Interactive provider picker  (shows only validated-healthy providers)
# ─────────────────────────────────────────────────────────────────────────────
def pick_provider_interactive(catalogue: List[ProviderEntry]) -> ProviderEntry:
    # Filter to healthy providers only; fall back to full catalogue if cache empty
    healthy = [e for e in catalogue if e in _HEALTHY_PROVIDERS] if _HEALTHY_PROVIDERS else catalogue
    if not healthy:
        healthy = catalogue   # nothing validated yet — show everything

    W_LABEL = 46; W_NOTE = 18
    border  = "─" * (W_LABEL + W_NOTE + 10)
    print(f"\n┌{border}┐")
    print(f"│  🤖  Select LLM Provider  (✓ = validated healthy){' ' * (len(border) - 50)}│")
    print(f"├────┬{'─'*W_LABEL}┬{'─'*W_NOTE}┤")
    print(f"│ #  │ {'Provider / Model'.ljust(W_LABEL-1)}│ {'Context / Notes'.ljust(W_NOTE-1)}│")
    print(f"├────┼{'─'*W_LABEL}┼{'─'*W_NOTE}┤")
    for i, e in enumerate(healthy, 1):
        label = e.label[:W_LABEL-1].ljust(W_LABEL-1)
        note  = e.context_note[:W_NOTE-1].ljust(W_NOTE-1)
        print(f"│ {str(i).ljust(2)} │ {label}│ {note}│")
    print(f"└────┴{'─'*W_LABEL}┴{'─'*W_NOTE}┘")
    print()
    print("  💡 Tip: use --provider or-qwen30b for best results (free, 131k ctx, no trimming)")
    print("         use --auto to skip this menu\n")
    while True:
        raw = input(f"  Enter number [1-{len(healthy)}] (or 'q' to quit): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        if raw.isdigit() and 1 <= int(raw) <= len(healthy):
            chosen = healthy[int(raw) - 1]
            print(f"  ✔  Using: {chosen.label}\n")
            return chosen
        print(f"  ⚠  Enter a number between 1 and {len(healthy)}")


def get_provider(
    catalogue:   List[ProviderEntry],
    provider_id: Optional[str],
    auto:        bool,
) -> List[ProviderEntry]:
    if not catalogue:
        raise ValueError(
            "No LLM providers available.\n\n"
            "Set at least one in your .env:\n"
            "  GROQ_API_KEY=gsk_...          https://console.groq.com/\n"
            "  OPENROUTER_API_KEY=sk-or-...  https://openrouter.ai/\n"
            "  GEMINI_API_KEY=AIza...        https://aistudio.google.com/app/apikey\n"
            "  NVIDIA_API_KEY=nvapi-...      https://build.nvidia.com/\n"
            "\nOr use Ollama locally (no key needed):\n"
            "  ollama serve && ollama pull qwen2.5:7b\n"
        )
    if provider_id:
        matches = [e for e in catalogue if e.id == provider_id]
        if not matches:
            valid = ", ".join(e.id for e in catalogue)
            raise ValueError(f"Provider '{provider_id}' not found. Available: {valid}")
        return matches
    if auto:
        return catalogue
    chosen = pick_provider_interactive(catalogue)
    return [chosen]


# ─────────────────────────────────────────────────────────────────────────────
# Token / size helpers
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ─────────────────────────────────────────────────────────────────────────────
# Context trimmer  — unchanged; still used for Groq's tiny free-tier window.
# For large-context providers the synthesis pipeline's PromptBuilder handles
# budget management instead.
# ─────────────────────────────────────────────────────────────────────────────
def _trim_chunks_to_budget(
    chunks:         List[RetrievedChunk],
    system_prompt:  str,
    user_prompt_base: str,
    entry:          ProviderEntry,
) -> List[RetrievedChunk]:
    """
    Simplified trimmer: for large-context providers returns all chunks.
    For Groq free tier applies the original character-budget logic.
    """
    model      = entry.model
    provider   = entry.provider
    ctx_budget = _MODEL_CTX.get(model) or (
        _MODEL_CTX["_ollama_default"] if provider == "ollama" else _DEFAULT_CTX
    )

    if ctx_budget >= _LARGE_CONTEXT_THRESHOLD:
        log.info(
            f"  Large-context provider ({model}: {ctx_budget:,} tok) — "
            f"sending all {len(chunks)} chunks without trimming"
        )
        return chunks

    token_budget     = ctx_budget - GROQ_MAX_TOKENS
    available_tokens = (
        token_budget
        - _estimate_tokens(system_prompt)
        - _estimate_tokens(user_prompt_base)
        - 200
    )
    if available_tokens <= 0:
        return chunks[:3]

    kept = []; used_tok = 0; used_chars = len(system_prompt) + len(user_prompt_base)
    for chunk in chunks:
        ct = _estimate_tokens(chunk.text) + 60
        cc = len(chunk.text) + 250
        if (used_tok + ct) > available_tokens:
            break
        if provider == "groq" and (used_chars + cc) > _GROQ_MAX_PAYLOAD_CHARS:
            break
        kept.append(chunk); used_tok += ct; used_chars += cc

    if len(kept) < len(chunks):
        log.info(f"  Trimmed {len(chunks)} → {len(kept)} chunks ({model})")
    return kept or chunks[:1]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini native call
# ─────────────────────────────────────────────────────────────────────────────
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
        timeout=110,   # Gemini can be slow on first call after idle
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


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible call
# ─────────────────────────────────────────────────────────────────────────────
def _call_openai_compat(system_prompt: str, user_prompt: str,
                        entry: ProviderEntry) -> dict:
    merge   = entry.model in _NO_SYSTEM_ROLE
    headers = {"Content-Type": "application/json"}
    if entry.api_key:
        headers["Authorization"] = f"Bearer {entry.api_key}"
    if "openrouter" in entry.api_url:
        # These headers are required by OpenRouter — missing them causes 403/429
        headers["HTTP-Referer"] = "https://github.com/Import-Saurabh/FinancialRag"
        headers["X-Title"]      = "FinancialRAG"
    messages = (
        [{"role": "user", "content": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
        if merge else
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": user_prompt}]
    )

    # Per-provider timeouts so a single stall doesn't eat the whole 180s budget
    if entry.provider == "groq":
        call_timeout = 55          # Groq is fast; 55s is generous
    elif entry.provider in ("openrouter", "nvidia"):
        call_timeout = 90          # Free-tier models can be slow
    else:
        call_timeout = 75

    payload = {
        "model":       entry.model,
        "messages":    messages,
        "max_tokens":  GROQ_MAX_TOKENS,
        "temperature": GROQ_TEMPERATURE,
        "stream":      False,      # never stream — we need the full JSON response
    }
    # Qwen3 thinking/reasoning mode is ON by default; it adds 10-40s and extra
    # tokens that blow the context budget — disable it for RAG queries.
    if "qwen3" in entry.model.lower():
        payload["thinking"] = {"type": "disabled"}

    resp = requests.post(
        entry.api_url,
        json=payload,
        headers=headers,
        timeout=call_timeout,
    )
    if not resp.ok:
        # Surface the actual API error message (e.g. wrong model slug gives 404)
        log.warning(
            f"  [{entry.provider}] HTTP {resp.status_code} for {entry.model!r} — "
            f"{resp.text[:400]}"
        )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Retry wrapper
# ─────────────────────────────────────────────────────────────────────────────
def _call_with_retry(system_prompt: str, user_prompt: str,
                     entry: ProviderEntry, max_retries: int = 3) -> dict:
    """
    Retry policy (revised):
      • 429 (rate-limit)  → back off and retry (max 2 retries, short delays)
      • 404/400/401/403   → fail immediately, no retry (bad key / wrong model)
      • Timeout           → retry once with no delay (transient network blip)
      • Any other error   → fail immediately
    The old policy (4 retries × up to 60s sleep) could burn 110s in sleeps alone,
    easily triggering the 180s client timeout before the LLM even responds.
    """
    # Short delays only for 429 — don't spend minutes waiting on a free-tier cap
    rate_limit_delays = [10, 20]; last_exc = None
    timeout_retries   = 1
    timeouts_seen     = 0

    for attempt in range(max_retries):
        try:
            if entry.provider == "gemini":
                return _call_gemini(system_prompt, user_prompt, entry.model, entry.api_key)
            if entry.provider == "anthropic":
                return _call_anthropic(system_prompt, user_prompt, entry)
            return _call_openai_compat(system_prompt, user_prompt, entry)

        except requests.exceptions.Timeout as e:
            last_exc = e
            timeouts_seen += 1
            if timeouts_seen <= timeout_retries:
                log.warning(
                    f"  Timeout on {entry.model} (attempt {attempt+1}), retrying once..."
                )
                continue          # immediate retry — no sleep
            raise                 # second timeout: give up on this provider

        except requests.HTTPError as e:
            status   = e.response.status_code
            last_exc = e
            if status == 429 and attempt < len(rate_limit_delays):
                wait = rate_limit_delays[attempt]
                log.warning(
                    f"  429 rate-limit on {entry.model} "
                    f"(attempt {attempt+1}/{max_retries}), retry in {wait}s"
                )
                time.sleep(wait)
            else:
                # 404 = wrong model slug, 401/403 = bad key, 5xx = server error
                # None of these benefit from sleeping — fail fast so the caller
                # can move to the next provider.
                raise

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# [SYNTHESIS] Lazy-init pipeline singleton
# One instance is shared for the process lifetime (stateless + thread-safe).
# ─────────────────────────────────────────────────────────────────────────────
_synthesis_pipeline = None

def _get_synthesis_pipeline():
    global _synthesis_pipeline
    if _synthesis_pipeline is None:
        try:
            from synthesis.pipeline import SynthesisPipeline
            _synthesis_pipeline = SynthesisPipeline()
            log.info("[rag_engine] SynthesisPipeline initialised")
        except Exception as exc:
            log.warning(f"[rag_engine] SynthesisPipeline unavailable: {exc} — using legacy path")
            _synthesis_pipeline = False   # sentinel: don't retry
    return _synthesis_pipeline if _synthesis_pipeline is not False else None


# ─────────────────────────────────────────────────────────────────────────────
# RAGResponse
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RAGResponse:
    answer:        str
    model_used:    str
    chunks_used:   int
    sources:       List[dict]
    tokens_used:   int
    latency_sec:   float
    # [SYNTHESIS] extra diagnostics (zero when legacy path is used)
    sql_rows:      int  = 0
    insights:      int  = 0
    pipeline_mode: str  = "vector_only"


# ─────────────────────────────────────────────────────────────────────────────
# Legacy prompt builders (kept intact; used when synthesis pipeline degrades)
# ─────────────────────────────────────────────────────────────────────────────

_FINANCIAL_STATEMENT_RULES = """\
FINANCIAL STATEMENT RETRIEVAL RULES (apply to every query):
A. EXACT METRIC MATCHING: If the user asks for EBIT but only EBITDA is in the context,
   flag this explicitly.  Never silently substitute one metric for another.
B. BALANCE SHEET: Look for pages titled "Balance Sheet" or "Consolidated Balance Sheet".
C. CASH FLOW STATEMENT: Look for "Statement of Cash Flows".
   Free cash flow = OCF − capex.
D. EPS NOTE: In Indian annual reports EPS is disclosed under "Earnings Per Share" (Ind AS 33).
E. RATIOS: ROCE = EBIT / Capital Employed.  ROE = PAT / Avg Shareholders Equity.
   Show numerator and denominator before computing the ratio."""

ANNUAL_SYSTEM_PROMPT = """\
You are a senior equity research analyst specialising in Indian listed companies.

YOUR RULES:
1. RECENCY FIRST: Always lead with the most recent fiscal year available.
2. USE ONLY CONTEXT: Do not use prior knowledge.
3. SHOW YOUR MATH: Write formula and numbers for every growth/trend calculation.
4. CURRENCY: State amounts exactly as shown in source (Crore / Lakh / Million).
5. CITE EVERY NUMBER: After each data point write [FY<year>, AR, Page <n>].
6. FLAG GAPS — ONLY FOR EXPLICITLY REQUESTED YEARS.
7. NO HALLUCINATION.

""" + _FINANCIAL_STATEMENT_RULES

CONCALL_SYSTEM_PROMPT = """\
You are a senior buy-side equity analyst reviewing earnings call transcripts.

YOUR RULES:
1. RECENCY FIRST: Lead with the most recent concall.
2. FORWARD-LOOKING PRIORITY: Prioritise guidance / outlook phrases.
3. QUOTE ACCURATELY: Name the speaker and their role.
4. FLAG GAPS — ONLY FOR EXPLICITLY REQUESTED YEARS.
5. USE ONLY CONTEXT."""

COMBINED_SYSTEM_PROMPT = """\
You are a senior equity research analyst with access to both annual reports
and earnings call transcripts for an Indian listed company.

YOUR RULES:
1. RECENCY FIRST.
2. CROSS-SOURCE SYNTHESIS: Label each source [Annual Report] or [Concall].
3. SHOW MATH: For any YOY growth write (new − old) / old × 100.
4. TABLES FOR TRENDS: Multi-year comparisons MUST be in a Markdown table.
5. CITE SOURCES: [FY<year>, AR/CC, Page <n>] after each data point.
6. FLAG MISSING DATA — ONLY FOR EXPLICITLY REQUESTED YEARS.
7. NO HALLUCINATION.

""" + _FINANCIAL_STATEMENT_RULES


def _build_context_legacy(chunks: List[RetrievedChunk]) -> str:
    def sort_key(c):
        yr  = c.metadata.get("year", 0)
        typ = 0 if c.metadata.get("doc_type") == "annual_report" else 1
        return (-yr, typ)
    sep   = "\n\n" + "─" * 60 + "\n\n"
    parts = []
    for i, chunk in enumerate(sorted(chunks, key=sort_key), 1):
        meta      = chunk.metadata
        doc_label = "Annual Report" if meta.get("doc_type") == "annual_report" else "Concall Transcript"
        section   = (meta.get("section") or meta.get("speaker") or "")[:60]
        tag = (
            f"[Source {i} | {meta.get('symbol','')} | {doc_label} | "
            f"FY{meta.get('year','')} | {section} | Page {meta.get('page_start','')}]"
        )
        parts.append(f"{tag}\n{chunk.text.strip()}")
    return sep.join(parts)


def _build_user_prompt_legacy(
    query: str, context: str, doc_type: str,
    resolved_years: Optional[List[int]] = None,
    explicit_years: Optional[List[int]] = None,
) -> str:
    year_note = (
        f"\n\nDATA SEARCHED: FY{'/'.join(str(y) for y in resolved_years)} documents."
        if resolved_years else ""
    )
    if explicit_years:
        gap_note = (
            f"\n\nGAP FLAG INSTRUCTION: Emit ⚠ ONLY for "
            f"FY{'/'.join(str(y) for y in explicit_years)} if data is missing."
        )
    else:
        gap_note = (
            "\n\nGAP FLAG INSTRUCTION: User did NOT specify years. "
            "Do NOT emit ⚠ gap flags."
        )
    q_lower = query.lower()
    intent  = (
        "\n\nINTENT NOTE: FORWARD-LOOKING query. Prioritise guidance/outlook chunks."
        if any(kw in q_lower for kw in ["outlook", "guidance", "expect", "h1", "h2",
                                         "demand environment", "going forward", "forecast"])
        else ""
    )
    return (
        f"CONTEXT FROM FINANCIAL DOCUMENTS (most recent first):\n"
        f"{'='*60}\n{context}\n{'='*60}"
        f"{year_note}{gap_note}{intent}\n\n"
        f"QUESTION: {query}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer using ONLY the context above.\n"
        f"- Lead with the most recent year's data.\n"
        f"- Show calculations explicitly for any growth/trend figures.\n"
        f"- Use a table if comparing across multiple years.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point  [SYNTHESIS-PATCHED]
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(
    query:          str,
    chunks:         List[RetrievedChunk],
    doc_type:       str = "annual_report",
    api_key:        Optional[str]       = None,   # legacy compat
    resolved_years: Optional[List[int]] = None,
    explicit_years: Optional[List[int]] = None,
    years:          Optional[List[int]] = None,   # legacy alias
    provider_id:    Optional[str]       = None,
    auto:           bool                = False,
    symbol:         Optional[str]       = None,   # NEW: forwarded to pipeline
) -> RAGResponse:
    """
    Generate a cited answer using the full synthesis pipeline.

    New parameters vs previous version
    ────────────────────────────────────
    symbol   — company ticker, forwarded to SynthesisPipeline so the
               decomposer can inject it onto atoms whose symbol could
               not be inferred from the query text alone.

    All other parameters are unchanged from the previous version.
    """
    # Back-compat aliases
    if resolved_years is None and years is not None:
        resolved_years = years
    if explicit_years is None:
        explicit_years = resolved_years or []

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

    catalogue = build_provider_catalogue()
    entries   = get_provider(catalogue, provider_id, auto)
    t0        = time.time()
    last_err  = None

    for entry in entries:
        log.info(f"  → {entry.label}")

        # ── [SYNTHESIS] Build (system, user) via the pipeline ─────────────────
        pipeline = _get_synthesis_pipeline()

        if pipeline is not None:
            try:
                sr = pipeline.run(
                    query          = query,
                    chunks         = chunks,
                    symbol         = symbol,
                    resolved_years = resolved_years,
                    explicit_years = explicit_years,
                    doc_type       = doc_type,
                    model          = entry.model,
                )
                system_prompt = sr.system_prompt
                user_prompt   = sr.user_prompt
                pipeline_mode = sr.pipeline_mode
                sql_rows      = sr.sql_rows
                n_insights    = sr.insights
                safe_chunks   = chunks   # pipeline manages its own budget

                if sr.warnings:
                    for w in sr.warnings:
                        log.warning(f"  [synthesis] {w}")

                log.info(
                    f"  [synthesis] mode={pipeline_mode} | "
                    f"sql_rows={sql_rows} insights={n_insights} "
                    f"chunks_used={sr.chunks_used}"
                )

            except Exception as exc:
                log.warning(
                    f"  [synthesis] pipeline raised unexpectedly: {exc} "
                    f"— falling back to legacy prompts"
                )
                pipeline = None   # trigger legacy path below

        # ── Legacy path (synthesis unavailable or crashed) ────────────────────
        if pipeline is None:
            legacy_system = {
                "annual_report": ANNUAL_SYSTEM_PROMPT,
                "concall":       CONCALL_SYSTEM_PROMPT,
            }.get(doc_type, COMBINED_SYSTEM_PROMPT)

            safe_chunks = _trim_chunks_to_budget(
                chunks, legacy_system,
                _build_user_prompt_legacy(query, "", doc_type, resolved_years, explicit_years),
                entry,
            )
            context       = _build_context_legacy(safe_chunks)
            system_prompt = legacy_system
            user_prompt   = _build_user_prompt_legacy(
                query, context, doc_type, resolved_years, explicit_years,
            )
            pipeline_mode = "vector_only"
            sql_rows      = 0
            n_insights    = 0

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            result  = _call_with_retry(system_prompt, user_prompt, entry)
            latency = time.time() - t0
            answer  = result["choices"][0]["message"]["content"]
            usage   = result.get("usage", {})

            log.info(
                f"  LLM ✓ [{entry.label}]: "
                f"{usage.get('completion_tokens', 0)} tokens | "
                f"{latency:.1f}s | {len(safe_chunks)}/{len(chunks)} chunks"
            )

            return RAGResponse(
                answer        = answer,
                model_used    = entry.label,
                chunks_used   = len(safe_chunks),
                sources       = [
                    {
                        "symbol":   c.metadata.get("symbol"),
                        "year":     c.metadata.get("year"),
                        "doc_type": c.metadata.get("doc_type"),
                        "section":  (c.metadata.get("section") or c.metadata.get("speaker", ""))[:50],
                        "page":     c.metadata.get("page_start"),
                        "score":    round(c.score, 4),
                    }
                    for c in safe_chunks
                ],
                tokens_used   = usage.get("total_tokens", 0),
                latency_sec   = round(latency, 2),
                sql_rows      = sql_rows,
                insights      = n_insights,
                pipeline_mode = pipeline_mode,
            )

        except (requests.HTTPError, Exception) as e:
            status  = getattr(getattr(e, "response", None), "status_code", "ERR")
            is_last = (entry == entries[-1])
            log.warning(
                f"  {entry.label} → HTTP {status} | "
                f"{'all providers exhausted' if is_last else 'trying next'}"
            )
            last_err = e

    raise RuntimeError(
        f"All selected providers failed. Last error: {last_err}\n\n"
        "Quick fixes:\n"
        "  • Best option: add OPENROUTER_API_KEY and use Qwen3-30B (free, 131k ctx)\n"
        "      https://openrouter.ai/ → pick 'or-qwen30b'\n"
        "  • Local Ollama: ollama serve && ollama pull qwen2.5:7b\n"
        "  • Groq free tier: wait ~60s then retry\n"
    )