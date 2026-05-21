"""
query_client.py — drop-in replacement for query.py
────────────────────────────────────────────────────
Shows the same provider menu as query.py, then sends the choice to the
running server.py (which keeps all models warm between queries).

USAGE:
    python query_client.py --symbol ADANIPORTS "revenue FY25"
    python query_client.py --symbol ADANIPORTS --auto "revenue FY25"
    python query_client.py --symbol ADANIPORTS --doc-type annual "segment breakdown"

Make sure server.py is running first:
    C:\\Users\\hp\\Downloads\\FinRag\\venv\\Scripts\\python.exe server.py
"""

import argparse
import sys

try:
    import requests
except ImportError:
    print("[error] pip install requests")
    sys.exit(1)

SERVER_URL = "http://localhost:8000"

# ── Provider menu (mirrors rag_engine.py build_provider_catalogue) ────────────
# Fetched dynamically from server so it stays in sync with rag_engine
FALLBACK_PROVIDERS = [
    # ── Fast / recommended ────────────────────────────────────────────────────
    {"id": "groq-llama",              "label": "Groq — llama-3.3-70b-versatile ★ FASTEST",   "note": "~5.5k tok (free)"},
    {"id": "or-llama70b",             "label": "OpenRouter — Llama 3.3 70B [FREE] ★ BEST",   "note": "131k ctx, FREE"},
    # ── OpenRouter free tier (working) ────────────────────────────────────────
    {"id": "or-gemini",               "label": "OpenRouter — Gemini 2.0 Flash [FREE]",       "note": "1M ctx, FREE"},
    # ── Direct APIs ───────────────────────────────────────────────────────────
    {"id": "gemini",                  "label": "Google Gemini — gemini-2.0-flash (direct)",  "note": "1M ctx, 15 RPM"},
    {"id": "nvidia",                  "label": "NVIDIA NIM — llama-3.3-70b ⚠ SLOW (~90s)",  "note": "128k ctx, slow"},
    # ── Local Ollama ──────────────────────────────────────────────────────────
    {"id": "ollama-llama3.1-latest",  "label": "Ollama local — llama3.1:latest",            "note": "4.9 GB, local"},
    {"id": "ollama-phi3-latest",      "label": "Ollama local — phi3:latest",                "note": "2.2 GB, local"},
    # ── Last resort ───────────────────────────────────────────────────────────
    {"id": "groq-llama-8b",           "label": "Groq — llama-3.1-8b-instant (fast fallback)", "note": "6k tok"},
]


def fetch_providers():
    """Try to get live provider list from server; fall back to hardcoded."""
    try:
        r = requests.get(f"{SERVER_URL}/providers", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return FALLBACK_PROVIDERS


def pick_provider(providers) -> str:
    W_LABEL, W_NOTE = 46, 18
    sep = "├────┼" + "─" * W_LABEL + "┼" + "─" * W_NOTE + "┤"
    top = "┌────┬" + "─" * W_LABEL + "┬" + "─" * W_NOTE + "┐"
    mid = "├────┼" + "─" * W_LABEL + "┼" + "─" * W_NOTE + "┤"
    bot = "└────┴" + "─" * W_LABEL + "┴" + "─" * W_NOTE + "┘"

    print()
    print(top)
    print(f"│ {'🤖  Select LLM Provider':<{W_LABEL + W_NOTE + 5}}│")
    print(mid)
    print(f"│ #  │ {'Provider / Model':<{W_LABEL-1}}│ {'Context / Notes':<{W_NOTE-1}}│")
    print(mid)
    for i, p in enumerate(providers, 1):
        label = p["label"][:W_LABEL-1].ljust(W_LABEL-1)
        note  = p.get("note", "")[:W_NOTE-1].ljust(W_NOTE-1)
        print(f"│ {i:<3}│ {label}│ {note}│")
    print(bot)
    print()
    print("  💡 Tip: groq-llama = fastest (2s LLM) | or-llama70b = best free quality (131k ctx)")
    print("         ⚠  Avoid NVIDIA NIM for interactive use — frequently times out at 160s")
    print("         use --auto flag to skip this menu")
    print()

    while True:
        try:
            raw = input(f"  Enter number [1-{len(providers)}] (or 'q' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if raw.lower() == "q":
            sys.exit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(providers):
            chosen = providers[int(raw) - 1]
            print(f"  ✔  Using: {chosen['label']}\n")
            return chosen["id"]
        print(f"  ⚠  Enter a number between 1 and {len(providers)}")


def main():
    parser = argparse.ArgumentParser(description="FinRAG query client")
    parser.add_argument("query",        help="Natural language query")
    parser.add_argument("--symbol",     default=None)
    parser.add_argument("--doc-type",   default="both",
                        choices=["both", "annual_report", "concall"])
    parser.add_argument("--year",       type=int, default=None)
    parser.add_argument("--auto",       action="store_true",
                        help="Skip provider menu, use best available automatically")
    parser.add_argument("--provider",   default=None,
                        help="Skip menu and use this provider ID directly")
    args = parser.parse_args()

    # ── Check server is up ────────────────────────────────────────────────────
    try:
        requests.get(f"{SERVER_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"\n[error] Cannot connect to {SERVER_URL}")
        print("        Start the server first:")
        print("        C:\\Users\\hp\\Downloads\\FinRag\\venv\\Scripts\\python.exe server.py\n")
        sys.exit(1)

    # ── Provider selection ────────────────────────────────────────────────────
    if args.auto:
        provider_id = "auto"
    elif args.provider:
        provider_id = args.provider
    else:
        providers = fetch_providers()
        provider_id = pick_provider(providers)

    # ── Send query ────────────────────────────────────────────────────────────
    payload = {
        "query":    args.query,
        "symbol":   args.symbol,
        "doc_type": args.doc_type,
        "year":     args.year,
        "provider": provider_id,
    }

    print("🔍 Retrieving and re-ranking...")
    try:
        resp = requests.post(f"{SERVER_URL}/query", json=payload, timeout=175)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        print(
            "\n[error] Client timed out waiting for server (>175s).\n"
            "  The server is still processing — this is likely a slow/stalled LLM provider.\n"
            "  Tips:\n"
            "    • Use groq-llama (option 1) — fastest and most reliable\n"
            "    • Use --auto to let the server try providers in order\n"
            "    • Free OpenRouter models can queue for 60-120s under load\n"
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        print(f"[error] Server returned HTTP {resp.status_code}:\n  {detail}")
        sys.exit(1)

    data = resp.json()

    # ── Print answer ──────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("ANSWER")
    print("=" * 74)
    print(data["answer"])

    if data.get("sources"):
        print(f"\n── Sources ({data['chunks_used']} chunks used) ──")
        for i, src in enumerate(data["sources"], 1):
            print(f"  [{i}] {src}")

    print(f"\n── Meta ──")
    print(f"  Model  : {data['model_used']}")
    print(f"  Latency: {data['latency_sec']}s  |  Mode: {data['pipeline_mode']}")
    print(f"  SQL    : {data['sql_rows']} rows  |  Insights: {data['insights']}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()