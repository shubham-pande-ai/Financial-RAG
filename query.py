#!/usr/bin/env python3
"""
query.py  — patched to surface synthesis pipeline diagnostics.

Changes vs previous version
────────────────────────────
[SYNTHESIS] run_query now passes symbol= to generate_answer so the
            SynthesisPipeline can inject the correct ticker onto atoms.

[SYNTHESIS] print_answer now shows sql_rows / insights / pipeline_mode
            when verbose=True.

All CLI flags (--symbol, --type, --year, --year-range, --provider, --auto,
--interactive, --verbose) are unchanged.
"""

import argparse
import os
import sys

from config.settings import LOG_DIR, COMBINED_ANNUAL_CHUNKS, COMBINED_CONCALL_CHUNKS
from db.database import init_db
from pipeline.retrieval import retrieve
from pipeline.retrieval.reranker import rerank, rerank_separate
from rag.rag_engine import generate_answer
from utils.logger import get_logger

log = get_logger(__name__, LOG_DIR)


# ─────────────────────────────────────────────
# Pretty print answer
# ─────────────────────────────────────────────
def print_answer(response, verbose: bool = False):
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.answer)

    print(f"\n── Sources ({response.chunks_used} chunks used) ──")
    for i, src in enumerate(response.sources, 1):
        dt      = "AR" if src["doc_type"] == "annual_report" else "CC"
        section = (src.get("section") or "")[:40]
        print(f"  [{i}] {src['symbol']} FY{src['year']} [{dt}] | {section} | page {src['page']} | score {src['score']}")

    if verbose:
        print(
            f"\n── Model: {response.model_used} | "
            f"Tokens: {response.tokens_used} | "
            f"Latency: {response.latency_sec}s ──"
        )
        # [SYNTHESIS] extra diagnostics
        print(
            f"── Pipeline: mode={response.pipeline_mode} | "
            f"sql_rows={response.sql_rows} | "
            f"insights={response.insights} ──"
        )
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# Core query function
# ─────────────────────────────────────────────
def run_query(
    query:      str,
    doc_type:   str,
    symbol:     str   = None,
    year:       int   = None,
    year_range: tuple = None,
    verbose:    bool  = False,
):
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENROUTER_API_KEY") \
            and not os.getenv("GEMINI_API_KEY") and not os.getenv("NVIDIA_API_KEY"):
        print("\n⚠  No LLM API key found.")
        print("Set at least one of:")
        print("  GROQ_API_KEY          https://console.groq.com/")
        print("  OPENROUTER_API_KEY    https://openrouter.ai/  (Qwen3-30B FREE)")
        print("  GEMINI_API_KEY        https://aistudio.google.com/app/apikey")
        print("  NVIDIA_API_KEY        https://build.nvidia.com/")
        print()
        print("Or use local Ollama (no key):")
        print("  ollama serve && ollama pull qwen2.5:7b")
        print()
        sys.exit(1)

    log.info(f"Query: '{query}'")
    log.info(f"  doc_type={doc_type} | symbol={symbol} | year={year}")

    # Step 1: Retrieve
    print(f"\n🔍 Retrieving relevant chunks...")
    result = retrieve(
        query=query,
        doc_type=doc_type,
        symbol=symbol,
        year=year,
        year_range=year_range,
    )

    # Resolve years the same way retrieve() does
    from pipeline.retrieval.retriever import parse_year_intent
    if year:
        resolved_years = [year]
    elif year_range:
        resolved_years = list(range(year_range[0], year_range[1] + 1))
    else:
        resolved_years = parse_year_intent(query)

    print(f"  📅 Year filter applied: FY{resolved_years}")

    # Step 2: Re-rank
    if doc_type == "both":
        annual_candidates, concall_candidates = result
        if not annual_candidates and not concall_candidates:
            print("❌ No relevant documents found.")
            print("Have you ingested documents? Run: python ingest.py --symbol <SYMBOL>")
            return
        total = len(annual_candidates) + len(concall_candidates)
        print(
            f"📊 Re-ranking {total} candidates "
            f"({len(annual_candidates)} annual + {len(concall_candidates)} concall) separately..."
        )
        top_chunks = rerank_separate(
            query,
            annual_candidates,
            concall_candidates,
            annual_top_k=COMBINED_ANNUAL_CHUNKS,
            concall_top_k=COMBINED_CONCALL_CHUNKS,
        )
    else:
        candidates = result
        if not candidates:
            print("❌ No relevant documents found.")
            print("Have you ingested documents? Run: python ingest.py --symbol <SYMBOL>")
            return
        print(f"📊 Re-ranking {len(candidates)} candidates...")
        top_chunks = rerank(query, candidates, doc_type)

    # Step 3: Generate — [SYNTHESIS] symbol is now forwarded
    print(f"🤖 Generating answer ({len(top_chunks)} chunks)...")
    response = generate_answer(
        query          = query,
        chunks         = top_chunks,
        doc_type       = doc_type,
        resolved_years = resolved_years,
        symbol         = symbol,    # [SYNTHESIS] new
    )

    print_answer(response, verbose)


# ─────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────
def interactive_mode():
    print("\n Financial RAG — Interactive Mode")
    print(" Type 'exit' to quit, 'help' for options\n")

    symbol = input("Company symbol (or press Enter for all): ").strip().upper() or None
    doc_type_input = input("Document type [annual/concall/both] (default: both): ").strip().lower()
    doc_type = {"annual": "annual_report", "concall": "concall"}.get(doc_type_input, "both")
    year_input = input("Year filter (or press Enter for all): ").strip()
    year = int(year_input) if year_input.isdigit() else None

    print(f"\nSettings: symbol={symbol} | type={doc_type} | year={year}")
    print("─" * 40)

    while True:
        try:
            query = input("\n❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if query.lower() in ("exit", "quit", "q"):
            break
        if query.lower() == "help":
            print("  Commands: exit | quit | help")
            continue
        if not query:
            continue
        run_query(query, doc_type, symbol=symbol, year=year, verbose=True)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Financial RAG — Query")
    ap.add_argument("query",        nargs="?",   help="Question to ask")
    ap.add_argument("--symbol", "-s",            help="Filter by company symbol")
    ap.add_argument("--type",   "-t",
                    choices=["annual", "concall", "both"], default="both",
                    help="Document type to search")
    ap.add_argument("--year",       type=int,    help="Filter by specific year")
    ap.add_argument("--year-range", nargs=2, type=int, metavar=("FROM", "TO"),
                    help="Filter by year range e.g. --year-range 2021 2024")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show model/token/pipeline info")
    ap.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    ap.add_argument("--provider",    default=None,
                    help="Pin to provider ID (e.g. or-qwen30b, groq-llama, gemini)")
    ap.add_argument("--auto",        action="store_true",
                    help="Skip interactive provider menu (use first available)")
    args = ap.parse_args()

    init_db()

    if args.interactive:
        interactive_mode()
        return

    if not args.query:
        ap.print_help()
        sys.exit(1)

    doc_type   = {"annual": "annual_report", "concall": "concall", "both": "both"}[args.type]
    year_range = tuple(args.year_range) if args.year_range else None

    run_query(
        query      = args.query,
        doc_type   = doc_type,
        symbol     = args.symbol,
        year       = args.year,
        year_range = year_range,
        verbose    = args.verbose,
    )


if __name__ == "__main__":
    main()