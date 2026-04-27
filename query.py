#!/usr/bin/env python3
"""
query.py
CLI to query the Financial RAG system.

Usage:
    python query.py "What is Reliance's revenue for FY24?"
    python query.py --symbol RELIANCE "What is the revenue growth trend?"
    python query.py --symbol RELIANCE --type concall "What did management say about capex?"
    python query.py --symbol RELIANCE --year 2024 "Summarise the MD&A section"
    python query.py --symbol RELIANCE --year-range 2021 2024 "Debt trend over 3 years"
    python query.py --interactive   # REPL mode
"""

import argparse
import json
import os
import sys

from config.settings import LOG_DIR
from db.database import init_db
from pipeline.retrieval import retrieve, rerank
from rag.rag_engine import generate_answer
from utils.logger import get_logger

log = get_logger(__name__, LOG_DIR)


# ─────────────────────────────────────────────
# Pretty print answer
# ─────────────────────────────────────────────
def print_answer(response, verbose: bool = False):
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(response.answer)

    print(f"\n── Sources ({response.chunks_used} chunks used) ──")
    for i, src in enumerate(response.sources, 1):
        dt = "AR" if src["doc_type"] == "annual_report" else "CC"
        section = (src.get("section") or "")[:40]
        print(f"  [{i}] {src['symbol']} FY{src['year']} [{dt}] | {section} | page {src['page']} | score {src['score']}")

    if verbose:
        print(f"\n── Model: {response.model_used} | Tokens: {response.tokens_used} | Latency: {response.latency_sec}s ──")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# Core query function
# ─────────────────────────────────────────────
def run_query(
    query: str,
    doc_type: str,
    symbol: str = None,
    year: int = None,
    year_range: tuple = None,
    verbose: bool = False,
):
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠  GROQ_API_KEY not set.")
        print("Get a free key at: https://console.groq.com/")
        print()
        print("Then add it to your .env file in the project root:")
        print('  GROQ_API_KEY=gsk_your_key_here')
        print()
        print("Or set it in your terminal:")
        print("  Windows PowerShell : $env:GROQ_API_KEY='gsk_...'")
        print("  Windows CMD        : set GROQ_API_KEY=gsk_...")
        print("  Linux/Mac          : export GROQ_API_KEY=gsk_...")
        print()
        sys.exit(1)

    log.info(f"Query: '{query}'")
    log.info(f"  doc_type={doc_type} | symbol={symbol} | year={year}")

    # Step 1: Retrieve
    print(f"\n🔍 Retrieving relevant chunks...")
    candidates = retrieve(
        query=query,
        doc_type=doc_type,
        symbol=symbol,
        year=year,
        year_range=year_range,
    )

    if not candidates:
        print("❌ No relevant documents found.")
        print("Have you ingested documents? Run: python ingest.py --symbol <SYMBOL>")
        return

    # Step 2: Re-rank
    print(f"📊 Re-ranking {len(candidates)} candidates...")
    top_chunks = rerank(query, candidates, doc_type)

    # Step 3: Generate
    print(f"🤖 Generating answer with Groq ({len(top_chunks)} chunks)...")
    response = generate_answer(query, top_chunks, doc_type)

    print_answer(response, verbose)


# ─────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────
def interactive_mode():
    print("\n Financial RAG — Interactive Mode")
    print(" Type 'exit' to quit, 'help' for options\n")

    symbol = input("Company symbol (or press Enter for all): ").strip().upper() or None
    doc_type_input = input("Document type [annual/concall/both] (default: both): ").strip().lower()
    doc_type = {
        "annual": "annual_report",
        "concall": "concall",
    }.get(doc_type_input, "both")

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
            print("  Just type your question about the company's financials")
            continue
        if not query:
            continue

        run_query(query, doc_type, symbol=symbol, year=year, verbose=True)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Financial RAG — Query")
    ap.add_argument("query", nargs="?", help="Question to ask")
    ap.add_argument("--symbol", "-s", help="Filter by company symbol")
    ap.add_argument(
        "--type", "-t",
        choices=["annual", "concall", "both"],
        default="both",
        help="Document type to search",
    )
    ap.add_argument("--year", type=int, help="Filter by specific year")
    ap.add_argument("--year-range", nargs=2, type=int, metavar=("FROM", "TO"),
                    help="Filter by year range e.g. --year-range 2021 2024")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show model/token info")
    ap.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    args = ap.parse_args()

    init_db()

    if args.interactive:
        interactive_mode()
        return

    if not args.query:
        ap.print_help()
        sys.exit(1)

    doc_type = {
        "annual": "annual_report",
        "concall": "concall",
        "both": "both",
    }[args.type]

    year_range = tuple(args.year_range) if args.year_range else None

    run_query(
        query=args.query,
        doc_type=doc_type,
        symbol=args.symbol,
        year=args.year,
        year_range=year_range,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()