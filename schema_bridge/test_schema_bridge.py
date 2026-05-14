"""
schema_bridge/test_schema_bridge.py

Tests for the SchemaBridge.

Two test modes
──────────────
1. UNIT (no DB required)  — verifies _build_sql() output, channel routing,
   comparative expansion, and FY date mapping.  Runs anywhere.

2. INTEGRATION (optional) — fires fetch() against a real DB path.
   Skipped automatically if the DB file is not found.

Run:
    python -m schema_bridge.test_schema_bridge
    python -m schema_bridge.test_schema_bridge --db /path/to/financial_rag.db
"""

import sys
import os
import argparse

# ── path bootstrap (run from project root) ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decomposer.atomic_decomposer import AtomicNeed, NeedType, TimeHorizon
from schema_bridge.schema_bridge import (
    SchemaBridge,
    BridgeResult,
    SqlAtomResult,
    VectorAtomResult,
    _build_sql,
    _classify,
    _expand_comparative,
    _fy_date_range,
)

WIDTH = 70


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atom(sub_type, symbol=None, years=None, period="annual",
          need_type=NeedType.QUANTITATIVE, symbols=None):
    a = AtomicNeed(
        need_type   = need_type,
        sub_type    = sub_type,
        metric      = sub_type,
        symbol      = symbol,
        symbols     = symbols or [],
        years       = years or [],
        time_horizon= TimeHorizon.HISTORICAL if years else TimeHorizon.CURRENT,
        period_type = period,
    )
    a.resolve_schema()
    return a


def _pass(msg): print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}"); return False


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fy_date_range():
    start, end = _fy_date_range(2024)
    assert start == "2023-04-01", f"Expected 2023-04-01 got {start}"
    assert end   == "2024-03-31", f"Expected 2024-03-31 got {end}"
    _pass("FY2024 → 2023-04-01 … 2024-03-31")


def test_sql_revenue():
    a = _atom("revenue", symbol="RELIANCE", years=[2024])
    sql, params = _build_sql(a)
    assert "annual_results" in sql,              "Table missing"
    assert "sales"          in sql,              "Column missing"
    assert "symbol = ?"     in sql,              "Symbol filter missing"
    assert "BETWEEN"        in sql,              "Date range missing"
    assert "RELIANCE"       in params,           "Symbol not in params"
    assert "2023-04-01"     in params,           "FY start not in params"
    assert "2024-03-31"     in params,           "FY end not in params"
    _pass(f"revenue SQL: {sql[:80]}…")


def test_sql_revenue_multiyear():
    a = _atom("revenue", symbol="TCS", years=[2022, 2023, 2024])
    sql, params = _build_sql(a)
    assert "BETWEEN" in sql,       "BETWEEN expected for multi-year"
    assert "2021-04-01" in params, "Multi-year start wrong"
    assert "2024-03-31" in params, "Multi-year end wrong"
    _pass(f"multi-year revenue SQL ok ({len(params)} params)")


def test_sql_no_symbol_macro():
    a = _atom("repo_rate", symbol=None, years=[])
    sql, params = _build_sql(a)
    assert "symbol" not in sql.lower().split("from")[0], \
        "Macro table should not have symbol filter"
    assert "rbi_rates" in sql
    _pass(f"macro (repo_rate) SQL has no symbol filter: {sql[:70]}…")


def test_sql_balance_sheet_period_type():
    a = _atom("net_debt", symbol="HDFCBANK", years=[2025], period="annual")
    sql, params = _build_sql(a)
    assert "period_type = ?" in sql
    assert "annual"          in params
    _pass("balance_sheet: period_type filter present")


def test_sql_quarterly():
    a = _atom("revenue_q", symbol="ICICIBANK", years=[2024], period="quarterly")
    sql, params = _build_sql(a)
    assert "quarterly_results" in sql
    _pass(f"quarterly_results table used for revenue_q")


def test_sql_corporate_action_type():
    a = _atom("dividend", symbol="ITC", years=[])
    sql, params = _build_sql(a)
    assert "corporate_actions" in sql
    assert "action_type = ?"   in sql
    assert "Dividend"          in params
    _pass("corporate_actions: action_type = 'Dividend' filter present")


def test_sql_current_limit():
    """No years → ORDER BY date DESC LIMIT used."""
    a = _atom("pe", symbol="INFY", years=[])
    sql, params = _build_sql(a)
    assert "LIMIT" in sql, "Expected LIMIT for current query"
    assert "DESC"  in sql
    _pass(f"current (no year) query has DESC LIMIT: …{sql[-30:]}")


def test_channel_routing():
    cases = [
        ("revenue",        NeedType.QUANTITATIVE,   "sql"),
        ("mda",            NeedType.QUALITATIVE,    "vector"),
        ("concall_outlook",NeedType.FORWARD_LOOKING,"vector"),
        ("rsi",            NeedType.TECHNICAL,       "sql"),
        ("promoter",       NeedType.OWNERSHIP,       "sql"),
        ("repo_rate",      NeedType.MACRO,           "sql"),
    ]
    for sub_type, need_type, expected in cases:
        a = _atom(sub_type, need_type=need_type)
        got = _classify(a)
        assert got == expected, f"{sub_type}: expected {expected!r} got {got!r}"
    _pass(f"channel routing correct for {len(cases)} atom types")


def test_comparative_expansion():
    a = _atom("roce", symbols=["WIPRO", "INFOSYS"],
              need_type=NeedType.COMPARATIVE)
    expanded = _expand_comparative(a)
    assert len(expanded) == 2
    assert {e.symbol for e in expanded} == {"WIPRO", "INFOSYS"}
    assert all(e.need_type == NeedType.QUANTITATIVE for e in expanded)
    _pass("COMPARATIVE atom expands to one atom per symbol")


def test_vector_backed_sub_type():
    """Atoms whose sql_table starts with 'chromadb:' must route to vector."""
    a = _atom("risk_factors", need_type=NeedType.QUALITATIVE)
    channel = _classify(a)
    assert channel == "vector", f"risk_factors should be vector, got {channel}"
    _pass("risk_factors routes to vector channel")


def test_sql_ownership():
    a = _atom("promoter", symbol="ADANIPORTS", years=[2024])
    sql, params = _build_sql(a)
    assert "ownership_history" in sql
    assert "promoter_pct"      in sql
    _pass(f"ownership (promoter) SQL: {sql[:70]}…")


# ─────────────────────────────────────────────────────────────────────────────
# Integration test (requires real DB)
# ─────────────────────────────────────────────────────────────────────────────

def test_integration_fetch(db_path: str):
    """
    db_path should point at Ai_Hedge_Fund.db (the structured financial DB),
    NOT financial_rag.db (which only holds RAG metadata).

    Example:
        python -m schema_bridge.test_schema_bridge \
            --db "C:/Users/hp/Downloads/Fund/database/Ai_Hedge_Fund.db"
    """
    from pathlib import Path
    p = Path(db_path)
    if not p.exists():
        print(f"  ⚠  DB not found at {db_path} — skipping integration test")
        print(f"      Pass Ai_Hedge_Fund.db, not financial_rag.db")
        return True

    # Quick schema check — warn clearly if wrong DB passed
    import sqlite3
    conn = sqlite3.connect(str(p))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    if "annual_results" not in tables:
        print(f"  ⚠  {p.name} has no 'annual_results' table.")
        print(f"      Tables found: {sorted(tables)}")
        print(f"      You likely passed financial_rag.db — use Ai_Hedge_Fund.db instead.")
        return True

    bridge = SchemaBridge(finance_db_path=p)

    # SQL-only atoms (vector atoms need the embedding model online)
    atoms = [
        _atom("revenue",    symbol="RELIANCE", years=[2024]),
        _atom("net_profit", symbol="RELIANCE", years=[2024]),
        _atom("net_debt",   symbol="RELIANCE", years=[2024]),
        _atom("roce",       symbol="RELIANCE"),
    ]

    result: BridgeResult = bridge.fetch(atoms)

    print(f"  SQL results   : {len(result.sql_results)}")
    for r in result.sql_results:
        tag = "✓" if not r.error else "✗"
        print(f"    {tag} {r.atom.sub_type}: {len(r.rows)} row(s)"
              + (f"  ERROR: {r.error}" if r.error else ""))

    print(f"  Vector results: {len(result.vector_results)}")
    for r in result.vector_results:
        tag = "✓" if not r.error else "✗"
        print(f"    {tag} {r.atom.sub_type}: {len(r.chunks)} chunk(s)"
              + (f"  ERROR: {r.error}" if r.error else ""))

    if result.errors:
        print(f"  Errors: {result.errors}")

    # Soft assertion: no hard crashes, results returned
    _pass(f"integration fetch completed (errors={len(result.errors)})")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

UNIT_TESTS = [
    test_fy_date_range,
    test_sql_revenue,
    test_sql_revenue_multiyear,
    test_sql_no_symbol_macro,
    test_sql_balance_sheet_period_type,
    test_sql_quarterly,
    test_sql_corporate_action_type,
    test_sql_current_limit,
    test_channel_routing,
    test_comparative_expansion,
    test_vector_backed_sub_type,
    test_sql_ownership,
]


def run_all(db_path: str = None):
    print("\n" + "═" * WIDTH)
    print("  Schema Bridge — Unit Tests")
    print("═" * WIDTH)

    passed = failed = 0
    for fn in UNIT_TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            _fail(f"{fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            _fail(f"{fn.__name__}: unexpected exception — {e}")
            failed += 1

    print()
    if db_path:
        print("── Integration test ──")
        test_integration_fetch(db_path)

    print("\n" + "─" * WIDTH)
    print(f"  Results: {passed}/{len(UNIT_TESTS)} unit tests passed "
          f"| {failed} failed")
    print("─" * WIDTH + "\n")
    return failed == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None,
                    help="Path to financial_rag.db for integration test")
    args = ap.parse_args()
    ok = run_all(db_path=args.db)
    sys.exit(0 if ok else 1)