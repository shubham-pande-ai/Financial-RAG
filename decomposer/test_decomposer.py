"""
decomposer/test_decomposer.py

Run:  python -m decomposer.test_decomposer
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decomposer.atomic_decomposer import (
    AtomicDecomposer, NeedType, TimeHorizon, SUBTYPE_TABLE_MAP
)

decomposer = AtomicDecomposer()

# ─────────────────────────────────────────────────────────────────────────────
# Test cases: (query, expected_sub_types_present, expected_channels)
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    # Single quantitative
    (
        "What is ADANIPORTS revenue for FY24?",
        ["revenue"],
        ["sql"],
        ["ADANIPORTS"],
        [2024],
    ),
    # Multi-metric annual
    (
        "Show me RELIANCE EBITDA margin, ROCE and net debt for FY23 FY24 FY25",
        ["ebitda_margin", "roce", "net_debt"],
        ["sql"],
        ["RELIANCE"],
        [2023, 2024, 2025],
    ),
    # Cash flow
    (
        "What is TCS free cash flow and capex over last 3 years?",
        ["fcf", "capex"],
        ["sql"],
        ["TCS"],
        [],   # "last 3 years" → no explicit FY years extracted
    ),
    # Balance sheet
    (
        "HDFCBANK total assets, borrowings and cash equivalents FY25",
        ["total_assets", "borrowings", "cash"],
        ["sql"],
        ["HDFCBANK"],
        [2025],
    ),
    # Valuation
    (
        "What is INFY P/E ratio, P/B and EV/EBITDA?",
        ["pe", "pb", "ev_ebitda"],
        ["sql"],
        ["INFY"],
        [],
    ),
    # Growth metrics
    (
        "TATAMOTORS revenue CAGR and profit CAGR 5 year",
        ["revenue_cagr", "profit_cagr"],
        ["sql"],
        [],
        [],
    ),
    # Return ratios
    (
        "Compare ROE, ROCE and ROA for WIPRO vs INFOSYS",
        ["roe", "roce", "roa"],
        ["sql", "vector"],
        [],
        [],
    ),
    # Technical
    (
        "What is the RSI and MACD for ICICIBANK? Is it above 200 SMA?",
        ["rsi", "macd", "sma"],
        ["sql"],
        ["ICICIBANK"],
        [],
    ),
    # Macro
    (
        "What is the current repo rate and USD/INR forex rate?",
        ["repo_rate", "forex"],
        ["sql"],
        [],
        [],
    ),
    # Ownership
    (
        "What is ADANIPORTS promoter holding and FII stake?",
        ["promoter", "fii"],
        ["sql"],
        ["ADANIPORTS"],
        [],
    ),
    # Qualitative
    (
        "Summarise the MD&A section and risk factors in RELIANCE annual report",
        ["mda", "risk_factors"],
        ["vector"],
        ["RELIANCE"],
        [],
    ),
    # Forward-looking
    (
        "What guidance did ADANIPORTS management give for FY26 capex and margins?",
        ["concall_capex", "concall_margin"],
        ["concall"],
        ["ADANIPORTS"],
        [],
    ),
    # Mixed: quantitative + qualitative + forward
    (
        "What is TATASTEEL EBITDA for FY24 and what did management say about demand outlook?",
        ["ebitda", "concall_outlook"],
        ["sql", "concall"],
        ["TATASTEEL"],
        [2024],
    ),
    # Working capital
    (
        "Show BAJFINANCE DSO, DIO and cash conversion cycle",
        ["dso", "dio", "ccc"],
        ["sql"],
        [],
        [],
    ),
    # EPS + estimates
    (
        "What is SUNPHARMA TTM EPS and analyst EPS estimate for FY26?",
        ["eps", "eps_estimate"],
        ["sql"],
        ["SUNPHARMA"],
        [2026],
    ),
    # Corporate action
    (
        "What dividends has ITC paid in the last 5 years?",
        ["dividend"],
        ["sql"],
        ["ITC"],
        [],
    ),
    # Single quantitative — 52w merged
    (
        "What is COALINDIA 52-week high and low?",
        ["52w_high"],       # merged into one atom; both high+low from same table
        ["sql"],
        ["COALINDIA"],
        [],
    ),
    # Quarterly — period_type="quarterly" distinguishes, sub_type=net_profit is fine
    (
        "What are quarterly sales and net profit for MARUTI in Q3 FY25?",
        ["revenue_q", "net_profit"],   # net_profit_q deduped since same sub_type diff; net_profit fires
        ["sql"],
        ["MARUTI"],
        [2025],
    ),
]


def run_tests():
    passed = 0
    failed = 0
    width  = 72

    print("\n" + "═" * width)
    print("  Atomic Decomposer Test Suite")
    print("═" * width)

    for i, (query, expected_subtypes, expected_channels, exp_symbols, exp_years) in enumerate(TEST_CASES, 1):
        result  = decomposer.decompose_verbose(query)
        atoms   = result["atoms"]
        found_s = {a["sub_type"] for a in atoms}
        found_c = set(result["channels"])

        missing_subtypes  = [s for s in expected_subtypes if s not in found_s]
        missing_channels  = [c for c in expected_channels if c not in found_c]
        # Symbols / years: soft check (warn, don't fail)
        sym_ok = all(s in result["symbols"] for s in exp_symbols)
        yr_ok  = all(y in result["years"]   for y in exp_years)

        ok = not missing_subtypes and not missing_channels

        status = "✓ PASS" if ok else "✗ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{i:02d}] {status}  {query[:65]}")
        print(f"     Found atoms : {sorted(found_s)}")
        print(f"     Channels    : {sorted(found_c)}")
        if result["symbols"]:
            print(f"     Symbols     : {result['symbols']}")
        if result["years"]:
            print(f"     Years       : {result['years']}")
        print(f"     Elapsed     : {result['elapsed_ms']} ms")

        if missing_subtypes:
            print(f"     ❌ Missing sub_types : {missing_subtypes}")
        if missing_channels:
            print(f"     ❌ Missing channels  : {missing_channels}")
        if not sym_ok:
            print(f"     ⚠  Expected symbols {exp_symbols}, got {result['symbols']}")
        if not yr_ok:
            print(f"     ⚠  Expected years   {exp_years}, got {result['years']}")

    print("\n" + "─" * width)
    print(f"  Results: {passed}/{len(TEST_CASES)} passed | {failed} failed")
    print("─" * width)

    # ── Schema map coverage check ───────────────────────────────────────────
    print("\n── SUBTYPE_TABLE_MAP coverage ──")
    print(f"  {len(SUBTYPE_TABLE_MAP)} sub-types registered")
    chromadb_types = [k for k, v in SUBTYPE_TABLE_MAP.items() if "chromadb" in v[0]]
    sql_types      = [k for k, v in SUBTYPE_TABLE_MAP.items() if "chromadb" not in v[0]]
    print(f"  SQL-backed  : {len(sql_types)}")
    print(f"  Vector-backed: {len(chromadb_types)}")
    print()

    # ── Detailed verbose output for one interesting query ────────────────────
    demo_query = (
        "Compare ADANIPORTS and TATASTEEL FCF, net debt and "
        "EBITDA margin for FY23-25 and what is management guidance?"
    )
    print("── Demo: complex multi-intent query ──")
    print(f"  Query: {demo_query}")
    demo = decomposer.decompose_verbose(demo_query)
    print(f"  Atoms found: {demo['atom_count']}")
    print(f"  Channels   : {demo['channels']}")
    print(f"  Symbols    : {demo['symbols']}")
    print(f"  Years      : {demo['years']}")
    print(f"  Need types : {demo['need_types']}")
    print()
    for a in demo["atoms"]:
        tbl = a.get("sql_table") or "—"
        cols = ", ".join(a.get("sql_columns") or []) or "—"
        print(f"    [{a['need_type']:>16}]  {a['metric']:<30}  table={tbl}  cols=[{cols}]")

    print()
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)