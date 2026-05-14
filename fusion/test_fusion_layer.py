"""
fusion/test_fusion_layer.py

Self-contained unit tests for FusionLayer.
No DB, no embedding model, no network required.

Run:
    python -m fusion.test_fusion_layer
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decomposer.atomic_decomposer import AtomicNeed, NeedType, TimeHorizon
from schema_bridge.schema_bridge   import BridgeResult, SqlAtomResult, VectorAtomResult
from pipeline.retrieval.retriever  import RetrievedChunk
from fusion.fusion_layer import (
    FusionLayer, FusionResult, InsightType,
    MetricRow, ConcallClaim,
    _extract_numeric_claims, _parse_year_from_period, _pct_divergence,
)

WIDTH = 70
fusion = FusionLayer()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _atom(sub_type, symbol="RELIANCE", years=None):
    a = AtomicNeed(
        need_type=NeedType.QUANTITATIVE, sub_type=sub_type,
        metric=sub_type, symbol=symbol, years=years or [],
        time_horizon=TimeHorizon.HISTORICAL if years else TimeHorizon.CURRENT,
    )
    a.resolve_schema()
    return a

def _sql_result(sub_type, rows, symbol="RELIANCE", error=None):
    return SqlAtomResult(atom=_atom(sub_type, symbol), rows=rows, error=error)

def _chunk(text, doc_type="concall", speaker="CFO", role="management",
           year=2024, score=0.85):
    return RetrievedChunk(
        chunk_id="test", text=text, score=score,
        vector_score=score, bm25_score=score,
        metadata={"doc_type": doc_type, "speaker": speaker,
                  "speaker_role": role, "year": year, "symbol": "RELIANCE"},
    )

def _vec_result(sub_type, chunks):
    a = _atom(sub_type)
    return VectorAtomResult(atom=a, chunks=chunks)

def _bridge(sql=None, vec=None, errors=None):
    return BridgeResult(
        sql_results    = sql or [],
        vector_results = vec or [],
        errors         = errors or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg): print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}"); return False


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_year():
    assert _parse_year_from_period("2024-03-31") == 2024
    assert _parse_year_from_period("2023-09-30") == 2023
    assert _parse_year_from_period("")           is None
    _pass("_parse_year_from_period handles Mar/Sep/empty")


def test_pct_divergence():
    assert _pct_divergence(100.0, 110.0) == 0.10
    assert _pct_divergence(100.0,  90.0) == 0.10
    assert _pct_divergence(0.0,    10.0) == 0.0    # zero-actual guard
    _pass("_pct_divergence symmetric, zero-safe")


def test_number_extraction_percent():
    chunk = _chunk(
        "Going forward we expect EBITDA margins to improve to 22-23% "
        "driven by O2C segment recovery."
    )
    claims = _extract_numeric_claims(chunk, "ebitda_margin",
                                     ["ebitda margin", "operating margin"], "%")
    assert len(claims) >= 1
    c = claims[0]
    assert c.value_low  == 22.0
    assert c.value_high == 23.0
    assert c.unit       == "%"
    assert c.is_forward is True
    _pass(f"% range extracted: {c.value_low}–{c.value_high}% (forward={c.is_forward})")


def test_number_extraction_crore():
    chunk = _chunk(
        "Our capex guidance for FY25 is Rs 15,000 crore towards port expansion."
    )
    claims = _extract_numeric_claims(chunk, "capex",
                                     ["capex", "capital expenditure"], "crore")
    assert len(claims) >= 1
    c = claims[0]
    assert c.value_low == 15000.0
    assert c.unit      == "crore"
    assert c.is_forward is True
    _pass(f"crore extracted: {c.value_low} crore (forward={c.is_forward})")


def test_number_extraction_no_unit_ignored():
    """Numbers without a recognised unit should NOT be captured."""
    chunk = _chunk("Revenue grew 12 percent year on year in FY24.")
    # "12 percent" — the word "percent" is not in the unit set, only "%"
    claims = _extract_numeric_claims(chunk, "revenue", ["revenue", "sales"], "crore")
    # Should not find crore-unit numbers here
    assert all(c.unit not in ("%",) for c in claims), \
        "Should not have extracted % claims for crore metric"
    _pass("Numbers without matching unit are ignored")


def test_fuse_unmatched_no_concall():
    """SQL has revenue data, no concall chunks → UNMATCHED insight."""
    bridge = _bridge(
        sql=[_sql_result("revenue", [
            {"symbol": "RELIANCE", "period_end": "2024-03-31", "sales": 897658.0}
        ])]
    )
    result = fusion.fuse(bridge)
    assert len(result.metric_rows) == 1
    assert result.metric_rows[0].value == 897658.0
    assert len(result.unmatched()) == 1
    assert result.unmatched()[0].insight_type == InsightType.UNMATCHED
    _pass("UNMATCHED insight emitted when no concall data present")


def test_fuse_forward_guidance():
    """SQL has EBITDA margin, concall has forward-looking guidance → FORWARD."""
    bridge = _bridge(
        sql=[_sql_result("ebitda_margin", [
            {"symbol": "RELIANCE", "as_of_date": "2024-03-31",
             "ebitda": 180000.0, "ebitda_margin_pct": 20.0}
        ])],
        vec=[_vec_result("concall_margin", [
            _chunk("Going forward we expect EBITDA margins to improve to 22-23% "
                   "in FY25 driven by Jio subscriber growth.")
        ])]
    )
    result = fusion.fuse(bridge)
    fwd = result.forward_guidance()
    assert len(fwd) >= 1
    assert fwd[0].insight_type == InsightType.FORWARD
    assert "22" in fwd[0].note or "23" in fwd[0].note
    _pass(f"FORWARD insight: {fwd[0].note[:80]}…")


def test_fuse_confirm():
    """Reported EBITDA margin 20%, concall says ~20% (historical) → CONFIRM."""
    bridge = _bridge(
        sql=[_sql_result("ebitda_margin", [
            {"symbol": "RELIANCE", "as_of_date": "2024-03-31",
             "ebitda": 180000.0, "ebitda_margin_pct": 20.0}
        ])],
        vec=[_vec_result("concall_margin", [
            _chunk("Our EBITDA margin for the year stood at 20% as reported in Q4.")
        ])]
    )
    result = fusion.fuse(bridge)
    confirms = result.confirmations()
    assert len(confirms) >= 1
    assert confirms[0].insight_type == InsightType.CONFIRM
    _pass(f"CONFIRM insight: {confirms[0].note[:80]}…")


def test_fuse_contradict():
    """Reported net profit 79,000 cr, concall claims 50,000 cr (huge gap) → CONTRADICT."""
    bridge = _bridge(
        sql=[_sql_result("net_profit", [
            {"symbol": "RELIANCE", "period_end": "2024-03-31",
             "net_profit": 79000.0, "eps": 11.7}
        ])],
        vec=[_vec_result("concall_guidance", [
            _chunk("Our net profit for the year was approximately 50,000 crore.")
        ])]
    )
    result = fusion.fuse(bridge)
    contras = result.contradictions()
    assert len(contras) >= 1
    assert contras[0].insight_type == InsightType.CONTRADICT
    assert contras[0].divergence_pct > 15.0
    _pass(f"CONTRADICT insight (div={contras[0].divergence_pct:.1f}%): "
          f"{contras[0].note[:80]}…")


def test_fuse_empty_bridge():
    """Empty BridgeResult → empty FusionResult, no crash."""
    result = fusion.fuse(_bridge())
    assert result.metric_rows    == []
    assert result.concall_claims == []
    assert result.insights       == []
    _pass("Empty BridgeResult handled gracefully")


def test_fuse_sql_error_skipped():
    """SQL atoms with errors are skipped gracefully."""
    bridge = _bridge(
        sql=[_sql_result("revenue", [], error="table not found")]
    )
    result = fusion.fuse(bridge)
    assert result.metric_rows == []   # error row produces no MetricRow
    _pass("SQL error atoms produce no MetricRow (skipped)")


def test_annual_chunk_routing():
    """Annual report chunks go to annual_chunks, not concall_chunks."""
    bridge = _bridge(
        vec=[_vec_result("mda", [
            _chunk("The company's MD&A section discusses risk factors.",
                   doc_type="annual_report", score=0.9)
        ])]
    )
    result = fusion.fuse(bridge)
    assert len(result.annual_chunks)  == 1
    assert len(result.concall_chunks) == 0
    _pass("Annual report chunks routed to annual_chunks correctly")


def test_to_context_dict_structure():
    """to_context_dict() must contain all required keys."""
    result = fusion.fuse(_bridge(
        sql=[_sql_result("revenue", [
            {"symbol": "TCS", "period_end": "2024-03-31", "sales": 240893.0}
        ])],
        vec=[_vec_result("mda", [
            _chunk("MD&A discussion on risk.", doc_type="annual_report")
        ])]
    ))
    ctx = result.to_context_dict()
    required_keys = ["metric_table", "contradictions", "confirmations",
                     "forward_guidance", "unmatched", "concall_quotes",
                     "annual_excerpts", "errors"]
    for k in required_keys:
        assert k in ctx, f"Missing key: {k}"
    assert len(ctx["metric_table"]) == 1
    assert ctx["metric_table"][0]["value"] == 240893.0
    _pass(f"to_context_dict() has all {len(required_keys)} required keys")


def test_bridge_errors_propagated():
    """Errors from BridgeResult propagate into FusionResult.errors."""
    bridge = _bridge(errors=["SQLite error on annual_results"])
    result = fusion.fuse(bridge)
    assert "SQLite error on annual_results" in result.errors
    _pass("Bridge errors propagated into FusionResult.errors")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_parse_year,
    test_pct_divergence,
    test_number_extraction_percent,
    test_number_extraction_crore,
    test_number_extraction_no_unit_ignored,
    test_fuse_unmatched_no_concall,
    test_fuse_forward_guidance,
    test_fuse_confirm,
    test_fuse_contradict,
    test_fuse_empty_bridge,
    test_fuse_sql_error_skipped,
    test_annual_chunk_routing,
    test_to_context_dict_structure,
    test_bridge_errors_propagated,
]


def run():
    print("\n" + "═" * WIDTH)
    print("  Fusion Layer — Unit Tests")
    print("═" * WIDTH)

    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            _fail(f"{fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            _fail(f"{fn.__name__}: unexpected — {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "─" * WIDTH)
    print(f"  Results: {passed}/{len(TESTS)} passed | {failed} failed")
    print("─" * WIDTH + "\n")
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)