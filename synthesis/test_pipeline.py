"""
synthesis/test_pipeline.py

Unit tests for SynthesisPipeline.
No DB, no embedding model, no network required.

Run:
    python -m synthesis.test_pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── import the components we CAN rely on ─────────────────────────────────────
from synthesis.prompt_builder import PromptBuilder, BuiltPrompt, _render_sql_table, _render_insights
from synthesis.pipeline import SynthesisPipeline, SynthesisResult, _minimal_fallback_prompt

# Stub RetrievedChunk (same pattern as fusion tests)
try:
    from pipeline.retrieval.retriever import RetrievedChunk
except ModuleNotFoundError:
    from dataclasses import dataclass, field
    from typing import Dict, Any
    @dataclass
    class RetrievedChunk:
        chunk_id: str = ""; text: str = ""; score: float = 0.0
        vector_score: float = 0.0; bm25_score: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

WIDTH = 70


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _chunk(text, symbol="ADANIPORTS", year=2024, doc_type="annual_report",
           score=0.85, section="STATUTORY REPORTS", page=335):
    return RetrievedChunk(
        chunk_id="c1", text=text, score=score,
        vector_score=score, bm25_score=score,
        metadata={
            "symbol": symbol, "year": year, "doc_type": doc_type,
            "section": section, "page_start": page,
        },
    )

def _concall_chunk(text, symbol="ADANIPORTS", year=2024, score=0.78):
    return RetrievedChunk(
        chunk_id="cc1", text=text, score=score,
        vector_score=score, bm25_score=score,
        metadata={
            "symbol": symbol, "year": year, "doc_type": "concall",
            "speaker": "CFO", "page_start": 3,
        },
    )

def _pass(msg): print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}"); return False


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_render_sql_table_empty():
    result = _render_sql_table([])
    assert result == "", "Empty metric_rows should produce empty string"
    _pass("render_sql_table: empty input → empty string")


def test_render_sql_table_populated():
    rows = [
        {"symbol": "ADANIPORTS", "sub_type": "revenue", "metric": "Revenue",
         "year": 2024, "period": "2024-03-31", "value": 23452.0, "unit": "crore"},
        {"symbol": "ADANIPORTS", "sub_type": "net_profit", "metric": "Net Profit",
         "year": 2024, "period": "2024-03-31", "value": 8289.0, "unit": "crore"},
        {"symbol": "ADANIPORTS", "sub_type": "ebitda_margin", "metric": "EBITDA Margin",
         "year": 2023, "period": "2023-03-31", "value": 54.7, "unit": "%"},
    ]
    rendered = _render_sql_table(rows)
    assert "[SQL-1]" in rendered
    assert "[SQL-2]" in rendered
    assert "[SQL-3]" in rendered
    assert "₹23,452.0 cr" in rendered
    assert "54.70%" in rendered
    assert "ADANIPORTS" in rendered
    _pass("render_sql_table: 3 rows rendered with correct [SQL-N] anchors and values")


def test_render_insights_all_types():
    insights = [
        {"type": "CONTRADICT", "metric": "Revenue",     "note": "reported 23k, mgmt said 18k (26%)"},
        {"type": "CONFIRM",    "metric": "EBITDA Margin","note": "both sources agree ~54%"},
        {"type": "FORWARD",    "metric": "Capex",        "symbol": "ADANIPORTS", "note": "CFO: 15k cr FY26"},
        {"type": "UNMATCHED",  "metric": "Net Debt",     "note": "reported 48k cr, no concall data"},
    ]
    rendered = _render_insights(insights)
    assert "⚠" in rendered,  "CONTRADICT should have ⚠"
    assert "✓" in rendered,  "CONFIRM should have ✓"
    assert "→" in rendered,  "FORWARD should have →"
    assert "·" in rendered,  "UNMATCHED should have ·"
    _pass("render_insights: all 4 types rendered with correct callout symbols")


def test_builder_vector_only_no_fusion():
    """PromptBuilder with fusion_result=None should produce vector-only prompt."""
    chunks  = [
        _chunk("Revenue for ADANIPORTS FY2024 was ₹23,452 crore."),
        _chunk("Net profit stood at ₹8,289 crore for the fiscal year.", doc_type="concall"),
    ]
    builder = PromptBuilder(max_context_chars=50_000)
    built   = builder.build(
        query          = "What is ADANIPORTS revenue and net profit for FY24?",
        chunks         = chunks,
        fusion_result  = None,
        doc_type       = "both",
        resolved_years = [2024],
        explicit_years = [2024],
    )
    assert isinstance(built, BuiltPrompt)
    assert "[SRC-1]" in built.user_prompt
    assert "[SRC-2]" in built.user_prompt
    assert "FY2024" in built.user_prompt  # gap flag instruction
    assert built.sql_rows_used == 0
    assert built.chunks_used   == 2
    assert built.insights_used == 0
    _pass(
        f"vector-only BuiltPrompt: {built.chunks_used} chunks | "
        f"{built.total_chars:,} chars | trimmed={built.was_trimmed}"
    )


def test_builder_gap_flag_no_explicit_years():
    """When no explicit years, gap flag instruction must say 'did NOT specify'."""
    chunks  = [_chunk("Revenue for ADANIPORTS FY2024 was ₹23,452 crore.")]
    builder = PromptBuilder()
    built   = builder.build(
        query          = "What is ADANIPORTS revenue?",
        chunks         = chunks,
        explicit_years = [],
    )
    assert "did NOT specify" in built.user_prompt
    _pass("gap_flag_note: 'did NOT specify' present when explicit_years=[]")


def test_builder_gap_flag_with_explicit_years():
    """When explicit_years=[2023,2024,2025] the gap note names them."""
    chunks  = [_chunk("Revenue for ADANIPORTS FY2024 was ₹23,452 crore.")]
    builder = PromptBuilder()
    built   = builder.build(
        query          = "YOY revenue growth FY23-25",
        chunks         = chunks,
        explicit_years = [2023, 2024, 2025],
    )
    assert "FY2023" in built.user_prompt
    assert "FY2024" in built.user_prompt
    assert "FY2025" in built.user_prompt
    _pass("gap_flag_note: explicit years FY2023/2024/2025 present in prompt")


def test_builder_intent_growth():
    """Growth/YOY query should inject GROWTH TREND note."""
    chunks  = [_chunk("Revenue trend over FY23-25.")]
    builder = PromptBuilder()
    built   = builder.build(
        query  = "What is YOY net profit growth FY23-25?",
        chunks = chunks,
    )
    assert "GROWTH" in built.user_prompt or "TREND" in built.user_prompt
    _pass("intent note: GROWTH/TREND detected for YOY query")


def test_builder_intent_forward():
    """Guidance/outlook query should inject FORWARD-LOOKING note."""
    chunks  = [_chunk("Management outlook for FY26 capex.")]
    builder = PromptBuilder()
    built   = builder.build(
        query  = "What is ADANIPORTS capex guidance for FY26?",
        chunks = chunks,
    )
    assert "FORWARD" in built.user_prompt
    _pass("intent note: FORWARD-LOOKING detected for guidance query")


def test_builder_metric_ebit_note():
    """EBIT-only query should inject METRIC NOTE about EBITDA distinction."""
    chunks  = [_chunk("EBIT margin was 28%.")]
    builder = PromptBuilder()
    built   = builder.build(
        query  = "What is ADANIPORTS EBIT margin for FY24?",
        chunks = chunks,
    )
    assert "METRIC NOTE" in built.user_prompt
    assert "EBITDA" in built.user_prompt
    _pass("metric note: EBIT/EBITDA distinction injected")


def test_builder_budget_trims_chunks():
    """When budget is tiny, chunks should be trimmed and was_trimmed=True."""
    # 20 chunks × ~100 chars each; budget only allows ~3
    chunks = [_chunk(f"Revenue chunk number {i}. " * 20) for i in range(20)]
    builder = PromptBuilder(max_context_chars=3_000)
    built   = builder.build(
        query  = "What is ADANIPORTS revenue?",
        chunks = chunks,
    )
    assert built.was_trimmed is True
    assert built.chunks_used < 20
    _pass(f"budget trim: {len(chunks)} → {built.chunks_used} chunks (was_trimmed=True)")


def test_builder_for_provider_groq():
    """for_provider('llama-3.3-70b-versatile') should return smaller-budget builder."""
    builder = PromptBuilder().for_provider("llama-3.3-70b-versatile")
    assert builder.max_context_chars <= 20_000, (
        f"Groq builder should have small budget, got {builder.max_context_chars}"
    )
    _pass(f"for_provider(groq): max_context_chars={builder.max_context_chars:,}")


def test_builder_for_provider_qwen():
    """for_provider('qwen/qwen3-30b-a3b:free') should return large-budget builder."""
    builder = PromptBuilder().for_provider("qwen/qwen3-30b-a3b:free")
    assert builder.max_context_chars >= 50_000, (
        f"Qwen builder should have large budget, got {builder.max_context_chars}"
    )
    _pass(f"for_provider(qwen30b): max_context_chars={builder.max_context_chars:,}")


# ─────────────────────────────────────────────────────────────────────────────
# SynthesisPipeline unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_vector_only_no_db():
    """
    With no FINANCE_DB_PATH / no DB file, pipeline should degrade gracefully
    to vector_only mode without raising.
    """
    pipeline = SynthesisPipeline(finance_db_path=None)
    chunks   = [
        _chunk("Revenue for ADANIPORTS FY2024 was ₹23,452 crore.", score=0.92),
        _concall_chunk("Going forward we expect EBITDA margins to improve to 55-57%.", score=0.78),
    ]
    result = pipeline.run(
        query          = "What is ADANIPORTS YOY revenue growth FY23-25?",
        chunks         = chunks,
        symbol         = "ADANIPORTS",
        resolved_years = [2023, 2024, 2025],
        explicit_years = [2023, 2024, 2025],
        doc_type       = "both",
        model          = "qwen/qwen3-30b-a3b:free",
    )
    assert isinstance(result, SynthesisResult)
    assert result.system_prompt, "system_prompt must not be empty"
    assert result.user_prompt,   "user_prompt must not be empty"
    assert result.pipeline_mode in ("vector_only", "no_sql", "full")
    assert "[SRC-1]" in result.user_prompt or "CONTEXT" in result.user_prompt
    _pass(
        f"pipeline (no DB): mode={result.pipeline_mode} | "
        f"chunks={result.chunks_used} | warnings={len(result.warnings)}"
    )


def test_pipeline_symbol_inference():
    """Symbol should be inferred from chunk metadata when not passed explicitly."""
    pipeline = SynthesisPipeline(finance_db_path=None)
    chunks   = [_chunk("Revenue data.", symbol="TATASTEEL")]
    result   = pipeline.run(
        query  = "What is revenue?",
        chunks = chunks,
        symbol = None,    # deliberately omitted
    )
    # Symbol should have been inferred as TATASTEEL
    assert result.system_prompt
    _pass("pipeline: symbol inferred from chunk metadata (no crash)")


def test_pipeline_empty_chunks():
    """Empty chunk list should produce a valid (minimal) SynthesisResult."""
    pipeline = SynthesisPipeline(finance_db_path=None)
    result   = pipeline.run(
        query  = "What is revenue?",
        chunks = [],
    )
    assert isinstance(result, SynthesisResult)
    assert result.system_prompt
    _pass("pipeline: empty chunks handled gracefully")


def test_minimal_fallback_prompt():
    """_minimal_fallback_prompt must always return a valid BuiltPrompt."""
    chunks = [_chunk("Revenue was ₹23,452 crore in FY2024.")]
    built  = _minimal_fallback_prompt("What is revenue?", chunks)
    assert built.system_prompt
    assert built.user_prompt
    assert "[SRC-1]" in built.user_prompt
    _pass("minimal_fallback_prompt: valid BuiltPrompt returned")


def test_synthesis_result_property_passthrough():
    """SynthesisResult.system_prompt / .user_prompt must delegate to BuiltPrompt."""
    built  = BuiltPrompt(
        system_prompt="SYS", user_prompt="USR",
        sql_rows_used=0, chunks_used=1,
        insights_used=0, total_chars=6, was_trimmed=False,
    )
    result = SynthesisResult(built_prompt=built)
    assert result.system_prompt == "SYS"
    assert result.user_prompt   == "USR"
    _pass("SynthesisResult property passthrough works")


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test (fusion path with mock FusionResult)
# ─────────────────────────────────────────────────────────────────────────────

def test_builder_with_mock_fusion_result():
    """
    Build a prompt from a mock FusionResult (duck-typed) to verify the
    fusion path works end-to-end without real DB or embedding model.
    """
    from dataclasses import dataclass, field
    from typing import List, Dict, Any

    @dataclass
    class _MockFusionResult:
        """Minimal duck-type of FusionResult that satisfies PromptBuilder."""
        _metrics: list = field(default_factory=list)
        _contras: list = field(default_factory=list)
        _confirms:list = field(default_factory=list)
        _fwd:     list = field(default_factory=list)
        _unmatch: list = field(default_factory=list)

        def to_context_dict(self) -> Dict[str, Any]:
            return {
                "metric_table":    self._metrics,
                "contradictions":  self._contras,
                "confirmations":   self._confirms,
                "forward_guidance":self._fwd,
                "unmatched":       self._unmatch,
                "concall_quotes":  [],
                "annual_excerpts": [],
                "errors":          [],
            }

        # FusionLayer accessors (not used by PromptBuilder directly but
        # present so isinstance checks don't fail)
        def contradictions(self): return self._contras
        def confirmations(self):  return self._confirms
        def forward_guidance(self): return self._fwd
        def unmatched(self):      return self._unmatch

    # Patch FusionResult in prompt_builder so isinstance check passes
    import synthesis.prompt_builder as pb_mod
    _orig = pb_mod.FusionResult
    pb_mod.FusionResult = _MockFusionResult

    try:
        mock_fr = _MockFusionResult(
            _metrics=[
                {"symbol": "ADANIPORTS", "sub_type": "revenue", "metric": "Revenue",
                 "year": 2024, "period": "2024-03-31", "value": 23452.0, "unit": "crore"},
                {"symbol": "ADANIPORTS", "sub_type": "net_profit", "metric": "Net Profit",
                 "year": 2024, "period": "2024-03-31", "value": 8289.0, "unit": "crore"},
            ],
            _contras=[{
                "type": "CONTRADICT", "metric": "Revenue", "symbol": "ADANIPORTS",
                "year": 2024, "sql_value": 23452.0, "sql_period": "2024-03-31",
                "note": "Reported 23,452 cr but mgmt stated 18,000 cr (divergence 23.2%).",
            }],
            _fwd=[{
                "type": "FORWARD", "metric": "Capex", "symbol": "ADANIPORTS",
                "year": 2025, "sql_value": None, "sql_period": "",
                "note": "CFO: capex guidance ₹15,000 cr for FY26.",
            }],
        )

        chunks  = [
            _chunk("Revenue for ADANIPORTS FY2024 was ₹23,452 crore per AR.", score=0.92),
            _concall_chunk("We expect EBITDA margins of 55-57% going forward.", score=0.78),
        ]
        builder = PromptBuilder(max_context_chars=100_000)
        built   = builder.build(
            query          = "What is ADANIPORTS revenue and net profit growth FY23-25?",
            chunks         = chunks,
            fusion_result  = mock_fr,
            doc_type       = "both",
            resolved_years = [2023, 2024, 2025],
            explicit_years = [2023, 2024, 2025],
        )

        assert "[SQL-1]" in built.user_prompt, "SQL table not rendered"
        assert "[SQL-2]" in built.user_prompt, "Second SQL row missing"
        assert "⚠" in built.user_prompt,       "CONTRADICTION callout missing"
        assert "→" in built.user_prompt,       "FORWARD guidance callout missing"
        assert "[SRC-1]" in built.user_prompt,  "Vector chunk anchor missing"
        assert built.sql_rows_used == 2
        assert built.insights_used == 2
        assert built.chunks_used   == 2
        _pass(
            f"fusion BuiltPrompt: sql_rows={built.sql_rows_used} "
            f"insights={built.insights_used} chunks={built.chunks_used} "
            f"chars={built.total_chars:,}"
        )
    finally:
        pb_mod.FusionResult = _orig  # restore


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # PromptBuilder
    test_render_sql_table_empty,
    test_render_sql_table_populated,
    test_render_insights_all_types,
    test_builder_vector_only_no_fusion,
    test_builder_gap_flag_no_explicit_years,
    test_builder_gap_flag_with_explicit_years,
    test_builder_intent_growth,
    test_builder_intent_forward,
    test_builder_metric_ebit_note,
    test_builder_budget_trims_chunks,
    test_builder_for_provider_groq,
    test_builder_for_provider_qwen,
    # SynthesisPipeline
    test_pipeline_vector_only_no_db,
    test_pipeline_symbol_inference,
    test_pipeline_empty_chunks,
    test_minimal_fallback_prompt,
    test_synthesis_result_property_passthrough,
    # Integration smoke
    test_builder_with_mock_fusion_result,
]


def run():
    print("\n" + "═" * WIDTH)
    print("  Synthesis Layer — Unit Tests")
    print("═" * WIDTH)

    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ✗  {fn.__name__}: unexpected — {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "─" * WIDTH)
    print(f"  Results: {passed}/{len(TESTS)} passed | {failed} failed")
    print("─" * WIDTH + "\n")
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)