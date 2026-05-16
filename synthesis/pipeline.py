"""
synthesis/pipeline.py

Orchestrates the full intent-decomposition → retrieval → fusion →
prompt-building chain that sits *between* the reranker and the LLM call.

                    ┌──────────────────────────────────┐
  top_chunks  ─────▶│                                  │
  raw query   ─────▶│   SynthesisPipeline.run()        │──▶ BuiltPrompt
  symbol/years ────▶│                                  │     (system + user)
                    └──────────────────────────────────┘
                         │            │          │
                   Decomposer    SchemaBridge  FusionLayer
                   (atoms)       (SQL+vec)     (insights)
                                               PromptBuilder

Design decisions
────────────────
1. FULLY OPTIONAL — if any upstream component is unavailable (missing DB,
   import error, no FINANCE_DB_PATH) the pipeline gracefully degrades to
   the vector-only prompt path so query.py never crashes.

2. SYMBOL FROM CHUNKS — if no symbol is passed explicitly, it is inferred
   from the metadata of the highest-scoring chunk.

3. FREE MODELS ONLY — provider-id and model name are forwarded from
   rag_engine so the prompt builder can size itself correctly.  No
   hard-coded Claude/GPT references anywhere.

4. NO NEW CLI FLAGS — query.py detects the new pipeline automatically.
   Old --provider / --auto flags still work.

5. THREAD SAFE — SchemaBridge already uses ThreadPoolExecutor internally;
   this wrapper is stateless.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── graceful imports ──────────────────────────────────────────────────────────
try:
    from decomposer.atomic_decomposer import AtomicDecomposer, AtomicNeed
    _HAS_DECOMPOSER = True
except ModuleNotFoundError:
    _HAS_DECOMPOSER = False

try:
    from schema_bridge.schema_bridge import SchemaBridge, BridgeResult
    _HAS_BRIDGE = True
except ModuleNotFoundError:
    _HAS_BRIDGE = False

try:
    from fusion.fusion_layer import FusionLayer, FusionResult
    _HAS_FUSION = True
except ModuleNotFoundError:
    _HAS_FUSION = False

try:
    from synthesis.prompt_builder import PromptBuilder, BuiltPrompt
    _HAS_BUILDER = True
except ModuleNotFoundError:
    _HAS_BUILDER = False

try:
    from pipeline.retrieval.retriever import RetrievedChunk
except ModuleNotFoundError:
    from dataclasses import dataclass as _dc, field as _f
    @_dc
    class RetrievedChunk:           # type: ignore[no-redef]
        chunk_id: str = ""; text: str = ""; score: float = 0.0
        vector_score: float = 0.0; bm25_score: float = 0.0
        metadata: Dict[str, Any] = _f(default_factory=dict)

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except ModuleNotFoundError:
    import logging
    log = logging.getLogger(__name__)

# ── FINANCE_DB_PATH — where the structured financial SQLite lives ─────────────
try:
    from config.settings import FINANCE_DB_PATH as _FINANCE_DB_PATH
except (ImportError, AttributeError):
    _FINANCE_DB_PATH = None   # bridge will skip SQL if None


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """Everything the LLM call needs, plus diagnostics."""

    built_prompt:    "BuiltPrompt"

    # Diagnostics (all optional — populated when pipeline ran fully)
    atoms:           List[Any]  = field(default_factory=list)   # AtomicNeed list
    fusion_result:   Any        = None                           # FusionResult | None
    sql_rows:        int        = 0
    insights:        int        = 0
    chunks_used:     int        = 0
    pipeline_mode:   str        = "vector_only"   # "full" | "no_sql" | "vector_only"
    warnings:        List[str]  = field(default_factory=list)

    # Convenience pass-throughs so callers don't need to unpack BuiltPrompt
    @property
    def system_prompt(self) -> str:
        return self.built_prompt.system_prompt

    @property
    def user_prompt(self) -> str:
        return self.built_prompt.user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Capability check helper
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_available() -> bool:
    """True only when every component is importable AND finance DB exists."""
    if not (_HAS_DECOMPOSER and _HAS_BRIDGE and _HAS_FUSION and _HAS_BUILDER):
        return False
    if _FINANCE_DB_PATH is None:
        return False
    if not Path(str(_FINANCE_DB_PATH)).exists():
        return False
    return True


def _builder_available() -> bool:
    """True when at least prompt builder is importable (vector-only path)."""
    return _HAS_BUILDER


# ─────────────────────────────────────────────────────────────────────────────
# SynthesisPipeline
# ─────────────────────────────────────────────────────────────────────────────

class SynthesisPipeline:
    """
    Runs the full chain: decompose → bridge → fuse → build prompt.

    Usage (from rag_engine.generate_answer):

        pipeline = SynthesisPipeline()
        result   = pipeline.run(
            query          = query,
            chunks         = safe_chunks,        # already reranked
            symbol         = "ADANIPORTS",
            resolved_years = [2023, 2024, 2025],
            explicit_years = [2023, 2024, 2025],
            doc_type       = "both",
            model          = entry.model,        # for prompt sizing
        )
        # result.system_prompt / result.user_prompt  → pass to _call_with_retry
    """

    def __init__(
        self,
        finance_db_path: Optional[Path] = None,
        max_workers:     int = 6,
    ):
        self._db = finance_db_path or (_FINANCE_DB_PATH and Path(str(_FINANCE_DB_PATH)))
        self._max_workers = max_workers

        # Lazy-init components — only instantiated when actually needed
        self._decomposer:  Optional[Any] = None
        self._bridge:      Optional[Any] = None
        self._fusion:      Optional[Any] = None

    # ── Lazy initialisers ─────────────────────────────────────────────────────

    def _get_decomposer(self) -> Optional[Any]:
        if not _HAS_DECOMPOSER:
            return None
        if self._decomposer is None:
            self._decomposer = AtomicDecomposer()
        return self._decomposer

    def _get_bridge(self) -> Optional[Any]:
        if not (_HAS_BRIDGE and self._db and Path(str(self._db)).exists()):
            return None
        if self._bridge is None:
            self._bridge = SchemaBridge(
                finance_db_path=Path(str(self._db)),
                max_workers=self._max_workers,
            )
        return self._bridge

    def _get_fusion(self) -> Optional[Any]:
        if not _HAS_FUSION:
            return None
        if self._fusion is None:
            self._fusion = FusionLayer()
        return self._fusion

    # ── Symbol inference ──────────────────────────────────────────────────────

    @staticmethod
    def _infer_symbol(
        chunks:  List["RetrievedChunk"],
        symbol:  Optional[str],
    ) -> Optional[str]:
        if symbol:
            return symbol.upper()
        for c in chunks:
            sym = c.metadata.get("symbol")
            if sym:
                return str(sym).upper()
        return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        symbol:         Optional[str]       = None,
        resolved_years: Optional[List[int]] = None,
        explicit_years: Optional[List[int]] = None,
        doc_type:       str                 = "both",
        model:          str                 = "",
    ) -> SynthesisResult:
        """
        Run the full pipeline. Never raises — on any error degrades gracefully.

        Returns SynthesisResult whose .system_prompt / .user_prompt are
        always populated (worst case: vector-only prompt matching old behaviour).
        """
        warnings: List[str] = []

        # ── Select prompt builder (size it for the target model) ───────────────
        if not _HAS_BUILDER:
            warnings.append("PromptBuilder not available — falling back to rag_engine prompts")
            # Caller must use the old rag_engine path; we return None signal
            return self._fallback_result(query, chunks, resolved_years,
                                         explicit_years, doc_type, warnings)

        builder = PromptBuilder().for_provider(model) if model else PromptBuilder()

        # ── Infer symbol from chunks if not provided ───────────────────────────
        eff_symbol = self._infer_symbol(chunks, symbol)

        # ── Decide which pipeline mode to run ─────────────────────────────────
        #
        # "full"        → decompose + SQL bridge + fusion + prompt builder
        # "no_sql"      → decompose + fusion (vector only) + prompt builder
        # "vector_only" → just prompt builder (legacy fallback)
        #
        full_ok = _pipeline_available()

        if full_ok:
            return self._run_full(
                query, chunks, eff_symbol,
                resolved_years, explicit_years, doc_type,
                builder, warnings,
            )
        else:
            # Log why full pipeline isn't available (once, at INFO level)
            if not _HAS_DECOMPOSER:
                warnings.append("decomposer not importable — using vector-only path")
            elif not _HAS_BRIDGE:
                warnings.append("schema_bridge not importable — using vector-only path")
            elif _FINANCE_DB_PATH is None:
                warnings.append("FINANCE_DB_PATH not set — using vector-only path")
            elif not Path(str(_FINANCE_DB_PATH)).exists():
                warnings.append(
                    f"FINANCE_DB_PATH={_FINANCE_DB_PATH} not found — using vector-only path"
                )
            else:
                warnings.append("fusion layer unavailable — using vector-only path")

            return self._run_vector_only(
                query, chunks, resolved_years, explicit_years,
                doc_type, builder, warnings,
            )

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def _run_full(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        symbol:         Optional[str],
        resolved_years: Optional[List[int]],
        explicit_years: Optional[List[int]],
        doc_type:       str,
        builder:        "PromptBuilder",
        warnings:       List[str],
    ) -> SynthesisResult:

        # Step 1 — decompose query into atoms
        try:
            decomposer = self._get_decomposer()
            atoms: List[AtomicNeed] = decomposer.decompose(query, symbol=symbol)

            # Override symbol on all atoms if we have one from chunks
            if symbol:
                for a in atoms:
                    if not a.symbol:
                        a.symbol = symbol
                    if not a.symbols and a.symbol:
                        pass   # keep as-is

            log.info(
                f"[synthesis] decomposed → {len(atoms)} atoms | "
                f"symbol={symbol} | years={resolved_years}"
            )
        except Exception as exc:
            warnings.append(f"Decomposer failed: {exc} — falling back to vector-only")
            log.warning(f"[synthesis] decomposer exception: {exc}")
            return self._run_vector_only(
                query, chunks, resolved_years, explicit_years,
                doc_type, builder, warnings,
            )

        # Step 2 — schema bridge: SQL + vector
        bridge_result = None
        try:
            bridge = self._get_bridge()
            if bridge and atoms:
                bridge_result = bridge.fetch(atoms)
                log.info(
                    f"[synthesis] bridge → "
                    f"{sum(len(r.rows) for r in bridge_result.sql_results)} SQL rows | "
                    f"{sum(len(r.chunks) for r in bridge_result.vector_results)} vec chunks"
                )
            else:
                warnings.append("Bridge skipped (no atoms or DB unavailable)")
        except Exception as exc:
            warnings.append(f"Bridge failed: {exc}")
            log.warning(f"[synthesis] bridge exception:\n{traceback.format_exc()}")

        # Step 3 — fusion layer
        fusion_result = None
        sql_rows = 0
        insights = 0
        if bridge_result is not None:
            try:
                fusion  = self._get_fusion()
                # Merge bridge vector results with the top reranked chunks
                # so the fusion layer sees the full picture
                augmented_bridge = _merge_chunks_into_bridge(bridge_result, chunks)
                fusion_result    = fusion.fuse(augmented_bridge)
                sql_rows  = len(fusion_result.metric_rows)
                insights  = len(fusion_result.insights)
                log.info(
                    f"[synthesis] fusion → "
                    f"{sql_rows} metric rows | "
                    f"{insights} insights "
                    f"({len(fusion_result.contradictions())} contradictions)"
                )
                if fusion_result.errors:
                    warnings.extend(fusion_result.errors)
            except Exception as exc:
                warnings.append(f"Fusion failed: {exc}")
                log.warning(f"[synthesis] fusion exception:\n{traceback.format_exc()}")

        # Step 4 — build prompt
        try:
            # Use all reranked chunks (they are already the best-ranked set)
            built = builder.build(
                query          = query,
                chunks         = chunks,
                fusion_result  = fusion_result,
                doc_type       = doc_type,
                resolved_years = resolved_years,
                explicit_years = explicit_years,
            )
        except Exception as exc:
            warnings.append(f"PromptBuilder failed: {exc} — falling back to vector-only")
            log.warning(f"[synthesis] prompt builder exception:\n{traceback.format_exc()}")
            return self._run_vector_only(
                query, chunks, resolved_years, explicit_years,
                doc_type, builder, warnings,
            )

        mode = "full" if fusion_result is not None else "no_sql"
        log.info(
            f"[synthesis] mode={mode} | "
            f"sql_rows={built.sql_rows_used} chunks={built.chunks_used} "
            f"insights={built.insights_used} chars={built.total_chars:,}"
        )

        return SynthesisResult(
            built_prompt  = built,
            atoms         = atoms,
            fusion_result = fusion_result,
            sql_rows      = sql_rows,
            insights      = insights,
            chunks_used   = built.chunks_used,
            pipeline_mode = mode,
            warnings      = warnings,
        )

    # ── Vector-only path ──────────────────────────────────────────────────────

    def _run_vector_only(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        resolved_years: Optional[List[int]],
        explicit_years: Optional[List[int]],
        doc_type:       str,
        builder:        "PromptBuilder",
        warnings:       List[str],
    ) -> SynthesisResult:
        try:
            built = builder.build(
                query          = query,
                chunks         = chunks,
                fusion_result  = None,
                doc_type       = doc_type,
                resolved_years = resolved_years,
                explicit_years = explicit_years,
            )
        except Exception as exc:
            # Last resort — return a minimal prompt so the LLM call never crashes
            warnings.append(f"PromptBuilder (vector-only) failed: {exc}")
            built = _minimal_fallback_prompt(query, chunks)

        return SynthesisResult(
            built_prompt  = built,
            chunks_used   = built.chunks_used,
            pipeline_mode = "vector_only",
            warnings      = warnings,
        )

    # ── Absolute fallback (PromptBuilder not importable) ──────────────────────

    def _fallback_result(
        self,
        query:          str,
        chunks:         List["RetrievedChunk"],
        resolved_years: Optional[List[int]],
        explicit_years: Optional[List[int]],
        doc_type:       str,
        warnings:       List[str],
    ) -> SynthesisResult:
        """
        Called only when PromptBuilder itself is missing.
        Returns a SynthesisResult whose .built_prompt is a minimal stub —
        rag_engine will detect this and use its own prompt-building code.
        """
        built = _minimal_fallback_prompt(query, chunks)
        return SynthesisResult(
            built_prompt  = built,
            pipeline_mode = "vector_only",
            warnings      = warnings + ["Using rag_engine legacy prompts"],
        )

PATCH_DESCRIPTION = {
    "file":   "synthesis/pipeline.py",
    "line":   303,
    "before": "atoms: List[AtomicNeed] = decomposer.decompose(query)",
    "after":  "atoms: List[AtomicNeed] = decomposer.decompose(query, symbol=symbol)",
    "reason": "Symbol was never passed to decomposer so SQL atoms had symbol=None "
              "and the bridge returned 0 rows (wrong/no company filter).",
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _merge_chunks_into_bridge(
    bridge: "BridgeResult",
    extra_chunks: List["RetrievedChunk"],
) -> "BridgeResult":
    """
    The bridge already fetched vector chunks via its own queries.
    Augment its vector_results with the reranker's top chunks so the
    fusion layer can cross-reference both.

    We inject the extra chunks as a synthetic VectorAtomResult tagged
    with sub_type="reranker_top" so the fusion layer can handle them.
    """
    if not extra_chunks:
        return bridge

    try:
        from schema_bridge.schema_bridge import VectorAtomResult
        from decomposer.atomic_decomposer import AtomicNeed, NeedType, TimeHorizon
        stub_atom = AtomicNeed(
            need_type=NeedType.QUALITATIVE,
            sub_type="reranker_top",
            metric="Reranker Top Chunks",
        )
        synthetic = VectorAtomResult(atom=stub_atom, chunks=extra_chunks)
        # Return a new BridgeResult with the extra chunks appended
        from schema_bridge.schema_bridge import BridgeResult
        return BridgeResult(
            sql_results    = bridge.sql_results,
            vector_results = bridge.vector_results + [synthetic],
            errors         = bridge.errors,
        )
    except Exception:
        return bridge   # if anything goes wrong, don't disturb the bridge


def _minimal_fallback_prompt(
    query: str,
    chunks: List["RetrievedChunk"],
) -> "BuiltPrompt":
    """
    Emergency fallback — builds the simplest possible BuiltPrompt
    without importing PromptBuilder.  Used only when every import fails.
    """
    # Import BuiltPrompt here so this function works even if PromptBuilder
    # module is broken (syntax error etc.)
    try:
        from synthesis.prompt_builder import BuiltPrompt
    except Exception:
        # absolute last resort — create inline dataclass
        from dataclasses import dataclass as _dc
        @_dc
        class BuiltPrompt:              # type: ignore[no-redef]
            system_prompt: str; user_prompt: str
            sql_rows_used: int = 0; chunks_used: int = 0
            insights_used: int = 0; total_chars: int = 0
            was_trimmed: bool = False

    sep = "\n" + "─" * 60 + "\n"
    context = sep.join(
        f"[SRC-{i}] {c.metadata.get('symbol','')} "
        f"FY{c.metadata.get('year','')} | "
        f"{(c.metadata.get('section') or '')[:40]}\n{c.text[:800]}"
        for i, c in enumerate(chunks[:12], 1)
    )
    system = (
        "You are a senior equity research analyst for Indian listed companies. "
        "Answer using ONLY the provided context. Cite sources as [SRC-N]. "
        "Show calculations explicitly. Never hallucinate numbers."
    )
    user = (
        f"CONTEXT:\n{'='*60}\n{context}\n{'='*60}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer using ONLY the context above. Cite [SRC-N] after every number."
    )
    return BuiltPrompt(
        system_prompt=system, user_prompt=user,
        sql_rows_used=0, chunks_used=min(len(chunks), 12),
        insights_used=0, total_chars=len(system)+len(user),
        was_trimmed=(len(chunks) > 12),
    )