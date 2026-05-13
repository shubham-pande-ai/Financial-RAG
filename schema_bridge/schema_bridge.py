"""
schema_bridge/schema_bridge.py

Layer 2 of the Quant CoPilot Intent Decomposition Pipeline.

Receives a list of AtomicNeed objects from the decomposer and translates
each one into a concrete data-fetch operation:

  QUANTITATIVE / TECHNICAL / MACRO / OWNERSHIP atoms
    → parameterized SQLite SELECT (table + columns from SUBTYPE_TABLE_MAP,
      filtered by symbol, fiscal year / date window)

  QUALITATIVE atoms
    → ChromaDB vector query via retrieve_annual()

  FORWARD_LOOKING atoms
    → ChromaDB concall query via retrieve_concall()

  COMPARATIVE atoms
    → both SQL and vector, one sub-call per symbol

All channels fire in parallel using concurrent.futures.ThreadPoolExecutor
so a mixed query (EBITDA + MD&A + guidance) costs roughly max(channel_latency)
instead of sum.

Returns a BridgeResult containing:
  sql_results    — list of SqlAtomResult (atom + rows fetched)
  vector_results — list of VectorAtomResult (atom + chunks fetched)
  errors         — any channel-level failures (non-fatal)

═══════════════════════════════════════════════════════════════════════════
Design principles
═══════════════════════════════════════════════════════════════════════════
1. ZERO hallucination surface — only columns that exist in SUBTYPE_TABLE_MAP
   and the actual DB schema are ever queried.  The bridge never generates
   free-form SQL; it assembles it from a whitelist.

2. Symbol filtering — every SQL query includes WHERE symbol = ? when a
   symbol is known.  Macro/rate tables (rbi_rates, forex_commodities,
   macro_indicators, market_indices) are symbol-free by design.

3. Year → date mapping
   FY year (e.g. 2024) maps to the date range [YYYY-04-01 … (YYYY+1)-03-31]
   for Indian financial year convention (Apr–Mar).
   "current" queries use ORDER BY <date_col> DESC LIMIT 1.

4. Period type routing
   annual    → annual_results, balance_sheet (period_type='annual'), cash_flow
   quarterly → quarterly_results, balance_sheet (period_type='quarterly')
   ttm       → fundamentals (has ttm_eps, ttm_pe columns)

5. Safety — all values are passed as SQL parameters (?), never interpolated.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── project imports ──────────────────────────────────────────────────────────
# Adjust the path prefix if the bridge lives in a different package directory.
from decomposer.atomic_decomposer import (
    AtomicNeed,
    NeedType,
    TimeHorizon,
    SUBTYPE_TABLE_MAP,
)
from pipeline.retrieval.retriever import (
    RetrievedChunk,
    retrieve_annual,
    retrieve_concall,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DB path (reuse the same DB as the rest of the system)
# ─────────────────────────────────────────────────────────────────────────────
from config.settings import DB_PATH   # noqa: E402  (import after path setup)

# ─────────────────────────────────────────────────────────────────────────────
# Table metadata: which column holds the date/period, and which holds symbol
# ─────────────────────────────────────────────────────────────────────────────

# Maps table_name → (date_column, has_symbol_column, has_period_type_column)
#   date_column        — used to filter by fiscal year or ORDER BY for "current"
#   has_symbol_column  — True when the table has a `symbol` TEXT column
#   has_period_type    — True when the table has a `period_type` TEXT column
#                        ('annual' / 'quarterly')
_TABLE_META: Dict[str, Tuple[str, bool, bool]] = {
    "annual_results":          ("period_end",     True,  False),
    "quarterly_results":       ("period_end",     True,  False),
    "fundamentals":            ("as_of_date",     True,  False),
    "balance_sheet":           ("period_end",     True,  True),
    "cash_flow":               ("period_end",     True,  True),
    "growth_metrics":          ("as_of_date",     True,  False),
    "annual_cashflow_derived": ("annual_end",     True,  False),
    "technical_indicators":    ("date",           True,  False),
    "price_daily":             ("date",           True,  False),
    "price_intraday":          ("ts",             True,  False),
    "corporate_actions":       ("action_date",    True,  False),
    "ownership":               ("snapshot_date",  True,  False),
    "ownership_history":       ("period_end",     True,  False),
    "earnings_history":        ("quarter_end",    True,  False),
    "earnings_estimates":      ("snapshot_date",  True,  False),
    "eps_trend":               ("snapshot_date",  True,  False),
    "eps_revisions":           ("snapshot_date",  True,  False),
    # Macro / market tables — no symbol column
    "rbi_rates":               ("effective_date", False, False),
    "forex_commodities":       ("snapshot_date",  False, False),
    "macro_indicators":        ("snapshot_date",  False, False),
    "market_indices":          ("snapshot_date",  False, False),
}

# Columns always fetched in addition to the atom's requested columns.
# This lets the LLM know what period the data belongs to.
_ALWAYS_SELECT = ["symbol", "period_end", "as_of_date", "annual_end", "date",
                  "snapshot_date", "action_date", "effective_date",
                  "quarter_end", "ts"]

# ─────────────────────────────────────────────────────────────────────────────
# Indian FY → calendar date range helper
# ─────────────────────────────────────────────────────────────────────────────
def _fy_date_range(fy_year: int) -> Tuple[str, str]:
    """
    Indian FY: April of (fy_year-1) → March of fy_year.
    e.g. FY2024 = 2023-04-01 to 2024-03-31
    """
    start = date(fy_year - 1, 4, 1).isoformat()
    end   = date(fy_year,     3, 31).isoformat()
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SqlAtomResult:
    atom:    AtomicNeed
    rows:    List[Dict[str, Any]] = field(default_factory=list)
    sql:     str = ""           # the SQL that was executed (for debugging)
    params:  tuple = ()
    error:   Optional[str] = None


@dataclass
class VectorAtomResult:
    atom:   AtomicNeed
    chunks: List[RetrievedChunk] = field(default_factory=list)
    error:  Optional[str] = None


@dataclass
class BridgeResult:
    sql_results:    List[SqlAtomResult]    = field(default_factory=list)
    vector_results: List[VectorAtomResult] = field(default_factory=list)
    errors:         List[str]              = field(default_factory=list)

    # Convenience: all SQL rows across all atoms, each tagged with sub_type
    def all_sql_rows(self) -> List[Dict[str, Any]]:
        out = []
        for r in self.sql_results:
            for row in r.rows:
                out.append({"_sub_type": r.atom.sub_type, **row})
        return out

    # Convenience: all vector chunks across all atoms, sorted by score desc
    def all_chunks(self) -> List[RetrievedChunk]:
        chunks = [c for r in self.vector_results for c in r.chunks]
        return sorted(chunks, key=lambda c: c.score, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# SQL query builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_sql(atom: AtomicNeed) -> Tuple[str, tuple]:
    """
    Build a parameterized SELECT for one SQL-backed AtomicNeed.

    Returns (sql_string, params_tuple).
    Raises ValueError if the table or columns cannot be resolved.
    """
    table   = atom.sql_table
    columns = list(atom.sql_columns) if atom.sql_columns else []

    if not table or table.startswith("chromadb:"):
        raise ValueError(f"Atom sub_type={atom.sub_type!r} is vector-backed; "
                         f"use vector channel instead.")

    meta = _TABLE_META.get(table)
    if meta is None:
        raise ValueError(f"Unknown table {table!r} — add it to _TABLE_META.")

    date_col, has_symbol, has_period_type = meta

    # --- SELECT clause -------------------------------------------------------
    # Always include the date column + symbol (if present) + requested columns.
    select_cols: List[str] = []
    if has_symbol:
        select_cols.append("symbol")
    if date_col not in select_cols:
        select_cols.append(date_col)
    if has_period_type:
        select_cols.append("period_type")
    # Requested metric columns (deduplicated, whitelisted)
    for col in columns:
        if col not in select_cols:
            select_cols.append(col)

    select_str = ", ".join(select_cols)

    # --- WHERE clause ---------------------------------------------------------
    conditions: List[str] = []
    params:     List[Any] = []

    # Symbol filter
    symbol = atom.symbol or (atom.symbols[0] if atom.symbols else None)
    if has_symbol and symbol:
        conditions.append("symbol = ?")
        params.append(symbol.upper())

    # Period type filter (only for tables that support it)
    if has_period_type:
        period_type_val = "quarterly" if atom.period_type == "quarterly" else "annual"
        conditions.append("period_type = ?")
        params.append(period_type_val)

    # Date / year filter
    if atom.years:
        if len(atom.years) == 1:
            fy = atom.years[0]
            start, end = _fy_date_range(fy)
            conditions.append(f"{date_col} BETWEEN ? AND ?")
            params.extend([start, end])
        else:
            # Multiple years: expand to full range
            min_fy, max_fy = min(atom.years), max(atom.years)
            start, _ = _fy_date_range(min_fy)
            _,   end = _fy_date_range(max_fy)
            conditions.append(f"{date_col} BETWEEN ? AND ?")
            params.extend([start, end])

    # Corporate actions: filter by action_type when the sub_type implies it
    if table == "corporate_actions" and atom.sub_type in ("dividend", "buyback",
                                                           "bonus", "split"):
        action_type_map = {
            "dividend": "Dividend",
            "buyback":  "Buyback",
            "bonus":    "Bonus",
            "split":    "Split",
        }
        conditions.append("action_type = ?")
        params.append(action_type_map[atom.sub_type])

    # Macro: filter by indicator_name when sub_type is "macro"
    if table == "macro_indicators" and atom.sub_type == "macro":
        # The raw_text might contain the indicator name; use it if short enough
        if atom.raw_text and len(atom.raw_text) < 30:
            conditions.append("LOWER(indicator_name) LIKE ?")
            params.append(f"%{atom.raw_text.lower()}%")

    where_str = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # --- ORDER BY / LIMIT ----------------------------------------------------
    # For current / no-year queries: newest row first, cap at 5 rows.
    # For historical queries with years: order newest first, no cap (user asked for all).
    if atom.time_horizon == TimeHorizon.CURRENT and not atom.years:
        order_limit = f"ORDER BY {date_col} DESC LIMIT 5"
    elif atom.years:
        order_limit = f"ORDER BY {date_col} DESC"
    else:
        order_limit = f"ORDER BY {date_col} DESC LIMIT 10"

    sql = f"SELECT {select_str} FROM {table} {where_str} {order_limit}".strip()
    return sql, tuple(params)


# ─────────────────────────────────────────────────────────────────────────────
# SQL executor
# ─────────────────────────────────────────────────────────────────────────────
def _execute_sql_atom(atom: AtomicNeed, db_path: Path) -> SqlAtomResult:
    """Run the SQL for one atom, return SqlAtomResult (never raises)."""
    try:
        sql, params = _build_sql(atom)
    except ValueError as e:
        log.warning(f"  [bridge] SQL build failed for {atom.sub_type}: {e}")
        return SqlAtomResult(atom=atom, error=str(e))

    log.debug(f"  [bridge] SQL: {sql} | params={params}")

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql, params)
            rows = [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

        log.info(f"  [bridge] {atom.sub_type}: {len(rows)} row(s) from {atom.sql_table}")
        return SqlAtomResult(atom=atom, rows=rows, sql=sql, params=params)

    except sqlite3.Error as e:
        msg = f"SQLite error for {atom.sub_type} ({atom.sql_table}): {e}"
        log.error(f"  [bridge] {msg}")
        return SqlAtomResult(atom=atom, sql=sql, params=params, error=msg)


# ─────────────────────────────────────────────────────────────────────────────
# Vector executor
# ─────────────────────────────────────────────────────────────────────────────
def _execute_vector_atom(atom: AtomicNeed) -> VectorAtomResult:
    """Run a ChromaDB vector query for one atom, return VectorAtomResult."""
    query  = atom.metric        # human-readable label is a good base query
    raw    = atom.raw_text or ""
    if raw and raw.lower() not in query.lower():
        query = f"{query} {raw}"   # enrich with original phrasing

    symbol = atom.symbol or (atom.symbols[0] if atom.symbols else None)
    years  = atom.years or None

    try:
        if atom.need_type == NeedType.FORWARD_LOOKING or \
                (atom.sql_table or "").startswith("chromadb:concalls"):
            chunks = retrieve_concall(query, symbol=symbol, years=years)
        else:
            # QUALITATIVE or fallback
            chunks = retrieve_annual(query, symbol=symbol, years=years)

        log.info(f"  [bridge] {atom.sub_type}: {len(chunks)} chunk(s) from vector store")
        return VectorAtomResult(atom=atom, chunks=chunks)

    except Exception as e:
        msg = f"Vector query failed for {atom.sub_type}: {e}"
        log.error(f"  [bridge] {msg}")
        return VectorAtomResult(atom=atom, error=msg)


# ─────────────────────────────────────────────────────────────────────────────
# Comparative expansion
# ─────────────────────────────────────────────────────────────────────────────
def _expand_comparative(atom: AtomicNeed) -> List[AtomicNeed]:
    """
    A COMPARATIVE atom targets multiple symbols.
    Expand it into one atom per symbol, each with the same sub_type but
    a single symbol.  The original COMPARATIVE atom is discarded.
    """
    symbols = atom.symbols if atom.symbols else ([atom.symbol] if atom.symbol else [])
    if not symbols:
        return [atom]

    expanded = []
    for sym in symbols:
        clone = AtomicNeed(
            need_type    = NeedType.QUANTITATIVE,   # route to SQL by default
            sub_type     = atom.sub_type,
            metric       = atom.metric,
            symbol       = sym,
            symbols      = [],
            years        = atom.years,
            time_horizon = atom.time_horizon,
            period_type  = atom.period_type,
            raw_text     = atom.raw_text,
            confidence   = atom.confidence,
            source       = atom.source,
        )
        clone.resolve_schema()
        expanded.append(clone)
    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# Channel classifier
# ─────────────────────────────────────────────────────────────────────────────
_SQL_NEED_TYPES = {
    NeedType.QUANTITATIVE,
    NeedType.TECHNICAL,
    NeedType.MACRO,
    NeedType.OWNERSHIP,
}
_VECTOR_NEED_TYPES = {
    NeedType.QUALITATIVE,
    NeedType.FORWARD_LOOKING,
}


def _classify(atom: AtomicNeed) -> str:
    """Return 'sql', 'vector', or 'both'."""
    if atom.need_type in _SQL_NEED_TYPES:
        # Some sub_types are actually vector-backed (chromadb:*)
        if (atom.sql_table or "").startswith("chromadb:"):
            return "vector"
        return "sql"
    if atom.need_type in _VECTOR_NEED_TYPES:
        return "vector"
    if atom.need_type == NeedType.COMPARATIVE:
        return "both"
    return "sql"   # safe default


# ─────────────────────────────────────────────────────────────────────────────
# Public API: SchemaBridge
# ─────────────────────────────────────────────────────────────────────────────
class SchemaBridge:
    """
    Translates a list of AtomicNeed objects into concrete data-fetch results.

    Usage:
        bridge = SchemaBridge()
        result = bridge.fetch(atoms)

        # All SQL rows across all atoms
        for row in result.all_sql_rows():
            print(row)

        # Vector chunks sorted by relevance
        for chunk in result.all_chunks():
            print(chunk.text[:120])

    Thread safety: each call to fetch() opens its own DB connection(s).
    The bridge object itself is stateless and safe to share.
    """

    def __init__(
        self,
        db_path:    Optional[Path] = None,
        max_workers: int = 8,
    ):
        self.db_path    = db_path or DB_PATH
        self.max_workers = max_workers

    # ── Main entry point ──────────────────────────────────────────────────────

    def fetch(self, atoms: List[AtomicNeed]) -> BridgeResult:
        """
        Dispatch all atoms to the appropriate channel(s), running them in
        parallel.  Returns a BridgeResult regardless of partial failures.
        """
        if not atoms:
            return BridgeResult()

        # Step 1: expand comparative atoms into per-symbol atoms
        expanded: List[AtomicNeed] = []
        for atom in atoms:
            if atom.need_type == NeedType.COMPARATIVE:
                expanded.extend(_expand_comparative(atom))
            else:
                expanded.append(atom)

        # Step 2: classify each atom
        sql_atoms:    List[AtomicNeed] = []
        vector_atoms: List[AtomicNeed] = []
        for atom in expanded:
            channel = _classify(atom)
            if channel in ("sql", "both"):
                sql_atoms.append(atom)
            if channel in ("vector", "both"):
                vector_atoms.append(atom)

        log.info(
            f"[bridge] Dispatching {len(sql_atoms)} SQL atom(s) + "
            f"{len(vector_atoms)} vector atom(s) in parallel"
        )

        # Step 3: fire all tasks in parallel
        sql_results:    List[SqlAtomResult]    = []
        vector_results: List[VectorAtomResult] = []
        errors:         List[str]              = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}

            for atom in sql_atoms:
                f = pool.submit(self._safe_sql, atom)
                futures[f] = ("sql", atom)

            for atom in vector_atoms:
                f = pool.submit(self._safe_vector, atom)
                futures[f] = ("vec", atom)

            for fut in as_completed(futures):
                kind, atom = futures[fut]
                try:
                    result = fut.result()
                    if kind == "sql":
                        sql_results.append(result)
                    else:
                        vector_results.append(result)
                    if result.error:
                        errors.append(result.error)
                except Exception as e:
                    msg = f"Unexpected failure in bridge ({atom.sub_type}): {e}"
                    log.error(f"  [bridge] {msg}")
                    errors.append(msg)

        log.info(
            f"[bridge] Done — {sum(len(r.rows) for r in sql_results)} SQL row(s), "
            f"{sum(len(r.chunks) for r in vector_results)} vector chunk(s), "
            f"{len(errors)} error(s)"
        )
        return BridgeResult(
            sql_results    = sql_results,
            vector_results = vector_results,
            errors         = errors,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _safe_sql(self, atom: AtomicNeed) -> SqlAtomResult:
        return _execute_sql_atom(atom, self.db_path)

    def _safe_vector(self, atom: AtomicNeed) -> VectorAtomResult:
        return _execute_vector_atom(atom)

    # ── Convenience: fetch for a single symbol across multiple sub_types ──────

    def fetch_symbol(
        self,
        symbol:    str,
        sub_types: List[str],
        years:     Optional[List[int]] = None,
    ) -> BridgeResult:
        """
        Helper for the common case: fetch several metrics for one company.

        Example:
            result = bridge.fetch_symbol("ADANIPORTS", ["revenue","net_debt","roce"])
        """
        atoms = []
        for st in sub_types:
            entry = SUBTYPE_TABLE_MAP.get(st)
            if not entry:
                log.warning(f"  [bridge] Unknown sub_type {st!r} — skipping")
                continue
            atom = AtomicNeed(
                need_type   = NeedType.QUANTITATIVE,
                sub_type    = st,
                metric      = st,
                symbol      = symbol.upper(),
                years       = years or [],
                time_horizon= TimeHorizon.HISTORICAL if years else TimeHorizon.CURRENT,
                period_type = "annual",
            )
            atom.resolve_schema()
            atoms.append(atom)
        return self.fetch(atoms)