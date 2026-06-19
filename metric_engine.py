"""
metric_engine.py
Fresh financial ratio computation from MySQL (Quant Copilot ETL schema).

No agent, no MCP.  Plain functions that query MySQL and return computed ratios.
Call these from the RAG answer-generation step or expose via a FastAPI route.

Ratios computed:
  Valuation   : PE, Forward PE, PB, PS, EV/EBITDA, EV/Sales, Market Cap
  Profitability: EBITDA Margin, PAT Margin, ROE, ROCE, Asset Turnover
  Leverage    : Debt/Equity, Net Debt/EBITDA, Interest Coverage
  Growth      : Revenue YoY, PAT YoY, EPS YoY (TTM vs prior year)

Table name assumptions (Quant Copilot ETL schema — update if yours differ):
  sm_price_data          — daily OHLCV + market cap
  sm_profit_loss         — annual/quarterly P&L line items
  sm_balance_sheet       — annual balance sheet line items
  sm_key_ratios          — pre-computed ratios from Screener (used as fallback)

All functions return a dict.  Missing data keys are omitted rather than
returning NaN/None so callers can check `"pe" in ratios` safely.

Usage:
    from metric_engine import get_ratios, get_valuation, get_profitability

    ratios = get_ratios("RELIANCE", year=2024)
    print(ratios["pe"], ratios["ev_ebitda"])
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import mysql.connector
from mysql.connector import pooling

from config.settings import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_POOL_SIZE
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Connection pool (separate from db/database.py pool to stay independent)
# ─────────────────────────────────────────────
_pool: Optional[pooling.MySQLConnectionPool] = None


def _get_pool() -> pooling.MySQLConnectionPool:
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name  = "metric_engine",
            pool_size  = min(DB_POOL_SIZE, 3),
            host       = DB_HOST,
            port       = DB_PORT,
            database   = DB_NAME,
            user       = DB_USER,
            password   = DB_PASSWORD,
            charset    = "utf8mb4",
            autocommit = True,
        )
    return _pool


@contextmanager
def _conn():
    conn = _get_pool().get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _q(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return list of dicts."""
    with _conn() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows


def _scalar(sql: str, params: tuple = ()) -> Optional[float]:
    """Return first column of first row as float, or None."""
    rows = _q(sql, params)
    if not rows:
        return None
    val = list(rows[0].values())[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# Low-level data fetchers
# Adjust column names to match your actual ETL schema.
# ─────────────────────────────────────────────

def _latest_price(symbol: str) -> Optional[float]:
    return _scalar(
        "SELECT close FROM sm_price_data WHERE symbol=%s ORDER BY date DESC LIMIT 1",
        (symbol.upper(),),
    )


def _market_cap(symbol: str) -> Optional[float]:
    """Market cap in Crore INR."""
    return _scalar(
        "SELECT market_cap FROM sm_price_data WHERE symbol=%s ORDER BY date DESC LIMIT 1",
        (symbol.upper(),),
    )


def _pl_annual(symbol: str, year: int) -> dict:
    """
    Fetch annual P&L for a given fiscal year.
    Expects columns: revenue, ebitda, depreciation, ebit, interest,
                     pbt, tax, pat, eps
    Returns dict (empty if not found).
    """
    rows = _q(
        """
        SELECT revenue, ebitda, depreciation, ebit, interest,
               pbt, tax, pat, eps
        FROM   sm_profit_loss
        WHERE  symbol=%s AND fiscal_year=%s AND period_type='annual'
        LIMIT  1
        """,
        (symbol.upper(), year),
    )
    return rows[0] if rows else {}


def _bs_annual(symbol: str, year: int) -> dict:
    """
    Fetch annual balance sheet.
    Expects columns: total_assets, total_equity, total_debt,
                     cash_equivalents, book_value_per_share
    """
    rows = _q(
        """
        SELECT total_assets, total_equity, total_debt,
               cash_equivalents, book_value_per_share
        FROM   sm_balance_sheet
        WHERE  symbol=%s AND fiscal_year=%s AND period_type='annual'
        LIMIT  1
        """,
        (symbol.upper(), year),
    )
    return rows[0] if rows else {}


def _ttm_eps(symbol: str) -> Optional[float]:
    """Sum of last 4 quarters' EPS."""
    return _scalar(
        """
        SELECT SUM(eps) FROM (
            SELECT eps FROM sm_profit_loss
            WHERE  symbol=%s AND period_type='quarterly'
            ORDER  BY fiscal_year DESC, quarter DESC
            LIMIT  4
        ) t
        """,
        (symbol.upper(),),
    )


def _forward_eps(symbol: str) -> Optional[float]:
    """
    Analyst consensus forward EPS stored by the ETL pipeline.
    Falls back to None if the table / column doesn't exist.
    """
    try:
        return _scalar(
            """
            SELECT forward_eps FROM sm_key_ratios
            WHERE  symbol=%s
            ORDER  BY updated_at DESC
            LIMIT  1
            """,
            (symbol.upper(),),
        )
    except Exception:
        return None


def _shares_outstanding(symbol: str) -> Optional[float]:
    """Shares outstanding in Crore."""
    try:
        return _scalar(
            """
            SELECT shares_outstanding FROM sm_key_ratios
            WHERE  symbol=%s
            ORDER  BY updated_at DESC LIMIT 1
            """,
            (symbol.upper(),),
        )
    except Exception:
        return None


# ─────────────────────────────────────────────
# Ratio computation helpers
# ─────────────────────────────────────────────

def _safe_div(num, den) -> Optional[float]:
    try:
        if den is None or den == 0:
            return None
        return round(float(num) / float(den), 2)
    except (TypeError, ValueError):
        return None


def _pct(val, base) -> Optional[float]:
    r = _safe_div(val, base)
    return round(r * 100, 2) if r is not None else None


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_valuation(symbol: str, year: int = None) -> dict:
    """
    Compute valuation multiples.
    year defaults to the latest available fiscal year.

    Returns dict with keys (only present if computable):
      pe, forward_pe, pb, ps, ev_ebitda, ev_sales,
      market_cap_cr, enterprise_value_cr, price
    """
    if year is None:
        year = _latest_fiscal_year(symbol) or 2024

    price = _latest_price(symbol)
    mcap  = _market_cap(symbol)
    pl    = _pl_annual(symbol, year)
    bs    = _bs_annual(symbol, year)

    out: dict = {}

    if price is not None:
        out["price"] = round(price, 2)

    if mcap is not None:
        out["market_cap_cr"] = round(mcap, 2)

    # PE (TTM)
    ttm_eps = _ttm_eps(symbol)
    if price and ttm_eps and ttm_eps > 0:
        out["pe"] = round(price / ttm_eps, 2)
    elif pl.get("eps") and pl["eps"] > 0 and price:
        out["pe"] = round(price / float(pl["eps"]), 2)

    # Forward PE
    fwd_eps = _forward_eps(symbol)
    if price and fwd_eps and fwd_eps > 0:
        out["forward_pe"] = round(price / fwd_eps, 2)

    # PB
    bvps = bs.get("book_value_per_share")
    if price and bvps and float(bvps) > 0:
        out["pb"] = round(price / float(bvps), 2)

    # PS
    shares = _shares_outstanding(symbol)
    if mcap and pl.get("revenue") and float(pl["revenue"]) > 0:
        out["ps"] = _safe_div(mcap, float(pl["revenue"]))

    # Enterprise Value = market_cap + total_debt - cash
    debt = bs.get("total_debt")
    cash = bs.get("cash_equivalents")
    if mcap is not None and debt is not None and cash is not None:
        ev = float(mcap) + float(debt) - float(cash)
        out["enterprise_value_cr"] = round(ev, 2)

        # EV/EBITDA
        ebitda = pl.get("ebitda")
        if ebitda and float(ebitda) > 0:
            out["ev_ebitda"] = round(ev / float(ebitda), 2)

        # EV/Sales
        revenue = pl.get("revenue")
        if revenue and float(revenue) > 0:
            out["ev_sales"] = round(ev / float(revenue), 2)

    return out


def get_profitability(symbol: str, year: int = None) -> dict:
    """
    Compute profitability and return ratios.

    Returns dict with keys (only present if computable):
      ebitda_margin_pct, pat_margin_pct, roe_pct, roce_pct,
      asset_turnover, ebitda_cr, pat_cr, revenue_cr
    """
    if year is None:
        year = _latest_fiscal_year(symbol) or 2024

    pl = _pl_annual(symbol, year)
    bs = _bs_annual(symbol, year)
    out: dict = {}

    revenue = pl.get("revenue")
    ebitda  = pl.get("ebitda")
    pat     = pl.get("pat")

    if revenue:
        out["revenue_cr"] = round(float(revenue), 2)
    if ebitda:
        out["ebitda_cr"]  = round(float(ebitda), 2)
    if pat:
        out["pat_cr"]     = round(float(pat), 2)

    if revenue and float(revenue) > 0:
        if ebitda:
            out["ebitda_margin_pct"] = _pct(float(ebitda), float(revenue))
        if pat:
            out["pat_margin_pct"]    = _pct(float(pat),    float(revenue))

    equity = bs.get("total_equity")
    assets = bs.get("total_assets")
    debt   = bs.get("total_debt")

    if pat and equity and float(equity) > 0:
        out["roe_pct"] = _pct(float(pat), float(equity))

    # ROCE = EBIT / Capital Employed  (Capital Employed = assets - current liabilities)
    # Approximation: assets - (assets - equity - debt) when current liabilities unavailable
    ebit = pl.get("ebit")
    if ebit and equity is not None and debt is not None:
        capital_employed = float(equity) + float(debt)
        if capital_employed > 0:
            out["roce_pct"] = _pct(float(ebit), capital_employed)

    if revenue and assets and float(assets) > 0:
        out["asset_turnover"] = _safe_div(float(revenue), float(assets))

    return out


def get_leverage(symbol: str, year: int = None) -> dict:
    """
    Compute leverage ratios.

    Returns dict with keys:
      debt_equity, net_debt_ebitda, interest_coverage, net_debt_cr
    """
    if year is None:
        year = _latest_fiscal_year(symbol) or 2024

    pl  = _pl_annual(symbol, year)
    bs  = _bs_annual(symbol, year)
    out: dict = {}

    debt   = bs.get("total_debt")
    equity = bs.get("total_equity")
    cash   = bs.get("cash_equivalents")
    ebitda = pl.get("ebitda")
    ebit   = pl.get("ebit")
    interest = pl.get("interest")

    if debt and equity and float(equity) > 0:
        out["debt_equity"] = _safe_div(float(debt), float(equity))

    if debt is not None and cash is not None:
        net_debt = float(debt) - float(cash)
        out["net_debt_cr"] = round(net_debt, 2)
        if ebitda and float(ebitda) > 0:
            out["net_debt_ebitda"] = _safe_div(net_debt, float(ebitda))

    if ebit and interest and float(interest) > 0:
        out["interest_coverage"] = _safe_div(float(ebit), float(interest))

    return out


def get_growth(symbol: str, year: int = None) -> dict:
    """
    Compute YoY growth rates comparing year vs year-1.

    Returns dict with keys:
      revenue_yoy_pct, pat_yoy_pct, ebitda_yoy_pct, eps_yoy_pct
    """
    if year is None:
        year = _latest_fiscal_year(symbol) or 2024

    cur  = _pl_annual(symbol, year)
    prev = _pl_annual(symbol, year - 1)
    out: dict = {}

    def _yoy(key: str) -> Optional[float]:
        c = cur.get(key)
        p = prev.get(key)
        if c is None or p is None or float(p) == 0:
            return None
        return round((float(c) - float(p)) / abs(float(p)) * 100, 2)

    for key, label in [
        ("revenue", "revenue_yoy_pct"),
        ("pat",     "pat_yoy_pct"),
        ("ebitda",  "ebitda_yoy_pct"),
        ("eps",     "eps_yoy_pct"),
    ]:
        v = _yoy(key)
        if v is not None:
            out[label] = v

    return out


def get_ratios(symbol: str, year: int = None) -> dict:
    """
    All ratios in one call.  Keys are flat — no nesting.

    Example:
        {
          "price": 2850.0,
          "market_cap_cr": 385000.0,
          "pe": 28.4,
          "forward_pe": 24.1,
          "pb": 2.1,
          "ev_ebitda": 14.2,
          "ebitda_margin_pct": 31.5,
          "pat_margin_pct": 11.2,
          "roe_pct": 15.3,
          "debt_equity": 0.42,
          "revenue_yoy_pct": 8.4,
          ...
        }
    """
    if year is None:
        year = _latest_fiscal_year(symbol) or 2024

    out = {}
    out.update(get_valuation(symbol,   year))
    out.update(get_profitability(symbol, year))
    out.update(get_leverage(symbol,    year))
    out.update(get_growth(symbol,      year))
    out["symbol"] = symbol.upper()
    out["fiscal_year"] = year
    return out


def get_ratios_multi_year(symbol: str, years: list[int]) -> list[dict]:
    """
    Return a list of ratio dicts for multiple fiscal years.
    Useful for trend analysis in the RAG answer step.
    """
    return [get_ratios(symbol, y) for y in years]


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def _latest_fiscal_year(symbol: str) -> Optional[int]:
    """Highest fiscal year with P&L data for this symbol."""
    return _scalar(
        """
        SELECT MAX(fiscal_year) FROM sm_profit_loss
        WHERE  symbol=%s AND period_type='annual'
        """,
        (symbol.upper(),),
    )


def available_years(symbol: str) -> list[int]:
    """All fiscal years with annual P&L data, descending."""
    rows = _q(
        """
        SELECT DISTINCT fiscal_year FROM sm_profit_loss
        WHERE  symbol=%s AND period_type='annual'
        ORDER  BY fiscal_year DESC
        """,
        (symbol.upper(),),
    )
    return [int(r["fiscal_year"]) for r in rows]


def get_snapshot(symbol: str) -> dict:
    """
    Quick one-liner summary dict — useful for LLM context injection.
    Returns latest ratios + 3-year revenue/PAT trend.
    """
    latest_year = _latest_fiscal_year(symbol)
    if latest_year is None:
        return {"symbol": symbol.upper(), "error": "no data"}

    latest  = get_ratios(symbol, latest_year)
    trend   = get_ratios_multi_year(
        symbol,
        list(range(max(latest_year - 2, 2018), latest_year + 1)),
    )

    return {
        "symbol":      symbol.upper(),
        "fiscal_year": latest_year,
        "latest":      latest,
        "trend":       trend,
    }