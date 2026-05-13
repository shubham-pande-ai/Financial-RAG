"""
decomposer/atomic_decomposer.py

Layer 1 of the Quant CoPilot Intent Decomposition Pipeline.

Receives a raw user query and decomposes it into a list of typed
AtomicNeed objects.  Each atom carries:
  - need_type      : which retrieval channel handles it
  - sub_type       : finer classification used by schema bridge
  - metric         : the specific metric/field being requested
  - symbol         : company ticker (if determinable from query)
  - years          : fiscal years explicitly mentioned
  - time_horizon   : "historical" | "current" | "forward_looking"
  - raw_text       : the slice of the query this atom came from

Need types map directly to retrieval channels:
  QUANTITATIVE   → SQL channel  (SQLite: fundamentals, annual_results,
                                  quarterly_results, balance_sheet,
                                  cash_flow, growth_metrics, technical_indicators)
  QUALITATIVE    → Vector channel (ChromaDB: annual report prose, MD&A)
  FORWARD_LOOKING→ Events/Concall channel (ChromaDB: concall transcripts)
  COMPARATIVE    → Multi-channel (SQL + Vector, same metric, multiple symbols)
  TECHNICAL      → SQL channel  (technical_indicators table)
  MACRO          → SQL channel  (rbi_rates, macro_indicators, forex_commodities)
  OWNERSHIP      → SQL channel  (ownership, ownership_history)

The decomposer uses a two-stage approach:
  Stage 1 — Rule-based fast path: pattern matching catches ~70% of queries
             with zero latency and zero API cost.
  Stage 2 — LLM fallback: for ambiguous / compound queries, a fast small
             LLM call (Groq llama3-8b or any OpenAI-compat endpoint)
             parses the remainder.

Both stages produce the same AtomicNeed schema so the schema bridge
(Layer 3) is unaware of which path produced an atom.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class NeedType(str, Enum):
    QUANTITATIVE    = "quantitative"     # hard numbers → SQL
    QUALITATIVE     = "qualitative"      # prose / narrative → Vector
    FORWARD_LOOKING = "forward_looking"  # guidance / outlook → Concall
    COMPARATIVE     = "comparative"      # company A vs B → multi-channel
    TECHNICAL       = "technical"        # RSI, MACD, SMA → SQL technical
    MACRO           = "macro"            # RBI, GDP, forex → SQL macro
    OWNERSHIP       = "ownership"        # promoter / FII / DII → SQL


class TimeHorizon(str, Enum):
    HISTORICAL      = "historical"
    CURRENT         = "current"
    FORWARD_LOOKING = "forward_looking"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-type catalogue  (used by schema bridge to pick the right SQL table/column)
# ─────────────────────────────────────────────────────────────────────────────

# Maps sub_type → (primary_table, [relevant_columns])
SUBTYPE_TABLE_MAP: Dict[str, tuple] = {
    # ── Income statement ─────────────────────────────────────────────────────
    "revenue":          ("annual_results",    ["sales"]),
    "revenue_q":        ("quarterly_results", ["sales"]),
    "operating_profit": ("annual_results",    ["operating_profit", "opm_pct"]),
    "net_profit":       ("annual_results",    ["net_profit", "eps"]),
    "net_profit_q":     ("quarterly_results", ["net_profit", "eps"]),
    "ebitda":           ("fundamentals",      ["ebitda", "ebitda_margin_pct"]),
    "ebit":             ("fundamentals",      ["ebit_margin_pct"]),
    "depreciation":     ("annual_results",    ["depreciation"]),
    "interest":         ("annual_results",    ["interest"]),
    "pbt":              ("annual_results",    ["profit_before_tax"]),
    "opm":              ("annual_results",    ["opm_pct"]),
    "eps":              ("fundamentals",      ["eps_annual", "ttm_eps"]),
    "eps_q":            ("quarterly_results", ["eps"]),
    "tax":              ("annual_results",    ["tax_pct"]),

    # ── Balance sheet ─────────────────────────────────────────────────────────
    "total_assets":     ("balance_sheet",     ["total_assets"]),
    "total_equity":     ("balance_sheet",     ["total_equity", "equity_capital", "reserves"]),
    "net_worth":        ("balance_sheet",     ["total_equity"]),
    "borrowings":       ("balance_sheet",     ["borrowings", "lt_borrowings", "st_borrowings"]),
    "net_debt":         ("balance_sheet",     ["net_debt"]),
    "cash":             ("balance_sheet",     ["cash_equivalents"]),
    "trade_receivables":("balance_sheet",     ["trade_receivables"]),
    "inventories":      ("balance_sheet",     ["inventories"]),
    "cwip":             ("balance_sheet",     ["cwip"]),
    "fixed_assets":     ("balance_sheet",     ["fixed_assets"]),
    "investments":      ("balance_sheet",     ["investments"]),

    # ── Cash flow ─────────────────────────────────────────────────────────────
    "ocf":              ("cash_flow",         ["cfo"]),
    "cfi":              ("cash_flow",         ["cfi"]),
    "cff":              ("cash_flow",         ["cff"]),
    "capex":            ("cash_flow",         ["capex"]),
    "fcf":              ("cash_flow",         ["free_cash_flow"]),
    "net_cashflow":     ("cash_flow",         ["net_cash_flow"]),
    "fcf_margin":       ("annual_cashflow_derived", ["fcf_margin_pct", "approx_fcf"]),

    # ── Ratios & valuation ────────────────────────────────────────────────────
    "roe":              ("fundamentals",      ["roe_pct"]),
    "roce":             ("fundamentals",      ["roce_pct"]),
    "roa":              ("fundamentals",      ["roa_pct"]),
    "pe":               ("fundamentals",      ["pe_ratio", "ttm_pe", "forward_pe"]),
    "pb":               ("fundamentals",      ["pb_ratio"]),
    "ev_ebitda":        ("fundamentals",      ["ev_ebitda"]),
    "ev_revenue":       ("fundamentals",      ["ev_revenue"]),
    "book_value":       ("fundamentals",      ["book_value"]),
    "graham_number":    ("fundamentals",      ["graham_number"]),
    "dividend_yield":   ("fundamentals",      ["dividend_yield_pct"]),
    "dividend_payout":  ("fundamentals",      ["dividend_payout_pct"]),
    "debt_equity":      ("fundamentals",      ["debt_to_equity"]),
    "current_ratio":    ("fundamentals",      ["current_ratio"]),
    "quick_ratio":      ("fundamentals",      ["quick_ratio"]),
    "interest_coverage":("fundamentals",      ["interest_coverage"]),
    "market_cap":       ("fundamentals",      ["market_cap"]),
    "ev":               ("fundamentals",      ["ev"]),
    "gross_margin":     ("fundamentals",      ["gross_margin_pct"]),
    "net_margin":       ("fundamentals",      ["net_profit_margin_pct"]),
    "ebitda_margin":    ("fundamentals",      ["ebitda_margin_pct"]),

    # ── Working capital ───────────────────────────────────────────────────────
    "dso":              ("fundamentals",      ["dso_days"]),
    "dio":              ("fundamentals",      ["dio_days"]),
    "dpo":              ("fundamentals",      ["dpo_days"]),
    "ccc":              ("fundamentals",      ["cash_conversion_cycle"]),
    "working_capital":  ("fundamentals",      ["working_capital_days"]),

    # ── Growth metrics ─────────────────────────────────────────────────────────
    "revenue_cagr":     ("growth_metrics",    ["sales_cagr_3y", "sales_cagr_5y", "sales_cagr_10y"]),
    "profit_cagr":      ("growth_metrics",    ["profit_cagr_3y", "profit_cagr_5y", "profit_cagr_10y"]),
    "eps_cagr":         ("growth_metrics",    ["eps_cagr_3y"]),
    "ebitda_cagr":      ("growth_metrics",    ["ebitda_cagr_3y"]),
    "fcf_cagr":         ("growth_metrics",    ["fcf_cagr_3y"]),
    "stock_cagr":       ("growth_metrics",    ["stock_cagr_3y", "stock_cagr_5y", "stock_cagr_10y"]),
    "roe_avg":          ("growth_metrics",    ["roe_3y", "roe_5y", "roe_10y"]),

    # ── Price & technicals ────────────────────────────────────────────────────
    "price":            ("price_daily",       ["close", "adj_close"]),
    "price_intraday":   ("price_intraday",    ["close", "open", "high", "low"]),
    "52w_high":         ("fundamentals",      ["high_52w"]),
    "52w_low":          ("fundamentals",      ["low_52w"]),
    "rsi":              ("technical_indicators", ["rsi_14"]),
    "macd":             ("technical_indicators", ["macd", "macd_signal", "macd_hist"]),
    "sma":              ("technical_indicators", ["sma_50", "sma_200"]),
    "ema":              ("technical_indicators", ["ema_21"]),
    "bb":               ("technical_indicators", ["bb_upper", "bb_mid", "bb_lower"]),
    "atr":              ("technical_indicators", ["atr_14"]),
    "adx":              ("technical_indicators", ["adx_14"]),
    "supertrend":       ("technical_indicators", ["supertrend", "supertrend_dir"]),
    "vwap":             ("technical_indicators", ["vwap_14"]),
    "obv":              ("technical_indicators", ["obv"]),

    # ── Macro / rates ──────────────────────────────────────────────────────────
    "repo_rate":        ("rbi_rates",         ["repo_rate"]),
    "rbi_policy":       ("rbi_rates",         ["repo_rate", "reverse_repo", "crr", "slr"]),
    "forex":            ("forex_commodities", ["last_price", "change_pct"]),
    "macro":            ("macro_indicators",  ["value", "unit"]),
    "market_index":     ("market_indices",    ["last_price", "change_pct"]),

    # ── Ownership ─────────────────────────────────────────────────────────────
    "promoter":         ("ownership_history", ["promoter_pct"]),
    "fii":              ("ownership",         ["fii_fpi_pct", "fii_net_buy_cr"]),
    "dii":              ("ownership",         ["dii_pct", "dii_net_buy_cr"]),
    "institutional":    ("ownership",         ["total_institutional_pct"]),
    "shareholders":     ("ownership",         ["num_shareholders"]),

    # ── Corporate actions ──────────────────────────────────────────────────────
    "dividend":         ("corporate_actions", ["value"]),
    "buyback":          ("corporate_actions", ["value", "notes"]),
    "split":            ("corporate_actions", ["value", "notes"]),
    "bonus":            ("corporate_actions", ["value", "notes"]),

    # ── Earnings estimates ─────────────────────────────────────────────────────
    "eps_estimate":     ("earnings_estimates", ["avg_eps", "growth_pct"]),
    "eps_surprise":     ("earnings_history",   ["eps_actual", "eps_estimate", "surprise_pct"]),
    "eps_revision":     ("eps_revisions",      ["up_last_7d", "down_last_7d"]),

    # ── Qualitative (vector) ───────────────────────────────────────────────────
    "mda":              ("chromadb:annual_reports", []),
    "risk_factors":     ("chromadb:annual_reports", []),
    "strategy":         ("chromadb:annual_reports", []),
    "business_overview":("chromadb:annual_reports", []),
    "concall_guidance": ("chromadb:concalls",       []),
    "concall_mgmt":     ("chromadb:concalls",       []),
    "concall_outlook":  ("chromadb:concalls",       []),
    "concall_capex":    ("chromadb:concalls",       []),
    "concall_margin":   ("chromadb:concalls",       []),
}


# ─────────────────────────────────────────────────────────────────────────────
# AtomicNeed dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AtomicNeed:
    """One atomic information need extracted from the user query."""

    need_type:     NeedType
    sub_type:      str                    # key into SUBTYPE_TABLE_MAP
    metric:        str                    # human-readable label
    symbol:        Optional[str]  = None  # e.g. "ADANIPORTS"
    symbols:       List[str]      = field(default_factory=list)  # comparative
    years:         List[int]      = field(default_factory=list)  # explicit FY years
    time_horizon:  TimeHorizon    = TimeHorizon.CURRENT
    period_type:   str            = "annual"   # "annual" | "quarterly" | "ttm"
    raw_text:      str            = ""         # originating query fragment
    confidence:    float          = 1.0        # 0-1, used by fusion layer
    source:        str            = "rule"     # "rule" | "llm"

    # Resolved at schema-bridge time (filled in by bridge, not decomposer)
    sql_table:     Optional[str]  = None
    sql_columns:   List[str]      = field(default_factory=list)

    def resolve_schema(self):
        """Populate sql_table and sql_columns from SUBTYPE_TABLE_MAP."""
        entry = SUBTYPE_TABLE_MAP.get(self.sub_type)
        if entry:
            self.sql_table, self.sql_columns = entry

    def to_dict(self) -> dict:
        d = asdict(self)
        d["need_type"]    = self.need_type.value
        d["time_horizon"] = self.time_horizon.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based pattern library
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (regex, need_type, sub_type, metric, time_hint)
# time_hint: "historical" | "current" | "forward" | None (→ inferred later)

_RULES: List[tuple] = [
    # ── Revenue / Sales ────────────────────────────────────────────────────────
    (r"\b(revenue|sales|topline|top[\s\-]line|income from operations)\b",
     NeedType.QUANTITATIVE, "revenue", "Revenue", None),
    (r"\bquarterly\s+(revenue|sales)\b",
     NeedType.QUANTITATIVE, "revenue_q", "Quarterly Revenue", None),

    # ── Profit ────────────────────────────────────────────────────────────────
    (r"\b(net\s+profit|pat|profit\s+after\s+tax|net\s+income|bottom[\s\-]?line)\b",
     NeedType.QUANTITATIVE, "net_profit", "Net Profit", None),
    (r"\b(quarterly\s+profit|quarterly\s+pat|quarterly\s+net\s+profit)\b",
     NeedType.QUANTITATIVE, "net_profit_q", "Quarterly Net Profit", None),
    (r"\b(pbt|profit\s+before\s+tax)\b",
     NeedType.QUANTITATIVE, "pbt", "PBT", None),
    (r"\b(operating\s+profit|ebit(?!da)|earnings\s+before\s+interest\s+and\s+tax)\b",
     NeedType.QUANTITATIVE, "operating_profit", "Operating Profit / EBIT", None),
    (r"\b(ebitda|earnings\s+before\s+interest.{0,6}tax.{0,6}depreciation)\b",
     NeedType.QUANTITATIVE, "ebitda", "EBITDA", None),

    # ── Margins ───────────────────────────────────────────────────────────────
    (r"\b(opm|operating\s+margin|operating\s+profit\s+margin)\b",
     NeedType.QUANTITATIVE, "opm", "OPM %", None),
    (r"\b(ebitda\s+margin)\b",
     NeedType.QUANTITATIVE, "ebitda_margin", "EBITDA Margin %", None),
    (r"\b(net\s+(profit\s+)?margin|npm)\b",
     NeedType.QUANTITATIVE, "net_margin", "Net Margin %", None),
    (r"\b(gross\s+margin)\b",
     NeedType.QUANTITATIVE, "gross_margin", "Gross Margin %", None),
    (r"\bebit\s+margin\b",
     NeedType.QUANTITATIVE, "ebit", "EBIT Margin %", None),

    # ── EPS ───────────────────────────────────────────────────────────────────
    (r"\b(eps|earnings\s+per\s+share|diluted\s+eps|basic\s+eps)\b",
     NeedType.QUANTITATIVE, "eps", "EPS", None),
    (r"\b(eps\s+estimate|forward\s+eps|projected\s+eps|consensus\s+eps)\b",
     NeedType.QUANTITATIVE, "eps_estimate", "EPS Estimate", TimeHorizon.FORWARD_LOOKING),
    (r"\b(eps\s+surprise|beat|miss)\b",
     NeedType.QUANTITATIVE, "eps_surprise", "EPS Surprise", None),
    (r"\b(eps\s+revision)\b",
     NeedType.QUANTITATIVE, "eps_revision", "EPS Revision", None),

    # ── Balance sheet ─────────────────────────────────────────────────────────
    (r"\b(total\s+assets?)\b",
     NeedType.QUANTITATIVE, "total_assets", "Total Assets", None),
    (r"\b(shareholders[\s\']*\s*equity|net\s+worth|book\s+equity|total\s+equity)\b",
     NeedType.QUANTITATIVE, "total_equity", "Total Equity / Net Worth", None),
    (r"\b(net\s+debt)\b",
     NeedType.QUANTITATIVE, "net_debt", "Net Debt", None),
    # [FIX] borrowings — also match bare "borrowings" and "debt" standalone
    (r"\b(total\s+(?:debt|borrowings?)|long[\s\-]term\s+debt|lt\s+borrowings?|borrowings?(?:\s+and)?)\b",
     NeedType.QUANTITATIVE, "borrowings", "Total Borrowings", None),
    (r"\b(cash\s+(?:and\s+(?:cash\s+)?equivalents?)?|cash\s+on\s+hand)\b",
     NeedType.QUANTITATIVE, "cash", "Cash & Equivalents", None),
    (r"\b(trade\s+receivables?|debtors?)\b",
     NeedType.QUANTITATIVE, "trade_receivables", "Trade Receivables", None),
    (r"\b(inventori(?:es|y)|stock[\s\-]in[\s\-]trade)\b",
     NeedType.QUANTITATIVE, "inventories", "Inventories", None),
    (r"\b(cwip|capital\s+work[\s\-]+in[\s\-]+progress)\b",
     NeedType.QUANTITATIVE, "cwip", "CWIP", None),
    (r"\b(fixed\s+assets?|net\s+block|property\s+plant\s+equipment|ppe)\b",
     NeedType.QUANTITATIVE, "fixed_assets", "Fixed Assets / Net Block", None),
    (r"\b(investments?)\b",
     NeedType.QUANTITATIVE, "investments", "Investments", None),

    # ── Cash flow ─────────────────────────────────────────────────────────────
    (r"\b(ocf|operating\s+cash\s+flow|cash\s+from\s+operations?|cfo)\b",
     NeedType.QUANTITATIVE, "ocf", "Operating Cash Flow", None),
    (r"\b(capex|capital\s+expenditure|cap(?:ital)?\s+ex)\b",
     NeedType.QUANTITATIVE, "capex", "Capex", None),
    (r"\b(fcf|free\s+cash\s+flow)\b",
     NeedType.QUANTITATIVE, "fcf", "Free Cash Flow", None),
    (r"\b(cfi|cash\s+from\s+investing)\b",
     NeedType.QUANTITATIVE, "cfi", "Cash from Investing", None),
    (r"\b(cff|cash\s+from\s+financing)\b",
     NeedType.QUANTITATIVE, "cff", "Cash from Financing", None),
    (r"\b(fcf\s+margin)\b",
     NeedType.QUANTITATIVE, "fcf_margin", "FCF Margin %", None),

    # ── Growth ────────────────────────────────────────────────────────────────
    (r"\b(revenue\s+cagr|sales\s+cagr|revenue\s+growth)\b",
     NeedType.QUANTITATIVE, "revenue_cagr", "Revenue CAGR", None),
    (r"\b(profit\s+cagr|earnings\s+cagr|pat\s+cagr)\b",
     NeedType.QUANTITATIVE, "profit_cagr", "Profit CAGR", None),
    (r"\b(eps\s+cagr|eps\s+growth)\b",
     NeedType.QUANTITATIVE, "eps_cagr", "EPS CAGR", None),
    (r"\b(ebitda\s+cagr|ebitda\s+growth)\b",
     NeedType.QUANTITATIVE, "ebitda_cagr", "EBITDA CAGR", None),
    (r"\b(stock\s+(?:price\s+)?cagr|price\s+cagr|stock\s+return)\b",
     NeedType.QUANTITATIVE, "stock_cagr", "Stock Price CAGR", None),

    # ── Return ratios ─────────────────────────────────────────────────────────
    (r"\b(roe|return\s+on\s+equity|return\s+on\s+net\s+worth|ronw)\b",
     NeedType.QUANTITATIVE, "roe", "ROE", None),
    (r"\b(roce|return\s+on\s+capital\s+employed)\b",
     NeedType.QUANTITATIVE, "roce", "ROCE", None),
    (r"\b(roa|return\s+on\s+assets?)\b",
     NeedType.QUANTITATIVE, "roa", "ROA", None),

    # ── Valuation ─────────────────────────────────────────────────────────────
    (r"\b(p/?e\s+ratio|price\s+to\s+earnings|pe\s+multiple|ttm\s+pe|forward\s+pe)\b",
     NeedType.QUANTITATIVE, "pe", "P/E Ratio", None),
    (r"\b(p[/\s]b(?:\s+(?:ratio|multiple))?|price[\s\-]to[\s\-]book|pb\s+(?:ratio|multiple)|price\s+to\s+book)\b",
     NeedType.QUANTITATIVE, "pb", "P/B Ratio", None),
    (r"\b(ev[/\s]ebitda|enterprise\s+value\s+to\s+ebitda)\b",
     NeedType.QUANTITATIVE, "ev_ebitda", "EV/EBITDA", None),
    (r"\b(market\s+cap(?:italisation)?|mcap)\b",
     NeedType.QUANTITATIVE, "market_cap", "Market Cap", None),
    (r"\b(graham\s+number)\b",
     NeedType.QUANTITATIVE, "graham_number", "Graham Number", None),
    (r"\b(book\s+value(?:\s+per\s+share)?|bvps)\b",
     NeedType.QUANTITATIVE, "book_value", "Book Value", None),
    (r"\b(dividend\s+yield)\b",
     NeedType.QUANTITATIVE, "dividend_yield", "Dividend Yield", None),
    (r"\b(dividend\s+payout)\b",
     NeedType.QUANTITATIVE, "dividend_payout", "Dividend Payout %", None),

    # ── Leverage ──────────────────────────────────────────────────────────────
    (r"\b(debt[\s\-]to[\s\-]equity|d[/\s]e\s+ratio|leverage\s+ratio)\b",
     NeedType.QUANTITATIVE, "debt_equity", "D/E Ratio", None),
    (r"\b(current\s+ratio)\b",
     NeedType.QUANTITATIVE, "current_ratio", "Current Ratio", None),
    (r"\b(quick\s+ratio|acid\s+test)\b",
     NeedType.QUANTITATIVE, "quick_ratio", "Quick Ratio", None),
    (r"\b(interest\s+coverage)\b",
     NeedType.QUANTITATIVE, "interest_coverage", "Interest Coverage", None),

    # ── Working capital efficiency ────────────────────────────────────────────
    (r"\b(dso|debtor\s+days?|days?\s+sales?\s+outstanding)\b",
     NeedType.QUANTITATIVE, "dso", "DSO (Days)", None),
    (r"\b(dio|inventory\s+days?|days?\s+inventory\s+outstanding)\b",
     NeedType.QUANTITATIVE, "dio", "DIO (Days)", None),
    (r"\b(dpo|payable\s+days?|days?\s+payable\s+outstanding)\b",
     NeedType.QUANTITATIVE, "dpo", "DPO (Days)", None),
    (r"\b(ccc|cash\s+conversion\s+cycle)\b",
     NeedType.QUANTITATIVE, "ccc", "Cash Conversion Cycle", None),
    (r"\b(working\s+capital\s+days?)\b",
     NeedType.QUANTITATIVE, "working_capital", "Working Capital Days", None),

    # ── Price / technicals ────────────────────────────────────────────────────
    (r"\b(52[\s\-]week\s+(?:high|low)|52[\s\-]?w[\s\-]?(?:high|low)|year[\s\-](?:high|low))\b",
     NeedType.TECHNICAL, "52w_high", "52-Week High/Low", TimeHorizon.CURRENT),
    (r"\b(rsi(?:[\s\-]14)?)\b",
     NeedType.TECHNICAL, "rsi", "RSI(14)", TimeHorizon.CURRENT),
    (r"\b(macd)\b",
     NeedType.TECHNICAL, "macd", "MACD", TimeHorizon.CURRENT),
    (r"\b(sma[\s\-]?50|50[\s\-](?:day\s+)?sma|50[\s\-]day\s+moving\s+average)\b",
     NeedType.TECHNICAL, "sma", "SMA-50", TimeHorizon.CURRENT),
    (r"\b(sma[\s\-]?200|200[\s\-](?:day\s+)?sma|200[\s\-]day\s+moving\s+average)\b",
     NeedType.TECHNICAL, "sma", "SMA-200", TimeHorizon.CURRENT),
    (r"\b(ema[\s\-]?21|21[\s\-]day\s+ema)\b",
     NeedType.TECHNICAL, "ema", "EMA-21", TimeHorizon.CURRENT),
    (r"\b(bollinger\s+bands?|bb\s+bands?)\b",
     NeedType.TECHNICAL, "bb", "Bollinger Bands", TimeHorizon.CURRENT),
    (r"\b(supertrend)\b",
     NeedType.TECHNICAL, "supertrend", "Supertrend", TimeHorizon.CURRENT),
    (r"\b(atr|average\s+true\s+range)\b",
     NeedType.TECHNICAL, "atr", "ATR(14)", TimeHorizon.CURRENT),
    (r"\b(adx|average\s+directional\s+(?:index|indicator))\b",
     NeedType.TECHNICAL, "adx", "ADX(14)", TimeHorizon.CURRENT),
    (r"\b(vwap)\b",
     NeedType.TECHNICAL, "vwap", "VWAP", TimeHorizon.CURRENT),
    (r"\b(obv|on[\s\-]balance\s+volume)\b",
     NeedType.TECHNICAL, "obv", "OBV", TimeHorizon.CURRENT),
    (r"\b(current\s+price|stock\s+price|share\s+price|cmp|ltp)\b",
     NeedType.TECHNICAL, "price", "Current Price", TimeHorizon.CURRENT),

    # ── Macro ─────────────────────────────────────────────────────────────────
    (r"\b(repo\s+rate|rbi\s+rate|policy\s+rate)\b",
     NeedType.MACRO, "repo_rate", "Repo Rate", TimeHorizon.CURRENT),
    (r"\b(rbi\s+policy|monetary\s+policy)\b",
     NeedType.MACRO, "rbi_policy", "RBI Policy", TimeHorizon.CURRENT),
    (r"\b(usd[/\-]inr|dollar[\s\-]rupee|forex|currency)\b",
     NeedType.MACRO, "forex", "Forex / USD-INR", TimeHorizon.CURRENT),
    (r"\b(gdp|gross\s+domestic\s+product)\b",
     NeedType.MACRO, "macro", "GDP", None),
    (r"\b(inflation|cpi|wpi)\b",
     NeedType.MACRO, "macro", "Inflation", None),
    (r"\b(nifty|sensex|nse\s+index|bse\s+index|market\s+index)\b",
     NeedType.MACRO, "market_index", "Market Index", TimeHorizon.CURRENT),

    # ── Ownership ─────────────────────────────────────────────────────────────
    (r"\b(promoter\s+(?:holding|stake|pledging?|ownership))\b",
     NeedType.OWNERSHIP, "promoter", "Promoter Holding", None),
    (r"\b(fii\s+(?:holding|stake|buying|selling|flow)|foreign\s+institutional)\b",
     NeedType.OWNERSHIP, "fii", "FII Holding", None),
    (r"\b(dii\s+(?:holding|stake|buying|selling|flow)|domestic\s+institutional)\b",
     NeedType.OWNERSHIP, "dii", "DII Holding", None),
    (r"\b(institutional\s+(?:holding|ownership))\b",
     NeedType.OWNERSHIP, "institutional", "Institutional Holding", None),
    (r"\b(number\s+of\s+shareholders?|shareholder\s+count)\b",
     NeedType.OWNERSHIP, "shareholders", "Shareholder Count", None),

    # ── Corporate actions ─────────────────────────────────────────────────────
    (r"\b(dividends?(?:\s+(?:history|declared|per\s+share|paid))?|dps)\b",
     NeedType.QUANTITATIVE, "dividend", "Dividend", None),
    (r"\b(buyback|buy[\s\-]back|share\s+repurchase)\b",
     NeedType.QUANTITATIVE, "buyback", "Buyback", None),
    (r"\b(bonus\s+(?:share|issue))\b",
     NeedType.QUANTITATIVE, "bonus", "Bonus Issue", None),
    (r"\b(stock\s+split|share\s+split)\b",
     NeedType.QUANTITATIVE, "split", "Stock Split", None),

    # ── Qualitative / Vector ──────────────────────────────────────────────────
    (r"\b(md&?a|management\s+discussion|management\s+(?:analysis|commentary))\b",
     NeedType.QUALITATIVE, "mda", "MD&A", None),
    (r"\b(risk\s+factors?|business\s+risks?|key\s+risks?)\b",
     NeedType.QUALITATIVE, "risk_factors", "Risk Factors", None),
    (r"\b(business\s+(?:overview|model|description|segment))\b",
     NeedType.QUALITATIVE, "business_overview", "Business Overview", None),
    (r"\b(strategy|strategic\s+(?:plan|direction|initiatives?))\b",
     NeedType.QUALITATIVE, "strategy", "Strategy", None),

    # ── Forward-looking / Concall ─────────────────────────────────────────────
    (r"\b(guidance|outlook|management\s+outlook|company\s+guidance)\b",
     NeedType.FORWARD_LOOKING, "concall_guidance", "Guidance / Outlook",
     TimeHorizon.FORWARD_LOOKING),
    (r"\b(concall|earnings\s+call|analyst\s+call|investor\s+call)\b",
     NeedType.FORWARD_LOOKING, "concall_mgmt", "Concall Commentary",
     TimeHorizon.FORWARD_LOOKING),
    (r"\b(management\s+(?:said|said\s+about|view|commentary\s+on)|ceo\s+said|cfo\s+said)\b",
     NeedType.FORWARD_LOOKING, "concall_mgmt", "Management Commentary",
     TimeHorizon.FORWARD_LOOKING),
    (r"\b(demand\s+environment|demand\s+scenario|demand\s+outlook)\b",
     NeedType.FORWARD_LOOKING, "concall_outlook", "Demand Outlook",
     TimeHorizon.FORWARD_LOOKING),
    (r"\b(capex\s+(?:plan|guidance|outlook|target)|capex\s+guidance)\b",
     NeedType.FORWARD_LOOKING, "concall_capex", "Capex Guidance",
     TimeHorizon.FORWARD_LOOKING),
    (r"\b(margins?\s+(?:guidance|outlook|target)|margin\s+guidance)\b",
     NeedType.FORWARD_LOOKING, "concall_margin", "Margin Guidance",
     TimeHorizon.FORWARD_LOOKING),

    # ── Comparative ───────────────────────────────────────────────────────────
    (r"\b(compare(?:\s+with)?|vs\.?|versus|comparison\s+(?:with|between))\b",
     NeedType.COMPARATIVE, "mda", "Comparative Analysis", None),
]

# Compile all patterns once
_COMPILED_RULES: List[tuple] = [
    (re.compile(pattern, re.IGNORECASE), need_type, sub_type, metric, time_hint)
    for pattern, need_type, sub_type, metric, time_hint in _RULES
]


# ─────────────────────────────────────────────────────────────────────────────
# Year extractor (reuse logic from retriever.py, self-contained copy)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_years(query: str) -> List[int]:
    """Extract explicitly mentioned fiscal years from a query string."""
    q = query.lower()
    years: set = set()

    # FY range: FY23-25, FY2023-2025
    m = re.search(r"fy[\s\-]*(\d{2,4})\s*(?:to|through|[-\/–])\s*(?:fy[\s\-]*)?(\d{2,4})", q)
    if m:
        y1 = int(m.group(1)); y2 = int(m.group(2))
        y1 = 2000 + y1 if y1 < 100 else y1
        y2 = 2000 + y2 if y2 < 100 else y2
        years.update(range(min(y1, y2), max(y1, y2) + 1))

    # Multiple FY mentions: FY23, FY24, FY25
    for raw in re.findall(r"\bfy[\s\-]*(\d{2,4})\b", q):
        y = int(raw); years.add(2000 + y if y < 100 else y)

    # Plain 4-digit years
    for raw in re.findall(r"\b(20\d{2})\b", q):
        y = int(raw)
        if 2010 <= y <= 2026:
            years.add(y)

    return sorted(years)


# ─────────────────────────────────────────────────────────────────────────────
# Symbol extractor
# ─────────────────────────────────────────────────────────────────────────────

# Well-known NSE symbols for fast extraction without DB lookup
_COMMON_SYMBOLS = {
    "reliance", "tcs", "infosys", "infy", "hdfcbank", "icicibank", "sbi",
    "adaniports", "adanient", "adanigreen", "adanipower", "tatamotors",
    "tatasteel", "bajfinance", "bajajfinsv", "wipro", "hcltech", "itc",
    "maruti", "sunpharma", "drreddy", "cipla", "asianpaint", "ultracemco",
    "titan", "nestleind", "britannia", "dabur", "marico", "godrejcp",
    "hindunilvr", "hul", "coalindia", "ongc", "bpcl", "ioc", "ntpc",
    "powergrid", "grasim", "m&m", "mahindra", "bosch", "bajaj-auto",
    "eichermot", "heromotoco", "tvsmotors", "apollohosp", "lici",
    "indusind", "kotakbank", "axisbank", "yesbank", "pfc", "recltd",
}

def _extract_symbols(query: str) -> List[str]:
    """Extract NSE symbols mentioned in the query (uppercase tokens)."""
    found = []
    # Match ALL-CAPS tokens 2-15 chars
    tokens = re.findall(r"\b([A-Z][A-Z0-9&\-]{1,14})\b", query)
    for tok in tokens:
        if tok.lower() in _COMMON_SYMBOLS or tok.upper() in _COMMON_SYMBOLS:
            found.append(tok.upper())
    # Also check lowercase against known set
    lower_query = query.lower()
    for sym in _COMMON_SYMBOLS:
        # word-boundary match
        if re.search(r'\b' + re.escape(sym) + r'\b', lower_query):
            up = sym.upper()
            if up not in found:
                found.append(up)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


# ─────────────────────────────────────────────────────────────────────────────
# Time horizon inference
# ─────────────────────────────────────────────────────────────────────────────

_FORWARD_SIGNALS = re.compile(
    r"\b(guidance|outlook|forecast|expect|anticipate|going\s+forward|"
    r"next\s+(?:year|quarter|fy)|h1\s+fy|h2\s+fy|future|projection|target|plan)\b",
    re.IGNORECASE,
)
_HISTORICAL_SIGNALS = re.compile(
    r"\b(fy\d{2,4}|last\s+\d+\s+years?|historical|trend|cagr|over\s+the\s+years)\b",
    re.IGNORECASE,
)

def _infer_time_horizon(query: str, explicit_hint: Optional[TimeHorizon]) -> TimeHorizon:
    if explicit_hint is not None:
        return explicit_hint
    if _FORWARD_SIGNALS.search(query):
        return TimeHorizon.FORWARD_LOOKING
    if _HISTORICAL_SIGNALS.search(query):
        return TimeHorizon.HISTORICAL
    return TimeHorizon.CURRENT


# ─────────────────────────────────────────────────────────────────────────────
# Period type inference  (annual / quarterly / ttm)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_period(query: str) -> str:
    q = query.lower()
    if re.search(r"\b(q[1-4]|quarterly|quarter|qtr)\b", q):
        return "quarterly"
    if re.search(r"\b(ttm|trailing\s+twelve\s+months?|last\s+twelve\s+months?)\b", q):
        return "ttm"
    return "annual"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Rule-based decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_decompose(query: str) -> List[AtomicNeed]:
    """Apply pattern rules to extract atomic needs. Returns list (may be empty)."""
    years   = _extract_years(query)
    symbols = _extract_symbols(query)
    period  = _infer_period(query)
    primary_symbol = symbols[0] if symbols else None

    # Dedup key = (sub_type, need_type) so that:
    #   - FORWARD_LOOKING/concall_capex  ≠  QUANTITATIVE/capex
    #   - QUANTITATIVE/net_profit_q      ≠  QUANTITATIVE/net_profit
    seen_keys: set = set()
    atoms: List[AtomicNeed] = []

    for compiled, need_type, sub_type, metric, time_hint in _COMPILED_RULES:
        m = compiled.search(query)
        if not m:
            continue

        dedup_key = (sub_type, need_type)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        horizon = _infer_time_horizon(query, time_hint)

        # For COMPARATIVE: capture both symbols
        atom_symbols = symbols if need_type == NeedType.COMPARATIVE else []

        atom = AtomicNeed(
            need_type    = need_type,
            sub_type     = sub_type,
            metric       = metric,
            symbol       = primary_symbol,
            symbols      = atom_symbols,
            years        = years,
            time_horizon = horizon,
            period_type  = period,
            raw_text     = m.group(0),
            confidence   = 1.0,
            source       = "rule",
        )
        atom.resolve_schema()
        atoms.append(atom)

    atoms = _upgrade_forward_atoms(atoms, query)
    return atoms


def _upgrade_forward_atoms(atoms: List[AtomicNeed], query: str) -> List[AtomicNeed]:
    """
    Post-processing: when the query has forward-looking signals AND a QUANTITATIVE
    capex or margin atom was extracted, also emit the FORWARD_LOOKING counterpart.

    Handles: "What guidance did management give for capex and margins?"
    """
    has_forward = bool(_FORWARD_SIGNALS.search(query))
    if not has_forward:
        return atoms

    existing = {(a.sub_type, a.need_type) for a in atoms}
    extra: List[AtomicNeed] = []

    # capex QUANT → concall_capex FORWARD
    if ("capex", NeedType.QUANTITATIVE) in existing and \
       ("concall_capex", NeedType.FORWARD_LOOKING) not in existing:
        base = next(a for a in atoms if a.sub_type == "capex")
        twin = AtomicNeed(
            need_type=NeedType.FORWARD_LOOKING, sub_type="concall_capex",
            metric="Capex Guidance", symbol=base.symbol, symbols=base.symbols,
            years=base.years, time_horizon=TimeHorizon.FORWARD_LOOKING,
            period_type=base.period_type, raw_text=base.raw_text,
            confidence=0.85, source="rule",
        )
        twin.resolve_schema()
        extra.append(twin)

    # margin word or margin QUANT atom → concall_margin FORWARD
    margin_subtypes = {"opm", "ebitda_margin", "net_margin", "gross_margin", "ebit"}
    has_margin = any((s, NeedType.QUANTITATIVE) in existing for s in margin_subtypes) \
                 or bool(re.search(r"\bmargins?\b", query, re.I))
    if has_margin and ("concall_margin", NeedType.FORWARD_LOOKING) not in existing:
        base = next((a for a in atoms if a.sub_type in margin_subtypes), None) \
               or (atoms[0] if atoms else None)
        if base:
            twin = AtomicNeed(
                need_type=NeedType.FORWARD_LOOKING, sub_type="concall_margin",
                metric="Margin Guidance", symbol=base.symbol, symbols=base.symbols,
                years=base.years, time_horizon=TimeHorizon.FORWARD_LOOKING,
                period_type=base.period_type, raw_text="margins",
                confidence=0.85, source="rule",
            )
            twin.resolve_schema()
            extra.append(twin)

    return atoms + extra
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
You are a financial query parser for Indian stock market analysis.

Given a raw user query, decompose it into a JSON array of atomic information needs.

Each atom must have:
  "need_type":    one of [quantitative, qualitative, forward_looking, comparative, technical, macro, ownership]
  "sub_type":     one of the keys in the sub-type catalogue (see below)
  "metric":       short human-readable label (e.g. "Revenue", "ROCE", "Guidance")
  "symbol":       NSE ticker symbol if mentioned, else null
  "symbols":      array of symbols for comparative queries, else []
  "years":        array of fiscal years explicitly mentioned (e.g. [2024, 2025]), else []
  "time_horizon": one of [historical, current, forward_looking]
  "period_type":  one of [annual, quarterly, ttm]
  "raw_text":     the exact phrase in the query that triggered this atom
  "confidence":   float 0.0–1.0
  "source":       always "llm"

Sub-type catalogue (partial):
  revenue, net_profit, ebitda, opm, ebitda_margin, net_margin, eps,
  total_assets, total_equity, borrowings, net_debt, cash, capex, fcf, ocf,
  roe, roce, roa, pe, pb, ev_ebitda, book_value, debt_equity, current_ratio,
  revenue_cagr, profit_cagr, stock_cagr, rsi, macd, sma, supertrend,
  repo_rate, forex, macro, market_index,
  promoter, fii, dii,
  mda, risk_factors, strategy, business_overview,
  concall_guidance, concall_mgmt, concall_outlook, concall_capex, concall_margin

Return ONLY valid JSON array. No preamble, no markdown, no explanation.
"""

def _llm_decompose(query: str, api_key: str, api_url: str, model: str) -> List[AtomicNeed]:
    """Call LLM to decompose a query into atoms."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if "openrouter" in api_url:
            headers["HTTP-Referer"] = "https://github.com/QuantCoPilot"
            headers["X-Title"]      = "Quant CoPilot Decomposer"

        payload = {
            "model":       model,
            "messages":    [
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            "max_tokens":  800,
            "temperature": 0.0,
        }
        resp = requests.post(api_url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()

        raw_text = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip possible markdown fences
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

        items = json.loads(raw_text)
        atoms = []
        for item in items:
            atom = AtomicNeed(
                need_type    = NeedType(item.get("need_type", "qualitative")),
                sub_type     = item.get("sub_type", "mda"),
                metric       = item.get("metric", ""),
                symbol       = item.get("symbol"),
                symbols      = item.get("symbols", []),
                years        = item.get("years", []),
                time_horizon = TimeHorizon(item.get("time_horizon", "current")),
                period_type  = item.get("period_type", "annual"),
                raw_text     = item.get("raw_text", ""),
                confidence   = float(item.get("confidence", 0.8)),
                source       = "llm",
            )
            atom.resolve_schema()
            atoms.append(atom)
        return atoms

    except Exception as e:
        return []   # silent fallback — caller handles empty list


# ─────────────────────────────────────────────────────────────────────────────
# AtomicDecomposer  (public interface)
# ─────────────────────────────────────────────────────────────────────────────

class AtomicDecomposer:
    """
    Main decomposer class.

    Usage:
        decomposer = AtomicDecomposer()
        atoms = decomposer.decompose("What is ADANIPORTS's ROCE and FCF for FY24?")

    LLM fallback fires when rule-based extraction returns fewer than
    min_rule_atoms atoms AND an API key is configured.
    """

    def __init__(
        self,
        # LLM fallback settings — reads from env if not provided
        api_key:       Optional[str] = None,
        api_url:       Optional[str] = None,
        model:         Optional[str] = None,
        min_rule_atoms: int = 1,  # fall back to LLM if rules find < N atoms
    ):
        self.api_key  = api_key  or os.getenv("GROQ_API_KEY", "") \
                                 or os.getenv("OPENROUTER_API_KEY", "")
        self.api_url  = api_url  or (
            "https://api.groq.com/openai/v1/chat/completions"
            if os.getenv("GROQ_API_KEY")
            else "https://openrouter.ai/api/v1/chat/completions"
        )
        self.model    = model or (
            "llama3-8b-8192"               # Groq free tier — fast & cheap
            if os.getenv("GROQ_API_KEY")
            else "qwen/qwen3-8b:free"      # OpenRouter free
        )
        self.min_rule_atoms = min_rule_atoms

    # ── Main entry point ──────────────────────────────────────────────────────

    def decompose(self, query: str) -> List[AtomicNeed]:
        """
        Decompose a user query into a list of AtomicNeed objects.

        Strategy:
          1. Always run rule-based extraction (free, instant).
          2. If result is sparse (< min_rule_atoms) AND LLM key is set,
             run LLM fallback and merge unique atoms.
          3. Deduplicate by sub_type — rule atoms take precedence.
        """
        rule_atoms = _rule_based_decompose(query)

        if len(rule_atoms) >= self.min_rule_atoms:
            return rule_atoms

        # LLM fallback
        if self.api_key:
            llm_atoms  = _llm_decompose(query, self.api_key, self.api_url, self.model)
            seen       = {a.sub_type for a in rule_atoms}
            merged     = list(rule_atoms)
            for atom in llm_atoms:
                if atom.sub_type not in seen:
                    merged.append(atom)
                    seen.add(atom.sub_type)
            return merged

        return rule_atoms

    def decompose_verbose(self, query: str) -> dict:
        """
        Same as decompose() but returns a diagnostic dict:
          {
            "query":       str,
            "atoms":       [atom.to_dict(), ...],
            "atom_count":  int,
            "need_types":  {type: count},
            "channels":    ["sql", "vector", "concall"],
            "symbols":     [...],
            "years":       [...],
          }
        """
        t0    = time.perf_counter()
        atoms = self.decompose(query)
        elapsed = time.perf_counter() - t0

        need_type_counts: Dict[str, int] = {}
        symbols_seen: set = set()
        years_seen:   set = set()
        channels:     set = set()

        for atom in atoms:
            k = atom.need_type.value
            need_type_counts[k] = need_type_counts.get(k, 0) + 1
            if atom.symbol:
                symbols_seen.add(atom.symbol)
            symbols_seen.update(atom.symbols)
            years_seen.update(atom.years)
            # Map need_type → channel
            if atom.need_type in (NeedType.QUANTITATIVE, NeedType.OWNERSHIP,
                                   NeedType.MACRO, NeedType.TECHNICAL):
                channels.add("sql")
            if atom.need_type == NeedType.QUALITATIVE:
                channels.add("vector")
            if atom.need_type == NeedType.FORWARD_LOOKING:
                channels.add("concall")
            if atom.need_type == NeedType.COMPARATIVE:
                channels.update(["sql", "vector"])

        return {
            "query":       query,
            "atoms":       [a.to_dict() for a in atoms],
            "atom_count":  len(atoms),
            "need_types":  need_type_counts,
            "channels":    sorted(channels),
            "symbols":     sorted(symbols_seen),
            "years":       sorted(years_seen),
            "elapsed_ms":  round(elapsed * 1000, 2),
        }