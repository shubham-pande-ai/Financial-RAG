"""
eval/eval_suite.py
──────────────────
Evaluation Suite for the Financial RAG Pipeline

Computes the two most important Information Retrieval metrics:

  Hit Rate@k   — "Does the correct chunk appear anywhere in the top-k results?"
                  Measures retrieval COVERAGE.  Goal: > 0.80 at k=10.

  MRR@k        — Mean Reciprocal Rank.  "How close to position 1 is the first
                  correct result?"  Rewards ranking quality.  Goal: > 0.60.

Additionally computes:
  Precision@k  — What fraction of the top-k are relevant?
  NDCG@k       — Normalised Discounted Cumulative Gain (graded relevance)

═══════════════════════════════════════════════════════════════════════
USAGE
═══════════════════════════════════════════════════════════════════════

1.  Build a golden dataset (one-time effort):

    from eval.eval_suite import GoldenSample, EvalDataset

    dataset = EvalDataset([
        GoldenSample(
            query         = "What was ADANIPORTS revenue in FY25?",
            symbol        = "ADANIPORTS",
            doc_type      = "annual_report",
            years         = [2025],
            relevant_ids  = ["adaniports_fy25_annual_page42",
                              "adaniports_fy25_annual_page43"],
        ),
        GoldenSample(
            query         = "ADANIPORTS EBITDA margin FY23 FY24 FY25",
            symbol        = "ADANIPORTS",
            doc_type      = "annual_report",
            years         = [2023, 2024, 2025],
            relevant_ids  = ["adaniports_fy25_annual_page44",
                              "adaniports_fy24_annual_page39"],
        ),
        # ... add 20-50 samples for meaningful results
    ])

    dataset.save("eval/golden_dataset.json")

2.  Run the evaluation:

    from eval.eval_suite import run_eval
    from pipeline.retrieval.retriever import retrieve_with_years
    from pipeline.retrieval.reranker  import rerank_separate

    results = run_eval(
        dataset        = dataset,
        retriever_fn   = retrieve_with_years,   # pass your retriever
        reranker_fn    = rerank_separate,        # pass your reranker (or None)
        k_values       = [5, 10, 20],
        verbose        = True,
    )
    results.print_report()
    results.save("eval/results_20250516.json")

3.  Interpret results:

    Hit Rate@10 > 0.80  → retriever is good; most answers can be found
    MRR@10      > 0.60  → reranker is doing its job
    MRR@10      < 0.40  → reranker needs tuning or top-k needs raising

═══════════════════════════════════════════════════════════════════════
HARDWARE NOTE (i5-1240p)
═══════════════════════════════════════════════════════════════════════
Running the full eval suite (50 samples × rerank) takes ~15-30 min on your
machine because of the reranker.  Run with reranker_fn=None first (5-10 min)
to get retriever-only metrics, then add the reranker once you're happy with
retrieval coverage.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from utils.logger import get_logger
    log = get_logger(__name__)
except Exception:
    import logging
    log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Golden dataset types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldenSample:
    """
    One evaluation example.

    relevant_ids  — list of chunk_ids that are considered correct answers.
                    At least one of these must appear in the top-k for a hit.
                    Provide 1–5 per query; more is better for NDCG.
    graded_ids    — optional dict {chunk_id: relevance_score} for NDCG.
                    relevance_score: 2 = perfect, 1 = acceptable, 0 = irrelevant.
                    If omitted, binary relevance (1 for all relevant_ids) is used.
    """
    query:        str
    symbol:       Optional[str]       = None
    doc_type:     str                 = "annual_report"
    years:        Optional[List[int]] = None
    relevant_ids: List[str]           = field(default_factory=list)
    graded_ids:   Dict[str, int]      = field(default_factory=dict)   # for NDCG
    notes:        str                 = ""

    def get_relevance(self, chunk_id: str) -> int:
        """Return graded relevance score (0/1/2) for a chunk."""
        if self.graded_ids:
            return self.graded_ids.get(chunk_id, 0)
        return 1 if chunk_id in self.relevant_ids else 0


@dataclass
class EvalDataset:
    """Collection of GoldenSamples, with load/save helpers."""
    samples: List[GoldenSample] = field(default_factory=list)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in self.samples], f, indent=2)
        log.info(f"[eval] dataset saved → {path} ({len(self.samples)} samples)")

    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        with open(path) as f:
            raw = json.load(f)
        samples = [GoldenSample(**r) for r in raw]
        log.info(f"[eval] dataset loaded ← {path} ({len(samples)} samples)")
        return cls(samples=samples)

    def add(self, sample: GoldenSample) -> None:
        self.samples.append(sample)

    def filter_by_doc_type(self, doc_type: str) -> "EvalDataset":
        return EvalDataset([s for s in self.samples if s.doc_type == doc_type])

    def filter_by_symbol(self, symbol: str) -> "EvalDataset":
        return EvalDataset([s for s in self.samples if s.symbol == symbol])


# ─────────────────────────────────────────────────────────────────────────────
# Per-query result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query:           str
    symbol:          Optional[str]
    doc_type:        str
    years:           Optional[List[int]]
    retrieved_ids:   List[str]          # ordered list of chunk_ids returned
    relevant_ids:    List[str]          # ground truth
    k_values:        List[int]

    # Computed metrics (filled by _compute_metrics)
    hit_at_k:        Dict[int, float]   = field(default_factory=dict)
    mrr_at_k:        Dict[int, float]   = field(default_factory=dict)
    precision_at_k:  Dict[int, float]   = field(default_factory=dict)
    ndcg_at_k:       Dict[int, float]   = field(default_factory=dict)
    first_hit_rank:  Optional[int]      = None   # 1-indexed position of first hit
    retrieval_ms:    float              = 0.0
    rerank_ms:       float              = 0.0
    error:           Optional[str]      = None


def _compute_metrics(
    result:   QueryResult,
    sample:   GoldenSample,
    k_values: List[int],
) -> None:
    """Compute and fill all metrics on a QueryResult in-place."""
    retrieved = result.retrieved_ids
    relevant  = set(result.relevant_ids)

    if not relevant:
        # Nothing to evaluate — mark all metrics as NaN
        for k in k_values:
            result.hit_at_k[k]       = float("nan")
            result.mrr_at_k[k]       = float("nan")
            result.precision_at_k[k] = float("nan")
            result.ndcg_at_k[k]      = float("nan")
        return

    # First hit rank
    for i, cid in enumerate(retrieved, 1):
        if cid in relevant:
            result.first_hit_rank = i
            break

    for k in k_values:
        top_k = retrieved[:k]

        # ── Hit Rate@k ────────────────────────────────────────────────────────
        hit = 1.0 if any(c in relevant for c in top_k) else 0.0
        result.hit_at_k[k] = hit

        # ── MRR@k ─────────────────────────────────────────────────────────────
        rr = 0.0
        for rank, cid in enumerate(top_k, 1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        result.mrr_at_k[k] = rr

        # ── Precision@k ───────────────────────────────────────────────────────
        n_relevant_in_k = sum(1 for c in top_k if c in relevant)
        result.precision_at_k[k] = n_relevant_in_k / k if k > 0 else 0.0

        # ── NDCG@k ────────────────────────────────────────────────────────────
        # Ideal DCG = sum of graded relevance at best possible ranking
        # Actual DCG = sum of graded relevance at actual ranking
        def _dcg(ranked_ids: List[str]) -> float:
            s = 0.0
            for i, cid in enumerate(ranked_ids, 1):
                rel = sample.get_relevance(cid)
                s  += rel / math.log2(i + 1)
            return s

        ideal_ids = sorted(
            [c for c in retrieved if sample.get_relevance(c) > 0],
            key=lambda c: sample.get_relevance(c),
            reverse=True,
        )
        idcg = _dcg(ideal_ids[:k])
        dcg  = _dcg(top_k)
        result.ndcg_at_k[k] = (dcg / idcg) if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    query_results:   List[QueryResult]
    k_values:        List[int]
    run_timestamp:   str = ""
    pipeline_config: Dict = field(default_factory=dict)

    # Aggregated metrics (mean over all queries, NaN-excluded)
    mean_hit_at_k:       Dict[int, float] = field(default_factory=dict)
    mean_mrr_at_k:       Dict[int, float] = field(default_factory=dict)
    mean_precision_at_k: Dict[int, float] = field(default_factory=dict)
    mean_ndcg_at_k:      Dict[int, float] = field(default_factory=dict)

    # Timing
    mean_retrieval_ms:   float = 0.0
    mean_rerank_ms:      float = 0.0
    mean_total_ms:       float = 0.0
    error_count:         int   = 0

    def aggregate(self) -> None:
        """Compute mean metrics across all query results."""
        for k in self.k_values:
            hits      = [r.hit_at_k.get(k, float("nan"))       for r in self.query_results]
            mrrs      = [r.mrr_at_k.get(k, float("nan"))       for r in self.query_results]
            precs     = [r.precision_at_k.get(k, float("nan")) for r in self.query_results]
            ndcgs     = [r.ndcg_at_k.get(k, float("nan"))      for r in self.query_results]

            self.mean_hit_at_k[k]       = _nanmean(hits)
            self.mean_mrr_at_k[k]       = _nanmean(mrrs)
            self.mean_precision_at_k[k] = _nanmean(precs)
            self.mean_ndcg_at_k[k]      = _nanmean(ndcgs)

        ret_ms   = [r.retrieval_ms for r in self.query_results if r.retrieval_ms > 0]
        rer_ms   = [r.rerank_ms    for r in self.query_results if r.rerank_ms    > 0]
        total_ms = [r.retrieval_ms + r.rerank_ms for r in self.query_results]

        self.mean_retrieval_ms = sum(ret_ms)   / len(ret_ms)   if ret_ms   else 0.0
        self.mean_rerank_ms    = sum(rer_ms)   / len(rer_ms)   if rer_ms   else 0.0
        self.mean_total_ms     = sum(total_ms) / len(total_ms) if total_ms else 0.0
        self.error_count       = sum(1 for r in self.query_results if r.error)

    def print_report(self) -> None:
        """Print a formatted evaluation report to stdout."""
        n = len(self.query_results)
        sep = "═" * 64

        print(f"\n{sep}")
        print(f"  Financial RAG — Evaluation Report")
        if self.run_timestamp:
            print(f"  Run: {self.run_timestamp}")
        print(f"  Samples: {n}  |  Errors: {self.error_count}")
        print(sep)

        # Header
        k_headers = "  ".join(f"@{k:>3}" for k in self.k_values)
        print(f"\n  Metric         {k_headers}")
        print("  " + "─" * (16 + 7 * len(self.k_values)))

        for label, metric_dict in [
            ("Hit Rate",    self.mean_hit_at_k),
            ("MRR",         self.mean_mrr_at_k),
            ("Precision",   self.mean_precision_at_k),
            ("NDCG",        self.mean_ndcg_at_k),
        ]:
            vals = "  ".join(
                f"{metric_dict.get(k, float('nan')):>6.3f}"
                for k in self.k_values
            )
            status = _status_emoji(label, metric_dict, self.k_values)
            print(f"  {label:<14} {vals}   {status}")

        print(f"\n  ── Latency (mean per query) ──────────────────────────")
        print(f"  Retrieval   : {self.mean_retrieval_ms:>7.0f} ms")
        print(f"  Reranking   : {self.mean_rerank_ms:>7.0f} ms")
        print(f"  Total       : {self.mean_total_ms:>7.0f} ms")

        print(f"\n  ── Target guidance ───────────────────────────────────")
        print(f"  Hit Rate@10 > 0.80  MRR@10 > 0.60  NDCG@10 > 0.65")
        print(f"{sep}\n")

        # Per-query detail (first 5 + last 5 to keep output manageable)
        if self.query_results:
            k10 = max(self.k_values) if 10 not in self.k_values else 10
            print(f"  Per-query detail (Hit@{k10} | MRR@{k10} | first hit rank)")
            print("  " + "─" * 60)
            for r in self.query_results[:5]:
                _print_query_row(r, k10)
            if len(self.query_results) > 10:
                print(f"  ... {len(self.query_results) - 10} queries omitted ...")
            if len(self.query_results) > 5:
                for r in self.query_results[-5:]:
                    _print_query_row(r, k10)
        print()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_timestamp":         self.run_timestamp,
            "pipeline_config":       self.pipeline_config,
            "n_samples":             len(self.query_results),
            "error_count":           self.error_count,
            "mean_hit_at_k":         self.mean_hit_at_k,
            "mean_mrr_at_k":         self.mean_mrr_at_k,
            "mean_precision_at_k":   self.mean_precision_at_k,
            "mean_ndcg_at_k":        self.mean_ndcg_at_k,
            "mean_retrieval_ms":     self.mean_retrieval_ms,
            "mean_rerank_ms":        self.mean_rerank_ms,
            "mean_total_ms":         self.mean_total_ms,
            "query_results":         [
                {
                    "query":          r.query,
                    "symbol":         r.symbol,
                    "doc_type":       r.doc_type,
                    "years":          r.years,
                    "first_hit_rank": r.first_hit_rank,
                    "hit_at_k":       r.hit_at_k,
                    "mrr_at_k":       r.mrr_at_k,
                    "precision_at_k": r.precision_at_k,
                    "ndcg_at_k":      r.ndcg_at_k,
                    "retrieval_ms":   r.retrieval_ms,
                    "rerank_ms":      r.rerank_ms,
                    "error":          r.error,
                }
                for r in self.query_results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"[eval] results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(
    dataset:      EvalDataset,
    retriever_fn: Callable,
    reranker_fn:  Optional[Callable] = None,
    k_values:     List[int]          = None,
    verbose:      bool               = True,
    pipeline_config: Dict            = None,
) -> EvalResults:
    """
    Run the evaluation loop over the dataset.

    Parameters
    ──────────
    dataset       : EvalDataset with GoldenSample objects
    retriever_fn  : callable matching signature of retrieve_with_years:
                      (query, doc_type, symbol, year, year_range, speaker_role)
                      → (chunks_or_tuple, resolved_years, explicit_years)
    reranker_fn   : callable matching rerank_separate or rerank.
                    If None, metrics are computed on raw retriever output.
    k_values      : list of k to evaluate at (default [5, 10, 20])
    verbose       : print progress to stdout
    pipeline_config: metadata dict stored in EvalResults for reproducibility

    Returns
    ───────
    EvalResults with aggregate metrics already computed (.aggregate() called).
    """
    k_values = k_values or [5, 10, 20]
    query_results: List[QueryResult] = []

    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    n = len(dataset)

    if verbose:
        print(f"\n[eval] Starting evaluation: {n} samples, k={k_values}")
        print(f"       Reranker: {'enabled' if reranker_fn else 'disabled (retriever only)'}")
        print()

    for idx, sample in enumerate(dataset, 1):
        if verbose:
            print(
                f"  [{idx:>3}/{n}] {sample.query[:55]:<55} "
                f"({sample.symbol or 'N/A'}, {sample.doc_type})",
                end=" ... ",
                flush=True,
            )

        qr = QueryResult(
            query        = sample.query,
            symbol       = sample.symbol,
            doc_type     = sample.doc_type,
            years        = sample.years,
            retrieved_ids= [],
            relevant_ids = sample.relevant_ids,
            k_values     = k_values,
        )

        try:
            # ── Retrieval ────────────────────────────────────────────────────
            t_ret0 = time.perf_counter()
            raw    = retriever_fn(
                query        = sample.query,
                doc_type     = sample.doc_type,
                symbol       = sample.symbol,
                year         = sample.years[0] if sample.years and len(sample.years) == 1 else None,
                year_range   = (
                    (sample.years[0], sample.years[-1])
                    if sample.years and len(sample.years) > 1
                    else None
                ),
            )
            qr.retrieval_ms = (time.perf_counter() - t_ret0) * 1000

            # Unpack (chunks_or_tuple, resolved_years, explicit_years)
            # from retrieve_with_years, or just a plain list from retrieve.
            if isinstance(raw, tuple) and len(raw) == 3:
                chunks_or_tuple, _, _ = raw
            else:
                chunks_or_tuple = raw

            # For "both" doc_type, flatten the tuple
            if isinstance(chunks_or_tuple, tuple):
                annual_chunks, concall_chunks = chunks_or_tuple
            else:
                annual_chunks  = chunks_or_tuple
                concall_chunks = []

            # ── Reranking ────────────────────────────────────────────────────
            if reranker_fn is not None:
                t_rer0 = time.perf_counter()
                if concall_chunks:
                    final_chunks = reranker_fn(
                        sample.query,
                        annual_chunks,
                        concall_chunks,
                    )
                else:
                    final_chunks = reranker_fn(
                        sample.query,
                        annual_chunks,
                        doc_type=sample.doc_type,
                        top_k=max(k_values),
                    )
                qr.rerank_ms = (time.perf_counter() - t_rer0) * 1000
            else:
                final_chunks = (
                    annual_chunks + concall_chunks
                    if concall_chunks
                    else annual_chunks
                )
                # Sort by score descending (retriever already does this, but be safe)
                final_chunks = sorted(final_chunks, key=lambda c: c.score, reverse=True)

            qr.retrieved_ids = [c.chunk_id for c in final_chunks]

            # ── Metrics ──────────────────────────────────────────────────────
            _compute_metrics(qr, sample, k_values)

            if verbose:
                h10 = qr.hit_at_k.get(10, qr.hit_at_k.get(k_values[-1], 0))
                m10 = qr.mrr_at_k.get(10, qr.mrr_at_k.get(k_values[-1], 0))
                rank_str = f"rank={qr.first_hit_rank}" if qr.first_hit_rank else "NO HIT"
                print(f"Hit@{k_values[-1]}={h10:.2f}  MRR={m10:.3f}  {rank_str}")

        except Exception as exc:
            qr.error = str(exc)
            if verbose:
                print(f"ERROR: {exc}")
            log.warning(f"[eval] sample {idx} failed: {exc}")

        query_results.append(qr)

    results = EvalResults(
        query_results    = query_results,
        k_values         = k_values,
        run_timestamp    = run_ts,
        pipeline_config  = pipeline_config or {},
    )
    results.aggregate()

    if verbose:
        results.print_report()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanmean(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def _status_emoji(metric: str, vals: Dict[int, float], k_values: List[int]) -> str:
    """Return a traffic-light emoji based on target thresholds."""
    targets = {
        "Hit Rate":  {10: 0.80, 5: 0.70},
        "MRR":       {10: 0.60, 5: 0.50},
        "Precision": {10: 0.20, 5: 0.30},
        "NDCG":      {10: 0.65, 5: 0.55},
    }
    t = targets.get(metric, {})
    k = 10 if 10 in k_values else k_values[-1]
    v = vals.get(k, float("nan"))
    goal = t.get(k, 0.5)
    if math.isnan(v):
        return "❓"
    if v >= goal:
        return "✅"
    if v >= goal * 0.8:
        return "⚠️ "
    return "❌"


def _print_query_row(r: QueryResult, k: int) -> None:
    hit  = r.hit_at_k.get(k,  float("nan"))
    mrr  = r.mrr_at_k.get(k, float("nan"))
    rank = str(r.first_hit_rank) if r.first_hit_rank else "—"
    err  = " [ERR]" if r.error else ""
    q    = r.query[:52].ljust(52)
    print(
        f"  {q}  "
        f"hit={hit:.2f}  mrr={mrr:.3f}  rank={rank:>4}{err}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder utility — makes it easy to create golden samples
# from manual annotations or a CSV
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Helper to build EvalDataset from various sources.

    Usage:
        builder = DatasetBuilder()
        builder.add_manual(
            query        = "ADANIPORTS revenue FY25",
            symbol       = "ADANIPORTS",
            doc_type     = "annual_report",
            years        = [2025],
            relevant_ids = ["adaniports_fy25_pg42", "adaniports_fy25_pg43"],
        )
        dataset = builder.build()
    """

    def __init__(self):
        self._samples: List[GoldenSample] = []

    def add_manual(
        self,
        query:        str,
        relevant_ids: List[str],
        symbol:       Optional[str]  = None,
        doc_type:     str            = "annual_report",
        years:        Optional[List[int]] = None,
        graded_ids:   Optional[Dict[str, int]] = None,
        notes:        str            = "",
    ) -> "DatasetBuilder":
        self._samples.append(GoldenSample(
            query        = query,
            symbol       = symbol,
            doc_type     = doc_type,
            years        = years,
            relevant_ids = relevant_ids,
            graded_ids   = graded_ids or {},
            notes        = notes,
        ))
        return self

    def from_csv(
        self,
        path: str,
        query_col:   str = "query",
        ids_col:     str = "relevant_ids",   # JSON array or comma-separated
        symbol_col:  str = "symbol",
        doc_type_col:str = "doc_type",
        years_col:   str = "years",
    ) -> "DatasetBuilder":
        """
        Load samples from a CSV file.

        CSV format example:
          query,symbol,doc_type,years,relevant_ids
          "ADANIPORTS revenue FY25","ADANIPORTS","annual_report","[2025]","[""id1"",""id2""]"
        """
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ids = json.loads(row.get(ids_col, "[]"))
                    if isinstance(ids, str):
                        ids = [i.strip() for i in ids.split(",")]
                    years_raw = row.get(years_col, "[]")
                    years = json.loads(years_raw) if years_raw else None
                    self._samples.append(GoldenSample(
                        query        = row[query_col],
                        symbol       = row.get(symbol_col) or None,
                        doc_type     = row.get(doc_type_col, "annual_report"),
                        years        = years,
                        relevant_ids = ids,
                    ))
                except Exception as e:
                    log.warning(f"[eval] CSV row skipped: {e} | row={row}")
        log.info(f"[eval] loaded {len(self._samples)} samples from {path}")
        return self

    def build(self) -> EvalDataset:
        return EvalDataset(samples=list(self._samples))


# ─────────────────────────────────────────────────────────────────────────────
# Quick unit tests (run with: python eval_suite.py)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    print("Running EvalSuite self-test...")
    print("=" * 50)

    # Fake retrieved chunks (just use plain objects with chunk_id + score)
    from dataclasses import make_dataclass
    Chunk = make_dataclass("Chunk", ["chunk_id", "score",
                                     ("text", str, ""), ("metadata", dict, None),
                                     ("vector_score", float, 0.0),
                                     ("bm25_score", float, 0.0)])

    def fake_retriever(query, doc_type="annual_report", symbol=None,
                       year=None, year_range=None, **kw):
        # Simulate retriever returning 20 chunks, correct ones at positions 3 and 7
        chunks = [Chunk(chunk_id=f"chunk_{i:03d}", score=1.0 - i * 0.04) for i in range(20)]
        chunks[2].chunk_id = "correct_a"    # rank 3
        chunks[6].chunk_id = "correct_b"    # rank 7
        return chunks, [2025], [2025]

    dataset = EvalDataset([
        GoldenSample(
            query        = "Test query A",
            symbol       = "TESTCO",
            doc_type     = "annual_report",
            years        = [2025],
            relevant_ids = ["correct_a", "correct_b"],
        ),
        GoldenSample(
            query        = "Test query B — no hit",
            symbol       = "TESTCO",
            doc_type     = "annual_report",
            years        = [2024],
            relevant_ids = ["correct_c"],   # not in results
        ),
    ])

    results = run_eval(
        dataset      = dataset,
        retriever_fn = fake_retriever,
        k_values     = [5, 10],
        verbose      = True,
    )

    assert results.mean_hit_at_k[10] == 0.5, \
        f"Hit@10 expected 0.5, got {results.mean_hit_at_k[10]}"
    assert results.mean_mrr_at_k[10] > 0.0, "MRR@10 should be > 0"
    print("✓ All assertions passed")


if __name__ == "__main__":
    _self_test()