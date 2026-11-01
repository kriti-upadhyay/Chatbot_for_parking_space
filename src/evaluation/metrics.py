"""
Retrieval evaluation metrics.

All metric functions are pure Python — no external dependencies — so they
are fast and easy to unit-test in isolation.

Metrics implemented:
    precision_at_k       — fraction of top-K retrieved docs that are relevant
    recall_at_k          — fraction of all relevant docs found in top-K
    reciprocal_rank      — 1/rank of the first relevant doc (0 if none found)
    mean_reciprocal_rank — MRR averaged across all queries
    compute_metrics      — run all metrics on a list of QueryResult objects
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from src.config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-query result container
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Holds the raw output of running one evaluation query."""

    query_id: str
    question: str
    relevant_doc_ids: list[str]          # ground-truth relevant doc IDs
    retrieved_doc_ids: list[str]         # doc IDs returned by the retriever
    answer: str = ""                     # LLM answer (empty if not run)
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    had_relevant_context: bool = False


# ---------------------------------------------------------------------------
# Aggregate metrics container
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics across all queries."""

    k: int
    num_queries: int

    # Retrieval quality
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0

    # Per-query breakdowns (parallel lists — same index = same query)
    per_query_precision: list[float] = field(default_factory=list)
    per_query_recall: list[float] = field(default_factory=list)
    per_query_rr: list[float] = field(default_factory=list)

    # Latency (ms)
    avg_retrieval_latency_ms: float = 0.0
    min_retrieval_latency_ms: float = 0.0
    max_retrieval_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Fraction of the top-K retrieved documents that are relevant.

    P@K = |{relevant} ∩ {top-K retrieved}| / K

    Returns 0.0 if K is 0 or no documents were retrieved.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Fraction of all relevant documents that appear in the top-K results.

    R@K = |{relevant} ∩ {top-K retrieved}| / |relevant|

    Returns 1.0 if there are no relevant documents (vacuously true).
    Returns 0.0 if K is 0.
    """
    if not relevant_ids:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    1 / rank of the first relevant document in the retrieved list.

    Returns 0.0 if no relevant document appears in the list.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results: list[tuple[list[str], set[str]]],
) -> float:
    """
    Average reciprocal rank across multiple queries.

    Args:
        results: List of (retrieved_ids, relevant_ids) pairs — one per query.

    Returns:
        MRR in [0.0, 1.0].
    """
    if not results:
        return 0.0
    rrs = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in results]
    return sum(rrs) / len(rrs)


# ---------------------------------------------------------------------------
# Aggregate metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[QueryResult], k: int) -> EvalMetrics:
    """
    Compute P@K, R@K, MRR, and latency statistics across all query results.

    Args:
        results: List of QueryResult objects (one per eval query).
        k:       Retrieval depth used for precision/recall.

    Returns:
        EvalMetrics instance with all computed values.
    """
    if not results:
        return EvalMetrics(k=k, num_queries=0)

    per_precision: list[float] = []
    per_recall: list[float] = []
    per_rr: list[float] = []
    retrieval_latencies: list[float] = []
    llm_latencies: list[float] = []
    total_latencies: list[float] = []

    for qr in results:
        relevant = set(qr.relevant_doc_ids)
        retrieved = qr.retrieved_doc_ids

        per_precision.append(precision_at_k(retrieved, relevant, k))
        per_recall.append(recall_at_k(retrieved, relevant, k))
        per_rr.append(reciprocal_rank(retrieved, relevant))
        retrieval_latencies.append(qr.retrieval_latency_ms)
        if qr.llm_latency_ms > 0:
            llm_latencies.append(qr.llm_latency_ms)
        total_latencies.append(qr.total_latency_ms)

    metrics = EvalMetrics(
        k=k,
        num_queries=len(results),
        precision_at_k=statistics.mean(per_precision),
        recall_at_k=statistics.mean(per_recall),
        mrr=statistics.mean(per_rr),
        per_query_precision=per_precision,
        per_query_recall=per_recall,
        per_query_rr=per_rr,
        avg_retrieval_latency_ms=statistics.mean(retrieval_latencies),
        min_retrieval_latency_ms=min(retrieval_latencies),
        max_retrieval_latency_ms=max(retrieval_latencies),
        avg_llm_latency_ms=statistics.mean(llm_latencies) if llm_latencies else 0.0,
        avg_total_latency_ms=statistics.mean(total_latencies),
    )

    logger.info(
        "Metrics computed | P@%d=%.3f R@%d=%.3f MRR=%.3f "
        "avg_retrieval=%.0fms avg_total=%.0fms",
        k, metrics.precision_at_k,
        k, metrics.recall_at_k,
        metrics.mrr,
        metrics.avg_retrieval_latency_ms,
        metrics.avg_total_latency_ms,
    )
    return metrics
