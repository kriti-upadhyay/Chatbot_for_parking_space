"""
Evaluation runner and report generator.

``run_evaluation()`` is the public entry point.  It:
  1. Loads the evaluation dataset.
  2. Runs each query through the retriever (and optionally the LLM).
  3. Computes P@K, R@K, MRR, and latency statistics.
  4. Writes a human-readable Markdown report to the reports directory.
  5. Returns the computed metrics dict.

The retriever can be injected for testing; if None it is built from settings
(requires a populated ChromaDB and the HuggingFace model).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_logger, settings
from src.evaluation.dataset import EvalQuery, load_dataset
from src.evaluation.metrics import (
    EvalMetrics,
    QueryResult,
    compute_metrics,
)

logger = get_logger(__name__)

_K_DEFAULT = 4


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def _run_query(
    query: EvalQuery,
    retriever,
    k: int,
) -> QueryResult:
    """Run a single evaluation query and return a QueryResult."""
    t0 = time.perf_counter()

    try:
        docs = retriever.invoke(query.question)
    except Exception:
        logger.exception("Retrieval failed for query %s", query.query_id)
        elapsed = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query_id=query.query_id,
            question=query.question,
            relevant_doc_ids=query.relevant_doc_ids,
            retrieved_doc_ids=[],
            retrieval_latency_ms=elapsed,
            total_latency_ms=elapsed,
        )

    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Deduplicate doc_ids while preserving rank order
    seen: set[str] = set()
    retrieved_ids: list[str] = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id", "unknown")
        if doc_id not in seen:
            seen.add(doc_id)
            retrieved_ids.append(doc_id)

    total_ms = (time.perf_counter() - t0) * 1000

    return QueryResult(
        query_id=query.query_id,
        question=query.question,
        relevant_doc_ids=query.relevant_doc_ids,
        retrieved_doc_ids=retrieved_ids,
        retrieval_latency_ms=retrieval_ms,
        total_latency_ms=total_ms,
        had_relevant_context=bool(set(retrieved_ids) & set(query.relevant_doc_ids)),
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _observations(metrics: EvalMetrics) -> str:
    lines: list[str] = []

    p = metrics.precision_at_k
    r = metrics.recall_at_k
    mrr = metrics.mrr

    if p >= 0.7:
        lines.append(
            f"- Precision@{metrics.k} ({p:.3f}) is strong, indicating that most "
            "retrieved chunks are directly relevant to the user's question."
        )
    elif p >= 0.4:
        lines.append(
            f"- Precision@{metrics.k} ({p:.3f}) is moderate. Consider increasing "
            "chunk granularity or tuning the similarity threshold."
        )
    else:
        lines.append(
            f"- Precision@{metrics.k} ({p:.3f}) is low. The documents may need "
            "re-chunking (smaller chunks) or the embedding model may need upgrading."
        )

    if r >= 0.7:
        lines.append(
            f"- Recall@{metrics.k} ({r:.3f}) is high, meaning most relevant "
            "documents appear in the top-K results."
        )
    else:
        lines.append(
            f"- Recall@{metrics.k} ({r:.3f}) suggests some relevant documents are "
            "not retrieved. Increasing K or using MMR retrieval could help."
        )

    if mrr >= 0.7:
        lines.append(
            f"- MRR ({mrr:.3f}) is strong — the first relevant result typically "
            "appears near the top of the ranked list."
        )
    else:
        lines.append(
            f"- MRR ({mrr:.3f}) indicates the first relevant result sometimes "
            "appears lower in the ranking. Re-ranking with a cross-encoder could improve this."
        )

    avg_ret = metrics.avg_retrieval_latency_ms
    if avg_ret < 100:
        lines.append(
            f"- Retrieval latency (avg {avg_ret:.1f} ms) is excellent."
        )
    elif avg_ret < 300:
        lines.append(
            f"- Retrieval latency (avg {avg_ret:.1f} ms) is acceptable for "
            "interactive use."
        )
    else:
        lines.append(
            f"- Retrieval latency (avg {avg_ret:.1f} ms) is high. Consider "
            "using a GPU for embeddings or reducing collection size."
        )

    lines.append(
        "- Possible improvements: (1) Use a larger embedding model such as "
        "all-mpnet-base-v2 for better semantic matching. "
        "(2) Apply MMR (Maximal Marginal Relevance) retrieval to reduce redundancy. "
        "(3) Add metadata filters (by document category) to narrow search scope."
    )

    return "\n".join(lines)


def _write_report(
    metrics: EvalMetrics,
    results: list[QueryResult],
    output_path: Path,
    dataset_path: Path,
) -> None:
    """Write a Markdown evaluation report to *output_path*."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    k = metrics.k

    lines: list[str] = [
        "# CityPark RAG System — Evaluation Report",
        "",
        f"**Generated:** {now}",
        f"**Dataset:** `{dataset_path}`",
        f"**K (retrieval depth):** {k}",
        f"**Queries evaluated:** {metrics.num_queries}",
        "",
        "---",
        "",
        "## 1. Methodology",
        "",
        "This evaluation measures the quality of the retrieval component of the "
        "CityPark parking chatbot RAG system.",
        "",
        "**Process:**",
        "1. Each evaluation question is embedded using the HuggingFace "
        "`all-MiniLM-L6-v2` model.",
        "2. The top-K most similar document chunks are retrieved from ChromaDB "
        "using cosine similarity.",
        "3. Retrieved document IDs are compared against ground-truth relevant "
        "document IDs from the evaluation dataset.",
        "4. Standard information retrieval metrics are computed.",
        "",
        "**Metrics used:**",
        f"- **Precision@{k}**: Fraction of top-{k} retrieved documents that are relevant.",
        f"- **Recall@{k}**: Fraction of all relevant documents found in top-{k} results.",
        "- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of the first "
        "relevant result across all queries.",
        "- **Retrieval latency**: Wall-clock time for ChromaDB similarity search (ms).",
        "",
        "---",
        "",
        "## 2. Dataset Description",
        "",
        f"The evaluation dataset contains **{metrics.num_queries} questions** "
        "covering all 6 parking knowledge documents:",
        "",
        "| Document | Topics Covered |",
        "|---|---|",
        "| `facility_overview` | Capacity, floors, CCTV, accessibility, contacts |",
        "| `location_directions` | Address, driving routes, landmarks, transport |",
        "| `rules_policies` | Speed limit, PCN fines, EV bay rules, prohibited acts |",
        "| `facilities_amenities` | Pay stations, EV chargers, lifts, toilets |",
        "| `reservation_guide` | How to book, modify, cancel; arrival instructions |",
        "| `faq` | Common questions: height, lost tickets, overnight, payment |",
        "",
        "Questions are drawn from realistic user scenarios — the same questions "
        "a member of the public might type into the chatbot.",
        "",
        "---",
        "",
        "## 3. Retrieval Metrics",
        "",
        f"| Metric | Value |",
        "|---|---|",
        f"| Precision@{k} | **{metrics.precision_at_k:.3f}** |",
        f"| Recall@{k} | **{metrics.recall_at_k:.3f}** |",
        f"| MRR | **{metrics.mrr:.3f}** |",
        "",
        "---",
        "",
        "## 4. Performance Measurements",
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| Avg retrieval latency | {metrics.avg_retrieval_latency_ms:.1f} ms |",
        f"| Min retrieval latency | {metrics.min_retrieval_latency_ms:.1f} ms |",
        f"| Max retrieval latency | {metrics.max_retrieval_latency_ms:.1f} ms |",
        f"| Avg total latency | {metrics.avg_total_latency_ms:.1f} ms |",
        "",
        "---",
        "",
        "## 5. Per-Query Results",
        "",
        f"| Query ID | Question (truncated) | P@{k} | R@{k} | RR | Retrieval (ms) | Retrieved Docs |",
        "|---|---|---|---|---|---|---|",
    ]

    for qr, p, r, rr in zip(
        results,
        metrics.per_query_precision,
        metrics.per_query_recall,
        metrics.per_query_rr,
    ):
        q_short = qr.question[:50] + ("…" if len(qr.question) > 50 else "")
        retrieved_str = ", ".join(qr.retrieved_doc_ids[:k]) if qr.retrieved_doc_ids else "—"
        lines.append(
            f"| {qr.query_id} | {q_short} | {p:.2f} | {r:.2f} | {rr:.2f} "
            f"| {qr.retrieval_latency_ms:.0f} | {retrieved_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Observations and Possible Improvements",
        "",
        _observations(metrics),
        "",
        "---",
        "",
        "*Report generated by `src/evaluation/report.py`*",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Evaluation report written to %s", output_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: Path | None = None,
    reports_dir: Path | None = None,
    k: int = _K_DEFAULT,
    retriever=None,
) -> dict[str, Any]:
    """
    Run the full retrieval evaluation and write a report.

    Args:
        dataset_path: Path to eval_dataset.json.
        reports_dir:  Directory to write the report file.
        k:            Retrieval depth for P@K / R@K.
        retriever:    Pre-built LangChain retriever. Built from settings if None.

    Returns:
        Dict with keys: precision_at_k, recall_at_k, mrr,
        avg_retrieval_latency_ms, avg_total_latency_ms, num_queries,
        report_path.
    """
    dataset_path = dataset_path or (settings.evaluation_dir / "eval_dataset.json")
    reports_dir = reports_dir or settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Build retriever if not injected ──────────────────────────────────────
    if retriever is None:
        logger.info("Building retriever from settings for evaluation...")
        from src.rag.retriever import build_retriever
        retriever = build_retriever(k=k)

    # ── Load dataset ─────────────────────────────────────────────────────────
    queries = load_dataset(dataset_path)
    logger.info("Running evaluation on %d queries (k=%d)...", len(queries), k)

    # ── Run each query ────────────────────────────────────────────────────────
    results: list[QueryResult] = []
    for query in queries:
        logger.debug("Evaluating query %s: %s", query.query_id, query.question)
        result = _run_query(query, retriever, k)
        results.append(result)

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = compute_metrics(results, k=k)

    # ── Write report ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"eval_report_{timestamp}.md"
    _write_report(metrics, results, report_path, dataset_path)

    summary = {
        "precision_at_k": round(metrics.precision_at_k, 4),
        "recall_at_k": round(metrics.recall_at_k, 4),
        "mrr": round(metrics.mrr, 4),
        "avg_retrieval_latency_ms": round(metrics.avg_retrieval_latency_ms, 1),
        "avg_total_latency_ms": round(metrics.avg_total_latency_ms, 1),
        "num_queries": metrics.num_queries,
        "report_path": str(report_path),
    }

    print("\n=== Evaluation Results ===")
    for key, val in summary.items():
        if key != "report_path":
            print(f"  {key}: {val}")
    print(f"\nReport written to: {report_path}")

    return summary
