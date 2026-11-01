"""
Tests for src/evaluation — dataset loader, metric functions, and report runner.

All ChromaDB / HuggingFace calls are mocked so these tests run offline.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(doc_id: str) -> Document:
    return Document(
        page_content=f"Content of {doc_id}.",
        metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}_chunk_0"},
    )


def _write_dataset(tmp_path: Path, queries: list[dict]) -> Path:
    path = tmp_path / "eval_dataset.json"
    path.write_text(
        json.dumps({"metadata": {}, "queries": queries}),
        encoding="utf-8",
    )
    return path


# ── Dataset loader tests ──────────────────────────────────────────────────────

class TestDatasetLoader:
    def test_loads_valid_dataset(self, tmp_path: Path):
        from src.evaluation.dataset import load_dataset

        path = _write_dataset(tmp_path, [
            {
                "query_id": "q01",
                "question": "How many spaces?",
                "relevant_doc_ids": ["facility_overview"],
                "expected_keywords": ["850"],
                "category": "general_faq",
            }
        ])
        queries = load_dataset(path)
        assert len(queries) == 1
        assert queries[0].query_id == "q01"
        assert queries[0].relevant_doc_ids == ["facility_overview"]

    def test_missing_file_raises(self, tmp_path: Path):
        from src.evaluation.dataset import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_empty_queries_raises(self, tmp_path: Path):
        from src.evaluation.dataset import load_dataset
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"queries": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="no queries"):
            load_dataset(path)

    def test_missing_required_field_raises(self, tmp_path: Path):
        from src.evaluation.dataset import load_dataset
        path = _write_dataset(tmp_path, [{"query_id": "q01"}])  # missing 'question'
        with pytest.raises(ValueError):
            load_dataset(path)

    def test_loads_real_eval_dataset(self):
        """The shipped eval_dataset.json must load without errors."""
        from src.evaluation.dataset import load_dataset
        queries = load_dataset()
        assert len(queries) == 15
        assert all(q.query_id for q in queries)
        assert all(q.relevant_doc_ids for q in queries)


# ── Precision@K tests ─────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_relevant_gives_one(self):
        from src.evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant_gives_zero(self):
        from src.evaluation.metrics import precision_at_k
        assert precision_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_half_relevant(self):
        from src.evaluation.metrics import precision_at_k
        result = precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4)
        assert abs(result - 0.5) < 1e-9

    def test_k_zero_returns_zero(self):
        from src.evaluation.metrics import precision_at_k
        assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_empty_retrieved_returns_zero(self):
        from src.evaluation.metrics import precision_at_k
        assert precision_at_k([], {"a"}, k=4) == 0.0

    def test_only_top_k_considered(self):
        from src.evaluation.metrics import precision_at_k
        # Relevant doc appears at position 3 (0-indexed 2), K=2 → not counted
        result = precision_at_k(["x", "y", "a"], {"a"}, k=2)
        assert result == 0.0


# ── Recall@K tests ────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_all_relevant_found(self):
        from src.evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_found_gives_zero(self):
        from src.evaluation.metrics import recall_at_k
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_recall(self):
        from src.evaluation.metrics import recall_at_k
        result = recall_at_k(["a", "x", "y"], {"a", "b"}, k=3)
        assert abs(result - 0.5) < 1e-9

    def test_no_relevant_docs_returns_one(self):
        from src.evaluation.metrics import recall_at_k
        assert recall_at_k(["a", "b"], set(), k=4) == 1.0

    def test_k_larger_than_retrieved(self):
        from src.evaluation.metrics import recall_at_k
        result = recall_at_k(["a"], {"a", "b"}, k=10)
        assert abs(result - 0.5) < 1e-9


# ── Reciprocal rank tests ─────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_relevant_at_rank_1(self):
        from src.evaluation.metrics import reciprocal_rank
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_relevant_at_rank_2(self):
        from src.evaluation.metrics import reciprocal_rank
        assert abs(reciprocal_rank(["x", "a", "b"], {"a"}) - 0.5) < 1e-9

    def test_relevant_at_rank_4(self):
        from src.evaluation.metrics import reciprocal_rank
        assert abs(reciprocal_rank(["x", "y", "z", "a"], {"a"}) - 0.25) < 1e-9

    def test_not_found_returns_zero(self):
        from src.evaluation.metrics import reciprocal_rank
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_uses_first_relevant_only(self):
        from src.evaluation.metrics import reciprocal_rank
        # "a" at rank 1, "b" at rank 2 → RR = 1.0 (first relevant)
        assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0


# ── MRR tests ─────────────────────────────────────────────────────────────────

class TestMRR:
    def test_perfect_mrr(self):
        from src.evaluation.metrics import mean_reciprocal_rank
        results = [(["a"], {"a"}), (["b"], {"b"})]
        assert mean_reciprocal_rank(results) == 1.0

    def test_mixed_mrr(self):
        from src.evaluation.metrics import mean_reciprocal_rank
        # Query 1: RR=1.0, Query 2: RR=0.5 → MRR=0.75
        results = [(["a", "b"], {"a"}), (["x", "b"], {"b"})]
        assert abs(mean_reciprocal_rank(results) - 0.75) < 1e-9

    def test_empty_list_returns_zero(self):
        from src.evaluation.metrics import mean_reciprocal_rank
        assert mean_reciprocal_rank([]) == 0.0

    def test_no_relevant_found_gives_zero(self):
        from src.evaluation.metrics import mean_reciprocal_rank
        results = [(["x", "y"], {"a"}), (["p", "q"], {"b"})]
        assert mean_reciprocal_rank(results) == 0.0


# ── compute_metrics tests ─────────────────────────────────────────────────────

class TestComputeMetrics:
    def _make_result(self, qid, retrieved, relevant, ret_ms=50.0):
        from src.evaluation.metrics import QueryResult
        return QueryResult(
            query_id=qid,
            question=f"Question {qid}",
            relevant_doc_ids=relevant,
            retrieved_doc_ids=retrieved,
            retrieval_latency_ms=ret_ms,
            total_latency_ms=ret_ms,
        )

    def test_returns_eval_metrics(self):
        from src.evaluation.metrics import EvalMetrics, compute_metrics
        results = [self._make_result("q1", ["a", "b"], ["a"])]
        m = compute_metrics(results, k=4)
        assert isinstance(m, EvalMetrics)

    def test_perfect_retrieval(self):
        from src.evaluation.metrics import compute_metrics
        results = [
            self._make_result("q1", ["a"], ["a"]),
            self._make_result("q2", ["b"], ["b"]),
        ]
        m = compute_metrics(results, k=4)
        assert m.precision_at_k > 0
        assert m.mrr == 1.0

    def test_empty_results_returns_zero_metrics(self):
        from src.evaluation.metrics import compute_metrics
        m = compute_metrics([], k=4)
        assert m.num_queries == 0
        assert m.precision_at_k == 0.0

    def test_latency_stats_computed(self):
        from src.evaluation.metrics import compute_metrics
        results = [
            self._make_result("q1", ["a"], ["a"], ret_ms=100.0),
            self._make_result("q2", ["b"], ["b"], ret_ms=200.0),
        ]
        m = compute_metrics(results, k=4)
        assert abs(m.avg_retrieval_latency_ms - 150.0) < 1e-9
        assert m.min_retrieval_latency_ms == 100.0
        assert m.max_retrieval_latency_ms == 200.0


# ── Report / run_evaluation tests ─────────────────────────────────────────────

class TestRunEvaluation:
    def _mock_retriever(self, doc_ids_per_call: list[list[str]]):
        """Build a mock retriever that returns specified docs on successive calls."""
        call_count = {"n": 0}

        def _invoke(question):
            idx = min(call_count["n"], len(doc_ids_per_call) - 1)
            call_count["n"] += 1
            return [_make_doc(did) for did in doc_ids_per_call[idx]]

        mock = MagicMock()
        mock.invoke.side_effect = _invoke
        return mock

    def test_run_evaluation_returns_metrics_dict(self, tmp_path: Path):
        from src.evaluation.report import run_evaluation

        path = _write_dataset(tmp_path, [
            {
                "query_id": "q01", "question": "How many spaces?",
                "relevant_doc_ids": ["facility_overview"],
                "expected_keywords": ["850"], "category": "general_faq",
            }
        ])
        retriever = self._mock_retriever([["facility_overview"]])
        result = run_evaluation(
            dataset_path=path,
            reports_dir=tmp_path / "reports",
            k=4,
            retriever=retriever,
        )
        assert "precision_at_k" in result
        assert "recall_at_k" in result
        assert "mrr" in result
        assert result["num_queries"] == 1

    def test_report_file_is_created(self, tmp_path: Path):
        from src.evaluation.report import run_evaluation

        path = _write_dataset(tmp_path, [
            {
                "query_id": "q01", "question": "Location?",
                "relevant_doc_ids": ["location_directions"],
                "expected_keywords": [], "category": "general_faq",
            }
        ])
        reports_dir = tmp_path / "reports"
        retriever = self._mock_retriever([["location_directions"]])
        result = run_evaluation(
            dataset_path=path,
            reports_dir=reports_dir,
            k=4,
            retriever=retriever,
        )
        report_path = Path(result["report_path"])
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Evaluation Report" in content
        assert "Precision" in content

    def test_perfect_retrieval_gives_mrr_one(self, tmp_path: Path):
        from src.evaluation.report import run_evaluation

        path = _write_dataset(tmp_path, [
            {
                "query_id": "q01", "question": "Q1?",
                "relevant_doc_ids": ["facility_overview"],
                "expected_keywords": [], "category": "general_faq",
            },
            {
                "query_id": "q02", "question": "Q2?",
                "relevant_doc_ids": ["faq"],
                "expected_keywords": [], "category": "general_faq",
            },
        ])
        retriever = self._mock_retriever(
            [["facility_overview", "faq"], ["faq", "facility_overview"]]
        )
        result = run_evaluation(
            dataset_path=path,
            reports_dir=tmp_path / "reports",
            k=4,
            retriever=retriever,
        )
        assert result["mrr"] == 1.0

    def test_zero_retrieval_gives_zero_precision(self, tmp_path: Path):
        from src.evaluation.report import run_evaluation

        path = _write_dataset(tmp_path, [
            {
                "query_id": "q01", "question": "Q?",
                "relevant_doc_ids": ["faq"],
                "expected_keywords": [], "category": "general_faq",
            }
        ])
        retriever = self._mock_retriever([[]])  # returns nothing
        result = run_evaluation(
            dataset_path=path,
            reports_dir=tmp_path / "reports",
            k=4,
            retriever=retriever,
        )
        assert result["precision_at_k"] == 0.0
        assert result["mrr"] == 0.0
