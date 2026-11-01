"""
Evaluation dataset loader.

Reads eval_dataset.json and returns a list of EvalQuery objects.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config import get_logger, settings

logger = get_logger(__name__)


@dataclass
class EvalQuery:
    query_id: str
    question: str
    relevant_doc_ids: list[str]
    expected_keywords: list[str]
    category: str


def load_dataset(path: Path | None = None) -> list[EvalQuery]:
    """
    Load evaluation queries from a JSON file.

    Args:
        path: Path to eval_dataset.json. Defaults to
              ``settings.evaluation_dir / "eval_dataset.json"``.

    Returns:
        List of EvalQuery objects.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    if path is None:
        path = settings.evaluation_dir / "eval_dataset.json"

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in evaluation dataset: {exc}") from exc

    raw_queries = data.get("queries", [])
    if not raw_queries:
        raise ValueError("Evaluation dataset contains no queries.")

    queries: list[EvalQuery] = []
    for item in raw_queries:
        try:
            queries.append(
                EvalQuery(
                    query_id=item["query_id"],
                    question=item["question"],
                    relevant_doc_ids=item["relevant_doc_ids"],
                    expected_keywords=item.get("expected_keywords", []),
                    category=item.get("category", "general_faq"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing required field {exc} in query: {item}") from exc

    logger.info("Loaded %d evaluation queries from %s", len(queries), path)
    return queries
