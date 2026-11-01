"""
Document loader for parking knowledge files.

Reads all .md and .txt files from the documents directory, applies light
text cleaning, and returns a list of LangChain Document objects with
source metadata attached.
"""

import re
from pathlib import Path

from langchain_core.documents import Document

from src.config import get_logger, settings

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".md", ".txt"}


def _clean_text(text: str) -> str:
    """
    Normalise whitespace while preserving paragraph structure.

    - Strip leading/trailing whitespace from each line.
    - Collapse runs of more than two blank lines to two blank lines.
    - Remove trailing whitespace left after markdown heading markers.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    joined = "\n".join(lines)
    # Collapse 3+ consecutive blank lines → 2
    cleaned = re.sub(r"\n{3,}", "\n\n", joined)
    return cleaned.strip()


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    """
    Load and clean all supported documents from *documents_dir*.

    Args:
        documents_dir: Directory to scan.  Defaults to ``settings.documents_dir``.

    Returns:
        List of Document objects, one per file, sorted by filename.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if documents_dir is None:
        documents_dir = settings.documents_dir

    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    paths = sorted(
        p for p in documents_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    )

    if not paths:
        logger.warning("No supported documents found in %s", documents_dir)
        return []

    documents: list[Document] = []
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
            content = _clean_text(raw)
            if not content:
                logger.warning("Skipping empty file: %s", path.name)
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "doc_id": path.stem,
                        "filename": path.name,
                    },
                )
            )
            logger.debug("Loaded document: %s (%d chars)", path.name, len(content))
        except Exception:
            logger.exception("Failed to load document: %s", path.name)

    logger.info("Loaded %d documents from %s", len(documents), documents_dir)
    return documents
