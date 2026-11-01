"""
Document chunker.

Splits LangChain Documents into smaller, overlapping chunks suitable for
embedding and vector retrieval.  Chunk metadata preserves the original
source so downstream retrieval can trace results back to their document.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_logger

logger = get_logger(__name__)

_DEFAULT_CHUNK_SIZE = 600
_DEFAULT_CHUNK_OVERLAP = 80


def chunk_documents(
    documents: list[Document],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split *documents* into overlapping text chunks.

    Each chunk inherits all metadata from its parent document and receives
    an additional ``chunk_index`` field for stable ID generation.

    Args:
        documents:     Documents to split.
        chunk_size:    Maximum character length of each chunk.
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        Flat list of chunk Documents ordered by (doc_id, chunk_index).
    """
    if not documents:
        logger.warning("chunk_documents called with an empty document list")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        raw_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(raw_chunks):
            chunk.metadata["chunk_index"] = idx
            # Stable, deterministic ID used by the embedder for upserts.
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata['doc_id']}_chunk_{idx}"
            )
            all_chunks.append(chunk)

        logger.debug(
            "Chunked '%s' into %d pieces", doc.metadata.get("doc_id"), len(raw_chunks)
        )

    logger.info(
        "Produced %d chunks from %d documents", len(all_chunks), len(documents)
    )
    return all_chunks
