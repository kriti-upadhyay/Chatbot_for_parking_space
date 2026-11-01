"""
ChromaDB retriever initialisation.

Wraps the persistent ChromaDB collection as a LangChain retriever so it
can be used both standalone and as part of a LangGraph node.
"""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_logger, settings

logger = get_logger(__name__)


def build_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Initialise the HuggingFace embedding model."""
    model = model_name or settings.embedding_model_name
    logger.info("Loading embedding model: %s", model)
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    embeddings: HuggingFaceEmbeddings | None = None,
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Open the persistent ChromaDB collection."""
    return Chroma(
        collection_name=collection_name or settings.chroma_collection_name,
        embedding_function=embeddings or build_embeddings(),
        persist_directory=str(chroma_dir or settings.chroma_dir),
    )


def build_retriever(
    k: int | None = None,
    embeddings: HuggingFaceEmbeddings | None = None,
    chroma_dir: Path | None = None,
    collection_name: str | None = None,
) -> VectorStoreRetriever:
    """
    Return a configured vector-store retriever.

    Args:
        k:               Number of documents to retrieve per query.
        embeddings:      Pre-built embeddings (built fresh if None).
        chroma_dir:      ChromaDB persistence directory.
        collection_name: ChromaDB collection name.
    """
    k = k if k is not None else settings.retrieval_k
    vector_store = build_vector_store(embeddings, chroma_dir, collection_name)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    logger.debug("Retriever built (k=%d, collection=%s)", k, collection_name or settings.chroma_collection_name)
    return retriever
