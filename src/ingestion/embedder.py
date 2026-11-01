"""
Embedder and ChromaDB persistence layer.

Responsibilities:
- Compute SHA-256 hashes of source files to detect changes.
- Skip re-embedding of unchanged documents.
- Delete stale chunks for changed documents before reinserting.
- Store embeddings in a persistent ChromaDB collection.

Public entry point: ``ingest_documents()``.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_logger, settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents

logger = get_logger(__name__)

_HASH_FILE_NAME = "doc_hashes.json"


# ── Hash helpers ──────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stored_hashes(storage_dir: Path) -> dict[str, str]:
    """Load the persisted hash map from disk.  Returns {} if not found."""
    hash_file = storage_dir / _HASH_FILE_NAME
    if hash_file.exists():
        try:
            return json.loads(hash_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Could not read hash file; will re-ingest all documents.")
    return {}


def _save_hashes(storage_dir: Path, hashes: dict[str, str]) -> None:
    """Persist the hash map to disk."""
    hash_file = storage_dir / _HASH_FILE_NAME
    hash_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _build_vector_store(
    chroma_dir: Path,
    collection_name: str,
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """Open (or create) a persistent Chroma collection."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )


def _delete_chunks_for_doc(vector_store: Chroma, doc_id: str) -> None:
    """Remove all chunks belonging to *doc_id* from the collection."""
    try:
        collection = vector_store._collection  # access underlying chromadb collection
        results = collection.get(where={"doc_id": {"$eq": doc_id}})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.debug("Deleted %d stale chunks for '%s'", len(ids_to_delete), doc_id)
    except Exception:
        logger.exception("Failed to delete stale chunks for '%s'", doc_id)


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_documents(
    documents_dir: Path | None = None,
    chroma_dir: Path | None = None,
    storage_dir: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Load, embed, and persist parking knowledge documents into ChromaDB.

    Only documents whose content has changed since the last run are
    re-processed.  Unchanged documents are skipped.

    Args:
        documents_dir:    Source document directory.
        chroma_dir:       Persistent ChromaDB directory.
        storage_dir:      Directory where the hash file is stored.
        collection_name:  ChromaDB collection name.
        embedding_model:  HuggingFace model identifier.
        force:            If True, re-ingest all documents regardless of hash.

    Returns:
        Dict with keys: ingested, skipped, deleted, total_chunks.
    """
    documents_dir = documents_dir or settings.documents_dir
    chroma_dir = chroma_dir or settings.chroma_dir
    storage_dir = storage_dir or settings.storage_dir
    collection_name = collection_name or settings.chroma_collection_name
    embedding_model = embedding_model or settings.embedding_model_name

    settings.ensure_directories()

    logger.info("Starting document ingestion (force=%s)", force)

    # ── 1. Load all source documents ─────────────────────────────────────────
    all_docs = load_documents(documents_dir)
    if not all_docs:
        logger.warning("No documents found; nothing to ingest.")
        return {"ingested": 0, "skipped": 0, "deleted": 0, "total_chunks": 0}

    # ── 2. Determine which documents have changed ─────────────────────────────
    stored_hashes = {} if force else _load_stored_hashes(storage_dir)
    current_hashes: dict[str, str] = {}
    docs_to_ingest: list[Document] = []
    docs_to_delete: list[str] = []

    for doc in all_docs:
        source_path = Path(doc.metadata["source"])
        doc_id = doc.metadata["doc_id"]
        current_hash = _file_hash(source_path)
        current_hashes[doc_id] = current_hash

        if not force and stored_hashes.get(doc_id) == current_hash:
            logger.debug("Skipping unchanged document: %s", doc_id)
        else:
            docs_to_ingest.append(doc)
            if doc_id in stored_hashes:
                docs_to_delete.append(doc_id)  # changed — delete old chunks first

    skipped = len(all_docs) - len(docs_to_ingest)
    logger.info(
        "%d documents to ingest, %d skipped (unchanged)", len(docs_to_ingest), skipped
    )

    if not docs_to_ingest:
        return {"ingested": 0, "skipped": skipped, "deleted": 0, "total_chunks": 0}

    # ── 3. Initialise embeddings and vector store ─────────────────────────────
    logger.info("Loading embedding model: %s", embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = _build_vector_store(chroma_dir, collection_name, embeddings)

    # ── 4. Delete stale chunks ────────────────────────────────────────────────
    for doc_id in docs_to_delete:
        _delete_chunks_for_doc(vector_store, doc_id)

    # ── 5. Chunk and embed new/changed documents ──────────────────────────────
    chunks = chunk_documents(docs_to_ingest)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info("Added %d chunks to ChromaDB collection '%s'", len(chunks), collection_name)

    # ── 6. Persist updated hashes ─────────────────────────────────────────────
    stored_hashes.update(current_hashes)
    _save_hashes(storage_dir, stored_hashes)

    stats = {
        "ingested": len(docs_to_ingest),
        "skipped": skipped,
        "deleted": len(docs_to_delete),
        "total_chunks": len(chunks),
    }
    logger.info("Ingestion complete: %s", stats)
    return stats
