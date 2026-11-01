"""
Tests for src/ingestion — loader, chunker, and embedder.

Embedding calls are mocked so these tests run offline without downloading
any model.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_loads_md_and_txt_files(self, tmp_dir: Path):
        (tmp_dir / "doc_a.md").write_text("# Title\n\nSome content here.", encoding="utf-8")
        (tmp_dir / "doc_b.txt").write_text("Plain text content.", encoding="utf-8")
        (tmp_dir / "ignore.pdf").write_text("Should be ignored.", encoding="utf-8")

        from src.ingestion.loader import load_documents

        docs = load_documents(tmp_dir)
        filenames = [d.metadata["filename"] for d in docs]

        assert len(docs) == 2
        assert "doc_a.md" in filenames
        assert "doc_b.txt" in filenames
        assert "ignore.pdf" not in filenames

    def test_metadata_fields_are_populated(self, tmp_dir: Path):
        (tmp_dir / "facility.md").write_text("Facility info.", encoding="utf-8")

        from src.ingestion.loader import load_documents

        docs = load_documents(tmp_dir)
        meta = docs[0].metadata

        assert meta["doc_id"] == "facility"
        assert meta["filename"] == "facility.md"
        assert "facility.md" in meta["source"]

    def test_cleans_excess_blank_lines(self, tmp_dir: Path):
        raw = "Line one.\n\n\n\n\nLine two."
        (tmp_dir / "messy.txt").write_text(raw, encoding="utf-8")

        from src.ingestion.loader import load_documents

        docs = load_documents(tmp_dir)
        assert "\n\n\n" not in docs[0].page_content

    def test_missing_directory_raises(self):
        from src.ingestion.loader import load_documents

        with pytest.raises(FileNotFoundError):
            load_documents(Path("/nonexistent/path/xyz"))

    def test_empty_directory_returns_empty_list(self, tmp_dir: Path):
        from src.ingestion.loader import load_documents

        docs = load_documents(tmp_dir)
        assert docs == []

    def test_documents_sorted_by_filename(self, tmp_dir: Path):
        (tmp_dir / "z_last.md").write_text("Last.", encoding="utf-8")
        (tmp_dir / "a_first.md").write_text("First.", encoding="utf-8")

        from src.ingestion.loader import load_documents

        docs = load_documents(tmp_dir)
        assert docs[0].metadata["filename"] == "a_first.md"
        assert docs[1].metadata["filename"] == "z_last.md"


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def _make_doc(self, content: str, doc_id: str = "test_doc") -> Document:
        return Document(
            page_content=content,
            metadata={"doc_id": doc_id, "source": f"/docs/{doc_id}.md", "filename": f"{doc_id}.md"},
        )

    def test_short_document_produces_single_chunk(self):
        from src.ingestion.chunker import chunk_documents

        doc = self._make_doc("Short content.")
        chunks = chunk_documents([doc], chunk_size=500)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short content."

    def test_long_document_produces_multiple_chunks(self):
        from src.ingestion.chunker import chunk_documents

        long_text = ("This is a sentence. " * 60)  # ~1200 chars
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1

    def test_chunk_metadata_inherits_doc_id(self):
        from src.ingestion.chunker import chunk_documents

        doc = self._make_doc("Some content.", doc_id="facility_overview")
        chunks = chunk_documents([doc])

        assert all(c.metadata["doc_id"] == "facility_overview" for c in chunks)

    def test_chunk_id_is_deterministic(self):
        from src.ingestion.chunker import chunk_documents

        doc = self._make_doc("Deterministic content.", doc_id="faq")
        chunks1 = chunk_documents([doc])
        chunks2 = chunk_documents([doc])

        assert [c.metadata["chunk_id"] for c in chunks1] == [
            c.metadata["chunk_id"] for c in chunks2
        ]

    def test_chunk_index_is_sequential(self):
        from src.ingestion.chunker import chunk_documents

        long_text = ("Word " * 200)
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_input_returns_empty_list(self):
        from src.ingestion.chunker import chunk_documents

        assert chunk_documents([]) == []


# ── Embedder / hash tests ─────────────────────────────────────────────────────

class TestEmbedder:
    def test_file_hash_is_deterministic(self, tmp_dir: Path):
        from src.ingestion.embedder import _file_hash

        f = tmp_dir / "test.txt"
        f.write_bytes(b"hello world")
        assert _file_hash(f) == _file_hash(f)

    def test_different_content_gives_different_hash(self, tmp_dir: Path):
        from src.ingestion.embedder import _file_hash

        f1 = tmp_dir / "a.txt"
        f2 = tmp_dir / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert _file_hash(f1) != _file_hash(f2)

    def test_hash_persistence_round_trip(self, tmp_dir: Path):
        from src.ingestion.embedder import _load_stored_hashes, _save_hashes

        hashes = {"doc_a": "abc123", "doc_b": "def456"}
        _save_hashes(tmp_dir, hashes)
        loaded = _load_stored_hashes(tmp_dir)
        assert loaded == hashes

    def test_ingest_documents_skips_unchanged(self, tmp_storage: dict):
        """Second ingestion call should skip all documents (hashes unchanged)."""
        from src.ingestion.embedder import ingest_documents

        docs_dir = tmp_storage["documents_dir"]
        (docs_dir / "info.md").write_text("Parking info content.", encoding="utf-8")

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384]
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}

        with (
            patch("src.ingestion.embedder.HuggingFaceEmbeddings", return_value=mock_embeddings),
            patch("src.ingestion.embedder._build_vector_store", return_value=mock_vs),
        ):
            stats1 = ingest_documents(
                documents_dir=docs_dir,
                chroma_dir=tmp_storage["chroma_dir"],
                storage_dir=tmp_storage["chroma_dir"],
            )
            stats2 = ingest_documents(
                documents_dir=docs_dir,
                chroma_dir=tmp_storage["chroma_dir"],
                storage_dir=tmp_storage["chroma_dir"],
            )

        assert stats1["ingested"] == 1
        assert stats2["ingested"] == 0
        assert stats2["skipped"] == 1

    def test_ingest_documents_re_ingests_changed_file(self, tmp_storage: dict):
        """A modified file should be detected and re-ingested."""
        from src.ingestion.embedder import ingest_documents

        docs_dir = tmp_storage["documents_dir"]
        doc_file = docs_dir / "info.md"
        doc_file.write_text("Version 1 content.", encoding="utf-8")

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384]
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}

        with (
            patch("src.ingestion.embedder.HuggingFaceEmbeddings", return_value=mock_embeddings),
            patch("src.ingestion.embedder._build_vector_store", return_value=mock_vs),
        ):
            ingest_documents(
                documents_dir=docs_dir,
                chroma_dir=tmp_storage["chroma_dir"],
                storage_dir=tmp_storage["chroma_dir"],
            )

            # Modify the file
            doc_file.write_text("Version 2 content — completely different.", encoding="utf-8")

            stats = ingest_documents(
                documents_dir=docs_dir,
                chroma_dir=tmp_storage["chroma_dir"],
                storage_dir=tmp_storage["chroma_dir"],
            )

        assert stats["ingested"] == 1
        assert stats["deleted"] == 1  # old chunks should be deleted first

    def test_force_flag_re_ingests_all(self, tmp_storage: dict):
        """force=True should re-ingest all documents regardless of hash."""
        from src.ingestion.embedder import ingest_documents

        docs_dir = tmp_storage["documents_dir"]
        (docs_dir / "info.md").write_text("Some content.", encoding="utf-8")

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384]
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_vs = MagicMock()
        mock_vs._collection.get.return_value = {"ids": []}

        with (
            patch("src.ingestion.embedder.HuggingFaceEmbeddings", return_value=mock_embeddings),
            patch("src.ingestion.embedder._build_vector_store", return_value=mock_vs),
        ):
            ingest_documents(
                documents_dir=docs_dir,
                chroma_dir=tmp_storage["chroma_dir"],
                storage_dir=tmp_storage["chroma_dir"],
            )
            stats = ingest_documents(
                documents_dir=docs_dir,
                chroma_dir=tmp_storage["chroma_dir"],
                storage_dir=tmp_storage["chroma_dir"],
                force=True,
            )

        assert stats["ingested"] == 1
        assert stats["skipped"] == 0
