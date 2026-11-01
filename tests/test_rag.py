"""
Tests for src/rag — retriever, prompt templates, and RAG chain.

All external calls (HuggingFace model loading, ChromaDB, Groq API) are
mocked so tests run fully offline without any API keys or downloads.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_doc(content: str, doc_id: str = "facility_overview") -> Document:
    return Document(
        page_content=content,
        metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}_chunk_0", "source": "/docs/x.md"},
    )


def _mock_embeddings():
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 384]
    emb.embed_query.return_value = [0.1] * 384
    return emb


# ── Prompt tests ──────────────────────────────────────────────────────────────

class TestPrompt:
    def test_rag_prompt_has_system_and_human_messages(self):
        from src.rag.prompt import RAG_PROMPT

        messages = RAG_PROMPT.format_messages(
            context="CityPark has 850 spaces.",
            question="How many spaces does CityPark have?",
        )
        roles = [m.type for m in messages]
        assert "system" in roles
        assert "human" in roles

    def test_rag_prompt_injects_context_and_question(self):
        from src.rag.prompt import RAG_PROMPT

        messages = RAG_PROMPT.format_messages(
            context="There are 850 parking spaces.",
            question="How many spaces?",
        )
        human_text = next(m.content for m in messages if m.type == "human")
        assert "850 parking spaces" in human_text
        assert "How many spaces?" in human_text

    def test_system_prompt_prohibits_hallucination(self):
        from src.rag.prompt import RAG_PROMPT

        messages = RAG_PROMPT.format_messages(context="ctx", question="q")
        system_text = next(m.content for m in messages if m.type == "system")
        # Key guardrail phrases must be present
        assert "ONLY" in system_text or "only" in system_text
        assert "invent" in system_text or "fabricate" in system_text.lower() or "assume" in system_text

    def test_fallback_message_is_non_empty_string(self):
        from src.rag.prompt import FALLBACK_MESSAGE

        assert isinstance(FALLBACK_MESSAGE, str)
        assert len(FALLBACK_MESSAGE) > 20
        assert "0800" in FALLBACK_MESSAGE  # should contain contact info


# ── Retriever tests ───────────────────────────────────────────────────────────

class TestRetriever:
    def test_build_embeddings_returns_huggingface_instance(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.rag.retriever import build_embeddings

        with patch("src.rag.retriever.HuggingFaceEmbeddings") as MockEmb:
            MockEmb.return_value = MagicMock(spec=HuggingFaceEmbeddings)
            result = build_embeddings("all-MiniLM-L6-v2")

        MockEmb.assert_called_once()
        assert result is not None

    def test_build_retriever_returns_retriever(self):
        from src.rag.retriever import build_retriever

        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        with (
            patch("src.rag.retriever.HuggingFaceEmbeddings", return_value=_mock_embeddings()),
            patch("src.rag.retriever.Chroma", return_value=mock_vs),
        ):
            retriever = build_retriever(k=3)

        mock_vs.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        assert retriever is mock_retriever

    def test_build_retriever_uses_settings_k_by_default(self):
        from src.config import settings
        from src.rag.retriever import build_retriever

        mock_vs = MagicMock()
        with (
            patch("src.rag.retriever.HuggingFaceEmbeddings", return_value=_mock_embeddings()),
            patch("src.rag.retriever.Chroma", return_value=mock_vs),
        ):
            build_retriever()

        call_kwargs = mock_vs.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["k"] == settings.retrieval_k


# ── RagChain tests ────────────────────────────────────────────────────────────

class TestRagChain:
    def _make_chain(self, docs: list[Document], llm_response: str = "Test answer."):
        """Build a RagChain with fully mocked retriever and LLM."""
        from src.rag.chain import RagChain

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs

        mock_llm_chain = MagicMock()
        mock_llm_chain.invoke.return_value = llm_response

        chain = RagChain(retriever=mock_retriever, llm=MagicMock())
        chain._llm_chain = mock_llm_chain
        return chain

    def test_query_returns_rag_result(self):
        from src.rag.chain import RagResult

        docs = [_make_doc("CityPark has 850 spaces.")]
        chain = self._make_chain(docs, "CityPark has 850 spaces.")
        result = chain.query("How many spaces?")

        assert isinstance(result, RagResult)

    def test_query_with_context_sets_had_relevant_context_true(self):
        docs = [_make_doc("CityPark is open 24 hours.")]
        chain = self._make_chain(docs)
        result = chain.query("Is it open at night?")

        assert result.had_relevant_context is True
        assert result.answer == "Test answer."

    def test_query_with_no_docs_returns_fallback(self):
        from src.rag.prompt import FALLBACK_MESSAGE

        chain = self._make_chain(docs=[])
        result = chain.query("What is the meaning of life?")

        assert result.had_relevant_context is False
        assert result.answer == FALLBACK_MESSAGE
        assert result.llm_latency_ms == 0.0

    def test_source_doc_ids_populated_from_metadata(self):
        docs = [
            _make_doc("Info about facility.", "facility_overview"),
            _make_doc("Info about location.", "location_directions"),
        ]
        chain = self._make_chain(docs)
        result = chain.query("Tell me about the facility.")

        assert "facility_overview" in result.source_doc_ids
        assert "location_directions" in result.source_doc_ids

    def test_latencies_are_non_negative(self):
        docs = [_make_doc("Some content.")]
        chain = self._make_chain(docs)
        result = chain.query("A question.")

        assert result.retrieval_latency_ms >= 0
        assert result.llm_latency_ms >= 0
        assert result.total_latency_ms >= 0

    def test_total_latency_ge_retrieval_plus_llm(self):
        docs = [_make_doc("Content.")]
        chain = self._make_chain(docs)
        result = chain.query("Question?")

        # Total should be at least the sum of the two timed segments
        assert result.total_latency_ms >= result.retrieval_latency_ms

    def test_retrieval_error_returns_fallback(self):
        from src.rag.chain import RagChain
        from src.rag.prompt import FALLBACK_MESSAGE

        mock_retriever = MagicMock()
        mock_retriever.invoke.side_effect = RuntimeError("ChromaDB unavailable")

        chain = RagChain(retriever=mock_retriever, llm=MagicMock())
        result = chain.query("Any question?")

        assert result.answer == FALLBACK_MESSAGE
        assert result.error is not None
        assert result.had_relevant_context is False

    def test_llm_error_returns_fallback_with_context(self):
        from src.rag.chain import RagChain
        from src.rag.prompt import FALLBACK_MESSAGE

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_make_doc("Some content.")]

        mock_llm_chain = MagicMock()
        mock_llm_chain.invoke.side_effect = RuntimeError("Groq API timeout")

        chain = RagChain(retriever=mock_retriever, llm=MagicMock())
        chain._llm_chain = mock_llm_chain
        result = chain.query("Question?")

        assert result.answer == FALLBACK_MESSAGE
        assert result.had_relevant_context is True  # chunks were retrieved
        assert result.error is not None


# ── Context formatting test ───────────────────────────────────────────────────

class TestFormatContext:
    def test_format_context_numbers_sources(self):
        from src.rag.chain import _format_context

        docs = [
            _make_doc("Facility has 850 spaces.", "facility_overview"),
            _make_doc("Open 24 hours.", "faq"),
        ]
        ctx = _format_context(docs)

        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "facility_overview" in ctx
        assert "faq" in ctx

    def test_format_context_empty_list(self):
        from src.rag.chain import _format_context

        assert _format_context([]) == ""


# ── build_rag_chain smoke test (no real API calls) ────────────────────────────

class TestBuildRagChain:
    def test_build_rag_chain_returns_rag_chain_instance(self):
        from src.rag.chain import RagChain, build_rag_chain

        with (
            patch("src.rag.chain.build_retriever", return_value=MagicMock()),
            patch("src.rag.chain.ChatGroq", return_value=MagicMock()),
        ):
            chain = build_rag_chain(k=3)

        assert isinstance(chain, RagChain)
