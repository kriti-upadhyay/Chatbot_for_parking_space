"""
RAG chain — retrieval-augmented generation end-to-end.

Public API
----------
RagResult     — typed dataclass returned by every query
RagChain      — class encapsulating the retriever + LLM; call .query()
build_rag_chain() — factory that creates a configured RagChain
"""

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from src.config import get_logger, settings
from src.rag.prompt import FALLBACK_MESSAGE, RAG_PROMPT
from src.rag.retriever import build_retriever

logger = get_logger(__name__)

_MIN_CHUNKS_REQUIRED = 1  # need at least this many chunks to attempt an LLM call


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RagResult:
    """Structured output from a single RAG query."""

    answer: str
    source_doc_ids: list[str]
    retrieved_chunks: list[Document]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    had_relevant_context: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks as a numbered, source-labelled context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("doc_id", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_doc_ids(docs: list[Document]) -> list[str]:
    return [doc.metadata.get("doc_id", "unknown") for doc in docs]


# ---------------------------------------------------------------------------
# RagChain
# ---------------------------------------------------------------------------

class RagChain:
    """
    Encapsulates the retriever and LLM so they are initialised once and
    reused across multiple queries within the same session.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: ChatGroq,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        # Build a minimal LCEL chain for the LLM step only
        # (retrieval is handled explicitly so we can measure latency)
        self._llm_chain = RAG_PROMPT | self._llm | StrOutputParser()

    def query(self, question: str) -> RagResult:
        """
        Run the full RAG pipeline for *question*.

        Steps:
          1. Retrieve relevant chunks from ChromaDB.
          2. If no chunks found, return FALLBACK_MESSAGE without calling LLM.
          3. Format context and call the LLM.
          4. Return a RagResult with the answer and timing data.
        """
        t_start = time.perf_counter()

        # ── Step 1: Retrieve ─────────────────────────────────────────────────
        t_retrieve_start = time.perf_counter()
        try:
            docs = self._retriever.invoke(question)
        except Exception as exc:
            logger.exception("Retrieval failed for question: %s", question)
            elapsed = (time.perf_counter() - t_start) * 1000
            return RagResult(
                answer=FALLBACK_MESSAGE,
                source_doc_ids=[],
                retrieved_chunks=[],
                retrieval_latency_ms=elapsed,
                llm_latency_ms=0.0,
                total_latency_ms=elapsed,
                had_relevant_context=False,
                error=str(exc),
            )
        retrieval_ms = (time.perf_counter() - t_retrieve_start) * 1000

        # ── Step 2: Fallback if nothing retrieved ────────────────────────────
        if len(docs) < _MIN_CHUNKS_REQUIRED:
            logger.info("No relevant chunks found for: %s", question)
            total_ms = (time.perf_counter() - t_start) * 1000
            return RagResult(
                answer=FALLBACK_MESSAGE,
                source_doc_ids=[],
                retrieved_chunks=[],
                retrieval_latency_ms=retrieval_ms,
                llm_latency_ms=0.0,
                total_latency_ms=total_ms,
                had_relevant_context=False,
            )

        # ── Step 3: Generate ─────────────────────────────────────────────────
        context = _format_context(docs)
        t_llm_start = time.perf_counter()
        try:
            answer = self._llm_chain.invoke(
                {"question": question, "context": context}
            )
        except Exception as exc:
            logger.exception("LLM call failed for question: %s", question)
            total_ms = (time.perf_counter() - t_start) * 1000
            return RagResult(
                answer=FALLBACK_MESSAGE,
                source_doc_ids=_extract_doc_ids(docs),
                retrieved_chunks=docs,
                retrieval_latency_ms=retrieval_ms,
                llm_latency_ms=0.0,
                total_latency_ms=total_ms,
                had_relevant_context=True,
                error=str(exc),
            )
        llm_ms = (time.perf_counter() - t_llm_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        logger.debug(
            "RAG query complete | retrieval=%.0fms llm=%.0fms total=%.0fms chunks=%d",
            retrieval_ms, llm_ms, total_ms, len(docs),
        )

        return RagResult(
            answer=answer,
            source_doc_ids=_extract_doc_ids(docs),
            retrieved_chunks=docs,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_ms,
            total_latency_ms=total_ms,
            had_relevant_context=True,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rag_chain(
    k: int | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> RagChain:
    """
    Build and return a fully configured RagChain.

    Args:
        k:           Number of chunks to retrieve per query.
        model_name:  Groq model identifier (defaults to settings.groq_model_name).
        temperature: LLM sampling temperature (0 = deterministic).
    """
    retriever = build_retriever(k=k)

    llm = ChatGroq(
        model=model_name or settings.groq_model_name,
        api_key=settings.groq_api_key,  # type: ignore[arg-type]
        temperature=temperature,
    )

    logger.info(
        "RAG chain built (model=%s, k=%d)",
        model_name or settings.groq_model_name,
        k or settings.retrieval_k,
    )
    return RagChain(retriever=retriever, llm=llm)
