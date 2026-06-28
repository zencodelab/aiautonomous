"""
Pinecone vector store integration for knowledge storage and retrieval.

Wraps ``langchain-pinecone`` to provide:
- Auto-creation of serverless Pinecone indexes
- Knowledge document ingestion
- Semantic similarity search
- Task execution history storage for future reference
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

if TYPE_CHECKING:
    from taskengine.config import Settings
    from taskengine.models import ExecutionReport

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Vector-backed knowledge store powered by Pinecone.

    Handles index lifecycle, document upsert, and similarity search.
    Past execution reports are stored so the planner can retrieve
    few-shot examples of similar tasks.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. "
                "KnowledgeStore requires a Pinecone API key."
            )
        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._ensure_index()
        self._vector_store = self._build_vector_store()

    # ── Index management ────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Create the Pinecone index if it does not already exist."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing:
            logger.info("Creating Pinecone index '%s' …", self._index_name)
            self._pc.create_index(
                name=self._index_name,
                dimension=self._settings.pinecone_dimension,
                metric=self._settings.pinecone_metric,
                spec=ServerlessSpec(
                    cloud=self._settings.pinecone_cloud,
                    region=self._settings.pinecone_region,
                ),
            )
            logger.info("Index '%s' created.", self._index_name)
        else:
            logger.debug("Index '%s' already exists.", self._index_name)

    def _build_vector_store(self):
        """Build a LangChain PineconeVectorStore wrapper."""
        from langchain_pinecone import PineconeVectorStore

        return PineconeVectorStore(
            index_name=self._index_name,
            embedding=self._embeddings,
            pinecone_api_key=self._settings.pinecone_api_key,
        )

    # ── Public API ──────────────────────────────────────────────────────

    def add_knowledge(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Upsert documents into the knowledge store.

        Args:
            texts: Document texts to embed and store.
            metadatas: Optional per-document metadata dicts.

        Returns:
            List of document IDs assigned by Pinecone.
        """
        logger.info("Adding %d documents to knowledge store …", len(texts))
        ids = self._vector_store.add_texts(texts=texts, metadatas=metadatas)
        logger.info("Added %d documents.", len(ids))
        return ids

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Run a similarity search against the knowledge store.

        Args:
            query: Natural-language search query.
            k: Number of results to return.

        Returns:
            List of matching LangChain ``Document`` objects.
        """
        logger.debug("Searching knowledge store: %r (k=%d)", query, k)
        results = self._vector_store.similarity_search(query=query, k=k)
        logger.debug("Found %d results.", len(results))
        return results

    def add_task_execution(self, report: ExecutionReport) -> list[str]:
        """Store a completed execution report for future retrieval.

        The report is serialised as a text summary with structured metadata
        so the planner can find similar past executions.
        """
        summary_text = (
            f"Task: {report.query}\n"
            f"Objective: {report.plan.objective}\n"
            f"Steps taken: {len(report.step_results)}\n"
            f"Success: {report.success}\n"
            f"Summary: {report.summary}"
        )

        step_descriptions = " → ".join(
            s.step_description for s in report.step_results
        )

        metadata = {
            "type": "execution_history",
            "query": report.query,
            "success": report.success,
            "step_count": len(report.step_results),
            "steps": step_descriptions[:500],  # Pinecone metadata size limit
            "duration": report.total_duration_seconds,
        }

        return self.add_knowledge(texts=[summary_text], metadatas=[metadata])
