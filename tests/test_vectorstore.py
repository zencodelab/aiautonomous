"""
Tests for the KnowledgeStore — Pinecone vector store integration.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from taskengine.models import (
    ExecutionReport,
    ReflectionResult,
    StepResult,
    TaskPlan,
    TaskStep,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.pinecone_api_key = "test-pinecone-key"
    settings.pinecone_index_name = "test-index"
    settings.pinecone_cloud = "aws"
    settings.pinecone_region = "us-east-1"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    return settings


class TestKnowledgeStore:
    """Tests for the KnowledgeStore class."""

    def test_index_creation(self, mock_settings):
        """Verify the store creates a Pinecone index if it doesn't exist."""
        with (
            patch("taskengine.vectorstore.Pinecone") as MockPC,
            patch("taskengine.vectorstore.OpenAIEmbeddings"),
            patch("taskengine.vectorstore.KnowledgeStore._build_vector_store"),
        ):
            # Simulate no existing indexes
            MockPC.return_value.list_indexes.return_value = []

            from taskengine.vectorstore import KnowledgeStore

            store = KnowledgeStore(mock_settings)

            MockPC.return_value.create_index.assert_called_once()

    def test_skips_existing_index(self, mock_settings):
        """Verify the store doesn't recreate an existing index."""
        with (
            patch("taskengine.vectorstore.Pinecone") as MockPC,
            patch("taskengine.vectorstore.OpenAIEmbeddings"),
            patch("taskengine.vectorstore.KnowledgeStore._build_vector_store"),
        ):
            mock_idx = MagicMock()
            mock_idx.name = "test-index"
            MockPC.return_value.list_indexes.return_value = [mock_idx]

            from taskengine.vectorstore import KnowledgeStore

            store = KnowledgeStore(mock_settings)

            MockPC.return_value.create_index.assert_not_called()

    def test_add_knowledge(self, mock_settings):
        """Verify documents are added to the vector store."""
        with (
            patch("taskengine.vectorstore.Pinecone") as MockPC,
            patch("taskengine.vectorstore.OpenAIEmbeddings"),
            patch("taskengine.vectorstore.KnowledgeStore._build_vector_store") as mock_build,
        ):
            MockPC.return_value.list_indexes.return_value = []

            from taskengine.vectorstore import KnowledgeStore

            store = KnowledgeStore(mock_settings)
            store._vector_store = MagicMock()
            store._vector_store.add_texts.return_value = ["id1", "id2"]

            ids = store.add_knowledge(
                texts=["Hello", "World"],
                metadatas=[{"source": "test"}, {"source": "test"}],
            )

            assert ids == ["id1", "id2"]
            store._vector_store.add_texts.assert_called_once()

    def test_search(self, mock_settings):
        """Verify similarity search returns documents."""
        with (
            patch("taskengine.vectorstore.Pinecone") as MockPC,
            patch("taskengine.vectorstore.OpenAIEmbeddings"),
            patch("taskengine.vectorstore.KnowledgeStore._build_vector_store"),
        ):
            MockPC.return_value.list_indexes.return_value = []

            from taskengine.vectorstore import KnowledgeStore

            store = KnowledgeStore(mock_settings)
            store._vector_store = MagicMock()
            store._vector_store.similarity_search.return_value = [
                Document(page_content="Result 1", metadata={"source": "test"}),
                Document(page_content="Result 2", metadata={"source": "test"}),
            ]

            results = store.search("test query", k=2)

            assert len(results) == 2
            assert results[0].page_content == "Result 1"

    def test_add_task_execution(self, mock_settings):
        """Verify execution reports are stored correctly."""
        with (
            patch("taskengine.vectorstore.Pinecone") as MockPC,
            patch("taskengine.vectorstore.OpenAIEmbeddings"),
            patch("taskengine.vectorstore.KnowledgeStore._build_vector_store"),
        ):
            MockPC.return_value.list_indexes.return_value = []

            from taskengine.vectorstore import KnowledgeStore

            store = KnowledgeStore(mock_settings)
            store._vector_store = MagicMock()
            store._vector_store.add_texts.return_value = ["exec-id"]

            report = ExecutionReport(
                query="Test query",
                plan=TaskPlan(
                    query="Test query",
                    objective="Test obj",
                    steps=[TaskStep(order=1, description="Step 1")],
                ),
                step_results=[
                    StepResult(
                        step_id="s1",
                        step_order=1,
                        step_description="Step 1",
                        output="Done",
                        success=True,
                        duration_seconds=1.0,
                    ),
                ],
                summary="All done.",
                success=True,
            )

            ids = store.add_task_execution(report)

            assert ids == ["exec-id"]
            call_args = store._vector_store.add_texts.call_args
            assert "execution_history" in str(call_args)
