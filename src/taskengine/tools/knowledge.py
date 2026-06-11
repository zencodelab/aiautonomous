"""
Knowledge retrieval tool — queries the Pinecone vector store.

This tool is created dynamically at engine startup so it can capture
a reference to the initialised ``KnowledgeStore`` instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool as _tool_decorator

if TYPE_CHECKING:
    from taskengine.vectorstore import KnowledgeStore


def create_knowledge_tool(store: KnowledgeStore):
    """Factory that returns a LangChain ``@tool`` bound to a live KnowledgeStore.

    This pattern avoids module-level singleton state — the tool is only
    created once the engine has a configured store instance.
    """

    @_tool_decorator
    def search_knowledge(query: str, num_results: int = 5) -> str:
        """Search the knowledge base for relevant information.

        Use this tool to find domain knowledge, past task execution
        examples, or any contextual information that might help
        complete the current step.

        Args:
            query: Natural-language search query.
            num_results: Number of results to return (default: 5).

        Returns:
            Formatted list of relevant knowledge excerpts.
        """
        try:
            docs = store.search(query=query, k=num_results)
            if not docs:
                return "No relevant knowledge found."

            parts: list[str] = []
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                source_info = ""
                if meta.get("type") == "execution_history":
                    source_info = f" (past execution, success={meta.get('success')})"
                parts.append(
                    f"{i}. {doc.page_content[:500]}{source_info}"
                )
            return "\n\n".join(parts)

        except Exception as exc:
            return f"Knowledge search error: {exc}"

    return search_knowledge
