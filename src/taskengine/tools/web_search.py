"""
Web search tool — uses DuckDuckGo for internet searches.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the internet for information using DuckDuckGo.

    Use this tool when you need to find current information, research a topic,
    or look up facts that are not in the knowledge base.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results found for: {query}"

        formatted: list[str] = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. **{r.get('title', 'N/A')}**\n"
                f"   {r.get('body', 'No snippet available.')}\n"
                f"   URL: {r.get('href', 'N/A')}"
            )
        return "\n\n".join(formatted)

    except ImportError:
        return (
            "Error: duckduckgo-search package is not installed. "
            "Run: pip install duckduckgo-search"
        )
    except Exception as exc:
        return f"Search error: {exc}"
