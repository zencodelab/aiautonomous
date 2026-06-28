"""
HTTP fetch tool — retrieves the text content of a URL.

Useful when the agent already knows a specific URL (documentation page,
raw API endpoint, GitHub raw file, etc.) and needs to read its content
without a search round-trip.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from langchain_core.tools import tool

_MAX_BYTES = 32_768  # truncate responses larger than 32 KB


@tool
def http_fetch(url: str, max_chars: int = 4000) -> str:
    """Fetch the text content of a URL.

    Use this when you already know the exact URL you want to read —
    documentation pages, raw GitHub files, public JSON APIs, etc.
    For discovering URLs, use ``web_search`` first.

    Args:
        url: The URL to fetch (must start with http:// or https://).
        max_chars: Maximum number of characters to return (default: 4000).

    Returns:
        The decoded text content of the response, truncated to *max_chars*,
        or an error message if the request fails.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://, got: {url!r}"

    max_chars = max(1, min(max_chars, _MAX_BYTES))

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TaskEngine/1.0 (+https://github.com/zencodelab/aiautonomous)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw = resp.read(_MAX_BYTES)

        # Detect encoding from Content-Type, fall back to utf-8
        content_type: str = resp.headers.get_content_charset() or "utf-8"
        text = raw.decode(content_type, errors="replace")

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[… truncated at {max_chars} chars]"
        return text

    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} error fetching {url}: {exc.reason}"
    except urllib.error.URLError as exc:
        return f"URL error fetching {url}: {exc.reason}"
    except TimeoutError:
        return f"Timeout fetching {url} (limit: 15 s)"
    except Exception as exc:  # noqa: BLE001
        return f"Error fetching {url}: {exc}"
