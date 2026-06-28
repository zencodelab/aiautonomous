"""Unit tests for the http_fetch tool."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from taskengine.tools.http_fetch import http_fetch


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mock_response(body: bytes, charset: str = "utf-8") -> MagicMock:
    """Build a mock urllib response context manager."""
    resp = MagicMock()
    resp.read.return_value = body
    resp.headers.get_content_charset.return_value = charset
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── Tests ────────────────────────────────────────────────────────────────────

class TestHttpFetch:
    """Tests for http_fetch tool."""

    def test_rejects_non_http_scheme(self) -> None:
        result = http_fetch.invoke({"url": "ftp://example.com/file.txt"})
        assert "Error" in result
        assert "http://" in result

    def test_rejects_bare_path(self) -> None:
        result = http_fetch.invoke({"url": "/etc/passwd"})
        assert "Error" in result

    def test_successful_fetch(self) -> None:
        body = b"Hello, world!"
        mock_resp = _mock_response(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = http_fetch.invoke({"url": "https://example.com/"})
        assert result == "Hello, world!"

    def test_truncates_long_content(self) -> None:
        body = b"A" * 10_000
        mock_resp = _mock_response(body)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = http_fetch.invoke({"url": "https://example.com/", "max_chars": 100})
        assert len(result) <= 200  # truncation marker adds a few chars
        assert "truncated" in result

    def test_http_error_returns_message(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://example.com/",
                code=404,
                msg="Not Found",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            ),
        ):
            result = http_fetch.invoke({"url": "https://example.com/missing"})
        assert "404" in result
        assert "Not Found" in result

    def test_url_error_returns_message(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(reason="Name or service not known"),
        ):
            result = http_fetch.invoke({"url": "https://nonexistent.invalid/"})
        assert "URL error" in result

    def test_timeout_returns_message(self) -> None:
        with patch("urllib.request.urlopen", side_effect=TimeoutError):
            result = http_fetch.invoke({"url": "https://slow.example.com/"})
        assert "Timeout" in result or "timeout" in result

    def test_non_utf8_charset(self) -> None:
        body = "Héllo".encode("latin-1")
        mock_resp = _mock_response(body, charset="latin-1")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = http_fetch.invoke({"url": "https://example.com/"})
        assert "H" in result  # basic sanity — at minimum the ASCII part survives

    def test_tool_is_importable_from_package(self) -> None:
        from taskengine.tools import http_fetch as imported  # noqa: PLC0415
        assert imported is not None
