"""
Tests for the file_ops sandbox — read, write, list, and path-traversal hardening.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from taskengine.tools.file_ops import (
    set_workspace,
    read_file,
    write_file,
    list_directory,
    _resolve,
)


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path):
    """Set up a fresh workspace for each test."""
    set_workspace(tmp_path)
    return tmp_path


class TestSandbox:
    """Sandbox path-traversal hardening tests."""

    def test_resolve_normal_path(self, workspace: Path):
        result = _resolve("foo.txt")
        assert result == workspace / "foo.txt"

    def test_resolve_nested_path(self, workspace: Path):
        result = _resolve("a/b/c.txt")
        assert result == workspace / "a" / "b" / "c.txt"

    def test_resolve_dot_path(self, workspace: Path):
        result = _resolve(".")
        assert result == workspace

    def test_traversal_dotdot_raises(self, workspace: Path):
        with pytest.raises(PermissionError, match="sandbox"):
            _resolve("../escape.txt")

    def test_traversal_absolute_path_raises(self, workspace: Path):
        with pytest.raises(PermissionError, match="sandbox"):
            _resolve("/etc/passwd")

    def test_traversal_sibling_prefix_raises(self, workspace: Path):
        """Regression: ensure a sibling directory whose name starts with the
        workspace name does not pass the sandbox check.

        e.g. workspace = /tmp/pytest-abc/workspace0
             attacker path that resolved to /tmp/pytest-abc/workspace0_evil/x
        """
        # Create a sibling directory whose name *starts* with the workspace name
        sibling = workspace.parent / (workspace.name + "_evil")
        sibling.mkdir(exist_ok=True)
        with pytest.raises(PermissionError, match="sandbox"):
            _resolve(f"../{sibling.name}/x.txt")


class TestReadFile:
    def test_read_existing_file(self, workspace: Path):
        (workspace / "hello.txt").write_text("hi there")
        assert read_file.invoke({"file_path": "hello.txt"}) == "hi there"

    def test_read_missing_file(self):
        result = read_file.invoke({"file_path": "nope.txt"})
        assert "not found" in result.lower()

    def test_read_directory_returns_error(self, workspace: Path):
        (workspace / "adir").mkdir()
        result = read_file.invoke({"file_path": "adir"})
        assert "not a file" in result.lower()

    def test_read_truncates_large_file(self, workspace: Path):
        (workspace / "big.txt").write_text("x" * 20_000)
        result = read_file.invoke({"file_path": "big.txt"})
        assert "truncated" in result
        assert len(result) < 15_000

    def test_read_path_traversal_blocked(self):
        result = read_file.invoke({"file_path": "../secret.txt"})
        assert "permission denied" in result.lower()


class TestWriteFile:
    def test_write_new_file(self, workspace: Path):
        result = write_file.invoke({"file_path": "out.txt", "content": "data"})
        assert "successfully" in result.lower()
        assert (workspace / "out.txt").read_text() == "data"

    def test_write_creates_parents(self, workspace: Path):
        write_file.invoke({"file_path": "sub/dir/out.txt", "content": "nested"})
        assert (workspace / "sub" / "dir" / "out.txt").read_text() == "nested"

    def test_write_path_traversal_blocked(self):
        result = write_file.invoke({"file_path": "../evil.txt", "content": "bad"})
        assert "permission denied" in result.lower()


class TestListDirectory:
    def test_list_root(self, workspace: Path):
        (workspace / "a.txt").write_text("a")
        (workspace / "b.txt").write_text("b")
        result = list_directory.invoke({"directory_path": "."})
        assert "a.txt" in result
        assert "b.txt" in result

    def test_list_missing_dir(self):
        result = list_directory.invoke({"directory_path": "ghost"})
        assert "not found" in result.lower()

    def test_list_file_returns_error(self, workspace: Path):
        (workspace / "f.txt").write_text("x")
        result = list_directory.invoke({"directory_path": "f.txt"})
        assert "not a directory" in result.lower()

    def test_list_empty_dir(self, workspace: Path):
        (workspace / "empty").mkdir()
        result = list_directory.invoke({"directory_path": "empty"})
        assert "empty" in result.lower()

    def test_list_path_traversal_blocked(self):
        result = list_directory.invoke({"directory_path": "../"})
        assert "permission denied" in result.lower()
