"""
File operation tools — sandboxed read, write, and list operations.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

# Lazy-loaded workspace root (set by engine at startup).
_WORKSPACE: Path | None = None


def set_workspace(path: Path) -> None:
    """Configure the sandboxed workspace directory for file operations."""
    global _WORKSPACE  # noqa: PLW0603
    _WORKSPACE = path.resolve()
    _WORKSPACE.mkdir(parents=True, exist_ok=True)


def _resolve(relative_path: str) -> Path:
    """Resolve a user-provided path relative to the workspace, ensuring sandboxing."""
    if _WORKSPACE is None:
        raise RuntimeError("Workspace not configured. Call set_workspace() first.")
    resolved = (_WORKSPACE / relative_path).resolve()
    if not resolved.is_relative_to(_WORKSPACE):
        raise PermissionError(
            f"Path escapes workspace sandbox: {relative_path}"
        )
    return resolved


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file from the workspace directory.

    Use this tool when you need to read data from a file. The path is
    relative to the workspace directory.

    Args:
        file_path: Relative path to the file within the workspace.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        target = _resolve(file_path)
        if not target.exists():
            return f"File not found: {file_path}"
        if not target.is_file():
            return f"Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        # Truncate very large files to avoid context overflow
        if len(content) > 10_000:
            return content[:10_000] + f"\n\n… [truncated — file is {len(content)} chars]"
        return content
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the workspace directory.

    Use this tool when you need to save results, create documents, or
    store data. The path is relative to the workspace directory.
    Parent directories are created automatically.

    Args:
        file_path: Relative path to the file within the workspace.
        content: The text content to write.

    Returns:
        Confirmation message or an error description.
    """
    try:
        target = _resolve(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """List files and directories in the workspace.

    Use this tool to explore the workspace file structure.

    Args:
        directory_path: Relative path to list (default: workspace root).

    Returns:
        Formatted directory listing or an error message.
    """
    try:
        target = _resolve(directory_path)
        if not target.exists():
            return f"Directory not found: {directory_path}"
        if not target.is_dir():
            return f"Not a directory: {directory_path}"

        entries: list[str] = []
        for item in sorted(target.iterdir()):
            kind = "📁" if item.is_dir() else "📄"
            size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            entries.append(f"  {kind} {item.name}{size}")

        if not entries:
            return f"Directory is empty: {directory_path}"

        header = f"Contents of {directory_path}/:"
        return header + "\n" + "\n".join(entries)
    except PermissionError as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error listing directory: {exc}"
