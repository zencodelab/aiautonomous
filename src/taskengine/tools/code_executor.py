"""
Python code execution tool — runs snippets in a sandboxed subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool


@tool
def execute_python(code: str, timeout: int = 30) -> str:
    """Execute a Python code snippet and return the output.

    Use this tool when you need to run calculations, data processing,
    or any programmatic operation. The code runs in a separate process
    with a timeout for safety.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds (default: 30).

    Returns:
        stdout output, stderr output, and the return code.
    """
    if not code.strip():
        return "Error: No code provided."

    # Write code to a temporary file so the subprocess can execute it
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=min(timeout, 60),  # Hard cap at 60s
            cwd=tempfile.gettempdir(),
        )

        parts: list[str] = []
        if result.stdout.strip():
            parts.append(f"STDOUT:\n{result.stdout.strip()}")
        if result.stderr.strip():
            parts.append(f"STDERR:\n{result.stderr.strip()}")
        parts.append(f"Return code: {result.returncode}")

        output = "\n\n".join(parts)

        # Truncate if output is huge
        if len(output) > 5_000:
            output = output[:5_000] + "\n\n… [output truncated]"

        return output

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds."
    except Exception as exc:
        return f"Error executing code: {exc}"
    finally:
        Path(script_path).unlink(missing_ok=True)
