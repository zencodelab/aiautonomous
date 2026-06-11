#!/usr/bin/env python3
"""
CLI entry point for the Autonomous Task Execution Engine.

Built with Typer + Rich for beautiful terminal output.

Usage::

    # Execute a task
    python cli.py run "Research Python 3.13 features and create a summary"

    # Generate a plan without executing
    python cli.py plan "Calculate compound interest and save to a file"

    # Add knowledge documents
    python cli.py knowledge add path/to/document.txt

    # Start the API server
    python cli.py serve
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ── CLI app & console ──────────────────────────────────────────────────────

app = typer.Typer(
    name="taskengine",
    help="🤖 Autonomous Task Execution Engine — powered by LangChain + Pinecone",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# Sub-commands group for knowledge management
knowledge_app = typer.Typer(help="📚 Manage the knowledge base")
app.add_typer(knowledge_app, name="knowledge")


# ── Logging setup ──────────────────────────────────────────────────────────


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


# ── Progress callback ─────────────────────────────────────────────────────


def _make_progress_callback(progress_display: Progress):
    """Create a progress callback that updates the Rich progress display."""
    task_id = progress_display.add_task("Initialising…", total=None)

    def callback(event: str, data: dict) -> None:
        if event == "planning":
            progress_display.update(
                task_id, description="🧠 Planning task decomposition…"
            )
        elif event == "plan_created":
            progress_display.update(
                task_id,
                description=(
                    f"📋 Plan created: {data['step_count']} steps — "
                    f"{data['objective'][:60]}"
                ),
            )
        elif event == "executing":
            progress_display.update(
                task_id,
                description=f"⚡ Executing {data['step_count']} steps…",
            )
        elif event == "step_start":
            progress_display.update(
                task_id,
                description=(
                    f"  🔧 Step {data['order']}: {data['description'][:50]}…"
                ),
            )
        elif event == "step_done":
            icon = "✅" if data["success"] else "❌"
            progress_display.update(
                task_id,
                description=(
                    f"  {icon} Step {data['order']} done "
                    f"({data['duration']:.1f}s)"
                ),
            )
        elif event == "reflecting":
            progress_display.update(
                task_id, description="🔍 Reflecting on results…"
            )
        elif event == "reflection_done":
            progress_display.update(
                task_id,
                description=(
                    f"📊 Quality score: {data['score']:.2f}"
                ),
            )
        elif event == "replanning":
            progress_display.update(
                task_id,
                description=(
                    f"🔄 Re-planning (attempt {data['attempt']}/{data['max']})…"
                ),
            )
        elif event == "completed":
            icon = "🎉" if data["success"] else "⚠️"
            progress_display.update(
                task_id,
                description=(
                    f"{icon} Completed in {data['duration']:.1f}s "
                    f"(score: {data['score']:.2f})"
                ),
            )

    return callback


# ── Commands ───────────────────────────────────────────────────────────────


@app.command()
def run(
    query: str = typer.Argument(help="Plain-text task description to execute."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
):
    """🚀 Execute an autonomous task from a plain-text query."""
    _setup_logging(verbose)

    # Load env
    from dotenv import load_dotenv
    load_dotenv()

    from taskengine.config import get_settings
    from taskengine.engine import TaskEngine

    console.print(
        Panel(
            f"[bold cyan]{query}[/bold cyan]",
            title="🤖 Task Engine",
            subtitle="Autonomous Execution",
            border_style="bright_blue",
        )
    )

    settings = get_settings()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        callback = _make_progress_callback(progress)
        engine = TaskEngine(settings, on_progress=callback)
        report = asyncio.run(engine.run(query))

    # Display results
    console.print()
    _display_report(report)


@app.command()
def plan(
    query: str = typer.Argument(help="Plain-text task to plan (dry run)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """📋 Generate a task plan without executing it."""
    _setup_logging(verbose)

    from dotenv import load_dotenv
    load_dotenv()

    from taskengine.config import get_settings
    from taskengine.engine import TaskEngine

    settings = get_settings()
    engine = TaskEngine(settings)
    task_plan = asyncio.run(engine.plan_only(query))

    console.print(
        Panel(
            f"[bold]{task_plan.objective}[/bold]",
            title="📋 Task Plan",
            border_style="green",
        )
    )

    table = Table(title="Execution Steps", show_lines=True)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Description", style="white")
    table.add_column("Tool Hint", style="yellow", width=20)

    for step in task_plan.steps:
        table.add_row(
            str(step.order),
            step.description,
            step.tool_hint or "—",
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Server host."),
    port: int = typer.Option(8000, help="Server port."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """🌐 Start the FastAPI REST server."""
    _setup_logging(verbose)

    from dotenv import load_dotenv
    load_dotenv()

    import uvicorn

    console.print(
        Panel(
            f"[bold green]Starting API server on {host}:{port}[/bold green]\n"
            f"Docs: http://{host}:{port}/docs",
            title="🌐 TaskEngine API",
            border_style="green",
        )
    )

    uvicorn.run(
        "taskengine.api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


@knowledge_app.command("add")
def knowledge_add(
    file_path: str = typer.Argument(help="Path to a text file to add."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """📄 Add a text file to the knowledge base."""
    _setup_logging(verbose)

    from dotenv import load_dotenv
    load_dotenv()

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(code=1)

    content = path.read_text(encoding="utf-8")

    from taskengine.config import get_settings
    from taskengine.vectorstore import KnowledgeStore

    settings = get_settings()
    store = KnowledgeStore(settings)

    # Split into chunks if the file is large
    chunk_size = 1000
    chunks = [
        content[i : i + chunk_size]
        for i in range(0, len(content), chunk_size)
    ]

    metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]
    ids = store.add_knowledge(texts=chunks, metadatas=metadatas)

    console.print(
        f"[green]✅ Added {len(ids)} chunk(s) from {path.name} "
        f"to the knowledge base.[/green]"
    )


# ── Display helpers ────────────────────────────────────────────────────────


def _display_report(report) -> None:
    """Render an execution report in the terminal."""
    # Header
    status_icon = "✅" if report.success else "❌"
    status_color = "green" if report.success else "red"
    console.print(
        Panel(
            f"[bold {status_color}]{status_icon} "
            f"{'Task Completed Successfully' if report.success else 'Task Completed with Issues'}[/bold {status_color}]",
            title="Execution Report",
            border_style=status_color,
        )
    )

    # Summary
    if report.summary:
        console.print(Panel(Markdown(report.summary), title="📝 Summary", border_style="blue"))

    # Step results table
    table = Table(title="Step Results", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Description", style="white")
    table.add_column("Status", width=8)
    table.add_column("Duration", style="cyan", width=10)
    table.add_column("Tools", style="yellow", width=20)

    for r in report.step_results:
        icon = "✅" if r.success else "❌"
        tools = ", ".join(tc["name"] for tc in r.tool_calls) or "—"
        table.add_row(
            str(r.step_order),
            r.step_description[:60],
            icon,
            f"{r.duration_seconds:.1f}s",
            tools,
        )

    console.print(table)

    # Reflection
    if report.reflection:
        score = report.reflection.quality_score
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        score_color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
        console.print(
            f"\n[bold]Quality Score:[/bold] [{score_color}]{bar} {score:.0%}[/{score_color}]"
        )
        if report.replan_count > 0:
            console.print(f"[dim]Re-planning cycles: {report.replan_count}[/dim]")

    # Timing
    console.print(
        f"\n[dim]Total duration: {report.total_duration_seconds:.1f}s[/dim]"
    )


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
