"""
FastAPI REST interface for the task engine.

Provides endpoints for submitting tasks, checking status, managing
knowledge, and health checks.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from taskengine.config import get_settings
from taskengine.engine import TaskEngine
from taskengine.models import ExecutionReport

logger = logging.getLogger(__name__)

# ── In-memory task storage ──────────────────────────────────────────────────

_tasks: dict[str, dict[str, Any]] = {}
_engine: TaskEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the engine on startup."""
    global _engine  # noqa: PLW0603
    settings = get_settings()
    _engine = TaskEngine(settings)
    logger.info("Task engine initialised.")
    yield
    logger.info("Shutting down.")


# ── App creation ────────────────────────────────────────────────────────────

app = FastAPI(
    title="TaskEngine API",
    description=(
        "Autonomous task-execution engine that transforms plain-text "
        "queries into structured, multi-step operations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────


class TaskRequest(BaseModel):
    """Request to submit a new task."""

    query: str = Field(description="Plain-text task description.")


class TaskResponse(BaseModel):
    """Response after submitting a task."""

    task_id: str = Field(description="Unique task identifier.")
    status: str = Field(description="Current status of the task.")
    message: str = Field(description="Human-readable status message.")


class KnowledgeRequest(BaseModel):
    """Request to add knowledge documents."""

    texts: list[str] = Field(description="Document texts to add.")
    metadatas: list[dict] | None = Field(
        default=None, description="Optional metadata for each document."
    )


class KnowledgeResponse(BaseModel):
    """Response after adding knowledge."""

    document_ids: list[str] = Field(description="IDs of added documents.")
    count: int = Field(description="Number of documents added.")


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "engine": _engine is not None}


@app.post("/tasks", response_model=TaskResponse)
async def submit_task(request: TaskRequest):
    """Submit a plain-text query for autonomous execution.

    The task runs asynchronously in the background. Use the returned
    task_id to poll for status and results.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "report": None, "error": None}

    # Launch execution in the background
    asyncio.create_task(_run_task(task_id, request.query))

    return TaskResponse(
        task_id=task_id,
        status="running",
        message="Task submitted. Poll GET /tasks/{task_id} for status.",
    )


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get the status and results of a submitted task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found.")

    task = _tasks[task_id]

    if task["status"] == "running":
        return {"task_id": task_id, "status": "running"}

    if task["status"] == "failed":
        return {
            "task_id": task_id,
            "status": "failed",
            "error": task["error"],
        }

    report: ExecutionReport = task["report"]
    return {
        "task_id": task_id,
        "status": "completed",
        "success": report.success,
        "summary": report.summary,
        "total_duration_seconds": report.total_duration_seconds,
        "step_count": len(report.step_results),
        "quality_score": report.reflection.quality_score if report.reflection else None,
        "replan_count": report.replan_count,
        "plan": {
            "objective": report.plan.objective,
            "steps": [
                {
                    "order": s.order,
                    "description": s.description,
                    "status": s.status.value,
                }
                for s in report.plan.steps
            ],
        },
        "step_results": [
            {
                "order": r.step_order,
                "description": r.step_description,
                "success": r.success,
                "output": r.output[:500] if r.output else None,
                "error": r.error,
                "duration": r.duration_seconds,
                "tools_used": [tc["name"] for tc in r.tool_calls],
            }
            for r in report.step_results
        ],
    }


@app.post("/tasks/{task_id}/plan")
async def plan_only(request: TaskRequest):
    """Generate a plan without executing it (dry run)."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")

    plan = await _engine.plan_only(request.query)
    return {
        "objective": plan.objective,
        "step_count": len(plan.steps),
        "steps": [
            {
                "order": s.order,
                "description": s.description,
                "tool_hint": s.tool_hint,
            }
            for s in plan.steps
        ],
    }


@app.post("/knowledge", response_model=KnowledgeResponse)
async def add_knowledge(request: KnowledgeRequest):
    """Add knowledge documents to the vector store."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")

    ids = _engine.add_knowledge(request.texts, request.metadatas)
    return KnowledgeResponse(document_ids=ids, count=len(ids))


# ── Background task runner ──────────────────────────────────────────────────


async def _run_task(task_id: str, query: str) -> None:
    """Execute a task in the background and store the result."""
    try:
        report = await _engine.run(query)  # type: ignore[union-attr]
        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["report"] = report
    except Exception as exc:
        logger.error("Task %s failed: %s", task_id, exc)
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(exc)
