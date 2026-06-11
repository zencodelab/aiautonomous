"""
Pydantic data models for task planning, execution, and reporting.

All models use Pydantic v2 with strict validation. These are the shared
data structures that flow between the Planner, Executor, Reflector, and Engine.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────────────


class StepStatus(str, Enum):
    """Lifecycle status of an individual task step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    """Lifecycle status of an entire task plan."""

    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


# ── Task Step ───────────────────────────────────────────────────────────────


class TaskStep(BaseModel):
    """A single atomic step in a task plan."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Short unique identifier for this step.",
    )
    order: int = Field(description="Execution order (1-indexed).")
    description: str = Field(
        description="Clear, actionable description of what this step does."
    )
    tool_hint: str | None = Field(
        default=None,
        description="Suggested tool to use (e.g., 'web_search', 'calculate'). "
        "The agent may override this.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of steps that must complete before this one.",
    )
    status: StepStatus = Field(default=StepStatus.PENDING)
    result: str | None = Field(
        default=None, description="Output produced by this step after execution."
    )


# ── Task Plan ───────────────────────────────────────────────────────────────


class TaskPlan(BaseModel):
    """Structured plan produced by the Planner from a plain-text query."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique plan identifier.",
    )
    query: str = Field(description="The original plain-text user query.")
    objective: str = Field(
        description="High-level objective restated by the planner."
    )
    steps: list[TaskStep] = Field(
        description="Ordered list of atomic execution steps."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    status: PlanStatus = Field(default=PlanStatus.DRAFT)

    @property
    def pending_steps(self) -> list[TaskStep]:
        """Return steps that haven't been executed yet."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def completed_steps(self) -> list[TaskStep]:
        """Return steps that completed successfully."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]


# ── Step Result ─────────────────────────────────────────────────────────────


class StepResult(BaseModel):
    """Outcome of executing a single step."""

    step_id: str = Field(description="ID of the step that was executed.")
    step_order: int = Field(description="Order of the step in the plan.")
    step_description: str = Field(description="Description of the step.")
    output: str = Field(description="Output / result produced by the step.")
    success: bool = Field(description="Whether the step completed successfully.")
    error: str | None = Field(
        default=None, description="Error message if the step failed."
    )
    duration_seconds: float = Field(
        description="Wall-clock time taken to execute the step."
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools invoked during this step (name + args).",
    )


# ── Reflection ──────────────────────────────────────────────────────────────


class ReflectionResult(BaseModel):
    """Assessment produced by the Reflector after all steps complete."""

    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score from 0.0 (poor) to 1.0 (excellent).",
    )
    assessment: str = Field(
        description="Detailed explanation of the quality assessment."
    )
    needs_replan: bool = Field(
        description="Whether the engine should re-plan and re-execute."
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Specific suggestions for improvement if re-planning.",
    )
    revised_query: str | None = Field(
        default=None,
        description="Revised query for re-planning (if needed).",
    )


# ── Execution Report ───────────────────────────────────────────────────────


class ExecutionReport(BaseModel):
    """Final report of the entire task execution lifecycle."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique execution identifier.",
    )
    query: str = Field(description="Original user query.")
    plan: TaskPlan = Field(description="The plan that was executed.")
    step_results: list[StepResult] = Field(
        description="Results for each executed step."
    )
    reflection: ReflectionResult | None = Field(
        default=None, description="Final quality reflection."
    )
    summary: str = Field(
        default="", description="LLM-generated summary of the execution."
    )
    total_duration_seconds: float = Field(
        default=0.0, description="Total wall-clock time for the entire execution."
    )
    success: bool = Field(
        default=False, description="Whether the overall task succeeded."
    )
    replan_count: int = Field(
        default=0, description="Number of re-planning cycles performed."
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
