"""
Task Engine — the main orchestrator.

Coordinates the Plan → Execute → Reflect → (Re-plan) lifecycle:
1. Planner decomposes the user query into structured steps
2. Executor runs each step with a ReAct agent
3. Reflector evaluates quality; if insufficient, triggers re-planning
4. Completed executions are stored in Pinecone for future reference

Usage::

    from taskengine.engine import TaskEngine
    from taskengine.config import get_settings

    engine = TaskEngine(get_settings())
    report = await engine.run("Your task query here")
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langchain_openai import ChatOpenAI

from taskengine.config import Settings
from taskengine.executor import StepExecutor
from taskengine.models import (
    ExecutionReport,
    PlanStatus,
    StepResult,
    TaskPlan,
)
from taskengine.planner import TaskPlanner
from taskengine.reflector import TaskReflector
from taskengine.tools import (
    calculate,
    execute_python,
    list_directory,
    read_file,
    web_search,
    write_file,
)
from taskengine.tools.file_ops import set_workspace
from taskengine.tools.knowledge import create_knowledge_tool
from taskengine.vectorstore import KnowledgeStore

logger = logging.getLogger(__name__)

# Type alias for progress callbacks
ProgressCallback = Callable[[str, dict[str, Any]], None]


class TaskEngine:
    """Autonomous task-execution engine.

    Orchestrates the full lifecycle of task planning, step-by-step
    execution, quality reflection, and optional re-planning.
    """

    def __init__(
        self,
        settings: Settings,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings
        self._on_progress = on_progress or (lambda *_: None)

        # Configure workspace for file tools
        set_workspace(settings.workspace_path)

        # Initialise components
        self._knowledge_store = KnowledgeStore(settings)
        self._planner = TaskPlanner(settings, self._knowledge_store)
        self._reflector = TaskReflector(settings)

        # Build the tool set
        knowledge_tool = create_knowledge_tool(self._knowledge_store)
        self._tools = [
            web_search,
            calculate,
            execute_python,
            read_file,
            write_file,
            list_directory,
            knowledge_tool,
        ]

        self._executor = StepExecutor(settings, self._tools)

        # LLM for summary generation
        self._llm = ChatOpenAI(
            model=settings.openai_model_name,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

    # ── Public API ──────────────────────────────────────────────────────

    async def run(self, query: str) -> ExecutionReport:
        """Execute a task from a plain-text query.

        This is the main entry point. It orchestrates:
        Plan → Execute → Reflect → (Re-plan if needed) → Report

        Args:
            query: Natural-language task description.

        Returns:
            An ``ExecutionReport`` with full results and metadata.
        """
        logger.info("═" * 60)
        logger.info("Task Engine: starting execution")
        logger.info("Query: %s", query)
        logger.info("═" * 60)

        start_time = time.monotonic()
        replan_count = 0

        # Phase 1: Plan
        self._emit("planning", {"query": query})
        plan = await self._planner.plan(query)
        self._emit("plan_created", {
            "objective": plan.objective,
            "step_count": len(plan.steps),
        })

        while True:
            # Phase 2: Execute
            plan.status = PlanStatus.EXECUTING
            self._emit("executing", {"step_count": len(plan.steps)})
            results = await self._execute_plan(plan)

            # Phase 3: Reflect
            self._emit("reflecting", {"step_count": len(results)})
            reflection = await self._reflector.reflect(query, plan, results)
            self._emit("reflection_done", {
                "score": reflection.quality_score,
                "needs_replan": reflection.needs_replan,
            })

            # Phase 4: Re-plan if needed (and within limits)
            if reflection.needs_replan and replan_count < self._settings.max_replans:
                replan_count += 1
                plan.status = PlanStatus.REPLANNING
                self._emit("replanning", {
                    "attempt": replan_count,
                    "max": self._settings.max_replans,
                    "suggestions": reflection.suggestions,
                })
                plan = await self._planner.replan(
                    original_query=query,
                    failed_plan=plan,
                    suggestions=reflection.suggestions,
                )
                continue  # Re-execute with the new plan

            # Done — either quality is good or we've exhausted re-plans
            break

        # Phase 5: Generate summary
        plan.status = PlanStatus.COMPLETED if reflection.quality_score >= 0.6 else PlanStatus.FAILED
        summary = await self._generate_summary(query, results)

        elapsed = time.monotonic() - start_time

        report = ExecutionReport(
            query=query,
            plan=plan,
            step_results=results,
            reflection=reflection,
            summary=summary,
            total_duration_seconds=round(elapsed, 2),
            success=reflection.quality_score >= 0.6,
            replan_count=replan_count,
        )

        # Store execution in Pinecone for future reference
        try:
            self._knowledge_store.add_task_execution(report)
            logger.info("Execution stored in knowledge base.")
        except Exception as exc:
            logger.warning("Failed to store execution: %s", exc)

        self._emit("completed", {
            "success": report.success,
            "duration": report.total_duration_seconds,
            "score": reflection.quality_score,
        })

        logger.info("═" * 60)
        logger.info("Task Engine: execution complete")
        logger.info("Success: %s | Duration: %.1fs | Score: %.2f",
                     report.success, elapsed, reflection.quality_score)
        logger.info("═" * 60)

        return report

    async def plan_only(self, query: str) -> TaskPlan:
        """Generate a plan without executing it (dry run).

        Args:
            query: Natural-language task description.

        Returns:
            A ``TaskPlan`` ready for review.
        """
        return await self._planner.plan(query)

    def add_knowledge(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Add knowledge documents to the vector store.

        Args:
            texts: Document texts to embed and store.
            metadatas: Optional per-document metadata.

        Returns:
            List of document IDs.
        """
        return self._knowledge_store.add_knowledge(texts, metadatas)

    # ── Private helpers ─────────────────────────────────────────────────

    async def _execute_plan(self, plan: TaskPlan) -> list[StepResult]:
        """Execute all steps in a plan sequentially, propagating context."""
        results: list[StepResult] = []
        context: dict[str, str] = {}

        for step in plan.steps:
            self._emit("step_start", {
                "order": step.order,
                "description": step.description,
                "tool_hint": step.tool_hint,
            })

            result = await self._executor.execute_step(step, context)
            results.append(result)

            # Propagate output to future steps
            if result.success and result.output:
                context[step.id] = result.output

            self._emit("step_done", {
                "order": step.order,
                "success": result.success,
                "duration": result.duration_seconds,
            })

        return results

    async def _generate_summary(
        self,
        query: str,
        results: list[StepResult],
    ) -> str:
        """Generate a human-readable summary of the execution."""
        step_outputs = "\n".join(
            f"Step {r.step_order} ({'✅' if r.success else '❌'}): "
            f"{r.output[:200] if r.output else r.error or 'no output'}"
            for r in results
        )

        from langchain_core.messages import HumanMessage

        response = await self._llm.ainvoke([
            HumanMessage(
                content=(
                    f"Summarise the following task execution in 2-3 paragraphs.\n\n"
                    f"**Query**: {query}\n\n"
                    f"**Step Results**:\n{step_outputs}\n\n"
                    f"Provide a clear, concise summary of what was accomplished, "
                    f"any issues encountered, and the final outcome."
                )
            )
        ])
        return response.content

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        """Emit a progress event via the callback."""
        self._on_progress(event, data)
        logger.debug("Event: %s — %s", event, data)
