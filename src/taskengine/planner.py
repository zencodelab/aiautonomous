"""
Task Planner — decomposes plain-text queries into structured step plans.

Uses an LLM with structured output to produce a ``TaskPlan`` with ordered,
atomic steps. Retrieves similar past tasks from Pinecone for few-shot context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from taskengine.models import TaskPlan, TaskStep

if TYPE_CHECKING:
    from taskengine.config import Settings
    from taskengine.vectorstore import KnowledgeStore

logger = logging.getLogger(__name__)

# ── System prompt for the planner LLM ──────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are an expert task planner for an autonomous execution engine.

Your job is to decompose a user's plain-text query into a structured plan
consisting of ordered, atomic steps that a ReAct agent can execute.

## Rules

1. **Atomic steps**: Each step should do ONE thing. If a step requires
   multiple actions, split it into separate steps.
2. **Order matters**: Steps are executed sequentially. Put dependencies first.
3. **Tool hints**: Suggest which tool is best for each step. Available tools:
   - `web_search` — Search the internet for information
   - `http_fetch` — Retrieve the full text of a known URL (docs, APIs, raw files)
   - `calculate` — Evaluate mathematical expressions
   - `execute_python` — Run Python code snippets
   - `read_file` / `write_file` — File operations in the workspace
   - `list_directory` — Explore workspace files
   - `search_knowledge` — Query the knowledge base for relevant context
4. **Be specific**: Step descriptions should be clear enough that another
   LLM can execute them without ambiguity.
5. **Include validation**: If appropriate, add a final step to verify or
   summarise the results.
6. **Keep it lean**: Don't add unnecessary steps. Aim for the minimum number
   of steps needed to accomplish the goal completely.
7. **Handle failures**: If a step might fail, note an alternative approach
   in the description.

## Output Format

Return a valid JSON object matching the TaskPlan schema with:
- `query`: The original query (copy verbatim)
- `objective`: A one-sentence restatement of the goal
- `steps`: Array of step objects with `order`, `description`, `tool_hint`
"""


# ── Planner class ──────────────────────────────────────────────────────────


class TaskPlanner:
    """Decomposes user queries into structured execution plans."""

    def __init__(
        self,
        settings: Settings,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        self._settings = settings
        self._knowledge_store = knowledge_store
        self._llm = ChatOpenAI(
            model=settings.openai_model_name,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
            max_retries=3,
        )

    async def plan(self, query: str) -> TaskPlan:
        """Decompose a plain-text query into a structured ``TaskPlan``.

        Args:
            query: The user's natural-language task description.

        Returns:
            A fully populated ``TaskPlan`` ready for execution.
        """
        logger.info("Planning task: %r", query)

        # Retrieve similar past executions for few-shot context
        context = await self._get_context(query)

        # Build messages
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        ]

        if context:
            messages.append(
                HumanMessage(
                    content=(
                        "Here are similar past tasks for reference:\n\n"
                        f"{context}\n\n"
                        "Use these as examples but adapt the plan to the "
                        "specific query below."
                    )
                )
            )

        messages.append(HumanMessage(content=f"Plan the following task:\n\n{query}"))

        # Invoke with structured output
        structured_llm = self._llm.with_structured_output(TaskPlan)
        plan: TaskPlan = await structured_llm.ainvoke(messages)

        # Enforce limits
        if len(plan.steps) > self._settings.max_steps:
            logger.warning(
                "Plan has %d steps (max %d). Truncating.",
                len(plan.steps),
                self._settings.max_steps,
            )
            plan.steps = plan.steps[: self._settings.max_steps]

        # Ensure the query is preserved exactly
        plan.query = query

        logger.info(
            "Plan created: %d steps, objective=%r",
            len(plan.steps),
            plan.objective,
        )
        return plan

    async def replan(
        self,
        original_query: str,
        failed_plan: TaskPlan,
        suggestions: list[str],
    ) -> TaskPlan:
        """Create a revised plan incorporating reflection feedback.

        Args:
            original_query: The original user query.
            failed_plan: The plan that didn't meet quality standards.
            suggestions: Improvement suggestions from the reflector.

        Returns:
            A new ``TaskPlan`` with revised steps.
        """
        logger.info("Re-planning task: %r", original_query)

        step_summary = "\n".join(
            f"  {s.order}. [{s.status.value}] {s.description}"
            for s in failed_plan.steps
        )
        suggestion_text = "\n".join(f"  - {s}" for s in suggestions)

        replan_prompt = (
            f"The previous plan did not fully satisfy the objective.\n\n"
            f"Original query: {original_query}\n\n"
            f"Previous plan steps:\n{step_summary}\n\n"
            f"Improvement suggestions:\n{suggestion_text}\n\n"
            f"Create a revised plan that addresses these issues."
        )

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=replan_prompt),
        ]

        structured_llm = self._llm.with_structured_output(TaskPlan)
        plan: TaskPlan = await structured_llm.ainvoke(messages)
        plan.query = original_query

        logger.info("Revised plan created: %d steps", len(plan.steps))
        return plan

    async def _get_context(self, query: str) -> str:
        """Retrieve relevant context from the knowledge store."""
        if self._knowledge_store is None:
            return ""

        try:
            docs = self._knowledge_store.search(query=query, k=3)
            if not docs:
                return ""
            parts = [
                f"---\n{doc.page_content[:400]}" for doc in docs
            ]
            return "\n".join(parts)
        except Exception as exc:
            logger.warning("Failed to retrieve context: %s", exc)
            return ""
