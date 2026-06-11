"""
Task Reflector — validates execution quality and triggers re-planning.

After all steps complete, the reflector evaluates whether the execution
adequately addressed the original query. If quality is insufficient,
it provides suggestions and can trigger a re-planning cycle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from taskengine.models import ReflectionResult, StepResult, TaskPlan

if TYPE_CHECKING:
    from taskengine.config import Settings

logger = logging.getLogger(__name__)

# ── System prompt for the reflector LLM ────────────────────────────────────

REFLECTOR_SYSTEM_PROMPT = """\
You are a quality assurance reviewer for an autonomous task execution engine.

Your job is to evaluate whether a sequence of executed steps adequately
satisfies the user's original query.

## Evaluation Criteria

1. **Completeness**: Were all aspects of the query addressed?
2. **Correctness**: Are the results factually accurate and logically sound?
3. **Quality**: Is the output useful, well-formatted, and actionable?
4. **Efficiency**: Were the steps necessary, or were there redundant actions?

## Scoring

- **0.0 – 0.3**: Poor — critical failures, missing outputs, or wrong results
- **0.3 – 0.6**: Partial — some goals met but significant gaps remain
- **0.6 – 0.8**: Good — mostly complete with minor issues
- **0.8 – 1.0**: Excellent — fully addresses the query

## Output

Return a JSON object matching the ReflectionResult schema:
- `quality_score`: float from 0.0 to 1.0
- `assessment`: detailed explanation of your evaluation
- `needs_replan`: true if quality_score < 0.6 or critical issues found
- `suggestions`: list of specific improvements (if needs_replan is true)
- `revised_query`: optional refined query for re-planning
"""


class TaskReflector:
    """Evaluates execution quality and decides whether to re-plan."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatOpenAI(
            model=settings.openai_model_name,
            temperature=0.0,  # Deterministic for evaluation
            api_key=settings.openai_api_key,
            max_retries=3,
        )

    async def reflect(
        self,
        query: str,
        plan: TaskPlan,
        results: list[StepResult],
    ) -> ReflectionResult:
        """Evaluate execution quality and return a reflection.

        Args:
            query: The original user query.
            plan: The executed plan.
            results: Results from each executed step.

        Returns:
            A ``ReflectionResult`` with quality score and suggestions.
        """
        logger.info("Reflecting on execution for query: %r", query)

        # Build the evaluation prompt
        step_details = self._format_steps(results)

        eval_prompt = (
            f"## Original Query\n{query}\n\n"
            f"## Plan Objective\n{plan.objective}\n\n"
            f"## Execution Results\n{step_details}\n\n"
            f"Evaluate the execution quality and return your assessment."
        )

        messages = [
            SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt),
        ]

        structured_llm = self._llm.with_structured_output(ReflectionResult)
        reflection: ReflectionResult = await structured_llm.ainvoke(messages)

        logger.info(
            "Reflection: score=%.2f, needs_replan=%s",
            reflection.quality_score,
            reflection.needs_replan,
        )

        return reflection

    @staticmethod
    def _format_steps(results: list[StepResult]) -> str:
        """Format step results for the evaluation prompt."""
        parts: list[str] = []
        for r in results:
            status = "✅" if r.success else "❌"
            tools_used = ", ".join(tc["name"] for tc in r.tool_calls) or "none"
            output_preview = r.output[:300] if r.output else "(no output)"
            error_info = f"\n   Error: {r.error}" if r.error else ""

            parts.append(
                f"### Step {r.step_order}: {r.step_description}\n"
                f"- Status: {status} ({'success' if r.success else 'failed'})\n"
                f"- Tools used: {tools_used}\n"
                f"- Duration: {r.duration_seconds}s\n"
                f"- Output: {output_preview}{error_info}"
            )

        return "\n\n".join(parts)
