"""
Step Executor — runs individual task steps using a LangGraph ReAct agent.

Each step gets its own agent invocation with the full tool set plus
context from previously completed steps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from taskengine.models import StepResult, StepStatus, TaskStep

if TYPE_CHECKING:
    from taskengine.config import Settings

logger = logging.getLogger(__name__)


class StepExecutor:
    """Executes individual task steps using a ReAct agent powered by LangGraph.

    Each step is executed by invoking a fresh agent with:
    - The step description as the primary instruction
    - Context from previously completed steps
    - The full set of bound tools
    """

    def __init__(
        self,
        settings: Settings,
        tools: list[Any],
    ) -> None:
        self._settings = settings
        self._tools = tools
        self._llm = ChatOpenAI(
            model=settings.openai_model_name,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
            max_retries=3,
        )

    async def execute_step(
        self,
        step: TaskStep,
        context: dict[str, str],
    ) -> StepResult:
        """Execute a single task step using the ReAct agent.

        Args:
            step: The step to execute.
            context: Results from previously completed steps,
                     keyed by step ID.

        Returns:
            A ``StepResult`` capturing the output, success, and timing.
        """
        logger.info("Executing step %d: %s", step.order, step.description)
        step.status = StepStatus.RUNNING
        start_time = time.monotonic()

        try:
            # Build the agent
            agent = create_react_agent(model=self._llm, tools=self._tools)

            # Construct the prompt with step context
            prompt = self._build_prompt(step, context)

            # Run the agent with a timeout
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": [HumanMessage(content=prompt)]}),
                timeout=self._settings.step_timeout_seconds,
            )

            # Extract the final output and tool call history
            output, tool_calls = self._extract_output(result)

            elapsed = time.monotonic() - start_time
            step.status = StepStatus.COMPLETED
            step.result = output

            logger.info(
                "Step %d completed in %.1fs (%d tool calls)",
                step.order,
                elapsed,
                len(tool_calls),
            )

            return StepResult(
                step_id=step.id,
                step_order=step.order,
                step_description=step.description,
                output=output,
                success=True,
                duration_seconds=round(elapsed, 2),
                tool_calls=tool_calls,
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            step.status = StepStatus.FAILED
            error_msg = (
                f"Step timed out after {self._settings.step_timeout_seconds}s"
            )
            logger.error("Step %d timed out.", step.order)
            return StepResult(
                step_id=step.id,
                step_order=step.order,
                step_description=step.description,
                output="",
                success=False,
                error=error_msg,
                duration_seconds=round(elapsed, 2),
            )

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            step.status = StepStatus.FAILED
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Step %d failed: %s", step.order, error_msg)
            return StepResult(
                step_id=step.id,
                step_order=step.order,
                step_description=step.description,
                output="",
                success=False,
                error=error_msg,
                duration_seconds=round(elapsed, 2),
            )

    def _build_prompt(self, step: TaskStep, context: dict[str, str]) -> str:
        """Build the agent prompt for a step, including prior context."""
        parts: list[str] = [
            "You are executing one step of a larger task plan.",
            "",
            f"## Current Step (#{step.order})",
            f"**Instruction**: {step.description}",
        ]

        if step.tool_hint:
            parts.append(f"**Suggested tool**: `{step.tool_hint}`")

        if context:
            parts.append("")
            parts.append("## Context from Previous Steps")
            for step_id, result_text in context.items():
                parts.append(f"- **Step {step_id}**: {result_text[:500]}")

        parts.extend(
            [
                "",
                "## Instructions",
                "- Execute this step completely and accurately.",
                "- Use the appropriate tool(s) to accomplish the task.",
                "- Provide a clear, concise summary of what you accomplished.",
                "- If you encounter an error, explain what went wrong.",
            ]
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_output(result: dict[str, Any]) -> tuple[str, list[dict]]:
        """Extract the final text output and tool call log from agent result.

        Returns:
            Tuple of (final_output_text, list_of_tool_call_dicts).
        """
        messages = result.get("messages", [])
        tool_calls: list[dict] = []
        final_output = ""

        for msg in messages:
            # Capture tool calls from AI messages
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "name": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                        }
                    )

        # The last AI message is the final response
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                msg_type = getattr(msg, "type", "")
                if msg_type == "ai":
                    final_output = msg.content
                    break

        if not final_output:
            # Fallback: use the last message with content
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    final_output = msg.content
                    break

        return final_output, tool_calls
