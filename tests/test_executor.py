"""
Tests for the StepExecutor — verifies step execution with mocked tools.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from taskengine.models import StepResult, StepStatus, TaskStep


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.openai_model_name = "gpt-4o"
    settings.openai_temperature = 0.1
    settings.step_timeout_seconds = 30
    return settings


@pytest.fixture
def sample_step():
    return TaskStep(
        id="test-step",
        order=1,
        description="Calculate 2+2",
        tool_hint="calculate",
    )


class TestStepExecutor:
    """Tests for the StepExecutor class."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_settings, sample_step):
        """Verify successful step execution produces a valid StepResult."""
        from taskengine.executor import StepExecutor

        mock_ai_msg = MagicMock()
        mock_ai_msg.type = "ai"
        mock_ai_msg.content = "The answer is 4."
        mock_ai_msg.tool_calls = []

        mock_result = {"messages": [mock_ai_msg]}

        with patch("taskengine.executor.ChatOpenAI"):
            with patch("taskengine.executor.create_react_agent") as mock_agent:
                mock_agent.return_value.ainvoke = AsyncMock(
                    return_value=mock_result
                )

                executor = StepExecutor(mock_settings, tools=[])
                result = await executor.execute_step(sample_step, context={})

                assert isinstance(result, StepResult)
                assert result.success is True
                assert result.output == "The answer is 4."
                assert result.step_id == "test-step"
                assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_failed_execution(self, mock_settings, sample_step):
        """Verify failed step execution captures the error."""
        from taskengine.executor import StepExecutor

        with patch("taskengine.executor.ChatOpenAI"):
            with patch("taskengine.executor.create_react_agent") as mock_agent:
                mock_agent.return_value.ainvoke = AsyncMock(
                    side_effect=RuntimeError("LLM error")
                )

                executor = StepExecutor(mock_settings, tools=[])
                result = await executor.execute_step(sample_step, context={})

                assert result.success is False
                assert "RuntimeError" in result.error
                assert sample_step.status == StepStatus.FAILED

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_settings, sample_step):
        """Verify step execution handles timeouts gracefully."""
        import asyncio
        from taskengine.executor import StepExecutor

        mock_settings.step_timeout_seconds = 0  # Instant timeout

        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(10)
            return {"messages": []}

        with patch("taskengine.executor.ChatOpenAI"):
            with patch("taskengine.executor.create_react_agent") as mock_agent:
                mock_agent.return_value.ainvoke = slow_invoke

                executor = StepExecutor(mock_settings, tools=[])
                result = await executor.execute_step(sample_step, context={})

                assert result.success is False
                assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_context_propagation(self, mock_settings, sample_step):
        """Verify prior step context is included in the prompt."""
        from taskengine.executor import StepExecutor

        mock_ai_msg = MagicMock()
        mock_ai_msg.type = "ai"
        mock_ai_msg.content = "Done with context."
        mock_ai_msg.tool_calls = []

        captured_args = {}

        async def capture_invoke(input_dict):
            captured_args["input"] = input_dict
            return {"messages": [mock_ai_msg]}

        with patch("taskengine.executor.ChatOpenAI"):
            with patch("taskengine.executor.create_react_agent") as mock_agent:
                mock_agent.return_value.ainvoke = capture_invoke

                executor = StepExecutor(mock_settings, tools=[])
                context = {"prev-step": "The result was 42."}
                await executor.execute_step(sample_step, context=context)

                # Verify the context was passed in the prompt
                messages = captured_args["input"]["messages"]
                prompt_text = messages[0].content
                assert "42" in prompt_text

    @pytest.mark.asyncio
    async def test_tool_call_extraction(self, mock_settings, sample_step):
        """Verify tool calls are extracted from agent messages."""
        from taskengine.executor import StepExecutor

        mock_tool_msg = MagicMock()
        mock_tool_msg.type = "ai"
        mock_tool_msg.content = ""
        mock_tool_msg.tool_calls = [
            {"name": "calculate", "args": {"expression": "2+2"}}
        ]

        mock_final_msg = MagicMock()
        mock_final_msg.type = "ai"
        mock_final_msg.content = "Result: 4"
        mock_final_msg.tool_calls = []

        mock_result = {"messages": [mock_tool_msg, mock_final_msg]}

        with patch("taskengine.executor.ChatOpenAI"):
            with patch("taskengine.executor.create_react_agent") as mock_agent:
                mock_agent.return_value.ainvoke = AsyncMock(
                    return_value=mock_result
                )

                executor = StepExecutor(mock_settings, tools=[])
                result = await executor.execute_step(sample_step, context={})

                assert len(result.tool_calls) == 1
                assert result.tool_calls[0]["name"] == "calculate"
