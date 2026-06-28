"""
Tests for the TaskReflector — validates quality evaluation and re-plan decisions.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from taskengine.models import (
    ReflectionResult,
    StepResult,
    TaskPlan,
    TaskStep,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.openai_model_name = "gpt-4o"
    settings.quality_threshold = 0.6
    return settings


@pytest.fixture
def sample_plan():
    return TaskPlan(
        query="Summarise the latest AI news",
        objective="Retrieve and summarise recent AI news articles",
        steps=[
            TaskStep(order=1, description="Search for AI news"),
            TaskStep(order=2, description="Summarise the findings"),
        ],
    )


def _make_result(
    *,
    order: int = 1,
    description: str = "Do something",
    success: bool = True,
    output: str = "some output",
    error: str | None = None,
    duration: float = 1.0,
    tool_calls: list | None = None,
) -> StepResult:
    return StepResult(
        step_id=f"step{order}",
        step_order=order,
        step_description=description,
        output=output,
        success=success,
        error=error,
        duration_seconds=duration,
        tool_calls=tool_calls or [],
    )


# ── Unit tests ──────────────────────────────────────────────────────────────


class TestTaskReflectorInit:
    """TaskReflector constructs without errors when LLM is mocked."""

    def test_init_bakes_threshold_into_prompt(self, mock_settings):
        from taskengine.reflector import TaskReflector, REFLECTOR_SYSTEM_PROMPT_TEMPLATE

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        expected_prompt = REFLECTOR_SYSTEM_PROMPT_TEMPLATE.format(threshold=0.6)
        assert reflector._system_prompt == expected_prompt

    def test_custom_threshold_reflected_in_prompt(self, mock_settings):
        from taskengine.reflector import TaskReflector

        mock_settings.quality_threshold = 0.8

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        assert "0.8" in reflector._system_prompt


class TestFormatSteps:
    """_format_steps produces readable markdown for the evaluation prompt."""

    def test_successful_step_shows_checkmark(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        result = _make_result(order=1, description="search the web", success=True)
        formatted = reflector._format_steps([result])

        assert "✅" in formatted
        assert "Step 1" in formatted
        assert "search the web" in formatted

    def test_failed_step_shows_cross_and_error(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        result = _make_result(
            order=2, success=False, error="timeout after 30s", output=""
        )
        formatted = reflector._format_steps([result])

        assert "❌" in formatted
        assert "timeout after 30s" in formatted

    def test_long_output_is_truncated(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        long_output = "x" * 600
        result = _make_result(output=long_output)
        formatted = reflector._format_steps([result])

        # Only first 300 chars should appear
        assert "x" * 300 in formatted
        assert "x" * 301 not in formatted

    def test_no_output_shows_placeholder(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        result = _make_result(output="")
        formatted = reflector._format_steps([result])

        assert "(no output)" in formatted

    def test_tool_calls_listed(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        result = _make_result(
            tool_calls=[{"name": "web_search"}, {"name": "calculate"}]
        )
        formatted = reflector._format_steps([result])

        assert "web_search" in formatted
        assert "calculate" in formatted

    def test_no_tool_calls_shows_none(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        result = _make_result(tool_calls=[])
        formatted = reflector._format_steps([result])

        assert "none" in formatted

    def test_multiple_steps_separated(self, mock_settings):
        from taskengine.reflector import TaskReflector

        with patch("taskengine.reflector.ChatOpenAI"):
            reflector = TaskReflector(mock_settings)

        results = [_make_result(order=i) for i in range(1, 4)]
        formatted = reflector._format_steps(results)

        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "Step 3" in formatted


class TestReflect:
    """TaskReflector.reflect delegates to the LLM and returns ReflectionResult."""

    @pytest.mark.asyncio
    async def test_reflect_returns_reflection_result(
        self, mock_settings, sample_plan
    ):
        from taskengine.reflector import TaskReflector

        expected = ReflectionResult(
            quality_score=0.9,
            assessment="All goals met.",
            needs_replan=False,
        )

        with patch("taskengine.reflector.ChatOpenAI") as MockLLM:
            mock_structured = AsyncMock(return_value=expected)
            MockLLM.return_value.with_structured_output.return_value.ainvoke = (
                mock_structured
            )

            reflector = TaskReflector(mock_settings)
            result = await reflector.reflect(
                query="Summarise AI news",
                plan=sample_plan,
                results=[_make_result()],
            )

        assert isinstance(result, ReflectionResult)
        assert result.quality_score == 0.9
        assert result.needs_replan is False

    @pytest.mark.asyncio
    async def test_reflect_low_score_triggers_replan(
        self, mock_settings, sample_plan
    ):
        from taskengine.reflector import TaskReflector

        expected = ReflectionResult(
            quality_score=0.3,
            assessment="Output was incomplete.",
            needs_replan=True,
            suggestions=["Try a different search query"],
        )

        with patch("taskengine.reflector.ChatOpenAI") as MockLLM:
            mock_structured = AsyncMock(return_value=expected)
            MockLLM.return_value.with_structured_output.return_value.ainvoke = (
                mock_structured
            )

            reflector = TaskReflector(mock_settings)
            result = await reflector.reflect(
                query="Summarise AI news",
                plan=sample_plan,
                results=[_make_result(success=False, output="")],
            )

        assert result.needs_replan is True
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_reflect_passes_query_and_objective_to_llm(
        self, mock_settings, sample_plan
    ):
        """The eval prompt sent to the LLM must contain the query and objective."""
        from taskengine.reflector import TaskReflector

        captured: list[list] = []

        async def capture_invoke(messages):
            captured.append(messages)
            return ReflectionResult(
                quality_score=0.7, assessment="ok", needs_replan=False
            )

        with patch("taskengine.reflector.ChatOpenAI") as MockLLM:
            MockLLM.return_value.with_structured_output.return_value.ainvoke = (
                capture_invoke
            )

            reflector = TaskReflector(mock_settings)
            await reflector.reflect(
                query="unique-query-string",
                plan=sample_plan,
                results=[_make_result()],
            )

        assert captured, "LLM was never called"
        human_msg = captured[0][1]
        assert "unique-query-string" in human_msg.content
        assert sample_plan.objective in human_msg.content
