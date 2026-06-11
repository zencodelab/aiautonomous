"""
Tests for the TaskPlanner — verifies query decomposition into structured plans.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from taskengine.models import TaskPlan, TaskStep, PlanStatus


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.openai_model_name = "gpt-4o"
    settings.openai_temperature = 0.1
    settings.max_steps = 15
    return settings


@pytest.fixture
def sample_plan():
    """Create a sample TaskPlan for testing."""
    return TaskPlan(
        query="Calculate 2+2 and save the result",
        objective="Perform arithmetic and save output to a file",
        steps=[
            TaskStep(
                id="step1",
                order=1,
                description="Calculate 2+2 using the math tool",
                tool_hint="calculate",
            ),
            TaskStep(
                id="step2",
                order=2,
                description="Save the result to result.txt",
                tool_hint="write_file",
                dependencies=["step1"],
            ),
        ],
    )


class TestTaskPlan:
    """Tests for the TaskPlan model."""

    def test_plan_creation(self, sample_plan: TaskPlan):
        assert sample_plan.query == "Calculate 2+2 and save the result"
        assert len(sample_plan.steps) == 2
        assert sample_plan.status == PlanStatus.DRAFT

    def test_pending_steps(self, sample_plan: TaskPlan):
        assert len(sample_plan.pending_steps) == 2

    def test_completed_steps(self, sample_plan: TaskPlan):
        assert len(sample_plan.completed_steps) == 0

    def test_step_dependencies(self, sample_plan: TaskPlan):
        step2 = sample_plan.steps[1]
        assert "step1" in step2.dependencies

    def test_step_order(self, sample_plan: TaskPlan):
        orders = [s.order for s in sample_plan.steps]
        assert orders == [1, 2]


class TestTaskPlanner:
    """Tests for the TaskPlanner class."""

    @pytest.mark.asyncio
    async def test_plan_returns_task_plan(self, mock_settings):
        """Verify the planner returns a valid TaskPlan."""
        from taskengine.planner import TaskPlanner

        expected_plan = TaskPlan(
            query="test query",
            objective="Test objective",
            steps=[
                TaskStep(
                    order=1,
                    description="Do something",
                    tool_hint="calculate",
                ),
            ],
        )

        with patch("taskengine.planner.ChatOpenAI") as MockLLM:
            mock_structured = AsyncMock(return_value=expected_plan)
            MockLLM.return_value.with_structured_output.return_value.ainvoke = (
                mock_structured
            )

            planner = TaskPlanner(mock_settings)
            result = await planner.plan("test query")

            assert isinstance(result, TaskPlan)
            assert result.query == "test query"
            assert len(result.steps) >= 1

    @pytest.mark.asyncio
    async def test_plan_enforces_max_steps(self, mock_settings):
        """Verify the planner truncates plans that exceed max_steps."""
        from taskengine.planner import TaskPlanner

        mock_settings.max_steps = 2
        many_steps = [
            TaskStep(order=i, description=f"Step {i}")
            for i in range(1, 6)
        ]
        big_plan = TaskPlan(
            query="test",
            objective="Test",
            steps=many_steps,
        )

        with patch("taskengine.planner.ChatOpenAI") as MockLLM:
            mock_structured = AsyncMock(return_value=big_plan)
            MockLLM.return_value.with_structured_output.return_value.ainvoke = (
                mock_structured
            )

            planner = TaskPlanner(mock_settings)
            result = await planner.plan("test")

            assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_replan_incorporates_suggestions(self, mock_settings):
        """Verify re-planning includes failure context."""
        from taskengine.planner import TaskPlanner

        revised_plan = TaskPlan(
            query="test",
            objective="Revised objective",
            steps=[
                TaskStep(order=1, description="Improved step"),
            ],
        )

        failed_plan = TaskPlan(
            query="test",
            objective="Original",
            steps=[TaskStep(order=1, description="Failed step")],
        )

        with patch("taskengine.planner.ChatOpenAI") as MockLLM:
            mock_structured = AsyncMock(return_value=revised_plan)
            MockLLM.return_value.with_structured_output.return_value.ainvoke = (
                mock_structured
            )

            planner = TaskPlanner(mock_settings)
            result = await planner.replan(
                original_query="test",
                failed_plan=failed_plan,
                suggestions=["Be more specific"],
            )

            assert result.objective == "Revised objective"
