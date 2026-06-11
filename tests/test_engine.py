"""
Integration tests for the TaskEngine orchestrator.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from taskengine.models import (
    ExecutionReport,
    PlanStatus,
    ReflectionResult,
    StepResult,
    StepStatus,
    TaskPlan,
    TaskStep,
)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "test-key"
    settings.openai_model_name = "gpt-4o"
    settings.openai_temperature = 0.1
    settings.pinecone_api_key = "test-pinecone-key"
    settings.pinecone_index_name = "test-index"
    settings.pinecone_cloud = "aws"
    settings.pinecone_region = "us-east-1"
    settings.pinecone_dimension = 1536
    settings.pinecone_metric = "cosine"
    settings.max_steps = 15
    settings.max_replans = 2
    settings.step_timeout_seconds = 30
    settings.workspace_dir = "./test_workspace"
    settings.workspace_path = MagicMock()
    return settings


@pytest.fixture
def sample_plan():
    return TaskPlan(
        query="Test task",
        objective="Test objective",
        steps=[
            TaskStep(id="s1", order=1, description="Step 1", tool_hint="calculate"),
            TaskStep(id="s2", order=2, description="Step 2", tool_hint="write_file"),
        ],
    )


@pytest.fixture
def good_reflection():
    return ReflectionResult(
        quality_score=0.9,
        assessment="Excellent execution.",
        needs_replan=False,
    )


@pytest.fixture
def poor_reflection():
    return ReflectionResult(
        quality_score=0.3,
        assessment="Poor execution — missing key steps.",
        needs_replan=True,
        suggestions=["Add data validation step"],
    )


class TestTaskEngine:
    """Integration tests for the full engine lifecycle."""

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, mock_settings, sample_plan, good_reflection
    ):
        """Verify the full plan → execute → reflect lifecycle."""
        step_results = [
            StepResult(
                step_id="s1",
                step_order=1,
                step_description="Step 1",
                output="Result 1",
                success=True,
                duration_seconds=1.0,
            ),
            StepResult(
                step_id="s2",
                step_order=2,
                step_description="Step 2",
                output="Result 2",
                success=True,
                duration_seconds=1.5,
            ),
        ]

        with (
            patch("taskengine.engine.KnowledgeStore") as MockStore,
            patch("taskengine.engine.TaskPlanner") as MockPlanner,
            patch("taskengine.engine.StepExecutor") as MockExecutor,
            patch("taskengine.engine.TaskReflector") as MockReflector,
            patch("taskengine.engine.ChatOpenAI") as MockLLM,
            patch("taskengine.engine.set_workspace"),
            patch("taskengine.engine.create_knowledge_tool", return_value=MagicMock()),
        ):
            # Configure mocks
            MockPlanner.return_value.plan = AsyncMock(return_value=sample_plan)
            MockExecutor.return_value.execute_step = AsyncMock(
                side_effect=step_results
            )
            MockReflector.return_value.reflect = AsyncMock(
                return_value=good_reflection
            )
            mock_summary_msg = MagicMock()
            mock_summary_msg.content = "Task completed successfully."
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_summary_msg)
            MockStore.return_value.add_task_execution = MagicMock()

            from taskengine.engine import TaskEngine

            engine = TaskEngine(mock_settings)
            report = await engine.run("Test task")

            assert isinstance(report, ExecutionReport)
            assert report.success is True
            assert report.replan_count == 0
            assert len(report.step_results) == 2

    @pytest.mark.asyncio
    async def test_replan_cycle(
        self, mock_settings, sample_plan, poor_reflection, good_reflection
    ):
        """Verify the engine re-plans when reflection quality is low."""
        step_result = StepResult(
            step_id="s1",
            step_order=1,
            step_description="Step 1",
            output="Partial result",
            success=True,
            duration_seconds=1.0,
        )

        revised_plan = TaskPlan(
            query="Test task",
            objective="Revised objective",
            steps=[
                TaskStep(id="r1", order=1, description="Revised step"),
            ],
        )

        with (
            patch("taskengine.engine.KnowledgeStore") as MockStore,
            patch("taskengine.engine.TaskPlanner") as MockPlanner,
            patch("taskengine.engine.StepExecutor") as MockExecutor,
            patch("taskengine.engine.TaskReflector") as MockReflector,
            patch("taskengine.engine.ChatOpenAI") as MockLLM,
            patch("taskengine.engine.set_workspace"),
            patch("taskengine.engine.create_knowledge_tool", return_value=MagicMock()),
        ):
            MockPlanner.return_value.plan = AsyncMock(return_value=sample_plan)
            MockPlanner.return_value.replan = AsyncMock(return_value=revised_plan)
            MockExecutor.return_value.execute_step = AsyncMock(
                return_value=step_result
            )
            # First reflect returns poor, second returns good
            MockReflector.return_value.reflect = AsyncMock(
                side_effect=[poor_reflection, good_reflection]
            )
            mock_summary_msg = MagicMock()
            mock_summary_msg.content = "Revised execution summary."
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_summary_msg)
            MockStore.return_value.add_task_execution = MagicMock()

            from taskengine.engine import TaskEngine

            engine = TaskEngine(mock_settings)
            report = await engine.run("Test task")

            assert report.replan_count == 1
            assert report.success is True

    @pytest.mark.asyncio
    async def test_plan_only(self, mock_settings, sample_plan):
        """Verify plan_only returns a plan without executing."""
        with (
            patch("taskengine.engine.KnowledgeStore"),
            patch("taskengine.engine.TaskPlanner") as MockPlanner,
            patch("taskengine.engine.StepExecutor"),
            patch("taskengine.engine.TaskReflector"),
            patch("taskengine.engine.ChatOpenAI"),
            patch("taskengine.engine.set_workspace"),
            patch("taskengine.engine.create_knowledge_tool", return_value=MagicMock()),
        ):
            MockPlanner.return_value.plan = AsyncMock(return_value=sample_plan)

            from taskengine.engine import TaskEngine

            engine = TaskEngine(mock_settings)
            plan = await engine.plan_only("Test task")

            assert isinstance(plan, TaskPlan)
            assert len(plan.steps) == 2
