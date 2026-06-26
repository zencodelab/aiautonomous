"""
Unit tests for the FastAPI REST interface (taskengine.api.server).

All external dependencies (TaskEngine, Settings) are mocked so tests run
without live API keys.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from taskengine.models import (
    ExecutionReport,
    PlanStatus,
    ReflectionResult,
    StepResult,
    StepStatus,
    TaskPlan,
    TaskStep,
)


# ── Lifespan stub (prevents real engine initialisation) ──────────────────────


@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_plan() -> TaskPlan:
    return TaskPlan(
        query="test query",
        objective="test objective",
        steps=[
            TaskStep(id="s1", order=1, description="Step one", tool_hint="calculate"),
        ],
    )


@pytest.fixture()
def sample_report(sample_plan: TaskPlan) -> ExecutionReport:
    step_result = StepResult(
        step_id="s1",
        step_order=1,
        step_description="Step one",
        success=True,
        output="42",
        tool_calls=[{"name": "calculate"}],
        duration_seconds=0.1,
    )
    reflection = ReflectionResult(
        quality_score=0.9,
        assessment="Looks good.",
        needs_replan=False,
        suggestions=[],
    )
    return ExecutionReport(
        query="test query",
        plan=sample_plan,
        step_results=[step_result],
        reflection=reflection,
        success=True,
        summary="Done.",
        total_duration_seconds=0.5,
        replan_count=0,
    )


@pytest.fixture()
def mock_engine(sample_plan: TaskPlan, sample_report: ExecutionReport) -> MagicMock:
    engine = MagicMock()
    engine.run = AsyncMock(return_value=sample_report)
    engine.plan_only = AsyncMock(return_value=sample_plan)
    engine.add_knowledge = MagicMock(return_value=["doc-1", "doc-2"])
    return engine


@pytest.fixture()
def client(mock_engine: MagicMock) -> TestClient:
    """Return a TestClient with the engine pre-injected, bypassing real lifespan."""
    from taskengine.api import server as srv

    srv._tasks.clear()
    with patch.object(srv.app.router, "lifespan_context", _noop_lifespan):
        srv._engine = mock_engine
        with TestClient(srv.app, raise_server_exceptions=True) as c:
            yield c
    srv._engine = None
    srv._tasks.clear()


# ── Health check ─────────────────────────────────────────────────────────────


def test_health_check_engine_present(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["engine"] is True


def test_health_check_engine_absent() -> None:
    from taskengine.api import server as srv

    srv._tasks.clear()
    with patch.object(srv.app.router, "lifespan_context", _noop_lifespan):
        srv._engine = None
        with TestClient(srv.app, raise_server_exceptions=True) as c:
            resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["engine"] is False


# ── Submit task ───────────────────────────────────────────────────────────────


def test_submit_task_returns_running(client: TestClient) -> None:
    resp = client.post("/tasks", json={"query": "do something"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert "task_id" in body


def test_submit_task_without_engine() -> None:
    from taskengine.api import server as srv

    srv._tasks.clear()
    with patch.object(srv.app.router, "lifespan_context", _noop_lifespan):
        srv._engine = None
        with TestClient(srv.app, raise_server_exceptions=True) as c:
            resp = c.post("/tasks", json={"query": "fail"})
    assert resp.status_code == 503


# ── Get task status ───────────────────────────────────────────────────────────


def test_get_task_not_found(client: TestClient) -> None:
    resp = client.get("/tasks/nonexistent-id")
    assert resp.status_code == 404


def test_get_task_running(client: TestClient) -> None:
    from taskengine.api import server as srv

    task_id = "test-running-id"
    srv._tasks[task_id] = {"status": "running", "report": None, "error": None}
    resp = client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


def test_get_task_failed(client: TestClient) -> None:
    from taskengine.api import server as srv

    task_id = "test-failed-id"
    srv._tasks[task_id] = {"status": "failed", "report": None, "error": "boom"}
    resp = client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"] == "boom"


def test_get_task_completed(client: TestClient, sample_report: ExecutionReport) -> None:
    from taskengine.api import server as srv

    task_id = "test-completed-id"
    srv._tasks[task_id] = {"status": "completed", "report": sample_report, "error": None}
    resp = client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["success"] is True
    assert body["quality_score"] == pytest.approx(0.9)
    assert body["replan_count"] == 0
    assert len(body["plan"]["steps"]) == 1
    assert body["step_results"][0]["tools_used"] == ["calculate"]
    assert body["step_results"][0]["output"] == "42"


# ── Plan-only endpoint ────────────────────────────────────────────────────────


def test_plan_only_returns_plan(client: TestClient, mock_engine: MagicMock) -> None:
    resp = client.post("/tasks/any-id/plan", json={"query": "plan this"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["objective"] == "test objective"
    assert body["step_count"] == 1
    assert body["steps"][0]["description"] == "Step one"
    mock_engine.plan_only.assert_awaited_once_with("plan this")


def test_plan_only_without_engine() -> None:
    from taskengine.api import server as srv

    srv._tasks.clear()
    with patch.object(srv.app.router, "lifespan_context", _noop_lifespan):
        srv._engine = None
        with TestClient(srv.app, raise_server_exceptions=True) as c:
            resp = c.post("/tasks/any-id/plan", json={"query": "x"})
    assert resp.status_code == 503


# ── Add knowledge ─────────────────────────────────────────────────────────────


def test_add_knowledge(client: TestClient, mock_engine: MagicMock) -> None:
    resp = client.post(
        "/knowledge",
        json={"texts": ["doc a", "doc b"], "metadatas": [{"src": "x"}, {"src": "y"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["document_ids"] == ["doc-1", "doc-2"]
    mock_engine.add_knowledge.assert_called_once_with(
        ["doc a", "doc b"], [{"src": "x"}, {"src": "y"}]
    )


def test_add_knowledge_without_engine() -> None:
    from taskengine.api import server as srv

    srv._tasks.clear()
    with patch.object(srv.app.router, "lifespan_context", _noop_lifespan):
        srv._engine = None
        with TestClient(srv.app, raise_server_exceptions=True) as c:
            resp = c.post("/knowledge", json={"texts": ["x"]})
    assert resp.status_code == 503


# ── Background runner stores result ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_task_stores_completed(
    mock_engine: MagicMock, sample_report: ExecutionReport
) -> None:
    from taskengine.api import server as srv

    srv._engine = mock_engine
    srv._tasks["bg-id"] = {"status": "running", "report": None, "error": None}
    await srv._run_task("bg-id", "test query")
    assert srv._tasks["bg-id"]["status"] == "completed"
    assert srv._tasks["bg-id"]["report"] is sample_report


@pytest.mark.asyncio
async def test_run_task_stores_error_on_exception(mock_engine: MagicMock) -> None:
    from taskengine.api import server as srv

    mock_engine.run = AsyncMock(side_effect=RuntimeError("exploded"))
    srv._engine = mock_engine
    srv._tasks["err-id"] = {"status": "running", "report": None, "error": None}
    await srv._run_task("err-id", "fail query")
    assert srv._tasks["err-id"]["status"] == "failed"
    assert "exploded" in srv._tasks["err-id"]["error"]
