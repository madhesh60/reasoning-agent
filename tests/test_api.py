import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "endpoint" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_readiness_endpoint():
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
    }

    # 1. Missing case
    with patch.dict(os.environ, {}, clear=True):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/readiness")
            assert response.status_code == 503
            assert "Missing environment variables" in response.json()["detail"]

    # 2. Ready case
    with patch.dict(os.environ, env_vars, clear=False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/readiness")
            assert response.status_code == 200
            assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_ask_endpoint():
    mock_workflow_result = {
        "status": "completed",
        "query": "test query",
        "report": None,
        "confidence_score": 0.9,
        "metadata": {"completed_tasks": ["planner", "researcher"], "failed_tasks": []},
    }

    with patch(
        "src.orchestration.research_workflow.ResearchWorkflow.execute", new_callable=AsyncMock
    ) as mock_execute:
        mock_execute.return_value = mock_workflow_result

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            payload = {"query": "test query", "max_retries": 2, "enable_web_search": True}
            response = await ac.post("/ask", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["query"] == "test query"
            assert data["confidence"] == 0.9
            assert "report" in data
            assert data["reasoning_steps"] == ["planner", "researcher"]


@pytest.mark.asyncio
async def test_ask_endpoint_validation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"query": "test query", "max_retries": 0, "enable_web_search": True}
        response = await ac.post("/ask", json=payload)
        assert response.status_code == 422

        payload["max_retries"] = 6
        response = await ac.post("/ask", json=payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_competitive_endpoint_503_when_unset():
    with patch.dict(
        os.environ, {"AZURE_EXISTING_AGENT_ID": "", "AZURE_PROJECT_ENDPOINT": ""}, clear=False
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            payload = {"query": "test query", "company": "test company"}
            response = await ac.post("/competitive", json=payload)
            assert response.status_code == 503
            assert "not configured in .env" in response.json()["detail"]
