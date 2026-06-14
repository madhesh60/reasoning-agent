"""
Test suite for Research-to-Report Multi-Agent System

This test suite validates the individual agents, orchestration, and protocol implementations.
"""

from unittest.mock import patch

import pytest

# Test configuration
TEST_QUERY = "What are the top 3 investment risks in the Indian EV market?"


# Mock LLM for testing
class MockLLM:
    """Mock LLM for testing without Azure OpenAI credentials."""

    async def ainvoke(self, messages):
        """Mock LLM invoke that returns structured responses."""
        from langchain_core.messages import AIMessage

        # Parse the last message to determine response
        last_message = messages[-1] if messages else ""

        # Generate mock response based on the message content
        if "decompos" in str(last_message).lower():
            print("MOCK_LLM: Matched DECOMPOSE branch")
            response_text = '''
            {
                "plan_id": "plan_001",
                "original_query": "What are the top 3 investment risks in the Indian EV market?",
                "intent_summary": "Analyze and present the top 3 investment risks in the Indian EV market with supporting data and analysis.",
                "total_tasks": 5,
                "estimated_total_time_seconds": 120,
                "tasks": [
                    {
                        "task_id": "task_001",
                        "task_type": "web_search",
                        "description": "Search for current information on Indian EV market",
                        "priority": "critical",
                        "depends_on": [],
                        "estimated_duration_seconds": 30,
                        "agent": "Researcher",
                        "output_format": "json",
                        "search_queries": ["Indian EV market 2024", "EV investment risks India"],
                        "key_aspects": ["market size", "regulatory environment", "competition"]
                    }
                ],
                "execution_order": ["task_001", "task_002", "task_003", "task_004", "task_005"],
                "required_tools": ["mcp_web_search"],
                "confidence_score": 0.85,
                "reasoning": "The query requires web research, analysis, and report generation."
            }
            '''
        elif "search" in str(last_message).lower():
            print("MOCK_LLM: Matched SEARCH branch")
            response_text = '''
            {
                "key_findings": [
                    {
                        "category": "regulatory",
                        "statement": "Regulatory uncertainty is a major risk factor",
                        "confidence": 0.85,
                        "evidence": ["Policy changes", "Subsidy revisions"],
                        "evidence_strength": "strong",
                        "caveats": [],
                        "source_count": 5
                    }
                ],
                "risks": [
                    {
                        "risk_id": "risk_1",
                        "title": "Regulatory Policy Uncertainty",
                        "description": "Frequent changes in government policies",
                        "level": "high",
                        "probability": 0.7,
                        "impact": 0.8,
                        "risk_score": 0.56,
                        "factors": ["Policy changes", "Subsidy revisions"],
                        "mitigation": ["Diversify investments"],
                        "evidence": ["News articles", "Policy documents"],
                        "confidence": 0.85,
                        "data_quality_issues": []
                    }
                ],
                "patterns": ["Increasing government focus on EVs"],
                "comparisons": {},
                "overall_confidence": 0.8,
                "reasoning_chain": ["Identified main risks", "Cross-referenced sources"],
                "limitations": []
            }
            '''
        elif "report" in str(last_message).lower():
            response_text = '''
            {
                "title": "Investment Risks in the Indian EV Market",
                "executive_summary": "The Indian EV market presents significant opportunities but also carries substantial investment risks that require careful consideration.",
                "sections": [
                    {
                        "section_id": "findings",
                        "title": "Key Findings",
                        "content": "Based on comprehensive analysis, we identified regulatory uncertainty, supply chain dependencies, and competitive pressures as the top risks.",
                        "data": {},
                        "sources": []
                    }
                ],
                "conclusions": [
                    "Regulatory uncertainty remains the primary concern",
                    "Supply chain vulnerabilities pose operational risks",
                    "Competitive landscape requires strategic positioning"
                ],
                "recommendations": [
                    "Diversify supplier relationships",
                    "Monitor regulatory changes closely",
                    "Invest in local manufacturing capabilities"
                ],
                "appendices": []
            }
            '''
        else:
            response_text = '{"status": "success"}'

        return AIMessage(content=response_text)


@pytest.fixture
def mock_llm():
    """Provide mock LLM for tests."""
    return MockLLM()


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return {
        "endpoint": "https://mock.openai.azure.com",
        "api_key": "mock-key",
        "deployment": "gpt-4o",
        "api_version": "2024-02-01"
    }


# Agent Tests
class TestPlannerAgent:
    """Tests for the Planner Agent."""

    @pytest.mark.asyncio
    async def test_decompose_task(self, mock_llm, mock_config):
        """Test task decomposition."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.planner import PlannerAgent

            agent = PlannerAgent(llm=mock_llm)
            result = await agent.decompose_task(TEST_QUERY)

            assert result is not None
            assert result.plan_id == "plan_001"
            assert result.total_tasks > 0
            assert result.confidence_score > 0

    @pytest.mark.asyncio
    async def test_validate_plan(self, mock_llm, mock_config):
        """Test plan validation."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.planner import (
                PlannerAgent,
                ResearchPlan,
                SubTask,
                TaskPriority,
                TaskType,
            )

            agent = PlannerAgent(llm=mock_llm)

            # Create a simple plan for validation
            tasks = [
                SubTask(
                    task_id="task_001",
                    task_type=TaskType.WEB_SEARCH,
                    description="Search for information",
                    priority=TaskPriority.HIGH,
                    agent="Researcher"
                )
            ]

            plan = ResearchPlan(
                plan_id="test_plan",
                original_query=TEST_QUERY,
                intent_summary="Test intent",
                total_tasks=1,
                estimated_total_time_seconds=30,
                tasks=tasks,
                execution_order=["task_001"],
                required_tools=["web_search"],
                confidence_score=0.8,
                reasoning="Test plan"
            )

            result = await agent.validate_plan(plan)
            assert result is not None
            assert "is_valid" in result


class TestResearcherAgent:
    """Tests for the Researcher Agent."""

    @pytest.mark.asyncio
    async def test_search(self, mock_llm, mock_config):
        """Test web search."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.researcher import ResearcherAgent

            agent = ResearcherAgent(llm=mock_llm)
            result = await agent.search(TEST_QUERY, max_results=5)

            assert result is not None
            assert result.query == TEST_QUERY
            assert result.total_sources >= 0

    @pytest.mark.asyncio
    async def test_batch_search(self, mock_llm, mock_config):
        """Test batch search."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.researcher import ResearcherAgent

            agent = ResearcherAgent(llm=mock_llm)
            queries = ["EV market India", "renewable energy trends"]
            results = await agent.batch_search(queries, max_results_per_query=3)

            assert len(results) == len(queries)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("queries,expected_len", [
        ([], 0),
        (["single query"], 1),
        (["query1", "query2", "query3"], 3),
    ])
    async def test_batch_search_edge_cases(self, mock_llm, mock_config, queries, expected_len):
        """Test batch search with varying inputs including empty and single queries."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.researcher import ResearcherAgent

            agent = ResearcherAgent(llm=mock_llm)
            results = await agent.batch_search(queries, max_results_per_query=2)
            assert len(results) == expected_len


class TestAnalystAgent:
    """Tests for the Analyst Agent."""

    @pytest.mark.asyncio
    async def test_analyze(self, mock_llm, mock_config):
        """Test analysis."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.analyst import AnalystAgent

            agent = AnalystAgent(llm=mock_llm)

            research_data = {
                "sources": [{"title": "Test", "snippet": "Test content", "source_name": "Test"}],
                "search_results": []
            }

            result = await agent.analyze(TEST_QUERY, research_data)

            assert result is not None
            assert result.query == TEST_QUERY
            assert result.overall_confidence >= 0


class TestWriterAgent:
    """Tests for the Writer Agent."""

    @pytest.mark.asyncio
    async def test_generate_report(self, mock_llm, mock_config):
        """Test report generation."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.agents.writer import WriterAgent

            agent = WriterAgent(llm=mock_llm)

            analysis_data = {
                "key_findings": [
                    {"category": "risk", "statement": "Test finding", "confidence": 0.8}
                ],
                "risks_identified": [
                    {"risk_id": "r1", "title": "Test Risk", "level": "high", "risk_score": 0.5}
                ],
                "patterns_detected": ["Pattern 1"],
                "reasoning_chain": ["Step 1", "Step 2"],
                "overall_confidence": 0.8,
                "data_sources_analyzed": 5
            }

            result = await agent.generate_report(TEST_QUERY, analysis_data)

            assert result is not None
            assert result.metadata.query == TEST_QUERY
            assert len(result.sections) > 0



# MCP Tools Tests
class TestMCPWebSearchTool:
    """Tests for MCP web search tool."""

    @pytest.mark.asyncio
    async def test_search_tool_initialization(self):
        """Test search tool can be initialized."""
        from src.mcp_tools.web_search import MCPWebSearchTool

        tool = MCPWebSearchTool()
        assert tool is not None


# Orchestration Tests
class TestResearchWorkflow:
    """Tests for the Research Workflow."""

    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_llm, mock_config):
        """Test workflow can be initialized."""
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config):
            from src.orchestration.research_workflow import ResearchWorkflow

            workflow = ResearchWorkflow()
            assert workflow is not None
            assert workflow.max_retries == 3


# Integration Test
class TestSystemIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mock_llm, mock_config):
        """Test complete workflow from query to report."""
        mock_response = """
        {
            "metadata": {
                "title": "Indian EV Market Risks",
                "report_id": "r123",
                "confidence_score": 0.95
            },
            "executive_summary": "Summary of EV risks.",
            "sections": [
                {
                    "title": "Regulatory Risks",
                    "content": "Policy adjustments and subsidy concerns."
                }
            ],
            "conclusions": ["Conclusion 1"],
            "recommendations": ["Recommendation 1"],
            "citations": []
        }
        """
        with patch("src.utils.config.get_azure_openai_config", return_value=mock_config), \
             patch("src.orchestration.research_workflow._responses_call", return_value=mock_response):
            from src.orchestration.research_workflow import ResearchWorkflow

            workflow = ResearchWorkflow()

            result = await workflow.execute(TEST_QUERY)

            assert result is not None
            assert "status" in result
            assert result["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
