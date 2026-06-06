#!/usr/bin/env python3
"""
A2A Unified Multi-Agent Server

This script launches a unified FastAPI server hosting all four agents
(Planner, Researcher, Analyst, Writer) under their respective paths
on port 8080.
"""

import sys
import uvicorn
from fastapi import FastAPI
import structlog

# Configure logging
from src.utils.logging import configure_logging
from src.utils.config import load_environment, get_a2a_config

configure_logging(log_level="INFO", json_format=False)
logger = structlog.get_logger(__name__)

# Load environment
load_environment()

import os
if not os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") == "your-api-key-here":
    logger.info("no_credentials_detected_running_in_mock_mode")
    os.environ["AZURE_OPENAI_API_KEY"] = "mock-key"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://mock.openai.azure.com"

    from langchain_core.messages import AIMessage
    class MockLLM:
        async def ainvoke(self, messages, *args, **kwargs):
            last_message = messages[-1] if messages else ""
            if "decompos" in str(last_message).lower():
                response_text = '''
                {
                    "plan_id": "plan_001",
                    "original_query": "What are the top 3 investment risks in the Indian EV market?",
                    "intent_summary": "Analyze and present the top 3 investment risks in the Indian EV market with supporting data and analysis.",
                    "total_tasks": 2,
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
                        },
                        {
                            "task_id": "task_002",
                            "task_type": "report_generation",
                            "description": "Generate final report",
                            "priority": "high",
                            "depends_on": ["task_001"],
                            "estimated_duration_seconds": 60,
                            "agent": "Writer",
                            "output_format": "text",
                            "search_queries": [],
                            "key_aspects": []
                        }
                    ],
                    "execution_order": ["task_001", "task_002"],
                    "required_tools": ["mcp_web_search"],
                    "confidence_score": 0.85,
                    "reasoning": "The query requires web research, analysis, and report generation."
                }
                '''
            elif "search" in str(last_message).lower() or "sources" in str(last_message).lower():
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
            elif "report" in str(last_message).lower() or "executive" in str(last_message).lower():
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

        def invoke(self, messages, *args, **kwargs):
            import asyncio
            return asyncio.run(self.ainvoke(messages, *args, **kwargs))

    from unittest.mock import patch
    mock_llm = MockLLM()
    patcher = patch("langchain_openai.AzureChatOpenAI.ainvoke", new=mock_llm.ainvoke)
    patcher.start()
    patcher_sync = patch("langchain_openai.AzureChatOpenAI.invoke", new=mock_llm.invoke)
    patcher_sync.start()

from src.agents.planner import PlannerAgent, ResearchPlan
from src.agents.researcher import ResearcherAgent, ResearchResults
from src.agents.analyst import AnalystAgent, AnalysisResults, AnalysisInsight
from src.agents.writer import WriterAgent, GeneratedReport, ReportFormat
from src.a2a.server import A2AServer

# Create the parent FastAPI app
app = FastAPI(
    title="A2A Unified Multi-Agent Server",
    description="Unified endpoint hosting all research agents under path routing",
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# 1. Planner Agent Server Setup
# ------------------------------------------------------------------------------
planner_agent = PlannerAgent()
planner_server = A2AServer(
    agent_name="planner",
    agent_description="Task decomposition and research planning agent",
    agent_version="1.0.0"
)
planner_server.register_capability("task_execution")
planner_server.register_capability("coordination")

async def planner_decompose_task_handler(query: str):
    plan = await planner_agent.decompose_task(query)
    return plan.model_dump()

async def planner_validate_plan_handler(plan: dict):
    plan_obj = ResearchPlan(**plan)
    validation = await planner_agent.validate_plan(plan_obj)
    return validation

async def planner_refine_plan_handler(plan: dict, feedback: str):
    plan_obj = ResearchPlan(**plan)
    refined = await planner_agent.refine_plan(plan_obj, feedback)
    return refined.model_dump()

planner_server.register_endpoint(
    "decompose_task",
    planner_decompose_task_handler,
    description="Decompose a query into subtasks",
    parameters={"query": {"type": "string"}}
)
planner_server.register_endpoint(
    "validate_plan",
    planner_validate_plan_handler,
    description="Validate a research plan",
    parameters={"plan": {"type": "object"}}
)
planner_server.register_endpoint(
    "refine_plan",
    planner_refine_plan_handler,
    description="Refine a plan based on feedback",
    parameters={"plan": {"type": "object"}, "feedback": {"type": "string"}}
)

# ------------------------------------------------------------------------------
# 2. Researcher Agent Server Setup
# ------------------------------------------------------------------------------
researcher_agent = ResearcherAgent()
researcher_server = A2AServer(
    agent_name="researcher",
    agent_description="Web search and document gathering agent",
    agent_version="1.0.0"
)
researcher_server.register_capability("data_retrieval")

async def researcher_search_handler(query: str, max_results: int = 10, search_type: str = "general"):
    results = await researcher_agent.search(query, max_results, search_type)
    return results.model_dump()

async def researcher_batch_search_handler(queries: list[str], max_results_per_query: int = 5):
    results_list = await researcher_agent.batch_search(queries, max_results_per_query)
    return [r.model_dump() for r in results_list]

async def researcher_fetch_document_handler(url: str):
    doc = await researcher_agent.fetch_document(url)
    return doc

researcher_server.register_endpoint(
    "search",
    researcher_search_handler,
    description="Search the web for information",
    parameters={
        "query": {"type": "string"},
        "max_results": {"type": "integer", "default": 10},
        "search_type": {"type": "string", "default": "general"}
    }
)
researcher_server.register_endpoint(
    "batch_search",
    researcher_batch_search_handler,
    description="Execute multiple searches in parallel",
    parameters={
        "queries": {"type": "array", "items": {"type": "string"}},
        "max_results_per_query": {"type": "integer", "default": 5}
    }
)
researcher_server.register_endpoint(
    "fetch_document",
    researcher_fetch_document_handler,
    description="Fetch document content from URL",
    parameters={"url": {"type": "string"}}
)

# ------------------------------------------------------------------------------
# 3. Analyst Agent Server Setup
# ------------------------------------------------------------------------------
analyst_agent = AnalystAgent()
analyst_server = A2AServer(
    agent_name="analyst",
    agent_description="Data reasoning and insight extraction agent",
    agent_version="1.0.0"
)
analyst_server.register_capability("analysis")
analyst_server.register_capability("risk_assessment")

async def analyst_analyze_handler(query: str, research_data: dict):
    results = await analyst_agent.analyze(query, research_data)
    return results.model_dump()

async def analyst_assess_risk_handler(risk_data: dict):
    risk = await analyst_agent.assess_risk(risk_data)
    return risk.model_dump()

async def analyst_verify_findings_handler(findings: list, sources: list):
    findings_objs = [AnalysisInsight(**f) for f in findings]
    verification = await analyst_agent.verify_findings(findings_objs, sources)
    return verification

analyst_server.register_endpoint(
    "analyze",
    analyst_analyze_handler,
    description="Analyze research data and draw findings",
    parameters={"query": {"type": "string"}, "research_data": {"type": "object"}}
)
analyst_server.register_endpoint(
    "assess_risk",
    analyst_assess_risk_handler,
    description="Perform detailed risk assessment",
    parameters={"risk_data": {"type": "object"}}
)
analyst_server.register_endpoint(
    "verify_findings",
    analyst_verify_findings_handler,
    description="Verify findings against original sources",
    parameters={
        "findings": {"type": "array", "items": {"type": "object"}},
        "sources": {"type": "array", "items": {"type": "object"}}
    }
)

# ------------------------------------------------------------------------------
# 4. Writer Agent Server Setup
# ------------------------------------------------------------------------------
writer_agent = WriterAgent()
writer_server = A2AServer(
    agent_name="writer",
    agent_description="Report generation and formatting agent",
    agent_version="1.0.0"
)
writer_server.register_capability("reporting")

async def writer_generate_report_handler(query: str, analysis_results: dict):
    report = await writer_agent.generate_report(query, analysis_results)
    return report.model_dump()

async def writer_format_output_handler(report: dict, format_type: str):
    report_obj = GeneratedReport(**report)
    fmt = ReportFormat(format_type)
    formatted = await writer_agent.format_output(report_obj, fmt)
    return formatted

writer_server.register_endpoint(
    "generate_report",
    writer_generate_report_handler,
    description="Generate comprehensive report from analysis results",
    parameters={"query": {"type": "string"}, "analysis_results": {"type": "object"}}
)
writer_server.register_endpoint(
    "format_output",
    writer_format_output_handler,
    description="Format report in markdown or HTML",
    parameters={"report": {"type": "object"}, "format_type": {"type": "string"}}
)

# ------------------------------------------------------------------------------
# Mount Sub-Applications
# ------------------------------------------------------------------------------
app.mount("/planner", planner_server.app)
app.mount("/researcher", researcher_server.app)
app.mount("/analyst", analyst_server.app)
app.mount("/writer", writer_server.app)

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "A2A Unified Multi-Agent Server",
        "agents": {
            "planner": "/planner/info",
            "researcher": "/researcher/info",
            "analyst": "/analyst/info",
            "writer": "/writer/info"
        }
    }

def main():
    a2a_config = get_a2a_config()
    host = a2a_config["host"]
    port = a2a_config["port"]

    logger.info("starting_unified_a2a_server", host=host, port=port)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
