#!/usr/bin/env python3
"""
Sample script demonstrating the Research-to-Report Multi-Agent System.

This script shows how to use the multi-agent system programmatically.

Usage:
    python scripts/run_sample.py
"""

import asyncio
import json
from datetime import datetime

# Configure logging
from src.utils.logging import configure_logging
from src.utils.config import load_environment

configure_logging(log_level="INFO", json_format=False)

import structlog
logger = structlog.get_logger(__name__)


async def demo_individual_agents():
    """Demonstrate individual agent functionality."""
    print("=" * 60)
    print("DEMO: Individual Agent Testing")
    print("=" * 60)

    # Load environment
    load_environment()

    # Demo Planner Agent
    print("\n--- PLANNER AGENT ---")
    from src.agents.planner import PlannerAgent

    planner = PlannerAgent()
    query = "What are the top 3 investment risks in the Indian EV market?"

    print(f"Query: {query}\n")
    plan = await planner.decompose_task(query)

    print(f"Plan ID: {plan.plan_id}")
    print(f"Intent: {plan.intent_summary}")
    print(f"Tasks: {plan.total_tasks}")
    print(f"Confidence: {plan.confidence_score:.2f}")

    # Demo Researcher Agent
    print("\n--- RESEARCHER AGENT ---")
    from src.agents.researcher import ResearcherAgent

    researcher = ResearcherAgent()
    research_results = await researcher.search(query, max_results=5)

    print(f"Search completed: {research_results.total_sources} sources found")
    print(f"High confidence sources: {len(research_results.high_confidence_sources)}")

    # Demo Analyst Agent
    print("\n--- ANALYST AGENT ---")
    from src.agents.analyst import AnalystAgent

    analyst = AnalystAgent()
    research_data = {
        "sources": [
            {"title": r.title, "snippet": r.snippet, "source_name": r.source_name}
            for r in research_results.high_confidence_sources
        ],
        "search_results": research_results.sources_used
    }

    analysis_results = await analyst.analyze(query, research_data)

    print(f"Analysis ID: {analysis_results.analysis_id}")
    print(f"Key findings: {len(analysis_results.key_findings)}")
    print(f"Risks identified: {len(analysis_results.risks_identified)}")
    print(f"Overall confidence: {analysis_results.overall_confidence:.2f}")

    # Demo Writer Agent
    print("\n--- WRITER AGENT ---")
    from src.agents.writer import WriterAgent

    writer = WriterAgent()
    analysis_data = {
        "key_findings": [
            {"category": f.category, "statement": f.statement, "confidence": f.confidence}
            for f in analysis_results.key_findings
        ],
        "risks_identified": [
            {"title": r.title, "level": r.level.value, "risk_score": r.risk_score}
            for r in analysis_results.risks_identified
        ],
        "patterns_detected": analysis_results.patterns_detected,
        "reasoning_chain": analysis_results.reasoning_chain,
        "overall_confidence": analysis_results.overall_confidence,
        "data_sources_analyzed": analysis_results.data_sources_analyzed
    }

    report = await writer.generate_report(query, analysis_data)

    print(f"Report ID: {report.metadata.report_id}")
    print(f"Title: {report.metadata.title}")
    print(f"Sections: {len(report.sections)}")
    print(f"Recommendations: {len(report.recommendations)}")

    return report


async def demo_workflow():
    """Demonstrate the complete workflow."""
    print("\n" + "=" * 60)
    print("DEMO: Complete Workflow Execution")
    print("=" * 60)

    # Load environment
    load_environment()

    from src.orchestration.research_workflow import ResearchWorkflow

    workflow = ResearchWorkflow(
        enable_a2a=True,
        enable_mcp=True,
        max_retries=3
    )

    # Run sample query
    query = "What are the top 3 investment risks in the Indian EV market?"

    print(f"\nExecuting workflow for query:")
    print(f"  {query}\n")

    result = await workflow.execute(query)

    print("\n--- RESULTS ---")
    print(f"Status: {result['status']}")
    print(f"Confidence: {result.get('confidence_score', 'N/A'):.2f}" if result.get('confidence_score') else "N/A")

    if result.get("metadata"):
        print(f"\nCompleted Tasks:")
        for task in result["metadata"].get("completed_tasks", []):
            print(f"  ✓ {task}")

    if result.get("report"):
        report = result["report"]
        print(f"\n--- REPORT ---")
        print(f"Title: {report.metadata.title}")
        print(f"\nExecutive Summary:")
        print(f"  {report.executive_summary[:300]}...")

        if report.conclusions:
            print(f"\nKey Conclusions:")
            for c in report.conclusions[:3]:
                print(f"  • {c}")

        if report.recommendations:
            print(f"\nRecommendations:")
            for r in report.recommendations[:3]:
                print(f"  • {r}")

    return result


async def demo_a2a():
    """Demonstrate A2A communication."""
    print("\n" + "=" * 60)
    print("DEMO: A2A Protocol Communication")
    print("=" * 60)

    from src.a2a.client import A2ARouter
    from src.a2a.server import A2AServer

    # Create router
    router = A2ARouter()

    # Register agents
    router.register_agent("planner", "http://localhost:8080/planner", timeout=30)
    router.register_agent("researcher", "http://localhost:8080/researcher", timeout=30)
    router.register_agent("analyst", "http://localhost:8080/analyst", timeout=30)
    router.register_agent("writer", "http://localhost:8080/writer", timeout=30)

    print("\nRegistered Agents:")
    for agent_name in router.list_agents():
        print(f"  • {agent_name}")

    # Note: A2A calls require running agent servers
    print("\nNote: A2A calls require agent servers to be running.")
    print("To test A2A calls, start the agent servers first.")

    return router


async def demo_mcp():
    """Demonstrate MCP tools."""
    print("\n" + "=" * 60)
    print("DEMO: MCP Tool Integration")
    print("=" * 60)

    from src.mcp_tools.web_search import MCPWebSearchTool

    # Create MCP tool
    tool = MCPWebSearchTool()

    # Execute search
    query = "Indian EV market investment risks 2024"
    print(f"\nExecuting MCP search: {query}\n")

    response = await tool.search(query, max_results=5)

    print(f"Query: {response.query}")
    print(f"Results: {response.total_results}")
    print(f"Execution time: {response.execution_time_ms:.2f}ms")

    print("\nSearch Results:")
    for i, result in enumerate(response.results[:3], 1):
        print(f"\n  {i}. {result.title}")
        print(f"     Source: {result.source}")
        print(f"     Relevance: {result.relevance_score:.2f}")

    await tool.close()

    return response


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("RESEARCH-TO-REPORT MULTI-AGENT SYSTEM")
    print("Sample Demonstration Script")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.utcnow().isoformat()}")
    print("\n" + "-" * 60)

    try:
        # Run individual agent demo
        await demo_individual_agents()

        # Run workflow demo
        await demo_workflow()

        # Run A2A demo
        await demo_a2a()

        # Run MCP demo
        await demo_mcp()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)

    except Exception as e:
        logger.error("demo_error", error=str(e))
        print(f"\nError during demonstration: {e}")
        print("\nNote: Some demos require Azure credentials to be configured.")
        print("Check your .env file and ensure Azure OpenAI is accessible.")


if __name__ == "__main__":
    asyncio.run(main())