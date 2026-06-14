"""
Research-to-Report Multi-Agent System

A production-ready multi-agent system for conducting comprehensive research-to-report
workflows using Microsoft's agent framework, LangGraph orchestration, A2A protocol
for inter-agent communication, and MCP tools for external capabilities.

Key Features:
- Intelligent task decomposition with Planner agent
- Real-time web research via MCP-connected Researcher agent
- Multi-step reasoning and insight extraction with Analyst agent
- Professional report generation with Writer agent
- Agent-to-agent communication via A2A protocol
- Production deployment on Azure AI Foundry

Usage:
    from src.orchestration.research_workflow import ResearchWorkflow

    workflow = ResearchWorkflow()
    result = await workflow.execute("What are the top 3 investment risks in the Indian EV market?")

For more information, see:
- docs/architecture.md - System architecture documentation
- docs/deployment.md - Azure AI Foundry deployment guide
- README.md - Project overview and quick start
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent Research Team"
__license__ = "MIT"

# Core components
# Agent exports
from .agents import (
    AnalysisInsight,
    AnalysisResults,
    AnalystAgent,
    GeneratedReport,
    PlannerAgent,
    ReportFormat,
    ReportSection,
    ResearcherAgent,
    ResearchPlan,
    ResearchResults,
    RiskAssessment,
    SearchResult,
    SubTask,
    TaskType,
    WriterAgent,
)

# MCP exports
from .mcp_tools.web_search import MCPDocumentSearchTool, MCPWebSearchTool
from .orchestration.research_workflow import ResearchWorkflow

# Configuration
from .utils.config import get_app_config, get_azure_openai_config, load_environment
from .utils.logging import configure_logging, get_logger

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core workflow
    "ResearchWorkflow",
    # Agents
    "PlannerAgent",
    "ResearcherAgent",
    "AnalystAgent",
    "WriterAgent",
    # Agent data types
    "ResearchPlan",
    "SubTask",
    "TaskType",
    "ResearchResults",
    "SearchResult",
    "AnalysisResults",
    "AnalysisInsight",
    "RiskAssessment",
    "GeneratedReport",
    "ReportFormat",
    "ReportSection",
    # MCP tools
    "MCPWebSearchTool",
    "MCPDocumentSearchTool",
    # Configuration
    "load_environment",
    "get_azure_openai_config",
    "get_app_config",
    "configure_logging",
    "get_logger",
]
