"""
Research-to-Report Multi-Agent System
Main Entry Point and Orchestration

This module provides the main interface for executing research workflows
using the multi-agent system with LangGraph orchestration.

Usage:
    from src.orchestration.research_workflow import ResearchWorkflow
    workflow = ResearchWorkflow()
    result = await workflow.execute("What are the top 3 investment risks in the Indian EV market?")
"""

from typing import Any, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END
import structlog

from ..agents.planner import PlannerAgent, ResearchPlan, SubTask, TaskType
from ..agents.researcher import ResearcherAgent, ResearchResults
from ..agents.analyst import AnalystAgent, AnalysisResults
from ..agents.writer import WriterAgent, GeneratedReport, ReportFormat
from ..a2a.client import A2AClient
from ..a2a.server import A2AServer

logger = structlog.get_logger(__name__)


class WorkflowState(TypedDict):
    """State managed throughout the LangGraph workflow execution."""
    # Core state
    query: str
    plan: ResearchPlan | None
    research_results: ResearchResults | None
    analysis_results: AnalysisResults | None
    report: GeneratedReport | None

    # Progress tracking
    current_task: str | None
    completed_tasks: list[str]
    failed_tasks: list[str]

    # Error handling
    errors: list[dict[str, Any]]
    retry_count: int
    validation_result: dict[str, Any] | None

    # Metadata
    start_time: str
    end_time: str | None
    confidence_scores: dict[str, float]


class ResearchWorkflow:
    """
    Main orchestration class for the Research-to-Report workflow.

    This class implements the LangGraph-based multi-agent workflow that:
    1. Decomposes complex queries into manageable tasks
    2. Orchestrates research, analysis, and report generation
    3. Manages state and data flow between agents
    4. Handles errors and retries
    5. Produces structured output reports
    """

    def __init__(
        self,
        enable_a2a: bool = True,
        enable_mcp: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the Research Workflow.

        Args:
            enable_a2a: Enable A2A protocol for agent communication
            enable_mcp: Enable MCP tools for web search
            max_retries: Maximum number of retries for failed tasks
        """
        self.enable_a2a = enable_a2a
        self.enable_mcp = enable_mcp
        self.max_retries = max_retries

        # Initialize agents
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()

        # Initialize A2A components if enabled
        if enable_a2a:
            self._setup_a2a()

        # Build the workflow graph
        self.graph = self._build_graph()

        logger.info(
            "research_workflow_initialized",
            enable_a2a=enable_a2a,
            enable_mcp=enable_mcp,
            max_retries=max_retries
        )

    def _setup_a2a(self):
        """Set up A2A protocol for inter-agent communication."""
        try:
            self.a2a_clients = {
                "planner": A2AClient("http://localhost:8080/planner"),
                "researcher": A2AClient("http://localhost:8080/researcher"),
                "analyst": A2AClient("http://localhost:8080/analyst"),
                "writer": A2AClient("http://localhost:8080/writer"),
            }
            logger.info("a2a_clients_initialized", agents=list(self.a2a_clients.keys()))
        except Exception as e:
            logger.warning("a2a_setup_failed", error=str(e))
            self.a2a_clients = {}

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow state machine."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("validate_plan", self._validate_plan_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("write_report", self._write_report_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "validate_plan")
        workflow.add_conditional_edges(
            "validate_plan",
            self._should_proceed,
            {
                "proceed": "research",
                "refine": "plan",
                "fail": END
            }
        )
        workflow.add_edge("research", "analyze")
        workflow.add_edge("analyze", "write_report")
        workflow.add_edge("write_report", END)

        # Error handling edge
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    async def _plan_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the planning node."""
        logger.info("workflow_plan_node_start", query=state["query"][:50])

        try:
            plan = None
            if self.enable_a2a and self.a2a_clients and "planner" in self.a2a_clients:
                try:
                    logger.info("a2a_planner_decompose_task_start")
                    response = await self.a2a_clients["planner"].call_agent(
                        method="decompose_task",
                        params={"query": state["query"]}
                    )
                    if response.status == "success" and response.result:
                        plan = ResearchPlan(**response.result)
                    else:
                        logger.warning("a2a_planner_decompose_task_failed", error=response.error)
                except Exception as e:
                    logger.warning("a2a_planner_decompose_task_exception", error=str(e))

            if plan is None:
                plan = await self.planner.decompose_task(state["query"])

            # Update confidence tracking
            confidence_scores = state.get("confidence_scores", {})
            confidence_scores["planning"] = plan.confidence_score

            # Increment retry count if we are refining an existing plan
            retry_count = state.get("retry_count", 0)
            if state.get("plan") is not None:
                retry_count += 1

            return {
                "plan": plan,
                "current_task": "planning",
                "completed_tasks": state.get("completed_tasks", []) + ["plan"],
                "errors": state.get("errors", []),
                "confidence_scores": confidence_scores,
                "retry_count": retry_count
            }
        except Exception as e:
            logger.error("planning_failed", error=str(e))
            return {
                "errors": state.get("errors", []) + [{"node": "plan", "error": str(e)}],
                "failed_tasks": state.get("failed_tasks", []) + ["plan"]
            }

    async def _validate_plan_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the plan validation node."""
        plan = state.get("plan")
        if not plan:
            return {"errors": [{"node": "validate_plan", "error": "No plan available"}]}

        logger.info("workflow_validate_plan_start", plan_id=plan.plan_id)

        validation = None
        if self.enable_a2a and self.a2a_clients and "planner" in self.a2a_clients:
            try:
                logger.info("a2a_planner_validate_plan_start")
                response = await self.a2a_clients["planner"].call_agent(
                    method="validate_plan",
                    params={"plan": plan.model_dump()}
                )
                if response.status == "success" and response.result:
                    validation = response.result
                else:
                    logger.warning("a2a_planner_validate_plan_failed", error=response.error)
            except Exception as e:
                logger.warning("a2a_planner_validate_plan_exception", error=str(e))

        if validation is None:
            validation = await self.planner.validate_plan(plan)

        return {
            "current_task": "validation",
            "confidence_scores": state.get("confidence_scores", {}),
            "validation_result": validation
        }

    def _should_proceed(self, state: WorkflowState) -> str:
        """Determine if workflow should proceed based on validation."""
        validation = state.get("validation_result", {})

        if not state.get("plan"):
            return "fail"

        if validation.get("can_proceed", False):
            return "proceed"
        elif validation.get("warnings", []) and len(validation.get("warnings", [])) < 3:
            return "proceed"  # Proceed with warnings
        elif state.get("retry_count", 0) < self.max_retries:
            return "refine"
        else:
            return "fail"

    async def _research_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the research node."""
        plan = state.get("plan")
        if not plan:
            return {"errors": [{"node": "research", "error": "No plan available"}]}

        logger.info("workflow_research_node_start", plan_id=plan.plan_id)

        try:
            # Extract search queries from plan tasks
            search_tasks = [
                task for task in plan.tasks
                if task.task_type == TaskType.WEB_SEARCH
            ]

            research_results = None
            if self.enable_a2a and self.a2a_clients and "researcher" in self.a2a_clients:
                try:
                    logger.info("a2a_researcher_search_start")
                    response = await self.a2a_clients["researcher"].call_agent(
                        method="search",
                        params={"query": state["query"], "max_results": 10}
                    )
                    if response.status == "success" and response.result:
                        research_results = ResearchResults(**response.result)
                    else:
                        logger.warning("a2a_researcher_search_failed", error=response.error)
                except Exception as e:
                    logger.warning("a2a_researcher_search_exception", error=str(e))

            if research_results is None:
                if search_tasks:
                    # Use the first search task's queries
                    search_queries = search_tasks[0].search_queries
                    if not search_queries:
                        search_queries = [state["query"]]

                    # Execute batch search
                    research_results = await self.researcher.search(
                        query=state["query"],
                        max_results=10
                    )
                else:
                    # Fallback to basic search
                    research_results = await self.researcher.search(
                        query=state["query"],
                        max_results=10
                    )

            # Update confidence tracking
            confidence_scores = state.get("confidence_scores", {})
            confidence_scores["research"] = research_results.confidence_score

            return {
                "research_results": research_results,
                "current_task": "research",
                "completed_tasks": state.get("completed_tasks", []) + ["research"],
                "confidence_scores": confidence_scores
            }
        except Exception as e:
            logger.error("research_failed", error=str(e))
            return {
                "errors": state.get("errors", []) + [{"node": "research", "error": str(e)}],
                "failed_tasks": state.get("failed_tasks", []) + ["research"]
            }

    async def _analyze_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the analysis node."""
        research_results = state.get("research_results")
        if not research_results:
            return {"errors": [{"node": "analyze", "error": "No research results available"}]}

        logger.info("workflow_analyze_node_start", query=state["query"][:50])

        try:
            # Convert research results to analysis input format
            research_data = {
                "sources": [
                    {
                        "title": r.title,
                        "snippet": r.snippet,
                        "source_name": r.source_name,
                        "url": r.url
                    }
                    for r in (research_results.high_confidence_sources +
                              research_results.medium_confidence_sources)
                ],
                "search_results": research_results.sources_used
            }

            analysis_results = None
            if self.enable_a2a and self.a2a_clients and "analyst" in self.a2a_clients:
                try:
                    logger.info("a2a_analyst_analyze_start")
                    response = await self.a2a_clients["analyst"].call_agent(
                        method="analyze",
                        params={"query": state["query"], "research_data": research_data}
                    )
                    if response.status == "success" and response.result:
                        analysis_results = AnalysisResults(**response.result)
                    else:
                        logger.warning("a2a_analyst_analyze_failed", error=response.error)
                except Exception as e:
                    logger.warning("a2a_analyst_analyze_exception", error=str(e))

            if analysis_results is None:
                analysis_results = await self.analyst.analyze(
                    query=state["query"],
                    research_data=research_data
                )

            # Update confidence tracking
            confidence_scores = state.get("confidence_scores", {})
            confidence_scores["analysis"] = analysis_results.overall_confidence

            return {
                "analysis_results": analysis_results,
                "current_task": "analysis",
                "completed_tasks": state.get("completed_tasks", []) + ["analyze"],
                "confidence_scores": confidence_scores
            }
        except Exception as e:
            logger.error("analysis_failed", error=str(e))
            return {
                "errors": state.get("errors", []) + [{"node": "analyze", "error": str(e)}],
                "failed_tasks": state.get("failed_tasks", []) + ["analyze"]
            }

    async def _write_report_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the report writing node."""
        analysis_results = state.get("analysis_results")
        if not analysis_results:
            return {"errors": [{"node": "write_report", "error": "No analysis results available"}]}

        logger.info("workflow_write_report_start", query=state["query"][:50])

        try:
            # Convert analysis results to report input format
            analysis_data = {
                "key_findings": [
                    {
                        "category": f.category,
                        "statement": f.statement,
                        "confidence": f.confidence,
                        "evidence": f.evidence,
                        "evidence_strength": f.evidence_strength.value,
                        "source_count": f.source_count
                    }
                    for f in analysis_results.key_findings
                ],
                "risks_identified": [
                    {
                        "risk_id": r.risk_id,
                        "title": r.title,
                        "description": r.description,
                        "level": r.level.value,
                        "probability": r.probability,
                        "impact": r.impact,
                        "risk_score": r.risk_score,
                        "mitigation": r.mitigation,
                        "confidence": r.confidence
                    }
                    for r in analysis_results.risks_identified
                ],
                "patterns_detected": analysis_results.patterns_detected,
                "reasoning_chain": analysis_results.reasoning_chain,
                "overall_confidence": analysis_results.overall_confidence,
                "data_sources_analyzed": analysis_results.data_sources_analyzed
            }

            # Add sources from research
            research_results = state.get("research_results")
            if research_results:
                analysis_data["sources"] = [
                    {"title": r.title, "source_name": r.source_name, "url": r.url}
                    for r in (research_results.high_confidence_sources +
                              research_results.medium_confidence_sources)
                ]

            report = None
            if self.enable_a2a and self.a2a_clients and "writer" in self.a2a_clients:
                try:
                    logger.info("a2a_writer_generate_report_start")
                    response = await self.a2a_clients["writer"].call_agent(
                        method="generate_report",
                        params={"query": state["query"], "analysis_results": analysis_data}
                    )
                    if response.status == "success" and response.result:
                        report = GeneratedReport(**response.result)
                    else:
                        logger.warning("a2a_writer_generate_report_failed", error=response.error)
                except Exception as e:
                    logger.warning("a2a_writer_generate_report_exception", error=str(e))

            if report is None:
                report = await self.writer.generate_report(
                    query=state["query"],
                    analysis_results=analysis_data
                )

            return {
                "report": report,
                "current_task": "report_generation",
                "completed_tasks": state.get("completed_tasks", []) + ["write_report"],
                "end_time": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("report_generation_failed", error=str(e))
            return {
                "errors": state.get("errors", []) + [{"node": "write_report", "error": str(e)}],
                "failed_tasks": state.get("failed_tasks", []) + ["write_report"]
            }

    async def _handle_error_node(self, state: WorkflowState) -> dict[str, Any]:
        """Handle workflow errors."""
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)

        logger.error("workflow_error_handling", errors=errors, retry_count=retry_count)

        if retry_count < self.max_retries:
            return {
                "retry_count": retry_count + 1,
                "current_task": "retry"
            }

        return {"end_time": datetime.utcnow().isoformat()}

    async def execute(self, query: str) -> dict[str, Any]:
        """
        Execute the complete research-to-report workflow.

        Args:
            query: The research query to process

        Returns:
            Dictionary containing the final report and workflow metadata
        """
        logger.info("workflow_execute_start", query=query[:100])

        # Initialize state
        initial_state: WorkflowState = {
            "query": query,
            "plan": None,
            "research_results": None,
            "analysis_results": None,
            "report": None,
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "errors": [],
            "retry_count": 0,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "confidence_scores": {}
        }

        # Execute the workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)

            # Build response
            response = {
                "status": "completed" if not final_state.get("errors") else "completed_with_errors",
                "query": query,
                "report": final_state.get("report"),
                "metadata": {
                    "start_time": final_state.get("start_time"),
                    "end_time": final_state.get("end_time"),
                    "completed_tasks": final_state.get("completed_tasks", []),
                    "failed_tasks": final_state.get("failed_tasks", []),
                    "confidence_scores": final_state.get("confidence_scores", {}),
                    "errors": final_state.get("errors", [])
                }
            }

            # Calculate overall metrics
            if final_state.get("report"):
                response["confidence_score"] = final_state["report"].metadata.confidence_score
                response["processing_time_seconds"] = final_state["report"].metadata.processing_time_seconds

            logger.info(
                "workflow_execute_complete",
                status=response["status"],
                completed_tasks=len(final_state.get("completed_tasks", [])),
                confidence=response.get("confidence_score", 0)
            )

            return response

        except Exception as e:
            logger.error("workflow_execute_failed", error=str(e))
            return {
                "status": "failed",
                "query": query,
                "error": str(e),
                "start_time": initial_state["start_time"],
                "end_time": datetime.utcnow().isoformat()
            }

    async def execute_streaming(self, query: str):
        """
        Execute workflow with streaming updates.

        Args:
            query: The research query to process

        Yields:
            Status updates as the workflow progresses
        """
        logger.info("workflow_execute_streaming_start", query=query[:50])

        initial_state: WorkflowState = {
            "query": query,
            "plan": None,
            "research_results": None,
            "analysis_results": None,
            "report": None,
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "errors": [],
            "retry_count": 0,
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "confidence_scores": {}
        }

        # Stream through nodes
        nodes = ["plan", "validate_plan", "research", "analyze", "write_report"]

        for node in nodes:
            yield {"stage": node, "status": "in_progress"}

            if node == "plan":
                state = await self._plan_node(initial_state)
            elif node == "validate_plan":
                state = await self._validate_plan_node(state)
            elif node == "research":
                state = await self._research_node(state)
            elif node == "analyze":
                state = await self._analyze_node(state)
            elif node == "write_report":
                state = await self._write_report_node(state)

            if state.get("errors"):
                yield {"stage": node, "status": "error", "errors": state["errors"]}
                break

            yield {"stage": node, "status": "completed", "state": state}

        yield {"stage": "complete", "status": "completed", "result": state}


async def main():
    """Demo function for testing the Research Workflow."""
    from ..utils.config import load_environment

    load_environment()

    print("=" * 60)
    print("RESEARCH-TO-REPORT MULTI-AGENT SYSTEM")
    print("=" * 60)
    print("\nInitializing workflow...\n")

    # Initialize workflow
    workflow = ResearchWorkflow(
        enable_a2a=True,
        enable_mcp=True,
        max_retries=3
    )

    # Sample query
    query = "What are the top 3 investment risks in the Indian EV market?"

    print(f"Query: {query}")
    print("\nExecuting workflow...\n")

    # Execute workflow
    result = await workflow.execute(query)

    # Display results
    print("\n" + "=" * 60)
    print("WORKFLOW RESULTS")
    print("=" * 60)

    print(f"\nStatus: {result['status']}")
    print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
    print(f"Processing Time: {result.get('processing_time_seconds', 'N/A')}s")

    if result.get("metadata"):
        print("\nCompleted Tasks:")
        for task in result["metadata"].get("completed_tasks", []):
            print(f"  ✓ {task}")

        if result["metadata"].get("failed_tasks"):
            print("\nFailed Tasks:")
            for task in result["metadata"].get("failed_tasks", []):
                print(f"  ✗ {task}")

    if result.get("report"):
        report = result["report"]
        print("\n" + "-" * 60)
        print("GENERATED REPORT")
        print("-" * 60)
        print(f"\nTitle: {report.metadata.title}")
        print(f"Report ID: {report.metadata.report_id}")
        print(f"\nExecutive Summary:\n{report.executive_summary[:500]}...")

        if report.conclusions:
            print("\nKey Conclusions:")
            for i, c in enumerate(report.conclusions[:3], 1):
                print(f"  {i}. {c}")

        if report.recommendations:
            print("\nRecommendations:")
            for i, r in enumerate(report.recommendations[:3], 1):
                print(f"  {i}. {r}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())