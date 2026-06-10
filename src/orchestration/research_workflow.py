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
import uuid
import os
import sqlite3
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite import SqliteStore
import structlog

from ..agents.planner import PlannerAgent, ResearchPlan, SubTask, TaskType
from ..agents.researcher import ResearcherAgent, ResearchResults
from ..agents.analyst import AnalystAgent, AnalysisResults
from ..agents.writer import WriterAgent, GeneratedReport, ReportFormat
from ..agents.competitive_analysis import CompetitiveLandscapeAgent, CompetitiveAnalysisResult
from ..foundry.client import FoundryAgentClient

logger = structlog.get_logger(__name__)


class WorkflowState(TypedDict):
    """State managed throughout the LangGraph workflow execution."""
    # Core state
    query: str
    plan: ResearchPlan | None
    research_results: ResearchResults | None
    analysis_results: AnalysisResults | None
    competitive_analysis: CompetitiveAnalysisResult | None
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
        max_retries: int = 3,
        memory_db_path: str = "memory.sqlite"
    ):
        """
        Initialize the Research Workflow.

        Args:
            enable_a2a: Enable A2A protocol for agent communication
            enable_mcp: Enable MCP tools for web search
            max_retries: Maximum number of retries for failed tasks
            memory_db_path: Path to the SQLite memory database
        """
        self.enable_a2a = enable_a2a
        self.enable_mcp = enable_mcp
        self.max_retries = max_retries

        # Initialize agents
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.competitive_analyst = CompetitiveLandscapeAgent()

        # Set up memory db paths
        self.memory_db_path = memory_db_path
        self.store_db_path = "store.sqlite"
        self.store = None

        # Initialize A2A components if enabled
        if enable_a2a:
            self._setup_a2a()

        # Graph compilation will happen at execution time for async context

        logger.info(
            "research_workflow_initialized",
            enable_a2a=enable_a2a,
            enable_mcp=enable_mcp,
            enable_competitive_analysis=self.competitive_analyst.is_available,
            max_retries=max_retries
        )

    def _setup_a2a(self):
        """Set up A2A protocol for real Azure Foundry Agents."""
        try:
            self.a2a_clients = {
                "planner": FoundryAgentClient(agent_name="planner-agent"),
                "researcher": FoundryAgentClient(agent_name="researcher-agent"),
                "analyst": FoundryAgentClient(agent_name="analyst-agent"),
                "writer": FoundryAgentClient(agent_name="writer-agent"),
            }
            logger.info("a2a_foundry_clients_initialized", agents=list(self.a2a_clients.keys()))
        except Exception as e:
            logger.warning("a2a_setup_failed", error=str(e))
            self.a2a_clients = {}

    def _build_graph(self, checkpointer, store) -> StateGraph:
        """Build the LangGraph workflow state machine."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("validate_plan", self._validate_plan_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("competitive_analysis", self._competitive_analysis_node)
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
        workflow.add_edge("analyze", "competitive_analysis")
        workflow.add_edge("competitive_analysis", "write_report")
        workflow.add_edge("write_report", END)

        # Error handling edge
        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=checkpointer, store=store)

    async def _plan_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the planning node."""
        logger.info("workflow_plan_node_start", query=state["query"][:50])

        try:
            plan = None
            
            # Check long-term memory for previous strategies
            past_memories = self.store.search(
                ("research_plans",),
                query=state["query"],
                limit=1
            )
            past_context_str = None
            if past_memories:
                logger.info("found_past_research_plan", memory_id=past_memories[0].key)
                past_context_str = json.dumps(past_memories[0].value, indent=2)

            if self.enable_a2a and self.a2a_clients and "planner" in self.a2a_clients:
                try:
                    logger.info("a2a_planner_decompose_task_start")
                    context_prompt = f"\nPAST CONTEXT FROM PREVIOUS RESEARCH:\n{past_context_str}\n(Use this to avoid redundant work and leverage known insights)\n" if past_context_str else ""
                    prompt = f"Please decompose the following research query into tasks:\n\nQuery: {state['query']}{context_prompt}\n\nRespond strictly with JSON matching the ResearchPlan schema."
                    result = await self.a2a_clients["planner"].call_agent_json(prompt)
                    if result:
                        plan = ResearchPlan(**result)
                except Exception as e:
                    logger.warning("a2a_planner_decompose_task_exception", error=str(e))

            if plan is None:
                plan = await self.planner.decompose_task(state["query"], past_context=past_context_str)

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
                prompt = f"Please validate the following research plan:\n\nPlan:\n{plan.model_dump_json(indent=2)}\n\nRespond strictly with JSON matching the validation schema (can_proceed, warnings, etc.)."
                result = await self.a2a_clients["planner"].call_agent_json(prompt)
                if result:
                    validation = result
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
                    prompt = f"Please perform research for the following query:\n\nQuery: {state['query']}\nMax Results: 10\n\nRespond strictly with JSON matching the ResearchResults schema."
                    result = await self.a2a_clients["researcher"].call_agent_json(prompt)
                    if result:
                        research_results = ResearchResults(**result)
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
                    import json
                    logger.info("a2a_analyst_analyze_start")
                    prompt = f"Please analyze the following research data for the query.\n\nQuery: {state['query']}\n\nResearch Data:\n{json.dumps(research_data, indent=2)}\n\nRespond strictly with JSON matching the AnalysisResults schema."
                    result = await self.a2a_clients["analyst"].call_agent_json(prompt)
                    if result:
                        analysis_results = AnalysisResults(**result)
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

    async def _competitive_analysis_node(self, state: WorkflowState) -> dict[str, Any]:
        """Execute the competitive landscape analysis node."""
        logger.info("workflow_competitive_analysis_start", query=state["query"][:50])

        try:
            comp_result = await self.competitive_analyst.analyze_competitive_landscape(
                query=state["query"]
            )

            # Update confidence tracking
            confidence_scores = state.get("confidence_scores", {})
            confidence_scores["competitive_analysis"] = comp_result.confidence_score

            return {
                "competitive_analysis": comp_result,
                "current_task": "competitive_analysis",
                "completed_tasks": state.get("completed_tasks", []) + ["competitive_analysis"],
                "confidence_scores": confidence_scores
            }
        except Exception as e:
            logger.warning("competitive_analysis_failed", error=str(e))
            # Non-fatal: continue without competitive analysis
            return {
                "competitive_analysis": None,
                "completed_tasks": state.get("completed_tasks", []) + ["competitive_analysis"],
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

            # Enrich with competitive landscape analysis if available
            comp_analysis = state.get("competitive_analysis")
            if comp_analysis:
                analysis_data["competitive_landscape"] = {
                    "market_overview": comp_analysis.market_overview,
                    "key_competitors": [
                        {"name": c.competitor_name, "insight": c.insight, "category": c.category}
                        for c in comp_analysis.key_competitors
                    ],
                    "market_trends": comp_analysis.market_trends,
                    "swot_analysis": comp_analysis.swot_analysis,
                    "strategic_recommendations": comp_analysis.strategic_recommendations,
                }

            report = None
            if self.enable_a2a and self.a2a_clients and "writer" in self.a2a_clients:
                try:
                    import json
                    logger.info("a2a_writer_generate_report_start")
                    prompt = f"Please write a comprehensive report based on the following analysis.\n\nQuery: {state['query']}\n\nAnalysis Data:\n{json.dumps(analysis_data, indent=2)}\n\nRespond strictly with JSON matching the GeneratedReport schema."
                    result = await self.a2a_clients["writer"].call_agent_json(prompt)
                    if result:
                        report = GeneratedReport(**result)
                except Exception as e:
                    logger.warning("a2a_writer_generate_report_exception", error=str(e))

            if report is None:
                report = await self.writer.generate_report(
                    query=state["query"],
                    analysis_results=analysis_data
                )

            # Save the final report metadata to long-term memory
            if self.store:
                self.store.put(
                    ("research_plans",),
                    str(uuid.uuid4()),
                    {
                        "query": state["query"],
                        "report_id": report.metadata.report_id,
                        "confidence_score": report.metadata.confidence_score,
                        "executive_summary": report.executive_summary,
                        "conclusions": report.conclusions,
                        "recommendations": report.recommendations
                    }
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

    async def execute(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        """
        Execute the complete research-to-report workflow.

        Args:
            query: The research query to process
            session_id: Optional session identifier for short-term memory continuity.

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
            "competitive_analysis": None,
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
            sid = session_id or str(uuid.uuid4())
            config = {"configurable": {"thread_id": sid}}
            
            with SqliteStore.from_conn_string(self.store_db_path) as store:
                store.setup()
                self.store = store
                
                async with AsyncSqliteSaver.from_conn_string(self.memory_db_path) as checkpointer:
                    await checkpointer.setup()
                    self.graph = self._build_graph(checkpointer, store)
                    final_state = await self.graph.ainvoke(initial_state, config=config)
                    
                self.store = None

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

    async def execute_streaming(self, query: str, session_id: str | None = None):
        """
        Execute workflow with streaming updates.

        Args:
            query: The research query to process
            session_id: Optional session identifier for short-term memory continuity.

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

        sid = session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": sid}}

        with SqliteStore.from_conn_string(self.store_db_path) as store:
            store.setup()
            self.store = store

            # Stream through nodes
            nodes = ["plan", "validate_plan", "research", "analyze", "competitive_analysis", "write_report"]

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

            self.store = None

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
            print(f"  - {task}")

        if result["metadata"].get("failed_tasks"):
            print("\nFailed Tasks:")
            for task in result["metadata"].get("failed_tasks", []):
                print(f"  - {task}")

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