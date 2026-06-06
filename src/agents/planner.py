"""
Planner Agent - Task Decomposition Engine

This agent is responsible for breaking down complex research queries into manageable subtasks.
It analyzes the user's query, identifies required research components, and creates a structured
execution plan that guides the other agents through the research workflow.

Key capabilities:
- Query intent analysis
- Task decomposition into subtasks
- Tool and resource identification
- Execution plan generation
- Plan validation and refinement

Used by: Orchestrator (via A2A protocol)
"""

from typing import Any, Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from enum import Enum
import json
import structlog

logger = structlog.get_logger(__name__)


class TaskType(str, Enum):
    """Types of tasks that can be decomposed from a query."""
    WEB_SEARCH = "web_search"
    DOCUMENT_ANALYSIS = "document_analysis"
    DATA_EXTRACTION = "data_extraction"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SYNTHESIS = "synthesis"
    FACT_CHECK = "fact_check"
    REPORT_GENERATION = "report_generation"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SubTask(BaseModel):
    """Represents a single subtask in the research plan."""
    task_id: str = Field(..., description="Unique identifier for the task")
    task_type: TaskType = Field(..., description="Type of task")
    description: str = Field(..., description="Detailed description of what needs to be done")
    priority: TaskPriority = Field(..., description="Task priority level")
    depends_on: list[str] = Field(default_factory=list, description="Task IDs this task depends on")
    estimated_duration_seconds: int = Field(default=30, description="Estimated time to complete")
    agent: str = Field(..., description="Which agent should handle this task")
    output_format: str = Field(default="json", description="Expected output format")
    search_queries: list[str] = Field(default_factory=list, description="Search queries needed")
    key_aspects: list[str] = Field(default_factory=list, description="Key aspects to investigate")


class ResearchPlan(BaseModel):
    """Complete research plan with all subtasks."""
    plan_id: str = Field(..., description="Unique identifier for this plan")
    original_query: str = Field(..., description="The original user query")
    intent_summary: str = Field(..., description="Summary of the intended outcome")
    total_tasks: int = Field(..., description="Total number of subtasks")
    estimated_total_time_seconds: int = Field(..., description="Estimated total execution time")
    tasks: list[SubTask] = Field(..., description="List of all subtasks")
    execution_order: list[str] = Field(..., description="Recommended execution order (task IDs)")
    required_tools: list[str] = Field(..., description="Tools needed for execution")
    confidence_score: float = Field(..., description="Confidence in plan quality (0-1)")
    reasoning: str = Field(..., description="Explanation of how the plan was derived")


class PlannerAgent:
    """
    Planner Agent for intelligent task decomposition.

    This agent analyzes complex research queries and breaks them down into
    structured, executable subtasks with clear dependencies and resource requirements.
    """

    SYSTEM_PROMPT = """You are the Planner Agent in a multi-agent research system.
Your role is to analyze user queries and decompose them into structured, executable subtasks.

For each query, you must:
1. Understand the user's intent and desired outcome
2. Identify all information needs and research components
3. Break down complex tasks into manageable subtasks
4. Define clear dependencies between tasks
5. Assign appropriate agents to each task
6. Estimate resource requirements and duration

Output format:
- Use structured JSON for all outputs
- Include clear reasoning for each decomposition decision
- Define task dependencies to ensure correct execution order
- Assign specific agents (Planner, Researcher, Analyst, Writer) to tasks
- Include specific search queries for research tasks

Task type definitions:
- web_search: Find current information from the internet
- document_analysis: Analyze existing documents or data
- data_extraction: Extract specific data points from sources
- comparative_analysis: Compare multiple options or perspectives
- risk_assessment: Evaluate risks and their impacts
- synthesis: Combine multiple sources into coherent understanding
- fact_check: Verify claims against reliable sources
- report_generation: Create structured output reports

Always ensure the plan is:
- Feasible: All tasks can be executed with available tools
- Complete: All information needs are addressed
- Efficient: Minimize redundant work
- Clear: Each task has well-defined inputs and outputs
"""

    def __init__(self, llm: Any | None = None):
        """
        Initialize the Planner Agent.

        Args:
            llm: Optional language model. If not provided, will be loaded from config.
        """
        self.llm = llm
        self._setup_llm()

    def _setup_llm(self):
        """Set up the language model from configuration."""
        if self.llm is None:
            from ..utils.config import get_chat_model
            self.llm = get_chat_model(temperature=0.3)
        self.chain = self.llm

    async def decompose_task(self, query: str) -> ResearchPlan:
        """
        Decompose a complex query into a structured research plan.

        Args:
            query: The user's research query

        Returns:
            ResearchPlan with all subtasks and execution details
        """
        logger.info("planner_decompose_task_start", query=query[:100])

        prompt = f"""
Analyze the following research query and create a detailed decomposition plan:

QUERY: {query}

Your response must be a valid JSON object with this structure:
{{
    "plan_id": "plan_001",
    "original_query": "<the original query>",
    "intent_summary": "<2-3 sentence summary of what the user wants to achieve>",
    "total_tasks": <number of subtasks>,
    "estimated_total_time_seconds": <estimated total time>,
    "tasks": [
        {{
            "task_id": "task_001",
            "task_type": "<one of: web_search, document_analysis, data_extraction, comparative_analysis, risk_assessment, synthesis, fact_check, report_generation>",
            "description": "<detailed description of what this task accomplishes>",
            "priority": "<critical|high|medium|low>",
            "depends_on": ["<list of task_ids this depends on>"],
            "estimated_duration_seconds": <time in seconds>,
            "agent": "<Planner|Researcher|Analyst|Writer>",
            "output_format": "<json|text|structured>",
            "search_queries": ["<specific search queries if applicable>"],
            "key_aspects": ["<key things to investigate or find>"]
        }}
    ],
    "execution_order": ["task_001", "task_002", ...],
    "required_tools": ["<list of tools needed>"],
    "confidence_score": <0.0-1.0>,
    "reasoning": "<explanation of how the plan was derived>"
}}

Consider:
1. What information does the user need?
2. What data sources should be consulted?
3. What analysis is required?
4. What is the logical flow from query to answer?
5. Are there dependencies between tasks?
6. What could go wrong and how to mitigate?

IMPORTANT: Return ONLY the JSON object, no additional text or explanation.
"""

        response = await self.chain.ainvoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        try:
            # Parse the response to extract the plan using clean_and_parse_json
            from ..utils.config import clean_and_parse_json
            raw_data = clean_and_parse_json(response.content)
            
            # Adapt the parsed data to match the expected schema if necessary
            plan_data = self._adapt_json_to_plan(raw_data, query)

            # Convert task dictionaries to SubTask objects
            tasks = [SubTask(**task) for task in plan_data["tasks"]]

            plan = ResearchPlan(
                plan_id=plan_data["plan_id"],
                original_query=plan_data["original_query"],
                intent_summary=plan_data["intent_summary"],
                total_tasks=plan_data["total_tasks"],
                estimated_total_time_seconds=plan_data["estimated_total_time_seconds"],
                tasks=tasks,
                execution_order=plan_data["execution_order"],
                required_tools=plan_data["required_tools"],
                confidence_score=plan_data["confidence_score"],
                reasoning=plan_data["reasoning"]
            )

            logger.info(
                "planner_decompose_task_complete",
                plan_id=plan.plan_id,
                task_count=plan.total_tasks,
                confidence=plan.confidence_score
            )

            return plan

        except Exception as e:
            logger.error("planner_json_parse_error", error=str(e), response=response.content[:200])
            raise ValueError(f"Failed to parse planner response as JSON: {e}")

    def _adapt_json_to_plan(self, data: Any, original_query: str) -> dict[str, Any]:
        """Adapt a custom or loose JSON planner response to match the ResearchPlan schema."""
        from datetime import datetime
        
        # 1. If it already has the exact expected keys and format
        if isinstance(data, dict) and "tasks" in data and "plan_id" in data:
            tasks_data = data["tasks"]
            if isinstance(tasks_data, list):
                # Ensure each task has task_id and task_type
                all_good = True
                for t in tasks_data:
                    if not isinstance(t, dict):
                        all_good = False
                        break
                    # Fix minor typos
                    if "task__id" in t and "task_id" not in t:
                        t["task_id"] = t["task__id"]
                    if "task_id" not in t:
                        all_good = False
                if all_good:
                    return data

        # 2. Otherwise, adapt from decompositionPlan or generic list
        raw_tasks = []
        if isinstance(data, dict):
            if "decompositionPlan" in data:
                raw_tasks = data["decompositionPlan"]
            elif "tasks" in data:
                raw_tasks = data["tasks"]
            else:
                # Find any list
                for k, v in data.items():
                    if isinstance(v, list):
                        raw_tasks = v
                        break
        elif isinstance(data, list):
            raw_tasks = data

        if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
            # Create a minimal fallback plan if no tasks list is found
            raw_tasks = [
                {
                    "taskType": "web_search",
                    "agent": "Researcher",
                    "description": "Perform web search to gather information",
                    "specificQueries": [original_query]
                },
                {
                    "taskType": "report_generation",
                    "agent": "Writer",
                    "description": "Generate final report",
                    "dependencies": ["web_search"]
                }
            ]

        tasks = []
        execution_order = []
        task_name_to_id = {}

        # First pass to build name mapping for dependency resolution
        for idx, t in enumerate(raw_tasks):
            if not isinstance(t, dict):
                continue
            
            # Extract task ID or name
            t_id = t.get("task_id", t.get("task__id", t.get("taskType", t.get("task_type", f"task_{idx+1}"))))
            t_id_clean = str(t_id).lower().replace(" ", "_").strip()
            
            task_id = f"task_{idx+1:03d}"
            
            task_name_to_id[t_id_clean] = task_id
            task_name_to_id[str(t.get("agent", "")).lower()] = task_id
            task_name_to_id[task_id] = task_id
            
            # Match keywords in description
            desc = str(t.get("description", "")).lower()
            for kw in ["search", "analyze", "extract", "compare", "assess", "synthesis", "report", "fact_check"]:
                if kw in desc:
                    task_name_to_id[kw] = task_id

        # Second pass to build tasks
        for idx, t in enumerate(raw_tasks):
            if not isinstance(t, dict):
                continue
            task_id = f"task_{idx+1:03d}"
            execution_order.append(task_id)

            # Map task type
            raw_type = t.get("taskType", t.get("task_type", "web_search"))
            mapped_type = "web_search"
            raw_type_lower = str(raw_type).lower()
            
            for enum_type in ["web_search", "document_analysis", "data_extraction", "comparative_analysis", "risk_assessment", "synthesis", "fact_check", "report_generation"]:
                if enum_type in raw_type_lower or raw_type_lower in enum_type:
                    mapped_type = enum_type
                    break

            # Map dependencies
            depends_on = []
            raw_deps = t.get("dependencies", t.get("depends_on", []))
            if isinstance(raw_deps, list):
                for dep in raw_deps:
                    dep_str = str(dep).lower().strip()
                    if dep_str in task_name_to_id:
                        depends_on.append(task_name_to_id[dep_str])
                    elif dep_str.replace("task_", "").isdigit():
                        depends_on.append(f"task_{int(dep_str.replace('task_', '')):03d}")

            # Duration mapping
            duration_val = t.get("estimated_duration_seconds", t.get("duration", 30))
            if isinstance(duration_val, str):
                if duration_val.isdigit():
                    duration = int(duration_val)
                elif "day" in duration_val:
                    duration = 120
                else:
                    duration = 30
            elif isinstance(duration_val, (int, float)):
                duration = int(duration_val)
            else:
                duration = 30

            subtask = {
                "task_id": task_id,
                "task_type": mapped_type,
                "description": t.get("description", f"Execute {mapped_type} task"),
                "priority": str(t.get("priority", "medium")).lower(),
                "depends_on": depends_on,
                "estimated_duration_seconds": duration,
                "agent": t.get("agent", "Researcher"),
                "output_format": t.get("output_format", "json"),
                "search_queries": t.get("specificQueries", t.get("search_queries", [original_query])),
                "key_aspects": t.get("key_aspects", [])
            }
            tasks.append(subtask)

        return {
            "plan_id": data.get("plan_id", f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}" if isinstance(data, dict) else "plan_001") if isinstance(data, dict) else "plan_001",
            "original_query": original_query,
            "intent_summary": data.get("intent_summary", f"Research plan for: {original_query}") if isinstance(data, dict) else f"Research plan for: {original_query}",
            "total_tasks": len(tasks),
            "estimated_total_time_seconds": sum(tk["estimated_duration_seconds"] for tk in tasks),
            "tasks": tasks,
            "execution_order": execution_order,
            "required_tools": data.get("required_tools", ["mcp_web_search"]) if isinstance(data, dict) else ["mcp_web_search"],
            "confidence_score": data.get("confidence_score", 0.8) if isinstance(data, dict) else 0.8,
            "reasoning": data.get("reasoning", "Adapted from model decomposition plan") if isinstance(data, dict) else "Adapted from model decomposition plan"
        }

    async def validate_plan(self, plan: ResearchPlan) -> dict[str, Any]:
        """
        Validate a research plan for completeness and feasibility.

        Args:
            plan: The research plan to validate

        Returns:
            Validation results with any issues found
        """
        logger.info("planner_validate_plan_start", plan_id=plan.plan_id)

        issues = []
        warnings = []

        # Check for circular dependencies
        task_ids = {task.task_id for task in plan.tasks}
        for task in plan.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    issues.append(f"Task {task.task_id} depends on non-existent task {dep}")

        # Check for missing dependencies
        for task in plan.tasks:
            for dep in task.depends_on:
                if dep == task.task_id:
                    issues.append(f"Task {task.task_id} depends on itself")

        # Check execution order validity
        executed = set()
        for task_id in plan.execution_order:
            task = next((t for t in plan.tasks if t.task_id == task_id), None)
            if task:
                for dep in task.depends_on:
                    if dep not in executed and dep in task_ids:
                        warnings.append(f"Task {task_id} executes before its dependency {dep}")
            executed.add(task_id)

        # Check plan completeness
        if plan.total_tasks != len(plan.tasks):
            issues.append(f"Total tasks mismatch: declared {plan.total_tasks}, actual {len(plan.tasks)}")

        # Check for report generation task
        has_report_task = any(t.task_type == TaskType.REPORT_GENERATION for t in plan.tasks)
        if not has_report_task:
            warnings.append("No report generation task found - final output may be incomplete")

        is_valid = len(issues) == 0

        result = {
            "plan_id": plan.plan_id,
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "confidence_score": plan.confidence_score,
            "can_proceed": is_valid and len(warnings) < 3
        }

        logger.info(
            "planner_validate_plan_complete",
            plan_id=plan.plan_id,
            is_valid=is_valid,
            issue_count=len(issues),
            warning_count=len(warnings)
        )

        return result

    async def refine_plan(self, plan: ResearchPlan, feedback: str) -> ResearchPlan:
        """
        Refine an existing plan based on feedback.

        Args:
            plan: The current research plan
            feedback: Feedback or issues to address

        Returns:
            Refined research plan
        """
        logger.info("planner_refine_plan_start", plan_id=plan.plan_id)

        prompt = f"""
The following research plan needs refinement based on this feedback:

ORIGINAL QUERY: {plan.original_query}

CURRENT PLAN:
{json.dumps(plan.model_dump(), indent=2)}

FEEDBACK:
{feedback}

Create a refined plan that addresses the feedback while maintaining the original intent.
Return a complete JSON plan object following the same structure as before.
"""

        response = await self.chain.ainvoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        plan_data = json.loads(content.strip())
        tasks = [SubTask(**task) for task in plan_data["tasks"]]

        refined_plan = ResearchPlan(
            plan_id=f"{plan.plan_id}_refined",
            original_query=plan_data["original_query"],
            intent_summary=plan_data["intent_summary"],
            total_tasks=plan_data["total_tasks"],
            estimated_total_time_seconds=plan_data["estimated_total_time_seconds"],
            tasks=tasks,
            execution_order=plan_data["execution_order"],
            required_tools=plan_data["required_tools"],
            confidence_score=plan_data["confidence_score"],
            reasoning=plan_data["reasoning"]
        )

        logger.info("planner_refine_plan_complete", plan_id=refined_plan.plan_id)

        return refined_plan


# A2A Protocol Implementation
A2A_METADATA = {
    "name": "planner",
    "description": "Task decomposition and research planning agent",
    "version": "1.0.0",
    "capabilities": ["decompose_task", "validate_plan", "refine_plan"],
    "endpoint": "/planner",
    "methods": {
        "decompose_task": {
            "description": "Decompose a complex query into subtasks",
            "parameters": {"query": "str"},
            "returns": "ResearchPlan"
        },
        "validate_plan": {
            "description": "Validate a research plan",
            "parameters": {"plan": "ResearchPlan"},
            "returns": "dict"
        },
        "refine_plan": {
            "description": "Refine an existing plan",
            "parameters": {"plan": "ResearchPlan", "feedback": "str"},
            "returns": "ResearchPlan"
        }
    }
}


async def main():
    """Demo function for testing the Planner Agent."""
    from ..utils.config import load_environment

    load_environment()

    planner = PlannerAgent()

    # Test with sample query
    query = "What are the top 3 investment risks in the Indian EV market?"

    print("=" * 60)
    print("PLANNER AGENT - TASK DECOMPOSITION DEMO")
    print("=" * 60)
    print(f"\nOriginal Query: {query}\n")

    # Decompose the task
    plan = await planner.decompose_task(query)

    print(f"Plan ID: {plan.plan_id}")
    print(f"Intent: {plan.intent_summary}")
    print(f"Total Tasks: {plan.total_tasks}")
    print(f"Estimated Time: {plan.estimated_total_time_seconds}s")
    print(f"Confidence: {plan.confidence_score:.2f}")
    print("\nTasks:")
    for task in plan.tasks:
        print(f"  [{task.task_id}] {task.task_type.value} - {task.description[:60]}...")
        print(f"          Agent: {task.agent}, Priority: {task.priority.value}")

    print(f"\nExecution Order: {' -> '.join(plan.execution_order)}")
    print(f"\nReasoning: {plan.reasoning}")

    # Validate the plan
    print("\n" + "-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)

    validation = await planner.validate_plan(plan)
    print(f"Valid: {validation['is_valid']}")
    print(f"Can Proceed: {validation['can_proceed']}")
    print(f"Issues: {len(validation['issues'])}")
    print(f"Warnings: {len(validation['warnings'])}")

    if validation['issues']:
        print("\nIssues:")
        for issue in validation['issues']:
            print(f"  - {issue}")

    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())