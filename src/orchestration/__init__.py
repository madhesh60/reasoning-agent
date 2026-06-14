# Orchestration package
from .research_workflow import ResearchWorkflow

# WorkflowState is no longer used but kept for backward compatibility
WorkflowState = dict

__all__ = ["ResearchWorkflow", "WorkflowState"]
