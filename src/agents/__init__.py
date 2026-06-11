# Agents package
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from typing import Any

# --- Task & Plan Models ---
class TaskType(str, Enum):
    WEB_SEARCH = "web_search"
    DATA_EXTRACTION = "data_extraction"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SYNTHESIS = "synthesis"
    REPORT_GENERATION = "report_generation"
    FACT_CHECK = "fact_check"

class SubTask(BaseModel):
    id: str = Field(default="", description="ID of the subtask")
    name: str = Field(default="", description="Name of the subtask")
    type: str = Field(default="", description="Type of the subtask")
    agent: str = Field(default="", description="Target agent")
    description: str = Field(default="", description="Task description")
    inputs: list[str] = Field(default_factory=list, description="List of input IDs")
    outputs: list[str] = Field(default_factory=list, description="List of output names")
    search_queries: list[str] = Field(default_factory=list, description="Search queries")
    estimated_duration: str = Field(default="", description="Estimated time")
    dependencies: list[str] = Field(default_factory=list, description="Dependencies")

    @model_validator(mode="after")
    def populate_defaults(self) -> "SubTask":
        if not self.description:
            self.description = self.name or self.id or "Research subtask"
        if not self.name:
            self.name = self.description or self.id or "Subtask"
        return self

class ResearchPlan(BaseModel):
    plan_id: str = Field(default="", description="ID of the plan")
    query: str = Field(default="", description="Original query")
    reasoning: str = Field(default="", description="Planner's reasoning")
    tasks: list[SubTask] = Field(default_factory=list, description="Tasks to run")
    estimated_total_duration: str = Field(default="", description="Estimated total duration")
    confidence_score: float = Field(default=1.0, description="Confidence score")
    output: str = Field(default="", description="Raw output text")

# --- Researcher Models ---
class SearchResult(BaseModel):
    """Represents a single search result from web research."""
    title: str = Field(default="", description="Title of the search result")
    url: str = Field(default="", description="URL of the source")
    snippet: str = Field(default="", description="Brief excerpt from the source")
    source_name: str = Field(default="", description="Name of the source website")
    published_date: str | None = Field(default=None, description="Publication date if available")
    relevance_score: float = Field(default=0.0, description="Relevance to query (0-1)")
    authority_score: float = Field(default=0.0, description="Source authority (0-1)")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Raw data from search API")

class ResearchResults(BaseModel):
    """Complete research results from a search session."""
    query: str = Field(default="", description="The search query")
    timestamp: str = Field(default="", description="When the research was conducted")
    total_sources: int = Field(default=0, description="Total number of sources found")
    high_confidence_sources: list[SearchResult] = Field(default_factory=list, description="High relevance sources")
    medium_confidence_sources: list[SearchResult] = Field(default_factory=list, description="Medium relevance sources")
    sources_used: list[str] = Field(default_factory=list, description="URLs of all sources consulted")
    search_metadata: dict[str, Any] = Field(default_factory=dict, description="Search execution metadata")
    confidence_score: float = Field(default=1.0, description="Overall confidence in results (0-1)")
    gaps_identified: list[str] = Field(default_factory=list, description="Information gaps found")

    @field_validator("sources_used", "gaps_identified", mode="before")
    @classmethod
    def convert_string_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            if "\n" in v:
                lines = [line.strip().lstrip("-*•").strip() for line in v.split("\n")]
                return [line for line in lines if line]
            return [v]
        return v

# --- Analyst Models ---
class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"

class AnalysisInsight(BaseModel):
    category: str = Field(default="", description="Insight category")
    statement: str = Field(default="", description="Insight statement")
    confidence: float = Field(default=0.0, description="Confidence score")
    evidence: str = Field(default="", description="Supporting evidence")
    evidence_strength: EvidenceStrength = Field(default=EvidenceStrength.MEDIUM, description="Strength of evidence")
    source_count: int = Field(default=0, description="Number of sources")

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskAssessment(BaseModel):
    risk_id: str = Field(default="", description="Unique risk ID")
    title: str = Field(default="", description="Risk title")
    description: str = Field(default="", description="Risk description")
    level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk level")
    probability: float = Field(default=0.0, description="Probability (0-1)")
    impact: float = Field(default=0.0, description="Impact (0-1)")
    risk_score: float = Field(default=0.0, description="Overall risk score")
    mitigation: str = Field(default="", description="Proposed mitigation strategy")
    confidence: float = Field(default=0.0, description="Mitigation confidence")

class AnalysisResults(BaseModel):
    query: str = Field(default="", description="The research query")
    key_findings: list[AnalysisInsight] = Field(default_factory=list, description="Key insights")
    risks_identified: list[RiskAssessment] = Field(default_factory=list, description="Risks identified")
    patterns_detected: list[str] = Field(default_factory=list, description="Patterns detected")
    reasoning_chain: list[str] = Field(default_factory=list, description="Reasoning steps")
    overall_confidence: float = Field(default=1.0, description="Overall confidence")
    data_sources_analyzed: list[str] = Field(default_factory=list, description="Data sources analyzed")

    @field_validator("patterns_detected", "reasoning_chain", "data_sources_analyzed", mode="before")
    @classmethod
    def convert_string_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            if "\n" in v:
                lines = [line.strip().lstrip("-*•").strip() for line in v.split("\n")]
                return [line for line in lines if line]
            return [v]
        return v

# --- Competitive Analysis Models ---
class CompetitorInsight(BaseModel):
    """A single competitive insight from the landscape analysis."""
    competitor_name: str = Field(default="", description="Name of the competitor or market player")
    insight: str = Field(default="", description="Key competitive insight")
    category: str = Field(default="general", description="Category: market_share, technology, pricing, strategy")
    confidence: float = Field(default=0.5, description="Confidence in this insight (0-1)")
    source: str = Field(default="", description="Source of the insight")

class CompetitiveAnalysisResult(BaseModel):
    """Complete competitive landscape analysis result."""
    query: str = Field(default="", description="Original analysis query")
    analysis_id: str = Field(default="", description="Unique analysis identifier")
    timestamp: str = Field(default="", description="When the analysis was performed")
    market_overview: str = Field(default="", description="High-level market overview")
    key_competitors: list[CompetitorInsight] = Field(default_factory=list, description="Key competitor insights")
    market_trends: list[str] = Field(default_factory=list, description="Identified market trends")
    strategic_recommendations: list[str] = Field(default_factory=list, description="Strategic recommendations")
    swot_analysis: dict[str, list[str]] = Field(
        default_factory=lambda: {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},
        description="SWOT analysis"
    )
    confidence_score: float = Field(default=0.5, description="Overall confidence (0-1)")
    raw_response: str = Field(default="", description="Raw response from the agent")

    @field_validator("market_trends", "strategic_recommendations", mode="before")
    @classmethod
    def convert_string_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            if "\n" in v:
                lines = [line.strip().lstrip("-*•").strip() for line in v.split("\n")]
                return [line for line in lines if line]
            return [v]
        return v

# --- Writer Models ---
class ReportFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"

class ReportMetadata(BaseModel):
    title: str = Field(default="", description="Title of the report")
    report_id: str = Field(default="", description="ID of the report")
    confidence_score: float = Field(default=1.0, description="Overall report confidence")
    processing_time_seconds: float = Field(default=0.0, description="Processing duration")
    generated_at: str = Field(default="", description="Generation timestamp")

class ReportSection(BaseModel):
    title: str = Field(default="", description="Section title")
    content: str = Field(default="", description="Section content in markdown format")

class GeneratedReport(BaseModel):
    metadata: ReportMetadata = Field(default_factory=ReportMetadata, description="Report metadata")
    executive_summary: str = Field(default="", description="Executive summary")
    sections: list[ReportSection] = Field(default_factory=list, description="Main sections")
    conclusions: list[str] = Field(default_factory=list, description="List of conclusions")
    recommendations: list[str] = Field(default_factory=list, description="List of recommendations")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="List of references / citations")

    @field_validator("conclusions", "recommendations", mode="before")
    @classmethod
    def convert_string_to_list(cls, v: Any) -> Any:
        if isinstance(v, str):
            if "\n" in v:
                lines = [line.strip().lstrip("-*•").strip() for line in v.split("\n")]
                return [line for line in lines if line]
            return [v]
        return v

# --- Dummy fallback Agent classes ---
class PlannerAgent:
    async def decompose_task(self, query: str) -> ResearchPlan:
        return ResearchPlan(query=query)
    async def validate_plan(self, plan: ResearchPlan) -> dict:
        return {"can_proceed": True, "is_valid": True}

class ResearcherAgent:
    async def search(self, query: str, max_results: int = 10) -> ResearchResults:
        return ResearchResults(query=query)

class AnalystAgent:
    async def analyze(self, query: str, research_data: dict) -> AnalysisResults:
        return AnalysisResults(query=query)

class WriterAgent:
    async def generate_report(self, query: str, analysis_results: dict) -> GeneratedReport:
        return GeneratedReport()

class CompetitiveLandscapeAgent:
    async def analyze_competitive_landscape(self, query: str, industry: str = "", region: str = "") -> CompetitiveAnalysisResult:
        return CompetitiveAnalysisResult(query=query, analysis_id="dummy", timestamp="")

__all__ = [
    "PlannerAgent",
    "ResearcherAgent",
    "AnalystAgent",
    "WriterAgent",
    "CompetitiveLandscapeAgent",
    "ResearchPlan",
    "SubTask",
    "TaskType",
    "SearchResult",
    "ResearchResults",
    "EvidenceStrength",
    "AnalysisInsight",
    "RiskLevel",
    "RiskAssessment",
    "AnalysisResults",
    "CompetitorInsight",
    "CompetitiveAnalysisResult",
    "ReportFormat",
    "ReportMetadata",
    "ReportSection",
    "GeneratedReport",
]