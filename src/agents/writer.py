"""
Writer Agent - Structured Report Generation

This agent is responsible for transforming analyzed research data into well-structured,
comprehensive reports. It handles formatting, citation management, and produces outputs
that meet professional standards.

Key capabilities:
- Structured report generation
- Multi-format output (JSON, Markdown, HTML)
- Citation and source management
- Executive summary creation
- Conclusion and recommendation synthesis

Used by: Orchestrator (via A2A protocol)
"""

from typing import Any, Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from enum import Enum
import json
import structlog

logger = structlog.get_logger(__name__)


class ReportFormat(str, Enum):
    """Available report output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    EXECUTIVE_SUMMARY = "executive_summary"


class ReportSection(BaseModel):
    """Represents a section within the report."""
    section_id: str = Field(..., description="Unique identifier for the section")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    subsections: list["ReportSection"] = Field(default_factory=list, description="Child sections")
    data: dict[str, Any] = Field(default_factory=dict, description="Structured data for section")
    sources: list[str] = Field(default_factory=list, description="Source citations")


class ReportMetadata(BaseModel):
    """Metadata about the generated report."""
    report_id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    query: str = Field(..., description="Original research query")
    created_at: str = Field(..., description="Report generation timestamp")
    version: str = Field(default="1.0", description="Report version")
    authors: list[str] = Field(default_factory=lambda: ["Research-to-Report Multi-Agent System"], description="Report authors")
    confidence_score: float = Field(..., description="Overall confidence in the report (0-1)")
    processing_time_seconds: float = Field(..., description="Time taken to generate report")
    data_sources: int = Field(..., description="Number of data sources used")
    agents_used: list[str] = Field(default_factory=list, description="Agents that contributed")


class GeneratedReport(BaseModel):
    """Complete generated report with all components."""
    metadata: ReportMetadata = Field(..., description="Report metadata")
    executive_summary: str = Field(..., description="High-level summary of findings")
    sections: list[ReportSection] = Field(..., description="All report sections")
    conclusions: list[str] = Field(..., description="Key conclusions")
    recommendations: list[str] = Field(default_factory=list, description="Actionable recommendations")
    citations: list[dict[str, Any]] = Field(..., description="All source citations")
    appendices: list[dict[str, Any]] = Field(default_factory=list, description="Supplementary materials")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Original analysis data")


class WriterAgent:
    """
    Writer Agent for structured report generation.

    This agent transforms research and analysis into professional reports:
    - Executive summaries for quick understanding
    - Structured sections with clear hierarchies
    - Proper citations and source tracking
    - Multiple output formats
    - Recommendations and conclusions
    """

    SYSTEM_PROMPT = """You are the Writer Agent in a multi-agent research system.
Your role is to transform research data and analysis into well-structured, professional reports.

For each report generation task, you must:
1. Create a compelling executive summary
2. Structure content logically with clear sections
3. Present findings in accessible language
4. Include proper citations and references
5. Synthesize conclusions and recommendations
6. Maintain consistency throughout
7. Format appropriately for the output type

Report structure guidelines:
- Executive Summary: 2-3 paragraphs, key findings, main conclusions
- Introduction: Context and scope
- Findings: Organized by theme or topic
- Analysis: Interpretations and insights
- Conclusions: Summarized takeaways
- Recommendations: Actionable next steps
- References: Full source citations

Tone and style:
- Professional and objective
- Clear and accessible
- Evidence-based
- Avoid jargon where possible
- Use active voice

Always:
- Cite sources for claims
- Distinguish facts from interpretations
- Acknowledge uncertainties
- Provide actionable recommendations
- Consider the target audience

Output should be structured JSON or Markdown as appropriate.
"""

    def __init__(self, llm: Any | None = None):
        """
        Initialize the Writer Agent.

        Args:
            llm: Optional language model. If not provided, will be loaded from config.
        """
        self.llm = llm
        self._setup_llm()

    def _setup_llm(self):
        """Set up the language model from configuration."""
        if self.llm is None:
            from ..utils.config import get_chat_model
            self.llm = get_chat_model(temperature=0.4)

    async def generate_report(
        self,
        query: str,
        analysis_results: dict[str, Any],
        output_format: ReportFormat = ReportFormat.JSON
    ) -> GeneratedReport:
        """
        Generate a comprehensive report from analysis results.

        Args:
            query: The original research query
            analysis_results: Data from the Analyst agent
            output_format: Desired output format

        Returns:
            GeneratedReport with all components
        """
        logger.info("writer_generate_report_start", query=query[:50], format=output_format.value, analysis_results_type=type(analysis_results).__name__)
        start_time = datetime.utcnow()

        if isinstance(analysis_results, str):
            try:
                analysis_results = json.loads(analysis_results)
            except Exception as e:
                logger.error("failed_to_parse_analysis_results_string", error=str(e))

        # Extract key data
        insights = analysis_results.get("key_findings", []) if isinstance(analysis_results, dict) else []
        risks = analysis_results.get("risks_identified", []) if isinstance(analysis_results, dict) else []
        patterns = analysis_results.get("patterns_detected", []) if isinstance(analysis_results, dict) else []
        reasoning_chain = analysis_results.get("reasoning_chain", []) if isinstance(analysis_results, dict) else []
        overall_confidence = analysis_results.get("overall_confidence", 0.5) if isinstance(analysis_results, dict) else 0.5
        data_sources = analysis_results.get("data_sources_analyzed", 0) if isinstance(analysis_results, dict) else 0

        # Generate report content using LLM
        report_data = await self._generate_report_content(
            query, insights, risks, patterns, reasoning_chain
        )

        # Build metadata
        metadata = ReportMetadata(
            report_id=f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            title=report_data.get("title", f"Research Report: {query[:50]}..."),
            query=query,
            created_at=datetime.utcnow().isoformat(),
            confidence_score=overall_confidence,
            processing_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            data_sources=data_sources,
            agents_used=["Planner", "Researcher", "Analyst", "Writer"]
        )

        # Build sections
        sections = self._build_sections(report_data, risks)

        # Create citations from sources
        citations = self._build_citations(analysis_results.get("sources", []))

        report = GeneratedReport(
            metadata=metadata,
            executive_summary=report_data.get("executive_summary", ""),
            sections=sections,
            conclusions=report_data.get("conclusions", []),
            recommendations=report_data.get("recommendations", []),
            citations=citations,
            appendices=report_data.get("appendices", []),
            raw_data=analysis_results
        )

        logger.info(
            "writer_generate_report_complete",
            report_id=metadata.report_id,
            sections=len(sections),
            recommendations=len(report.recommendations)
        )

        return report

    async def _generate_report_content(
        self,
        query: str,
        insights: list,
        risks: list,
        patterns: list,
        reasoning_chain: list
    ) -> dict[str, Any]:
        """Generate structured report content using LLM."""
        # Convert insights and risks to text for the prompt
        insights_text = "\n".join([
            f"- {i.get('statement', str(i))} (Confidence: {i.get('confidence', 0.5):.2f})"
            for i in insights[:5]
        ]) or "No key insights available."

        risks_text = "\n".join([
            f"- {r.get('title', str(r))}: Level={r.get('level', 'unknown')}, Score={r.get('risk_score', 0):.2f}"
            for r in risks[:5]
        ]) or "No risks identified."

        patterns_text = "\n".join([f"- {p}" for p in patterns[:5]]) or "No patterns identified."

        reasoning_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(reasoning_chain[:5])]) or "No reasoning chain available."

        prompt = f"""
Generate a comprehensive research report for the following query:

QUERY: {query}

KEY INSIGHTS:
{insights_text}

RISKS IDENTIFIED:
{risks_text}

PATTERNS DETECTED:
{patterns_text}

REASONING CHAIN:
{reasoning_text}

Create a detailed JSON report with this structure:
{{
    "title": "<professional report title>",
    "executive_summary": "<2-3 paragraph executive summary covering key findings and main conclusions>",
    "sections": [
        {{
            "section_id": "section_1",
            "title": "<section title>",
            "content": "<section content as well-structured text>",
            "data": {{}},
            "sources": []
        }}
    ],
    "conclusions": ["<conclusion 1>", "<conclusion 2>", ...],
    "recommendations": ["<recommendation 1>", "<recommendation 2>", ...],
    "appendices": []
}}

Ensure:
- Executive summary is compelling and self-contained
- Sections are logically organized (typically: Introduction, Findings, Analysis, Risks, Recommendations)
- Conclusions are supported by evidence
- Recommendations are specific and actionable
- Content is professional and well-written

IMPORTANT: Return ONLY the JSON object, no additional text.
"""

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        try:
            from ..utils.config import clean_and_parse_json
            report_data = clean_and_parse_json(response.content)
            return report_data
        except Exception as e:
            logger.error("writer_json_parse_error", error=str(e))
            return self._create_fallback_report(query)

    def _build_sections(self, report_data: dict, risks: list) -> list[ReportSection]:
        """Build structured sections from report data."""
        sections = []

        # Introduction section
        sections.append(ReportSection(
            section_id="introduction",
            title="Introduction",
            content=report_data.get("executive_summary", ""),
            data={"scope": "research_query", "approach": "multi-agent analysis"},
            sources=[]
        ))

        # Process report sections
        for i, section_data in enumerate(report_data.get("sections", [])):
            section = ReportSection(
                section_id=section_data.get("section_id", f"section_{i+1}"),
                title=section_data.get("title", f"Section {i+1}"),
                content=section_data.get("content", ""),
                subsections=[],
                data=section_data.get("data", {}),
                sources=section_data.get("sources", [])
            )
            sections.append(section)

        # Add risks section if risks are available
        if risks:
            risks_content = "\n".join([
                f"**{r.get('title', f'Risk {i+1}')}**\n"
                f"Level: {r.get('level', 'unknown').upper()}\n"
                f"Risk Score: {r.get('risk_score', 0):.2f}\n"
                f"Description: {r.get('description', 'No description available')}\n"
                f"Mitigation: {', '.join(r.get('mitigation', ['No mitigation strategies identified'])[:3])}"
                for i, r in enumerate(risks[:5])
            ])

            sections.append(ReportSection(
                section_id="risk_assessment",
                title="Risk Assessment",
                content=risks_content,
                data={"risks_count": len(risks)},
                sources=[]
            ))

        # Add conclusions section
        if report_data.get("conclusions"):
            conclusions_content = "\n".join([
                f"{i+1}. {c}" for i, c in enumerate(report_data.get("conclusions", []))
            ])
            sections.append(ReportSection(
                section_id="conclusions",
                title="Conclusions",
                content=conclusions_content,
                data={"conclusions_count": len(report_data.get("conclusions", []))},
                sources=[]
            ))

        return sections

    def _build_citations(self, sources: list) -> list[dict[str, Any]]:
        """Build formatted citations from sources."""
        citations = []

        for i, source in enumerate(sources[:10], 1):
            citations.append({
                "id": f"ref_{i}",
                "title": source.get("title", "Unknown Title"),
                "source": source.get("source_name", source.get("source", "Unknown Source")),
                "url": source.get("url", ""),
                "accessed": datetime.utcnow().isoformat(),
                "relevance": source.get("relevance_score", source.get("relevance", 0.5))
            })

        return citations

    async def format_output(
        self,
        report: GeneratedReport,
        format_type: ReportFormat
    ) -> str:
        """
        Format the report in the specified output format.

        Args:
            report: The generated report
            format_type: Desired output format

        Returns:
            Formatted report string
        """
        logger.info("writer_format_output", format_type=format_type.value)

        if format_type == ReportFormat.MARKDOWN:
            return self._format_markdown(report)
        elif format_type == ReportFormat.HTML:
            return self._format_html(report)
        elif format_type == ReportFormat.EXECUTIVE_SUMMARY:
            return self._format_executive_summary(report)
        else:
            return report.model_dump_json(indent=2)

    def _format_markdown(self, report: GeneratedReport) -> str:
        """Format report as Markdown."""
        md = []

        # Title and metadata
        md.append(f"# {report.metadata.title}\n")
        md.append(f"**Generated:** {report.metadata.created_at}\n")
        md.append(f"**Confidence Score:** {report.metadata.confidence_score:.2f}\n")
        md.append(f"**Data Sources:** {report.metadata.data_sources}\n")
        md.append("---\n")

        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"{report.executive_summary}\n")

        # Sections
        for section in report.sections:
            md.append(f"## {section.title}\n")
            md.append(f"{section.content}\n")

        # Conclusions
        if report.conclusions:
            md.append("## Conclusions\n")
            for conclusion in report.conclusions:
                md.append(f"- {conclusion}\n")

        # Recommendations
        if report.recommendations:
            md.append("\n## Recommendations\n")
            for rec in report.recommendations:
                md.append(f"- {rec}\n")

        # Citations
        if report.citations:
            md.append("\n## References\n")
            for citation in report.citations:
                md.append(f"- [{citation['title']}]({citation['url']}) - {citation['source']}\n")

        return "\n".join(md)

    def _format_html(self, report: GeneratedReport) -> str:
        """Format report as HTML."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report.metadata.title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }",
            "h2 { color: #34495e; }",
            ".metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; }",
            ".section { margin: 20px 0; }",
            ".citation { font-size: 0.9em; color: #7f8c8d; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report.metadata.title}</h1>",
            "<div class='metadata'>",
            f"<p><strong>Generated:</strong> {report.metadata.created_at}</p>",
            f"<p><strong>Confidence:</strong> {report.metadata.confidence_score:.2f}</p>",
            "</div>",
            "<h2>Executive Summary</h2>",
            f"<p>{report.executive_summary}</p>"
        ]

        for section in report.sections:
            html.append(f"<div class='section'><h2>{section.title}</h2>")
            html.append(f"<p>{section.content}</p></div>")

        if report.conclusions:
            html.append("<h2>Conclusions</h2><ul>")
            for c in report.conclusions:
                html.append(f"<li>{c}</li>")
            html.append("</ul>")

        if report.recommendations:
            html.append("<h2>Recommendations</h2><ul>")
            for r in report.recommendations:
                html.append(f"<li>{r}</li>")
            html.append("</ul>")

        html.extend(["</body>", "</html>"])

        return "\n".join(html)

    def _format_executive_summary(self, report: GeneratedReport) -> str:
        """Format as standalone executive summary."""
        summary = [
            "=" * 60,
            "EXECUTIVE SUMMARY",
            "=" * 60,
            "",
            f"Query: {report.metadata.query}",
            f"Generated: {report.metadata.created_at}",
            f"Confidence: {report.metadata.confidence_score:.2f}",
            "",
            "-" * 60,
            "",
            report.executive_summary,
            "",
            "-" * 60,
            "",
            "KEY FINDINGS:",
            ""
        ]

        for i, c in enumerate(report.conclusions[:5], 1):
            summary.append(f"{i}. {c}")

        if report.recommendations:
            summary.extend(["", "RECOMMENDATIONS:", ""])
            for i, r in enumerate(report.recommendations[:5], 1):
                summary.append(f"{i}. {r}")

        summary.extend(["", "=" * 60])

        return "\n".join(summary)

    def _create_fallback_report(self, query: str) -> dict[str, Any]:
        """Create a minimal fallback report when content generation fails."""
        return {
            "title": f"Research Report: {query[:50]}...",
            "executive_summary": "This report analyzes the research query and provides key findings based on available data.",
            "sections": [
                {
                    "section_id": "findings",
                    "title": "Key Findings",
                    "content": "The research has identified several important findings related to the query. Detailed analysis is provided in the sections below.",
                    "data": {},
                    "sources": []
                }
            ],
            "conclusions": ["Analysis completed with available data."],
            "recommendations": ["Further research may provide additional insights."],
            "appendices": []
        }


# A2A Protocol Implementation
A2A_METADATA = {
    "name": "writer",
    "description": "Report generation and formatting agent",
    "version": "1.0.0",
    "capabilities": ["generate_report", "format_output"],
    "endpoint": "/writer",
    "methods": {
        "generate_report": {
            "description": "Generate a comprehensive report from analysis results",
            "parameters": {"query": "str", "analysis_results": "dict", "output_format": "ReportFormat"},
            "returns": "GeneratedReport"
        },
        "format_output": {
            "description": "Format report in specified output type",
            "parameters": {"report": "GeneratedReport", "format_type": "ReportFormat"},
            "returns": "str"
        }
    }
}


async def main():
    """Demo function for testing the Writer Agent."""
    from ..utils.config import load_environment

    load_environment()

    writer = WriterAgent()

    # Test with sample data
    query = "What are the top 3 investment risks in the Indian EV market?"

    analysis_results = {
        "key_findings": [
            {
                "category": "regulatory",
                "statement": "Regulatory uncertainty is the primary risk factor for EV investments in India",
                "confidence": 0.85,
                "evidence": ["Government policy shifts", "Subsidy fluctuations"],
                "evidence_strength": "strong",
                "source_count": 5
            },
            {
                "category": "market",
                "statement": "Supply chain vulnerabilities present significant operational risks",
                "confidence": 0.78,
                "evidence": ["Battery component imports", "Lithium dependency"],
                "evidence_strength": "moderate",
                "source_count": 3
            }
        ],
        "risks_identified": [
            {
                "risk_id": "risk_1",
                "title": "Regulatory Policy Uncertainty",
                "description": "Frequent changes in government EV policies and subsidy programs create investment unpredictability",
                "level": "high",
                "probability": 0.7,
                "impact": 0.8,
                "risk_score": 0.56,
                "mitigation": ["Diversify market presence", "Engage with policymakers"],
                "confidence": 0.85
            },
            {
                "risk_id": "risk_2",
                "title": "Supply Chain Dependency",
                "description": "Heavy reliance on imported battery components and lithium creates supply chain vulnerability",
                "level": "high",
                "probability": 0.6,
                "impact": 0.75,
                "risk_score": 0.45,
                "mitigation": ["Develop local partnerships", "Invest in battery recycling"],
                "confidence": 0.78
            }
        ],
        "patterns_detected": ["Increasing government focus on domestic manufacturing"],
        "reasoning_chain": [
            "Identified regulatory uncertainty as primary concern",
            "Cross-referenced with policy documents and news",
            "Confirmed impact on investment decisions"
        ],
        "overall_confidence": 0.82,
        "data_sources_analyzed": 10
    }

    print("=" * 60)
    print("WRITER AGENT - REPORT GENERATION DEMO")
    print("=" * 60)
    print(f"\nQuery: {query}\n")

    # Generate report
    report = await writer.generate_report(query, analysis_results)

    print(f"Report ID: {report.metadata.report_id}")
    print(f"Title: {report.metadata.title}")
    print(f"Created: {report.metadata.created_at}")
    print(f"Confidence: {report.metadata.confidence_score:.2f}")
    print(f"Processing Time: {report.metadata.processing_time_seconds:.2f}s")

    print("\n" + "-" * 60)
    print("EXECUTIVE SUMMARY")
    print("-" * 60)
    print(report.executive_summary[:300] + "...")

    print("\n" + "-" * 60)
    print("SECTIONS")
    print("-" * 60)
    for section in report.sections:
        print(f"\n[{section.section_id}] {section.title}")
        print(f"   {section.content[:100]}...")

    print("\n" + "-" * 60)
    print("CONCLUSIONS")
    print("-" * 60)
    for i, conclusion in enumerate(report.conclusions, 1):
        print(f"{i}. {conclusion}")

    if report.recommendations:
        print("\n" + "-" * 60)
        print("RECOMMENDATIONS")
        print("-" * 60)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

    # Demonstrate formatting
    print("\n" + "-" * 60)
    print("FORMATTED OUTPUT (Markdown Preview)")
    print("-" * 60)
    markdown_output = await writer.format_output(report, ReportFormat.MARKDOWN)
    print(markdown_output[:500] + "\n...[truncated]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())