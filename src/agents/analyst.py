"""
Analyst Agent - Reasoning and Insight Extraction

This agent is responsible for analyzing research data, extracting insights,
identifying patterns, and making reasoned assessments. It applies structured
thinking to synthesize information and evaluate evidence.

Key capabilities:
- Data pattern recognition
- Risk assessment and evaluation
- Comparative analysis
- Evidence synthesis
- Confidence scoring
- Self-verification

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


class RiskLevel(str, Enum):
    """Risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceStrength(str, Enum):
    """Evidence strength classification."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ANECDOTAL = "anecdotal"


class AnalysisInsight(BaseModel):
    """A single insight from analysis."""
    insight_id: str = Field(..., description="Unique identifier for the insight")
    category: str = Field(..., description="Category or domain of the insight")
    statement: str = Field(..., description="The insight statement")
    confidence: float = Field(..., description="Confidence in the insight (0-1)")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")
    evidence_strength: EvidenceStrength = Field(..., description="Strength of evidence")
    caveats: list[str] = Field(default_factory=list, description="Limitations or conditions")
    source_count: int = Field(default=0, description="Number of sources supporting this")


class RiskAssessment(BaseModel):
    """Risk assessment with multiple dimensions."""
    risk_id: str = Field(..., description="Unique identifier for the risk")
    title: str = Field(..., description="Risk title")
    description: str = Field(..., description="Detailed description")
    level: RiskLevel = Field(..., description="Risk severity level")
    probability: float = Field(..., description="Probability of occurrence (0-1)")
    impact: float = Field(..., description="Impact magnitude (0-1)")
    risk_score: float = Field(..., description="Combined risk score (probability * impact)")
    factors: list[str] = Field(default_factory=list, description="Contributing factors")
    mitigation: list[str] = Field(default_factory=list, description="Potential mitigation strategies")
    evidence: list[str] = Field(default_factory=list, description="Evidence supporting assessment")
    confidence: float = Field(..., description="Confidence in assessment (0-1)")
    data_quality_issues: list[str] = Field(default_factory=list, description="Data quality concerns")


class AnalysisResults(BaseModel):
    """Complete analysis results from research data."""
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    query: str = Field(..., description="Original research query")
    timestamp: str = Field(..., description="When analysis was performed")
    key_findings: list[AnalysisInsight] = Field(..., description="Primary insights discovered")
    risks_identified: list[RiskAssessment] = Field(..., description="Risks evaluated")
    patterns_detected: list[str] = Field(default_factory=list, description="Patterns or trends identified")
    comparisons: dict[str, Any] = Field(default_factory=dict, description="Comparative analyses")
    overall_confidence: float = Field(..., description="Overall confidence in analysis (0-1)")
    reasoning_chain: list[str] = Field(default_factory=list, description="Steps in reasoning process")
    data_sources_analyzed: int = Field(default=0, description="Number of sources analyzed")
    methodology: str = Field(..., description="Analysis methodology used")
    limitations: list[str] = Field(default_factory=list, description="Analysis limitations")


class AnalystAgent:
    """
    Analyst Agent for reasoning and insight extraction.

    This agent implements structured analytical thinking:
    - Deductive reasoning from evidence
    - Inductive pattern recognition
    - Comparative analysis
    - Risk evaluation
    - Self-verification of conclusions
    """

    SYSTEM_PROMPT = """You are the Analyst Agent in a multi-agent research system.
Your role is to analyze research data, extract insights, identify patterns, and make reasoned assessments.

For each analysis task, you must:
1. Examine the evidence thoroughly
2. Identify patterns and correlations
3. Evaluate risks and their dimensions
4. Assess confidence in findings
5. Explain your reasoning chain
6. Acknowledge limitations and uncertainties
7. Verify conclusions against evidence

Analysis methodology:
- Start with data examination
- Identify key themes and patterns
- Form hypotheses based on evidence
- Test hypotheses against sources
- Assess confidence levels
- Synthesize conclusions
- Self-verify findings

Risk assessment dimensions:
- Probability: How likely is this risk?
- Impact: How severe would the consequences be?
- Velocity: How quickly could it manifest?
- Persistence: How long would effects last?
- Recoverability: How easily can we recover?

Always provide:
- Clear reasoning chains
- Evidence citations
- Confidence scores
- Acknowledgment of limitations
- Consideration of alternatives

Output format should always include structured reasoning steps.
"""

    def __init__(self, llm: Any | None = None):
        """
        Initialize the Analyst Agent.

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

    async def analyze(
        self,
        query: str,
        research_data: dict[str, Any]
    ) -> AnalysisResults:
        """
        Perform comprehensive analysis on research data.

        Args:
            query: The original research query
            research_data: Data from the Researcher agent

        Returns:
            AnalysisResults with insights, risks, and reasoning
        """
        logger.info("analyst_analyze_start", query=query[:50], research_data_type=type(research_data).__name__)

        if isinstance(research_data, str):
            try:
                research_data = json.loads(research_data)
            except Exception as e:
                logger.error("failed_to_parse_research_data_string", error=str(e))

        # Extract key information from research data
        sources = research_data.get("sources", []) if isinstance(research_data, dict) else []
        search_results = research_data.get("search_results", []) if isinstance(research_data, dict) else []

        # Generate analysis prompt
        prompt = self._create_analysis_prompt(query, sources, search_results)

        response = await self.llm.ainvoke([
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])

        try:
            from ..utils.config import clean_and_parse_json
            analysis_data = clean_and_parse_json(response.content)

            # Build structured results with robust sanitization
            key_findings = []
            for i, f in enumerate(analysis_data.get("key_findings", [])):
                if not isinstance(f, dict):
                    continue
                
                # Sanitize evidence strength enum
                raw_strength = str(f.get("evidence_strength", "moderate")).lower()
                strength = EvidenceStrength.MODERATE
                for val in EvidenceStrength:
                    if val.value == raw_strength:
                        strength = val
                        break
                        
                # Sanitize lists
                evidence = f.get("evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]
                elif not isinstance(evidence, list):
                    evidence = []
                    
                caveats = f.get("caveats", [])
                if isinstance(caveats, str):
                    caveats = [caveats]
                elif not isinstance(caveats, list):
                    caveats = []
                    
                key_findings.append(
                    AnalysisInsight(
                        insight_id=f.get("insight_id", f"insight_{i+1}"),
                        category=f.get("category", "general"),
                        statement=f.get("statement", ""),
                        confidence=float(f.get("confidence", 0.5)),
                        evidence=evidence,
                        evidence_strength=strength,
                        caveats=caveats,
                        source_count=int(f.get("source_count", 1))
                    )
                )

            risks = []
            for i, r in enumerate(analysis_data.get("risks", [])):
                if not isinstance(r, dict):
                    continue
                    
                # Sanitize level enum
                raw_level = str(r.get("level", "medium")).lower()
                level = RiskLevel.MEDIUM
                for val in RiskLevel:
                    if val.value == raw_level:
                        level = val
                        break
                
                # Sanitize lists
                factors = r.get("factors", [])
                if isinstance(factors, str):
                    factors = [factors]
                elif not isinstance(factors, list):
                    factors = []
                    
                mitigation = r.get("mitigation", [])
                if isinstance(mitigation, str):
                    mitigation = [mitigation]
                elif not isinstance(mitigation, list):
                    mitigation = []
                    
                evidence = r.get("evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]
                elif not isinstance(evidence, list):
                    evidence = []
                    
                data_quality = r.get("data_quality_issues", r.get("data_quality", []))
                if isinstance(data_quality, str):
                    data_quality = [data_quality]
                elif not isinstance(data_quality, list):
                    data_quality = []
                    
                # Convert numeric fields
                try:
                    prob = float(r.get("probability", 0.5))
                except (ValueError, TypeError):
                    prob = 0.5
                    
                try:
                    imp = float(r.get("impact", 0.5))
                except (ValueError, TypeError):
                    imp = 0.5
                    
                try:
                    score = float(r.get("risk_score", prob * imp))
                except (ValueError, TypeError):
                    score = prob * imp
                    
                try:
                    conf = float(r.get("confidence", 0.5))
                except (ValueError, TypeError):
                    conf = 0.5

                risks.append(
                    RiskAssessment(
                        risk_id=r.get("risk_id", f"risk_{i+1}"),
                        title=r.get("title", ""),
                        description=r.get("description", ""),
                        level=level,
                        probability=prob,
                        impact=imp,
                        risk_score=score,
                        factors=factors,
                        mitigation=mitigation,
                        evidence=evidence,
                        confidence=conf,
                        data_quality_issues=data_quality
                    )
                )

            results = AnalysisResults(
                analysis_id=f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                query=query,
                timestamp=datetime.utcnow().isoformat(),
                key_findings=key_findings,
                risks_identified=risks,
                patterns_detected=analysis_data.get("patterns", []),
                comparisons=analysis_data.get("comparisons", {}),
                overall_confidence=analysis_data.get("overall_confidence", 0.5),
                reasoning_chain=analysis_data.get("reasoning_chain", []),
                data_sources_analyzed=len(sources),
                methodology="Structured analytical reasoning with multi-source triangulation",
                limitations=analysis_data.get("limitations", [])
            )

            logger.info(
                "analyst_analyze_complete",
                analysis_id=results.analysis_id,
                insights=len(key_findings),
                risks=len(risks),
                confidence=results.overall_confidence
            )

            return results

        except Exception as e:
            logger.error("analyst_json_parse_error", error=str(e))
            # Return a minimal valid result
            return self._create_fallback_analysis(query)

    def _create_analysis_prompt(self, query: str, sources: list, search_results: list) -> str:
        """Create a structured prompt for analysis."""
        # Prioritize sources as it contains rich dict metadata; fallback to search_results
        items = sources if sources else search_results
        sources_text = "\n".join([
            f"- {s.get('title', 'Unknown') if isinstance(s, dict) else s}: {s.get('snippet', 'No snippet available') if isinstance(s, dict) else 'No details'} (Source: {s.get('source_name', 'Unknown') if isinstance(s, dict) else 'Unknown'})"
            for s in items[:10]
        ])

        return f"""
Perform a comprehensive analysis of the following research data for the query:
QUERY: {query}

SOURCES AND DATA:
{sources_text if sources_text else "No structured sources available. Analyze based on general knowledge."}

Create a detailed JSON analysis with this structure:
{{
    "key_findings": [
        {{
            "category": "<insight category>",
            "statement": "<the insight statement>",
            "confidence": <0.0-1.0>,
            "evidence": ["<evidence 1>", "<evidence 2>"],
            "evidence_strength": "<strong|moderate|weak|anecdotal>",
            "caveats": ["<limitation or condition>"],
            "source_count": <number of supporting sources>
        }}
    ],
    "risks": [
        {{
            "risk_id": "risk_1",
            "title": "<risk title>",
            "description": "<detailed description>",
            "level": "<critical|high|medium|low>",
            "probability": <0.0-1.0>,
            "impact": <0.0-1.0>,
            "risk_score": <probability * impact>,
            "factors": ["<contributing factor>"],
            "mitigation": ["<mitigation strategy>"],
            "evidence": ["<supporting evidence>"],
            "confidence": <0.0-1.0>,
            "data_quality_issues": ["<concern if any>"]
        }}
    ],
    "patterns": ["<identified pattern or trend>"],
    "comparisons": {{}},
    "overall_confidence": <0.0-1.0>,
    "reasoning_chain": [
        "Step 1: Initial observation...",
        "Step 2: Evidence examination...",
        "Step 3: Pattern recognition...",
        "Step 4: Conclusion..."
    ],
    "limitations": ["<analysis limitation>"]
}}

IMPORTANT: Return ONLY the JSON object, no additional text. Focus on identifying the TOP risks and insights.
"""

    async def assess_risk(self, risk_data: dict[str, Any]) -> RiskAssessment:
        """
        Perform detailed risk assessment on specific risk data.

        Args:
            risk_data: Information about the risk to assess

        Returns:
            RiskAssessment with detailed evaluation
        """
        logger.info("analyst_assess_risk_start", risk_title=risk_data.get("title", "unknown")[:50])

        prompt = f"""
Perform a detailed risk assessment for the following:

RISK: {risk_data.get('description', risk_data.get('title', 'Unknown'))}
CONTEXT: {risk_data.get('context', '')}

Evaluate:
1. Probability: How likely is this risk to occur? Consider historical data, current conditions.
2. Impact: How severe would the consequences be? Consider affected stakeholders, magnitude.
3. Velocity: How quickly could this risk manifest?
4. Persistence: How long would the effects last?
5. Uncertainty factors: What data gaps exist?

Provide a structured risk assessment with clear reasoning.

Return JSON with:
- probability (0-1)
- impact (0-1)
- risk_score (calculated)
- level (critical/high/medium/low)
- factors (contributing elements)
- mitigation (potential strategies)
- confidence (in assessment 0-1)
"""

        response = await self.llm.ainvoke([
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

        risk_json = json.loads(content)

        assessment = RiskAssessment(
            risk_id=risk_data.get("id", "manual_assessment"),
            title=risk_data.get("title", "Unknown Risk"),
            description=risk_data.get("description", ""),
            level=RiskLevel(risk_json.get("level", "medium")),
            probability=risk_json.get("probability", 0.5),
            impact=risk_json.get("impact", 0.5),
            risk_score=risk_json.get("risk_score", 0.25),
            factors=risk_json.get("factors", []),
            mitigation=risk_json.get("mitigation", []),
            evidence=risk_json.get("evidence", []),
            confidence=risk_json.get("confidence", 0.5),
            data_quality_issues=risk_json.get("data_quality_issues", [])
        )

        logger.info(
            "analyst_assess_risk_complete",
            risk_id=assessment.risk_id,
            risk_score=assessment.risk_score,
            level=assessment.level.value
        )

        return assessment

    async def verify_findings(self, findings: list[AnalysisInsight], sources: list[dict]) -> dict[str, Any]:
        """
        Self-verify findings against source data.

        Args:
            findings: Insights to verify
            sources: Source data for verification

        Returns:
            Verification results with any discrepancies
        """
        logger.info("analyst_verify_findings_start", finding_count=len(findings))

        verification_results = {
            "verified_findings": [],
            "modified_findings": [],
            "rejected_findings": [],
            "confidence_adjustments": {}
        }

        for finding in findings:
            # Check if evidence supports the confidence level
            supporting_sources = sum(
                1 for s in sources
                if any(evidence.lower() in str(s).lower() for evidence in finding.evidence)
            )

            # Adjust confidence if needed
            if supporting_sources < finding.source_count:
                new_confidence = max(0.3, finding.confidence - 0.1)
                verification_results["confidence_adjustments"][finding.insight_id] = new_confidence
                verification_results["modified_findings"].append(finding.insight_id)
            else:
                verification_results["verified_findings"].append(finding.insight_id)

        logger.info(
            "analyst_verify_findings_complete",
            verified=len(verification_results["verified_findings"]),
            modified=len(verification_results["modified_findings"]),
            rejected=len(verification_results["rejected_findings"])
        )

        return verification_results

    def _create_fallback_analysis(self, query: str) -> AnalysisResults:
        """Create a minimal fallback analysis when parsing fails."""
        return AnalysisResults(
            analysis_id=f"analysis_fallback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            query=query,
            timestamp=datetime.utcnow().isoformat(),
            key_findings=[],
            risks_identified=[],
            patterns_detected=["Unable to generate structured analysis"],
            comparisons={},
            overall_confidence=0.2,
            reasoning_chain=["Analysis generation failed - returning minimal result"],
            data_sources_analyzed=0,
            methodology="Fallback (failed)",
            limitations=["JSON parsing failed during analysis"]
        )


# A2A Protocol Implementation
A2A_METADATA = {
    "name": "analyst",
    "description": "Reasoning and insight extraction agent",
    "version": "1.0.0",
    "capabilities": ["analyze", "assess_risk", "verify_findings"],
    "endpoint": "/analyst",
    "methods": {
        "analyze": {
            "description": "Perform comprehensive analysis on research data",
            "parameters": {"query": "str", "research_data": "dict"},
            "returns": "AnalysisResults"
        },
        "assess_risk": {
            "description": "Perform detailed risk assessment",
            "parameters": {"risk_data": "dict"},
            "returns": "RiskAssessment"
        },
        "verify_findings": {
            "description": "Self-verify findings against source data",
            "parameters": {"findings": "list[AnalysisInsight]", "sources": "list[dict]"},
            "returns": "dict"
        }
    }
}


async def main():
    """Demo function for testing the Analyst Agent."""
    from ..utils.config import load_environment

    load_environment()

    analyst = AnalystAgent()

    # Test with sample query and simulated research data
    query = "What are the top 3 investment risks in the Indian EV market?"

    research_data = {
        "sources": [
            {"title": "India EV Market Report", "snippet": "Regulatory uncertainty affecting investments", "source_name": "Reuters"},
            {"title": "Electric Vehicle Analysis", "snippet": "Supply chain challenges in battery sector", "source_name": "Bloomberg"},
            {"title": "Market Trends", "snippet": "Competitive pressure from Chinese manufacturers", "source_name": "Economic Times"},
        ],
        "search_results": [
            {"title": "EV Investment Risks India", "snippet": "Key risks include regulatory, supply chain, and competitive factors", "source_name": "CNBC"},
            {"title": "Battery Manufacturing Challenges", "snippet": "Lithium imports dependency creating supply risks", "source_name": "Livemint"},
        ]
    }

    print("=" * 60)
    print("ANALYST AGENT - REASONING AND INSIGHTS DEMO")
    print("=" * 60)
    print(f"\nQuery: {query}\n")

    # Execute analysis
    results = await analyst.analyze(query, research_data)

    print(f"Analysis ID: {results.analysis_id}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Overall Confidence: {results.overall_confidence:.2f}")
    print(f"Sources Analyzed: {results.data_sources_analyzed}")

    print("\n" + "-" * 60)
    print("KEY FINDINGS")
    print("-" * 60)

    for i, finding in enumerate(results.key_findings, 1):
        print(f"\n{i}. [{finding.category}] {finding.statement}")
        print(f"   Confidence: {finding.confidence:.2f}")
        print(f"   Evidence Strength: {finding.evidence_strength.value}")
        print(f"   Sources: {finding.source_count}")

    print("\n" + "-" * 60)
    print("RISKS IDENTIFIED")
    print("-" * 60)

    for risk in results.risks_identified:
        print(f"\n[{risk.level.value.upper()}] {risk.title}")
        print(f"   Risk Score: {risk.risk_score:.2f}")
        print(f"   Probability: {risk.probability:.2f}, Impact: {risk.impact:.2f}")
        if risk.mitigation:
            print(f"   Mitigation: {risk.mitigation[0]}")

    if results.patterns_detected:
        print("\n" + "-" * 60)
        print("PATTERNS DETECTED")
        print("-" * 60)
        for pattern in results.patterns_detected:
            print(f"  - {pattern}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())