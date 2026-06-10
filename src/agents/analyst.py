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
from pydantic import BaseModel, Field, field_validator
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
    evidence_strength: EvidenceStrength = Field(default=EvidenceStrength.MODERATE, description="Strength of evidence")
    caveats: list[str] = Field(default_factory=list, description="Limitations or conditions")
    source_count: int = Field(default=0, description="Number of sources supporting this")

    @field_validator("evidence", "caveats", mode="before")
    @classmethod
    def coerce_list_to_str(cls, v):
        """Convert list-of-dicts or mixed lists to list[str]."""
        if not isinstance(v, list):
            return [str(v)] if v else []
        result = []
        for item in v:
            if isinstance(item, dict):
                # extract text from common dict shapes
                result.append(str(item.get("text") or item.get("value") or item.get("title") or next(iter(item.values()), "")))
            elif item is not None:
                result.append(str(item))
        return result

    @field_validator("evidence_strength", mode="before")
    @classmethod
    def coerce_evidence_strength(cls, v):
        """Accept dict like {'value':'strong'} or any string."""
        if isinstance(v, dict):
            v = v.get("value") or v.get("strength") or next(iter(v.values()), "moderate")
        if isinstance(v, str):
            v = v.lower().strip()
            mapping = {"strong": "strong", "moderate": "moderate", "weak": "weak", "anecdotal": "anecdotal"}
            return mapping.get(v, "moderate")
        return "moderate"


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

    @field_validator("factors", "mitigation", "evidence", "data_quality_issues", mode="before")
    @classmethod
    def coerce_list_to_str(cls, v):
        """Convert list-of-dicts or mixed lists to list[str]."""
        if not isinstance(v, list):
            return [str(v)] if v else []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(str(item.get("text") or item.get("value") or item.get("title") or next(iter(item.values()), "")))
            elif item is not None:
                result.append(str(item))
        return result

    @field_validator("level", mode="before")
    @classmethod
    def coerce_level(cls, v):
        """Accept dict or any string for level."""
        if isinstance(v, dict):
            v = v.get("value") or v.get("level") or next(iter(v.values()), "medium")
        if isinstance(v, str):
            return v.lower().strip()
        return "medium"


class AnalysisResults(BaseModel):
    """Complete analysis results from research data."""
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    query: str = Field(..., description="Original research query")
    timestamp: str = Field(..., description="When analysis was performed")
    key_findings: list[AnalysisInsight] = Field(default_factory=list, description="Primary insights discovered")
    risks_identified: list[RiskAssessment] = Field(default_factory=list, description="Risks evaluated")
    patterns_detected: list[str] = Field(default_factory=list, description="Patterns or trends identified")
    comparisons: dict[str, Any] = Field(default_factory=dict, description="Comparative analyses")
    overall_confidence: float = Field(default=0.5, description="Overall confidence in analysis (0-1)")
    reasoning_chain: list[str] = Field(default_factory=list, description="Steps in reasoning process")
    data_sources_analyzed: int = Field(default=0, description="Number of sources analyzed")
    methodology: str = Field(default="Multi-source analysis", description="Analysis methodology used")
    limitations: list[str] = Field(default_factory=list, description="Analysis limitations")

    @field_validator("patterns_detected", "reasoning_chain", "limitations", mode="before")
    @classmethod
    def coerce_str_list(cls, v):
        """Ensure list[str] — convert any dict items to their string representation."""
        if not isinstance(v, list):
            return [str(v)] if v else []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(str(item.get("text") or item.get("value") or next(iter(item.values()), "")))
            elif item is not None:
                result.append(str(item))
        return result


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

    SYSTEM_PROMPT = """You are a JSON-output Investment and Strategic Analyst. Think VERY briefly (3 sentences max), then output the JSON immediately.

Your ONLY job: read the sources, extract financial insights, strategic moats, and investment risks (bull/bear cases), and return structured JSON.
Apply financial frameworks (e.g. SWOT, competitive analysis). Do NOT write long reasoning. Output JSON right after your brief think."""


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
            # Use max_tokens=2500: analyst prompt includes research sources,
            # so we need room for a moderate prompt + structured JSON output
            self.llm = get_chat_model(temperature=0.3, max_tokens=2500)

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

            if isinstance(analysis_data, list):
                if len(analysis_data) > 0 and isinstance(analysis_data[0], dict):
                    analysis_data = analysis_data[0]
                else:
                    analysis_data = {
                        "key_findings": [
                            {"insight_id": f"insight_{idx}", "category": "general", "statement": str(item), "confidence": 0.5, "evidence": [], "evidence_strength": "moderate", "caveats": [], "source_count": 1}
                            for idx, item in enumerate(analysis_data)
                        ]
                    }
            if not isinstance(analysis_data, dict):
                analysis_data = {}

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
        """Create a compact prompt for analysis that fits in Phi-4-mini 4096 context."""
        items = sources if sources else search_results
        # Each source gets max 500 chars to keep prompt reasonable
        sources_text = "\n".join([
            f"{i+1}. {(s.get('title','?') if isinstance(s,dict) else str(s))[:100]}: {(s.get('snippet','') if isinstance(s,dict) else '')[:400]}"
            for i, s in enumerate(items[:8])
        ]) or "No sources provided."

        return f"""Analyze the provided sources to answer: {query[:200]}

Sources:
{sources_text}

Based on these sources, perform a thorough strategic and risk analysis.
Generate a structured JSON response. Replace the placeholder values in the template below with your actual synthesized findings and detailed analysis.

JSON Schema to populate:
{{
  "key_findings": [
    {{
      "insight_id": "i1",
      "category": "market/financial/regulatory/etc",
      "statement": "Detailed analysis statement summarizing a key finding...",
      "confidence": 0.85,
      "evidence": ["Brief quote or specific data point from sources"],
      "evidence_strength": "strong",
      "caveats": ["Any limitations or conditions of this finding"],
      "source_count": 1
    }}
  ],
  "risks": [
    {{
      "risk_id": "r1",
      "title": "Descriptive risk title",
      "description": "Thorough risk description detailing the threat...",
      "level": "high",
      "probability": 0.7,
      "impact": 0.8,
      "risk_score": 0.56,
      "factors": ["Contributing factor 1", "Contributing factor 2"],
      "mitigation": ["Actionable mitigation step 1", "Actionable mitigation step 2"],
      "evidence": ["Source reference or data point supporting risk"],
      "confidence": 0.8,
      "data_quality_issues": []
    }}
  ],
  "patterns": ["Detected trend or pattern 1", "Detected trend or pattern 2"],
  "comparisons": {{}},
  "overall_confidence": 0.75,
  "reasoning_chain": ["Step 1 of analytical logic", "Step 2 of analytical logic"],
  "limitations": ["Data gap or limitation 1"]
}}

Return ONLY valid JSON. Start with {{ and end with }}"""

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