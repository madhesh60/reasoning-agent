"""
Research-to-Report Orchestration — 6-Agent Pipeline
=====================================================
Mirrors the Foundry visual workflow exactly:

  planner-agent  →  researcher-agent  →  industry-news-trend-scanner
  →  competitive-landscape-researcher  →  analyst-agent  →  writer-agent

Tool-enabled agents (planner, researcher, industry-news-trend-scanner,
competitive-landscape-researcher, analyst) use the AgentsClient thread+run API
so their Bing web search / MCP toolbox calls execute correctly.

Writer-agent (no tools) uses the Responses API — faster and proven reliable.

FALLBACK: If any Foundry call fails, that stage is skipped and accumulated
context is passed to the next stage. If writer also fails → local fallback model.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any

import json_repair
import structlog

from ..agents import (
    GeneratedReport, ReportMetadata, ReportSection,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Agent config from .env
# ---------------------------------------------------------------------------
_EP = os.getenv(
    "AZURE_PROJECT_ENDPOINT",
    "https://reasoning-agent-hack2-resource.services.ai.azure.com/api/projects/reasoning-agent-hack2",
)

# Agent config — reads asst_xxx IDs (from setup_agent_ids.py) for thread-based calls
# Falls back to old GUID IDs if asst IDs not yet set
_AGENTS = {
    "planner": {
        "name":    os.getenv("PLANNER_AGENT_NAME",    "planner-agent"),
        "version": os.getenv("PLANNER_AGENT_VERSION", "12"),
        "id":      os.getenv("PLANNER_ASST_ID") or os.getenv("PLANNER_AGENT_ID") or "",
    },
    "researcher": {
        "name":    os.getenv("RESEARCHER_AGENT_NAME",    "researcher-agent"),
        "version": os.getenv("RESEARCHER_AGENT_VERSION", "12"),
        "id":      os.getenv("RESEARCHER_ASST_ID") or os.getenv("RESEARCHER_AGENT_ID") or "",
    },
    "industry_news": {
        "name":    os.getenv("INDUSTRY_NEWS_AGENT_NAME",    "industry-news-trend-scanner"),
        "version": os.getenv("INDUSTRY_NEWS_AGENT_VERSION", "5"),
        "id":      os.getenv("INDUSTRY_NEWS_ASST_ID") or os.getenv("INDUSTRY_NEWS_AGENT_ID") or "",
    },
    "competitive": {
        "name":    os.getenv("COMPETITIVE_AGENT_NAME",    "competitive-landscape-researcher"),
        "version": os.getenv("COMPETITIVE_AGENT_VERSION", "2"),
        "id":      os.getenv("COMPETITIVE_ASST_ID") or os.getenv("COMPETITIVE_AGENT_ID") or "",
    },
    "analyst": {
        "name":    os.getenv("ANALYST_AGENT_NAME",    "analyst-agent"),
        "version": os.getenv("ANALYST_AGENT_VERSION", "4"),
        "id":      os.getenv("ANALYST_ASST_ID") or os.getenv("ANALYST_AGENT_ID") or "",
    },
    "writer": {
        "name":    os.getenv("WRITER_AGENT_NAME",    "writer-agent"),
        "version": os.getenv("WRITER_AGENT_VERSION", "4"),
        "id":      None,   # No tools — uses Responses API (faster, proven to work)
    },
}


# ---------------------------------------------------------------------------
# Low-level callers
# ---------------------------------------------------------------------------
def _responses_call(name: str, version: str, prompt: str) -> str:
    """Responses API — works for all Foundry Workflow-built agents."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    with AIProjectClient(
        endpoint=_EP,
        credential=DefaultAzureCredential(exclude_interactive_browser_credential=True),
    ) as client:
        oc = client.get_openai_client()
        resp = oc.responses.create(
            input=[{"role": "user", "content": prompt}],
            extra_body={"agent_reference": {
                "name": name, "version": version, "type": "agent_reference"
            }},
        )
        return resp.output_text


async def _call(role: str, prompt: str, retries: int = 2) -> str | None:
    """
    Async wrapper with retries. Returns None on failure so pipeline continues.
    All agents use Responses API — the only API compatible with Foundry Workflow agents.
    """
    cfg  = _AGENTS[role]
    loop = asyncio.get_running_loop()

    for attempt in range(1, retries + 2):
        try:
            result = await loop.run_in_executor(
                None, _responses_call, cfg["name"], cfg["version"], prompt
            )
            logger.info("agent_success", role=role)
            return result
        except Exception as exc:
            logger.warning("agent_failed", role=role, attempt=attempt, error=str(exc)[:100])
            if attempt <= retries:
                await asyncio.sleep(min(2 ** attempt, 10))

    logger.warning("agent_skipped", role=role)
    return None


# ---------------------------------------------------------------------------
# Report parser
# ---------------------------------------------------------------------------
def _parse_report(raw: str, query: str, elapsed: float) -> GeneratedReport:
    try:
        data = json_repair.loads(raw)
        if isinstance(data, dict) and ("executive_summary" in data or "sections" in data):
            meta_d = data.get("metadata", {})
            return GeneratedReport(
                metadata=ReportMetadata(
                    title=meta_d.get("title", query[:80]),
                    report_id=meta_d.get("report_id", str(uuid.uuid4())[:8]),
                    confidence_score=float(meta_d.get("confidence_score", 0.88)),
                    processing_time_seconds=elapsed,
                    generated_at=datetime.utcnow().isoformat(),
                ),
                executive_summary=data.get("executive_summary", raw[:600]),
                sections=[
                    ReportSection(title=s.get("title",""), content=s.get("content",""))
                    for s in data.get("sections", []) if isinstance(s, dict)
                ],
                conclusions=data.get("conclusions", []),
                recommendations=data.get("recommendations", []),
                citations=data.get("citations", []),
            )
    except Exception:
        pass

    # Markdown fallback
    sections, cur_title, cur_lines = [], "", []
    for line in raw.split("\n"):
        if line.startswith("## "):
            if cur_title or cur_lines:
                sections.append(ReportSection(title=cur_title or "Overview", content="\n".join(cur_lines).strip()))
            cur_title, cur_lines = line[3:].strip(), []
        elif line.startswith("# "):
            cur_title = line[2:].strip()
        else:
            cur_lines.append(line)
    if cur_lines:
        sections.append(ReportSection(title=cur_title or "Analysis", content="\n".join(cur_lines).strip()))

    exec_sum = next((s.content[:600] for s in sections if s.content), raw[:600])
    return GeneratedReport(
        metadata=ReportMetadata(
            title=query[:80],
            report_id=str(uuid.uuid4())[:8],
            confidence_score=0.85,
            processing_time_seconds=elapsed,
            generated_at=datetime.utcnow().isoformat(),
        ),
        executive_summary=exec_sum,
        sections=sections,
        conclusions=[],
        recommendations=[],
        citations=[],
    )


# ---------------------------------------------------------------------------
# ResearchWorkflow — public API
# ---------------------------------------------------------------------------
class ResearchWorkflow:
    """
    Runs the 6-agent pipeline mirroring the Foundry visual workflow:

      planner → researcher → industry-news-trend-scanner
      → competitive-landscape-researcher → analyst → writer

    Each stage accumulates context and passes it forward.
    Failed stages are skipped gracefully — the pipeline always completes.
    """

    def __init__(
        self,
        enable_a2a: bool = True,
        enable_mcp: bool = False,
        max_retries: int = 3,
        memory_db_path: str = "memory.sqlite",
    ):
        self.enable_a2a  = enable_a2a
        self.max_retries = max_retries

    async def execute_streaming(
        self,
        query: str,
        session_id: str | None = None,
        memory_context: str = "",
        user_clarifications: str = "",
    ):
        stages = [
            "planner", "researcher", "industry_news",
            "competitive", "analyst", "writer",
        ]
        labels = {
            "planner":      "Planner: Decomposing query into research tasks...",
            "researcher":   "Researcher: Fetching latest data from the web...",
            "industry_news":"Industry News Scanner: Real-time trend signals...",
            "competitive":  "Competitive Intel: Mapping the competitive landscape...",
            "analyst":      "Analyst: Extracting insights and risk factors...",
            "writer":       "Writer: Generating the structured report...",
        }
        for s in stages:
            yield {"stage": s, "status": "in_progress", "label": labels[s]}

        try:
            result = await self.execute(
                query,
                session_id=session_id,
                memory_context=memory_context,
                user_clarifications=user_clarifications,
            )
            yield {"stage": "complete", "status": "completed", "result": result}
        except Exception as exc:
            yield {"stage": "complete", "status": "error", "errors": [str(exc)]}

    async def execute(
        self,
        query: str,
        session_id: str | None = None,
        memory_context: str = "",
        user_clarifications: str = "",
    ) -> dict[str, Any]:
        t0 = datetime.utcnow()
        logger.info("pipeline_start", query=query[:80])

        # Build the personalisation prefix once; every agent receives it
        _prefix = ""
        if memory_context:
            _prefix += memory_context + "\n"
        if user_clarifications:
            _prefix += user_clarifications + "\n"

        context = ""   # Accumulated stage output passed stage-to-stage
        stages_ok: list[str] = []
        stages_failed: list[str] = []

        if self.enable_a2a:
            # ── Stage 1: Planner ───────────────────────────────────────────────
            plan_prompt = (
                f"{_prefix}"
                f"Research Query: {query}\n\n"
                "Decompose this into a structured research plan with sub-tasks covering: "
                "current trends, key players, market data, risks, and opportunities. "
                "Be specific about what to research and why."
            )
            plan_out = await _call("planner", plan_prompt, self.max_retries)
            if plan_out:
                context += f"\n\n=== RESEARCH PLAN ===\n{plan_out}"
                stages_ok.append("planner")
            else:
                stages_failed.append("planner")

            # ── Stage 2: Researcher ────────────────────────────────────────────
            research_prompt = (
                f"Research Query: {query}\n"
                f"{context}\n\n"
                "Conduct comprehensive web research. Find:\n"
                "- Latest statistics, market data, and reports\n"
                "- Recent news and developments (last 6-12 months)\n"
                "- Key academic papers or industry publications\n"
                "- Specific URLs and sources\n"
                "Return detailed findings with source links."
            )
            research_out = await _call("researcher", research_prompt, self.max_retries)
            if research_out:
                context += f"\n\n=== WEB RESEARCH ===\n{research_out}"
                stages_ok.append("researcher")
            else:
                stages_failed.append("researcher")

            # ── Stage 3: Industry News Scanner ────────────────────────────────
            news_prompt = (
                f"Topic: {query}\n"
                f"{context}\n\n"
                "Scan for the most recent industry news, trend signals, and breaking "
                "developments relevant to this topic. Focus on the last 3-6 months. "
                "Include specific headlines and publication sources."
            )
            news_out = await _call("industry_news", news_prompt, self.max_retries)
            if news_out:
                context += f"\n\n=== INDUSTRY NEWS & TRENDS ===\n{news_out}"
                stages_ok.append("industry_news")
            else:
                stages_failed.append("industry_news")

            # ── Stage 4: Competitive Landscape Researcher ──────────────────────
            comp_prompt = (
                f"Topic: {query}\n"
                f"{context}\n\n"
                "Analyze the competitive landscape. Identify:\n"
                "- Key players and their positioning\n"
                "- Market share and differentiation strategies\n"
                "- Emerging challengers and disruptors\n"
                "- Gaps and white-space opportunities\n"
                "Include specific companies, products, and data points."
            )
            comp_out = await _call("competitive", comp_prompt, self.max_retries)
            if comp_out:
                context += f"\n\n=== COMPETITIVE LANDSCAPE ===\n{comp_out}"
                stages_ok.append("competitive")
            else:
                stages_failed.append("competitive")

            # ── Stage 5: Analyst ───────────────────────────────────────────────
            analysis_prompt = (
                f"Topic: {query}\n"
                f"{context}\n\n"
                "Synthesize all research above into a structured analysis:\n"
                "- Key insights and patterns\n"
                "- Risk assessment (high/medium/low)\n"
                "- Strategic opportunities\n"
                "- Confidence levels for each finding\n"
                "Be specific and evidence-based."
            )
            analysis_out = await _call("analyst", analysis_prompt, self.max_retries)
            if analysis_out:
                context += f"\n\n=== ANALYSIS ===\n{analysis_out}"
                stages_ok.append("analyst")
            else:
                stages_failed.append("analyst")

            # ── Stage 6: Writer ────────────────────────────────────────────────
            writer_prompt = (
                f"{_prefix}"
                f"Write a comprehensive, professional research report on:\n\n"
                f"TOPIC: {query}\n\n"
                f"Use all the following research and analysis:\n{context}\n\n"
                "Structure the report as:\n"
                "- Executive Summary (3-4 paragraphs)\n"
                "- Current State & Key Trends\n"
                "- Competitive Landscape\n"
                "- Opportunities & Risks\n"
                "- Strategic Recommendations (numbered, specific, actionable)\n"
                "- Resources & References (with real URLs)\n\n"
                "Be detailed, insightful, and professional. Include specific data points, "
                "company names, statistics, and source links throughout."
            )
            writer_out = await _call("writer", writer_prompt, self.max_retries)
            if writer_out:
                stages_ok.append("writer")
                elapsed = (datetime.utcnow() - t0).total_seconds()
                report  = _parse_report(writer_out, query, elapsed)
                path    = f"foundry_6agent ({'→'.join(stages_ok)})"

                logger.info("pipeline_complete", stages_ok=stages_ok, stages_failed=stages_failed)
                return self._result(report, query, t0, path, stages_ok, stages_failed)
            else:
                stages_failed.append("writer")

        # ── Fallback: Local Model ──────────────────────────────────────────────
        logger.info("using_local_fallback", stages_ok=stages_ok)
        report  = await self._local_fallback(query, context, t0)
        elapsed = (datetime.utcnow() - t0).total_seconds()
        return self._result(report, query, t0, "local_fallback", stages_ok, stages_failed)

    def _result(self, report, query, t0, path, ok_stages, fail_stages) -> dict:
        elapsed = (datetime.utcnow() - t0).total_seconds()
        return {
            "status":                   "completed",
            "query":                    query,
            "report":                   report,
            "path_used":                path,
            "confidence_score":         report.metadata.confidence_score,
            "processing_time_seconds":  elapsed,
            "metadata": {
                "start_time":      t0.isoformat(),
                "end_time":        datetime.utcnow().isoformat(),
                "completed_tasks": ok_stages,
                "failed_tasks":    fail_stages,
                "errors":          [],
                "confidence_scores": {"overall": report.metadata.confidence_score},
            },
        }

    async def _local_fallback(self, query: str, context: str, t0: datetime) -> GeneratedReport:
        try:
            from ..utils.config import get_chat_model
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = get_chat_model(temperature=0.4, max_tokens=3000)
            prompt = (
                f"Write a comprehensive research report on: {query}\n\n"
                + (f"Use this research context:\n{context[:3000]}\n\n" if context else "")
                + "Include: Executive Summary, Current Trends, Competitive Landscape, "
                "Opportunities & Risks, Strategic Recommendations, Resources."
            )
            resp    = await llm.ainvoke([
                SystemMessage(content="You are an expert research analyst. Write detailed, insightful reports."),
                HumanMessage(content=prompt),
            ])
            elapsed = (datetime.utcnow() - t0).total_seconds()
            return _parse_report(resp.content, query, elapsed)
        except Exception as exc:
            logger.error("local_fallback_failed", error=str(exc))
            elapsed = (datetime.utcnow() - t0).total_seconds()
            return GeneratedReport(
                metadata=ReportMetadata(
                    title=query[:80], report_id=str(uuid.uuid4())[:8],
                    confidence_score=0.5, processing_time_seconds=elapsed,
                    generated_at=datetime.utcnow().isoformat(),
                ),
                executive_summary=f"Could not complete research on '{query}'. Check Azure credentials.",
                sections=[], conclusions=[],
                recommendations=["Run `az login` and verify AZURE_PROJECT_ENDPOINT in .env"],
                citations=[],
            )