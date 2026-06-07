#!/usr/bin/env python3
"""
quick_demo.py  — Lightweight end-to-end smoke-test (no full orchestration)
===========================================================================
Tests each agent individually so you can confirm each piece works before
running the complete pipeline.

Usage:
    python quick_demo.py               # all tests
    python quick_demo.py --only model  # just model connection
    python quick_demo.py --only search # just web search
    python quick_demo.py --only planner
    python quick_demo.py --only full   # full 4-agent pipeline
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Terminal colours ──────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"; B=BOLD; C="\033[96m"; G="\033[92m"
Y="\033[93m"; RE="\033[91m"; BL="\033[94m"; M="\033[95m"; D="\033[2m"

def hdr(t,c=C): print(f"\n{c}{B}{'─'*64}\n  {t}\n{'─'*64}{R}")
def ok(m):  print(f"  {G}✓{R}  {m}")
def fail(m):print(f"  {RE}✗{R}  {m}")
def info(m):print(f"  {D}→{R}  {m}")
def sec(m): print(f"\n{BL}{B}[{m}]{R}")

# ── Test 1: Model connection ──────────────────────────────────────────────────
async def test_model():
    hdr("TEST 1 — Azure Foundry Model Connection", C)
    from openai import OpenAI

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT","")
    api_key  = os.environ.get("AZURE_OPENAI_API_KEY","")
    deploy   = os.environ.get("AZURE_OPENAI_DEPLOYMENT","phi-4-mini-reasoning")

    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint[:-len("/chat/completions")]

    info(f"Endpoint   : {endpoint}")
    info(f"Deployment : {deploy}")

    client = OpenAI(base_url=endpoint, api_key=api_key)
    try:
        r = client.chat.completions.create(
            model=deploy,
            messages=[
                {"role":"system","content":"You are a concise reasoning assistant. Show your thinking briefly."},
                {"role":"user","content":"In 2 sentences: What is LangGraph and why is it useful for multi-agent systems?"}
            ],
            max_tokens=300,
        )
        print(f"\n{G}Model reply:{R}")
        print(r.choices[0].message.content)
        ok(f"Tokens used: {r.usage.total_tokens if r.usage else 'N/A'}")
        return True
    except Exception as e:
        fail(f"Model error: {e}")
        return False

# ── Test 2: MCP Web Search ────────────────────────────────────────────────────
async def test_search():
    hdr("TEST 2 — MCP Web Search Tool", M)
    from src.mcp_tools.web_search import MCPWebSearchTool

    tool = MCPWebSearchTool(
        azure_project_endpoint=os.environ.get("AZURE_PROJECT_ENDPOINT"),
        azure_toolbox_name=os.environ.get("AZURE_TOOLBOX_NAME","reasoning-agent-web-search"),
        azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )

    q = "India electric vehicle market 2025 growth"
    info(f"Search query: {q}")
    try:
        resp = await tool.search(q, max_results=4)
        ok(f"Results: {resp.total_results}  Time: {resp.execution_time_ms:.0f}ms")
        for i, r in enumerate(resp.results, 1):
            print(f"  {i}. {C}{r.title[:60]}{R}  {D}({r.source}){R}")
        await tool.close()
        return True
    except Exception as e:
        fail(f"Search error: {e}")
        return False

# ── Test 3: Planner agent ─────────────────────────────────────────────────────
async def test_planner():
    hdr("TEST 3 — Planner Agent (Task Decomposition)", Y)
    from src.utils.config import load_environment
    load_environment()

    from src.agents.planner import PlannerAgent
    planner = PlannerAgent()
    q = "What are the main risks of investing in AI startups in India?"
    info(f"Query: {q}")
    try:
        plan = await planner.decompose_task(q)
        ok(f"Plan ID    : {plan.plan_id}")
        ok(f"Tasks      : {plan.total_tasks}")
        ok(f"Confidence : {plan.confidence_score:.2f}")
        print(f"\n  {B}Sub-tasks breakdown:{R}")
        for t in plan.tasks:
            print(f"    {BL}[{t.task_id}]{R} {t.task_type.value:22s} → {t.description[:55]}…")
        print(f"\n  {B}Execution order:{R} {' → '.join(plan.execution_order)}")
        return True
    except Exception as e:
        fail(f"Planner error: {e}")
        import traceback; traceback.print_exc()
        return False

# ── Test 4: Researcher agent ──────────────────────────────────────────────────
async def test_researcher():
    hdr("TEST 4 — Researcher Agent (Search + Rank)", BL)
    from src.utils.config import load_environment
    load_environment()

    from src.agents.researcher import ResearcherAgent
    researcher = ResearcherAgent()
    q = "Top 3 investment risks in Indian EV market"
    info(f"Query: {q}")
    try:
        results = await researcher.search(q, max_results=8)
        ok(f"Total sources       : {results.total_sources}")
        ok(f"High-confidence     : {len(results.high_confidence_sources)}")
        ok(f"Medium-confidence   : {len(results.medium_confidence_sources)}")
        ok(f"Confidence score    : {results.confidence_score:.2f}")
        if results.high_confidence_sources:
            print(f"\n  {B}Top sources:{R}")
            for r in results.high_confidence_sources[:3]:
                print(f"    • {C}{r.title[:55]}{R}  {D}[{r.source_name}]{R}")
        return True
    except Exception as e:
        fail(f"Researcher error: {e}")
        return False

# ── Test 5: Analyst agent ─────────────────────────────────────────────────────
async def test_analyst():
    hdr("TEST 5 — Analyst Agent (Insight Extraction)", M)
    from src.utils.config import load_environment
    load_environment()

    from src.agents.analyst import AnalystAgent
    analyst = AnalystAgent()
    q = "What are the top 3 investment risks in the Indian EV market?"
    research_data = {
        "sources": [
            {"title": "India EV Policy Report", "snippet": "Regulatory uncertainty and subsidy fluctuations remain top concerns", "source_name": "Reuters"},
            {"title": "EV Battery Supply Chain", "snippet": "India imports 80% of lithium—supply chain risk is significant", "source_name": "Bloomberg"},
            {"title": "Market Competition 2024", "snippet": "Chinese OEMs entering India with lower-cost models", "source_name": "Economic Times"},
        ],
        "search_results": []
    }
    info(f"Analyzing {len(research_data['sources'])} sources for: {q[:60]}…")
    try:
        results = await analyst.analyze(q, research_data)
        ok(f"Insights found : {len(results.key_findings)}")
        ok(f"Risks flagged  : {len(results.risks_identified)}")
        ok(f"Confidence     : {results.overall_confidence:.2f}")
        if results.key_findings:
            print(f"\n  {B}Key findings:{R}")
            for f in results.key_findings[:3]:
                print(f"    [{f.category}] {f.statement[:65]}…")
        if results.risks_identified:
            print(f"\n  {B}Identified risks:{R}")
            for r in results.risks_identified[:3]:
                print(f"    {RE}[{r.level.value.upper()}]{R} {r.title}  score={r.risk_score:.2f}")
        return True
    except Exception as e:
        fail(f"Analyst error: {e}")
        return False

# ── Test 6: Writer agent ──────────────────────────────────────────────────────
async def test_writer():
    hdr("TEST 6 — Writer Agent (Report Generation)", G)
    from src.utils.config import load_environment
    load_environment()

    from src.agents.writer import WriterAgent, ReportFormat
    writer = WriterAgent()
    q = "What are the top 3 investment risks in the Indian EV market?"
    analysis = {
        "key_findings": [
            {"category": "regulatory", "statement": "Policy uncertainty threatens long-term ROI",
             "confidence": 0.85, "evidence": ["FAME-II subsidy cuts"], "evidence_strength": "strong", "source_count": 4},
            {"category": "supply_chain", "statement": "Lithium import dependency creates operational risk",
             "confidence": 0.78, "evidence": ["80% import ratio"], "evidence_strength": "moderate", "source_count": 3},
        ],
        "risks_identified": [
            {"risk_id": "r1", "title": "Regulatory Policy Shifts", "description": "Subsidy changes reduce margins",
             "level": "high", "probability": 0.7, "impact": 0.8, "risk_score": 0.56,
             "mitigation": ["Diversify revenue", "Engage policymakers"], "confidence": 0.85},
        ],
        "patterns_detected": ["Growing domestic manufacturing push"],
        "reasoning_chain": ["Regulatory risk identified first", "Supply chain confirmed via import data"],
        "overall_confidence": 0.82,
        "data_sources_analyzed": 7
    }
    info(f"Generating report for: {q[:60]}…")
    try:
        report = await writer.generate_report(q, analysis)
        ok(f"Report ID    : {report.metadata.report_id}")
        ok(f"Sections     : {len(report.sections)}")
        ok(f"Conclusions  : {len(report.conclusions)}")
        ok(f"Confidence   : {report.metadata.confidence_score:.2f}")
        print(f"\n  {B}Executive Summary (preview):{R}")
        print(f"  {D}{report.executive_summary[:200]}…{R}")
        return True
    except Exception as e:
        fail(f"Writer error: {e}")
        return False

# ── Test 7: Full pipeline ─────────────────────────────────────────────────────
async def test_full_pipeline():
    hdr("TEST 7 — FULL AGENT PIPELINE (Planner→Researcher→Analyst→Writer)", G)
    from src.utils.config import load_environment
    load_environment()

    from src.orchestration.research_workflow import ResearchWorkflow
    workflow = ResearchWorkflow(enable_a2a=True, enable_mcp=True, max_retries=2)

    q = "What are the top 3 investment risks in the Indian EV market?"
    info(f"Query: {q}")
    info(f"Starting at {datetime.now().strftime('%H:%M:%S')} — this may take 30-90 seconds …")
    print()

    stages = ["📋 Planner", "🔍 Researcher", "🧠 Analyst", "📝 Writer"]
    for s in stages:
        print(f"  {D}{s} …{R}")

    t0 = datetime.utcnow()
    try:
        result = await workflow.execute(q)
        elapsed = (datetime.utcnow() - t0).total_seconds()

        status = result.get("status","")
        if "completed" in status:
            ok(f"Pipeline completed in {elapsed:.1f}s")
        else:
            fail(f"Status: {status}")

        meta = result.get("metadata", {})
        ok(f"Stages done : {', '.join(meta.get('completed_tasks',[]))}")
        cs = result.get("confidence_score")
        if cs:
            ok(f"Confidence  : {cs:.0%}")

        report = result.get("report")
        if report:
            ok(f"Report title: {report.metadata.title}")
            print(f"\n  {B}Executive Summary:{R}")
            print(f"  {report.executive_summary[:350]}…")
            if report.conclusions:
                print(f"\n  {B}Conclusions:{R}")
                for i, c in enumerate(report.conclusions[:3], 1):
                    print(f"  {i}. {c}")
        return "completed" in status
    except Exception as e:
        fail(f"Pipeline error: {e}")
        import traceback; traceback.print_exc()
        return False

# ── Main ──────────────────────────────────────────────────────────────────────
TESTS = {
    "model":      test_model,
    "search":     test_search,
    "planner":    test_planner,
    "researcher": test_researcher,
    "analyst":    test_analyst,
    "writer":     test_writer,
    "full":       test_full_pipeline,
}

async def main():
    p = argparse.ArgumentParser(description="Quick smoke-test for all agents")
    p.add_argument("--only", choices=list(TESTS.keys()), help="Run only this test")
    args = p.parse_args()

    hdr("Phi-4 Reasoning Multi-Agent — Quick Demo & Test Suite", C)
    info(f"Python  : {sys.version.split()[0]}")
    info(f"Root    : {ROOT}")
    info(f"Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests_to_run = {args.only: TESTS[args.only]} if args.only else TESTS
    results = {}

    for name, fn in tests_to_run.items():
        try:
            results[name] = await fn()
        except Exception as e:
            fail(f"Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    hdr("SUMMARY", G if all(results.values()) else RE)
    for name, passed in results.items():
        (ok if passed else fail)(f"{name:12s} {'PASS' if passed else 'FAIL'}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {G if passed==total else Y}{BOLD}{passed}/{total} tests passed{R}\n")

    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    asyncio.run(main())
