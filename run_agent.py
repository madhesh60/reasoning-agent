#!/usr/bin/env python3
"""
run_agent.py - One-command launcher for the Phi-4 Reasoning Multi-Agent System
===============================================================================
Usage:
    python run_agent.py
    python run_agent.py --query "What are the top AI trends in 2025?"
    python run_agent.py --query "..." --model phi-4-reasoning   # switch to full model
    python run_agent.py --stream                                # streaming output
    python run_agent.py --model-test                           # test model connection only
    python run_agent.py --search-test                          # test MCP web search only
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# ── Bootstrap path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Coloured terminal helpers (no deps) ──────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"

def hdr(text: str, color: str = CYAN):
    width = min(os.get_terminal_size().columns, 72) if hasattr(os, "get_terminal_size") else 72
    print(f"\n{color}{BOLD}{'═' * width}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'═' * width}{RESET}")

def step(icon: str, label: str, detail: str = "", color: str = BLUE):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{DIM}[{ts}]{RESET} {color}{BOLD}{icon} {label}{RESET}  {DIM}{detail}{RESET}")

def ok(msg: str):   print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg: str):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg: str): print(f"  {DIM}→{RESET}  {msg}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Phi-4 Reasoning Multi-Agent — Research-to-Report Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py
      → Interactive mode: prompts you for a query

  python run_agent.py --query "Top 3 risks in Indian EV market"
      → Single query, full agent pipeline

  python run_agent.py --query "..." --model phi-4-reasoning
      → Use the larger Phi-4 model

  python run_agent.py --model-test
      → Only test Azure Foundry model connection

  python run_agent.py --search-test
      → Only test MCP web search tool

  python run_agent.py --stream
      → Interactive mode with streaming stage-by-stage output
        """,
    )
    p.add_argument("--query", "-q", help="Research query to process")
    p.add_argument("--model", "-m", default=None,
                   help="Override deployment name (e.g. phi-4-reasoning or phi-4-mini-reasoning)")
    p.add_argument("--stream", action="store_true",
                   help="Stream stage-by-stage output instead of waiting for full result")
    p.add_argument("--model-test", action="store_true",
                   help="Run only the Azure model connection test and exit")
    p.add_argument("--search-test", action="store_true",
                   help="Run only the MCP web-search tool test and exit")
    p.add_argument("--format", choices=["markdown", "json", "summary"], default="markdown",
                   help="Output format for the final report (default: markdown)")
    return p.parse_args()

# ── Model connection test ─────────────────────────────────────────────────────
async def run_model_test(model_override: str | None = None):
    hdr("Azure AI Foundry — Model Connection Test", CYAN)

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deploy   = model_override or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "phi-4-mini-reasoning")

    if not endpoint or not api_key:
        err("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set in .env")
        return False

    # Strip /chat/completions if accidentally left in
    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint[:-len("/chat/completions")]

    step("🔌", "Connecting to endpoint:", endpoint, CYAN)
    step("🤖", "Using deployment   :", deploy, CYAN)

    from openai import OpenAI
    client = OpenAI(base_url=endpoint, api_key=api_key)

    messages = [
        {"role": "system", "content": (
            "You are a concise reasoning assistant. "
            "Think step by step, show your reasoning, then give a short answer."
        )},
        {"role": "user", "content": (
            "What are the top 3 investment risks in the Indian EV market? "
            "Think step-by-step then give a short bullet-point answer."
        )},
    ]

    step("📤", "Sending test prompt …", "", YELLOW)
    try:
        completion = client.chat.completions.create(
            model=deploy,
            messages=messages,
            max_tokens=900,
        )
        msg = completion.choices[0].message
        print(f"\n{GREEN}{BOLD}── Model Response ──────────────────────────────────{RESET}")
        print(msg.content)
        print(f"{GREEN}{BOLD}────────────────────────────────────────────────────{RESET}")
        ok(f"Finish reason : {completion.choices[0].finish_reason}")
        ok(f"Total tokens  : {completion.usage.total_tokens if completion.usage else 'N/A'}")
        return True
    except Exception as e:
        err(f"Model call failed: {e}")
        return False

# ── MCP web-search test ───────────────────────────────────────────────────────
async def run_search_test():
    hdr("MCP / Azure Foundry — Web Search Tool Test", MAGENTA)

    project_ep = os.environ.get("AZURE_PROJECT_ENDPOINT", "")
    if not project_ep:
        warn("AZURE_PROJECT_ENDPOINT not set — will use simulated fallback results")

    from src.mcp_tools.web_search import MCPWebSearchTool

    tool = MCPWebSearchTool(
        azure_project_endpoint=project_ep or None,
        azure_toolbox_name=os.environ.get("AZURE_TOOLBOX_NAME", "reasoning-agent-web-search"),
        azure_toolbox_version=os.environ.get("AZURE_TOOLBOX_VERSION", "1"),
        azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )

    query = "Latest AI research trends 2025"
    step("🔍", "Search query:", query, YELLOW)

    response = await tool.search(query, max_results=5)

    print(f"\n{BLUE}{BOLD}── Search Results ──────────────────────────────────{RESET}")
    print(f"  Query          : {response.query}")
    print(f"  Results found  : {response.total_results}")
    print(f"  Execution time : {response.execution_time_ms:.0f} ms")
    for i, r in enumerate(response.results, 1):
        print(f"\n  {CYAN}{i}. {r.title}{RESET}")
        print(f"     Source    : {r.source}")
        print(f"     URL       : {r.url}")
        print(f"     Relevance : {r.relevance_score:.2f}")
        snippet = r.snippet[:120].replace("\n", " ")
        print(f"     Snippet   : {DIM}{snippet}…{RESET}")
    print(f"{BLUE}{BOLD}────────────────────────────────────────────────────{RESET}\n")
    await tool.close()

# ── Full pipeline (streaming) ─────────────────────────────────────────────────
async def run_streaming(query: str, model_override: str | None = None):
    hdr(f"Research Agent — Streaming Mode", GREEN)
    info(f"Query: {query}")

    if model_override:
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = model_override

    from src.utils.config import load_environment
    load_environment()

    from src.orchestration.research_workflow import ResearchWorkflow
    workflow = ResearchWorkflow(enable_a2a=True, enable_mcp=True, max_retries=2)

    STAGE_ICONS = {
        "plan":          ("📋", "Planner Agent   — decomposing query into sub-tasks …",   CYAN),
        "validate_plan": ("✅", "Validator       — checking plan quality …",               YELLOW),
        "research":      ("🔍", "Researcher Agent — executing web searches …",             BLUE),
        "analyze":       ("🧠", "Analyst Agent   — extracting insights & risks …",         MAGENTA),
        "write_report":  ("📝", "Writer Agent    — generating structured report …",        GREEN),
        "complete":      ("🏁", "Pipeline complete",                                       GREEN),
        "error":         ("💥", "Error in stage",                                          RED),
    }

    async for update in workflow.execute_streaming(query):
        stage  = update.get("stage", "unknown")
        status = update.get("status", "")
        icon, label, color = STAGE_ICONS.get(stage, ("⚙️", stage, DIM))

        if status == "in_progress":
            step(icon, label, "", color)
        elif status == "completed" and stage != "complete":
            ok(f"{label.split('—')[0].strip()} done")
        elif stage == "complete":
            state = update.get("result", {})
            _print_report(query, state)
        elif status == "error":
            for e in update.get("errors", []):
                err(str(e))

# ── Full pipeline (non-streaming) ─────────────────────────────────────────────
async def run_query(query: str, model_override: str | None, fmt: str):
    hdr(f"Research-to-Report Multi-Agent  ·  Phi-4 Reasoning", GREEN)
    info(f"Query : {query}")
    if model_override:
        info(f"Model : {model_override}")
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = model_override

    from src.utils.config import load_environment
    load_environment()

    step("📋", "PLANNER       — breaking query into sub-tasks …",  "", CYAN)
    step("🔍", "RESEARCHER    — web search will run next …",         "", BLUE)
    step("🧠", "ANALYST       — reasoning & insight extraction …",   "", MAGENTA)
    step("📝", "WRITER        — report generation …",                "", GREEN)
    print(f"  {DIM}(all four agents running sequentially via LangGraph){RESET}\n")

    from src.orchestration.research_workflow import ResearchWorkflow
    workflow = ResearchWorkflow(enable_a2a=True, enable_mcp=True, max_retries=2)

    t0 = datetime.utcnow()
    result = await workflow.execute(query)
    elapsed = (datetime.utcnow() - t0).total_seconds()

    _print_result(query, result, fmt, elapsed)

# ── Pretty-print helpers ──────────────────────────────────────────────────────
def _print_result(query: str, result: dict, fmt: str, elapsed: float):
    status = result.get("status", "unknown")
    meta   = result.get("metadata", {})

    hdr("WORKFLOW RESULTS", GREEN if "completed" in status else RED)
    ok(f"Status          : {status}") if "completed" in status else err(f"Status: {status}")
    ok(f"Processing time : {elapsed:.1f} s")

    cs = result.get("confidence_score")
    if cs:
        ok(f"Confidence      : {cs:.0%}")

    # Completed tasks
    done = meta.get("completed_tasks", [])
    if done:
        print(f"\n  {BOLD}Completed stages:{RESET}")
        for t in done:
            ok(t)

    # Errors
    errs = meta.get("errors", [])
    if errs:
        print(f"\n  {BOLD}Errors encountered:{RESET}")
        for e in errs:
            warn(str(e))

    # Report
    report = result.get("report")
    if report:
        _print_report_object(report, fmt)
    else:
        err("No report generated — check errors above")

def _print_report(query: str, state: dict):
    """Used by streaming path where state is partial."""
    report = state.get("report")
    if report:
        _print_report_object(report, "markdown")
    else:
        warn("No report in final state")

def _print_report_object(report, fmt: str):
    from src.agents.writer import ReportFormat, WriterAgent
    import asyncio

    hdr(f"📄  {report.metadata.title}", GREEN)

    print(f"\n{BOLD}Executive Summary:{RESET}")
    print(report.executive_summary)

    if report.sections:
        print(f"\n{BOLD}Report Sections:{RESET}")
        for s in report.sections:
            print(f"\n  {CYAN}{BOLD}▶ {s.title}{RESET}")
            # Print up to 400 chars of each section
            content = s.content[:400].replace("\n", "\n    ")
            print(f"    {content}")
            if len(s.content) > 400:
                print(f"    {DIM}… [{len(s.content)-400} more chars]{RESET}")

    if report.conclusions:
        print(f"\n{BOLD}Key Conclusions:{RESET}")
        for i, c in enumerate(report.conclusions, 1):
            print(f"  {GREEN}{i}.{RESET} {c}")

    if report.recommendations:
        print(f"\n{BOLD}Recommendations:{RESET}")
        for i, r in enumerate(report.recommendations, 1):
            print(f"  {YELLOW}{i}.{RESET} {r}")

    if report.citations:
        print(f"\n{BOLD}Sources Referenced:{RESET}")
        for c in report.citations[:6]:
            title = c.get("title", "")[:60]
            url   = c.get("url", "")
            src   = c.get("source", "")
            print(f"  {DIM}• {title} — {src}{RESET}")
            if url and url.startswith("http"):
                print(f"    {DIM}{url}{RESET}")

    print(f"\n{DIM}Report ID: {report.metadata.report_id}  |  "
          f"Generated: {report.metadata.created_at[:19]}{RESET}\n")

    # Save markdown to file
    _save_report(report)

def _save_report(report):
    """Save the report as a markdown file in the project root."""
    try:
        from src.agents.writer import WriterAgent, ReportFormat
        writer = WriterAgent.__new__(WriterAgent)  # don't call __init__ again
        writer.llm = None

        md_lines = [
            f"# {report.metadata.title}\n",
            f"**Generated:** {report.metadata.created_at}  ",
            f"**Report ID:** `{report.metadata.report_id}`  ",
            f"**Confidence:** {report.metadata.confidence_score:.0%}\n",
            "---\n",
            "## Executive Summary\n",
            report.executive_summary + "\n",
        ]
        for s in report.sections:
            md_lines.append(f"## {s.title}\n")
            md_lines.append(s.content + "\n")
        if report.conclusions:
            md_lines.append("## Conclusions\n")
            for c in report.conclusions:
                md_lines.append(f"- {c}")
            md_lines.append("")
        if report.recommendations:
            md_lines.append("## Recommendations\n")
            for r in report.recommendations:
                md_lines.append(f"- {r}")
            md_lines.append("")
        if report.citations:
            md_lines.append("## References\n")
            for c in report.citations:
                md_lines.append(f"- [{c.get('title','')}]({c.get('url','')}) — {c.get('source','')}")

        fname = ROOT / f"report_{report.metadata.report_id}.md"
        fname.write_text("\n".join(md_lines), encoding="utf-8")
        ok(f"Report saved → {fname.name}")
    except Exception as e:
        warn(f"Could not save report: {e}")

# ── Interactive mode ──────────────────────────────────────────────────────────
async def interactive_mode(stream: bool, model_override: str | None, fmt: str):
    hdr("Research-to-Report — Interactive Mode  (Phi-4 Reasoning)", CYAN)
    print(f"  Type your research question and press Enter.")
    print(f"  Commands: {BOLD}exit{RESET} | {BOLD}model-test{RESET} | {BOLD}search-test{RESET}\n")

    while True:
        try:
            query = input(f"{CYAN}Query>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "model-test":
            await run_model_test(model_override)
            continue
        if query.lower() == "search-test":
            await run_search_test()
            continue

        try:
            if stream:
                await run_streaming(query, model_override)
            else:
                await run_query(query, model_override, fmt)
        except Exception as e:
            err(f"Pipeline error: {e}")

# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    args = parse_args()

    if args.model_test:
        ok_val = await run_model_test(args.model)
        sys.exit(0 if ok_val else 1)

    if args.search_test:
        await run_search_test()
        sys.exit(0)

    if args.query:
        if args.stream:
            await run_streaming(args.query, args.model)
        else:
            await run_query(args.query, args.model, args.format)
    else:
        await interactive_mode(args.stream, args.model, args.format)


if __name__ == "__main__":
    asyncio.run(main())
