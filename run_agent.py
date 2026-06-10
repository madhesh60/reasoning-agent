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

# ── Fix Windows console encoding to support Unicode output ─────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Bootstrap path & Configuration ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Suppress Warnings & Noisy Logs ────────────────────────────────────────────
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
import warnings
warnings.filterwarnings("ignore")

from src.utils.logging import configure_logging
configure_logging(log_level="WARNING", json_format=False)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)

# ── Rich UI Helpers ───────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.status import Status
from rich.prompt import Prompt

console = Console()

def hdr(text: str, color: str = "cyan"):
    console.print(Panel(f"[bold {color}]{text}[/]", expand=False, border_style=color))

def ok(msg: str):   console.print(f"  [bold green]✓[/] {msg}")
def warn(msg: str): console.print(f"  [bold yellow]⚠[/] {msg}")
def err(msg: str):  console.print(f"  [bold red]✗[/] {msg}")
def info(msg: str): console.print(f"  [dim]→[/] {msg}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Phi-4 Reasoning Multi-Agent — Research-to-Report Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    hdr("Azure AI Foundry — Model Connection Test", "cyan")

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deploy   = model_override or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "phi-4-mini-reasoning")

    if not endpoint or not api_key:
        err("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set in .env")
        return False

    if endpoint.endswith("/chat/completions"):
        endpoint = endpoint[:-len("/chat/completions")]

    info(f"Connecting to endpoint: {endpoint}")
    info(f"Using deployment: {deploy}")

    from openai import OpenAI
    client = OpenAI(base_url=endpoint, api_key=api_key)

    messages = [
        {"role": "system", "content": "You are a concise reasoning assistant."},
        {"role": "user", "content": "What are the top 3 investment risks in the Indian EV market? Give a short answer."},
    ]

    with console.status("[bold yellow]Sending test prompt...", spinner="dots"):
        try:
            completion = client.chat.completions.create(
                model=deploy,
                messages=messages,
                max_tokens=900,
            )
            msg = completion.choices[0].message
            console.print(Panel(msg.content, title="Model Response", border_style="green"))
            ok(f"Finish reason : {completion.choices[0].finish_reason}")
            ok(f"Total tokens  : {completion.usage.total_tokens if completion.usage else 'N/A'}")
            return True
        except Exception as e:
            err(f"Model call failed: {e}")
            return False

# ── MCP web-search test ───────────────────────────────────────────────────────
async def run_search_test():
    hdr("MCP / Azure Foundry — Web Search Tool Test", "magenta")

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
    info(f"Search query: {query}")

    with console.status("[bold yellow]Searching the web...", spinner="dots"):
        response = await tool.search(query, max_results=5)

    console.print(Panel(
        f"[bold]Query:[/] {response.query}\n"
        f"[bold]Results found:[/] {response.total_results}\n"
        f"[bold]Execution time:[/] {response.execution_time_ms:.0f} ms",
        title="Search Results", border_style="blue"
    ))
    
    for i, r in enumerate(response.results, 1):
        console.print(f"\n  [bold cyan]{i}. {r.title}[/]")
        console.print(f"     [dim]Source:[/] {r.source}")
        console.print(f"     [dim]URL:[/] {r.url}")
        snippet = r.snippet[:120].replace("\n", " ")
        console.print(f"     [dim]Snippet:[/] {snippet}…")
    
    console.print("")
    await tool.close()

# ── Full pipeline ─────────────────────────────────────────────────────────────
async def execute_workflow(query: str, model_override: str | None = None, stream: bool = True, session_id: str | None = None):
    hdr("Research Agent", "green")
    console.print(f"[bold]Query:[/] {query}")

    if model_override:
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = model_override

    from src.utils.config import load_environment
    load_environment()

    from src.orchestration.research_workflow import ResearchWorkflow
    workflow = ResearchWorkflow(enable_a2a=True, enable_mcp=True, max_retries=2)

    STAGE_LABELS = {
        "plan":          "Planner: Decomposing query into sub-tasks...",
        "validate_plan": "Validator: Checking plan quality...",
        "research":      "Researcher: Executing web searches...",
        "analyze":       "Analyst: Extracting insights & risks...",
        "write_report":  "Writer: Generating structured report...",
    }

    if stream:
        status = Status("[bold green]Initializing workflow...", spinner="dots")
        status.start()
        try:
            async for update in workflow.execute_streaming(query, session_id=session_id):
                stage  = update.get("stage", "unknown")
                state_status = update.get("status", "")
                
                if state_status == "in_progress":
                    label = STAGE_LABELS.get(stage, f"Processing {stage}...")
                    status.update(f"[bold cyan]{label}")
                elif state_status == "completed" and stage != "complete":
                    if stage == "validate_plan":
                        status.stop()
                        state_data = update.get("state", {})
                        plan = state_data.get("plan")
                        if plan:
                            console.print(Panel(
                                "\n".join(f"[bold cyan]{i+1}.[/] {t.description} [dim]({t.agent})[/]" for i, t in enumerate(plan.tasks)),
                                title="[bold magenta]Proposed Research Plan[/]",
                                border_style="magenta"
                            ))
                        
                        console.print("\n[bold yellow]Allow running this research plan?[/]")
                        console.print("  [green]1.[/] Yes, allow this time")
                        console.print("  [red]2.[/] No (abort workflow)")
                        
                        choice = Prompt.ask("\nSelect option", choices=["1", "2"], default="1")
                        if choice == "2":
                            console.print("[dim]Workflow aborted by user.[/]")
                            break
                        
                        console.print("")
                        status.start()
                elif stage == "complete":
                    status.stop()
                    state = update.get("result", {})
                    _print_report(query, state)
                elif state_status == "error":
                    status.stop()
                    for e in update.get("errors", []):
                        err(str(e))
        finally:
            status.stop()
    else:
        with console.status("[bold cyan]Executing workflow... (this may take a minute)", spinner="dots"):
            t0 = datetime.utcnow()
            result = await workflow.execute(query, session_id=session_id)
            elapsed = (datetime.utcnow() - t0).total_seconds()
        _print_result(query, result, elapsed)

# ── Pretty-print helpers ──────────────────────────────────────────────────────
def _print_result(query: str, result: dict, elapsed: float):
    status = result.get("status", "unknown")
    meta   = result.get("metadata", {})

    if "completed" in status:
        ok(f"Workflow completed in {elapsed:.1f}s")
    else:
        err(f"Workflow failed after {elapsed:.1f}s")

    errs = meta.get("errors", [])
    if errs:
        for e in errs:
            warn(str(e))

    report = result.get("report")
    if report:
        _print_report_object(report)
    else:
        err("No report generated")

def _print_report(query: str, state: dict):
    report = state.get("report")
    if report:
        _print_report_object(report)
    else:
        warn("No report in final state")

def _print_report_object(report):
    md_lines = [
        f"# {report.metadata.title}",
        f"**Confidence:** {report.metadata.confidence_score:.0%}  |  **Report ID:** `{report.metadata.report_id}`",
        "---",
        "## Executive Summary",
        report.executive_summary,
    ]
    for s in report.sections:
        md_lines.append(f"## {s.title}")
        md_lines.append(s.content)
    
    if report.conclusions:
        md_lines.append("## Conclusions")
        for c in report.conclusions:
            md_lines.append(f"- {c}")
            
    if report.recommendations:
        md_lines.append("## Recommendations")
        for r in report.recommendations:
            md_lines.append(f"- {r}")
            
    if report.citations:
        md_lines.append("## References")
        for c in report.citations[:6]:
            md_lines.append(f"- [{c.get('title', 'Link')}]({c.get('url', '#')}) - {c.get('source', '')}")

    md_content = "\n\n".join(md_lines)
    console.print("\n")
    console.print(Panel(Markdown(md_content), border_style="green", padding=(1, 2)))
    
    # Save markdown to file
    _save_report(report, md_content)

def _save_report(report, md_content):
    try:
        fname = ROOT / f"report_{report.metadata.report_id}.md"
        fname.write_text(md_content, encoding="utf-8")
        ok(f"Report saved to {fname.name}")
    except Exception as e:
        warn(f"Could not save report: {e}")

# ── Interactive mode ──────────────────────────────────────────────────────────
async def interactive_mode(stream: bool, model_override: str | None):
    hdr("Business & Investment Research Assistant", "blue")
    console.print("[dim]Type your research question, or 'exit' to quit.[/]")
    
    import uuid
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Query[/]")
            query = query.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/]")
            break
        if query.lower() == "model-test":
            await run_model_test(model_override)
            continue
        if query.lower() == "search-test":
            await run_search_test()
            continue

        try:
            await execute_workflow(query, model_override, stream=stream, session_id=session_id)
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
        await execute_workflow(args.query, args.model, stream=args.stream)
    else:
        await interactive_mode(args.stream, args.model)


if __name__ == "__main__":
    asyncio.run(main())
