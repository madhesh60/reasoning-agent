"""
logos/cli.py — LOGOS command-line interface
============================================

The entry point for the `logos` command.

Usage:
    logos                          # interactive session
    logos -q "your query"          # single-shot query
    logos --no-a2a -q "..."        # local model only, no Foundry agents
    logos --model-test             # verify model connection

This CLI is story-driven: it guides the user through a research investigation
the way an intelligence analyst briefs a client — methodical, direct, no noise.
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Windows UTF-8 ─────────────────────────────────────────────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Resolve project root (for local development) ──────────────────────────────
_CLI_DIR = Path(__file__).resolve().parent          # logos/
_ROOT    = _CLI_DIR.parent                          # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load .env — CWD first, then ~/.logos/.env ─────────────────────────────────
from dotenv import load_dotenv

_LOGOS_HOME = Path.home() / ".logos"
_LOGOS_HOME.mkdir(parents=True, exist_ok=True)

_loaded = load_dotenv(Path.cwd() / ".env", override=False)
if not _loaded:
    load_dotenv(_LOGOS_HOME / ".env", override=False)

# ── Silence noisy loggers ─────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

# ── Rich UI ───────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console(highlight=False)

# Restrained colour palette — no emojis, no colour overload
C_RULE    = "grey46"
C_TITLE   = "bold white"
C_ACCENT  = "steel_blue1"
C_DIM     = "grey46"
C_OK      = "green3"
C_WARN    = "yellow3"
C_ERR     = "red3"
C_LABEL   = "white"
C_BORDER  = "grey30"

# ── Imports (after path setup) ────────────────────────────────────────────────
from logos.memory.store import MemoryStore
from logos.hitl.questioner import generate_questions

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rule(title: str = "") -> None:
    if title:
        console.print(Rule(f"[{C_DIM}]{title}[/]", style=C_RULE))
    else:
        console.print(Rule(style=C_RULE))


def _print(msg: str, indent: int = 2) -> None:
    console.print(" " * indent + msg)


def _blank() -> None:
    console.print()


def _ok(msg: str)   -> None: _print(f"[{C_OK}]{msg}[/]")
def _warn(msg: str) -> None: _print(f"[{C_WARN}]{msg}[/]")
def _err(msg: str)  -> None: _print(f"[{C_ERR}]{msg}[/]")
def _dim(msg: str)  -> None: _print(f"[{C_DIM}]{msg}[/]")


def _ask(prompt_text: str) -> str:
    """Prompt for a single line of input. Returns stripped string."""
    console.print()
    try:
        val = console.input(f"  [{C_ACCENT}]{prompt_text}[/]  ")
        return val.strip()
    except (KeyboardInterrupt, EOFError):
        raise


# ── Banner ────────────────────────────────────────────────────────────────────

def _banner() -> None:
    _blank()
    console.print(f"  [{C_TITLE}]L O G O S[/]   [{C_DIM}]— Autonomous Research Intelligence[/]")
    _print(f"[{C_DIM}]{'─' * 54}[/]")
    _blank()


# ── First-time setup ──────────────────────────────────────────────────────────

async def _first_time_setup(memory: MemoryStore) -> None:
    _blank()
    _print(f"[{C_LABEL}]This is your first session.[/]")
    _blank()
    _dim(
        "A few questions to personalize your experience.\n"
        "  Press Enter to skip any."
    )
    _blank()

    name = _ask("Your name  >")
    role = _ask("Your role  (e.g. analyst, founder, researcher)  >")
    org  = _ask("Organization or project (optional)  >")
    domain = _ask("Primary domain of research  (e.g. AI, fintech, biotech)  >")

    _blank()
    _print(f"[{C_LABEL}]Preferred report depth:[/]")
    _print(f"  [{C_DIM}]1[/]  Concise — executive-level summaries")
    _print(f"  [{C_DIM}]2[/]  Detailed — full analysis with evidence")
    depth_choice = _ask("Choice  [1/2]  >")
    depth = "concise" if depth_choice.strip() == "1" else "detailed"

    kwargs: dict[str, str] = {"depth_preference": depth}
    if name:   kwargs["name"]         = name
    if role:   kwargs["role"]         = role
    if org:    kwargs["organization"] = org
    if domain: kwargs["domain"]       = domain

    memory.set_profile(**kwargs)

    _blank()
    _ok("Profile saved.")
    _dim("LOGOS will remember your preferences across sessions.")
    _blank()
    _rule()


# ── Session greeting ──────────────────────────────────────────────────────────

def _greet(memory: MemoryStore) -> None:
    profile    = memory.get_user_profile()
    recent     = memory.get_recent_queries(3)
    total      = memory.total_query_count()
    session_no = memory.session_number()
    entities   = memory.get_tracked_entities(4)
    now_str    = datetime.now().strftime("%A, %d %B %Y  |  %H:%M")

    name = profile.get("name", "")
    greeting_line = f"Session {session_no}  |  {now_str}"
    if name:
        greeting_line += f"  |  Welcome back, {name}"

    _print(f"[{C_DIM}]{greeting_line}[/]")
    _blank()

    if total > 0:
        _dim(f"Your research history spans {total} {'query' if total == 1 else 'queries'}.")

        if entities:
            top = ", ".join(e["name"] for e in entities[:3])
            _dim(f"Frequently investigated: {top}.")

        if recent:
            _dim(f"Last query: \"{recent[0]['query'][:70]}\".")

    _blank()


# ── Human-in-the-loop question flow ──────────────────────────────────────────

async def _run_hitl(query: str, memory: MemoryStore) -> str:
    """
    Ask the user 2-3 clarifying questions before dispatching the research team.
    Returns a formatted string of the Q&A pairs for the agent context.
    """
    _blank()
    _rule("Investigation received")
    _blank()
    _print(f"[{C_LABEL}]Before dispatching the research team, a few clarifying questions.[/]")
    _dim("Your answers will shape the depth and direction of the analysis.")
    _blank()

    memory_ctx = memory.build_context_string() if not memory.is_first_run() else ""

    with console.status(
        f"  [{C_DIM}]Preparing questions...[/]",
        spinner="line",
        spinner_style=C_DIM,
    ):
        questions = await generate_questions(query, memory_ctx, max_questions=3)

    if not questions:
        return ""

    qa_pairs: list[str] = []
    total = len(questions)

    for i, q_text in enumerate(questions, 1):
        _rule(f"Question {i} of {total}")
        _blank()
        _print(f"[{C_LABEL}]{q_text}[/]")
        _blank()

        try:
            answer = _ask(">")
        except (KeyboardInterrupt, EOFError):
            answer = ""

        if answer:
            qa_pairs.append(f"Q: {q_text}\nA: {answer}")
        else:
            _dim("(skipped)")

    _blank()
    _rule()
    _blank()
    _dim("Understood. The research team has been briefed.")
    _dim("Dispatching all six agents now.")
    _blank()

    if not qa_pairs:
        return ""

    return (
        "=== USER CLARIFICATIONS ===\n"
        + "\n\n".join(qa_pairs)
        + "\n=== END CLARIFICATIONS ==="
    )


# ── Pipeline status display ───────────────────────────────────────────────────

_PIPELINE = [
    ("Planner",                "Decomposing the query into research sub-tasks"),
    ("Researcher",             "Fetching data, reports, and source material"),
    ("Industry News Scanner",  "Scanning real-time signals and press"),
    ("Competitive Intel",      "Mapping the competitive landscape"),
    ("Analyst",                "Synthesising findings and assessing risk"),
    ("Writer",                 "Composing the structured report"),
]


def _show_pipeline() -> None:
    t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    t.add_column(justify="right", style=C_DIM,   no_wrap=True, width=4)
    t.add_column(style=C_LABEL,  no_wrap=True,   width=24)
    t.add_column(style=C_DIM)

    for i, (name, desc) in enumerate(_PIPELINE, 1):
        t.add_row(str(i), name, desc)

    console.print(Panel(
        t,
        title=f"[{C_DIM}]Pipeline[/]",
        border_style=C_BORDER,
        padding=(0, 2),
    ))


# ── Report renderer ───────────────────────────────────────────────────────────

def _render_report(report, elapsed: float, path: str) -> None:
    _blank()
    _rule("Report")
    _blank()

    # Stats row
    profile  = report.metadata
    conf     = profile.confidence_score
    conf_col = C_OK if conf >= 0.8 else (C_WARN if conf >= 0.6 else C_ERR)
    _print(
        f"[{C_DIM}]Generated in[/] [{C_LABEL}]{elapsed:.1f}s[/]   "
        f"[{C_DIM}]Confidence[/] [{conf_col}]{conf:.0%}[/]   "
        f"[{C_DIM}]Path[/] [{C_LABEL}]{path.replace('_',' ').replace('foundry ','').title()}[/]   "
        f"[{C_DIM}]ID[/] [{C_DIM}]{profile.report_id}[/]"
    )
    _blank()

    # Executive summary
    if report.executive_summary:
        console.print(Panel(
            Markdown(report.executive_summary),
            title=f"[{C_LABEL}]Executive Summary[/]",
            border_style=C_BORDER,
            padding=(0, 2),
        ))
        _blank()

    # Sections
    for sec in report.sections:
        if sec.content and sec.title:
            console.print(Panel(
                Markdown(sec.content),
                title=f"[{C_LABEL}]{sec.title}[/]",
                border_style=C_BORDER,
                padding=(0, 2),
            ))
            _blank()

    # Conclusions
    if report.conclusions:
        md = "\n".join(f"- {c}" for c in report.conclusions)
        console.print(Panel(
            Markdown(md),
            title=f"[{C_LABEL}]Conclusions[/]",
            border_style=C_BORDER,
            padding=(0, 2),
        ))
        _blank()

    # Recommendations
    if report.recommendations:
        md = "\n".join(f"{i}. {r}" for i, r in enumerate(report.recommendations, 1))
        console.print(Panel(
            Markdown(md),
            title=f"[{C_LABEL}]Strategic Recommendations[/]",
            border_style=C_BORDER,
            padding=(0, 2),
        ))
        _blank()

    # Resources
    if report.citations:
        lines = []
        for c in report.citations[:12]:
            title = c.get("title", "Source")
            url   = c.get("url", "")
            line  = f"- {title}" + (f"  {url}" if url else "")
            lines.append(line)
        console.print(Panel(
            Markdown("\n".join(lines)),
            title=f"[{C_LABEL}]Resources & References[/]",
            border_style=C_BORDER,
            padding=(0, 2),
        ))
        _blank()


# ── Save report to file ───────────────────────────────────────────────────────

def _save_report(report, query: str) -> Path:
    slug  = query[:40].lower().replace(" ", "_").replace("/", "").replace("\\", "")
    fname = Path.cwd() / f"report_{slug}_{report.metadata.report_id}.md"

    lines = [
        f"# {report.metadata.title or query}",
        f"\n**ID:** `{report.metadata.report_id}`  "
        f"| **Confidence:** {report.metadata.confidence_score:.0%}  "
        f"| **Generated:** {report.metadata.generated_at}\n",
        "---\n",
        "## Executive Summary\n",
        report.executive_summary, "\n",
    ]
    for sec in report.sections:
        lines += [f"\n## {sec.title}\n", sec.content, "\n"]
    if report.conclusions:
        lines += ["\n## Conclusions\n"] + [f"- {c}\n" for c in report.conclusions]
    if report.recommendations:
        lines += ["\n## Recommendations\n"] + [
            f"{i}. {r}\n" for i, r in enumerate(report.recommendations, 1)
        ]
    if report.citations:
        lines += ["\n## Resources\n"] + [
            f"- [{c.get('title','Source')}]({c.get('url','#')})\n"
            for c in report.citations[:12]
        ]

    fname.write_text("\n".join(lines), encoding="utf-8")
    return fname


# ── Post-report memory save + bookmark ───────────────────────────────────────

async def _save_to_memory(
    query: str,
    report,
    memory: MemoryStore,
    path_used: str,
) -> int:
    """Save the query and report summary to long-term memory."""
    # Extract topics from section titles + report content
    topics: list[str] = []
    if report.sections:
        topics = [s.title for s in report.sections[:5] if s.title]

    # Extract entity names from tracked entities in report text
    full_text = report.executive_summary + " ".join(
        s.content for s in report.sections
    )
    # Simple heuristic: save any all-caps words >3 chars as potential entities
    import re
    raw_entities = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b(?:\s[A-Z][a-zA-Z]{2,})*", full_text)
    entities = list(dict.fromkeys(raw_entities))[:10]  # deduplicated

    summary = report.executive_summary[:300] if report.executive_summary else query

    query_id = memory.save_query(
        query    = query,
        summary  = summary,
        topics   = topics,
        entities = entities,
        path_used= path_used,
    )

    for e in entities[:6]:
        memory.track_entity(e)

    return query_id


async def _offer_bookmark(query_id: int, report, memory: MemoryStore) -> None:
    """Offer the user a chance to bookmark a key finding."""
    _blank()
    _rule("Memory")
    _blank()
    _print(f"[{C_LABEL}]Report saved to memory.[/]")
    _dim(
        "You can bookmark a key finding for your long-term record.\n"
        "  Type it now, or press Enter to continue."
    )

    try:
        insight = _ask("Bookmark  >")
    except (KeyboardInterrupt, EOFError):
        insight = ""

    if insight:
        memory.save_insight(query_id, insight)
        _ok("Saved to long-term memory.")
    else:
        _dim("Nothing bookmarked.")

    _blank()


# ── Main query execution ──────────────────────────────────────────────────────

async def execute_query(
    query: str,
    memory: MemoryStore,
    enable_a2a: bool = True,
    skip_hitl: bool = False,
) -> None:
    from src.utils.config import load_environment
    load_environment()

    from src.orchestration.research_workflow import ResearchWorkflow

    # Memory context for agents
    memory_ctx = memory.build_context_string() if not memory.is_first_run() else ""

    # Human-in-the-loop
    clarifications = ""
    if not skip_hitl:
        try:
            clarifications = await _run_hitl(query, memory)
        except (KeyboardInterrupt, EOFError):
            _blank()
            _dim("Clarification skipped. Proceeding with the original query.")
            _blank()
    else:
        _blank()

    # Show pipeline
    _show_pipeline()
    _blank()

    # Run
    workflow = ResearchWorkflow(enable_a2a=enable_a2a, max_retries=1)
    t0 = time.time()

    with console.status(
        f"  [{C_DIM}]Research in progress...[/]",
        spinner="line",
        spinner_style=C_DIM,
    ):
        result = await workflow.execute(
            query,
            memory_context=memory_ctx,
            user_clarifications=clarifications,
        )

    elapsed = time.time() - t0
    report  = result.get("report")
    path    = result.get("path_used", "unknown")

    if report and report.executive_summary:
        _ok(f"Complete.  {elapsed:.1f}s elapsed.")
        _render_report(report, elapsed, path)

        # Save report file
        fname = _save_report(report, query)
        _dim(f"Report file: {fname.name}")

        # Save to memory
        query_id = await _save_to_memory(query, report, memory, path)

        # Offer bookmark
        await _offer_bookmark(query_id, report, memory)
    else:
        _err("The research pipeline did not produce a report.")
        _warn("Check your Azure credentials and network connection.")


# ── Interactive REPL ──────────────────────────────────────────────────────────

async def interactive_mode(
    memory: MemoryStore,
    enable_a2a: bool = True,
) -> None:
    _banner()

    # First run setup
    if memory.is_first_run():
        await _first_time_setup(memory)
    else:
        _greet(memory)

    _print(f"[{C_DIM}]State your investigation. Type 'help' for commands, 'exit' to leave.[/]")
    _blank()

    while True:
        try:
            query = _ask(">")
        except (KeyboardInterrupt, EOFError):
            _blank()
            _dim("Session closed.")
            _blank()
            break

        if not query:
            continue

        low = query.lower()

        if low in ("exit", "quit", "q"):
            _blank()
            _dim("Session closed.")
            _blank()
            break

        if low == "help":
            _blank()
            _print(f"[{C_DIM}]Commands:[/]")
            _dim("  Any text          Run a full research investigation")
            _dim("  memory            Show what LOGOS remembers about you")
            _dim("  profile           Update your profile")
            _dim("  insights          View your bookmarked findings")
            _dim("  clear memory      Wipe your memory database")
            _dim("  exit / quit       End the session")
            _blank()
            continue

        if low == "memory":
            _show_memory(memory)
            continue

        if low == "profile":
            await _first_time_setup(memory)
            continue

        if low == "insights":
            _show_insights(memory)
            continue

        if low == "clear memory":
            _blank()
            confirm = _ask("Type 'confirm' to erase all memory  >")
            if confirm.lower() == "confirm":
                memory.db_path.unlink(missing_ok=True)
                memory.initialize()
                _ok("Memory cleared.")
            else:
                _dim("Cancelled.")
            _blank()
            continue

        try:
            await execute_query(query, memory, enable_a2a=enable_a2a)
        except KeyboardInterrupt:
            _blank()
            _warn("Investigation interrupted.")
            _blank()
        except Exception as exc:
            _err(f"Pipeline error: {exc}")
            _blank()

    memory.close()


# ── Memory display commands ───────────────────────────────────────────────────

def _show_memory(memory: MemoryStore) -> None:
    _blank()
    _rule("Memory")
    _blank()

    profile  = memory.get_user_profile()
    recent   = memory.get_recent_queries(5)
    entities = memory.get_tracked_entities(8)

    if profile:
        for k, v in profile.items():
            _print(f"[{C_DIM}]{k:<22}[/] {v}")
        _blank()

    if recent:
        _print(f"[{C_LABEL}]Recent investigations:[/]")
        for q in recent:
            ts = q["created_at"][:10]
            _dim(f"  {ts}  {q['query'][:70]}")
        _blank()

    if entities:
        _print(f"[{C_LABEL}]Frequently researched:[/]")
        for e in entities:
            _dim(f"  {e['name']:<30} mentioned {e['count']}x")
        _blank()

    if not profile and not recent:
        _dim("No memory stored yet.")
        _blank()


def _show_insights(memory: MemoryStore) -> None:
    _blank()
    _rule("Bookmarked Insights")
    _blank()
    insights = memory.get_recent_insights(10)
    if insights:
        for i, ins in enumerate(insights, 1):
            _print(f"[{C_DIM}]{i}.[/]  {ins}")
            _blank()
    else:
        _dim("No insights bookmarked yet.")
        _dim("After a report is generated, you will be offered the chance to bookmark findings.")
    _blank()


# ── Model test ────────────────────────────────────────────────────────────────

async def run_model_test() -> None:
    _banner()
    _rule("Connection Test")
    _blank()

    endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "phi-4-mini-reasoning")
    api_key    = os.environ.get("AZURE_OPENAI_API_KEY", "")

    _dim(f"Endpoint   : {endpoint[:60]}")
    _dim(f"Deployment : {deployment}")
    _blank()

    from openai import AzureOpenAI
    client = AzureOpenAI(base_url=endpoint.rstrip("/"), api_key=api_key)

    with console.status(f"  [{C_DIM}]Pinging model...[/]", spinner="line"):
        try:
            r = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Reply in one sentence about AI in 2025."}],
                max_tokens=100,
            )
            text = r.choices[0].message.content or "(no content)"
            _ok("Model responded.")
            _blank()
            console.print(Panel(text, title="Reply", border_style=C_BORDER, padding=(0, 2)))
        except Exception as exc:
            _err(f"Connection failed: {exc}")

    _blank()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="logos",
        description="LOGOS — Autonomous Research Intelligence Agent",
    )
    p.add_argument("-q", "--query",     help="Run a single research query non-interactively")
    p.add_argument("--no-a2a",          action="store_true", help="Skip Foundry agents, use local model only")
    p.add_argument("--no-hitl",         action="store_true", help="Skip clarifying questions")
    p.add_argument("--model-test",      action="store_true", help="Test model connection and exit")
    p.add_argument("--memory-path",     default=None,        help="Custom path for memory database")
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()

    if args.model_test:
        await run_model_test()
        return

    memory = MemoryStore(db_path=args.memory_path)
    memory.initialize()

    if args.query:
        _banner()
        if memory.is_first_run():
            _dim("First session detected. Run 'logos' interactively to set up your profile.")
            _blank()
        else:
            _greet(memory)
        await execute_query(
            args.query,
            memory,
            enable_a2a=not args.no_a2a,
            skip_hitl=args.no_hitl,
        )
        memory.close()
    else:
        await interactive_mode(memory, enable_a2a=not args.no_a2a)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
