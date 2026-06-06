#!/usr/bin/env python3
"""
Research-to-Report Multi-Agent System
Main Entry Point

This script provides the main CLI interface for the multi-agent system.
It handles command-line arguments and orchestrates the research workflow.

Usage:
    python -m src.main --query "What are the top 3 investment risks in the Indian EV market?"
    python -m src.main --interactive
    python -m src.main --demo
"""

import sys
import asyncio
import argparse
import json
from typing import Any

# Configure logging
from src.utils.logging import configure_logging
from src.utils.config import load_environment

configure_logging(log_level="INFO", json_format=False)

import structlog
logger = structlog.get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Research-to-Report Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a single query
  python -m src.main --query "What are the top 3 investment risks in the Indian EV market?"

  # Run in interactive mode
  python -m src.main --interactive

  # Run demo workflow
  python -m src.main --demo

  # Output as JSON
  python -m src.main --query "..." --format json

  # Save output to file
  python -m src.main --query "..." --output results.json
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Research query to process"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with multiple queries"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the demo workflow with sample data"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown", "html"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save output to file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def run_single_query(query: str, output_format: str) -> dict[str, Any]:
    """Run the workflow for a single query."""
    from src.orchestration.research_workflow import ResearchWorkflow

    logger.info("executing_query", query=query[:100])

    workflow = ResearchWorkflow(
        enable_a2a=True,
        enable_mcp=True,
        max_retries=3
    )

    result = await workflow.execute(query)

    # Format output
    if output_format == "markdown" and result.get("report"):
        from src.agents.writer import ReportFormat
        writer = result["report"].__class__.__name__
        # Get markdown from writer
        result["formatted_output"] = "markdown_output"

    return result


async def run_interactive_mode():
    """Run the system in interactive mode."""
    print("=" * 60)
    print("RESEARCH-TO-REPORT MULTI-AGENT SYSTEM")
    print("Interactive Mode")
    print("=" * 60)
    print("\nType your research query and press Enter.")
    print("Type 'exit' or 'quit' to end the session.\n")

    from src.orchestration.research_workflow import ResearchWorkflow

    workflow = ResearchWorkflow(
        enable_a2a=True,
        enable_mcp=True,
        max_retries=3
    )

    while True:
        try:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            print("\nProcessing...")
            result = await workflow.execute(query)

            print("\n" + "-" * 60)
            print("RESULT")
            print("-" * 60)

            if result.get("status") == "completed":
                print(f"Status: SUCCESS")
                print(f"Confidence: {result.get('confidence_score', 'N/A')}")
                print(f"Processing Time: {result.get('processing_time_seconds', 'N/A')}s")

                if result.get("report"):
                    print(f"\nReport: {result['report'].metadata.title}")
                    print(f"\nExecutive Summary:\n{result['report'].executive_summary[:300]}...")

                if result.get("metadata", {}).get("completed_tasks"):
                    print(f"\nCompleted Tasks: {', '.join(result['metadata']['completed_tasks'])}")
            else:
                print(f"Status: {result.get('status', 'unknown')}")
                if result.get("error"):
                    print(f"Error: {result['error']}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


async def run_demo():
    """Run the demo workflow with sample queries."""
    print("=" * 60)
    print("RESEARCH-TO-REPORT MULTI-AGENT SYSTEM")
    print("Demo Mode")
    print("=" * 60)

    from src.orchestration.research_workflow import ResearchWorkflow

    # Sample queries for demo
    demo_queries = [
        "What are the top 3 investment risks in the Indian EV market?",
        "Analyze the competitive landscape of AI coding assistants",
        "What are the key challenges in adopting renewable energy in data centers?",
    ]

    workflow = ResearchWorkflow(
        enable_a2a=True,
        enable_mcp=True,
        max_retries=3
    )

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"DEMO {i}/3")
        print(f"{'=' * 60}")
        print(f"\nQuery: {query}\n")

        result = await workflow.execute(query)

        print(f"\nStatus: {result.get('status', 'unknown')}")
        print(f"Confidence: {result.get('confidence_score', 'N/A'):.2f}" if result.get('confidence_score') else "")
        print(f"Processing Time: {result.get('processing_time_seconds', 'N/A'):.2f}s" if result.get('processing_time_seconds') else "")

        if result.get("report"):
            print(f"\nReport Title: {result['report'].metadata.title}")
            print(f"\nExecutive Summary:\n{result['report'].executive_summary[:200]}...")

        if i < len(demo_queries):
            print("\n" + "-" * 60)
            print("Press Enter to continue to next demo...")
            input()


async def main_async(args: argparse.Namespace):
    """Main async entry point."""
    if args.verbose:
        configure_logging(log_level="DEBUG", json_format=True)

    # Load environment
    load_environment()

    if args.interactive:
        await run_interactive_mode()
    elif args.demo:
        await run_demo()
    elif args.query:
        result = await run_single_query(args.query, args.format)

        # Output result
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))
    else:
        print("Error: Please provide --query, --interactive, or --demo")
        print("Run 'python -m src.main --help' for usage information")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error("main_error", error=str(e))
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()