"""
main.py — FastAPI Server for the Reasoning-to-Report Multi-Agent System
=======================================================================
Connects our 4-agent pipeline (Planner → Researcher → Analyst → Writer)
with the Azure AI Foundry competitive-landscape-researcher agent via the
AZURE_EXISTING_AGENT_ID and AZURE_EXISTING_AIPROJECT_ENDPOINT in .env.

Hackathon demo endpoints:
  POST /ask          — run the full 4-agent pipeline
  POST /competitive  — run the Azure Foundry competitive agent
  POST /research     — combined: our agents + competitive intelligence
  GET  /health       — liveness probe

Run locally:
  uvicorn main:app --reload --port 8000

Open docs:
  http://localhost:8000/docs
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Reasoning-to-Report Multi-Agent API",
    description="""
Microsoft Agent League Hackathon — Research-to-Report Multi-Agent System.

Uses Phi-4-mini-reasoning on Azure AI Foundry with:
- **LangGraph** orchestration (Planner → Researcher → Analyst → Writer)
- **A2A Protocol** for inter-agent communication
- **MCP Toolbox** for Bing web search
- **Azure AI Foundry** competitive-landscape agent
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    max_retries: int = 2
    enable_web_search: bool = True


class CompetitiveRequest(BaseModel):
    query: str
    company: str = ""


class ResearchRequest(BaseModel):
    query: str
    include_competitive: bool = True


class AgentResponse(BaseModel):
    status: str
    query: str
    report: dict
    reasoning_steps: list[str] = []
    confidence: float = 0.0
    execution_time_seconds: float = 0.0
    agents_used: list[str] = []
    timestamp: str = ""


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """Liveness probe — returns 200 if the API is running."""
    return {
        "status": "healthy",
        "model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "phi-4-mini-reasoning"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "")[:60] + "...",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Main Pipeline: POST /ask ─────────────────────────────────────────────────
@app.post("/ask", response_model=AgentResponse, tags=["Research Pipeline"])
async def ask(request: AskRequest):
    """
    Run the full 4-agent research pipeline.

    Flow: Planner → Researcher → Analyst → Writer

    Returns a structured report with reasoning steps, confidence score,
    executive summary, and detailed sections.
    """
    t0 = datetime.utcnow()
    logger.info("api_ask_start", query=request.query[:100])

    try:
        from src.utils.config import load_environment
        load_environment()

        from src.orchestration.research_workflow import ResearchWorkflow
        workflow = ResearchWorkflow(
            enable_a2a=True,
            enable_mcp=request.enable_web_search,
            max_retries=request.max_retries
        )

        result = await workflow.execute(request.query)
        elapsed = (datetime.utcnow() - t0).total_seconds()

        # Extract report data
        report_obj = result.get("report")
        report_dict = {}
        if report_obj:
            report_dict = {
                "title": report_obj.metadata.title if report_obj.metadata else request.query,
                "executive_summary": report_obj.executive_summary,
                "sections": [
                    {"title": s.title, "content": s.content[:500]}
                    for s in report_obj.sections[:5]
                ],
                "conclusions": report_obj.conclusions[:5],
                "recommendations": report_obj.recommendations[:3],
                "confidence": report_obj.metadata.confidence_score if report_obj.metadata else 0.5,
            }
        else:
            report_dict = {
                "title": request.query,
                "executive_summary": result.get("summary", "Research completed."),
                "sections": [],
                "conclusions": [],
                "recommendations": [],
            }

        meta = result.get("metadata", {})
        return AgentResponse(
            status=result.get("status", "completed"),
            query=request.query,
            report=report_dict,
            reasoning_steps=meta.get("completed_tasks", []),
            confidence=result.get("confidence_score", 0.7),
            execution_time_seconds=elapsed,
            agents_used=["Planner", "Researcher", "Analyst", "Writer"],
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logger.error("api_ask_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ── Competitive Analysis: POST /competitive ───────────────────────────────────
@app.post("/competitive", tags=["Competitive Intelligence"])
async def competitive_analysis(request: CompetitiveRequest):
    """
    Call the Azure AI Foundry 'competitive-landscape-researcher' agent
    (AZURE_EXISTING_AGENT_ID from .env) to get competitive intelligence.

    This uses the pre-deployed agent from Azure AI Foundry directly.
    """
    t0 = datetime.utcnow()
    logger.info("api_competitive_start", query=request.query[:100])

    agent_id = os.environ.get("AZURE_EXISTING_AGENT_ID", "")
    project_endpoint = os.environ.get("AZURE_PROJECT_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

    if not agent_id or not project_endpoint:
        raise HTTPException(
            status_code=503,
            detail="AZURE_EXISTING_AGENT_ID or AZURE_PROJECT_ENDPOINT not configured in .env"
        )

    try:
        from azure.ai.projects import AIProjectClient
        from azure.core.credentials import AzureKeyCredential

        client = AIProjectClient(
            endpoint=project_endpoint,
            credential=AzureKeyCredential(api_key),
        )

        # Parse agent_id (format: "name:version")
        agent_name = agent_id.split(":")[0] if ":" in agent_id else agent_id

        query = request.query
        if request.company:
            query = f"Analyze competitive landscape for {request.company}: {request.query}"

        # Create a thread and run the agent
        thread = client.agents.create_thread()
        client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=query,
        )

        run = client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent_name,
        )

        # Get messages
        messages = client.agents.list_messages(thread_id=thread.id)
        answer = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        answer = block.text.value
                        break
                if answer:
                    break

        elapsed = (datetime.utcnow() - t0).total_seconds()
        return {
            "status": "completed",
            "query": request.query,
            "agent_id": agent_id,
            "answer": answer or "No response from competitive agent.",
            "execution_time_seconds": elapsed,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("api_competitive_error", error=str(e))
        # Fallback: run our own analyst agent on the query
        logger.info("api_competitive_fallback", using="local_analyst")
        try:
            from src.utils.config import load_environment
            load_environment()
            from src.agents.analyst import AnalystAgent
            analyst = AnalystAgent()
            result = await analyst.analyze(
                request.query,
                {"sources": [], "search_results": []}
            )
            elapsed = (datetime.utcnow() - t0).total_seconds()
            return {
                "status": "completed_via_fallback",
                "query": request.query,
                "agent_id": "local_analyst",
                "answer": "; ".join([f.statement for f in result.key_findings[:3]]) or "Analysis completed.",
                "confidence": result.overall_confidence,
                "execution_time_seconds": elapsed,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Both competitive agent and fallback failed: {str(e2)}")


# ── Combined Research: POST /research ────────────────────────────────────────
@app.post("/research", tags=["Research Pipeline"])
async def combined_research(request: ResearchRequest):
    """
    Run our 4-agent pipeline + competitive intelligence in parallel.
    Returns a merged report with both deep research and market positioning.

    This is the FLAGSHIP endpoint for the hackathon demo.
    """
    t0 = datetime.utcnow()
    logger.info("api_research_start", query=request.query[:100])

    # Run pipeline (always) and competitive (if enabled) concurrently
    pipeline_task = asyncio.create_task(_run_pipeline(request.query))

    if request.include_competitive:
        competitive_task = asyncio.create_task(
            _run_competitive(request.query)
        )
        pipeline_result, competitive_result = await asyncio.gather(
            pipeline_task, competitive_task, return_exceptions=True
        )
    else:
        pipeline_result = await pipeline_task
        competitive_result = None

    elapsed = (datetime.utcnow() - t0).total_seconds()

    # Merge results
    report = pipeline_result if isinstance(pipeline_result, dict) else {"error": str(pipeline_result)}
    competitive = competitive_result if isinstance(competitive_result, dict) else {}

    return {
        "status": "completed",
        "query": request.query,
        "execution_time_seconds": elapsed,
        "research_report": report,
        "competitive_intelligence": competitive,
        "timestamp": datetime.utcnow().isoformat(),
        "agents_used": ["Planner", "Researcher", "Analyst", "Writer", "CompetitiveLandscapeResearcher"],
    }


async def _run_pipeline(query: str) -> dict:
    """Helper: run the 4-agent pipeline and return a dict."""
    try:
        from src.utils.config import load_environment
        load_environment()
        from src.orchestration.research_workflow import ResearchWorkflow
        workflow = ResearchWorkflow(enable_a2a=True, enable_mcp=True, max_retries=2)
        result = await workflow.execute(query)
        report_obj = result.get("report")
        if report_obj:
            return {
                "title": report_obj.metadata.title if report_obj.metadata else query,
                "executive_summary": report_obj.executive_summary,
                "conclusions": report_obj.conclusions[:5],
                "recommendations": report_obj.recommendations[:3],
                "confidence": report_obj.metadata.confidence_score if report_obj.metadata else 0.5,
            }
        return {"title": query, "executive_summary": result.get("summary", ""), "conclusions": []}
    except Exception as e:
        return {"error": str(e), "title": query}


async def _run_competitive(query: str) -> dict:
    """Helper: run competitive analysis and return a dict."""
    agent_id = os.environ.get("AZURE_EXISTING_AGENT_ID", "")
    project_endpoint = os.environ.get("AZURE_PROJECT_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not agent_id or not project_endpoint:
        return {"status": "skipped", "reason": "AZURE_EXISTING_AGENT_ID or AZURE_PROJECT_ENDPOINT not set"}
    try:
        from azure.ai.projects import AIProjectClient
        from azure.core.credentials import AzureKeyCredential
        client = AIProjectClient(endpoint=project_endpoint, credential=AzureKeyCredential(api_key))
        agent_name = agent_id.split(":")[0] if ":" in agent_id else agent_id
        thread = client.agents.create_thread()
        client.agents.create_message(thread_id=thread.id, role="user", content=f"Competitive analysis: {query}")
        client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent_name)
        messages = client.agents.list_messages(thread_id=thread.id)
        answer = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        answer = block.text.value
                        break
                if answer:
                    break
        return {"agent_id": agent_id, "analysis": answer or "No competitive data."}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("A2A_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
