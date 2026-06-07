"""
Azure AI Foundry — MCP Toolbox Web-Search Agent
================================================
Wires a LangGraph ReAct agent to the Bing web-search toolbox deployed in
your Azure AI Foundry project.  Uses API-key auth instead of DefaultAzureCredential
so no 'az login' is needed.

Run standalone:
    python -m src.mcp_tools.azure_web_search

Or import create_agent_with_toolbox / call_agent_with_toolbox from elsewhere.
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

# ── Configuration — all read from .env ──────────────────────────────────────
ENDPOINT         = os.environ["AZURE_PROJECT_ENDPOINT"]        # .../projects/<name>
TOOLBOX_NAME     = os.environ.get("AZURE_TOOLBOX_NAME", "reasoning-agent-web-search")
TOOLBOX_VERSION  = os.environ.get("AZURE_TOOLBOX_VERSION", "1")
MODEL_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "phi-4-mini-reasoning")
API_KEY          = os.environ["AZURE_OPENAI_API_KEY"]

# Derive the OpenAI-compatible endpoint from the project endpoint
# e.g. https://<resource>.services.ai.azure.com/api/projects/<name>
#   -> https://<resource>.services.ai.azure.com/openai/v1
from urllib.parse import urlparse as _up
_p = _up(ENDPOINT)
AZURE_OPENAI_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    f"{_p.scheme}://{_p.netloc}/openai/v1",
)
if AZURE_OPENAI_ENDPOINT.endswith("/chat/completions"):
    AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT[:-len("/chat/completions")]

# ── Agent singleton ──────────────────────────────────────────────────────────
_agent = None


async def create_agent_with_toolbox():
    """Create a LangGraph ReAct agent wired to the Foundry Bing toolbox via MCP."""
    global _agent

    from langchain_openai import ChatOpenAI
    from langchain_core.tools import Tool
    import httpx

    print(f"[toolbox] Connecting to project  : {ENDPOINT}")
    print(f"[toolbox] Toolbox                : {TOOLBOX_NAME} v{TOOLBOX_VERSION}")
    print(f"[toolbox] Model deployment       : {MODEL_DEPLOYMENT}")

    llm = ChatOpenAI(
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key=API_KEY,
        model=MODEL_DEPLOYMENT,
        temperature=0.3,
        timeout=120.0,
        max_tokens=2000,
    )

    # ── Build a web-search tool via the MCP toolbox endpoint ────────────────
    mcp_url   = os.environ.get("MCP_SERVER_URL", "")
    mcp_token = os.environ.get("MCP_AUTH_TOKEN", API_KEY)

    async def _mcp_web_search(query: str) -> str:
        """Call the Foundry MCP web-search toolbox."""
        if not mcp_url:
            return f"[web_search] MCP_SERVER_URL not set. Query was: {query}"
        headers = {"Authorization": f"Bearer {mcp_token}", "Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "bing_search", "arguments": {"query": query, "count": 5}}
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(mcp_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("result", {}).get("content", [])
                if isinstance(results, list):
                    return "\n".join(
                        r.get("text", str(r))[:500] for r in results[:5]
                    )
                return str(results)[:1000]
        except Exception as e:
            return f"[web_search error] {e}. Query: {query}"

    web_search_tool = Tool(
        name="bing_web_search",
        description="Search the web using Bing via Azure AI Foundry MCP toolbox. Input: search query string.",
        func=lambda q: asyncio.get_event_loop().run_until_complete(_mcp_web_search(q)),
        coroutine=_mcp_web_search,
    )
    tools = [web_search_tool]
    print(f"[toolbox] Tools loaded           : {[t.name for t in tools]}")

    from langgraph.prebuilt import create_react_agent
    _agent = create_react_agent(llm, tools)
    return _agent



async def call_agent_with_toolbox(query: str) -> str:
    """
    Send *query* to the toolbox agent and return the final response string.
    Prints each step to the terminal so you can watch the reasoning.
    """
    if _agent is None:
        raise RuntimeError("call create_agent_with_toolbox() first")

    print(f"\n[agent] Query: {query}")
    print("[agent] Thinking (may take 10-30 s for phi-4-reasoning)...")

    result = await _agent.ainvoke({"messages": [("user", query)]})

    # Print intermediate tool calls
    for msg in result["messages"][:-1]:
        role = getattr(msg, "type", type(msg).__name__)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [tool_call] {tc['name']}({tc.get('args', {})})")
        elif role == "tool":
            content_preview = str(msg.content)[:200]
            print(f"  [tool_result] {content_preview}...")

    final = result["messages"][-1].content
    return final


# ── Standalone demo ──────────────────────────────────────────────────────────
async def main():
    print("=" * 64)
    print("Azure AI Foundry — MCP Toolbox Web-Search Demo")
    print("=" * 64)

    await create_agent_with_toolbox()

    questions = [
        "What tools are available in this toolbox?",
        "Search the web: What are the latest EV policy updates in India 2024?",
    ]

    for q in questions:
        answer = await call_agent_with_toolbox(q)
        print("\n[agent] Answer:")
        print(answer)
        print("-" * 64)


if __name__ == "__main__":
    asyncio.run(main())
