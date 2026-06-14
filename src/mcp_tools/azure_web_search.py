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
from urllib.parse import urlparse as _up

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

# ── Configuration — all read from .env ──────────────────────────────────────
ENDPOINT         = os.environ["AZURE_PROJECT_ENDPOINT"]        # .../projects/<name>
TOOLBOX_NAME     = os.environ.get("AZURE_TOOLBOX_NAME", "reasoning-agent-web-search")
TOOLBOX_VERSION  = os.environ.get("AZURE_TOOLBOX_VERSION", "1")
MODEL_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
API_KEY          = os.environ["AZURE_OPENAI_API_KEY"]

# Derive the OpenAI-compatible endpoint from the project endpoint
# e.g. https://<resource>.services.ai.azure.com/api/projects/<name>
#   -> https://<resource>.services.ai.azure.com/openai/v1

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

    import httpx
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI

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
            "params": {"name": "web_search", "arguments": {"search_query": query}}
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

    class CustomReActAgent:
        def __init__(self, llm, tools):
            self.llm = llm
            self.tools = {t.name: t for t in tools}

        async def ainvoke(self, input_dict: dict, config: dict | None = None) -> dict:
            import uuid

            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

            from src.utils.config import clean_and_parse_json

            raw_messages = input_dict.get("messages", [])
            messages = []
            for msg in raw_messages:
                if isinstance(msg, tuple):
                    role, content = msg
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                else:
                    messages.append(msg)

            from datetime import datetime
            today_str = datetime.now().strftime("%Y-%m-%d")

            system_prompt = (
                "You are a helpful assistant with access to the following tools:\n"
                + "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
                + f"\n\nToday's date is {today_str}. If the user asks you to search the web, gather information, or asks about recent/current events, "
                "you MUST call the 'bing_web_search' tool first to retrieve the facts. Do not answer from your pre-trained knowledge directly without searching.\n\n"
                "To use a tool, you MUST respond in this format (wrapped in a ```json code block):\n"
                "```json\n"
                "{\n"
                '  "action": "<tool_name>",\n'
                '  "action_input": "<your tool input here>"\n'
                "}\n"
                "```\n\n"
                "When you have the final answer to the user's question (after calling tools if needed), you MUST respond in this format (wrapped in a ```json code block):\n"
                "```json\n"
                "{\n"
                '  "action": "final_answer",\n'
                '  "action_input": "<your final answer here>"\n'
                "}\n"
                "```\n\n"
                "Always respond with one of these two JSON formats. Do not output anything else outside the json block except your <think> reasoning."
            )

            run_messages = [SystemMessage(content=system_prompt)] + messages

            for _ in range(5):
                response = await self.llm.ainvoke(run_messages)
                content = response.content

                try:
                    parsed = clean_and_parse_json(content)
                except Exception:
                    ai_msg = AIMessage(content=content)
                    run_messages.append(ai_msg)
                    break

                if not isinstance(parsed, dict):
                    ai_msg = AIMessage(content=str(parsed))
                    run_messages.append(ai_msg)
                    break

                action = parsed.get("action")
                action_input = parsed.get("action_input")

                if action == "final_answer" or not action:
                    ai_msg = AIMessage(content=str(action_input or content))
                    run_messages.append(ai_msg)
                    break

                if action in self.tools:
                    tool = self.tools[action]
                    call_id = f"call_{uuid.uuid4().hex[:8]}"

                    ai_msg = AIMessage(
                        content=content,
                        tool_calls=[{
                            "name": action,
                            "args": {"query": action_input} if isinstance(action_input, str) else action_input,
                            "id": call_id
                        }]
                    )
                    run_messages.append(ai_msg)

                    # Execute the tool via its coroutine
                    tool_result = await tool.coroutine(action_input)

                    tool_msg = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=call_id,
                        name=action
                    )
                    run_messages.append(tool_msg)
                else:
                    error_msg = f"Error: Tool '{action}' is not available. Available tools: {list(self.tools.keys())}."
                    ai_msg = AIMessage(content=content)
                    run_messages.append(ai_msg)

                    tool_msg = ToolMessage(
                        content=error_msg,
                        tool_call_id=f"call_{uuid.uuid4().hex[:8]}",
                        name=action
                    )
                    run_messages.append(tool_msg)

            return {"messages": run_messages[1:]}

    _agent = CustomReActAgent(llm, tools)
    return _agent



async def call_agent_with_toolbox(query: str) -> str:
    """
    Send *query* to the toolbox agent and return the final response string.
    Prints each step to the terminal so you can watch the reasoning.
    """
    if _agent is None:
        raise RuntimeError("call create_agent_with_toolbox() first")

    print(f"\n[agent] Query: {query}")
    print("[agent] Thinking...")

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
    import sys
    print("=" * 64)
    print("Azure AI Foundry — MCP Toolbox Web-Search Demo")
    print("=" * 64)

    await create_agent_with_toolbox()

    # Check if a query was passed as command-line arguments
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer = await call_agent_with_toolbox(query)
        print("\n[agent] Answer:")
        print(answer)
        print("=" * 64)
    else:
        # Interactive Mode
        print("Entering interactive mode. Type 'exit' or 'quit' to end.")
        while True:
            try:
                # Use standard print/input for compatibility
                print("\nAsk the web-search agent a question: ", end="", flush=True)
                query = sys.stdin.readline().strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit"):
                    print("Exiting web-search demo.")
                    break
                answer = await call_agent_with_toolbox(query)
                print("\n[agent] Answer:")
                print(answer)
                print("-" * 64)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting web-search demo.")
                break


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to python path so 'src' can be imported when running standalone
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    asyncio.run(main())
