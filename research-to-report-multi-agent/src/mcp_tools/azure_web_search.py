# pip install azure-identity httpx langchain-azure-ai==1.2.3 langchain-mcp-adapters==0.2.2 langchain-openai==1.1.13 langgraph==1.1.6
import asyncio

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from langchain_azure_ai.tools import AzureAIProjectToolbox
from langgraph.prebuilt import create_react_agent

# ── Configuration ─────────────────────────────────────────────────────────────

endpoint = "https://reasoning-agent-hack2-resource.services.ai.azure.com/api/projects/reasoning-agent-hack2"
toolbox_name = "reasoning-agent-web-search"
toolbox_version = "1"
model_deployment = "<your-model-deployment-name>"

from urllib.parse import urlparse
_parsed = urlparse(endpoint)
azure_openai_endpoint = f"{_parsed.scheme}://{_parsed.netloc}"

# ── Reusable functions (can be pulled into a hosted agent main.py) ────────────

# [START langgraph_toolbox]
_agent = None

async def create_agent_with_toolbox():
    """Create a LangGraph ReAct agent wired to a Foundry toolbox."""
    global _agent

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )

    llm = AzureChatOpenAI(
        azure_endpoint=azure_openai_endpoint,
        azure_deployment=model_deployment,
        azure_ad_token_provider=token_provider,
        api_version="2025-03-01-preview",
    )

    toolbox = AzureAIProjectToolbox(
        project_endpoint=endpoint,
        toolbox_name=toolbox_name,
        toolbox_version=toolbox_version,
        credential=DefaultAzureCredential(),
    )
    tools = await toolbox.get_tools()

    for t in tools:
        t.handle_tool_error = True

    _agent = create_react_agent(llm, tools)


async def call_agent_with_toolbox(input_messages):
    """Send a message to the toolbox agent and print the response."""
    result = await _agent.ainvoke(input_messages)
    print(result["messages"][-1].content)
# [END langgraph_toolbox]


# ── Script entry point ────────────────────────────────────────────────────
async def main():
    await create_agent_with_toolbox()
    await call_agent_with_toolbox({"messages": [("user", "What tools are available?")]})

asyncio.run(main())
