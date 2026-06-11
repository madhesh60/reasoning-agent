# Before running the sample:
#    pip install azure-ai-projects>=2.1.0

import os
import sys
import time
import subprocess
import json
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AccessToken
from dotenv import load_dotenv

# Fix Windows console encoding to support Unicode output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load env variables from .env
load_dotenv()

class AzureCliTokenCredential:
    """
    Credential that invokes Azure CLI to fetch tokens.
    Necessary on Windows when DefaultAzureCredential fails due to Python subprocess
    limitations with .cmd file resolution.
    """
    def get_token(self, *scopes, **kwargs) -> AccessToken:
        resource = "https://management.core.windows.net"
        if scopes:
            scope = scopes[0]
            if scope.endswith("/.default"):
                resource = scope[:-9]
            else:
                resource = scope
        
        # Invoke az account get-access-token for the resource
        cmd = ["az.cmd", "account", "get-access-token", "--resource", resource]
        res = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        expires_on = data.get("expiresOnTime")
        if expires_on is None:
            expires_str = data.get("expiresOn")
            if expires_str:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(expires_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    expires_on = int(dt.timestamp())
                except:
                    expires_on = int(time.time()) + 3600
            else:
                expires_on = int(time.time()) + 3600
        else:
            expires_on = int(expires_on)
            
        return AccessToken(data["accessToken"], expires_on)

# Determine query to run
if len(sys.argv) > 1:
    query = sys.argv[1]
else:
    query = input("Enter your research question/query: ").strip()

if not query:
    print("Error: Query cannot be empty.")
    sys.exit(1)

endpoint = "https://reasoning-agent-hack2-resource.services.ai.azure.com/api/projects/reasoning-agent-hack2"

# Prefer our custom Azure CLI credential to avoid DefaultAzureCredential invocation failures on Windows
credential = AzureCliTokenCredential()

print(f"Connecting to AI Project Client...")
project_client = AIProjectClient(
    endpoint=endpoint,
    credential=credential,
)

with project_client:
    workflow = {
        "name": "investment-research-workflow",
        "version": "4",
    }
    
    openai_client = project_client.get_openai_client()

    print("Creating conversation...")
    conversation = openai_client.conversations.create()
    print(f"Created conversation (id: {conversation.id})")

    print(f"Submitting query: '{query}'")
    stream = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={"agent_reference": {"name": workflow["name"], "version": workflow["version"], "type": "agent_reference"}},
        input=query,
        stream=True,
        metadata={"x-ms-debug-mode-enabled": "1"},
    )

    print("\n--- Streaming Report Output ---\n")
    for event in stream:
        event_type = getattr(event, "type", "")
        
        if event_type == "response.output_text.done":
            text_content = getattr(event, "text", "")
            if text_content:
                print(text_content)
        elif event_type == "response.output_text.delta":
            print(getattr(event, "delta", ""), end="", flush=True)
        elif event_type == "response.output_item.added" and hasattr(event, "item") and event.item.type == "workflow_action":
            print(f"\n[Actor: {event.item.action_id} started]\n")
        elif event_type == "response.output_item.done" and hasattr(event, "item") and event.item.type == "workflow_action":
            print(f"\n[Actor: {event.item.action_id} completed with status: {event.item.status}]\n")
        elif event_type == "response.failed":
            print(f"\nError: Response failed.")
            if hasattr(event, "response") and getattr(event.response, "error", None):
                print(f"Details: {event.response.error.message}")

    print("\n--- Stream Complete ---\n")
    
    print("Deleting conversation...")
    openai_client.conversations.delete(conversation_id=conversation.id)
    print("Conversation deleted.")