import os
import asyncio
import structlog
from typing import Optional, Dict, Any
import json
import json_repair

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

logger = structlog.get_logger(__name__)

class FoundryAgentClient:
    """
    Client for calling real Azure Foundry Agents via the Azure AI Projects SDK.
    Uses the Responses API to interact with agents dynamically using their reference name and version.
    """
    def __init__(self, agent_name: str, agent_version: str, endpoint: Optional[str] = None):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.endpoint = endpoint or os.getenv("AZURE_PROJECT_ENDPOINT") or "https://reasoning-agent-hack2-resource.services.ai.azure.com/api/projects/reasoning-agent-hack2"
        
        if not self.endpoint:
            logger.warning("foundry_client_missing_endpoint", agent=agent_name)

    async def call_agent_raw(self, prompt: str) -> str:
        """
        Connects to the Azure Foundry agent using the agent_reference, sends a prompt, and returns the raw string response.
        """
        if not self.endpoint:
            raise ValueError("Missing endpoint for Foundry Agent Client.")
            
        logger.info("calling_foundry_agent_raw", agent=self.agent_name, version=self.agent_version)
        
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            with AIProjectClient(
                endpoint=self.endpoint,
                credential=DefaultAzureCredential(),
            ) as project_client:
                openai_client = project_client.get_openai_client()
                response = openai_client.responses.create(
                    input=[{"role": "user", "content": prompt}],
                    extra_body={
                        "agent_reference": {
                            "name": self.agent_name,
                            "version": self.agent_version,
                            "type": "agent_reference"
                        }
                    },
                )
                return response.output_text
                    
        try:
            return await loop.run_in_executor(None, _sync_call)
        except Exception as e:
            logger.error("foundry_agent_call_failed", agent=self.agent_name, version=self.agent_version, error=str(e))
            raise e

    async def call_agent_json(self, prompt: str) -> Dict[str, Any]:
        """
        Calls the agent and automatically parses the response into JSON.
        Uses json_repair to handle slightly malformed outputs.
        """
        raw_response = await self.call_agent_raw(prompt)
        logger.info("parsing_foundry_agent_response", agent=self.agent_name)
        try:
            parsed = json_repair.loads(raw_response)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed output is not a dictionary.")
            return parsed
        except Exception as e:
            logger.error("foundry_agent_json_parse_failed", error=str(e), raw_snippet=raw_response[:100])
            raise ValueError(f"Failed to parse agent response as JSON: {e}")

