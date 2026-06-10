import os
import uuid
import asyncio
import structlog
from typing import Optional, Dict, Any
import json
import json_repair

from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import VersionRefIndicator

logger = structlog.get_logger(__name__)

class FoundryAgentClient:
    """
    Client for calling real Azure Foundry Agents via the Azure AI Projects SDK.
    Uses the Responses API to interact with agents dynamically.
    """
    def __init__(self, agent_name: str, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.agent_name = agent_name
        self.endpoint = endpoint or os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        
        if not self.endpoint or not self.api_key:
            logger.warning("foundry_client_missing_credentials", agent=agent_name)

    async def call_agent_raw(self, prompt: str) -> str:
        """
        Connects to the Azure Foundry agent session, sends a prompt, and returns the raw string response.
        """
        if not self.endpoint or not self.api_key:
            raise ValueError("Missing endpoint or API key for Foundry Agent Client.")
            
        logger.info("calling_foundry_agent_raw", agent=self.agent_name)
        
        loop = asyncio.get_event_loop()
        
        def _sync_call():
            with AIProjectClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
                allow_preview=True,
            ) as project_client:
                isolation_key = f"iso-{uuid.uuid4().hex[:8]}"
                
                # Fetch the agent to determine the latest version
                agent = project_client.agents.get(agent_name=self.agent_name)
                agent_version = None
                
                if hasattr(agent, "versions") and agent.versions:
                    if "latest" in agent.versions:
                        agent_version = agent.versions["latest"].version
                    else:
                        versions = list(agent.versions.values())
                        if versions:
                            agent_version = versions[-1].version
                            
                session_args = {
                    "agent_name": self.agent_name,
                    "isolation_key": isolation_key,
                }
                if agent_version:
                    session_args["version_indicator"] = VersionRefIndicator(agent_version=agent_version)
                    
                session = project_client.beta.agents.create_session(**session_args)
                
                try:
                    openai_client = project_client.get_openai_client(agent_name=self.agent_name)
                    response = openai_client.responses.create(
                        input=prompt,
                        extra_body={
                            "agent_session_id": session.agent_session_id,
                        },
                    )
                    return response.output_text
                finally:
                    project_client.beta.agents.delete_session(
                        agent_name=self.agent_name,
                        session_id=session.agent_session_id,
                        isolation_key=isolation_key,
                    )
                    
        try:
            return await loop.run_in_executor(None, _sync_call)
        except Exception as e:
            logger.error("foundry_agent_call_failed", agent=self.agent_name, error=str(e))
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
