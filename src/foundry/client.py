"""
Azure AI Foundry Agent Client
==============================
Calls Azure AI Foundry hosted agents using the Responses API
with agent_reference (name + version).

This is the ONLY calling strategy used — it matches exactly what works
in the Foundry portal sample code:

    openai_client.responses.create(
        input=[{"role": "user", "content": "..."}],
        extra_body={
            "agent_reference": {
                "name":    "planner-agent",
                "version": "9",
                "type":    "agent_reference"
            }
        },
    )

All 6 Foundry agents are called this way.
"""

from __future__ import annotations

import os
import asyncio
import structlog
from typing import Any

logger = structlog.get_logger(__name__)

import time
from azure.core.credentials import AccessToken

class CustomApiKeyCredential:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_token(self, *scopes, **kwargs) -> AccessToken:
        return AccessToken(self.api_key, int(time.time()) + 3600)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


class FoundryAgentClient:
    """
    Client for calling a single Azure AI Foundry hosted agent
    via the Responses API (name + version).

    Args:
        agent_name:    The exact name shown in Foundry portal
                       e.g. "planner-agent", "researcher-agent"
        agent_version: The version number shown in Foundry portal
                       e.g. "9", "7", "4"
        agent_id:      Ignored — kept only for backwards compatibility.
    """

    def __init__(
        self,
        agent_name: str,
        agent_version: str,
        agent_id: str | None = None,   # kept for compat, not used
    ) -> None:
        self.agent_name = agent_name
        self.agent_version = agent_version

        # Project endpoint — e.g.
        # https://reasoning-agent-hack2-resource.services.ai.azure.com/api/projects/reasoning-agent-hack2
        self._endpoint: str = os.getenv("AZURE_PROJECT_ENDPOINT", "")
        self._api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")

        logger.info(
            "foundry_client_initialized",
            agent=agent_name,
            version=agent_version,
            endpoint_set=bool(self._endpoint),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call_agent_json(self, prompt: str) -> dict[str, Any]:
        """
        Send a prompt to the Foundry agent, return parsed JSON dict.
        Retries up to _MAX_RETRIES times on transient errors.
        Raises on final failure so the caller falls back to local agent.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "foundry_agent_call_start",
                    agent=self.agent_name,
                    version=self.agent_version,
                    attempt=attempt,
                )
                result = await self._call_responses_api(prompt)
                logger.info(
                    "foundry_agent_call_success",
                    agent=self.agent_name,
                    attempt=attempt,
                )
                return result

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "foundry_agent_call_failed",
                    agent=self.agent_name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_DELAY * attempt)

        logger.error(
            "foundry_agent_all_retries_exhausted",
            agent=self.agent_name,
            error=str(last_error),
        )
        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Responses API call
    # ------------------------------------------------------------------

    async def _call_responses_api(self, prompt: str) -> dict[str, Any]:
        """
        Call the Foundry Responses API exactly as shown in the portal sample:

            openai_client.responses.create(
                input=[{"role": "user", "content": "..."}],
                extra_body={
                    "agent_reference": {
                        "name": "planner-agent",
                        "version": "9",
                        "type": "agent_reference"
                    }
                },
            )

        The agent's own system prompt (set in Foundry portal) handles
        instructions — we only need to send the user message.
        """
        from azure.ai.projects import AIProjectClient

        # Build the project client using CustomApiKeyCredential
        project_client = AIProjectClient(
            endpoint=self._endpoint,
            credential=CustomApiKeyCredential(self._api_key),
        )

        # Get the OpenAI-compatible client attached to this Foundry project
        openai_client = project_client.get_openai_client()

        # Add JSON instruction to the user prompt so the agent outputs clean JSON
        json_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond ONLY with valid JSON. "
            f"No prose. No markdown fences. No <think> blocks. "
            f"Start with {{ and end with }}."
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai_client.responses.create(
                input=[{"role": "user", "content": json_prompt}],
                extra_body={
                    "agent_reference": {
                        "name":    self.agent_name,
                        "version": self.agent_version,
                        "type":    "agent_reference",
                    }
                },
            ),
        )

        reply: str = getattr(response, "output_text", None) or ""
        if not reply.strip():
            raise ValueError(
                f"Foundry agent '{self.agent_name}' v{self.agent_version} "
                f"returned empty output_text."
            )

        return _parse_json_reply(reply, self.agent_name)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def _parse_json_reply(text: str, agent_name: str) -> dict[str, Any]:
    """Parse JSON from agent reply using the battle-tested clean_and_parse_json."""
    try:
        from src.utils.config import clean_and_parse_json
        result = clean_and_parse_json(text)
        if not isinstance(result, dict):
            raise ValueError(
                f"Expected JSON object from '{agent_name}', "
                f"got {type(result).__name__}"
            )
        return result
    except Exception as exc:
        logger.warning(
            "foundry_json_parse_failed",
            agent=agent_name,
            error=str(exc),
            preview=text[:300],
        )
        raise ValueError(
            f"Could not parse JSON from '{agent_name}': {exc}"
        ) from exc