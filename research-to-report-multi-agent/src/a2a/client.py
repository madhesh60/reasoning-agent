"""
A2A Protocol Implementation - Client Side

This module provides the client-side implementation of the Agent-to-Agent (A2A) protocol,
enabling agents to communicate and call each other's methods remotely.

The A2A protocol allows agents to:
- Discover available agents and their capabilities
- Call agent methods with structured parameters
- Handle responses and errors
- Support both sync and async communication

Reference: Microsoft A2A Protocol Specification
"""

from typing import Any, Literal
import json
import structlog
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import httpx

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """A2A message types."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"


class AgentCapability(str, Enum):
    """Agent capability types."""
    TASK_EXECUTION = "task_execution"
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    COORDINATION = "coordination"


@dataclass
class A2AMessage:
    """Represents an A2A protocol message."""
    message_id: str
    message_type: MessageType
    sender: str
    receiver: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    payload: Any = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: str | None = None


@dataclass
class A2AResponse:
    """Represents an A2A protocol response."""
    status: str
    result: Any = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    description: str
    version: str
    endpoint: str
    capabilities: list[str]
    methods: dict[str, dict[str, Any]]
    status: str = "available"


class A2AClient:
    """
    Client for A2A protocol communication.

    This client enables agents to call other agents' methods using the A2A protocol.
    It handles request/response serialization, error handling, and retry logic.
    """

    def __init__(
        self,
        agent_endpoint: str,
        timeout: int = 30,
        max_retries: int = 3,
        api_key: str | None = None
    ):
        """
        Initialize the A2A client.

        Args:
            agent_endpoint: URL of the target agent's A2A endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            api_key: Optional API key for authentication
        """
        self.agent_endpoint = agent_endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=self._build_headers()
        )

        logger.info(
            "a2a_client_initialized",
            endpoint=agent_endpoint,
            timeout=timeout
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers including auth if provided."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "A2A-Client/1.0",
            "X-A2A-Protocol-Version": "1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def call_agent(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        correlation_id: str | None = None
    ) -> A2AResponse:
        """
        Call an agent's method via A2A protocol.

        Args:
            method: The method name to call
            params: Method parameters
            correlation_id: Optional correlation ID for tracking

        Returns:
            A2AResponse with the result or error
        """
        message_id = f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        correlation_id = correlation_id or message_id

        message = A2AMessage(
            message_id=message_id,
            message_type=MessageType.REQUEST,
            sender="client",
            receiver=self.agent_endpoint.split("/")[-1],
            method=method,
            params=params or {},
            correlation_id=correlation_id
        )

        logger.info(
            "a2a_call_agent",
            method=method,
            endpoint=self.agent_endpoint,
            correlation_id=correlation_id
        )

        for attempt in range(self.max_retries):
            try:
                response = await self._send_request(message)
                return response
            except httpx.TimeoutException:
                logger.warning(
                    "a2a_timeout",
                    attempt=attempt + 1,
                    max_retries=self.max_retries
                )
                if attempt == self.max_retries - 1:
                    return A2AResponse(
                        status="timeout",
                        error={"code": "TIMEOUT", "message": f"Request timed out after {self.max_retries} attempts"}
                    )
            except httpx.HTTPError as e:
                logger.error(
                    "a2a_http_error",
                    error=str(e),
                    attempt=attempt + 1
                )
                if attempt == self.max_retries - 1:
                    return A2AResponse(
                        status="error",
                        error={"code": "HTTP_ERROR", "message": str(e)}
                    )

        return A2AResponse(
            status="error",
            error={"code": "MAX_RETRIES", "message": "Maximum retries exceeded"}
        )

    async def _send_request(self, message: A2AMessage) -> A2AResponse:
        """Send the actual HTTP request."""
        url = f"{self.agent_endpoint}/call"
        payload = {
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "sender": message.sender,
            "receiver": message.receiver,
            "method": message.method,
            "params": message.params,
            "timestamp": message.timestamp,
            "correlation_id": message.correlation_id
        }

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return A2AResponse(
            status=data.get("status", "success"),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )

    async def discover_capabilities(self) -> AgentInfo | None:
        """
        Discover the capabilities of the target agent.

        Returns:
            AgentInfo if successful, None otherwise
        """
        try:
            response = await self._client.get(f"{self.agent_endpoint}/info")
            response.raise_for_status()
            data = response.json()

            return AgentInfo(
                name=data["name"],
                description=data["description"],
                version=data["version"],
                endpoint=data["endpoint"],
                capabilities=data["capabilities"],
                methods=data["methods"],
                status=data.get("status", "available")
            )
        except Exception as e:
            logger.error("a2a_discover_capabilities_failed", error=str(e))
            return None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class A2ARouter:
    """
    A2A Router for managing multiple agent connections.

    This router maintains connections to multiple agents and provides
    a centralized way to route requests to appropriate agents.
    """

    def __init__(self):
        """Initialize the A2A router."""
        self._clients: dict[str, A2AClient] = {}
        self._agent_registry: dict[str, AgentInfo] = {}
        logger.info("a2a_router_initialized")

    def register_agent(self, name: str, endpoint: str, **kwargs) -> A2AClient:
        """
        Register an agent with the router.

        Args:
            name: Agent name/identifier
            endpoint: Agent's A2A endpoint URL
            **kwargs: Additional client configuration

        Returns:
            A2AClient instance for the agent
        """
        client = A2AClient(endpoint, **kwargs)
        self._clients[name] = client
        logger.info("a2a_agent_registered", name=name, endpoint=endpoint)
        return client

    def get_client(self, name: str) -> A2AClient | None:
        """Get a client for a registered agent."""
        return self._clients.get(name)

    async def discover_all_agents(self):
        """Discover capabilities of all registered agents."""
        for name, client in self._clients.items():
            info = await client.discover_capabilities()
            if info:
                self._agent_registry[name] = info

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._clients.keys())

    def get_agent_info(self, name: str) -> AgentInfo | None:
        """Get information about a specific agent."""
        return self._agent_registry.get(name)

    def find_agents_by_capability(self, capability: AgentCapability) -> list[AgentInfo]:
        """Find agents that support a specific capability."""
        return [
            info for info in self._agent_registry.values()
            if capability.value in info.capabilities
        ]

    async def route_request(
        self,
        agent_name: str,
        method: str,
        params: dict[str, Any] | None = None
    ) -> A2AResponse:
        """
        Route a request to a specific agent.

        Args:
            agent_name: Name of the target agent
            method: Method to call
            params: Method parameters

        Returns:
            A2AResponse from the agent
        """
        client = self.get_client(agent_name)
        if not client:
            return A2AResponse(
                status="error",
                error={"code": "AGENT_NOT_FOUND", "message": f"Agent '{agent_name}' not found"}
            )

        return await client.call_agent(method, params)

    async def close_all(self):
        """Close all agent connections."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._agent_registry.clear()
        logger.info("a2a_router_closed")


async def main():
    """Demo function for testing A2A client."""
    print("=" * 60)
    print("A2A PROTOCOL CLIENT DEMO")
    print("=" * 60)

    # Create router
    router = A2ARouter()

    # Register agents (simulated endpoints for demo)
    router.register_agent(
        "planner",
        "http://localhost:8080/planner",
        timeout=30
    )
    router.register_agent(
        "researcher",
        "http://localhost:8080/researcher",
        timeout=30
    )
    router.register_agent(
        "analyst",
        "http://localhost:8080/analyst",
        timeout=30
    )
    router.register_agent(
        "writer",
        "http://localhost:8080/writer",
        timeout=30
    )

    print("\nRegistered Agents:")
    for agent_name in router.list_agents():
        print(f"  - {agent_name}")

    # Note: In production, this would connect to actual agent endpoints
    print("\nNote: To test actual A2A calls, deploy the agent server first.")
    print("This demonstrates the client interface structure.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())