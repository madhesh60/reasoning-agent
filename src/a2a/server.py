"""
A2A Protocol Implementation - Server Side

This module provides the server-side implementation of the Agent-to-Agent (A2A) protocol,
enabling agents to receive and process requests from other agents.

The A2A server handles:
- Agent registration and capability publishing
- Request routing and processing
- Response serialization
- Error handling and notifications

Reference: Microsoft A2A Protocol Specification
"""

from typing import Any, Callable, Awaitable
import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """A2A message types."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"


@dataclass
class A2AEndpoint:
    """Represents an A2A agent endpoint."""
    method: str
    handler: Callable[..., Awaitable[Any]]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRegistration:
    """Agent registration information."""
    name: str
    description: str
    version: str
    capabilities: list[str]
    endpoints: dict[str, A2AEndpoint]
    metadata: dict[str, Any] = field(default_factory=dict)


class A2AServer:
    """
    Server for A2A protocol communication.

    This server hosts agents and exposes their methods via the A2A protocol.
    It handles request parsing, method routing, and response generation.
    """

    def __init__(
        self,
        agent_name: str,
        agent_description: str = "",
        agent_version: str = "1.0.0",
        port: int = 8080,
        host: str = "0.0.0.0"
    ):
        """
        Initialize the A2A server.

        Args:
            agent_name: Name of the agent being hosted
            agent_description: Description of the agent
            agent_version: Version of the agent
            port: Port to listen on
            host: Host address to bind to
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.agent_version = agent_version
        self.port = port
        self.host = host

        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"A2A Server - {agent_name}",
            description=agent_description,
            version=agent_version
        )

        # Agent capabilities and endpoints
        self._endpoints: dict[str, A2AEndpoint] = {}
        self._capabilities: list[str] = []

        # Register base routes
        self._setup_routes()

        logger.info(
            "a2a_server_initialized",
            agent_name=agent_name,
            port=port,
            host=host
        )

    def _setup_routes(self):
        """Set up FastAPI routes."""
        @self.app.get("/")
        async def root():
            return {
                "agent": self.agent_name,
                "status": "running",
                "version": self.agent_version,
                "endpoints": list(self._endpoints.keys())
            }

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "agent": self.agent_name}

        @self.app.get("/info")
        async def info():
            return {
                "name": self.agent_name,
                "description": self.agent_description,
                "version": self.agent_version,
                "capabilities": self._capabilities,
                "methods": {
                    method: {
                        "description": endpoint.description,
                        "parameters": endpoint.parameters
                    }
                    for method, endpoint in self._endpoints.items()
                },
                "status": "available"
            }

        @self.app.post("/call")
        async def call(request: Request):
            try:
                body = await request.json()
                return await self._handle_call(body)
            except Exception as e:
                logger.error("a2a_call_error", error=str(e))
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": str(e)
                        }
                    }
                )

        @self.app.post("/notify")
        async def notify(request: Request):
            try:
                body = await request.json()
                return await self._handle_notification(body)
            except Exception as e:
                logger.error("a2a_notify_error", error=str(e))
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": str(e)}
                )

    def register_endpoint(
        self,
        method: str,
        handler: Callable[..., Awaitable[Any]],
        description: str = "",
        parameters: dict[str, Any] | None = None
    ):
        """
        Register an endpoint with the A2A server.

        Args:
            method: Method name to expose
            handler: Async function to handle the method
            description: Description of what the method does
            parameters: Schema for method parameters
        """
        self._endpoints[method] = A2AEndpoint(
            method=method,
            handler=handler,
            description=description,
            parameters=parameters or {}
        )
        logger.info("a2a_endpoint_registered", method=method, agent=self.agent_name)

    def register_capability(self, capability: str):
        """Register a capability with the agent."""
        if capability not in self._capabilities:
            self._capabilities.append(capability)
            logger.info("a2a_capability_registered", capability=capability, agent=self.agent_name)

    async def _handle_call(self, request_body: dict[str, Any]) -> JSONResponse:
        """Handle an A2A call request."""
        method = request_body.get("method")
        params = request_body.get("params", {})
        message_id = request_body.get("message_id", "")

        logger.info(
            "a2a_handle_call",
            method=method,
            message_id=message_id,
            agent=self.agent_name
        )

        if not method:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "error": {
                        "code": "MISSING_METHOD",
                        "message": "Method name is required"
                    }
                }
            )

        if method not in self._endpoints:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "error": {
                        "code": "METHOD_NOT_FOUND",
                        "message": f"Method '{method}' not found"
                    }
                }
            )

        try:
            endpoint = self._endpoints[method]
            result = await endpoint.handler(**params)

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result),
                    "metadata": {
                        "method": method,
                        "message_id": message_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            )
        except Exception as e:
            logger.error(
                "a2a_handler_error",
                method=method,
                error=str(e)
            )
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": {
                        "code": "HANDLER_ERROR",
                        "message": str(e)
                    }
                }
            )

    async def _handle_notification(self, request_body: dict[str, Any]) -> JSONResponse:
        """Handle an A2A notification."""
        method = request_body.get("method")
        params = request_body.get("params", {})

        logger.info(
            "a2a_handle_notification",
            method=method,
            agent=self.agent_name
        )

        if method and method in self._endpoints:
            try:
                endpoint = self._endpoints[method]
                await endpoint.handler(**params)
            except Exception as e:
                logger.error("a2a_notification_error", error=str(e))

        return JSONResponse(
            status_code=200,
            content={"status": "acknowledged"}
        )

    def run(self, blocking: bool = True):
        """
        Run the A2A server.

        Args:
            blocking: If True, block the current thread. If False, run in background.
        """
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info(
            "a2a_server_starting",
            agent=self.agent_name,
            host=self.host,
            port=self.port
        )

        if blocking:
            server.run()
        else:
            import asyncio
            asyncio.create_task(server.serve())


class A2AMultiAgentServer:
    """
    Server that hosts multiple agents on a single port.

    This server uses path-based routing to serve multiple agents,
    each with their own set of endpoints.
    """

    def __init__(self, port: int = 8080, host: str = "0.0.0.0"):
        """
        Initialize the multi-agent server.

        Args:
            port: Port to listen on
            host: Host address to bind to
        """
        self.port = port
        self.host = host
        self._agents: dict[str, A2AServer] = {}

        self.app = FastAPI(
            title="A2A Multi-Agent Server",
            description="Agent-to-Agent Protocol Server hosting multiple agents",
            version="1.0.0"
        )

        self._setup_routes()

        logger.info("a2a_multi_agent_server_initialized", port=port)

    def register_agent(self, agent: A2AServer):
        """Register an agent with the multi-agent server."""
        self._agents[agent.agent_name] = agent
        logger.info("a2a_agent_registered", agent_name=agent.agent_name)

    def _setup_routes(self):
        """Set up multi-agent routing routes."""
        @self.app.get("/")
        async def root():
            return {
                "server": "A2A Multi-Agent Server",
                "version": "1.0.0",
                "agents": list(self._agents.keys()),
                "status": "running"
            }

        @self.app.get("/agents")
        async def list_agents():
            return {
                "agents": [
                    {
                        "name": agent.agent_name,
                        "description": agent.agent_description,
                        "version": agent.agent_version,
                        "endpoints": list(agent._endpoints.keys())
                    }
                    for agent in self._agents.values()
                ]
            }

        @self.app.get("/agents/{agent_name}")
        async def get_agent_info(agent_name: str):
            if agent_name not in self._agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
            agent = self._agents[agent_name]
            return {
                "name": agent.agent_name,
                "description": agent.agent_description,
                "version": agent.agent_version,
                "capabilities": agent._capabilities,
                "methods": list(agent._endpoints.keys())
            }

        @self.app.post("/agents/{agent_name}/call")
        async def agent_call(agent_name: str, request: Request):
            if agent_name not in self._agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

            body = await request.json()
            agent = self._agents[agent_name]
            return await agent._handle_call(body)

        @self.app.post("/agents/{agent_name}/notify")
        async def agent_notify(agent_name: str, request: Request):
            if agent_name not in self._agents:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

            body = await request.json()
            agent = self._agents[agent_name]
            return await agent._handle_notification(body)

    def run(self, blocking: bool = True):
        """Run the multi-agent server."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        logger.info(
            "a2a_multi_agent_server_starting",
            host=self.host,
            port=self.port,
            agents=len(self._agents)
        )

        if blocking:
            server.run()
        else:
            import asyncio
            asyncio.create_task(server.serve())


async def main():
    """Demo function for testing A2A server."""
    print("=" * 60)
    print("A2A PROTOCOL SERVER DEMO")
    print("=" * 60)

    # Create server for planner agent
    server = A2AServer(
        agent_name="planner",
        agent_description="Task decomposition and research planning agent",
        agent_version="1.0.0",
        port=8080
    )

    # Register capabilities
    server.register_capability("task_execution")
    server.register_capability("coordination")

    # Register endpoints
    async def decompose_task_handler(query: str):
        from ..agents.planner import PlannerAgent
        planner = PlannerAgent()
        plan = await planner.decompose_task(query)
        return plan.model_dump()

    async def validate_plan_handler(plan: dict):
        from ..agents.planner import PlannerAgent, ResearchPlan
        planner = PlannerAgent()
        plan_obj = ResearchPlan(**plan)
        validation = await planner.validate_plan(plan_obj)
        return validation

    server.register_endpoint(
        "decompose_task",
        decompose_task_handler,
        description="Decompose a complex query into subtasks",
        parameters={"query": {"type": "string", "description": "The research query"}}
    )

    server.register_endpoint(
        "validate_plan",
        validate_plan_handler,
        description="Validate a research plan",
        parameters={"plan": {"type": "object", "description": "The research plan to validate"}}
    )

    print(f"\nAgent: {server.agent_name}")
    print(f"Version: {server.agent_version}")
    print(f"Description: {server.agent_description}")
    print("\nRegistered Endpoints:")
    for method in server._endpoints.keys():
        print(f"  - {method}")
    print("\nRegistered Capabilities:")
    for cap in server._capabilities:
        print(f"  - {cap}")

    print("\nNote: Start the server with server.run() to expose endpoints.")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())