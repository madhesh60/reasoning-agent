"""
A2A Protocol - Message Definitions and Utilities

This module provides shared protocol definitions, message types,
and utility functions for A2A communication between agents.

Reference: Microsoft A2A Protocol Specification
"""

from typing import Any, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import json
import structlog

logger = structlog.get_logger(__name__)


# Message Types
class MessageType(str, Enum):
    """A2A message types."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    NOTIFICATION = "notification"


# Status Codes
class StatusCode(str, Enum):
    """Standard status codes for A2A responses."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"


@dataclass
class A2AHeader:
    """Header information for A2A messages."""
    protocol_version: str = "1.0"
    message_id: str = ""
    correlation_id: str | None = None
    timestamp: str = ""
    sender: str = ""
    receiver: str = ""


@dataclass
class A2AMessageBody:
    """Body of an A2A message."""
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    payload: Any = None


@dataclass
class A2ARequest:
    """A2A request message."""
    header: A2AHeader
    body: A2AMessageBody


@dataclass
class A2AResponseBody:
    """Body of an A2A response."""
    status: StatusCode
    result: Any = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class A2AResponse:
    """A2A response message."""
    header: A2AHeader
    body: A2AResponseBody


class AgentMessage(TypedDict):
    """Type definition for agent messages."""
    message_id: str
    message_type: str
    sender: str
    receiver: str
    method: str
    params: dict[str, Any]
    timestamp: str
    correlation_id: str | None


def create_request_message(
    sender: str,
    receiver: str,
    method: str,
    params: dict[str, Any] | None = None,
    message_id: str | None = None,
    correlation_id: str | None = None
) -> AgentMessage:
    """
    Create a standardized A2A request message.

    Args:
        sender: Name of the sending agent
        receiver: Name of the receiving agent
        method: Method to invoke
        params: Method parameters
        message_id: Optional message ID
        correlation_id: Optional correlation ID

    Returns:
        AgentMessage dictionary
    """
    from datetime import datetime

    return AgentMessage(
        message_id=message_id or f"msg_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        message_type=MessageType.REQUEST.value,
        sender=sender,
        receiver=receiver,
        method=method,
        params=params or {},
        timestamp=datetime.utcnow().isoformat(),
        correlation_id=correlation_id
    )


def create_response_message(
    original_message: AgentMessage,
    status: StatusCode,
    result: Any = None,
    error: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Create a standardized A2A response message.

    Args:
        original_message: The original request message
        status: Response status
        result: Result data
        error: Error information

    Returns:
        Response message dictionary
    """
    from datetime import datetime

    return {
        "message_id": f"resp_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "message_type": MessageType.RESPONSE.value,
        "sender": original_message["receiver"],
        "receiver": original_message["sender"],
        "status": status.value,
        "result": result,
        "error": error,
        "timestamp": datetime.utcnow().isoformat(),
        "correlation_id": original_message.get("correlation_id") or original_message["message_id"]
    }


def create_error_message(
    original_message: AgentMessage,
    error_code: str,
    error_message: str,
    details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Create a standardized A2A error message.

    Args:
        original_message: The original request message
        error_code: Error code
        error_message: Error description
        details: Additional error details

    Returns:
        Error message dictionary
    """
    return create_response_message(
        original_message=original_message,
        status=StatusCode.ERROR,
        error={
            "code": error_code,
            "message": error_message,
            "details": details or {}
        }
    )


def parse_message(raw_message: str | dict) -> AgentMessage | None:
    """
    Parse a raw message into an AgentMessage.

    Args:
        raw_message: JSON string or dictionary

    Returns:
        AgentMessage if valid, None otherwise
    """
    try:
        if isinstance(raw_message, str):
            data = json.loads(raw_message)
        else:
            data = raw_message

        # Validate required fields
        required_fields = ["message_id", "message_type", "sender", "receiver", "method"]
        for field in required_fields:
            if field not in data:
                logger.warning("parse_message_missing_field", field=field)
                return None

        return AgentMessage(
            message_id=data["message_id"],
            message_type=data["message_type"],
            sender=data["sender"],
            receiver=data["receiver"],
            method=data["method"],
            params=data.get("params", {}),
            timestamp=data.get("timestamp", ""),
            correlation_id=data.get("correlation_id")
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("parse_message_error", error=str(e))
        return None


def serialize_message(message: AgentMessage) -> str:
    """
    Serialize an AgentMessage to JSON string.

    Args:
        message: AgentMessage to serialize

    Returns:
        JSON string
    """
    return json.dumps(message, indent=2)


class MessageValidator:
    """Validator for A2A messages."""

    @staticmethod
    def validate_request(message: AgentMessage) -> tuple[bool, str | None]:
        """
        Validate an A2A request message.

        Args:
            message: Message to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if message["message_type"] != MessageType.REQUEST.value:
            return False, f"Invalid message type: expected 'request'"

        if not message["method"]:
            return False, "Missing method name"

        if not message["sender"]:
            return False, "Missing sender"

        if not message["receiver"]:
            return False, "Missing receiver"

        return True, None

    @staticmethod
    def validate_response(message: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Validate an A2A response message.

        Args:
            message: Message to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if message.get("message_type") == MessageType.ERROR.value:
            if not message.get("error"):
                return False, "Error message missing error details"

        return True, None


# Protocol Constants
PROTOCOL_VERSION = "1.0"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Well-known agent names
AGENT_PLANNER = "planner"
AGENT_RESEARCHER = "researcher"
AGENT_ANALYST = "analyst"
AGENT_WRITER = "writer"
AGENT_ORCHESTRATOR = "orchestrator"

# Well-known methods
METHOD_DECOMPOSE_TASK = "decompose_task"
METHOD_VALIDATE_PLAN = "validate_plan"
METHOD_SEARCH = "search"
METHOD_BATCH_SEARCH = "batch_search"
METHOD_ANALYZE = "analyze"
METHOD_ASSESS_RISK = "assess_risk"
METHOD_GENERATE_REPORT = "generate_report"
METHOD_FORMAT_OUTPUT = "format_output"


__all__ = [
    "MessageType",
    "StatusCode",
    "A2AHeader",
    "A2AMessageBody",
    "A2ARequest",
    "A2AResponseBody",
    "A2AResponse",
    "AgentMessage",
    "create_request_message",
    "create_response_message",
    "create_error_message",
    "parse_message",
    "serialize_message",
    "MessageValidator",
    "PROTOCOL_VERSION",
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "MAX_MESSAGE_SIZE",
    "AGENT_PLANNER",
    "AGENT_RESEARCHER",
    "AGENT_ANALYST",
    "AGENT_WRITER",
    "AGENT_ORCHESTRATOR",
    "METHOD_DECOMPOSE_TASK",
    "METHOD_VALIDATE_PLAN",
    "METHOD_SEARCH",
    "METHOD_BATCH_SEARCH",
    "METHOD_ANALYZE",
    "METHOD_ASSESS_RISK",
    "METHOD_GENERATE_REPORT",
    "METHOD_FORMAT_OUTPUT",
]