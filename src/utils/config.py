"""
Configuration Management

This module provides centralized configuration management for the multi-agent system,
loading settings from environment variables and .env files.

Usage:
    from src.utils.config import load_environment, get_azure_openai_config

    # Load environment from .env file
    load_environment()

    # Get configuration for specific service
    config = get_azure_openai_config()
"""

import os
import structlog
from typing import Any
from dataclasses import dataclass
from dotenv import load_dotenv

logger = structlog.get_logger(__name__)


@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration."""
    endpoint: str
    api_key: str
    deployment: str
    api_version: str


@dataclass
class AzureFoundryConfig:
    """Azure AI Foundry configuration."""
    endpoint: str
    project: str
    api_key: str


@dataclass
class MCPConfig:
    """MCP server configuration."""
    server_url: str
    auth_token: str | None


@dataclass
class A2AConfig:
    """A2A protocol configuration."""
    host: str
    port: int
    timeout: int


@dataclass
class AgentConfig:
    """Agent configuration."""
    temperature: float
    max_tokens: int
    retry_attempts: int


@dataclass
class WebSearchConfig:
    """Web search configuration."""
    max_results: int
    timeout: int


@dataclass
class AppConfig:
    """Complete application configuration."""
    azure_openai: AzureOpenAIConfig
    azure_foundry: AzureFoundryConfig
    mcp: MCPConfig
    a2a: A2AConfig
    agent: AgentConfig
    web_search: WebSearchConfig
    log_level: str


def load_environment(env_file: str | None = None) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_file: Optional path to .env file. Defaults to .env in project root.
    """
    if env_file is None:
        env_file = ".env"

    loaded = load_dotenv(env_file)

    if loaded:
        logger.info("environment_loaded", env_file=env_file)
    else:
        logger.info("environment_using_existing", env_file=env_file)


def get_azure_openai_config() -> dict[str, Any]:
    """
    Get Azure OpenAI configuration from environment.

    Returns:
        Dictionary with Azure OpenAI settings
    """
    return {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    }


def get_azure_foundry_config() -> dict[str, Any]:
    """
    Get Azure AI Foundry configuration from environment.

    Returns:
        Dictionary with Azure Foundry settings
    """
    return {
        "endpoint": os.getenv("AZURE_FOUNDRY_ENDPOINT", ""),
        "project": os.getenv("AZURE_FOUNDRY_PROJECT", ""),
        "api_key": os.getenv("AZURE_FOUNDRY_API_KEY", "")
    }


def get_mcp_config() -> dict[str, Any]:
    """
    Get MCP server configuration from environment.

    Returns:
        Dictionary with MCP settings
    """
    return {
        "server_url": os.getenv("MCP_SERVER_URL", "https://mcp.ai.azure.com"),
        "auth_token": os.getenv("MCP_AUTH_TOKEN")
    }


def get_a2a_config() -> dict[str, Any]:
    """
    Get A2A protocol configuration from environment.

    Returns:
        Dictionary with A2A settings
    """
    return {
        "host": os.getenv("A2A_HOST", "0.0.0.0"),
        "port": int(os.getenv("A2A_PORT", "8080")),
        "timeout": int(os.getenv("A2A_TIMEOUT", "30"))
    }


def get_agent_config() -> dict[str, Any]:
    """
    Get agent configuration from environment.

    Returns:
        Dictionary with agent settings
    """
    return {
        "temperature": float(os.getenv("AGENT_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("AGENT_MAX_TOKENS", "4000")),
        "retry_attempts": int(os.getenv("AGENT_RETRY_ATTEMPTS", "3"))
    }


def get_web_search_config() -> dict[str, Any]:
    """
    Get web search configuration from environment.

    Returns:
        Dictionary with web search settings
    """
    return {
        "max_results": int(os.getenv("WEB_SEARCH_MAX_RESULTS", "10")),
        "timeout": int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
    }


def get_app_config() -> AppConfig:
    """
    Get complete application configuration.

    Returns:
        AppConfig dataclass with all settings
    """
    return AppConfig(
        azure_openai=AzureOpenAIConfig(**get_azure_openai_config()),
        azure_foundry=AzureFoundryConfig(**get_azure_foundry_config()),
        mcp=MCPConfig(**get_mcp_config()),
        a2a=A2AConfig(**get_a2a_config()),
        agent=AgentConfig(**get_agent_config()),
        web_search=WebSearchConfig(**get_web_search_config()),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )


def validate_config(config: AppConfig) -> list[str]:
    """
    Validate configuration and return list of warnings.

    Args:
        config: AppConfig to validate

    Returns:
        List of validation warnings
    """
    warnings = []

    # Check Azure OpenAI
    if not config.azure_openai.endpoint:
        warnings.append("AZURE_OPENAI_ENDPOINT not set")
    if not config.azure_openai.api_key:
        warnings.append("AZURE_OPENAI_API_KEY not set")

    # Check Azure Foundry
    if not config.azure_foundry.project:
        warnings.append("AZURE_FOUNDRY_PROJECT not set - using local mode")

    return warnings


# Centralized LLM Model Builder
def get_chat_model(temperature: float = 0.3, max_tokens: int = 4000) -> Any:
    """
    Get a configured Chat Model (either ChatOpenAI or AzureChatOpenAI).

    If the AZURE_OPENAI_ENDPOINT contains "services.ai.azure.com", "/openai/v1", or "/v1",
    it uses ChatOpenAI configured with standard OpenAI completions.
    Otherwise, it defaults to AzureChatOpenAI.

    max_tokens: caps total output tokens. Critical for Phi-4 reasoning models which
    can consume thousands of tokens in <think> blocks before outputting JSON.
    """
    from langchain_openai import ChatOpenAI, AzureChatOpenAI

    config = get_azure_openai_config()
    endpoint = config["endpoint"]
    deployment = config["deployment"]
    api_key = config["api_key"]
    api_version = config["api_version"]

    is_azure_ai_foundry = any(x in endpoint for x in ["services.ai.azure.com", "/openai/v1", "/v1"])

    if is_azure_ai_foundry:
        # Base url should be stripped of /chat/completions if present
        base_url = endpoint
        if base_url.endswith("/chat/completions"):
            base_url = base_url[:-17]

        logger.info(
            "instantiating_chat_openai_for_foundry",
            base_url=base_url,
            model=deployment,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=deployment,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120.0,  # Reasoning models can have high latency
        )
    else:
        logger.info(
            "instantiating_azure_chat_openai",
            endpoint=endpoint,
            deployment=deployment,
            temperature=temperature
        )
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration for agent initialization."""
    config = get_azure_openai_config()
    return {
        "azure_endpoint": config["endpoint"],
        "azure_deployment": config["deployment"],
        "api_key": config["api_key"],
        "api_version": config["api_version"]
    }


def get_search_config() -> dict[str, Any]:
    """Get search tool configuration."""
    return {
        "mcp_server_url": os.getenv("MCP_SERVER_URL", "https://mcp.ai.azure.com"),
        "api_key": os.getenv("MCP_AUTH_TOKEN"),
        "timeout": int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
    }


def get_a2a_client_config() -> dict[str, Any]:
    """Get A2A client configuration."""
    return {
        "timeout": int(os.getenv("A2A_TIMEOUT", "30")),
        "max_retries": int(os.getenv("AGENT_RETRY_ATTEMPTS", "3")),
        "api_key": os.getenv("A2A_API_KEY")
    }



def clean_and_parse_json(text: str) -> Any:
    """
    Cleans a text response by removing reasoning <think>...</think> blocks
    and extracting the JSON object or array content.

    Handles common model output issues:
    - <think> reasoning traces (Phi-4 reasoning models)
    - Markdown code fences (```json ... ```)
    - Invalid backslash escape sequences  (\\ in unexpected places)
    - Truncated JSON (finish_reason=length)
    - Surrounding prose before/after the JSON
    """
    import re
    import json

    # ── 1. Strip <think>...</think> blocks (handles closed and unclosed) ────────
    # Phi-4 reasoning models output potentially very long <think> blocks.
    # Case A: block is properly closed with </think>
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Case B: block is never closed (model ran out of space thinking)
    # In this case, strip from <think> to the end, then try from the start.
    if '<think>' in text_clean:
        think_start = text_clean.find('<think>')
        text_clean = text_clean[:think_start]
    text_clean = text_clean.strip()

    # ── 2. Strip markdown code fences ─────────────────────────────────────────
    text_clean = re.sub(r'^```(?:json)?\s*', '', text_clean, flags=re.MULTILINE)
    text_clean = re.sub(r'\s*```$', '', text_clean, flags=re.MULTILINE)
    text_clean = text_clean.strip()

    # ── 3. Extract the outermost JSON object or array ─────────────────────────
    first_brace   = text_clean.find('{')
    first_bracket = text_clean.find('[')

    start_idx = -1
    is_object = False

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
        is_object = True
    elif first_bracket != -1:
        start_idx = first_bracket
        is_object = False

    if start_idx != -1:
        end_char = '}' if is_object else ']'
        end_idx  = text_clean.rfind(end_char)
        if end_idx >= start_idx:
            text_clean = text_clean[start_idx:end_idx + 1]

    # ── 4. Fix invalid escape sequences (\x where x is not a valid JSON escape)
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    def _fix_escapes(s: str) -> str:
        result = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_ch = s[i + 1]
                if next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                    result.append(s[i])          # keep valid escape
                else:
                    result.append('\\\\')        # double it so it becomes a literal backslash
                i += 1
            else:
                result.append(s[i])
            i += 1
        return ''.join(result)

    text_clean = _fix_escapes(text_clean)

    # ── 5. First attempt: direct parse ────────────────────────────────────────
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError as first_err:
        pass

    # ── 6. Second attempt: repair truncated JSON (model hit token limit) ──────
    def _repair_truncated(s: str) -> str:
        """
        Add missing closing brackets/braces for truncated JSON.
        Tracks nesting depth with a stack, skips characters inside strings.
        """
        stack = []
        in_string = False
        escape_next = False
        OPEN  = {'{': '}', '[': ']'}
        CLOSE = {'}', ']'}

        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in OPEN:
                stack.append(OPEN[ch])
            elif ch in CLOSE and stack:
                if stack[-1] == ch:
                    stack.pop()

        # If we ended inside a string, close it first
        if in_string:
            s += '"'

        # Close remaining open structures in reverse order
        s += ''.join(reversed(stack))
        return s

    repaired = _repair_truncated(text_clean)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # ── 7. Third attempt: strip trailing commas (common model mistake) ─────────
    no_trailing = re.sub(r',\s*([}\]])', r'\1', repaired)
    try:
        return json.loads(no_trailing)
    except json.JSONDecodeError:
        pass

    # ── 8. Nuclear option: json-repair library (handles mid-stream corruption) ─
    # json-repair is purpose-built for LLM outputs: missing commas, unclosed
    # strings, unquoted keys, partial outputs, etc.
    try:
        import json_repair  # pip install json-repair
        # Feed it the raw extracted JSON region (not the repaired one which may
        # have introduced extra closings that confuse json-repair's own parser)
        candidate = text_clean  # use pre-repair version
        result = json_repair.repair_json(candidate, return_objects=True)
        if isinstance(result, (dict, list)) and result:
            logger.info("json_repair_succeeded", char_len=len(candidate))
            return result
    except ImportError:
        logger.warning("json_repair_not_installed", msg="pip install json-repair to improve JSON robustness")
    except Exception as repair_err:
        logger.warning("json_repair_failed", error=str(repair_err))

    raise ValueError(
        f"Could not parse JSON from model response after all repair attempts.\n"
        f"Last error: Expecting ',' delimiter\n"
        f"Text preview: {text[:300]}"
    )


if __name__ == "__main__":
    # Demo configuration loading
    print("=" * 60)
    print("CONFIGURATION MANAGEMENT DEMO")
    print("=" * 60)

    # Load environment
    load_environment()

    # Display configuration (sanitized)
    print("\nCurrent Configuration:")

    openai_config = get_azure_openai_config()
    print(f"\nAzure OpenAI:")
    print(f"  Endpoint: {openai_config['endpoint'][:50]}..." if openai_config['endpoint'] else "  Endpoint: Not set")
    print(f"  Deployment: {openai_config['deployment']}")
    print(f"  API Key: {'***' + openai_config['api_key'][-4:] if openai_config['api_key'] else 'Not set'}")

    foundry_config = get_azure_foundry_config()
    print(f"\nAzure Foundry:")
    print(f"  Project: {foundry_config['project'] or 'Not set'}")
    print(f"  Endpoint: {foundry_config['endpoint'][:50]}..." if foundry_config['endpoint'] else "  Endpoint: Not set")

    mcp_config = get_mcp_config()
    print(f"\nMCP Server:")
    print(f"  URL: {mcp_config['server_url']}")

    a2a_config = get_a2a_config()
    print(f"\nA2A Protocol:")
    print(f"  Host: {a2a_config['host']}")
    print(f"  Port: {a2a_config['port']}")

    agent_config = get_agent_config()
    print(f"\nAgent Settings:")
    print(f"  Temperature: {agent_config['temperature']}")
    print(f"  Max Tokens: {agent_config['max_tokens']}")

    # Validate configuration
    app_config = get_app_config()
    warnings = validate_config(app_config)

    if warnings:
        print("\nConfiguration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nConfiguration: OK")

    print("\n" + "=" * 60)