"""
MCP Tools - Web Search Integration

This module provides MCP (Model Context Protocol) tool integration for web search,
connecting the multi-agent system to external search capabilities.

The MCP protocol allows agents to access tools like:
- Web search for real-time information
- Document retrieval from various sources
- Code execution for data analysis

Reference: Microsoft MCP Protocol Specification
"""

from typing import Any, Literal
import json
import structlog
from dataclasses import dataclass
from enum import Enum
import httpx

logger = structlog.get_logger(__name__)


class SearchEngine(str, Enum):
    """Supported search engines."""
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    AZURE_AI_SEARCH = "azure_ai_search"


@dataclass
class SearchResult:
    """Standardized search result format."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: str | None = None
    relevance_score: float = 0.0
    authority_score: float = 0.0


@dataclass
class SearchResponse:
    """Response from web search operation."""
    query: str
    results: list[SearchResult]
    total_results: int
    execution_time_ms: float
    sources: list[str]


class MCPWebSearchTool:
    """
    MCP tool for web search operations.

    This tool provides access to web search capabilities through the MCP protocol,
    allowing agents to retrieve current information from the internet.
    """

    def __init__(
        self,
        mcp_server_url: str = "https://mcp.ai.azure.com",
        api_key: str | None = None,
        timeout: int = 15,
        azure_project_endpoint: str | None = None,
        azure_toolbox_name: str | None = None,
        azure_toolbox_version: str | None = None,
        azure_openai_api_key: str | None = None
    ):
        """
        Initialize the MCP web search tool.

        Args:
            mcp_server_url: URL of the MCP server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            azure_project_endpoint: Optional project endpoint for Azure AI Foundry
            azure_toolbox_name: Name of the toolbox in Azure AI Foundry
            azure_toolbox_version: Version of the toolbox
            azure_openai_api_key: API Key for Azure AI Foundry tools authentication
        """
        import os
        self.mcp_server_url = mcp_server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

        self.azure_project_endpoint = azure_project_endpoint or os.getenv("AZURE_PROJECT_ENDPOINT")
        self.azure_toolbox_name = azure_toolbox_name or os.getenv("AZURE_TOOLBOX_NAME", "reasoning-agent-web-search")
        self.azure_toolbox_version = azure_toolbox_version or os.getenv("AZURE_TOOLBOX_VERSION", "1")
        self.azure_openai_api_key = azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")

        logger.info(
            "mcp_web_search_initialized",
            server_url=mcp_server_url,
            use_azure_toolbox=self.azure_project_endpoint is not None
        )

    def _build_headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MCP-Client/1.0",
            "X-MCP-Protocol-Version": "1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_azure_search_response(self, text: str, query: str) -> list[dict[str, Any]]:
        """Parse the structured Azure AI Web Search response text into detailed SearchResults."""
        import re
        results = []
        
        # Extract markdown links pattern: [anchor](url)
        link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
        
        # Split text into sections by Markdown headings
        sections = re.split(r'\n(?:###|##)\s+', text)
        
        # The first section is usually the intro
        intro = sections[0].strip() if sections else ""
        if intro:
            intro_links = link_pattern.findall(intro)
            primary_url = intro_links[0][1] if intro_links else "https://ai.azure.com"
            primary_source = intro_links[0][0] if intro_links else "Azure AI Web Search"
            results.append({
                "title": f"Web Search Summary: {query}",
                "url": primary_url,
                "snippet": intro,
                "source": primary_source,
                "relevance_score": 1.0,
                "authority_score": 0.9
            })
            
        # Process other sections
        for i, section in enumerate(sections[1:]):
            section = section.strip()
            if not section:
                continue
                
            # Split title and content
            lines = section.split('\n', 1)
            title = lines[0].strip('* \t')
            content = lines[1].strip() if len(lines) > 1 else ""
            
            if not content:
                content = title
                
            links = link_pattern.findall(content)
            
            if links:
                # Create a SearchResult for each unique link in this section
                seen_urls = set()
                for anchor, url in links:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    
                    results.append({
                        "title": f"{title} ({anchor})",
                        "url": url,
                        "snippet": content,
                        "source": anchor,
                        "relevance_score": max(0.4, 0.9 - (i * 0.05)),
                        "authority_score": 0.8
                    })
            else:
                # Default result if no links are found
                results.append({
                    "title": title,
                    "url": "https://ai.azure.com",
                    "snippet": content,
                    "source": "Azure AI Web Search",
                    "relevance_score": max(0.4, 0.8 - (i * 0.05)),
                    "authority_score": 0.7
                })
                
        return results

    async def _execute_azure_search(self, query: str, max_results: int) -> SearchResponse:
        """Execute search using Azure AI Project Toolbox."""
        import time
        start_time = time.time()
        
        logger.info(
            "azure_toolbox_search_start",
            query=query[:50],
            project_endpoint=self.azure_project_endpoint,
            toolbox=self.azure_toolbox_name
        )
        
        try:
            from langchain_azure_ai.tools import AzureAIProjectToolbox
            
            toolbox = AzureAIProjectToolbox(
                project_endpoint=self.azure_project_endpoint,
                toolbox_name=self.azure_toolbox_name,
                toolbox_version=self.azure_toolbox_version,
                credential=self.azure_openai_api_key,
            )
            
            tools = await toolbox.get_tools()
            if not tools:
                raise ValueError("No tools found in Azure AI Project Toolbox")
                
            web_search_tool = tools[0]
            # Invoke the tool with search_query parameter
            res = await web_search_tool.ainvoke({"search_query": query})
            
            # Parse results
            results = []
            if isinstance(res, list) and len(res) > 0 and "text" in res[0]:
                text = res[0]["text"]
                results = self._parse_azure_search_response(text, query)
            elif isinstance(res, str):
                results = self._parse_azure_search_response(res, query)
            else:
                # Fallback in case of unexpected format
                results = [{
                    "title": f"Web Search Summary: {query}",
                    "url": "https://ai.azure.com",
                    "snippet": str(res),
                    "source": "Azure AI Web Search",
                    "relevance_score": 1.0,
                    "authority_score": 0.9
                }]
                
            # Wrap as SearchResult objects
            search_results = []
            for r in results[:max_results]:
                search_results.append(SearchResult(
                    title=r["title"],
                    url=r["url"],
                    snippet=r["snippet"],
                    source=r["source"],
                    published_date=r.get("published_date"),
                    relevance_score=r.get("relevance_score", 0.8),
                    authority_score=r.get("authority_score", 0.8)
                ))
                
            execution_time = (time.time() - start_time) * 1000
            
            logger.info(
                "azure_toolbox_search_complete",
                query=query[:50],
                results_count=len(search_results),
                execution_time_ms=execution_time
            )
            
            return SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                execution_time_ms=execution_time,
                sources=list(set([r.source for r in search_results]))
            )
        except Exception as e:
            logger.error("azure_toolbox_search_error_falling_back", query=query[:50], error=str(e))
            # Fall back to simulated results
            simulated = self._get_simulated_results(query, max_results)
            results = self._parse_search_results(simulated, query)
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                execution_time_ms=(time.time() - start_time) * 1000,
                sources=list(set([r.source for r in results]))
            )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_engine: SearchEngine = SearchEngine.BING,
        language: str = "en",
        safe_search: bool = True
    ) -> SearchResponse:
        """
        Execute a web search using MCP tools.

        Args:
            query: Search query string
            max_results: Maximum number of results
            search_engine: Search engine to use
            language: Language for results
            safe_search: Enable safe search filtering

        Returns:
            SearchResponse with results
        """
        # Route to Azure AI Project Toolbox if endpoint is set
        if self.azure_project_endpoint:
            return await self._execute_azure_search(query, max_results)

        import time
        start_time = time.time()

        logger.info(
            "mcp_search_start",
            query=query[:50],
            max_results=max_results,
            engine=search_engine.value
        )

        try:
            # Build MCP request payload
            payload = {
                "tool": "web_search",
                "method": "search",
                "params": {
                    "query": query,
                    "max_results": max_results,
                    "search_engine": search_engine.value,
                    "language": language,
                    "safe_search": safe_search
                }
            }

            # Execute search via MCP
            response = await self._execute_mcp_call(payload)

            # Parse response
            results = self._parse_search_results(response, query)

            execution_time = (time.time() - start_time) * 1000

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                execution_time_ms=execution_time,
                sources=list(set([r.source for r in results]))
            )

            logger.info(
                "mcp_search_complete",
                query=query[:50],
                results_count=len(results),
                execution_time_ms=execution_time
            )

            return search_response

        except Exception as e:
            logger.error("mcp_search_error", query=query[:50], error=str(e))
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                execution_time_ms=(time.time() - start_time) * 1000,
                sources=[]
            )

    async def _execute_mcp_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP call to the server."""
        url = f"{self.mcp_server_url}/tools/web_search"

        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=self._build_headers()
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning("mcp_server_unavailable", error=str(e))
            # Return simulated results for demo
            return self._get_simulated_results(payload["params"]["query"], payload["params"]["max_results"])

    def _parse_search_results(self, response: dict[str, Any], query: str) -> list[SearchResult]:
        """Parse search results from MCP response."""
        results = []

        raw_results = response.get("results", [])

        for i, result in enumerate(raw_results[:10]):
            search_result = SearchResult(
                title=result.get("title", "Unknown"),
                url=result.get("url", ""),
                snippet=result.get("snippet", result.get("description", "")),
                source=result.get("source", result.get("displayLink", "Unknown")),
                published_date=result.get("publishedDate"),
                relevance_score=1.0 - (i * 0.1),  # Higher = more relevant
                authority_score=result.get("authority", 0.7)
            )
            results.append(search_result)

        return results

    def _get_simulated_results(self, query: str, max_results: int) -> dict[str, Any]:
        """Generate simulated search results for demonstration."""
        sources = [
            {"name": "Reuters", "authority": 0.9},
            {"name": "Bloomberg", "authority": 0.9},
            {"name": "The Economic Times", "authority": 0.8},
            {"name": "CNBC", "authority": 0.85},
            {"name": "Financial Times", "authority": 0.9},
        ]

        results = []
        for i in range(min(max_results, len(sources))):
            source = sources[i]
            results.append({
                "title": f"{query.title()} - {source['name']} Analysis",
                "url": f"https://www.{source['name'].lower().replace(' ', '')}.com/article/{i+1}",
                "snippet": f"Comprehensive analysis and data-driven insights on {query} with market trends and expert opinions.",
                "source": source["name"],
                "publishedDate": "2024-06-01",
                "authority": source["authority"]
            })

        return {"results": results}

    async def batch_search(self, queries: list[str], max_results_per_query: int = 5) -> list[SearchResponse]:
        """
        Execute multiple searches in parallel.

        Args:
            queries: List of search queries
            max_results_per_query: Max results for each query

        Returns:
            List of SearchResponse objects
        """
        logger.info("mcp_batch_search_start", query_count=len(queries))

        import asyncio

        tasks = [
            self.search(query, max_results_per_query)
            for query in queries
        ]

        responses = await asyncio.gather(*tasks)

        logger.info("mcp_batch_search_complete", total_responses=len(responses))

        return responses

    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        days_back: int = 7
    ) -> SearchResponse:
        """
        Search for recent news articles.

        Args:
            query: Search query
            max_results: Maximum number of results
            days_back: How many days back to search

        Returns:
            SearchResponse with news results
        """
        logger.info("mcp_news_search_start", query=query[:50], days_back=days_back)

        return await self.search(
            query=query,
            max_results=max_results,
            safe_search=True
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class MCPDocumentSearchTool:
    """
    MCP tool for document search and retrieval.

    This tool provides access to Azure AI Search and other document sources.
    """

    def __init__(
        self,
        azure_search_endpoint: str | None = None,
        azure_search_key: str | None = None,
        index_name: str = "default",
        timeout: int = 30
    ):
        """
        Initialize the MCP document search tool.

        Args:
            azure_search_endpoint: Azure AI Search endpoint URL
            azure_search_key: Azure AI Search API key
            index_name: Name of the search index
            timeout: Request timeout in seconds
        """
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key
        self.index_name = index_name
        self.timeout = timeout

        if azure_search_endpoint:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        else:
            self._client = None

        logger.info(
            "mcp_document_search_initialized",
            endpoint=azure_search_endpoint,
            index=index_name
        )

    async def search_documents(
        self,
        query: str,
        max_results: int = 10,
        filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Search documents in the index.

        Args:
            query: Search query
            max_results: Maximum results
            filters: Optional filters

        Returns:
            Search results
        """
        if not self._client or not self.azure_search_endpoint:
            return {"results": [], "total": 0, "message": "Azure Search not configured"}

        logger.info("mcp_document_search", query=query[:50], index=self.index_name)

        try:
            url = f"{self.azure_search_endpoint}/indexes/{self.index_name}/docs/search"

            payload = {
                "search": query,
                "top": max_results,
                "select": "*",
                "searchMode": "any"
            }

            if filters:
                payload["filter"] = self._build_filter_string(filters)

            response = await self._client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.azure_search_key
                }
            )

            response.raise_for_status()
            data = response.json()

            return {
                "results": data.get("value", []),
                "total": data.get("@odata.count", len(data.get("value", []))),
                "query": query
            }

        except Exception as e:
            logger.error("mcp_document_search_error", error=str(e))
            return {"results": [], "total": 0, "error": str(e)}

    def _build_filter_string(self, filters: dict[str, Any]) -> str:
        """Build OData filter string from filters dict."""
        filter_parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                filter_parts.append(f"{key} eq '{value}'")
            elif isinstance(value, (int, float, bool)):
                filter_parts.append(f"{key} eq {value}")
            elif isinstance(value, list):
                value_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                filter_parts.append(f"{key} in ({value_str})")
        return " and ".join(filter_parts)


# Tool factory functions
def create_web_search_tool(config: dict[str, Any]) -> MCPWebSearchTool:
    """Create a configured web search tool."""
    return MCPWebSearchTool(
        mcp_server_url=config.get("mcp_server_url", "https://mcp.ai.azure.com"),
        api_key=config.get("api_key"),
        timeout=config.get("timeout", 15)
    )


def create_document_search_tool(config: dict[str, Any]) -> MCPDocumentSearchTool:
    """Create a configured document search tool."""
    return MCPDocumentSearchTool(
        azure_search_endpoint=config.get("azure_search_endpoint"),
        azure_search_key=config.get("azure_search_key"),
        index_name=config.get("index_name", "default"),
        timeout=config.get("timeout", 30)
    )


async def main():
    """Demo function for testing MCP tools."""
    from ..utils.config import load_environment
    load_environment()

    print("=" * 60)
    print("MCP WEB SEARCH TOOL DEMO")
    print("=" * 60)

    # Create web search tool
    tool = MCPWebSearchTool()

    # Test search
    query = "Indian EV market investment risks 2024"
    print(f"\nExecuting search: {query}\n")

    response = await tool.search(query, max_results=5)

    print(f"Query: {response.query}")
    print(f"Total Results: {response.total_results}")
    print(f"Execution Time: {response.execution_time_ms:.2f}ms")

    print("\n" + "-" * 60)
    print("SEARCH RESULTS")
    print("-" * 60)

    # Convert print outputs to avoid potential Windows terminal unicode encoding errors
    for i, result in enumerate(response.results, 1):
        try:
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Source: {result.source}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Authority: {result.authority_score:.2f}")
            print(f"   Snippet: {result.snippet[:100]}...")
        except UnicodeEncodeError:
            print(f"\n{i}. {result.title.encode('ascii', 'ignore').decode('ascii')}")
            print(f"   URL: {result.url}")
            print(f"   Source: {result.source}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   Authority: {result.authority_score:.2f}")
            print(f"   Snippet: {result.snippet[:100].encode('ascii', 'ignore').decode('ascii')}...")

    await tool.close()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())