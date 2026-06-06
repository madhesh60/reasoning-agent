# Azure AI Foundry Multi-Agent Research System

A production-ready multi-agent system for conducting comprehensive research-to-report workflows using Microsoft's agent framework, LangGraph orchestration, A2A protocol for inter-agent communication, and MCP tools for external capabilities.

## Overview

This system implements a **Research-to-Report Multi-Agent System** that transforms complex research questions into structured, well-reasoned reports. It demonstrates advanced agent coordination patterns required for the Agents League Hackathon.

### Key Features

- **Intelligent Task Decomposition**: Planner agent breaks down complex queries into manageable subtasks
- **Real-time Web Research**: Researcher agent gathers current data via MCP-connected web search
- **Multi-step Reasoning**: Analyst agent applies structured thinking to extract insights
- **Self-Verification**: Built-in fact-checking and confidence scoring
- **Agent-to-Agent Communication**: A2A protocol for seamless agent orchestration
- **Production-Ready Deployment**: Azure AI Foundry hosted agents with automatic scaling

## System Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        User Query                               â”‚
â”‚              "What are the top 3 investment risks..."          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                â”‚
                                â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                     Orchestrator (LangGraph)                    â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”‚
â”‚  â”‚  [Planner] â”€â–º [Researcher] â”€â–º [Analyst] â”€â–º [Writer]     â”‚   â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚              â”‚             â”‚             â”‚
         â–¼              â–¼             â–¼             â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ MCP Web    â”‚  â”‚ MCP Web    â”‚  â”‚ Reasoning â”‚  â”‚ Report    â”‚
â”‚ Search     â”‚  â”‚ Search     â”‚  â”‚ Engine    â”‚  â”‚ Generator â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### Detailed Architecture

![Architecture Diagram](docs/architecture_diagram.svg)

For a detailed architecture explanation, see [Architecture Documentation](docs/architecture.md).

## Agent Roles

| Agent | Primary Function | Key Capabilities |
|-------|-----------------|------------------|
| **Planner** | Task Decomposition | Breaks queries into subtasks, identifies required tools |
| **Researcher** | Information Gathering | Web search, document retrieval, data sourcing |
| **Analyst** | Reasoning & Insights | Pattern detection, risk assessment, comparative analysis |
| **Writer** | Report Generation | Structured output, formatting, citation management |

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Agent Framework** | LangGraph | Reasoning flow orchestration |
| **Agent Runtime** | Azure AI Foundry | Hosted agent deployment |
| **Inter-Agent Comms** | A2A Protocol | Agent-to-agent messaging |
| **Tool Integration** | MCP (Model Context Protocol) | Web search, external tools |
| **LLM Backend** | Azure OpenAI (GPT-4o) | Reasoning and generation |

## Project Structure

```
research-to-report-multi-agent/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ agents/           # Individual agent implementations
â”‚   â”‚   â”œâ”€â”€ planner.py
â”‚   â”‚   â”œâ”€â”€ researcher.py
â”‚   â”‚   â”œâ”€â”€ analyst.py
â”‚   â”‚   â””â”€â”€ writer.py
â”‚   â”œâ”€â”€ orchestration/    # LangGraph workflow definitions
â”‚   â”‚   â””â”€â”€ research_workflow.py
â”‚   â”œâ”€â”€ a2a/              # A2A protocol implementation
â”‚   â”‚   â”œâ”€â”€ client.py
â”‚   â”‚   â”œâ”€â”€ server.py
â”‚   â”‚   â””â”€â”€ protocol.py
â”‚   â”œâ”€â”€ mcp_tools/        # MCP tool integrations
â”‚   â”‚   â”œâ”€â”€ web_search.py
â”‚   â”‚   â””â”€â”€ document_search.py
â”‚   â””â”€â”€ utils/            # Shared utilities
â”‚       â”œâ”€â”€ config.py
â”‚       â””â”€â”€ logging.py
â”œâ”€â”€ config/               # Configuration files
â”œâ”€â”€ docs/                 # Documentation and diagrams
â”œâ”€â”€ tests/                # Unit and integration tests
â”œâ”€â”€ scripts/              # Deployment and utility scripts
â”œâ”€â”€ pyproject.toml        # Project metadata
â”œâ”€â”€ requirements.txt      # Python dependencies
â””â”€â”€ README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.11+
- Azure subscription with AI Foundry access
- Azure OpenAI resource with GPT-4o deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/research-to-report-multi-agent.git
cd research-to-report-multi-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your Azure credentials
```

### Configuration

Update `.env` with your Azure credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_FOUNDRY_PROJECT=your-project-name
MCP_SERVER_URL=https://mcp.ai.azure.com
A2A_PORT=8080
```

### Running the System

**Local Development:**

```bash
# Run the orchestrator directly
python -m src.orchestration.research_workflow

# Run with sample query
python scripts/run_sample.py
```

**Deploy to Azure Foundry:**

```bash
# Deploy all agents
./scripts/deploy_to_foundry.sh

# Check deployment status
az foundry agent list -p your-project
```

## Usage Examples

### Example Query

```python
from src.orchestration.research_workflow import ResearchWorkflow

workflow = ResearchWorkflow()

query = """
What are the top 3 investment risks in the Indian EV market?
Include market data, regulatory challenges, and competitive analysis.
"""

result = workflow.execute(query)
print(result["report"])
```

### Sample Output Structure

```json
{
  "query": "Investment risks in Indian EV market",
  "status": "completed",
  "confidence_score": 0.87,
  "report": {
    "executive_summary": "...",
    "risk_1": {
      "title": "Regulatory Uncertainty",
      "analysis": "...",
      "evidence": [...],
      "impact_level": "high"
    },
    "risk_2": {...},
    "risk_3": {...},
    "conclusion": "..."
  },
  "sources": [...],
  "processing_time_seconds": 45
}
```

## A2A Protocol

This system uses Microsoft's Agent-to-Agent protocol for inter-agent communication. Each agent exposes a callable endpoint that other agents can invoke.

### Agent Endpoints

| Agent | A2A Endpoint | Methods |
|-------|--------------|---------|
| Planner | `http://localhost:8080/planner` | `decompose_task`, `validate_plan` |
| Researcher | `http://localhost:8080/researcher` | `search`, `fetch_document` |
| Analyst | `http://localhost:8080/analyst` | `analyze`, `assess_risk` |
| Writer | `http://localhost:8080/writer` | `generate_report`, `format_output` |

### Making A2A Calls

```python
from src.a2a.client import A2AClient

client = A2AClient("http://localhost:8080/researcher")

response = await client.call_agent(
    method="search",
    params={"query": "Indian EV market trends 2024", "max_results": 10}
)
```

## MCP Tools

The system integrates with Azure's MCP server for tool access:

- **Web Search**: Real-time internet search for current data
- **Document Search**: Azure AI Search for structured documents
- **Code Interpreter**: Execute Python for data analysis

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test suite
pytest tests/agents/ tests/orchestration/
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## Deployment

### Azure AI Foundry Deployment

```bash
# Login to Azure
az login

# Set up Foundry project
az foundry project create -n research-agents-project

# Deploy agents
./scripts/deploy_agents.sh --environment production

# Verify deployment
az foundry agent list -p research-agents-project
```

### Docker Deployment (Optional)

```bash
# Build Docker image
docker build -t research-agents:latest .

# Run container
docker run -p 8080:8080 \
  -e AZURE_OPENAI_API_KEY=$AZURE_KEY \
  research-agents:latest
```

## Evaluation Criteria

This system is designed to score highly in the following hackathon categories:

| Category | Weight | How Addressed |
|----------|--------|---------------|
| **Reasoning + Multi-step Thinking** | 20% | Multi-agent reasoning chain with self-verification |
| **Agent Collaboration (A2A)** | 15% | Native A2A protocol implementation |
| **Tool Integration (MCP)** | 15% | MCP web search and document tools |
| **Production Readiness** | 15% | Azure Foundry deployment, monitoring |
| **Innovation** | 15% | Novel agent orchestration patterns |
| **User Experience** | 10% | Clean API, comprehensive output |
| **Documentation** | 10% | This README, architecture diagrams |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Microsoft Azure AI Foundry team
- LangGraph community
- Hackathon organizers and judges

---

**Built for the Agents League Hackathon 2025**