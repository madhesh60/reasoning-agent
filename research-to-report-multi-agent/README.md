# Azure AI Foundry Multi-Agent Research System

A multi-agent system for conducting research-to-report workflows using Microsoft's agent framework, LangGraph orchestration, and A2A protocol for inter-agent communication.

## Overview

This system implements a **Research-to-Report Multi-Agent System** that transforms complex research questions into structured, well-reasoned reports.

### Key Features

- **Intelligent Task Decomposition**: Planner agent breaks down complex queries into manageable subtasks
- **Real-time Web Research**: Researcher agent gathers current data via MCP-connected web search
- **Multi-step Reasoning**: Analyst agent applies structured thinking to extract insights
- **Agent-to-Agent Communication**: A2A protocol for seamless agent orchestration

## Quick Start

### Prerequisites

- Python 3.11+
- Azure subscription with AI Foundry access

### Installation

```bash
git clone https://github.com/madhesh60/reasoning-agent.git
cd reasoning-agent
python -m venv venv
pip install -r requirements.txt
cp .env.example .env
```

## Project Structure

```
research-to-report-multi-agent/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ agents/           # Agent implementations
â”‚   â”œâ”€â”€ orchestration/    # LangGraph workflows
â”‚   â”œâ”€â”€ a2a/              # A2A protocol
â”‚   â”œâ”€â”€ mcp_tools/        # MCP tool integrations
â”‚   â””â”€â”€ utils/            # Shared utilities
â”œâ”€â”€ tests/
â”œâ”€â”€ docs/
â”œâ”€â”€ scripts/
â””â”€â”€ README.md
```

## License

MIT License