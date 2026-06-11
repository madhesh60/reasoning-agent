# LOGOS — Autonomous Research Intelligence Agent

> A 6-agent AI research pipeline with persistent memory, human-in-the-loop clarifications, and a story-driven CLI. Powered by Azure AI Foundry.

## Install

```bash
pip install logos-research
```

## Usage

```bash
# Interactive session (recommended)
logos

# Single query
logos -q "What are the latest trends in NLP?"

# Local model only, no Foundry agents
logos --no-a2a -q "..."

# Skip clarifying questions
logos --no-hitl -q "..."

# Verify model connection
logos --model-test
```

## Configuration

On first run, LOGOS will prompt you to set up your profile. It stores memory in `~/.logos/memory.db`.

Copy `.env.example` to `.env` in your working directory and fill in your Azure credentials:

```bash
cp .env.example .env
```

Required variables:

```
AZURE_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/deployments/your-deployment
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

## How It Works

```
Your query
  │
  ├── Memory loaded (past queries, preferences, tracked entities)
  │
  ├── Clarifying questions (2-3, shaped by your context)
  │
  └── 6-agent Foundry pipeline:
        1  Planner               Decomposes your query
        2  Researcher            Web search and data retrieval
        3  Industry News         Real-time trend signals
        4  Competitive Intel     Market landscape mapping
        5  Analyst               Synthesises findings and risk
        6  Writer                Structured professional report
```

## Memory

LOGOS remembers across sessions:
- Your profile (name, role, domain, depth preference)
- Past research queries and summaries
- Frequently investigated entities
- Bookmarked findings

All stored locally in `~/.logos/memory.db`. Type `memory` in the interactive session to review it.

## License

MIT
