# 🤖 Phi-4 Reasoning Multi-Agent — Run Commands & Guide

> **Your Azure AI Foundry models:** `phi-4-mini-reasoning` (fast) · `phi-4-reasoning` (powerful)
> **Pipeline:** Planner → Researcher → Analyst → Writer

---

## ✅ Step 0 — One-Time Setup

### Activate your virtual environment
```powershell
# From the project root
cd "C:\Users\RAJ\Desktop\research-to-report-multi-agent"
python -m venv venv
.\venv\Scripts\activate
```

### Install all dependencies
```powershell
pip install -r requirements.txt

# Also install langchain-azure-ai (needed for MCP toolbox)
pip install langchain-azure-ai
```

---

## ✅ Step 1 — Configure Your `.env` File

Your `.env` file already exists. Make sure these keys are filled in:

```env
# ── Required: Azure AI Foundry model endpoint ────────────────────────────────
AZURE_OPENAI_ENDPOINT=https://<your-resource>.services.ai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=phi-4-mini-reasoning   # or phi-4-reasoning

# ── Required: Azure AI Foundry project (for MCP web search) ─────────────────
AZURE_PROJECT_ENDPOINT=https://<your-resource>.services.ai.azure.com/api/projects/<project-name>
AZURE_TOOLBOX_NAME=reasoning-agent-web-search
AZURE_TOOLBOX_VERSION=1

# ── Optional: MCP token (if your Foundry uses bearer auth) ──────────────────
MCP_AUTH_TOKEN=<your-mcp-token>
```

> **Where to find these values:**
> 1. Go to [ai.azure.com](https://ai.azure.com) → your project → **Models + Endpoints**
> 2. Click on your `phi-4-mini-reasoning` or `phi-4-reasoning` deployment
> 3. Copy the **Endpoint URL** and **API Key**
> 4. For `AZURE_PROJECT_ENDPOINT`: in the project overview sidebar, copy the project URL

---

## Step 2 — Test Your Model Connection (Run First!)

```powershell
# Test that your Azure Foundry model responds
python run_agent.py --model-test
```

**Expected output:**
```
Azure AI Foundry — Model Connection Test
=========================================
Connecting to: https://...services.ai.azure.com/openai/v1
Deployment   : phi-4-mini-reasoning

Model reply:
<think>
Let me reason step-by-step about EV investment risks in India...
</think>

Top 3 EV investment risks in India:
1. Regulatory uncertainty — subsidy policy changes (FAME-II)
2. Supply chain risk — 80%+ lithium imports
3. Competition — Chinese OEMs entering with lower prices

Finish reason : stop
Total tokens  : 487
```

> If you see `<think>...</think>` blocks — that's the **Phi-4 reasoning trace**. This is working correctly!

---

## Step 3 — Test Web Search (MCP Tool)

```powershell
# Test the Azure Foundry MCP Bing search tool
python run_agent.py --search-test
```

**Expected output:**
```
  Results: 5   Time: 2340ms
  1. Latest AI Research Trends 2025 (MIT Technology Review)
  2. Top AI Breakthroughs in 2025 (Nature)
  ...
```

> If `AZURE_PROJECT_ENDPOINT` is not set, it falls back to **simulated results** (still works for testing the pipeline).

---

## Step 4 — Run Individual Agent Tests

Use `quick_demo.py` to test each agent one-by-one:

```powershell
# Test only model connection
python quick_demo.py --only model

# Test only web search
python quick_demo.py --only search

# Test only Planner agent (task decomposition)
python quick_demo.py --only planner

# Test only Researcher agent (web search + ranking)
python quick_demo.py --only researcher

# Test only Analyst agent (insight extraction, risk scoring)
python quick_demo.py --only analyst

# Test only Writer agent (report generation)
python quick_demo.py --only writer

# Run the FULL 4-agent pipeline end-to-end
python quick_demo.py --only full

# Run ALL tests in sequence
python quick_demo.py
```

---

## Step 5 — Run the Full Agent Pipeline

### Option A: Single query (recommended first run)
```powershell
python run_agent.py --query "What are the top 3 investment risks in the Indian EV market?"
```

### Option B: Switch to the larger Phi-4 Reasoning model
```powershell
python run_agent.py --query "Analyze competitive landscape of AI coding assistants" --model phi-4-reasoning
```

### Option C: Interactive mode (ask multiple questions)
```powershell
python run_agent.py
# Then type your question at the prompt:
# Query> What are the risks of investing in AI startups in India?
```

### Option D: Streaming mode (see each stage as it runs)
```powershell
python run_agent.py --stream
# OR with a query:
python run_agent.py --query "Renewable energy challenges in India" --stream
```

---

## Step 6 — Test the A2A Protocol (Agent-to-Agent HTTP endpoints)

```powershell
# Start the A2A server (in a separate terminal)
python -m uvicorn src.a2a.server:app --reload --host 0.0.0.0 --port 8080
```

```powershell
# In another terminal — start the FastAPI main app
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
# Then open: http://localhost:8000/docs
```

---

## Step 7 — Run the Original Module-Level Demos

Each agent has its own standalone demo:

```powershell
# Test model connection script (src/utils/run_model.py)
python -m src.utils.run_model

# Test planner agent standalone
python -m src.agents.planner

# Test researcher agent standalone
python -m src.agents.researcher

# Test analyst agent standalone
python -m src.agents.analyst

# Test writer agent standalone
python -m src.agents.writer

# Test MCP web search standalone
python -m src.mcp_tools.web_search

# Test Azure MCP toolbox agent (LangGraph ReAct + Bing search)
python -m src.mcp_tools.azure_web_search

# Run the full orchestration workflow
python -m src.orchestration.research_workflow

# Run the main CLI
python -m src.main --query "What are the top 3 investment risks in the Indian EV market?"
python -m src.main --interactive
python -m src.main --demo
```

---

## Git Commits — Recommended Workflow

```powershell
cd "C:\Users\RAJ\Desktop\research-to-report-multi-agent"

# After testing model connection works:
git add run_agent.py quick_demo.py RUN_COMMANDS.md
git commit -m "feat: add run_agent.py interactive runner and quick_demo.py test suite"

# After confirming full pipeline works:
git add .
git commit -m "feat: verified full Planner-Researcher-Analyst-Writer pipeline with phi-4-reasoning"

# After adding web search integration:
git add src/mcp_tools/ src/agents/researcher.py
git commit -m "feat: connect MCP Azure Foundry web search toolbox to researcher agent"

# Push to GitHub
git push origin main
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `AZURE_OPENAI_ENDPOINT not set` | Check `.env` — copy exact URL from Azure portal |
| `Model not found (404)` | Check `AZURE_OPENAI_DEPLOYMENT` matches your deployment name exactly |
| `Timeout after 120s` | phi-4-reasoning is slow — normal. Or switch to `phi-4-mini-reasoning` |
| `<think> blocks showing` | Expected! That is the reasoning trace from Phi-4 |
| `Search returns simulated results` | Set `AZURE_PROJECT_ENDPOINT` in `.env` for real Bing search |
| `langchain_azure_ai not found` | Run: `pip install langchain-azure-ai` |
| `JSON parse error` | The `clean_and_parse_json` in config.py strips `<think>` blocks automatically |
| `A2A connection refused` | Normal — agents fall back to direct Python calls automatically |

---

## Project Structure Quick Reference

```
research-to-report-multi-agent/
├── run_agent.py                  <- YOUR MAIN ENTRY POINT
├── quick_demo.py                 <- Test each agent individually
├── RUN_COMMANDS.md               <- This guide
├── .env                          <- Your Azure credentials
├── src/
│   ├── agents/
│   │   ├── planner.py            <- Breaks query into sub-tasks
│   │   ├── researcher.py         <- Web search + source ranking
│   │   ├── analyst.py            <- Insight extraction + risk scoring
│   │   └── writer.py             <- Report generation (MD/HTML/JSON)
│   ├── mcp_tools/
│   │   ├── web_search.py         <- MCP web search (Azure toolbox)
│   │   └── azure_web_search.py   <- LangGraph ReAct agent with toolbox
│   ├── orchestration/
│   │   └── research_workflow.py  <- LangGraph StateGraph pipeline
│   ├── a2a/
│   │   ├── server.py             <- A2A FastAPI server
│   │   └── client.py             <- A2A HTTP client
│   ├── utils/
│   │   ├── config.py             <- Env config + LLM factory
│   │   ├── run_model.py          <- Quick model test
│   │   └── logging.py            <- Structured logging
│   └── main.py                   <- CLI entry point
└── requirements.txt
```

---

## Sample Queries to Try

```powershell
python run_agent.py --query "What are the top 3 investment risks in the Indian EV market?"
python run_agent.py --query "Analyze the competitive landscape of AI coding assistants in 2025"
python run_agent.py --query "What are the key challenges in adopting renewable energy in Indian data centers?"
python run_agent.py --query "What are the risks and opportunities in India's semiconductor manufacturing push?"
python run_agent.py --query "How is generative AI changing the pharmaceutical drug discovery process?"
```

> **Tip:** Use `--model phi-4-reasoning` for deeper analysis (takes 60-120s per query).
> Use `phi-4-mini-reasoning` (default) for faster results (20-40s per query).
