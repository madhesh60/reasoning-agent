# Research-to-Report Multi-Agent System
# Deployment Configuration for Azure AI Foundry

## Deployment Overview

This document provides step-by-step instructions for deploying the multi-agent system to Azure AI Foundry.

---

## Prerequisites

1. Azure subscription with sufficient permissions
2. Azure CLI installed and configured (`az login`)
3. Python 3.11+ environment
4. Access to Azure AI Foundry portal

---

## Step 1: Prepare Configuration

```bash
# Clone the repository
git clone https://github.com/your-org/research-to-report-multi-agent.git
cd research-to-report-multi-agent

# Create .env file
cp .env.example .env

# Edit .env with your Azure credentials
vim .env
```

Required environment variables:
```env
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_FOUNDRY_PROJECT=your-project-name
AZURE_FOUNDRY_API_KEY=your-foundry-key
MCP_SERVER_URL=https://mcp.ai.azure.com
```

---

## Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Build Agents

Each agent can be built and deployed individually or as a group:

### Build All Agents

```bash
# Build the agent package
python -m build

# Verify build
ls dist/
```

### Individual Agent Build

```bash
# Build Planner agent
python -m src.agents.planner --build

# Build Researcher agent
python -m src.agents.researcher --build

# Build Analyst agent
python -m src.agents.analyst --build

# Build Writer agent
python -m src.agents.writer --build
```

---

## Step 4: Deploy to Azure AI Foundry

### Option A: Using Azure CLI

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription YOUR_SUBSCRIPTION_ID

# Create Foundry project (if not exists)
az foundry project create \
    --name research-multi-agent \
    --resource-group YOUR_RESOURCE_GROUP \
    --location eastus

# Deploy agents
./scripts/deploy_to_foundry.sh
```

### Option B: Using Foundry Portal

1. Navigate to [ai.azure.com](https://ai.azure.com)
2. Select your project
3. Go to "Agents" tab
4. Click "Create Agent"
5. Upload agent code and configuration
6. Configure endpoints and scaling

### Option C: Using Scripts

```bash
# Deploy all agents
./scripts/deploy_agents.sh --environment production

# Check deployment status
az foundry agent list -p research-multi-agent

# Verify agent health
./scripts/health_check.sh
```

---

## Step 5: Configure A2A Endpoints

After deploying agents, configure A2A communication:

```bash
# Set up A2A server for each agent
./scripts/setup_a2a.sh --agent planner --port 8080
./scripts/setup_a2a.sh --agent researcher --port 8081
./scripts/setup_a2a.sh --agent analyst --port 8082
./scripts/setup_a2a.sh --agent writer --port 8083

# Verify A2A connectivity
./scripts/test_a2a_connection.sh
```

---

## Step 6: Connect MCP Tools

Configure MCP tool access:

```bash
# Connect MCP web search
az foundry agent tool connect \
    --name researcher \
    --tool mcp \
    --endpoint https://mcp.ai.azure.com \
    --auth-type api-key

# Connect Azure AI Search (optional)
az foundry agent tool connect \
    --name researcher \
    --tool azure_ai_search \
    --endpoint YOUR_SEARCH_ENDPOINT \
    --auth-type managed-identity
```

---

## Step 7: Configure Authentication

Set up Azure AD authentication:

```bash
# Create service principal
az ad sp create-for-rbac \
    --name research-agents-sp \
    --role contributor \
    --scopes /subscriptions/YOUR_SUBSCRIPTION/resourceGroups/YOUR_RG

# Assign roles to agents
az role assignment create \
    --assignee YOUR_SP_APP_ID \
    --role "Cognitive Services OpenAI User" \
    --scope /subscriptions/YOUR_SUBSCRIPTION/resourceGroups/YOUR_RG
```

---

## Step 8: Verify Deployment

Run the verification script:

```bash
# Test complete workflow
python scripts/verify_deployment.py

# Expected output:
# - All agents responding
# - A2A connections established
# - MCP tools connected
# - Workflow executes successfully
```

---

## Deployment Configuration

### Agent Configuration

```yaml
# config/agent_config.yaml
agents:
  planner:
    name: planner
    version: 1.0.0
    resources:
      cpu: 1
      memory: 2Gi
    scaling:
      min_replicas: 1
      max_replicas: 5
      target_cpu: 70

  researcher:
    name: researcher
    version: 1.0.0
    resources:
      cpu: 2
      memory: 4Gi
    scaling:
      min_replicas: 1
      max_replicas: 10
      target_cpu: 70

  analyst:
    name: analyst
    version: 1.0.0
    resources:
      cpu: 1
      memory: 2Gi
    scaling:
      min_replicas: 1
      max_replicas: 5
      target_cpu: 70

  writer:
    name: writer
    version: 1.0.0
    resources:
      cpu: 1
      memory: 2Gi
    scaling:
      min_replicas: 1
      max_replicas: 5
      target_cpu: 70
```

### A2A Configuration

```yaml
# config/a2a_config.yaml
a2a:
  server:
    host: 0.0.0.0
    port: 8080
    timeout: 30

  agents:
    - name: planner
      endpoint: /planner
      methods:
        - decompose_task
        - validate_plan

    - name: researcher
      endpoint: /researcher
      methods:
        - search
        - batch_search

    - name: analyst
      endpoint: /analyst
      methods:
        - analyze
        - assess_risk

    - name: writer
      endpoint: /writer
      methods:
        - generate_report
        - format_output
```

---

## Monitoring

### View Logs

```bash
# View agent logs
az foundry agent logs -p research-multi-agent -n planner --tail 100

# Search logs
az foundry agent logs -p research-multi-agent -n researcher --filter "level eq 'ERROR'"
```

### View Metrics

```bash
# List available metrics
az monitor metrics list-definitions --resource /subscriptions/YOUR_SUB/resourceGroups/YOUR_RG/providers/microsoft.insights/components/YOUR_APP_INSIGHTS

# Get agent metrics
az monitor metrics list --resource /subscriptions/YOUR_SUB/resourceGroups/YOUR_RG/providers/microsoft.insights/components/YOUR_APP_INSIGHTS --metric "Requests"
```

---

## Troubleshooting

### Common Issues

**Issue: Agent not responding**
```bash
# Check agent status
az foundry agent list -p research-multi-agent

# Restart agent
az foundry agent restart -p research-multi-agent -n YOUR_AGENT
```

**Issue: A2A connection failed**
```bash
# Verify endpoints
curl https://YOUR_AGENT_ENDPOINT/health

# Check A2A logs
az foundry agent logs -p research-multi-agent -n YOUR_AGENT | grep A2A
```

**Issue: MCP tools not working**
```bash
# Test MCP connection
python scripts/test_mcp_connection.py

# Check MCP server status
curl https://mcp.ai.azure.com/health
```

---

## Scaling Configuration

| Environment | Agents | Replicas | Max Concurrent |
|-------------|--------|----------|-----------------|
| Development | 4 | 1 each | 5 |
| Staging | 4 | 2 each | 20 |
| Production | 4 | 3-10 each | 100 |

---

## Cleanup

To remove the deployment:

```bash
# Delete agents
az foundry agent delete -p research-multi-agent --all

# Delete project
az foundry project delete -n research-multi-agent

# Delete resource group (optional)
az group delete -n YOUR_RESOURCE_GROUP --yes
```

---

**Deployment Version:** 1.0.0
**Last Updated:** June 5, 2024