# Azure AI Foundry & Model Deployment Guide

This guide provides step-by-step instructions on setting up your Azure environment, deploying LLM models (specifically GPT-4o) on Azure AI Foundry and Azure OpenAI, and configuring your local environment variables to run this multi-agent system.

---

## 1. Overview of Azure Resources Needed

To run this system, you need access to:
1. **Azure Subscription**: An active Azure subscription.
2. **Azure OpenAI Resource**: To deploy reasoning and generation models (e.g., GPT-4o).
3. **Azure AI Foundry Project**: To manage, scale, and optionally deploy hosted agents, tools, and evaluate their outputs.
4. **Azure AI Search (Optional)**: If you intend to use document retrieval / Vector search via `MCPDocumentSearchTool`.
5. **Azure AI Foundry Model Catalog / Deployment**: To get model endpoints, API keys, and connection credentials.

---

## 2. Step-by-Step Setup Guide

### Step 2.1: Create an Azure OpenAI Resource & Deploy GPT-4o

1. **Sign in to Azure Portal**:
   - Go to [portal.azure.com](https://portal.azure.com) and log in.
2. **Create Azure OpenAI Resource**:
   - Search for **Azure OpenAI** in the top search bar.
   - Click **Create**.
   - Select your Subscription and Resource Group (or create a new one).
   - Select a Region (e.g., `East US` or `Sweden Central` are recommended for model availability).
   - Enter a Resource Name (e.g., `my-openai-resource`).
   - Select the `S0` Pricing Tier.
   - Click **Review + Create**, then **Create**.
3. **Deploy the GPT-4o Model**:
   - Once the deployment is complete, go to the resource.
   - Under the resource menu, click **Model Deployments** under **Resource Management**.
   - Click **Manage Deployments** to open the **Azure AI Studio / Azure OpenAI Studio**.
   - In Azure OpenAI Studio, navigate to **Deployments** (under **Shared Resources**).
   - Click **Create new deployment**.
   - Select **gpt-4o** as the model.
   - Select the **model version** (e.g., `2024-05-13` or `2024-08-06`).
   - Set the Deployment Name to **gpt-4o** (this matches `AZURE_OPENAI_DEPLOYMENT` in your `.env`).
   - Set the Token Limit (TPM) as needed.
   - Click **Create**.
4. **Retrieve API Credentials**:
   - Return to your Azure OpenAI resource page in the main **Azure Portal**.
   - Under **Resource Management**, click **Keys and Endpoint**.
   - Copy:
     - **KEY 1** (maps to `AZURE_OPENAI_API_KEY`)
     - **Endpoint** (maps to `AZURE_OPENAI_ENDPOINT`, e.g., `https://my-openai-resource.openai.azure.com/`)
     - **API Version** (e.g., `2024-02-01` or `2024-05-01-preview`)

---

### Step 2.2: Create an Azure AI Foundry Project

Azure AI Foundry (formerly Azure AI Studio) organizes resources under hubs and projects.

1. **Open Azure AI Foundry Portal**:
   - Go to [ai.azure.com](https://ai.azure.com).
2. **Create a Hub** (if you don't have one):
   - Click **All resources** -> **New resource** -> **AI Hub**.
   - Fill in the resource details (Subscription, Resource Group, Location).
   - Click **Next** and **Create**.
3. **Create a Project**:
   - Inside your Hub, click **Create Project** (or **New Project**).
   - Name the project (e.g., `research-multi-agent`).
   - Click **Create**.
4. **Retrieve Project Information**:
   - Once the project is created, copy the **Project Name** (maps to `AZURE_FOUNDRY_PROJECT`).
   - Copy the **Connection String** or project URL from the project homepage dashboard.
   - Generate or obtain the Foundry Project API Key (if using API key auth) under project settings (maps to `AZURE_FOUNDRY_API_KEY`).

---

### Step 2.3: Connect MCP Tools (Azure AI Search / Bing Search)

If you are using the Azure AI Foundry hosted agent endpoints:
1. Go to your project dashboard on [ai.azure.com](https://ai.azure.com).
2. Navigate to **Tools** or **Connected Resources**.
3. Click **Add connection**.
4. To connect **Azure AI Search**:
   - Select **Azure AI Search** from the list of services.
   - Choose your existing Azure AI Search service.
   - Set authentication to **Managed Identity** (recommended) or **API Key**.
5. To connect **Bing Search** (MCP Web Search):
   - Create a connection to the hosted MCP server or Bing Web Search service.
   - Set `MCP_SERVER_URL` to `https://mcp.ai.azure.com` or your hosted server endpoint.
   - Retrieve your **MCP token / API Key** (maps to `MCP_AUTH_TOKEN`).

---

## 3. Configuration Mapping (.env Template)

Edit your `.env` file in the root of the project with the credentials retrieved above:

```env
# ==============================================================================
# Azure OpenAI Configuration
# ==============================================================================
# The endpoint URL from the "Keys and Endpoint" page of your Azure OpenAI resource.
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Either KEY 1 or KEY 2 from the "Keys and Endpoint" page of your Azure OpenAI resource.
AZURE_OPENAI_API_KEY=your-api-key-here

# The exact deployment name of your gpt-4o model deployment (default is gpt-4o).
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# The API version you want to query (e.g., 2024-02-01 or 2024-08-01-preview).
AZURE_OPENAI_API_VERSION=2024-02-01


# ==============================================================================
# Azure AI Foundry Configuration
# ==============================================================================
# The endpoint for the Azure AI Foundry management plane/API.
AZURE_FOUNDRY_ENDPOINT=https://ai.azure.com/api/foundry

# The name of the project you created in the Azure AI Foundry portal.
AZURE_FOUNDRY_PROJECT=research-multi-agent

# The API key associated with your Azure AI Foundry project.
AZURE_FOUNDRY_API_KEY=your-foundry-api-key-here


# ==============================================================================
# MCP Server (Model Context Protocol) Configuration
# ==============================================================================
# The URL of your Model Context Protocol server for web search tools.
MCP_SERVER_URL=https://mcp.ai.azure.com

# The token used to authenticate calls to the MCP server.
MCP_AUTH_TOKEN=your-mcp-auth-token-here


# ==============================================================================
# A2A Protocol Configuration (For local development/testing)
# ==============================================================================
# Host address to bind the A2A server to (0.0.0.0 binds to all interfaces).
A2A_HOST=0.0.0.0

# The port where the A2A multi-agent server will listen for requests.
A2A_PORT=8080

# Timeout in seconds for inter-agent API calls.
A2A_TIMEOUT=30
```

---

## 4. Deploying Agents to Azure AI Foundry

For hosted agent deployment, refer to `docs/deployment.md`. You can deploy each agent package (`src/agents/planner.py`, etc.) as a hosted Agent under the **Agents** tab in your Azure AI Foundry project, exposing their endpoints over Azure AD authenticated channels.
