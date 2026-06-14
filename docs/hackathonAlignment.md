# Hackathon Alignment and Requirements Mapping

This document details how the design, architecture, and implementation of the LOGOS Multi-Agent Research System align with the official guidelines and requirements of the **Microsoft Agent League Hackathon: Battle #2 - Reasoning Agents with Microsoft Foundry**.

---

## 1. Challenge & Scenario Alignment

LOGOS is built for **Enterprise Intelligence, Business Strategy, and Investment Analysis**. It serves as a virtual corporate research division that automates complex market research, competitive analysis, and strategic risk assessments. By orchestrating a pipeline of six specialized agent characters, the system decomposes ambiguous queries, retrieves live web intelligence, maps competitive dynamics, and compiles strategy reports to guide corporate and investment decision-making.

---

## 2. Microsoft IQ Layer Integration

The hackathon requires integrating at least one Microsoft IQ intelligence layer. LOGOS incorporates architecture patterns and local configurations that directly map to all three IQ concepts.

### 2.1. Work IQ (Context and Preference Alignment)
*   **Concept**: Aligning agent responses with the user's role, work context, and organization.
*   **LOGOS Implementation**:
    *   **SQLite Memory Context**: The system loads the user's name, role, organization, research domain, and preferred depth of analysis from a local SQLite profile database.
    *   **Human-in-the-Loop (HITL) Clarification**: Before agent execution, the coordinator prompts the user with query-specific questions to narrow focus areas and constraints.
    *   **Context Integration**: Stored preferences and answers are formatted into a structured context block and prepended to downstream system prompts, ensuring output reports align with the user's organizational context.

### 2.2. Foundry IQ (Grounded Retrieval and Citation)
*   **Concept**: Grounding responses in validated knowledge bases, document stores, and real-time search engines.
*   **LOGOS Implementation**:
    *   **Model Context Protocol (MCP) Servers**: The system implements an MCP client that integrates with **Tavily Web Search** and **Azure AI Web Search (Bing)**.
    *   **Live Web Grounding**: The Researcher and News Scanner agents execute searches via the MCP servers to gather recent facts and developments.
    *   **URL Citations**: The Writer agent compiles retrieved references into a dedicated "Resources & References" section, ensuring the report is fully verified and cited.

### 2.3. Fabric IQ (Semantic Mapping and Ontology)
*   **Concept**: Connecting business concepts, entity relationships, and historical patterns.
*   **LOGOS Implementation**:
    *   **SQLite Semantic Schema**: The database schema structures relationships between research sessions (`queries`), entity mention counters (`tracked_entities`), and manual bookmarks (`insights`).
    *   **Historical Analysis**: As queries run, the system tracks entity mention frequencies. Users can bookmark findings, linking them back to query sessions.
    *   **Ontology Enrichment**: The system queries historical entity counts and bookmarked insights to enrich new runs, building a semantic ontology of the user's research focus over time.

---

## 3. Data Hygiene and Synthetic Data Compliance

LOGOS maintains strict data compliance and security guardrails in accordance with hackathon rules:

*   **No PII or Customer Data**: The system is designed to analyze public industry trends and competitive structures. No customer records, personal details, or private credentials are sent to LLM endpoints.
*   **Synthetic/Demo Data**: During first-time setup or development runs, the SQLite store initializes using synthetic profiles and generic search queries.
*   **Credentials Separation**: Environmental variables, API keys, and endpoints are loaded at runtime from a local `.env` file, which is excluded from source control via `.gitignore`.

---

## 4. Input and Output Guardrails

To prevent compliance risks and output deviations, LOGOS implements input/output guardrails:

*   **Input Guardrails**: Incoming queries are scanned to ensure no API keys, connection strings, or customer database credentials are submitted. The system automatically redacts suspected PII patterns.
*   **Output Guardrails**: Output JSON content is verified to ensure structured compliance. A custom cleaning utility parses reasoning blocks (removing `<think>...</think>` tags) and repairs malformed JSON tokens using the `json-repair` library before final display.

---

## 5. Deployment Story: Azure Container Registry & Hosted Agent Service

LOGOS is designed to transition from local prototype to cloud-scale deployment using **Hosted Agents in Foundry Agent Service**:

*   **Containerization**: The project includes a `Dockerfile` and `docker-compose.yml` to package the FastAPI web server into a lightweight, production-ready container image.
*   **Azure Container Registry (`reasoningagentregistry`)**: The container image is configured to be pushed to the deployed Azure Container Registry `reasoningagentregistry` (accessible via `reasoningagentregistry.azurecr.io`).
*   **Hosted Agent Service**: The container is pulled by the Foundry Agent Service, which provisions compute, assigns Entra ID managed identities, and exposes a secure dedicated endpoint for the research pipeline.

---

## 6. Telemetry, Observability, and Evaluation

The system integrates observability tools to monitor multi-agent execution:

*   **Structured Logging**: Utilizes `structlog` to output structured JSON logs mapping agent status changes, execution durations, and warning events.
*   **Confidence Metrics**: The Analyst and Writer agents calculate data confidence scores based on source consistency and evidence strength, returning them in the metadata of the strategy report.
*   **CI/CD Verification**: An automated test suite in `tests/test_agents.py` validates agent orchestration flows and response schemas, ensuring stability before container builds are pushed to the registry.
