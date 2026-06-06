# Research-to-Report Multi-Agent System

## Architecture Documentation

This document provides detailed architecture information for the multi-agent research system, including system design, component interactions, and deployment architecture.

---

## 1. System Overview

The **Research-to-Report Multi-Agent System** is a sophisticated agent orchestration platform that transforms complex research queries into structured, well-reasoned reports through coordinated multi-agent collaboration.

### 1.1 Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Agent Autonomy** | Each agent operates independently with clear responsibilities |
| **Structured Communication** | A2A protocol enables reliable inter-agent messaging |
| **Tool Abstraction** | MCP provides unified access to external capabilities |
| **Reasoning Transparency** | LangGraph enables visible reasoning chains |
| **Production Readiness** | Azure Foundry provides scalable deployment |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                         (API / Web / CLI Input)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    LangGraph State Machine                          │    │
│  │                                                                      │    │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │    │
│  │   │    PLAN      │────►│   RESEARCH   │────►│   ANALYZE    │       │    │
│  │   └──────────────┘     └──────────────┘     └──────────────┘       │    │
│  │         │                    │                    │                │    │
│  │         │                    │                    │                │    │
│  │         ▼                    ▼                    ▼                │    │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │    │
│  │   │   VALIDATE   │     │   GATHER     │     │   EXTRACT    │       │    │
│  │   │     PLAN     │     │   DATA       │     │   INSIGHTS   │       │    │
│  │   └──────────────┘     └──────────────┘     └──────────────┘       │    │
│  │         │                    │                    │                │    │
│  │         └────────────────────┴────────────────────┘                │    │
│  │                              │                                      │    │
│  │                              ▼                                      │    │
│  │   ┌──────────────────────────────────────────────────────────┐    │    │
│  │   │                    WRITE REPORT                           │    │    │
│  │   └──────────────────────────────────────────────────────────┘    │    │
│  │                              │                                      │    │
│  └──────────────────────────────┼──────────────────────────────────────┘    │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│    PLANNER        │  │   RESEARCHER       │  │    ANALYST        │
│    AGENT          │  │   AGENT            │  │    AGENT          │
│                   │  │                   │  │                   │
│ • Task           │  │ • Web Search      │  │ • Pattern        │
│   Decomposition  │  │ • Data Gathering  │  │   Recognition    │
│ • Plan Generation│  │ • Source          │  │ • Risk           │
│ • Validation     │  │   Verification   │  │   Assessment    │
└───────────────────┘  └───────────────────┘  └───────────────────┘
           │                    │                     │
           │                    │                     │
           └────────────────────┼─────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────────┐  ┌───────────────────┐
            │       A2A         │  │       MCP         │
            │    PROTOCOL       │  │     TOOLS         │
            │                   │  │                   │
            │ • Agent Discovery │  │ • Web Search     │
            │ • Method Calls    │  │ • Document       │
            │ • Response Handle │  │   Search         │
            │ • Error Handling  │  │ • Code           │
            │                   │  │   Interpreter    │
            └───────────────────┘  └───────────────────┘
                    │                       │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │   AZURE AI        │
                    │   FOUNDRY         │
                    │                   │
                    │ • Agent Hosting   │
                    │ • Auto-scaling    │
                    │ • Identity/Auth   │
                    │ • Monitoring      │
                    └───────────────────┘
```

---

## 3. Agent Architecture

### 3.1 Agent Design Pattern

```
┌────────────────────────────────────────────────────────────────┐
│                         AGENT CORE                             │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    SYSTEM PROMPT                         │  │
│  │  Defines agent role, capabilities, and behavior         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   LLM INTERFACE                          │  │
│  │  Connects to Azure OpenAI (GPT-4o / o3-mini)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  CAPABILITY LAYER                        │  │
│  │                                                          │  │
│  │   Primary Methods:                                       │  │
│  │   • analyze()     • search()        • write()           │  │
│  │   • assess()      • decompose()     • format()          │  │
│  │                                                          │  │
│  │   Secondary Methods:                                     │  │
│  │   • validate()    • verify()        • refine()          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    A2A ENDPOINT                          │  │
│  │  Exposes agent methods via A2A protocol                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Individual Agent Specifications

#### Planner Agent
```
┌─────────────────────────────────────────────────────────────────┐
│ PLANNER AGENT                                                    │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Break down complex queries into executable tasks       │
│                                                                 │
│ Input:  User query (string)                                      │
│ Output: ResearchPlan with subtasks                              │
│                                                                 │
│ Core Methods:                                                    │
│ ├── decompose_task(query: str) -> ResearchPlan                 │
│ │   Breaks query into subtasks with dependencies                │
│ │                                                               │
│ ├── validate_plan(plan: ResearchPlan) -> ValidationResult      │
│ │   Checks plan completeness and feasibility                   │
│ │                                                               │
│ └── refine_plan(plan: ResearchPlan, feedback: str) -> ResearchPlan│
│     Adjusts plan based on feedback                              │
│                                                                 │
│ A2A Endpoint: /planner                                           │
│ Capabilities: task_execution, coordination                      │
└─────────────────────────────────────────────────────────────────┘
```

#### Researcher Agent
```
┌─────────────────────────────────────────────────────────────────┐
│ RESEARCHER AGENT                                                │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Gather information from web and documents              │
│                                                                 │
│ Input:  Search queries, research plan                           │
│ Output: ResearchResults with ranked sources                     │
│                                                                 │
│ Core Methods:                                                    │
│ ├── search(query: str, max_results: int) -> ResearchResults     │
│ │   Executes web search and ranks results                      │
│ │                                                               │
│ ├── batch_search(queries: list[str]) -> list[ResearchResults]  │
│ │   Parallel search for multiple queries                       │
│ │                                                               │
│ └── fetch_document(url: str) -> dict                            │
│     Retrieves and extracts content from URLs                    │
│                                                                 │
│ A2A Endpoint: /researcher                                        │
│ Capabilities: data_retrieval                                    │
│ Tools: MCP Web Search, Azure AI Search                         │
└─────────────────────────────────────────────────────────────────┘
```

#### Analyst Agent
```
┌─────────────────────────────────────────────────────────────────┐
│ ANALYST AGENT                                                   │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Analyze data, extract insights, assess risks            │
│                                                                 │
│ Input:  Research data, key findings                             │
│ Output: AnalysisResults with insights and risk assessments      │
│                                                                 │
│ Core Methods:                                                    │
│ ├── analyze(query: str, research_data: dict) -> AnalysisResults │
│ │   Performs comprehensive analysis on research data            │
│ │                                                               │
│ ├── assess_risk(risk_data: dict) -> RiskAssessment              │
│ │   Detailed evaluation of specific risks                      │
│ │                                                               │
│ └── verify_findings(findings, sources) -> VerificationResult   │
│     Self-verification of findings against sources              │
│                                                                 │
│ A2A Endpoint: /analyst                                          │
│ Capabilities: analysis, risk_assessment                         │
└─────────────────────────────────────────────────────────────────┘
```

#### Writer Agent
```
┌─────────────────────────────────────────────────────────────────┐
│ WRITER AGENT                                                    │
├─────────────────────────────────────────────────────────────────┤
│ Purpose: Generate structured, professional reports              │
│                                                                 │
│ Input:  Analysis results, key insights                          │
│ Output: GeneratedReport with formatted sections                │
│                                                                 │
│ Core Methods:                                                    │
│ ├── generate_report(query, analysis_results, format) -> Report  │
│ │   Creates comprehensive report from analysis                 │
│ │                                                               │
│ └── format_output(report, format_type) -> str                   │
│     Formats report as JSON, Markdown, or HTML                  │
│                                                                 │
│ A2A Endpoint: /writer                                           │
│ Capabilities: reporting                                          │
│ Output Formats: JSON, Markdown, HTML, Executive Summary         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Communication Architecture

### 4.1 A2A Protocol Flow

```
┌─────────────┐                    ┌─────────────┐                    ┌─────────────┐
│   AGENT A   │                    │   A2A       │                    │   AGENT B   │
│  (Caller)   │                    │   SERVER    │                    │  (Callee)   │
└─────────────┘                    └─────────────┘                    └─────────────┘
       │                               │                                │
       │  1. POST /call                │                                │
       │  {method, params}             │                                │
       │──────────────────────────────►                                │
       │                               │                                │
       │                               │  2. Route Request             │
       │                               │───────────────────────────────►
       │                               │                                │
       │                               │  3. Execute Handler            │
       │                               │        ┌─────────────┐         │
       │                               │        │ Agent Logic │         │
       │                               │        └─────────────┘         │
       │                               │                                │
       │                               │  4. Return Response            │
       │                               │◄──────────────────────────────
       │                               │                                │
       │  5. POST /call response       │                                │
       │  {status, result}            │                                │
       │◄──────────────────────────────│                                │
       │                               │                                │
```

### 4.2 Message Format

```json
{
  "message_id": "msg_20240605_143000123456",
  "message_type": "request",
  "sender": "planner",
  "receiver": "researcher",
  "method": "search",
  "params": {
    "query": "Indian EV market risks",
    "max_results": 10
  },
  "timestamp": "2024-06-05T14:30:00.123Z",
  "correlation_id": "workflow_123"
}
```

---

## 5. LangGraph State Machine

### 5.1 Workflow States

```
                    ┌─────────────────┐
                    │   QUERY_INPUT   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      PLAN       │──────► Task Decomposition
                    │   (Planner)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  VALIDATE_PLAN  │──────► Check Feasibility
                    │   (Planner)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐   ┌─────────────┐   ┌──────┐
        │ PROCEED │   │   REFINE    │   │ FAIL │
        └────┬────┘   └──────┬──────┘   └───┬──┘
             │              │              │
             │              ▼              │
             │      ┌─────────────┐        │
             │      │    PLAN     │        │
             │      │   (Again)   │        │
             │      └──────┬──────┘        │
             │             │              │
             │             └──────────────► │
             │                              │
             ▼                              ▼
      ┌─────────────┐               ┌─────────────┐
      │  RESEARCH   │──────────────►│   ANALYZE   │
      │ (Researcher)│               │  (Analyst)  │
      └──────┬──────┘               └──────┬──────┘
             │                              │
             │                              │
             │                              ▼
             │                      ┌─────────────┐
             │                      │WRITE_REPORT │
             │                      │  (Writer)   │
             │                      └──────┬──────┘
             │                             │
             └─────────────────────────────►│
                                            │
                                            ▼
                                     ┌─────────────┐
                                     │   OUTPUT    │
                                     │   (Report)  │
                                     └─────────────┘
```

### 5.2 State Schema

```python
class WorkflowState(TypedDict):
    # Core state
    query: str
    plan: ResearchPlan | None
    research_results: ResearchResults | None
    analysis_results: AnalysisResults | None
    report: GeneratedReport | None

    # Progress tracking
    current_task: str | None
    completed_tasks: list[str]
    failed_tasks: list[str]

    # Error handling
    errors: list[dict]
    retry_count: int

    # Metadata
    start_time: str
    end_time: str | None
    confidence_scores: dict[str, float]
```

---

## 6. MCP Tool Integration

### 6.1 MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT LAYER                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Planner │  │Research │  │ Analyst │  │ Writer  │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
└───────┼───────────┼───────────┼───────────┼───────────────────┘
        │           │           │           │
        └───────────┴─────┬─────┴───────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │         MCP CLIENT LAYER              │
        │                                        │
        │   ┌─────────────┐    ┌─────────────┐  │
        │   │ MCPWebSearch│    │MCPDocSearch │  │
        │   │     Tool    │    │     Tool    │  │
        │   └──────┬──────┘    └──────┬──────┘  │
        └──────────┼─────────────────┼──────────┘
                   │                 │
                   ▼                 ▼
        ┌──────────────────────────────────────┐
        │        MCP SERVER LAYER              │
        │                                       │
        │   mcp.ai.azure.com                    │
        │   - Web Search API                   │
        │   - Document Search API              │
        │   - Code Interpreter API             │
        └──────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────────────┐
        │         EXTERNAL SERVICES             │
        │                                       │
        │   • Bing Search                       │
        │   • Azure AI Search                   │
        │   • Code Execution Runtime            │
        └──────────────────────────────────────┘
```

---

## 7. Azure AI Foundry Deployment

### 7.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AZURE CLOUD                                     │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    RESOURCE GROUP                                  │ │
│  │                                                                     │ │
│  │   ┌────────────────────────────────────────────────────────────┐  │ │
│  │   │              AZURE AI FOUNDRY PROJECT                        │  │ │
│  │   │                                                              │  │ │
│  │   │   ┌──────────────────────────────────────────────────────┐  │  │ │
│  │   │   │              HOSTED AGENTS                             │  │  │ │
│  │   │   │                                                        │  │  │ │
│  │   │   │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │  │  │ │
│  │   │   │   │Planner │ │Research│ │Analyst │ │Writer  │         │  │  │ │
│  │   │   │   │ Agent  │ │ Agent  │ │ Agent  │ │ Agent  │         │  │  │ │
│  │   │   │   └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘         │  │  │ │
│  │   │   │        │         │         │         │               │  │  │ │
│  │   │   │        └─────────┴────┬────┴─────────┘               │  │  │ │
│  │   │   │                        │                              │  │  │ │
│  │   │   │                        │ A2A Protocol                │  │  │ │
│  │   │   │                        │                              │  │  │ │
│  │   │   └────────────────────────┼──────────────────────────────┘  │  │ │
│  │   │                            │                                  │  │ │
│  │   └────────────────────────────┼──────────────────────────────────┘  │ │
│  │                                │                                     │ │
│  └────────────────────────────────┼─────────────────────────────────────┘ │
│                                   │                                        │
│  ┌────────────────────────────────┼─────────────────────────────────────┐ │
│  │                           DATA SERVICES                              │ │
│  │                                                                         │ │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │ │
│  │   │  Azure OpenAI   │  │ Azure AI Search │  │   Azure CDN     │     │ │
│  │   │   (GPT-4o)      │  │   (Documents)   │  │   (Static)      │     │ │
│  │   └──────────────────┘  └──────────────────┘  └──────────────────┘     │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Scaling Configuration

| Component | SKU | Autoscaling | Notes |
|-----------|-----|-------------|-------|
| Planner Agent | Standard | Yes | CPU-bound, moderate load |
| Researcher Agent | Standard | Yes | I/O-bound, high load |
| Analyst Agent | Standard | Yes | CPU-bound, moderate load |
| Writer Agent | Standard | Yes | CPU-bound, moderate load |
| A2A Server | Basic | Yes | Low resource usage |
| MCP Server | (External) | N/A | Managed by Azure |

---

## 8. Security Architecture

### 8.1 Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUTHENTICATION FLOW                         │
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ │
│  │   User    │───►│   Azure   │───►│   Foundry │───►│  Agent │ │
│  │  Request  │    │   AD      │    │   Project │    │  Call  │ │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ │
│                        │                 │                      │
│                        ▼                 ▼                      │
│                   ┌─────────┐       ┌─────────┐                  │
│                   │Validate │       │Validate │                  │
│                   │ Token   │       │ Managed │                  │
│                   │         │       │ Identity│                  │
│                   └─────────┘       └─────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Security Controls

| Control | Implementation |
|---------|----------------|
| Authentication | Azure Active Directory |
| Authorization | Role-based access control |
| Encryption | TLS 1.2+ in transit |
| Secrets | Azure Key Vault |
| Network | Private endpoints (optional) |
| Monitoring | Azure Monitor, Defender |

---

## 9. Monitoring and Observability

### 9.1 Metrics Collection

| Metric | Source | Destination |
|--------|--------|-------------|
| Agent latency | Instrumentation | Azure Monitor |
| A2A call count | A2A Server logs | Log Analytics |
| Error rate | Exception handling | Application Insights |
| Token usage | OpenAI API | Cost Management |
| MCP calls | MCP Client logs | Log Analytics |

### 9.2 Logging Structure

```json
{
  "timestamp": "2024-06-05T14:30:00.123Z",
  "level": "INFO",
  "logger": "researcher",
  "event": "search_complete",
  "query": "Indian EV market",
  "results_count": 10,
  "latency_ms": 150,
  "correlation_id": "workflow_123"
}
```

---

## 10. Error Handling

### 10.1 Error Flow

```
                    ┌──────────────┐
                    │     ERROR    │
                    │   DETECTED   │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Error Classification │
              └────────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  RETRY    │    │  FALLBACK │    │   FAIL    │
    │ (Transient│    │ (Non-    │    │ (Fatal)   │
    │  Errors)  │    │  critical)│    │           │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                │                │
          ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │  Re-run   │    │  Use      │    │  Return   │
    │  Task     │    │  Cached   │    │  Error    │
    │           │    │  Data     │    │  Response │
    └───────────┘    └───────────┘    └───────────┘
```

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-06-05 | Initial architecture documentation |
| 1.0.1 | 2024-06-05 | Added deployment architecture |
| 1.0.2 | 2024-06-05 | Added security architecture |

---

**Document Version:** 1.0.2
**Last Updated:** June 5, 2024
**Authors:** Multi-Agent Research Team