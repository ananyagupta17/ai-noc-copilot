# 🛡️ AI NOC Copilot

> An agentic AI system for network operations.
> Investigates incidents, correlates alerts, retrieves runbooks via RAG,
> and generates explainable root cause analyses with confidence scoring.

---

## What It Does

Network Operations Centers (NOCs) at large telecom companies receive hundreds of alerts per hour. Engineers manually correlate signals, search runbooks, diagnose root causes, and write incident reports — a process that takes 30 minutes to 2 hours per incident and is mentally exhausting at scale.

The AI NOC Copilot automates this investigation loop:

1. An engineer describes an incident in natural language (or a structured alert is ingested)
2. The system correlates related alerts and filters noise
3. A LangGraph agent investigates systematically — calling tools, retrieving runbooks, analysing topology
4. The agent produces a structured RCA with a computed confidence score, evidence trail, incident timeline, and recommended actions
5. Everything is visible in a real-time dashboard with full AI observability

---

## Architecture

```
Incident / Alert Input
        ↓
Alert Correlation Engine        ← TF-IDF embeddings + agglomerative clustering
        ↓                           groups noisy alerts into clean clusters
LangGraph Agent                 ← Gemini 2.5 Flash orchestrated via LangGraph
        ↓  ↑  (investigation loop)
   Agent Tools        RAG Pipeline
   ─────────          ────────────
   get_incident       ChromaDB vector store
   search_logs        8 hand-written runbooks
   get_topology       Cosine similarity retrieval
   get_blast_radius
   get_device_metrics
   find_similar_incidents
        ↓
Evidence Scoring Engine         ← weighted confidence score, not LLM-asserted
        ↓
Timeline Reconstruction         ← chronological event sequence from logs + alerts
        ↓
RCA Output                      ← cause · confidence · evidence · impact · actions
        ↓
FastAPI (REST + WebSocket) → Streamlit Dashboard
```

---

## Features

### Core Agent
- **LangGraph investigation loop** — multi-step agentic reasoning with up to 10 tool calls per investigation
- **14 tools** across 6 modules: incidents, alerts, logs, topology, metrics, runbooks
- **Gemini 2.5 Flash** with API key rotation across 3 project quotas for uninterrupted operation

### Alert Correlation Engine
- Embeds alert messages using TF-IDF vectorisation with bigram support
- Clusters similar alerts using average-linkage agglomerative clustering on a
  cosine distance matrix (DBSCAN retained as an automatic fallback)
- Reduces alert noise before the agent investigates — solving the #1 NOC pain point

### Explainable RCA with Evidence Scoring
- Every claim is backed by a typed evidence item with a weight
- Confidence score is computed from weighted evidence, not asserted by the LLM
- Evidence types: `log` (0.30) · `historical_incident` (0.25) · `runbook` (0.20) · `alert` (0.15) · `topology` (0.10) · `metric` (0.10)
- Type caps prevent single-source inflation

### Incident Timeline Reconstruction
- Parses timestamps from syslog and ISO formats across logs and alerts
- Classifies events using 16 regex patterns (BGP drop, interface down, CRC error, optical degradation, etc.)
- Deduplicates events within a 30-second window
- Infers customer impact milestone automatically when first CRITICAL event fires

### Live Investigation Streaming
- The UI streams agent activity in real time as the investigation runs — no waiting for a spinner to resolve
- Each node emits typed events (`start`, `reason`, `tool`, `output`) polled every 1.5 seconds from the server
- Engineers see exactly which tool is being called, which reasoning loop is running, and whether the agent hit a rate limit and rotated keys — before the RCA is ready
- After the investigation completes the trace collapses into an expandable "Last investigation trace" panel

### AI Observability Layer
- **Runtime tracer** — records every tool call with timing, input params, and output summary
- **RAG quality scorer** — tracks retrieval quality per query
- **Loop recorder** — captures confidence evolution across reasoning loops
- **Structured JSONL audit log** — persistent, queryable, one line per event
- All visible in the Streamlit observability panel

### Data Layer
- 80 synthetic incidents, 277 alerts, 40 syslog files, 8 runbooks, 33-node topology graph
- Generated with Faker + NetworkX + hand-written SOPs
- Stored in SQLite (structured) + ChromaDB (vectors)

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash (via LangChain) |
| Agent framework | LangGraph |
| RAG | ChromaDB + all-MiniLM-L6-v2 embeddings |
| Alert correlation | scikit-learn (agglomerative clustering, DBSCAN fallback) + NumPy |
| Backend | FastAPI + WebSockets |
| Frontend | Streamlit |
| Storage | SQLite + ChromaDB |
| Topology | NetworkX |
| Data generation | Faker |

---

## Project Structure

```
noc_copilot/
├── agent/
│   ├── graph.py           # LangGraph investigation loop (4 nodes)
│   ├── state.py           # AgentState + RCAOutput Pydantic schemas
│   ├── evidence.py        # Weighted confidence scoring
│   ├── timeline.py        # Incident timeline reconstruction
│   └── tools/
│       ├── __init__.py    # Tool registry with @tool decorators
│       ├── incidents.py   # SQLite incident queries
│       ├── alerts.py      # SQLite alert queries
│       ├── logs.py        # Syslog file search
│       ├── topology.py    # NetworkX graph traversal
│       ├── metrics.py     # Device metrics (simulated)
│       └── runbooks.py    # ChromaDB RAG retrieval
├── alert_correlation/
│   ├── embedder.py        # TF-IDF alert vectorisation
│   └── clusterer.py       # agglomerative clustering (DBSCAN fallback) + noise reduction
├── observability/
│   ├── tracer.py          # Runtime tool call + RAG tracing
│   └── logger.py          # Persistent JSONL audit log
├── api/
│   └── main.py            # FastAPI — 17 REST endpoints + WebSocket
├── ui/
│   └── app.py             # Streamlit dashboard — 4 panels
├── scripts/
│   ├── generate_data.py   # Synthetic data generator
│   └── init_db.py         # SQLite + ChromaDB initialisation
├── data/
│   ├── incidents/         # 80 incident JSON records
│   ├── alerts/            # 277 alert JSON records
│   ├── logs/              # 40 syslog files
│   ├── runbooks/          # 8 SOP text files
│   └── topology/          # Network graph JSON
├── db/
│   ├── noc.db             # SQLite database
│   └── chroma/            # ChromaDB vector store
├── config.py
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites
- Python 3.10+
- A Google AI Studio API key (free at https://aistudio.google.com)

### Installation

```bash
git clone https://github.com/ananyagupta17/noc_copilot.git
cd noc_copilot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GOOGLE_API_KEY_1=your-first-key
GOOGLE_API_KEY_2=your-second-key
GOOGLE_API_KEY_3=your-third-key
```

### Generate Data and Initialise Database

```bash
python scripts/generate_data.py
python scripts/init_db.py
```

### Run

Terminal 1 — API:
```bash
uvicorn api.main:app --reload --port 8000
```

Terminal 2 — UI:
```bash
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.
FastAPI interactive docs at http://localhost:8000/docs.

---

## Usage

### Investigating an Incident

Enter a natural language description in the sidebar:

```
Singapore enterprise customers reporting intermittent connectivity.
NOC monitoring shows packet loss exceeding 15% on edge routers.
BGP session instability observed on SIN-CR-01. Latency spiked to
280ms on normally sub-50ms paths. Enterprise MPLS customers affected.
```

Select a region filter, keep alert correlation ON, and click Investigate.

### Running Tests

```bash
python test_run.py
```

Runs 6 test suites: agent tools, alert correlation, agent investigation,
evidence scoring, timeline reconstruction, and observability.

---

## Sample Output

```
Status       : Complete
Tool calls   : 17
Evidence     : 13 items
Loops        : 4
Confidence   : 86%

Probable cause:
  Physical layer degradation on Singapore backbone link causing CRC errors,
  BGP session instability, and downstream packet loss for enterprise customers.

Recommended actions:
  1. Dispatch field team to inspect optical transceivers on SIN-ER-01
  2. Reroute traffic via backup MPLS path immediately
  3. Run OTDR diagnostics on Singapore backbone fibre
  4. Monitor BGP stabilisation after reroute

Escalation: Fiber Operations Team
```

---

## Future Expansion

- Real-time Kafka ingestion from live monitoring systems
- Graph database (Neo4j) for topology-aware GraphRAG
- Predictive failure detection using historical incident patterns
- Autonomous remediation with approval workflows
- SIEM integration for security-correlated incident analysis
- Shift handoff automation and postmortem report generation

---
Built with LangGraph · Gemini 2.5 Flash · ChromaDB · FastAPI · Streamlit
