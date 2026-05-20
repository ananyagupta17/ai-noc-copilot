# AI NOC Copilot

An intelligent network operations assistant for Tata Communications.
Built with LangGraph, RAG, and MCP tool orchestration.

## Structure

```
noc_copilot/
├── data/
│   ├── incidents/       # Synthetic incident JSON files
│   ├── logs/            # Synthetic device/network logs
│   ├── alerts/          # Synthetic alert JSON files
│   ├── runbooks/        # Troubleshooting SOPs (text/PDF)
│   └── topology/        # Network topology graph (JSON)
├── db/
│   ├── noc.db           # SQLite database
│   └── chroma/          # ChromaDB vector store
├── rag/
│   ├── indexer.py       # Chunk + embed runbooks
│   └── retriever.py     # Query vector store
├── agent/
│   ├── graph.py         # LangGraph agent definition
│   ├── state.py         # Agent state schema
│   └── tools/           # MCP tool implementations
│       ├── incidents.py
│       ├── logs.py
│       ├── topology.py
│       ├── metrics.py
│       └── runbooks.py
├── api/
│   └── main.py          # FastAPI app
├── ui/
│   └── app.py           # Streamlit dashboard
├── scripts/
│   ├── generate_data.py # Synthetic data generator
│   └── init_db.py       # DB + vector store setup
├── config.py
├── requirements.txt
└── .env.example
```

## Quickstart

```bash
cp .env.example .env
# Add your OpenAI key to .env

pip install -r requirements.txt
python scripts/generate_data.py
python scripts/init_db.py
uvicorn api.main:app --reload
streamlit run ui/app.py
```
