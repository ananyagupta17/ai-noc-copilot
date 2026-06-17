"""
Tool: Runbook retrieval
Queries ChromaDB to find relevant runbook sections for a given symptom.
This is the RAG retrieval step — the agent calls this to get grounding context.

Retrieval is semantic: both the indexed runbook chunks and the incoming query
are embedded with the all-MiniLM-L6-v2 sentence-transformer (384-dim), run
locally via ONNX through ChromaDB's DefaultEmbeddingFunction — no API key and
no rate limit. The same embedding function MUST be used here and in
scripts/init_db.py so query vectors live in the same space as the chunks.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHROMA_PATH, RUNBOOKS_DIR, TOP_K_RETRIEVAL

import chromadb
from chromadb.utils import embedding_functions

# all-MiniLM-L6-v2 via ONNX — local, semantic, 384-dim. Must match init_db.py.
_EMBEDDING_FN = embedding_functions.DefaultEmbeddingFunction()


# Load ChromaDB collection once
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection("runbooks", embedding_function=_EMBEDDING_FN)
    return _collection


def get_runbook(symptom: str, n_results: int = TOP_K_RETRIEVAL) -> dict:
    """
    Retrieve the most relevant runbook chunks for a symptom description.
    The agent passes a natural language query like:
    'packet loss on Singapore edge router with CRC errors'
    and gets back the most relevant SOP sections.
    """
    col = _get_collection()

    results = col.query(
        query_texts=[symptom],
        n_results=min(n_results, col.count())
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "content": doc
        })

    return {
        "query": symptom,
        "chunks_retrieved": len(chunks),
        "results": chunks
    }


def list_available_runbooks() -> dict:
    """
    List all runbooks available in the knowledge base.
    Agent can call this first to understand what SOPs exist.
    """
    files = sorted(RUNBOOKS_DIR.glob("*.txt"))
    runbooks = []
    for f in files:
        lines = f.read_text().splitlines()
        # Extract title and category from first few lines
        title = next((l.replace("RUNBOOK:", "").strip() for l in lines if l.startswith("RUNBOOK:")), f.stem)
        category = next((l.replace("Category:", "").strip() for l in lines if l.startswith("Category:")), "General")
        runbooks.append({
            "filename": f.name,
            "title": title,
            "category": category
        })

    return {
        "total": len(runbooks),
        "runbooks": runbooks
    }