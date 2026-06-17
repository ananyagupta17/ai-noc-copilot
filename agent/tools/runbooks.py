"""
Tool: Runbook retrieval
Queries ChromaDB to find relevant runbook sections for a given symptom.
This is the RAG retrieval step — the agent calls this to get grounding context.
"""

import hashlib
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHROMA_PATH, RUNBOOKS_DIR, TOP_K_RETRIEVAL

import chromadb
from chromadb import EmbeddingFunction, Embeddings


# Same embedding function used in init_db.py
# When you have an OpenAI key, swap this class for OpenAIEmbeddingFunction
class LocalHashEF(EmbeddingFunction):
    def __init__(self): pass
    def __call__(self, input: List[str]) -> Embeddings:
        out = []
        for text in input:
            h = hashlib.sha256(text.encode()).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            arr = np.pad(arr, (0, 384 - len(arr)))
            arr = arr / (np.linalg.norm(arr) + 1e-9)
            out.append(arr)
        return out


# Load ChromaDB collection once
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection("runbooks", embedding_function=LocalHashEF())
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