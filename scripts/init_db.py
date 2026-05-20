"""
Database initialization for AI NOC Copilot.
- Loads incidents + alerts into SQLite
- Chunks + embeds runbooks into ChromaDB

Run ONCE after generate_data.py:
    python scripts/init_db.py
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    SQLITE_PATH, CHROMA_PATH,
    INCIDENTS_DIR, ALERTS_DIR, RUNBOOKS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
)

# ─────────────────────────────────────────────
# PART 1 — SQLite
# Stores incidents and alerts as structured tables.
# MCP tools will query these with simple SQL.
# ─────────────────────────────────────────────

def init_sqlite():
    print("[SQLite] Creating database...")
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    # --- Incidents table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id       TEXT PRIMARY KEY,
            severity          TEXT,
            region            TEXT,
            affected_device   TEXT,
            symptom           TEXT,
            root_cause        TEXT,
            description       TEXT,
            customer_segment  TEXT,
            affected_customers INTEGER,
            detected_at       TEXT,
            acknowledged_at   TEXT,
            resolved_at       TEXT,
            mttr_minutes      INTEGER,
            resolution        TEXT,
            engineer          TEXT,
            tags              TEXT   -- stored as JSON string
        )
    """)

    # --- Alerts table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id          TEXT PRIMARY KEY,
            incident_id       TEXT,
            device            TEXT,
            region            TEXT,
            severity          TEXT,
            alert_type        TEXT,
            message           TEXT,
            metric_name       TEXT,
            metric_value      REAL,
            threshold_breached INTEGER,
            timestamp         TEXT,
            source            TEXT,
            FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
        )
    """)

    conn.commit()
    print("[SQLite] Tables created.")

    # --- Load incidents ---
    incidents_file = INCIDENTS_DIR / "incidents.json"
    incidents = json.loads(incidents_file.read_text())

    cur.executemany("""
        INSERT OR REPLACE INTO incidents VALUES (
            :incident_id, :severity, :region, :affected_device,
            :symptom, :root_cause, :description, :customer_segment,
            :affected_customers, :detected_at, :acknowledged_at,
            :resolved_at, :mttr_minutes, :resolution, :engineer, :tags
        )
    """, [
        {**inc, "tags": json.dumps(inc.get("tags", []))}
        for inc in incidents
    ])

    conn.commit()
    print(f"[SQLite] Loaded {len(incidents)} incidents.")

    # --- Load alerts ---
    alerts_file = ALERTS_DIR / "alerts.json"
    alerts = json.loads(alerts_file.read_text())

    cur.executemany("""
        INSERT OR REPLACE INTO alerts VALUES (
            :alert_id, :incident_id, :device, :region, :severity,
            :alert_type, :message, :metric_name, :metric_value,
            :threshold_breached, :timestamp, :source
        )
    """, [
        {
            **a,
            "metric_name":  a.get("metric_value", {}).get("name", ""),
            "metric_value": a.get("metric_value", {}).get("value", 0),
            "threshold_breached": int(a.get("threshold_breached", False)),
        }
        for a in alerts
    ])

    conn.commit()
    print(f"[SQLite] Loaded {len(alerts)} alerts.")
    conn.close()


# ─────────────────────────────────────────────
# PART 2 — ChromaDB
# Chunks each runbook into overlapping windows,
# embeds them, and stores in a local vector DB.
# The RAG retriever will query this collection.
#
# We use sentence-level chunking with overlap so
# that context isn't lost at chunk boundaries.
# ─────────────────────────────────────────────

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks by character count.
    overlap ensures a sentence cut at a boundary still appears
    in the next chunk with enough context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # drop tiny tail chunks


def init_chromadb():
    print("\n[ChromaDB] Setting up vector store...")

    try:
        import chromadb
        from chromadb import EmbeddingFunction, Embeddings
    except ImportError:
        print("  chromadb not installed. Run: pip install chromadb")
        return

    import hashlib
    import numpy as np
    from typing import List

    # Local hash-based embedding — no API key or model download needed.
    # On your machine, swap this for OpenAI embeddings once you have a key
    # (see rag/retriever.py). For init and dev this is fine.
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

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    ef = LocalHashEF()

    # Delete collection if re-running so we start fresh
    try:
        client.delete_collection("runbooks")
    except Exception:
        pass

    collection = client.create_collection(
        name="runbooks",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    runbook_files = list(RUNBOOKS_DIR.glob("*.txt"))
    total_chunks = 0

    for rb_file in runbook_files:
        text = rb_file.read_text()
        chunks = chunk_text(text)

        # Each chunk gets an ID, the source filename as metadata,
        # and the raw text as the document. ChromaDB embeds it.
        ids       = [f"{rb_file.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": rb_file.name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        total_chunks += len(chunks)
        print(f"  {rb_file.name}: {len(chunks)} chunks")

    print(f"[ChromaDB] Indexed {len(runbook_files)} runbooks → {total_chunks} total chunks.")
    print(f"[ChromaDB] Collection '{collection.name}' ready at {CHROMA_PATH}")


# ─────────────────────────────────────────────
# VERIFY — quick sanity checks after setup
# ─────────────────────────────────────────────

def verify():
    print("\n[Verify] Running sanity checks...")

    # SQLite check
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    inc_count = cur.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    alt_count = cur.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    # Sample query — find P1 incidents
    p1s = cur.execute(
        "SELECT incident_id, region, symptom FROM incidents WHERE severity='P1' LIMIT 3"
    ).fetchall()
    conn.close()

    print(f"  SQLite → incidents: {inc_count}, alerts: {alt_count}")
    print(f"  Sample P1 incidents: {p1s}")

    # ChromaDB check
    try:
        import chromadb, hashlib, numpy as np
        from chromadb import EmbeddingFunction, Embeddings
        from typing import List

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

        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col = client.get_collection("runbooks", embedding_function=LocalHashEF())
        results = col.query(query_texts=["packet loss troubleshooting steps"], n_results=2)
        print(f"  ChromaDB → collection size: {col.count()} chunks")
        print(f"  Sample RAG query result sources: {[m['source'] for m in results['metadatas'][0]]}")
    except Exception as e:
        print(f"  ChromaDB check skipped: {e}")

    print("\n✓ Database initialization complete.\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== AI NOC Copilot — DB Initialization ===\n")
    init_sqlite()
    init_chromadb()
    verify()
