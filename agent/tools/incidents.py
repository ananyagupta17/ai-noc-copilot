"""
Tool: Incident lookup
Queries SQLite for incident records.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SQLITE_PATH


def _connect():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def get_incident(incident_id: str) -> dict:
    """
    Fetch a single incident by ID.
    Returns the full incident record or an error dict.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    conn.close()

    if not row:
        return {"error": f"Incident {incident_id} not found"}

    result = dict(row)
    result["tags"] = json.loads(result.get("tags", "[]"))
    return result


def find_similar_incidents(symptom: str = None, region: str = None,
                           root_cause: str = None, limit: int = 5) -> list:
    """
    Find historical incidents matching symptom / region / root_cause.
    Agent uses this to ground its RCA in real past events.
    """
    conn = _connect()

    conditions = []
    params = []

    if symptom:
        conditions.append("symptom = ?")
        params.append(symptom)
    if region:
        conditions.append("region = ?")
        params.append(region)
    if root_cause:
        conditions.append("root_cause = ?")
        params.append(root_cause)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"""
        SELECT incident_id, severity, region, affected_device,
               symptom, root_cause, description, resolution, mttr_minutes,
               customer_segment, affected_customers, detected_at, resolved_at
        FROM incidents
        {where}
        ORDER BY detected_at DESC
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_recent_incidents(region: str = None, severity: str = None,
                         limit: int = 10) -> list:
    """
    Get the most recent incidents, optionally filtered by region or severity.
    """
    conn = _connect()
    conditions, params = [], []

    if region:
        conditions.append("region = ?")
        params.append(region)
    if severity:
        conditions.append("severity = ?")
        params.append(severity)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = conn.execute(f"""
        SELECT incident_id, severity, region, affected_device,
               symptom, description, detected_at, mttr_minutes
        FROM incidents {where}
        ORDER BY detected_at DESC LIMIT ?
    """, params + [limit]).fetchall()
    conn.close()

    return [dict(r) for r in rows]