"""
Tool: Alert lookup
Queries SQLite for alert records linked to incidents.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SQLITE_PATH


def _connect():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_alerts_for_incident(incident_id: str) -> list:
    """
    Get all alerts linked to a specific incident.
    Shows the agent how many signals fired and how severe they were.
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT alert_id, device, severity, alert_type,
               message, metric_name, metric_value, timestamp, source
        FROM alerts
        WHERE incident_id = ?
        ORDER BY timestamp ASC
    """, (incident_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_alerts_by_device(device: str, limit: int = 20) -> list:
    """
    Get recent alerts for a specific device.
    Helps agent identify if a device is a repeat offender.
    """
    conn = _connect()
    rows = conn.execute("""
        SELECT alert_id, incident_id, severity, alert_type,
               message, metric_name, metric_value, timestamp
        FROM alerts
        WHERE device = ?
        ORDER BY timestamp DESC LIMIT ?
    """, (device, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_critical_alerts(region: str = None, limit: int = 15) -> list:
    """
    Get the most recent CRITICAL alerts, optionally filtered by region.
    Used by agent during initial triage to assess blast radius.
    """
    conn = _connect()
    conditions = ["severity = 'CRITICAL'"]
    params = []

    if region:
        conditions.append("region = ?")
        params.append(region)

    rows = conn.execute(f"""
        SELECT alert_id, incident_id, device, region,
               alert_type, message, metric_value, timestamp
        FROM alerts
        WHERE {' AND '.join(conditions)}
        ORDER BY timestamp DESC LIMIT ?
    """, params + [limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]