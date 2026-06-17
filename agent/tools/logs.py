"""
Tool: Log search
Reads raw syslog files from data/logs/.
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import LOGS_DIR


def search_logs(incident_id: str = None, keyword: str = None,
                device: str = None, limit: int = 30) -> dict:
    """
    Search log files by incident ID, keyword, or device name.
    Returns matching lines with their source file.

    - incident_id: reads the specific log file for that incident
    - keyword: grep-style search across all log files
    - device: filter lines containing the device hostname
    """
    results = []

    if incident_id:
        # Direct lookup — one file per incident
        log_file = LOGS_DIR / f"{incident_id}.log"
        if not log_file.exists():
             # Fall back to keyword search using incident_id as keyword
            return search_logs(keyword=incident_id.split('-')[1], 
                             limit=limit)
        lines = log_file.read_text().splitlines()
        if keyword:
            lines = [l for l in lines if keyword.lower() in l.lower()]
        results = [{"source": log_file.name, "line": l} for l in lines[:limit]]

    else:
        # Scan across all log files
        log_files = sorted(LOGS_DIR.glob("*.log"))
        for lf in log_files:
            lines = lf.read_text().splitlines()
            for line in lines:
                match = True
                if keyword and keyword.lower() not in line.lower():
                    match = False
                if device and device.lower() not in line.lower():
                    match = False
                if match:
                    results.append({"source": lf.name, "line": line})
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

    return {
        "total_lines": len(results),
        "lines": results
    }


def get_error_summary(incident_id: str) -> dict:
    """
    Summarize error types found in a log file.
    Returns counts of each Cisco/Juniper error code found.
    Helps agent quickly understand what kind of failure occurred.
    """
    log_file = LOGS_DIR / f"{incident_id}.log"
    if not log_file.exists():
        return {"error": f"No log file for {incident_id}"}

    lines = log_file.read_text().splitlines()

    # Extract Cisco-style error codes e.g. %BGP-5-ADJCHANGE
    pattern = re.compile(r"%([A-Z]+-\d+-[A-Z_]+)")
    counts = {}
    for line in lines:
        match = pattern.search(line)
        if match:
            code = match.group(1)
            counts[code] = counts.get(code, 0) + 1

    return {
        "incident_id": incident_id,
        "total_log_lines": len(lines),
        "error_codes": counts,
        "dominant_error": max(counts, key=counts.get) if counts else None
    }