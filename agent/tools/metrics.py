"""
Tool: Device metrics
Returns simulated real-time metrics for devices.
In production this would call a real monitoring API (Prometheus, SolarWinds).
For now we generate realistic values seeded by device name.
"""

import hashlib
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

# Remove from config import TOPOLOGY_DIR from here!

# Load device list for validation
_devices = None

def _get_known_devices():
    global _devices
    if _devices is None:
        # MOVE THE IMPORT HERE inside the function scope
        from config import TOPOLOGY_DIR 
        
        data = json.loads((TOPOLOGY_DIR / "topology.json").read_text())
        _devices = {n["id"] for n in data["nodes"]}
    return _devices

def _seed_for_device(device_id: str) -> random.Random:
    """Deterministic RNG per device so metrics are consistent across calls."""
    seed = int(hashlib.md5(device_id.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def get_device_metrics(device_id: str) -> dict:
    """
    Get current performance metrics for a device.
    Returns CPU, memory, interface utilization, packet stats.
    """
    known = _get_known_devices()
    if device_id not in known:
        return {"error": f"Device {device_id} not found"}

    rng = _seed_for_device(device_id)

    # Slightly randomize on each call to simulate live data
    jitter = random.uniform(0.9, 1.1)

    cpu = round(min(99, rng.uniform(10, 60) * jitter), 1)
    memory = round(min(99, rng.uniform(30, 75) * jitter), 1)
    bandwidth_in = round(rng.uniform(100, 9000) * jitter, 1)   # Mbps
    bandwidth_out = round(rng.uniform(100, 9000) * jitter, 1)
    packet_loss = round(max(0, rng.uniform(-1, 5) * jitter), 2)
    latency_ms = round(rng.uniform(1, 80) * jitter, 1)
    interface_errors = int(rng.uniform(0, 200) * jitter)

    return {
        "device_id": device_id,
        "timestamp": datetime.now().isoformat(),
        "cpu_utilization_pct": cpu,
        "memory_utilization_pct": memory,
        "bandwidth_in_mbps": bandwidth_in,
        "bandwidth_out_mbps": bandwidth_out,
        "packet_loss_pct": packet_loss,
        "latency_ms": latency_ms,
        "interface_errors_per_min": interface_errors,
        "status": _derive_status(cpu, memory, packet_loss)
    }


def _derive_status(cpu, memory, packet_loss):
    if cpu > 90 or memory > 90 or packet_loss > 10:
        return "CRITICAL"
    elif cpu > 75 or memory > 75 or packet_loss > 3:
        return "DEGRADED"
    return "NORMAL"


def get_metrics_history(device_id: str, hours: int = 6) -> dict:
    """
    Simulated historical metrics — returns hourly snapshots.
    Useful for the agent to spot trends (is CPU rising over time?).
    """
    known = _get_known_devices()
    if device_id not in known:
        return {"error": f"Device {device_id} not found"}

    rng = _seed_for_device(device_id)
    history = []
    now = datetime.now()

    for i in range(hours, 0, -1):
        t = now - timedelta(hours=i)
        history.append({
            "timestamp": t.isoformat(),
            "cpu_pct": round(rng.uniform(10, 85), 1),
            "memory_pct": round(rng.uniform(30, 80), 1),
            "packet_loss_pct": round(max(0, rng.uniform(-1, 8)), 2),
            "latency_ms": round(rng.uniform(1, 100), 1),
        })

    return {
        "device_id": device_id,
        "hours_analyzed": hours,
        "history": history
    }