"""
agent/tools/__init__.py

Registers all MCP tools with LangGraph using the @tool decorator.
The decorator gives each function a JSON schema so Gemini knows
exactly what parameters to pass when it decides to call a tool.

The docstring of each tool is critical — Gemini reads it to decide
WHEN to call a tool. Write them like instructions to the LLM.
"""

from langchain.tools import tool

from agent.tools.incidents import (
    get_incident,
    find_similar_incidents,
    get_recent_incidents,
)
from agent.tools.alerts import (
    get_alerts_for_incident,
    get_alerts_by_device,
    get_critical_alerts,
)
from agent.tools.logs import search_logs, get_error_summary
from agent.tools.topology import (
    get_device_info,
    get_neighbors,
    get_blast_radius,
    get_path_between,
    get_region_devices,
)
from agent.tools.metrics import get_device_metrics, get_metrics_history
from agent.tools.runbooks import get_runbook, list_available_runbooks


@tool("get_incident")
def tool_get_incident(incident_id: str) -> dict:
    """
    Fetch metadata for a specific incident by its ID (e.g. INC-104332):
    severity, region, affected device, timestamps, and customer impact.
    Use this to establish the facts of the incident you are investigating.

    Note: this does NOT return the root cause or resolution. You must
    DERIVE the root cause yourself from logs, topology, metrics, and
    similar historical incidents — this tool only gives you the case facts.
    """
    # Strip the post-RCA answer fields. The agent must derive the root
    # cause from evidence, not read it off the incident record. The raw
    # get_incident() (used by the API/UI) still returns the full record.
    _LEAKY_FIELDS = ("root_cause", "resolution")
    record = get_incident(incident_id)
    return {k: v for k, v in record.items() if k not in _LEAKY_FIELDS}


@tool("find_similar_incidents")
def tool_find_similar_incidents(
    symptom: str = None,
    region: str = None,
    root_cause: str = None,
    limit: int = 5
) -> list:
    """
    Search historical incidents matching a symptom, region, or root cause.
    Use this early in every investigation to find past incidents that
    resemble the current one. Symptom options: packet_loss, high_latency,
    bgp_flap, interface_down, crc_errors, cpu_spike, memory_exhaustion,
    mpls_failure, dns_outage, optical_degradation.
    """
    return find_similar_incidents(
        symptom=symptom,
        region=region,
        root_cause=root_cause,
        limit=limit
    )


@tool("get_recent_incidents")
def tool_get_recent_incidents(
    region: str = None,
    severity: str = None,
    limit: int = 10
) -> list:
    """
    Get the most recent incidents, optionally filtered by region or severity.
    Use this for initial triage to understand recent activity in an area.
    Severity options: P1, P2, P3, P4.
    """
    return get_recent_incidents(
        region=region,
        severity=severity,
        limit=limit
    )


@tool("get_alerts_for_incident")
def tool_get_alerts_for_incident(incident_id: str) -> list:
    """
    Get all alerts linked to a specific incident ID.
    Use this to see how many signals fired and what metrics breached threshold.
    """
    return get_alerts_for_incident(incident_id)


@tool("get_critical_alerts")
def tool_get_critical_alerts(region: str = None, limit: int = 15) -> list:
    """
    Get the most recent CRITICAL severity alerts, optionally filtered by region.
    Use this during initial triage to understand the current blast radius.
    """
    return get_critical_alerts(region=region, limit=limit)


@tool("search_logs")
def tool_search_logs(
    incident_id: str = None,
    keyword: str = None,
    device: str = None,
    limit: int = 30
) -> dict:
    """
    Search syslog files for an incident or keyword.
    Use incident_id to get all logs for a specific incident.
    Use keyword to search across all logs (e.g. 'BGP', 'CRC', 'MPLS').
    Returns raw syslog lines with timestamps and error codes.
    """
    return search_logs(
        incident_id=incident_id,
        keyword=keyword,
        device=device,
        limit=limit
    )


@tool("get_error_summary")
def tool_get_error_summary(incident_id: str) -> dict:
    """
    Get a summary of error codes found in the logs for an incident.
    Returns counts of each Cisco/Juniper error code.
    Use this to quickly identify the dominant error type.
    """
    return get_error_summary(incident_id)


@tool("get_blast_radius")
def tool_get_blast_radius(device_id: str, hops: int = 2) -> dict:
    """
    Find all devices within N hops of an affected device.
    Use this to understand how far an outage can cascade.
    Critical for impact analysis and SLA breach assessment.
    """
    return get_blast_radius(device_id=device_id, hops=hops)


@tool("get_neighbors")
def tool_get_neighbors(device_id: str) -> dict:
    """
    Get all directly connected devices for a given device ID.
    Use this to identify upstream and downstream dependencies.
    """
    return get_neighbors(device_id)


@tool("get_region_devices")
def tool_get_region_devices(region: str) -> dict:
    """
    List all network devices in a given region.
    Region options: Singapore, Mumbai, London, Frankfurt,
    New York, Tokyo, Sydney, Dubai.
    """
    return get_region_devices(region)


@tool("get_device_metrics")
def tool_get_device_metrics(device_id: str) -> dict:
    """
    Get current performance metrics for a specific device.
    Returns CPU, memory, bandwidth, packet loss, latency, interface errors.
    Use this to confirm whether a device is under stress.
    """
    return get_device_metrics(device_id)


@tool("get_metrics_history")
def tool_get_metrics_history(device_id: str, hours: int = 6) -> dict:
    """
    Get hourly metric snapshots for a device over the last N hours.
    Use this to spot trends — rising CPU, worsening packet loss, etc.
    """
    return get_metrics_history(device_id=device_id, hours=hours)


@tool("get_runbook")
def tool_get_runbook(symptom: str) -> dict:
    """
    Retrieve the most relevant runbook sections for a given symptom.
    Pass a natural language description e.g. 'packet loss with CRC errors'.
    Always call this — it grounds your diagnosis in real SOPs.
    """
    return get_runbook(symptom)


@tool("list_runbooks")
def tool_list_runbooks() -> dict:
    """
    List all available runbooks in the knowledge base.
    Use this if unsure which runbook covers the current symptom.
    """
    return list_available_runbooks()


ALL_TOOLS = [
    tool_get_incident,
    tool_find_similar_incidents,
    tool_get_recent_incidents,
    tool_get_alerts_for_incident,
    tool_get_critical_alerts,
    tool_search_logs,
    tool_get_error_summary,
    tool_get_blast_radius,
    tool_get_neighbors,
    tool_get_region_devices,
    tool_get_device_metrics,
    tool_get_metrics_history,
    tool_get_runbook,
    tool_list_runbooks,
]