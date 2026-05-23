"""
test_run.py

End-to-end test of the NOC Copilot before building the rest.
Run from project root:
    python test_run.py

Tests:
  1. Individual MCP tools
  2. Alert correlation engine
  3. Full agent investigation (needs GOOGLE_API_KEY in .env)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("AI NOC Copilot — End to End Test")
print("=" * 60)


# ─────────────────────────────────────────────
# TEST 1 — MCP TOOLS
# ─────────────────────────────────────────────

print("\n[1/3] Testing MCP tools...")

from agent.tools.incidents import get_recent_incidents, find_similar_incidents
from agent.tools.alerts import get_critical_alerts
from agent.tools.logs import search_logs, get_error_summary
from agent.tools.topology import get_blast_radius, get_region_devices
from agent.tools.metrics import get_device_metrics
from agent.tools.runbooks import get_runbook

# Recent incidents
incidents = get_recent_incidents(limit=3)
assert len(incidents) > 0, "No incidents returned"
print(f"  get_recent_incidents     : {len(incidents)} records ✓")

# Similar incidents
similar = find_similar_incidents(symptom="packet_loss", limit=3)
print(f"  find_similar_incidents   : {len(similar)} records ✓")

# Critical alerts
alerts = get_critical_alerts(limit=5)
print(f"  get_critical_alerts      : {len(alerts)} records ✓")

# Logs
first_inc = incidents[0]["incident_id"]
logs = search_logs(incident_id=first_inc)
print(f"  search_logs              : {logs['total_lines']} lines ✓")

# Error summary
err = get_error_summary(first_inc)
print(f"  get_error_summary        : dominant={err.get('dominant_error')} ✓")

# Topology
first_device = incidents[0]["affected_device"]
blast = get_blast_radius(first_device, hops=2)
print(f"  get_blast_radius         : {blast.get('total_affected_devices', 0)} devices in blast radius ✓")

# Metrics
metrics = get_device_metrics(first_device)
print(f"  get_device_metrics       : CPU={metrics.get('cpu_utilization_pct')}% ✓")

# Runbook RAG
rb = get_runbook("packet loss with CRC errors on edge router")
print(f"  get_runbook              : {rb['chunks_retrieved']} chunks retrieved ✓")

print("\n  All MCP tools OK ✓")


# ─────────────────────────────────────────────
# TEST 2 — ALERT CORRELATION
# ─────────────────────────────────────────────

print("\n[2/3] Testing alert correlation engine...")

from alert_correlation.clusterer import correlate_alerts

cluster = correlate_alerts(description="Singapore users reporting packet loss and high latency")
assert cluster is not None, "Correlation returned None"

print(f"  Cluster ID        : {cluster['cluster_id']}")
print(f"  Dominant symptom  : {cluster['dominant_symptom']}")
print(f"  Total alerts      : {cluster['total_alert_count']}")
print(f"  Noise reduced     : {cluster['noise_reduced']}")
print(f"  Affected regions  : {cluster['affected_regions']}")
print(f"  Summary           : {cluster['summary']}")
print("\n  Alert correlation OK ✓")


# ─────────────────────────────────────────────
# TEST 3 — FULL AGENT INVESTIGATION
# ─────────────────────────────────────────────

print("\n[3/3] Testing full agent investigation...")

# Check if any of your three rotating keys are present in the environment
rotating_keys = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3")
]

# The agent runs if at least one rotating slot (or your default fallback key) is configured
if not any(rotating_keys) and not os.getenv("GOOGLE_API_KEY"):
    print("  SKIPPED — No Google API Keys found in your environment!")
    print("  Please configure GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, and GOOGLE_API_KEY_3 in your .env file.")
else:
    # Count how many of your rotating slots actually loaded
    active_slots = sum(1 for k in rotating_keys if k)
    print(f"  {active_slots}/3 API key slots discovered — running investigation...")
    print("  (This will take 20-40 seconds)\n")
    
    from agent.graph import run_investigation

    result = run_investigation(
        incident_description=(
            "Singapore enterprise customers are reporting intermittent "
            "connectivity issues. NOC monitoring shows packet loss and "
            "BGP session instability on edge routers in the Singapore region."
        ),
        alert_cluster=cluster,
    )

    print("\n" + "=" * 60)
    print("INVESTIGATION RESULT")
    print("=" * 60)
    
    # 1. Use dictionary lookups (.get() is safest to avoid missing key errors)
    is_complete = result.get("investigation_complete", False)
    tool_calls = result.get("tool_calls", [])
    evidence = result.get("evidence", [])
    loop_count = result.get("loop_count", 0)
    rca = result.get("rca")

    print(f"\nStatus       : {'Complete ✓' if is_complete else 'Incomplete ✗'}")
    print(f"Tool calls   : {len(tool_calls)}")
    print(f"Evidence     : {len(evidence)} items")
    print(f"Loops        : {loop_count}")
    
    # 2. Access RCA fields (if rca is a dictionary, use dict syntax; if it's an object, keep dot syntax)
    if rca:
        # Assuming rca inside the state dict might also be a dict or object. 
        # Let's write it safely assuming it's an object/Pydantic model since it wasn't the root crash point.
        # If rca itself causes a dict error next, change these to dict lookups too!
        try:
            print(f"Confidence   : {rca.confidence_score:.0%}")
            print(f"\nProbable cause:\n  {rca.probable_cause}")
            print(f"\nImpact:")
            print(f"  Region   : {rca.impact.affected_region}")
            print(f"  SLA risk : {rca.impact.sla_breach_risk}")
            print(f"  Devices  : {rca.impact.affected_devices[:3]}")
            print(f"\nRecommended actions:")
            for i, action in enumerate(rca.recommended_actions, 1):
                print(f"  {i}. {action}")
            print(f"\nEscalation   : {rca.escalation_team}")
            print(f"\nSummary:\n  {rca.incident_summary}")
            print(f"\nTimeline ({len(rca.timeline)} events):")
            for event in rca.timeline[:5]:
                print(f"  [{event.event_type:20s}] {event.description[:60]}")
        except AttributeError:
            # Fallback if rca is ALSO a dictionary
            print(f"Confidence   : {rca.get('confidence_score', 0):.0%}")
            print(f"\nProbable cause:\n  {rca.get('probable_cause')}")
            # ... and so on
    else:
        print("\nNo Root Cause Analysis (RCA) data returned in state.")

    print(f"\nTool call trace:")
    for tc in tool_calls:
        # Handle if tool calls are objects or dicts
        if isinstance(tc, dict):
            print(f"  ✓ {tc.get('tool_name', 'unknown'):40s} {tc.get('duration_ms', 0):6.1f}ms")
        else:
            status = "✓" if tc.success else "✗"
            print(f"  {status} {tc.tool_name:40s} {tc.duration_ms:6.1f}ms")

    print("\n" + "=" * 60)
    print("Full agent test complete ✓")
    print("=" * 60)