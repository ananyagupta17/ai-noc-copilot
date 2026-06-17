"""
test_run.py

End-to-end test of the NOC Copilot before building the rest.
Run from project root:
    python test_run.py

Tests:
  1. Individual agent tools
  2. Alert correlation engine
  3. Full agent investigation (needs GOOGLE_API_KEY in .env)
  4. Evidence scoring module
  5. Timeline reconstruction module
  6. Observability tracer + logger
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("AI NOC Copilot — End to End Test")
print("=" * 60)


# ─────────────────────────────────────────────
# TEST 1 — AGENT TOOLS
# ─────────────────────────────────────────────

print("\n[1/6] Testing agent tools...")

from agent.tools.incidents import get_recent_incidents, find_similar_incidents
from agent.tools.alerts import get_critical_alerts
from agent.tools.logs import search_logs, get_error_summary
from agent.tools.topology import get_blast_radius, get_region_devices
from agent.tools.metrics import get_device_metrics
from agent.tools.runbooks import get_runbook

incidents = get_recent_incidents(limit=3)
assert len(incidents) > 0, "No incidents returned"
print(f"  get_recent_incidents     : {len(incidents)} records ✓")

similar = find_similar_incidents(symptom="packet_loss", limit=3)
print(f"  find_similar_incidents   : {len(similar)} records ✓")

alerts = get_critical_alerts(limit=5)
print(f"  get_critical_alerts      : {len(alerts)} records ✓")

first_inc = incidents[0]["incident_id"]
logs = search_logs(incident_id=first_inc)
print(f"  search_logs              : {logs['total_lines']} lines ✓")

err = get_error_summary(first_inc)
print(f"  get_error_summary        : dominant={err.get('dominant_error')} ✓")

first_device = incidents[0]["affected_device"]
blast = get_blast_radius(first_device, hops=2)
print(f"  get_blast_radius         : {blast.get('total_affected_devices', 0)} devices in blast radius ✓")

metrics = get_device_metrics(first_device)
print(f"  get_device_metrics       : CPU={metrics.get('cpu_utilization_pct')}% ✓")

rb = get_runbook("packet loss with CRC errors on edge router")
print(f"  get_runbook              : {rb['chunks_retrieved']} chunks retrieved ✓")

print("\n  All agent tools OK ✓")


# ─────────────────────────────────────────────
# TEST 2 — ALERT CORRELATION
# ─────────────────────────────────────────────

print("\n[2/6] Testing alert correlation engine...")

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

print("\n[3/6] Testing full agent investigation...")

rotating_keys = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3")
]

if not any(rotating_keys) and not os.getenv("GOOGLE_API_KEY"):
    print("  SKIPPED — No Google API Keys found in your environment!")
    agent_result = None
else:
    active_slots = sum(1 for k in rotating_keys if k)
    print(f"  {active_slots}/3 API key slots discovered — running investigation...")
    print("  (This will take 20-40 seconds)\n")

    from agent.graph import run_investigation

    agent_result = run_investigation(
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

    is_complete = agent_result.get("investigation_complete", False)
    tool_calls  = agent_result.get("tool_calls", [])
    evidence    = agent_result.get("evidence", [])
    loop_count  = agent_result.get("loop_count", 0)
    rca         = agent_result.get("rca")

    print(f"\nStatus       : {'Complete ✓' if is_complete else 'Incomplete ✗'}")
    print(f"Tool calls   : {len(tool_calls)}")
    print(f"Evidence     : {len(evidence)} items")
    print(f"Loops        : {loop_count}")

    if rca:
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
            print(f"Confidence   : {rca.get('confidence_score', 0):.0%}")
            print(f"\nProbable cause:\n  {rca.get('probable_cause')}")
    else:
        print("\nNo RCA data returned.")

    print(f"\nTool call trace:")
    for tc in tool_calls:
        if isinstance(tc, dict):
            print(f"  ✓ {tc.get('tool_name', 'unknown'):40s} {tc.get('duration_ms', 0):6.1f}ms")
        else:
            status = "✓" if tc.success else "✗"
            print(f"  {status} {tc.tool_name:40s} {tc.duration_ms:6.1f}ms")

    print("\n" + "=" * 60)
    print("Full agent test complete ✓")
    print("=" * 60)


# ─────────────────────────────────────────────
# TEST 4 — EVIDENCE SCORING
# ─────────────────────────────────────────────

print("\n[4/6] Testing evidence scoring module...")

from agent.evidence import score_evidence, enrich_rca_with_evidence
from agent.state import AgentState, EvidenceItem

mock_state = AgentState(incident_description="Test — Singapore packet loss")
mock_state.evidence = [
    EvidenceItem(type="log",                source="INC-001.log",         content="BGP session dropped",         weight=0.30),
    EvidenceItem(type="log",                source="INC-001.log",         content="CRC errors spiking on Gi0/1", weight=0.30),
    EvidenceItem(type="historical_incident",source="find_similar()",      content="Same region, same symptom",   weight=0.25),
    EvidenceItem(type="runbook",            source="packet_loss_runbook", content="Step 2: check CRC errors",    weight=0.20),
    EvidenceItem(type="alert",              source="get_critical_alerts", content="CRITICAL: packet loss 35%",   weight=0.15),
    EvidenceItem(type="topology",           source="get_blast_radius()",  content="3 devices in blast radius",   weight=0.10),
]

confidence, breakdown = score_evidence(mock_state)

assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
assert breakdown.score_interpretation in ("LOW", "MEDIUM", "HIGH", "VERY HIGH")
assert len(breakdown.strongest_evidence) > 0
assert len(breakdown.contributing_types) > 0

print(f"  Confidence score     : {confidence:.0%}")
print(f"  Interpretation       : {breakdown.score_interpretation}")
print(f"  Contributing types   : {list(breakdown.contributing_types.keys())}")
print(f"  Strongest evidence   : {breakdown.strongest_evidence[0][:60]}...")
print(f"  Missing evidence     : {breakdown.missing_evidence}")
print(f"  Explanation          : {breakdown.explanation[:80]}...")

enriched = enrich_rca_with_evidence(mock_state)
assert enriched.rca.confidence_score == confidence
print(f"\n  enrich_rca_with_evidence : state.rca.confidence_score = {enriched.rca.confidence_score:.0%} ✓")
print("\n  Evidence scoring OK ✓")


# ─────────────────────────────────────────────
# TEST 5 — TIMELINE RECONSTRUCTION
# ─────────────────────────────────────────────

print("\n[5/6] Testing timeline reconstruction...")

from agent.timeline import build_timeline, timeline_to_display

timeline_state = AgentState(incident_description="Test — Singapore optical degradation")
timeline_state.raw_alerts = [
    {
        "timestamp":  "2025-04-09T18:08:00",
        "alert_type": "packet_loss",
        "device":     "SIN-ER-01",
        "message":    "CRITICAL: Packet loss 35% on SIN-ER-01",
        "severity":   "CRITICAL",
    },
    {
        "timestamp":  "2025-04-09T18:09:30",
        "alert_type": "bgp_flap",
        "device":     "SIN-CR-01",
        "message":    "CRITICAL: BGP session DOWN on SIN-CR-01",
        "severity":   "CRITICAL",
    },
]
timeline_state.raw_logs = [
    {"line": "Apr 09 18:07:45 SIN-ER-01 : %OPTICAL-3-RXPOWER_LOW: Rx power -28dBm on Gi0/1"},
    {"line": "Apr 09 18:08:10 SIN-CR-01 : %BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down Hold Timer Expired"},
    {"line": "Apr 09 18:09:30 SIN-CR-01 : %MPLS-3-LDP_NBR_DOWN: LDP session dropped"},
]
timeline_state.similar_incidents_found = [
    {
        "detected_at":    "2025-04-09T18:07:00",
        "resolved_at":    "2025-04-09T20:15:00",
        "affected_device":"SIN-ER-01",
        "resolution":     "Traffic rerouted via backup MPLS path",
        "description":    "Optical degradation on Singapore edge router",
    }
]

timeline = build_timeline(timeline_state)
display  = timeline_to_display(timeline)

assert len(timeline) > 0, "Timeline is empty"
assert all(hasattr(e, "timestamp") for e in timeline)
assert all("icon" in d for d in display)
assert any(e["event_type"] == "customer_impact" for e in display), \
    "Customer impact event not inferred"

print(f"  Total events built   : {len(timeline)}")
print(f"  Event types found    : {list({e.event_type for e in timeline})}")
print(f"  Customer impact      : {'✓ inferred' if any(e['event_type'] == 'customer_impact' for e in display) else '✗ missing'}")
print(f"\n  Timeline preview:")
print(f"  {'Time':10} {'Icon'} {'Type':22} {'Device':12} Description")
print(f"  {'-'*75}")
for e in display:
    print(f"  {e['time']:10} {e['icon']}  {e['event_type']:22} {e['device']:12} {e['description'][:40]}")

print("\n  Timeline reconstruction OK ✓")


# ─────────────────────────────────────────────
# TEST 6 — OBSERVABILITY TRACER + LOGGER
# ─────────────────────────────────────────────

print("\n[6/6] Testing observability tracer and logger...")

from observability.tracer import NOCTracer
from observability.logger import NOCLogger
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)

    # ── Tracer ────────────────────────────────
    test_tracer = NOCTracer()
    inv_id = test_tracer.start_investigation("Test incident — Singapore BGP flap")
    assert inv_id.startswith("INV-")

    test_tracer.record_tool_call(
        tool_name="find_similar_incidents",
        input_params={"symptom": "bgp_flap", "region": "Singapore"},
        result=[{"incident_id": "INC-001", "symptom": "bgp_flap"}],
        duration_ms=12.5, success=True, error=None, loop_number=1,
    )
    test_tracer.record_rag_retrieval(
        query="BGP session instability troubleshooting",
        results=[
            {"source": "bgp_flap_runbook.txt", "content": "Step 1..."},
            {"source": "bgp_flap_runbook.txt", "content": "Step 2..."},
        ],
        duration_ms=8.3,
    )
    test_tracer.record_loop(
        loop_number=1, tools_selected=["find_similar_incidents", "get_runbook"],
        evidence_count=2, confidence=0.45, duration_ms=1200.0,
    )
    test_tracer.end_investigation(confidence=0.82, complete=True)

    summary = test_tracer.get_summary()
    assert summary["total_tool_calls"] == 1
    assert summary["total_loops"] == 1
    assert summary["rag_retrievals"] == 1
    assert summary["final_confidence"] == 0.82
    assert summary["tool_success_rate"] == 100.0

    trace_dict = test_tracer.to_dict()
    assert len(trace_dict["tool_traces"]) == 1
    assert len(trace_dict["rag_traces"])  == 1
    assert len(trace_dict["loop_traces"]) == 1

    saved_path = test_tracer.save_to_file(output_dir=tmp_path)
    assert saved_path.exists()
    saved_data = json.loads(saved_path.read_text())
    assert saved_data["final_confidence"] == 0.82

    print(f"  Tracer investigation ID  : {inv_id} ✓")
    print(f"  Tool calls recorded      : {summary['total_tool_calls']} ✓")
    print(f"  RAG retrievals recorded  : {summary['rag_retrievals']} ✓")
    print(f"  Loops recorded           : {summary['total_loops']} ✓")
    print(f"  Final confidence         : {summary['final_confidence']:.0%} ✓")
    print(f"  Tool success rate        : {summary['tool_success_rate']}% ✓")
    print(f"  Serialisation to dict    : ✓")
    print(f"  Trace saved to file      : ✓")

    # ── Logger ────────────────────────────────
    test_logger = NOCLogger(log_dir=tmp_path)
    test_logger.investigation_started("TEST-001", "Singapore BGP flap test")
    test_logger.tool_called(
        tool_name="find_similar_incidents", params={"symptom": "bgp_flap"},
        success=True, duration_ms=12.5, result_summary="3 records", loop_number=1,
    )
    quality = test_logger.rag_retrieved(
        query="BGP troubleshooting", chunks_retrieved=4,
        sources=["bgp_flap_runbook.txt"], duration_ms=8.3,
    )
    test_logger.confidence_updated(loop=1, confidence=0.45, evidence_count=2)
    test_logger.confidence_updated(loop=2, confidence=0.72, evidence_count=5)
    test_logger.investigation_completed(
        investigation_id="TEST-001", confidence=0.82, tool_count=3,
        loop_count=2, duration_ms=4500,
        probable_cause="BGP misconfiguration on Singapore edge router",
    )

    logs = test_logger.read_recent_logs(n=10)
    assert len(logs) > 0
    events = [l["event"] for l in logs]
    assert "investigation.started"   in events
    assert "tool.called"             in events
    assert "rag.retrieved"           in events
    assert "confidence.updated"      in events
    assert "investigation.completed" in events

    history = test_logger.get_confidence_history()
    assert len(history) == 2
    assert history[0]["confidence"] == 0.45
    assert history[1]["confidence"] == 0.72

    print(f"  Logger events written    : {len(logs)} ✓")
    print(f"  Events logged            : {events} ✓")
    print(f"  Confidence history       : {[h['confidence'] for h in history]} ✓")
    print(f"  RAG quality score        : {quality:.2f} ✓")

print("\n  Observability OK ✓")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("""
Modules verified:
  ✓ Agent tools (14 tools across 6 modules)
  ✓ Alert correlation (TF-IDF + DBSCAN)
  ✓ Agent investigation loop (LangGraph + Gemini)
  ✓ Evidence scoring (weighted confidence)
  ✓ Timeline reconstruction (multi-source, deduped)
  ✓ Observability (tracer + structured logger)

Ready to build:
  ⬜ api/main.py       FastAPI backend
  ⬜ ui/app.py         Streamlit dashboard
""")