"""
agent/state.py

The state object is the single source of truth during an investigation.
LangGraph passes this between every node in the graph.
Every tool result, evidence item, and decision gets written here.

Think of it as the "investigation notebook" — it starts empty
and fills up as the agent gathers evidence.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ─────────────────────────────────────────────
# EVIDENCE ITEM
# One piece of collected evidence.
# Every tool result becomes one or more EvidenceItems.
# ─────────────────────────────────────────────

class EvidenceItem(BaseModel):
    type: str = Field(
        description="log | alert | runbook | historical_incident | topology | metric"
    )
    source: str = Field(
        description="Where this came from e.g. INC-104332.log, packet_loss_runbook.txt"
    )
    content: str = Field(
        description="The actual evidence text or summary"
    )
    weight: float = Field(
        default=0.0,
        description="Contribution to confidence score (0.0 to 1.0)"
    )
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


# ─────────────────────────────────────────────
# TIMELINE EVENT
# One chronological event in the incident timeline.
# Reconstructed from log and alert timestamps.
# ─────────────────────────────────────────────

class TimelineEvent(BaseModel):
    timestamp: str
    event_type: str = Field(
        description="alert_fired | log_error | bgp_drop | interface_down | customer_impact | resolution"
    )
    device: Optional[str] = None
    description: str
    source: str = Field(description="alerts | logs | incident_record")


# ─────────────────────────────────────────────
# TOOL CALL RECORD
# Logged by the observability layer.
# Tracks every tool the agent called, what it passed, and how long it took.
# ─────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    tool_name: str
    input_params: dict
    output_summary: str = Field(description="Short summary of what was returned")
    duration_ms: float
    called_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None


# ─────────────────────────────────────────────
# IMPACT ANALYSIS
# What is affected — region, devices, customers, SLA risk.
# Populated from topology blast radius + incident metadata.
# ─────────────────────────────────────────────

class ImpactAnalysis(BaseModel):
    affected_region: Optional[str] = None
    affected_devices: list[str] = Field(default_factory=list)
    customer_segment: Optional[str] = None
    estimated_affected_customers: Optional[int] = None
    sla_breach_risk: Optional[str] = Field(
        default=None,
        description="LOW | MEDIUM | HIGH | CRITICAL"
    )
    blast_radius_confirmed: bool = False


# ─────────────────────────────────────────────
# RCA OUTPUT
# The final deliverable. Everything the agent produces
# gets written into this object.
# ─────────────────────────────────────────────

class RCAOutput(BaseModel):
    probable_cause: Optional[str] = None
    confidence_score: float = Field(
        default=0.0,
        description="0.0 to 1.0 computed from evidence weights"
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    impact: ImpactAnalysis = Field(default_factory=ImpactAnalysis)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    escalation_team: Optional[str] = None
    incident_summary: Optional[str] = Field(
        default=None,
        description="Human-readable paragraph summarising the outage"
    )
    correlated_alert_cluster: Optional[dict] = Field(
        default=None,
        description="Output from alert correlation engine"
    )


# ─────────────────────────────────────────────
# AGENT STATE
# The full investigation state passed between LangGraph nodes.
# Starts mostly empty, fills up as the agent works.
# ─────────────────────────────────────────────

class AgentState(BaseModel):

    # ── Input ──────────────────────────────────
    incident_description: str = Field(
        description="The raw input from the engineer or alert system"
    )
    incident_id: Optional[str] = Field(
        default=None,
        description="If a specific incident ID was provided or found"
    )
    input_alert: Optional[dict] = Field(
        default=None,
        description="Structured alert JSON if input was an alert"
    )

    # ── Investigation progress ──────────────────
    messages: list[dict] = Field(
        default_factory=list,
        description="Full LangGraph message history — LLM conversation turns"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Every tool the agent has called so far"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence collected so far"
    )
    loop_count: int = Field(
        default=0,
        description="How many reasoning loops have run — used to enforce MAX_TOOL_CALLS"
    )

    # ── Intermediate findings ───────────────────
    identified_symptom: Optional[str] = None
    identified_region: Optional[str] = None
    identified_device: Optional[str] = None
    similar_incidents_found: list[dict] = Field(default_factory=list)
    raw_logs: list[dict] = Field(default_factory=list)
    raw_alerts: list[dict] = Field(default_factory=list)
    topology_data: Optional[dict] = None
    device_metrics: Optional[dict] = None
    runbook_chunks: list[dict] = Field(default_factory=list)

    # ── Alert correlation ───────────────────────
    alert_cluster: Optional[dict] = Field(
        default=None,
        description="Output from alert correlation engine"
    )

    # ── Final output ────────────────────────────
    rca: RCAOutput = Field(default_factory=RCAOutput)
    investigation_complete: bool = False
    error: Optional[str] = None

    # ── Metadata ────────────────────────────────
    started_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    completed_at: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# ─────────────────────────────────────────────
# HELPER — pretty print state summary
# Useful for debugging during development
# ─────────────────────────────────────────────

def summarise_state(state: AgentState) -> str:
    lines = [
        f"Incident     : {state.incident_description[:80]}...",
        f"Loop count   : {state.loop_count}",
        f"Tool calls   : {len(state.tool_calls)}",
        f"Evidence     : {len(state.evidence)} items",
        f"Confidence   : {state.rca.confidence_score:.2f}",
        f"Symptom      : {state.identified_symptom}",
        f"Region       : {state.identified_region}",
        f"Device       : {state.identified_device}",
        f"Complete     : {state.investigation_complete}",
    ]
    return "\n".join(lines)