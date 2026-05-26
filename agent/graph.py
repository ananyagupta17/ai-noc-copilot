from __future__ import annotations
"""
agent/graph.py

The LangGraph investigation loop.
This is the brain of the NOC Copilot.

Graph structure:
  START
    ↓
  [entry] — parse input, identify symptom/region/device
    ↓
  [reason] — GPT-4o decides what to do next
    ↓
  [tools]  — execute whatever tool GPT-4o selected
    ↓  ↑
  (loop back to reason until enough evidence OR max loops hit)
    ↓
  [output] — score evidence, build timeline, generate RCA
    ↓
  END
"""
import itertools
import os
import json
import time
from datetime import datetime
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from agent.evidence import enrich_rca_with_evidence
from agent.timeline import build_timeline, timeline_to_display
from observability.tracer import tracer
from observability.logger import noc_logger

# Note: We keep GOOGLE_API_KEY as a fallback default if your environment relies on it elsewhere
from config import LLM_MODEL, MAX_TOOL_CALLS, GOOGLE_API_KEY
from agent.state import (
    AgentState, EvidenceItem, TimelineEvent,
    ToolCallRecord, ImpactAnalysis, summarise_state
)
from agent.tools import ALL_TOOLS


# ─────────────────────────────────────────────
# API KEY ROTATION SETUP
#
# WHY WE ROTATE KEYS HERE:
#   This project runs on Gemini free-tier API keys, which enforce a
#   per-minute request quota (RPM limit). A single investigation can
#   trigger 6-10 LLM calls across reasoning loops, which exhausts the
#   free-tier RPM limit mid-investigation and causes 429 errors.
#
#   To work around this during development, we spread calls across
#   3 separate free-tier API keys using round-robin rotation, effectively
#   tripling the available RPM budget.
#
# IN PRODUCTION THIS WOULD NOT BE NEEDED:
#   A paid Gemini API key has a much higher (or configurable) RPM limit,
#   so a single key handles the full investigation load comfortably.
#   The correct production pattern would be:
#     - One API key
#     - Exponential backoff + retry on 429 responses
#   Key rotation is purely a free-tier development workaround.
# ─────────────────────────────────────────────

API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3")
]

# Quick validation check to notify you if any key failed to load from .env
if not all(API_KEYS):
    missing_indices = [i + 1 for i, key in enumerate(API_KEYS) if not key]
    print(f"[WARNING] Missing environment keys for slot(s): {missing_indices}")

# Fall back to the original key config array if everything in .env came back empty
if not any(API_KEYS):
    API_KEYS = [GOOGLE_API_KEY]

KEY_ROTATOR = itertools.cycle(API_KEYS)


# ─────────────────────────────────────────────
# LLM SETUP
# bind_tools() tells the LLM about all 14 tools
# ─────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0,
    # We pass a temporary placeholder key initialization here because 
    # we will dynamically overwrite it inside reason_node during invocation.
    google_api_key=API_KEYS[0] or GOOGLE_API_KEY,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# Tool lookup map — name → callable
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# This is what GPT-4o reads at the start of every investigation.
# It sets the persona, goals, and investigation strategy.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI NOC Copilot for Tata Communications.
You are an expert network operations engineer investigating incidents.

Your job is to investigate network incidents systematically:
1. Identify the symptom type and affected region/device from the description
2. Search for similar historical incidents first — they reveal likely root causes
3. Pull logs for the affected incident to get raw evidence
4. Get the error summary to identify dominant error codes
5. Check topology to understand blast radius and affected devices
6. Retrieve the relevant runbook to ground your diagnosis in SOPs
7. Check device metrics to confirm device health
8. Once you have sufficient evidence, stop calling tools

Investigation strategy:
- Always call tool_find_similar_incidents early
- Always call tool_get_runbook — it grounds your answer in real procedures
- Call tool_get_blast_radius if you have a device ID — it reveals customer impact
- Do NOT call the same tool twice with the same parameters
- Stop investigating when you have: logs + similar incidents + runbook + topology

You are methodical, precise, and evidence-driven.
Never guess — every claim must be backed by tool results.
"""


# ─────────────────────────────────────────────
# NODE 1 — ENTRY
# Parses the input and extracts key fields into state.
# Runs once at the start of every investigation.
# ─────────────────────────────────────────────

def entry_node(state: AgentState) -> AgentState:
    """
    Initialise the investigation.
    Parses the incident description to extract symptom, region, device.
    Builds the first message for the LLM conversation.
    """
    print(f"\n[NOC Agent] Starting investigation...")
    print(f"[NOC Agent] Input: {state.incident_description[:100]}")

    tracer.start_investigation(state.incident_description)
    noc_logger.investigation_started("active", state.incident_description)

    # Build the opening message — this is what GPT-4o first reads
    opening = f"""
New incident reported:

{state.incident_description}

Begin your investigation. Use the available tools systematically
to gather evidence and determine the root cause.
"""

    state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": opening.strip()},
    ]

    # If alert cluster exists from correlation engine, inject it
    if state.alert_cluster:
        cluster_summary = json.dumps(state.alert_cluster, indent=2)
        state.messages.append({
            "role": "user",
            "content": f"Alert correlation results:\n{cluster_summary}"
        })

    state.loop_count = 0
    return state


# ─────────────────────────────────────────────
# NODE 2 — REASON
# GPT-4o reads current state and decides what to do next.
# Either calls a tool or signals it's done investigating.
# This node is the "thinking" step in the loop.
# ─────────────────────────────────────────────

def reason_node(state: AgentState) -> AgentState:
    """
    GPT-4o decides what tool to call next, or signals completion.
    Reads the full message history including all previous tool results.
    """
    state.loop_count += 1
    print(f"[NOC Agent] Reasoning loop {state.loop_count}...")

    lc_messages = []
    for msg in state.messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(msg.get("_lc_obj") or HumanMessage(content=str(msg["content"])))
        elif msg["role"] == "tool":
            lc_messages.append(ToolMessage(
                content=str(msg["content"]),
                tool_call_id=msg.get("tool_call_id", "unknown")
            ))

    current_key = next(KEY_ROTATOR)
    key_preview = f"...{current_key[-6:]}" if current_key else "FALLBACK"
    print(f"[NOC Agent] Dynamic authentication token shift. Active key slot: {key_preview}")

    response = llm_with_tools.invoke(
        lc_messages,
        config={"configurable": {"google_api_key": current_key}}
    )

    state.messages.append({
        "role": "assistant",
        "content": response.content,
        "tool_calls": response.tool_calls if response.tool_calls else [],
        "_lc_obj": response
    })

    # Record reasoning loop for observability
    tracer.record_loop(
        loop_number=state.loop_count,
        tools_selected=[
            tc["name"] for tc in
            state.messages[-1].get("tool_calls", [])
        ],
        evidence_count=len(state.evidence),
        confidence=state.rca.confidence_score,
        duration_ms=0,
    )

    return state
# ─────────────────────────────────────────────
# ROUTER — after reason_node
# Decides whether to call a tool or go to output.
# LangGraph uses this to determine the next node.
# ─────────────────────────────────────────────

def should_continue(state: AgentState):
    """
    If the last LLM response contains tool calls → go to tools node.
    If no tool calls OR max loops hit → go to output node.
    """
    last_msg = state.messages[-1]
    tool_calls = last_msg.get("tool_calls", [])

    if state.loop_count >= MAX_TOOL_CALLS:
        print(f"[NOC Agent] Max loops ({MAX_TOOL_CALLS}) reached — moving to output")
        return "output"

    if tool_calls:
        print(f"[NOC Agent] Tool selected: {[tc['name'] for tc in tool_calls]}")
        return "tools"

    print("[NOC Agent] No tool calls — investigation complete")
    return "output"


# ─────────────────────────────────────────────
# NODE 3 — TOOLS
# Executes whichever tool GPT-4o selected.
# Records timing, result, and evidence into state.
# ─────────────────────────────────────────────

def tools_node(state: AgentState) -> AgentState:
    """
    Execute the tool(s) GPT-4o selected.
    Adds results to message history and evidence list.
    """
    last_msg = state.messages[-1]
    tool_calls = last_msg.get("tool_calls", [])

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc.get("id", f"call_{len(state.tool_calls)}")

        print(f"[NOC Agent] Calling {tool_name}({tool_args})")

        start = time.perf_counter()
        try:
            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                result = {"error": f"Tool {tool_name} not found"}
                success = False
            else:
                result = tool_fn.invoke(tool_args)
                success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        # Serialise result for message history
        result_str = json.dumps(result, indent=2, default=str)

        # Add to message history so LLM sees the result next loop
        state.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_str,
        })

        tracer.record_tool_call(
            tool_name=tool_name, input_params=tool_args, result=result,
            duration_ms=duration_ms, success=success, error=None, loop_number=state.loop_count,
        )
        noc_logger.tool_called(
            tool_name=tool_name, params=tool_args, success=success,
            duration_ms=duration_ms, result_summary=_summarise_result(tool_name, result),
            loop_number=state.loop_count,
        )

        # Record for observability
        state.tool_calls.append(ToolCallRecord(
            tool_name=tool_name,
            input_params=tool_args,
            output_summary=_summarise_result(tool_name, result),
            duration_ms=duration_ms,
            success=success,
            error=result.get("error") if not success else None,
        ))

        # Extract evidence from tool result
        _extract_evidence(state, tool_name, tool_args, result)

        # Extract key fields if found
        _extract_state_fields(state, tool_name, result)

        # RAG tracer — record runbook retrievals for observability panel
        if tool_name == "get_runbook" and success and isinstance(result, dict):
            tracer.record_rag_retrieval(
                query=tool_args.get("symptom", ""),
                results=result.get("results", []),
                duration_ms=duration_ms,
            )

        # Timeline fix — populate raw_alerts from get_critical_alerts
        if tool_name == "get_critical_alerts" and success and isinstance(result, list):
            state.raw_alerts.extend(result)

    return state


def _summarise_result(tool_name: str, result) -> str:
    """Short summary of a tool result for the observability trace."""
    if isinstance(result, dict) and "error" in result:
        return f"ERROR: {result['error']}"
    if isinstance(result, list):
        return f"{len(result)} records returned"
    if isinstance(result, dict):
        keys = list(result.keys())[:4]
        return f"Dict with keys: {keys}"
    return str(result)[:100]


def _extract_evidence(state: AgentState, tool_name: str, args: dict, result):
    """
    Convert a tool result into an EvidenceItem with appropriate weight.
    Evidence weights:
      logs            → 0.30  (hard signals — error codes don't lie)
      historical match→ 0.25  (precedent — same symptom resolved same way)
      runbook         → 0.20  (SOP grounding)
      alerts          → 0.15  (monitoring signals)
      topology        → 0.10  (structural confirmation)
      metrics         → 0.10  (device health confirmation)
    """
    weight_map = {
        "search_logs":             0.30,
        "get_error_summary":       0.30,
        "find_similar_incidents":  0.25,
        "get_incident":            0.20,
        "get_runbook":             0.20,
        "list_runbooks":           0.05,
        "get_alerts_for_incident": 0.15,
        "get_critical_alerts":     0.15,
        "get_blast_radius":        0.10,
        "get_neighbors":           0.10,
        "get_region_devices":      0.05,
        "get_device_metrics":      0.10,
        "get_metrics_history":     0.10,
        "get_recent_incidents":    0.10,
    }

    type_map = {
        "search_logs":             "log",
        "get_error_summary":       "log",
        "find_similar_incidents":  "historical_incident",
        "get_incident":            "historical_incident",
        "get_runbook":             "runbook",
        "get_alerts_for_incident": "alert",
        "get_critical_alerts":     "alert",
        "get_blast_radius":        "topology",
        "get_neighbors":           "topology",
        "get_region_devices":      "topology",
        "get_device_metrics":      "metric",
        "get_metrics_history":     "metric",
        "get_recent_incidents":    "historical_incident",
    }

    # Skip if result is empty or error
    if not result:
        return
    if isinstance(result, dict) and "error" in result:
        return
    if isinstance(result, list) and len(result) == 0:
        return

    content = json.dumps(result, default=str)[:500]  # truncate long results

    state.evidence.append(EvidenceItem(
        type=type_map.get(tool_name, "unknown"),
        source=f"{tool_name}({json.dumps(args, default=str)[:60]})",
        content=content,
        weight=weight_map.get(tool_name, 0.05),
    ))


def _extract_state_fields(state: AgentState, tool_name: str, result):
    """
    Pull key fields out of tool results and store in state directly.
    These are used by the output node for impact analysis and timeline.
    """
    if tool_name == "tool_find_similar_incidents" and isinstance(result, list):
        state.similar_incidents_found = result

    elif tool_name == "tool_search_logs" and isinstance(result, dict):
        state.raw_logs = result.get("lines", [])
        # Try to extract incident ID from log source filenames
        for line in state.raw_logs:
            src = line.get("source", "")
            if src.startswith("INC-") and not state.incident_id:
                state.incident_id = src.replace(".log", "")

    elif tool_name == "tool_get_alerts_for_incident" and isinstance(result, list):
        state.raw_alerts = result

    elif tool_name == "tool_get_critical_alerts" and isinstance(result, list):
        state.raw_alerts = result

    elif tool_name == "get_critical_alerts" and isinstance(result, list):
         state.raw_alerts.extend(result)

    elif tool_name == "tool_get_blast_radius" and isinstance(result, dict):
        state.topology_data = result
        if not state.identified_region and result.get("regions_affected"):
            state.identified_region = result["regions_affected"][0]

    elif tool_name == "tool_get_device_metrics" and isinstance(result, dict):
        state.device_metrics = result
        if not state.identified_device:
            state.identified_device = result.get("device_id")

    elif tool_name == "tool_get_runbook" and isinstance(result, dict):
        state.runbook_chunks = result.get("results", [])

    elif tool_name == "tool_find_similar_incidents" and isinstance(result, list):
        state.similar_incidents_found = result
        if result and not state.identified_region:
            state.identified_region = result[0].get("region")
        if result and not state.identified_symptom:
            state.identified_symptom = result[0].get("symptom")


# ─────────────────────────────────────────────
# NODE 4 — OUTPUT
# Computes confidence score, builds timeline,
# then asks GPT-4o to synthesise the final RCA.
# Runs once at the end of the investigation.
# ─────────────────────────────────────────────

def output_node(state: AgentState) -> AgentState:
    """
    Generate the final RCA output.
    1. Compute confidence score from evidence weights
    2. Build timeline from log/alert timestamps
    3. Ask GPT-4o to synthesise final RCA using all collected evidence
    """
    print("[NOC Agent] Generating RCA output...")

      # ── Step 1: Compute confidence score (via evidence.py) ────
    state = enrich_rca_with_evidence(state)
    confidence = state.rca.confidence_score

    # ── Step 2: Build timeline ─────────────────
    timeline = build_timeline(state)
    state.rca.timeline = timeline

    # ── Step 3: Build impact analysis ─────────
    state.rca.impact = _build_impact(state)

    # ── Step 4: Ask LLM to synthesise RCA ─────
    synthesis_prompt = f"""
Based on your investigation, provide a final RCA in this exact JSON format:

{{
  "probable_cause": "one clear sentence describing the root cause",
  "recommended_actions": [
    "action 1",
    "action 2",
    "action 3"
  ],
  "escalation_team": "which team to escalate to",
  "incident_summary": "2-3 sentence human-readable summary of what happened, impact, and resolution path"
}}

Evidence collected: {len(state.evidence)} items
Confidence score: {confidence}
Similar incidents found: {len(state.similar_incidents_found)}
Log lines retrieved: {len(state.raw_logs)}

Return ONLY the JSON object, no other text.
"""

    state.messages.append({"role": "user", "content": synthesis_prompt})

    lc_messages = []
    for msg in state.messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] in ("user",):
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "tool":
            lc_messages.append(ToolMessage(
                content=str(msg["content"]),
                tool_call_id=msg.get("tool_call_id", "unknown")
            ))
        elif msg["role"] == "assistant" and msg.get("content"):
            lc_messages.append(HumanMessage(content=f"[Assistant]: {msg['content']}"))

    # Use plain LLM here (no tools) — we just want text output
    response = llm.invoke(lc_messages)

    # Parse LLM JSON response
    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        rca_data = json.loads(raw.strip())

        state.rca.probable_cause   = rca_data.get("probable_cause")
        state.rca.recommended_actions = rca_data.get("recommended_actions", [])
        state.rca.escalation_team  = rca_data.get("escalation_team")
        state.rca.incident_summary = rca_data.get("incident_summary")

    except json.JSONDecodeError:
        # If LLM didn't return clean JSON, store raw text
        state.rca.incident_summary = response.content
        state.rca.probable_cause   = "See incident summary"
        state.rca.recommended_actions = ["Manual investigation required"]
        state.rca.escalation_team  = "L2 Network Engineering"

    # ── Step 5: Attach alert correlation ──────
    if state.alert_cluster:
        state.rca.correlated_alert_cluster = state.alert_cluster

    tracer.end_investigation(confidence=confidence, complete=True)
    tracer.save_to_file()
    noc_logger.investigation_completed(
        investigation_id="active", confidence=confidence,
        tool_count=len(state.tool_calls), loop_count=state.loop_count,
        duration_ms=0, probable_cause=state.rca.probable_cause,
    )
    
    state.investigation_complete = True
    state.completed_at = datetime.now().isoformat()

    print(f"[NOC Agent] Investigation complete.")
    print(f"[NOC Agent] Confidence: {confidence}")
    print(f"[NOC Agent] Probable cause: {state.rca.probable_cause}")

    return state


def _build_timeline(state: AgentState) -> list[TimelineEvent]:
    """
    Build a chronological timeline from raw logs and alerts.
    Parses timestamps and sorts all events.
    """
    events = []

    # Events from raw alerts
    for alert in state.raw_alerts[:10]:
        ts = alert.get("timestamp", "")
        if ts:
            events.append(TimelineEvent(
                timestamp=ts,
                event_type="alert_fired",
                device=alert.get("device"),
                description=alert.get("message", "Alert fired"),
                source="alerts"
            ))

    # Events from raw logs — extract timestamp from syslog format
    # Syslog format: "Apr 09 18:08:03 DEVICE : %ERROR-CODE: message"
    for log_entry in state.raw_logs[:15]:
        line = log_entry.get("line", "")
        if line:
            parts = line.split()
            if len(parts) >= 3:
                # Approximate timestamp — syslog has no year
                ts_str = f"{parts[0]} {parts[1]} {parts[2]}"
                device = parts[3] if len(parts) > 3 else None
                msg = " ".join(parts[5:]) if len(parts) > 5 else line
                events.append(TimelineEvent(
                    timestamp=ts_str,
                    event_type=_classify_log_event(line),
                    device=device,
                    description=msg[:120],
                    source="logs"
                ))

    # Add incident resolution event if we have incident data
    if state.similar_incidents_found:
        inc = state.similar_incidents_found[0]
        if inc.get("resolved_at"):
            events.append(TimelineEvent(
                timestamp=inc["resolved_at"],
                event_type="resolution",
                device=inc.get("affected_device"),
                description=inc.get("resolution", "Incident resolved"),
                source="incident_record"
            ))

    # Sort by timestamp string (works for ISO format and syslog)
    events.sort(key=lambda e: e.timestamp)
    return events


def _classify_log_event(line: str) -> str:
    """Classify a syslog line into an event type."""
    line_lower = line.lower()
    if "bgp" in line_lower and ("down" in line_lower or "adjchange" in line_lower):
        return "bgp_drop"
    if "interface" in line_lower and "down" in line_lower:
        return "interface_down"
    if "crc" in line_lower:
        return "log_error"
    if "cpu" in line_lower:
        return "log_error"
    if "optical" in line_lower or "los" in line_lower:
        return "log_error"
    return "log_error"


def _build_impact(state: AgentState) -> ImpactAnalysis:
    """Build impact analysis from topology and incident data."""
    impact = ImpactAnalysis()

    # Region from state
    if state.identified_region:
        impact.affected_region = state.identified_region

    # Affected devices from topology blast radius
    if state.topology_data:
        devices = list(state.topology_data.get("affected_devices", {}).keys())
        impact.affected_devices = devices[:10]  # cap at 10 for display
        regions = state.topology_data.get("regions_affected", [])
        if regions and not impact.affected_region:
            impact.affected_region = regions[0]
        impact.blast_radius_confirmed = True

    # Customer segment + SLA risk from similar incidents
    if state.similar_incidents_found:
        inc = state.similar_incidents_found[0]
        impact.customer_segment = inc.get("customer_segment")
        impact.estimated_affected_customers = inc.get("affected_customers")
        sev = inc.get("severity", "P3")
        impact.sla_breach_risk = {
            "P1": "CRITICAL", "P2": "HIGH",
            "P3": "MEDIUM",   "P4": "LOW"
        }.get(sev, "MEDIUM")

    return impact


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# Wire all nodes together with edges and the router.
# ─────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph investigation graph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("entry",  entry_node)
    graph.add_node("reason", reason_node)
    graph.add_node("tools",  tools_node)
    graph.add_node("output", output_node)

    # Add edges
    graph.add_edge(START,    "entry")   # always start at entry
    graph.add_edge("entry",  "reason")  # entry always goes to reason
    graph.add_edge("tools",  "reason")  # after tool call, reason again

    # Conditional edge — reason → tools OR output
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"tools": "tools", "output": "output"}
    )

    graph.add_edge("output", END)       # output is always the last node

    return graph.compile()


# ─────────────────────────────────────────────
# PUBLIC INTERFACE
# This is what api/main.py and ui/app.py call.
# ─────────────────────────────────────────────

# Compile once at import time
noc_graph = build_graph()


def run_investigation(
    incident_description: str,
    alert_cluster: dict = None,
    input_alert: dict = None,
) -> AgentState:
    """
    Run a full NOC investigation.

    Args:
        incident_description: Natural language description of the incident
        alert_cluster:        Output from alert correlation engine (optional)
        input_alert:          Raw structured alert JSON (optional)

    Returns:
        Completed AgentState with full RCA output
    """
    initial_state = AgentState(
        incident_description=incident_description,
        alert_cluster=alert_cluster,
        input_alert=input_alert,
    )

    final_state = noc_graph.invoke(initial_state)
    return final_state


def run_investigation_stream(
    incident_description: str,
    alert_cluster: dict = None,
):
    """
    Stream investigation updates as they happen.
    Each yielded item is a dict with the node name and current state.
    Used by the FastAPI WebSocket endpoint for real-time UI updates.
    """
    initial_state = AgentState(
        incident_description=incident_description,
        alert_cluster=alert_cluster,
    )

    for event in noc_graph.stream(initial_state):
        node_name = list(event.keys())[0]
        node_state = event[node_name]
        yield {
            "node": node_name,
            "loop_count": node_state.loop_count,
            "tools_called": len(node_state.tool_calls),
            "evidence_count": len(node_state.evidence),
            "complete": node_state.investigation_complete,
            "rca": node_state.rca.model_dump() if node_state.investigation_complete else None,
        }