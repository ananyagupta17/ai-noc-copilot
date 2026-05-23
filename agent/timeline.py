"""
agent/timeline.py

Incident timeline reconstruction.

What this solves:
  After an incident, engineers write postmortem reports manually.
  They piece together: when did the first alert fire? when did BGP drop?
  when was the customer impacted? when was it resolved?
  This takes 30-60 minutes of manual work per incident.

What we do:
  Read all logs and alerts collected during the investigation,
  parse every timestamp, sort them, classify each event,
  and produce a clean chronological timeline automatically.

Output is used in two places:
  1. state.rca.timeline — the structured timeline object
  2. Streamlit UI timeline panel — rendered as a visual sequence
"""

import re
from datetime import datetime, timezone
from typing import Optional
from agent.state import AgentState, TimelineEvent


# ─────────────────────────────────────────────
# EVENT CLASSIFIERS
# Map log keywords and alert types to human-readable event types
# ─────────────────────────────────────────────

# Syslog keyword → event type
LOG_EVENT_PATTERNS = [
    (r"bgp.*(down|adjchange|notification|hold.timer)", "bgp_drop"),
    (r"interface.*(down|updown)",                      "interface_down"),
    (r"interface.*(up)",                               "interface_up"),
    (r"crc.error",                                     "crc_error"),
    (r"optical.*(los|power|signal|degrad)",            "optical_degradation"),
    (r"cpu.*(high|hog|util)",                          "cpu_spike"),
    (r"memory.*(exhaust|critical|low)",                "memory_exhaustion"),
    (r"mpls|ldp.*(down|drop)",                         "mpls_failure"),
    (r"dns.*(fail|timeout|unreachable)",               "dns_failure"),
    (r"packet.loss|loss.rate",                         "packet_loss"),
    (r"reload|reboot|restart",                         "device_restart"),
    (r"config.*change|configured",                     "config_change"),
]

# Alert type → event type (direct mapping)
ALERT_EVENT_MAP = {
    "packet_loss":        "packet_loss",
    "high_latency":       "latency_spike",
    "bgp_flap":           "bgp_drop",
    "interface_down":     "interface_down",
    "crc_errors":         "crc_error",
    "cpu_spike":          "cpu_spike",
    "memory_exhaustion":  "memory_exhaustion",
    "mpls_failure":       "mpls_failure",
    "dns_outage":         "dns_failure",
    "optical_degradation":"optical_degradation",
}

# Severity of each event type for sorting ties and UI colour coding
EVENT_SEVERITY = {
    "bgp_drop":           "CRITICAL",
    "interface_down":     "CRITICAL",
    "optical_degradation":"CRITICAL",
    "mpls_failure":       "CRITICAL",
    "packet_loss":        "MAJOR",
    "crc_error":          "MAJOR",
    "latency_spike":      "MAJOR",
    "cpu_spike":          "MAJOR",
    "memory_exhaustion":  "MAJOR",
    "dns_failure":        "MAJOR",
    "interface_up":       "INFO",
    "device_restart":     "MAJOR",
    "config_change":      "INFO",
    "alert_fired":        "MAJOR",
    "customer_impact":    "CRITICAL",
    "resolution":         "INFO",
    "investigation_start":"INFO",
    "unknown":            "MINOR",
}


# ─────────────────────────────────────────────
# TIMESTAMP PARSERS
# Syslog and ISO formats need different parsing
# ─────────────────────────────────────────────

# Syslog format: "Apr 09 18:08:03"
SYSLOG_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_syslog_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse syslog-style timestamp like 'Apr 09 18:08:03'."""
    try:
        parts = ts_str.strip().split()
        if len(parts) < 3:
            return None
        month = SYSLOG_MONTHS.get(parts[0])
        if not month:
            return None
        day = int(parts[1])
        time_parts = parts[2].split(":")
        hour, minute, second = int(time_parts[0]), int(time_parts[1]), int(time_parts[2])
        # Use current year — syslog doesn't include it
        year = datetime.now().year
        return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError, KeyError):
        return None


def _parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp like '2025-04-09T18:08:03'."""
    try:
        # Handle both with and without timezone
        ts_str = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        # Strip timezone for consistent comparison
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Try both timestamp formats."""
    if not ts_str:
        return None
    # Try ISO first (more precise)
    dt = _parse_iso_timestamp(ts_str)
    if dt:
        return dt
    # Fall back to syslog
    return _parse_syslog_timestamp(ts_str)


# ─────────────────────────────────────────────
# EVENT CLASSIFIERS
# ─────────────────────────────────────────────

def _classify_log_line(line: str) -> str:
    """Classify a syslog line into an event type using regex patterns."""
    line_lower = line.lower()
    for pattern, event_type in LOG_EVENT_PATTERNS:
        if re.search(pattern, line_lower):
            return event_type
    return "unknown"


def _classify_alert(alert: dict) -> str:
    """Map an alert's alert_type field to an event type."""
    return ALERT_EVENT_MAP.get(alert.get("alert_type", ""), "alert_fired")


# ─────────────────────────────────────────────
# PARSE EVENTS FROM SOURCES
# ─────────────────────────────────────────────

def _events_from_alerts(alerts: list[dict]) -> list[TimelineEvent]:
    """Convert raw alert dicts into TimelineEvents."""
    events = []
    for alert in alerts:
        ts_str = alert.get("timestamp", "")
        dt = _parse_timestamp(ts_str)
        if not dt:
            continue

        event_type = _classify_alert(alert)
        events.append(TimelineEvent(
            timestamp=dt.isoformat(),
            event_type=event_type,
            device=alert.get("device"),
            description=alert.get("message", f"Alert: {event_type}"),
            source="alerts",
        ))
    return events


def _events_from_logs(log_lines: list[dict]) -> list[TimelineEvent]:
    """
    Parse syslog lines into TimelineEvents.
    Syslog format: "Apr 09 18:08:03 DEVICE : %ERROR-CODE: message"
    """
    events = []
    for entry in log_lines:
        line = entry.get("line", "")
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        # Parse timestamp from first 3 tokens
        ts_str = f"{parts[0]} {parts[1]} {parts[2]}"
        dt = _parse_timestamp(ts_str)
        if not dt:
            continue

        # Device is 4th token (before the colon)
        device = parts[3].rstrip(":")

        # Message is everything after " : "
        if " : " in line:
            msg = line.split(" : ", 1)[1].strip()
        else:
            msg = " ".join(parts[4:])

        event_type = _classify_log_line(line)
        events.append(TimelineEvent(
            timestamp=dt.isoformat(),
            event_type=event_type,
            device=device,
            description=msg[:150],
            source="logs",
        ))
    return events


def _events_from_incidents(incidents: list[dict]) -> list[TimelineEvent]:
    """
    Extract key timestamps from historical incident records.
    Adds: detection event, resolution event.
    """
    events = []
    for inc in incidents[:2]:   # only use first 2 to avoid noise

        # Detection event
        detected = _parse_timestamp(inc.get("detected_at", ""))
        if detected:
            events.append(TimelineEvent(
                timestamp=detected.isoformat(),
                event_type="alert_fired",
                device=inc.get("affected_device"),
                description=f"Incident detected: {inc.get('description', '')[:100]}",
                source="incident_record",
            ))

        # Resolution event
        resolved = _parse_timestamp(inc.get("resolved_at", ""))
        if resolved:
            events.append(TimelineEvent(
                timestamp=resolved.isoformat(),
                event_type="resolution",
                device=inc.get("affected_device"),
                description=inc.get("resolution", "Incident resolved"),
                source="incident_record",
            ))

    return events


# ─────────────────────────────────────────────
# DEDUPLICATION
# Multiple sources often produce the same event
# (alert fires and log fires at same time for same device)
# ─────────────────────────────────────────────

def _deduplicate(events: list[TimelineEvent],
                 window_seconds: int = 30) -> list[TimelineEvent]:
    """
    Remove near-duplicate events.
    Two events are duplicates if they have the same event_type,
    same device, and timestamps within window_seconds of each other.
    """
    if not events:
        return []

    deduped = [events[0]]
    for event in events[1:]:
        last = deduped[-1]

        # Check type and device match
        if event.event_type != last.event_type:
            deduped.append(event)
            continue
        if event.device != last.device:
            deduped.append(event)
            continue

        # Check timestamp proximity
        dt_curr = _parse_timestamp(event.timestamp)
        dt_last = _parse_timestamp(last.timestamp)
        if dt_curr and dt_last:
            diff = abs((dt_curr - dt_last).total_seconds())
            if diff > window_seconds:
                deduped.append(event)
        else:
            deduped.append(event)

    return deduped


# ─────────────────────────────────────────────
# ANNOTATE TIMELINE
# Add inferred events that weren't explicitly in logs/alerts
# e.g. "customer impact begins" when CRITICAL alert fires
# ─────────────────────────────────────────────

def _annotate(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """
    Add inferred milestone events to the timeline.
    Looks for the first CRITICAL event and inserts a customer_impact marker.
    """
    annotated = list(events)

    # Find first CRITICAL event → mark customer impact
    for i, event in enumerate(annotated):
        sev = EVENT_SEVERITY.get(event.event_type, "MINOR")
        if sev == "CRITICAL" and event.event_type != "customer_impact":
            impact_event = TimelineEvent(
                timestamp=event.timestamp,
                event_type="customer_impact",
                device=event.device,
                description="Customer impact begins — SLA clock starts",
                source="inferred",
            )
            # Insert right after the triggering event
            annotated.insert(i + 1, impact_event)
            break   # only add one customer_impact marker

    return annotated


# ─────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────

def build_timeline(state: AgentState) -> list[TimelineEvent]:
    """
    Build a complete chronological incident timeline from agent state.

    Sources used (in priority order):
      1. Raw alerts collected during investigation
      2. Raw log lines collected during investigation
      3. Historical incident records (detection + resolution timestamps)

    Returns a sorted, deduplicated, annotated list of TimelineEvents.
    """
    all_events: list[TimelineEvent] = []

    # Source 1 — alerts
    if state.raw_alerts:
        all_events.extend(_events_from_alerts(state.raw_alerts))

    # Source 2 — logs
    if state.raw_logs:
        all_events.extend(_events_from_logs(state.raw_logs))

    # Source 3 — historical incidents (gives us detection + resolution anchor points)
    if state.similar_incidents_found:
        all_events.extend(_events_from_incidents(state.similar_incidents_found))

    if not all_events:
        return []

    # Sort by parsed timestamp — handle mixed formats cleanly
    def sort_key(e: TimelineEvent):
        dt = _parse_timestamp(e.timestamp)
        return dt if dt else datetime.max

    all_events.sort(key=sort_key)

    # Deduplicate
    all_events = _deduplicate(all_events, window_seconds=30)

    # Annotate with inferred events
    all_events = _annotate(all_events)

    return all_events


def timeline_to_display(events: list[TimelineEvent]) -> list[dict]:
    """
    Convert timeline events to display-ready dicts for Streamlit.
    Adds severity and icon fields.
    """
    icons = {
        "bgp_drop":           "🔴",
        "interface_down":     "🔴",
        "interface_up":       "🟢",
        "optical_degradation":"🔴",
        "mpls_failure":       "🔴",
        "packet_loss":        "🟠",
        "crc_error":          "🟠",
        "latency_spike":      "🟠",
        "cpu_spike":          "🟠",
        "memory_exhaustion":  "🟠",
        "dns_failure":        "🟠",
        "device_restart":     "🟡",
        "config_change":      "🔵",
        "alert_fired":        "🟠",
        "customer_impact":    "🚨",
        "resolution":         "✅",
        "investigation_start":"🔍",
        "unknown":            "⚪",
    }

    display = []
    for e in events:
        dt = _parse_timestamp(e.timestamp)
        time_str = dt.strftime("%H:%M:%S") if dt else e.timestamp

        display.append({
            "time":        time_str,
            "timestamp":   e.timestamp,
            "event_type":  e.event_type,
            "severity":    EVENT_SEVERITY.get(e.event_type, "MINOR"),
            "icon":        icons.get(e.event_type, "⚪"),
            "device":      e.device or "—",
            "description": e.description,
            "source":      e.source,
        })

    return display


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from agent.state import AgentState, TimelineEvent

    state = AgentState(
        incident_description="Test — Singapore packet loss"
    )

    # Simulate some raw data
    state.raw_alerts = [
        {
            "timestamp": "2025-04-09T18:08:00",
            "alert_type": "packet_loss",
            "device": "SIN-ER-01",
            "message": "CRITICAL: Packet loss 35% on SIN-ER-01",
            "severity": "CRITICAL",
        },
        {
            "timestamp": "2025-04-09T18:09:00",
            "alert_type": "bgp_flap",
            "device": "SIN-CR-01",
            "message": "CRITICAL: BGP session DOWN on SIN-CR-01",
            "severity": "CRITICAL",
        },
    ]
    state.raw_logs = [
        {"line": "Apr 09 18:07:45 SIN-ER-01 : %OPTICAL-3-RXPOWER_LOW: Rx power -28dBm on Gi0/1"},
        {"line": "Apr 09 18:08:10 SIN-CR-01 : %BGP-5-ADJCHANGE: neighbor 10.1.1.1 Down Hold Timer Expired"},
        {"line": "Apr 09 18:09:30 SIN-CR-01 : %MPLS-3-LDP_NBR_DOWN: LDP session dropped"},
    ]
    state.similar_incidents_found = [
        {
            "detected_at": "2025-04-09T18:07:00",
            "resolved_at": "2025-04-09T20:15:00",
            "affected_device": "SIN-ER-01",
            "resolution": "Traffic rerouted via backup MPLS path",
            "description": "Optical degradation on Singapore edge router",
        }
    ]

    timeline = build_timeline(state)
    display = timeline_to_display(timeline)

    print(f"\nTimeline — {len(display)} events\n")
    print(f"{'Time':10} {'Icon'} {'Type':22} {'Severity':10} {'Device':12} Description")
    print("-" * 100)
    for e in display:
        print(
            f"{e['time']:10} {e['icon']}  "
            f"{e['event_type']:22} {e['severity']:10} "
            f"{e['device']:12} {e['description'][:55]}"
        )