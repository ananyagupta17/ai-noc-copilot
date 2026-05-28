"""
ui/app.py

Streamlit dashboard for the AI NOC Copilot.

Run:
    streamlit run ui/app.py

Layout:
  Sidebar   — incident input, recent incidents, alert cluster
  Tab 1     — RCA output (cause, confidence, evidence, impact, actions)
  Tab 2     — Incident timeline
  Tab 3     — Tool trace (what the agent called and when)
  Tab 4     — Observability (confidence evolution, RAG quality, logs)
"""

import json
import sys
import time
import requests
import websocket
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE = "http://localhost:8000"
WS_BASE  = "ws://localhost:8000"

st.set_page_config(
    page_title="AI NOC Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# Dark terminal aesthetic — fits a NOC environment
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.noc-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.noc-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(30,90,180,0.03) 2px, rgba(30,90,180,0.03) 4px
    );
    pointer-events: none;
}
.noc-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #4da6ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.noc-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #4a6fa5;
    margin-top: 4px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Confidence badge */
.confidence-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 600;
    padding: 12px 24px;
    border-radius: 8px;
    margin-bottom: 8px;
}
.conf-very-high { background: rgba(0,200,100,0.15); color: #00c864; border: 1px solid #00c864; }
.conf-high      { background: rgba(0,180,220,0.15); color: #00b4dc; border: 1px solid #00b4dc; }
.conf-medium    { background: rgba(255,180,0,0.15);  color: #ffb400; border: 1px solid #ffb400; }
.conf-low       { background: rgba(255,80,80,0.15);  color: #ff5050; border: 1px solid #ff5050; }

/* Cards */
.noc-card {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}
.noc-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a6fa5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.noc-card-value {
    font-size: 15px;
    color: #e2e8f0;
    line-height: 1.6;
}

/* Evidence items */
.evidence-item {
    border-left: 3px solid #1e3a5f;
    padding: 10px 16px;
    margin-bottom: 8px;
    background: rgba(30,58,95,0.2);
    border-radius: 0 6px 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
}
.evidence-type-log     { border-color: #ff6b6b; }
.evidence-type-historical { border-color: #4da6ff; }
.evidence-type-runbook { border-color: #00c864; }
.evidence-type-alert   { border-color: #ffb400; }
.evidence-type-topology{ border-color: #a78bfa; }
.evidence-type-metric  { border-color: #38bdf8; }

/* Timeline */
.timeline-event {
    display: flex;
    gap: 16px;
    padding: 12px 0;
    border-bottom: 1px solid #1e2d40;
    align-items: flex-start;
}
.timeline-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #4a6fa5;
    min-width: 80px;
    padding-top: 2px;
}
.timeline-icon { font-size: 18px; min-width: 24px; }
.timeline-content { flex: 1; }
.timeline-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a6fa5;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.timeline-desc { font-size: 13px; color: #cbd5e1; margin-top: 2px; }
.timeline-device {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4da6ff;
    margin-top: 2px;
}

/* Tool trace */
.tool-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    background: rgba(13,17,23,0.8);
    border: 1px solid #1e2d40;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.tool-success { border-left: 3px solid #00c864; }
.tool-fail    { border-left: 3px solid #ff5050; }
.tool-name  { color: #4da6ff; flex: 1; }
.tool-time  { color: #4a6fa5; min-width: 70px; text-align: right; }
.tool-loop  { color: #a78bfa; min-width: 50px; }
.tool-status-ok  { color: #00c864; }
.tool-status-err { color: #ff5050; }

/* Metric boxes */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #4da6ff;
}
.metric-lbl {
    font-size: 11px;
    color: #4a6fa5;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Action items */
.action-item {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #1e2d40;
    font-size: 14px;
    color: #cbd5e1;
    align-items: flex-start;
}
.action-num {
    font-family: 'JetBrains Mono', monospace;
    color: #4a6fa5;
    min-width: 24px;
}

/* Status pill */
.status-pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
}
.pill-critical { background: rgba(255,80,80,0.2);  color: #ff5050; }
.pill-high     { background: rgba(255,180,0,0.2);  color: #ffb400; }
.pill-medium   { background: rgba(0,180,220,0.2);  color: #00b4dc; }
.pill-low      { background: rgba(0,200,100,0.2);  color: #00c864; }

/* Sidebar */
.sidebar-section {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
}
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4a6fa5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Log viewer */
.log-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 4px 0;
    border-bottom: 1px solid #1a2332;
    display: flex;
    gap: 12px;
}
.log-ts    { color: #4a6fa5; min-width: 85px; }
.log-level-INFO  { color: #4da6ff; min-width: 50px; }
.log-level-ERROR { color: #ff5050; min-width: 50px; }
.log-level-WARN  { color: #ffb400; min-width: 50px; }
.log-level-DEBUG { color: #64748b; min-width: 50px; }
.log-event { color: #94a3b8; flex: 1; }

/* Blinking cursor for "live" feel */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.cursor { animation: blink 1s infinite; color: #4da6ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def api_get(path: str, params: dict = None):
    """GET request to FastAPI with graceful error handling."""
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure `uvicorn api.main:app --port 8000` is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, payload: dict):
    """POST request to FastAPI."""
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def confidence_class(score: float) -> str:
    if score >= 0.80: return "conf-very-high"
    if score >= 0.60: return "conf-high"
    if score >= 0.40: return "conf-medium"
    return "conf-low"


def confidence_label(score: float) -> str:
    if score >= 0.80: return "VERY HIGH"
    if score >= 0.60: return "HIGH"
    if score >= 0.40: return "MEDIUM"
    return "LOW"


def sla_pill(risk: str) -> str:
    mapping = {
        "CRITICAL": "pill-critical",
        "HIGH":     "pill-high",
        "MEDIUM":   "pill-medium",
        "LOW":      "pill-low",
    }
    cls = mapping.get(risk, "pill-medium")
    return f'<span class="status-pill {cls}">{risk}</span>'


def severity_pill(sev: str) -> str:
    mapping = {
        "P1": "pill-critical",
        "P2": "pill-high",
        "P3": "pill-medium",
        "P4": "pill-low",
    }
    cls = mapping.get(sev, "pill-medium")
    return f'<span class="status-pill {cls}">{sev}</span>'


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="noc-header">
    <p class="noc-title">🛡️ AI NOC Copilot</p>
    <p class="noc-subtitle">Network Operations Intelligence · Powered by Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "rca_result"    not in st.session_state: st.session_state.rca_result    = None
if "trace_result"  not in st.session_state: st.session_state.trace_result  = None
if "cluster"       not in st.session_state: st.session_state.cluster       = None
if "investigating" not in st.session_state: st.session_state.investigating = False
if "logs"          not in st.session_state: st.session_state.logs          = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-label">Incident Input</div>
    """, unsafe_allow_html=True)

    incident_input = st.text_area(
        label="Incident description",
        placeholder="e.g. Singapore enterprise customers reporting packet loss and BGP instability on edge routers...",
        height=120,
        label_visibility="collapsed",
    )

    region_input = st.selectbox(
        "Region filter (optional)",
        ["", "Singapore", "Mumbai", "London", "Frankfurt",
         "New York", "Tokyo", "Sydney", "Dubai"],
        label_visibility="visible",
    )

    run_correlation = st.checkbox("Run alert correlation", value=True)

    investigate_btn = st.button(
        "🔍 Investigate",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.investigating,
    )

    st.divider()

    # Recent incidents
    st.markdown('<div class="sidebar-label">Recent Incidents</div>', unsafe_allow_html=True)
    recent = api_get("/incidents", params={"limit": 8})
    if recent:
        for inc in recent:
            sev   = inc.get("severity", "P3")
            reg   = inc.get("region", "")
            sym   = inc.get("symptom", "").replace("_", " ")
            iid   = inc.get("incident_id", "")
            color = {"P1":"#ff5050","P2":"#ffb400","P3":"#4da6ff","P4":"#64748b"}.get(sev,"#4da6ff")
            st.markdown(f"""
            <div style="padding:8px 0;border-bottom:1px solid #1e2d40;font-size:12px">
                <span style="color:{color};font-family:'JetBrains Mono',monospace;font-weight:600">{sev}</span>
                <span style="color:#94a3b8;margin-left:8px">{iid}</span><br>
                <span style="color:#4a6fa5;font-size:11px">{reg} · {sym}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Alert cluster display
    if st.session_state.cluster:
        c = st.session_state.cluster
        st.markdown('<div class="sidebar-label">Alert Cluster</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-section">
            <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#4da6ff">{c.get('cluster_id','')}</div>
            <div style="font-size:13px;color:#e2e8f0;margin-top:6px">{c.get('dominant_symptom','').replace('_',' ').title()}</div>
            <div style="font-size:11px;color:#4a6fa5;margin-top:4px">
                {c.get('total_alert_count',0)} alerts · {c.get('noise_reduced',0)} noise filtered
            </div>
            <div style="font-size:11px;color:#94a3b8;margin-top:4px">{', '.join(c.get('affected_regions',[]))}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INVESTIGATION TRIGGER
# ─────────────────────────────────────────────

if investigate_btn and incident_input.strip():
    st.session_state.investigating = True

    with st.spinner("Running alert correlation..."):
        result = api_post("/investigate", {
            "incident_description": incident_input.strip(),
            "region": region_input if region_input else None,
            "run_correlation": run_correlation,
        })

    if result:
        st.session_state.rca_result   = result
        st.session_state.cluster      = result.get("alert_cluster")
        # Fetch trace after investigation
        st.session_state.trace_result = api_get("/observability/trace")
        st.session_state.logs         = api_get("/observability/logs") or []

    st.session_state.investigating = False
    st.rerun()

elif investigate_btn and not incident_input.strip():
    st.warning("Please enter an incident description.")


# ─────────────────────────────────────────────
# MAIN CONTENT — TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Root Cause Analysis",
    "⏱️ Incident Timeline",
    "🔧 Tool Trace",
    "📊 Observability",
])


# ══════════════════════════════════════════════
# TAB 1 — RCA
# ══════════════════════════════════════════════

with tab1:
    if not st.session_state.rca_result:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;color:#4a6fa5">
            <div style="font-size:48px;margin-bottom:16px">🔍</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:14px">
                Enter an incident description and click Investigate
            </div>
            <div style="font-size:12px;margin-top:8px;color:#2d4a6a">
                The AI agent will investigate, gather evidence, and generate a root cause analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state.rca_result
        rca    = result.get("rca", {})
        obs    = result.get("observability", {})

        # Top metrics row
        confidence = rca.get("confidence_score", 0)
        conf_pct   = f"{confidence:.0%}"
        conf_cls   = confidence_class(confidence)
        conf_lbl   = confidence_label(confidence)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="noc-card" style="text-align:center">
                <div class="noc-card-title">Confidence</div>
                <div class="confidence-badge {conf_cls}">{conf_pct}</div>
                <div style="font-size:11px;color:#4a6fa5;font-family:'JetBrains Mono',monospace">{conf_lbl}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="noc-card" style="text-align:center">
                <div class="noc-card-title">Tool Calls</div>
                <div class="metric-val">{result.get('tool_calls', 0)}</div>
                <div class="metric-lbl">executed</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="noc-card" style="text-align:center">
                <div class="noc-card-title">Evidence Items</div>
                <div class="metric-val">{result.get('evidence_count', 0)}</div>
                <div class="metric-lbl">collected</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="noc-card" style="text-align:center">
                <div class="noc-card-title">Agent Loops</div>
                <div class="metric-val">{result.get('loop_count', 0)}</div>
                <div class="metric-lbl">reasoning cycles</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            # Probable cause
            st.markdown(f"""
            <div class="noc-card">
                <div class="noc-card-title">Probable Root Cause</div>
                <div class="noc-card-value" style="font-size:16px;font-weight:500;color:#e2e8f0">
                    {rca.get('probable_cause', 'Under investigation')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Incident summary
            st.markdown(f"""
            <div class="noc-card">
                <div class="noc-card-title">Incident Summary</div>
                <div class="noc-card-value">{rca.get('incident_summary', '—')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Recommended actions
            actions = rca.get("recommended_actions", [])
            if actions:
                actions_html = "".join([
                    f'<div class="action-item"><span class="action-num">{i+1}.</span><span>{a}</span></div>'
                    for i, a in enumerate(actions)
                ])
                st.markdown(f"""
                <div class="noc-card">
                    <div class="noc-card-title">Recommended Actions</div>
                    {actions_html}
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            # Impact analysis
            impact = rca.get("impact", {})
            sla    = impact.get("sla_breach_risk", "MEDIUM")
            st.markdown(f"""
            <div class="noc-card">
                <div class="noc-card-title">Impact Analysis</div>
                <div style="margin-bottom:12px">{sla_pill(sla)}</div>
                <div style="font-size:13px;color:#94a3b8;margin-bottom:6px">
                    <span style="color:#4a6fa5">Region:</span>&nbsp;
                    {impact.get('affected_region') or '—'}
                </div>
                <div style="font-size:13px;color:#94a3b8;margin-bottom:6px">
                    <span style="color:#4a6fa5">Customer segment:</span>&nbsp;
                    {impact.get('customer_segment') or '—'}
                </div>
                <div style="font-size:13px;color:#94a3b8;margin-bottom:6px">
                    <span style="color:#4a6fa5">Est. affected:</span>&nbsp;
                    {impact.get('estimated_affected_customers') or '—'} customers
                </div>
                <div style="font-size:13px;color:#94a3b8">
                    <span style="color:#4a6fa5">Affected devices:</span>&nbsp;
                    {len(impact.get('affected_devices', []))} confirmed
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Escalation
            st.markdown(f"""
            <div class="noc-card">
                <div class="noc-card-title">Escalation Team</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:14px;color:#4da6ff">
                    {rca.get('escalation_team') or '—'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Evidence breakdown
            evidence = rca.get("evidence", [])
            if evidence:
                type_counts = {}
                for e in evidence:
                    t = e.get("type", "unknown") if isinstance(e, dict) else getattr(e, "type", "unknown")
                    type_counts[t] = type_counts.get(t, 0) + 1

                type_colors = {
                    "log": "#ff6b6b", "historical_incident": "#4da6ff",
                    "runbook": "#00c864", "alert": "#ffb400",
                    "topology": "#a78bfa", "metric": "#38bdf8",
                }

                st.markdown('<div class="noc-card"><div class="noc-card-title">Evidence Breakdown</div>', unsafe_allow_html=True)
                for t, count in type_counts.items():
                    color = type_colors.get(t, "#64748b")
                    label = f"{count} item{'s' if count > 1 else ''}"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                        f'border-bottom:1px solid #1e2d40;font-size:12px">'
                        f'<span style="color:{color};font-family:JetBrains Mono,monospace">{t}</span>'
                        f'<span style="color:#94a3b8">{label}</span></div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — TIMELINE
# ══════════════════════════════════════════════

with tab2:
    if not st.session_state.rca_result:
        st.markdown('<div style="padding:80px 0;text-align:center;color:#4a6fa5">Run an investigation to see the timeline.</div>', unsafe_allow_html=True)
    else:
        rca      = st.session_state.rca_result.get("rca", {})
        timeline = rca.get("timeline", [])

        if not timeline:
            st.markdown("""
            <div class="noc-card">
                <div class="noc-card-title">Timeline</div>
                <div style="color:#4a6fa5;font-size:13px;padding:20px 0;text-align:center">
                    No timeline events reconstructed.<br>
                    <span style="font-size:11px">This happens when log files for the investigated incident aren't available.
                    The agent retrieved evidence from other sources.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Import display helper
            sys.path.append(str(Path(__file__).parent.parent))
            from agent.timeline import timeline_to_display, TimelineEvent

            # Convert to display format
            if timeline and isinstance(timeline[0], dict):
                display_events = timeline
            else:
                display_events = timeline_to_display(timeline)

            st.markdown(f"""
            <div class="noc-card">
                <div class="noc-card-title">Incident Timeline — {len(display_events)} events</div>
            """, unsafe_allow_html=True)

            for ev in display_events:
                t    = ev.get("time", ev.get("timestamp", ""))[:8]
                icon = ev.get("icon", "⚪")
                etype= ev.get("event_type", "")
                sev  = ev.get("severity", "MINOR")
                dev  = ev.get("device", "—")
                desc = ev.get("description", "")
                src  = ev.get("source", "")

                sev_colors = {
                    "CRITICAL": "#ff5050", "MAJOR": "#ffb400",
                    "INFO": "#4da6ff", "MINOR": "#64748b"
                }
                sev_color = sev_colors.get(sev, "#64748b")

                st.markdown(f"""
                <div class="timeline-event">
                    <div class="timeline-time">{t}</div>
                    <div class="timeline-icon">{icon}</div>
                    <div class="timeline-content">
                        <div class="timeline-type" style="color:{sev_color}">{etype.replace('_',' ').upper()} · {sev}</div>
                        <div class="timeline-desc">{desc}</div>
                        <div class="timeline-device">{dev} · <span style="color:#4a6fa5">{src}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — TOOL TRACE
# ══════════════════════════════════════════════

with tab3:
    if not st.session_state.trace_result:
        st.markdown('<div style="padding:80px 0;text-align:center;color:#4a6fa5">Run an investigation to see the tool trace.</div>', unsafe_allow_html=True)
    else:
        trace      = st.session_state.trace_result
        tool_traces = trace.get("tool_traces", [])
        rag_traces  = trace.get("rag_traces", [])
        loop_traces = trace.get("loop_traces", [])

        # Summary row
        summary = trace.get("summary", {})
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("Total Duration", f"{summary.get('total_duration_ms',0):.0f}ms"),
            ("Tool Success Rate", f"{summary.get('tool_success_rate',0):.0f}%"),
            ("Avg Tool Time", f"{summary.get('avg_tool_time_ms',0):.1f}ms"),
            ("RAG Retrievals", str(summary.get("rag_retrievals", 0))),
        ]
        for col, (label, val) in zip([col1,col2,col3,col4], metrics):
            with col:
                st.markdown(f"""
                <div class="noc-card" style="text-align:center">
                    <div class="noc-card-title">{label}</div>
                    <div class="metric-val" style="font-size:22px">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            # Tool call waterfall
            st.markdown("""
            <div class="noc-card">
                <div class="noc-card-title">Tool Call Waterfall</div>
            """, unsafe_allow_html=True)

            for tc in tool_traces:
                success  = tc.get("success", True)
                cls      = "tool-success" if success else "tool-fail"
                status   = "✓" if success else "✗"
                scls     = "tool-status-ok" if success else "tool-status-err"
                name     = tc.get("tool_name", "")
                dur      = tc.get("duration_ms", 0)
                loop     = tc.get("loop_number", 0)
                summary_txt = tc.get("output_summary", "")

                st.markdown(f"""
                <div class="tool-row {cls}">
                    <span class="{scls}">{status}</span>
                    <span class="tool-loop">L{loop}</span>
                    <span class="tool-name">{name}</span>
                    <span style="color:#64748b;font-size:11px;flex:1">{summary_txt[:40]}</span>
                    <span class="tool-time">{dur:.1f}ms</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            # RAG retrievals
            if rag_traces:
                st.markdown("""
                <div class="noc-card">
                    <div class="noc-card-title">RAG Retrievals</div>
                """, unsafe_allow_html=True)

                for rt in rag_traces:
                    quality = rt.get("top_similarity", 0)
                    q_color = "#00c864" if quality > 0.7 else "#ffb400" if quality > 0.4 else "#ff5050"
                    st.markdown(f"""
                    <div style="padding:8px 0;border-bottom:1px solid #1e2d40">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                    color:#4a6fa5;margin-bottom:4px">
                            {rt.get('retrieved_at','')[:19]}
                        </div>
                        <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">
                            {rt.get('query','')[:60]}
                        </div>
                        <div style="display:flex;gap:12px;font-size:11px">
                            <span style="color:#4da6ff">{rt.get('chunks_retrieved',0)} chunks</span>
                            <span style="color:{q_color}">quality {quality:.0%}</span>
                            <span style="color:#4a6fa5">{rt.get('duration_ms',0):.1f}ms</span>
                        </div>
                        <div style="font-size:11px;color:#4a6fa5;margin-top:4px">
                            {', '.join(rt.get('sources_hit',[]))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # Loop summary
            if loop_traces:
                st.markdown("""
                <div class="noc-card">
                    <div class="noc-card-title">Reasoning Loops</div>
                """, unsafe_allow_html=True)

                for lt in loop_traces:
                    conf = lt.get("confidence_so_far", 0)
                    st.markdown(f"""
                    <div style="padding:8px 0;border-bottom:1px solid #1e2d40;font-size:12px">
                        <div style="display:flex;justify-content:space-between">
                            <span style="font-family:'JetBrains Mono',monospace;color:#4da6ff">
                                Loop {lt.get('loop_number',0)}
                            </span>
                            <span style="color:#00c864">{conf:.0%} confidence</span>
                        </div>
                        <div style="color:#4a6fa5;margin-top:4px">
                            {', '.join(lt.get('tools_selected',[]))}
                        </div>
                        <div style="color:#64748b;font-size:11px;margin-top:2px">
                            {lt.get('evidence_count',0)} evidence items · {lt.get('duration_ms',0):.0f}ms
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — OBSERVABILITY
# ══════════════════════════════════════════════

with tab4:
    st.markdown("""
    <div class="noc-card">
        <div class="noc-card-title">AI System Observability</div>
        <div style="font-size:12px;color:#4a6fa5">
            Real-time visibility into agent decisions, tool execution, and RAG quality.
            This panel makes the AI's reasoning transparent and auditable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Confidence evolution chart
        if st.session_state.trace_result:
            loop_traces = st.session_state.trace_result.get("loop_traces", [])
            if loop_traces:
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Loop": lt["loop_number"],
                        "Confidence": lt["confidence_so_far"] * 100,
                        "Evidence": lt["evidence_count"],
                    }
                    for lt in loop_traces
                ])
                st.markdown('<div class="noc-card-title">Confidence Evolution</div>', unsafe_allow_html=True)
                st.line_chart(df.set_index("Loop")[["Confidence"]], height=200)
                st.markdown('<div class="noc-card-title" style="margin-top:8px">Evidence Accumulation</div>', unsafe_allow_html=True)
                st.bar_chart(df.set_index("Loop")[["Evidence"]], height=150)
        else:
            st.markdown("""
            <div class="noc-card">
                <div class="noc-card-title">Confidence Evolution</div>
                <div style="color:#4a6fa5;text-align:center;padding:40px 0;font-size:13px">
                    Run an investigation to see confidence evolution
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # Live log viewer
        st.markdown('<div class="noc-card-title">Structured Audit Log</div>', unsafe_allow_html=True)

        if st.button("🔄 Refresh logs"):
            st.session_state.logs = api_get("/observability/logs") or []

        logs = st.session_state.logs
        if logs:
            log_html = '<div class="noc-card" style="max-height:380px;overflow-y:auto">'
            for log in logs[-30:]:
                ts    = log.get("ts", "")[:19].replace("T", " ")
                level = log.get("level", "INFO")
                event = log.get("event", "")
                lcls  = f"log-level-{level}"
                log_html += f"""
                <div class="log-line">
                    <span class="log-ts">{ts}</span>
                    <span class="{lcls}">{level}</span>
                    <span class="log-event">{event}</span>
                </div>"""
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="noc-card">
                <div style="color:#4a6fa5;text-align:center;padding:40px 0;font-size:13px">
                    No logs yet — run an investigation
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Full trace JSON viewer
    if st.session_state.trace_result:
        with st.expander("📋 Full Investigation Trace (JSON)"):
            st.json(st.session_state.trace_result)