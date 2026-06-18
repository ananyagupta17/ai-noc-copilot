"""
api/main.py

FastAPI backend for the AI NOC Copilot.

Two types of endpoints:
  REST      — list incidents, trigger investigations, expose observability
  WebSocket — stream the agent investigation in real time

The WebSocket endpoint is the important one.
Instead of the UI waiting 40 seconds for a complete response,
it receives updates as each node completes:
  → entry node done  (investigation started)
  → tools node done  (tool X called, result Y)
  → reason node done (loop N complete)
  → output node done (full RCA ready)

This is how production AI systems surface intermediate results.
Engineers see the investigation happening live, not a spinner.

Scope note: the agent itself does NOT call this API — it invokes its tools
in-process — so there is no need to expose a REST endpoint per tool. The
endpoints here are the ones the system consumes: the Streamlit UI, the
streaming interface, and observability. Per-device/topology/alert data-browse
endpoints were removed because nothing consumed them; the underlying tool
functions still exist for the agent's use and can be re-exposed if a future UI
needs them.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.graph import run_investigation, run_investigation_stream, _LIVE_EVENTS
from alert_correlation.clusterer import correlate_alerts
from observability.tracer import tracer
from observability.logger import noc_logger

# Data query tool — used by the /incidents REST endpoint
from agent.tools.incidents import get_recent_incidents


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI NOC Copilot",
    description="Agentic network operations assistant for large-scale telecom infrastructure",
    version="1.0.0",
)

# Allow Streamlit (running on port 8501) to call FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class InvestigationRequest(BaseModel):
    incident_description: str
    region: Optional[str] = None
    run_correlation: bool = True


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "AI NOC Copilot",
        "status":  "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# ─────────────────────────────────────────────
# REST — INCIDENTS
# ─────────────────────────────────────────────

@app.get("/incidents")
def list_incidents(
    region:   Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    limit:    int           = Query(20, le=100),
):
    """List recent incidents, optionally filtered by region or severity."""
    return get_recent_incidents(region=region, severity=severity, limit=limit)


# ─────────────────────────────────────────────
# REST — INVESTIGATE (synchronous)
# Runs a full investigation and returns when complete.
# Use this for testing. Use WebSocket for the UI.
# ─────────────────────────────────────────────

@app.post("/investigate")
def investigate(req: InvestigationRequest):
    """
    Run a full investigation synchronously.
    Returns complete RCA when done.
    For real-time streaming use the WebSocket endpoint instead.
    """
    # Step 1 — run alert correlation
    cluster = None
    if req.run_correlation:
        cluster = correlate_alerts(
            description=req.incident_description,
            region=req.region,
        )

    # Step 2 — run agent investigation
    result = run_investigation(
        incident_description=req.incident_description,
        alert_cluster=cluster,
    )

    # Step 3 — extract RCA from result dict
    rca = result.get("rca", {})
    if hasattr(rca, "model_dump"):
        rca = rca.model_dump()

    return {
        "status":          "complete",
        "investigation_id": tracer.get_current().investigation_id if tracer.get_current() else None,
        "tool_calls":      len(result.get("tool_calls", [])),
        "evidence_count":  len(result.get("evidence", [])),
        "loop_count":      result.get("loop_count", 0),
        "alert_cluster":   cluster,
        "rca":             rca,
        "observability":   tracer.get_summary(),
    }


# ─────────────────────────────────────────────
# WEBSOCKET — INVESTIGATE (streaming)
# Streams investigation updates as they happen.
# Each message is a JSON object with the current state.
# ─────────────────────────────────────────────

@app.websocket("/ws/investigate")
async def ws_investigate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time investigation streaming.

    Client sends:
      {"incident_description": "...", "region": "Singapore", "run_correlation": true}

    Server streams:
      {"event": "started",    "data": {...}}
      {"event": "tool_call",  "data": {"tool": "...", "loop": N}}
      {"event": "loop_done",  "data": {"loop": N, "evidence": N}}
      {"event": "complete",   "data": {"rca": {...}, "observability": {...}}}
      {"event": "error",      "data": {"message": "..."}}
    """
    await websocket.accept()

    try:
        # Receive investigation request
        raw = await websocket.receive_text()
        req_data = json.loads(raw)

        incident_description = req_data.get("incident_description", "")
        region               = req_data.get("region")
        run_correlation      = req_data.get("run_correlation", True)

        if not incident_description:
            await websocket.send_json({
                "event": "error",
                "data":  {"message": "incident_description is required"}
            })
            return

        # Send started event
        await websocket.send_json({
            "event": "started",
            "data":  {
                "incident_description": incident_description,
                "timestamp": datetime.now().isoformat(),
            }
        })

        # Run alert correlation
        cluster = None
        if run_correlation:
            cluster = correlate_alerts(
                description=incident_description,
                region=region,
            )
            if cluster:
                await websocket.send_json({
                    "event": "correlation_done",
                    "data":  cluster,
                })

        # Stream the agent investigation
        # run_investigation_stream yields after each node completes
        for update in run_investigation_stream(
            incident_description=incident_description,
            alert_cluster=cluster,
        ):
            node = update.get("node", "")

            if node == "tools":
                await websocket.send_json({
                    "event": "tool_call",
                    "data":  {
                        "loop":           update.get("loop_count"),
                        "tools_called":   update.get("tools_called"),
                        "evidence_count": update.get("evidence_count"),
                    }
                })

            elif node == "reason":
                await websocket.send_json({
                    "event": "loop_done",
                    "data":  {
                        "loop":           update.get("loop_count"),
                        "evidence_count": update.get("evidence_count"),
                    }
                })

            elif node == "output" and update.get("complete"):
                rca = update.get("rca", {})
                if hasattr(rca, "model_dump"):
                    rca = rca.model_dump()

                await websocket.send_json({
                    "event": "complete",
                    "data":  {
                        "rca":           rca,
                        "observability": tracer.get_summary(),
                        "tool_traces":   [
                            {
                                "tool":        t["tool_name"],
                                "duration_ms": t["duration_ms"],
                                "success":     t["success"],
                                "loop":        t["loop_number"],
                            }
                            for t in tracer.to_dict().get("tool_traces", [])
                        ],
                    }
                })
                break

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        noc_logger.error("websocket", str(e))
        try:
            await websocket.send_json({
                "event": "error",
                "data":  {"message": str(e)}
            })
        except Exception:
            pass


# ─────────────────────────────────────────────
# REST — OBSERVABILITY
# ─────────────────────────────────────────────

@app.get("/status/events")
def get_live_events():
    """Live event log polled by the UI while an investigation is running."""
    return _LIVE_EVENTS


@app.get("/observability/summary")
def observability_summary():
    """Get summary of the most recent investigation."""
    return tracer.get_summary()


@app.get("/observability/trace")
def observability_trace():
    """Get full trace of the most recent investigation."""
    return tracer.to_dict()


@app.get("/observability/logs")
def observability_logs(n: int = Query(50, le=200)):
    """Get recent structured log entries."""
    return noc_logger.read_recent_logs(n=n)


@app.get("/observability/confidence")
def confidence_history():
    """Get confidence evolution across loops for current investigation."""
    return noc_logger.get_confidence_history()


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
