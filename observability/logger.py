"""
observability/logger.py

Structured logging for the NOC agent.

Different from the tracer:
  tracer.py  = in-memory runtime recording (used during investigation)
  logger.py  = persistent structured logs written to disk (audit trail)

What gets logged:
  - Every investigation start/end with full context
  - Every tool call with params and result summary
  - Every RAG retrieval with query and quality score
  - Confidence evolution across loops
  - Any errors or unexpected behaviour

Log format: one JSON object per line (JSONL)
  - Easy to parse with any tool
  - Easy to query with Python
  - Easy to ingest into any log aggregator if this goes to production

The Streamlit observability panel reads from both
tracer (real-time) and logger (historical).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# LOG LEVELS
# ─────────────────────────────────────────────

INFO  = "INFO"
WARN  = "WARN"
ERROR = "ERROR"
DEBUG = "DEBUG"


# ─────────────────────────────────────────────
# STRUCTURED LOGGER
# ─────────────────────────────────────────────

class NOCLogger:

    def __init__(self, log_dir: Path = None):
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "db" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # One log file per day
        today = datetime.now().strftime("%Y-%m-%d")
        self._log_path = log_dir / f"noc_agent_{today}.jsonl"

        # Confidence evolution tracking
        self._confidence_history: list[dict] = []
        self._current_investigation_id: Optional[str] = None

    # ── Core write ────────────────────────────

    def _write(self, level: str, event: str, data: dict):
        """Write one structured log line."""
        record = {
            "ts":    datetime.now().isoformat(),
            "level": level,
            "event": event,
            "inv":   self._current_investigation_id,
            **data,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass    # never let logging break the main flow

    # ── Investigation lifecycle ───────────────

    def investigation_started(self, investigation_id: str, description: str):
        self._current_investigation_id = investigation_id
        self._confidence_history = []
        self._write(INFO, "investigation.started", {
            "investigation_id": investigation_id,
            "description":      description[:200],
        })

    def investigation_completed(
        self,
        investigation_id: str,
        confidence: float,
        tool_count: int,
        loop_count: int,
        duration_ms: float,
        probable_cause: Optional[str],
    ):
        self._write(INFO, "investigation.completed", {
            "investigation_id": investigation_id,
            "confidence":       confidence,
            "tool_count":       tool_count,
            "loop_count":       loop_count,
            "duration_ms":      duration_ms,
            "probable_cause":   probable_cause,
            "confidence_history": self._confidence_history,
        })

    # ── Tool call logging ─────────────────────

    def tool_called(
        self,
        tool_name: str,
        params: dict,
        success: bool,
        duration_ms: float,
        result_summary: str,
        loop_number: int,
    ):
        level = INFO if success else ERROR
        self._write(level, "tool.called", {
            "tool":          tool_name,
            "params":        params,
            "success":       success,
            "duration_ms":   round(duration_ms, 2),
            "result":        result_summary[:200],
            "loop":          loop_number,
        })

    # ── RAG quality logging ───────────────────

    def rag_retrieved(
        self,
        query: str,
        chunks_retrieved: int,
        sources: list[str],
        duration_ms: float,
    ):
        # Quality score: penalise zero results, reward diverse sources
        quality = 0.0
        if chunks_retrieved > 0:
            quality = min(1.0, (chunks_retrieved / 5) * 0.6 + (len(sources) / 3) * 0.4)

        self._write(INFO, "rag.retrieved", {
            "query":            query[:150],
            "chunks":           chunks_retrieved,
            "sources":          sources,
            "quality_score":    round(quality, 2),
            "duration_ms":      round(duration_ms, 2),
        })

        return quality

    # ── Confidence tracking ───────────────────

    def confidence_updated(self, loop: int, confidence: float, evidence_count: int):
        """Track how confidence evolves across loops — key for the UI chart."""
        point = {
            "loop":           loop,
            "confidence":     confidence,
            "evidence_count": evidence_count,
            "timestamp":      datetime.now().isoformat(),
        }
        self._confidence_history.append(point)
        self._write(DEBUG, "confidence.updated", point)

    # ── Error logging ─────────────────────────

    def error(self, context: str, error: str, data: dict = None):
        self._write(ERROR, "agent.error", {
            "context": context,
            "error":   str(error)[:300],
            "data":    data or {},
        })

    # ── Alert correlation logging ─────────────

    def correlation_completed(
        self,
        cluster_id: str,
        total_alerts: int,
        noise_reduced: int,
        dominant_symptom: str,
    ):
        self._write(INFO, "correlation.completed", {
            "cluster_id":       cluster_id,
            "total_alerts":     total_alerts,
            "noise_reduced":    noise_reduced,
            "dominant_symptom": dominant_symptom,
            "noise_reduction_pct": round(
                noise_reduced / total_alerts * 100 if total_alerts else 0, 1
            ),
        })

    # ── Historical log reader ─────────────────

    def read_recent_logs(self, n: int = 50) -> list[dict]:
        """
        Read the N most recent log entries.
        Used by the Streamlit observability panel.
        """
        if not self._log_path.exists():
            return []
        try:
            lines = self._log_path.read_text().strip().split("\n")
            lines = [l for l in lines if l.strip()]
            recent = lines[-n:]
            return [json.loads(l) for l in recent]
        except Exception:
            return []

    def get_confidence_history(self) -> list[dict]:
        """Return confidence evolution for the current investigation."""
        return self._confidence_history

    def get_log_path(self) -> Path:
        return self._log_path


# ─────────────────────────────────────────────
# GLOBAL LOGGER INSTANCE
# ─────────────────────────────────────────────

noc_logger = NOCLogger()