"""
alert_correlation/clusterer.py

Groups similar alerts into clusters using cosine similarity.
One cluster = one real incident, even if 20 alerts fired for it.

Why DBSCAN over K-Means:
  - K-Means needs you to specify the number of clusters upfront
    which we don't know for incoming alerts
  - DBSCAN finds clusters automatically based on density
  - DBSCAN handles noise — alerts that don't belong to any cluster
    get labelled as noise and filtered out
  - Perfect for NOC use case where cluster count is unknown
"""

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from alert_correlation.embedder import AlertEmbedder, load_alerts, load_alerts_for_incident


# ─────────────────────────────────────────────
# CLUSTER RESULT
# The output of the correlation engine.
# This is what gets injected into the agent state.
# ─────────────────────────────────────────────

@dataclass
class AlertCluster:
    cluster_id: str
    alerts: list[dict]          # all alerts in this cluster
    root_alerts: list[dict]     # the most representative alerts
    noise_alerts: list[dict]    # alerts DBSCAN labelled as noise
    dominant_symptom: str
    dominant_severity: str
    affected_devices: list[str]
    affected_regions: list[str]
    total_alert_count: int
    noise_reduced: int          # how many noise alerts were filtered
    time_span_minutes: float    # how long the alert storm lasted
    summary: str                # human-readable one-liner

    def to_dict(self) -> dict:
        return {
            "cluster_id":        self.cluster_id,
            "dominant_symptom":  self.dominant_symptom,
            "dominant_severity": self.dominant_severity,
            "affected_devices":  self.affected_devices,
            "affected_regions":  self.affected_regions,
            "total_alert_count": self.total_alert_count,
            "noise_reduced":     self.noise_reduced,
            "time_span_minutes": self.time_span_minutes,
            "root_alerts":       self.root_alerts[:3],   # top 3 for agent context
            "summary":           self.summary,
        }


# ─────────────────────────────────────────────
# ALERT CLUSTERER
# ─────────────────────────────────────────────

class AlertClusterer:

    def __init__(self, similarity_threshold: float = 0.6):
        """
        similarity_threshold: alerts with cosine similarity above this
        value get grouped together. 0.6 is a good starting point —
        lower = more aggressive grouping, higher = stricter separation.
        """
        # DBSCAN params:
        # eps = 1 - similarity_threshold (DBSCAN uses distance not similarity)
        # min_samples = 1 means even single alerts form a cluster
        self.eps = 1.0 - similarity_threshold
        self.min_samples = 1
        self.embedder = AlertEmbedder()

    def cluster(self, alerts: list[dict]) -> list[AlertCluster]:
        """
        Cluster a list of alerts.
        Returns a list of AlertCluster objects, one per group found.
        """
        if not alerts:
            return []

        if len(alerts) == 1:
            return [self._single_cluster(alerts)]

        # Step 1 — embed all alerts into vectors
        vectors = self.embedder.embed(alerts)

        # Step 2 — compute pairwise cosine similarity matrix
        # similarity[i][j] = how similar alert i and alert j are
        similarity_matrix = cosine_similarity(vectors)

        # Step 3 — convert similarity to distance for DBSCAN
        # Clip to [0,1] to avoid tiny negative floats from floating point math
        distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 2.0)
        np.fill_diagonal(distance_matrix, 0)

        # Step 4 — run DBSCAN
        db = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed"    # we're passing a precomputed distance matrix
        )
        labels = db.fit_predict(distance_matrix)

        # Step 5 — group alerts by cluster label
        # Label -1 means DBSCAN classified this alert as noise
        clusters = {}
        noise = []

        for i, label in enumerate(labels):
            if label == -1:
                noise.append(alerts[i])
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(alerts[i])

        # Step 6 — build AlertCluster objects
        result = []
        for label, cluster_alerts in clusters.items():
            cluster = self._build_cluster(
                cluster_id=f"CLU-{label:03d}",
                alerts=cluster_alerts,
                noise=noise,
                vectors=vectors,
                labels=labels,
                label=label,
            )
            result.append(cluster)

        # Sort by size descending — biggest cluster first
        result.sort(key=lambda c: c.total_alert_count, reverse=True)
        return result

    def correlate_incident(self, incident_id: str) -> AlertCluster | None:
        """
        Correlate all alerts for a specific incident ID.
        Convenience method called by the agent before investigation.
        """
        alerts = load_alerts_for_incident(incident_id)
        if not alerts:
            # Fall back to recent alerts if no incident-specific alerts
            alerts = load_alerts(time_window_minutes=60)
        if not alerts:
            return None
        clusters = self.cluster(alerts)
        return clusters[0] if clusters else None

    def correlate_from_description(self, description: str,
                                   region: str = None) -> AlertCluster | None:
        """
        Given a natural language incident description,
        load recent alerts and cluster them.
        Optionally filter by region for more focused results.
        """
        alerts = load_alerts(time_window_minutes=120)
        if region:
            alerts = [a for a in alerts if a.get("region", "").lower() == region.lower()]
        if not alerts:
            return None
        clusters = self.cluster(alerts)
        return clusters[0] if clusters else None

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _build_cluster(self, cluster_id, alerts, noise,
                       vectors, labels, label) -> AlertCluster:
        """Build an AlertCluster from a group of alerts."""

        # Find the most representative alerts (closest to cluster centroid)
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        cluster_vectors = vectors[cluster_indices]
        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        similarities_to_centroid = cosine_similarity(cluster_vectors, centroid).flatten()
        sorted_idx = np.argsort(similarities_to_centroid)[::-1]
        # alerts here is already the cluster subset — index directly
        root_alerts = [alerts[i] for i in sorted_idx[:min(3, len(alerts))]]

        # Dominant symptom — most common alert_type in cluster
        symptom_counts = {}
        for a in alerts:
            t = a.get("alert_type", "unknown")
            symptom_counts[t] = symptom_counts.get(t, 0) + 1
        dominant_symptom = max(symptom_counts, key=symptom_counts.get)

        # Dominant severity
        severity_order = {"CRITICAL": 3, "MAJOR": 2, "MINOR": 1}
        dominant_severity = max(
            (a.get("severity", "MINOR") for a in alerts),
            key=lambda s: severity_order.get(s, 0)
        )

        # Affected devices and regions
        devices = list({a.get("device") for a in alerts if a.get("device")})
        regions = list({a.get("region") for a in alerts if a.get("region")})

        # Time span
        time_span = self._compute_time_span(alerts)

        # Noise reduced = alerts that DBSCAN labelled as noise
        noise_reduced = len(noise)

        summary = (
            f"{len(alerts)} alerts correlated — "
            f"dominant: {dominant_symptom} ({dominant_severity}) "
            f"across {len(devices)} device(s) in {', '.join(regions) or 'unknown region'}. "
            f"{noise_reduced} noise alerts filtered."
        )

        return AlertCluster(
            cluster_id=cluster_id,
            alerts=alerts,
            root_alerts=root_alerts,
            noise_alerts=noise,
            dominant_symptom=dominant_symptom,
            dominant_severity=dominant_severity,
            affected_devices=devices,
            affected_regions=regions,
            total_alert_count=len(alerts),
            noise_reduced=noise_reduced,
            time_span_minutes=time_span,
            summary=summary,
        )

    def _single_cluster(self, alerts: list[dict]) -> AlertCluster:
        """Handle the edge case of a single alert."""
        a = alerts[0]
        return AlertCluster(
            cluster_id="CLU-000",
            alerts=alerts,
            root_alerts=alerts,
            noise_alerts=[],
            dominant_symptom=a.get("alert_type", "unknown"),
            dominant_severity=a.get("severity", "MINOR"),
            affected_devices=[a.get("device")] if a.get("device") else [],
            affected_regions=[a.get("region")] if a.get("region") else [],
            total_alert_count=1,
            noise_reduced=0,
            time_span_minutes=0.0,
            summary=f"Single alert: {a.get('message', 'No message')}",
        )

    def _compute_time_span(self, alerts: list[dict]) -> float:
        """Compute how many minutes the alert storm lasted."""
        timestamps = []
        for a in alerts:
            ts = a.get("timestamp", "")
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except ValueError:
                    pass
        if len(timestamps) < 2:
            return 0.0
        span = max(timestamps) - min(timestamps)
        return round(span.total_seconds() / 60, 1)


# ─────────────────────────────────────────────
# PUBLIC INTERFACE
# Called by api/main.py and agent/graph.py
# ─────────────────────────────────────────────

# Single shared instance
_clusterer = AlertClusterer(similarity_threshold=0.6)


def correlate_alerts(incident_id: str = None,
                     description: str = None,
                     region: str = None) -> dict | None:
    """
    Main entry point for alert correlation.

    Pass either:
      incident_id  → correlates alerts for that specific incident
      description  → correlates recent alerts, optionally filtered by region

    Returns a dict (cluster.to_dict()) ready to inject into AgentState,
    or None if no alerts found.
    """
    if incident_id:
        cluster = _clusterer.correlate_incident(incident_id)
    elif description:
        cluster = _clusterer.correlate_from_description(description, region=region)
    else:
        return None

    return cluster.to_dict() if cluster else None