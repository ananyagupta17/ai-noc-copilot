"""
alert_correlation/embedder.py

Converts alert messages into numerical vectors.
These vectors are what the clusterer uses to find similar alerts.

We use TF-IDF here — no API key needed, works offline,
and for short alert messages it performs as well as neural embeddings.

TF-IDF in plain english:
  - Words that appear often in ONE alert but rarely across ALL alerts
    get a high score — they're distinctive
  - Common words like "on", "the", "device" get low scores
  - Result: each alert becomes a vector where important words have
    high values and noise words have low values
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import ALERTS_DIR


# ─────────────────────────────────────────────
# ALERT EMBEDDER
# ─────────────────────────────────────────────

class AlertEmbedder:

    def __init__(self):
        # TF-IDF vectorizer — learns vocabulary from the alert corpus
        # ngram_range=(1,2) means it considers single words AND pairs
        # e.g. "packet loss" is treated as one feature, not two
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,       # top 500 most informative terms
            stop_words="english",   # ignore common words
            sublinear_tf=True,      # dampen very high term frequencies
        )
        self.fitted = False

    def fit(self, alerts: list[dict]) -> "AlertEmbedder":
        """
        Learn vocabulary from a list of alert dicts.
        Must be called before embed().
        """
        texts = [self._alert_to_text(a) for a in alerts]
        self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def embed(self, alerts: list[dict]) -> np.ndarray:
        """
        Convert a list of alerts into a matrix of vectors.
        Shape: (n_alerts, n_features)
        Each row is one alert's vector.
        """
        if not self.fitted:
            # Auto-fit on the same data if not fitted yet
            self.fit(alerts)

        texts = [self._alert_to_text(a) for a in alerts]
        matrix = self.vectorizer.transform(texts).toarray()

        # L2 normalise — makes cosine similarity = dot product
        # which is faster and numerically stable
        return normalize(matrix, norm="l2")

    def embed_single(self, alert: dict) -> np.ndarray:
        """Embed a single alert. Requires fit() to have been called."""
        if not self.fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")
        text = self._alert_to_text(alert)
        vec = self.vectorizer.transform([text]).toarray()
        return normalize(vec, norm="l2")[0]

    def _alert_to_text(self, alert: dict) -> str:
        """
        Flatten an alert dict into a single string for vectorisation.
        We concatenate the most informative fields with weights —
        repeating important fields makes them score higher in TF-IDF.
        """
        parts = []

        # Message is the most informative field — repeat it
        msg = alert.get("message", "")
        parts.extend([msg, msg])

        # Alert type and severity
        parts.append(alert.get("alert_type", ""))
        parts.append(alert.get("severity", ""))

        # Region and device — important for grouping geographically
        parts.append(alert.get("region", ""))
        parts.append(alert.get("device", ""))

        # Metric name gives strong signal
        metric = alert.get("metric_value", {})
        if isinstance(metric, dict):
            parts.append(metric.get("name", ""))

        return " ".join(p for p in parts if p).lower()


# ─────────────────────────────────────────────
# LOAD ALERTS HELPER
# ─────────────────────────────────────────────

def load_alerts(time_window_minutes: int = 60) -> list[dict]:
    """
    Load alerts from the JSON file.
    In production this would query a real monitoring system.
    For now, returns the most recent N minutes worth of alerts.
    """
    alerts_file = ALERTS_DIR / "alerts.json"
    if not alerts_file.exists():
        return []

    alerts = json.loads(alerts_file.read_text())

    # Sort by timestamp descending and take recent window
    alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)

    # For demo purposes return up to 50 recent alerts
    # In production: filter by actual timestamp delta
    return alerts[:50]


def load_alerts_for_incident(incident_id: str) -> list[dict]:
    """Load all alerts linked to a specific incident."""
    alerts_file = ALERTS_DIR / "alerts.json"
    if not alerts_file.exists():
        return []
    alerts = json.loads(alerts_file.read_text())
    return [a for a in alerts if a.get("incident_id") == incident_id]