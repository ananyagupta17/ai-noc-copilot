"""
agent/evidence.py

Weighted evidence scoring for RCA confidence.

The problem with pure LLM-generated confidence:
  The LLM will say "I'm 85% confident" based on vibes.
  That number means nothing — it's not computed from anything real.

What we do instead:
  Every piece of evidence has a type and a weight.
  Confidence = f(weights of all collected evidence).
  The score reflects how much real signal backs the diagnosis.

This makes the RCA auditable:
  "Confidence is 84% because we have 3 log signals (0.30 each),
   2 historical matches (0.25 each), and 1 runbook match (0.20)"
  — that's a defensible, explainable number.
"""

from dataclasses import dataclass, field
from typing import Optional
from agent.state import EvidenceItem, AgentState


# ─────────────────────────────────────────────
# EVIDENCE WEIGHTS
# How much each evidence type contributes to confidence.
# These are tuned for NOC investigations specifically.
# ─────────────────────────────────────────────

EVIDENCE_WEIGHTS = {
    # Logs are the strongest signal — error codes don't lie
    "log":                  0.30,

    # Historical incidents show this has happened before
    # and was resolved the same way — strong precedent
    "historical_incident":  0.25,

    # Runbook match means we have an SOP for this
    # grounds the answer in real procedure
    "runbook":              0.20,

    # Alerts confirm the monitoring system saw it too
    "alert":                0.15,

    # Topology confirms blast radius structurally
    "topology":             0.10,

    # Metrics corroborate device stress
    "metric":               0.10,

    # Unknown type — minimal weight
    "unknown":              0.05,
}

# How much a single piece of evidence of each type
# can contribute — prevents one strong source from
# dominating the score when many weak ones exist
MAX_WEIGHT_PER_TYPE = {
    "log":                  0.55,   # up to ~2 strong log signals
    "historical_incident":  0.45,   # up to ~2 historical matches
    "runbook":              0.20,   # one runbook is enough
    "alert":                0.30,   # up to 2 alert confirmations
    "topology":             0.10,   # one topology check
    "metric":               0.15,   # one metric check
    "unknown":              0.10,
}


# ─────────────────────────────────────────────
# EVIDENCE BREAKDOWN
# Human-readable explanation of the confidence score.
# Shown in the UI evidence trail panel.
# ─────────────────────────────────────────────

@dataclass
class EvidenceBreakdown:
    total_evidence_items: int
    confidence_score: float
    score_interpretation: str           # LOW / MEDIUM / HIGH / VERY HIGH
    contributing_types: dict            # type → capped weight contributed
    strongest_evidence: list[str]       # top 3 evidence item descriptions
    missing_evidence: list[str]         # what would strengthen the score
    explanation: str                    # one paragraph for the UI


# ─────────────────────────────────────────────
# SCORE EVIDENCE
# Main function — takes the agent state after investigation
# and returns a computed confidence score + breakdown.
# ─────────────────────────────────────────────

def score_evidence(state: AgentState) -> tuple[float, EvidenceBreakdown]:
    """
    Compute a confidence score from all evidence in agent state.

    Returns:
        (confidence_score, breakdown)
        confidence_score: float 0.0 to 1.0
        breakdown: EvidenceBreakdown with full explanation
    """
    evidence = state.evidence

    if not evidence:
        return 0.0, _empty_breakdown()

    # Step 1 — group evidence by type and sum weights per type
    type_weights: dict[str, float] = {}
    for item in evidence:
        t = item.type
        w = EVIDENCE_WEIGHTS.get(t, EVIDENCE_WEIGHTS["unknown"])
        type_weights[t] = type_weights.get(t, 0.0) + w

    # Step 2 — cap each type at its maximum contribution
    # Prevents log flooding from inflating the score
    capped_weights: dict[str, float] = {}
    for t, w in type_weights.items():
        cap = MAX_WEIGHT_PER_TYPE.get(t, 0.10)
        capped_weights[t] = min(w, cap)

    # Step 3 — sum capped weights and normalise to 0-1
    # We normalise against a "perfect investigation" score of ~1.30
    # (all evidence types present at their caps)
    perfect_score = sum(MAX_WEIGHT_PER_TYPE.values())  # ~1.85
    raw_score = sum(capped_weights.values())
    confidence = min(round(raw_score / perfect_score, 2), 1.0)

    # Step 4 — build breakdown
    breakdown = _build_breakdown(
        evidence=evidence,
        confidence=confidence,
        capped_weights=capped_weights,
    )

    return confidence, breakdown


def _build_breakdown(
    evidence: list[EvidenceItem],
    confidence: float,
    capped_weights: dict[str, float],
) -> EvidenceBreakdown:
    """Build the human-readable evidence breakdown."""

    # Interpretation
    if confidence >= 0.80:
        interpretation = "VERY HIGH"
    elif confidence >= 0.60:
        interpretation = "HIGH"
    elif confidence >= 0.40:
        interpretation = "MEDIUM"
    else:
        interpretation = "LOW"

    # Strongest evidence — top 3 by weight
    sorted_evidence = sorted(evidence, key=lambda e: e.weight, reverse=True)
    strongest = [
        f"[{e.type.upper()}] {e.source} — {e.content[:80]}..."
        for e in sorted_evidence[:3]
    ]

    # Missing evidence — what types are absent
    present_types = set(e.type for e in evidence)
    all_types = set(EVIDENCE_WEIGHTS.keys()) - {"unknown"}
    missing_types = all_types - present_types
    missing = _missing_evidence_hints(missing_types)

    # Explanation paragraph
    type_summary = ", ".join(
        f"{count} {t} signal(s)"
        for t, count in _count_by_type(evidence).items()
    )
    explanation = (
        f"Confidence score of {confidence:.0%} is based on {len(evidence)} "
        f"evidence items: {type_summary}. "
        f"The score is {interpretation.lower()} — "
        f"{_interpretation_detail(interpretation, missing)}"
    )

    return EvidenceBreakdown(
        total_evidence_items=len(evidence),
        confidence_score=confidence,
        score_interpretation=interpretation,
        contributing_types=capped_weights,
        strongest_evidence=strongest,
        missing_evidence=missing,
        explanation=explanation,
    )


def _count_by_type(evidence: list[EvidenceItem]) -> dict[str, int]:
    counts = {}
    for e in evidence:
        counts[e.type] = counts.get(e.type, 0) + 1
    return counts


def _missing_evidence_hints(missing_types: set[str]) -> list[str]:
    hints = {
        "log":               "No log signals — retrieve syslog for affected device",
        "historical_incident": "No historical match — check similar past incidents",
        "runbook":           "No runbook retrieved — get SOP for this symptom",
        "alert":             "No alert data — pull monitoring alerts for the device",
        "topology":          "No topology check — run blast radius analysis",
        "metric":            "No metrics — check device CPU/memory/packet-loss",
    }
    return [hints[t] for t in missing_types if t in hints]


def _interpretation_detail(interpretation: str, missing: list[str]) -> str:
    if interpretation == "VERY HIGH":
        return "multiple independent signal types corroborate the diagnosis."
    elif interpretation == "HIGH":
        return "strong evidence present across most signal types."
    elif interpretation == "MEDIUM":
        return (
            f"moderate evidence collected. "
            f"Consider gathering: {missing[0] if missing else 'additional signals'}."
        )
    else:
        return (
            f"limited evidence. Investigation incomplete. "
            f"Missing: {', '.join(missing[:2]) if missing else 'more data'}."
        )


def _empty_breakdown() -> EvidenceBreakdown:
    return EvidenceBreakdown(
        total_evidence_items=0,
        confidence_score=0.0,
        score_interpretation="LOW",
        contributing_types={},
        strongest_evidence=[],
        missing_evidence=list(_missing_evidence_hints(
            set(EVIDENCE_WEIGHTS.keys()) - {"unknown"}
        )),
        explanation="No evidence collected. Investigation did not run or all tools failed.",
    )


# ─────────────────────────────────────────────
# ENRICH STATE
# Called by output_node in graph.py.
# Updates state.rca with the computed score and breakdown.
# ─────────────────────────────────────────────

def enrich_rca_with_evidence(state: AgentState) -> AgentState:
    """
    Compute evidence score and inject into state.rca.
    Call this from output_node before LLM synthesis.
    """
    confidence, breakdown = score_evidence(state)

    state.rca.confidence_score = confidence
    state.rca.evidence = state.evidence

    # Store breakdown as a dict in the RCA for the UI to display
    state.rca.correlated_alert_cluster = state.rca.correlated_alert_cluster or {}
    state.rca.correlated_alert_cluster["evidence_breakdown"] = {
        "score": confidence,
        "interpretation": breakdown.score_interpretation,
        "contributing_types": breakdown.contributing_types,
        "strongest_evidence": breakdown.strongest_evidence,
        "missing_evidence": breakdown.missing_evidence,
        "explanation": breakdown.explanation,
    }

    return state


# ─────────────────────────────────────────────
# STANDALONE USAGE
# For testing evidence scoring independently
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from agent.state import AgentState, EvidenceItem

    # Simulate a state with some evidence
    state = AgentState(
        incident_description="Test incident — Singapore packet loss"
    )
    state.evidence = [
        EvidenceItem(type="log",                source="INC-001.log",          content="BGP session dropped", weight=0.30),
        EvidenceItem(type="log",                source="INC-001.log",          content="CRC errors spiking",  weight=0.30),
        EvidenceItem(type="historical_incident",source="find_similar()",       content="Same region, same symptom resolved via reroute", weight=0.25),
        EvidenceItem(type="runbook",            source="packet_loss_runbook",  content="Step 2: check CRC errors", weight=0.20),
        EvidenceItem(type="alert",              source="get_critical_alerts()",content="CRITICAL: packet loss 35%", weight=0.15),
        EvidenceItem(type="topology",           source="get_blast_radius()",   content="3 devices in blast radius", weight=0.10),
    ]

    confidence, breakdown = score_evidence(state)

    print(f"\nConfidence score : {confidence:.0%}")
    print(f"Interpretation   : {breakdown.score_interpretation}")
    print(f"\nContributing types:")
    for t, w in breakdown.contributing_types.items():
        print(f"  {t:25s}: {w:.2f}")
    print(f"\nStrongest evidence:")
    for e in breakdown.strongest_evidence:
        print(f"  {e}")
    print(f"\nMissing evidence:")
    for m in breakdown.missing_evidence:
        print(f"  - {m}")
    print(f"\nExplanation:\n  {breakdown.explanation}")