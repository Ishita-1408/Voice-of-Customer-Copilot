"""
V4 Product Priority Scoring and Deterministic Decision Classification Module.

Converts discovered customer feedback themes and V4 evidence bundles into
explainable product priority scores and deterministic decision recommendations
using the PRD formula:
    Priority = Frequency × Severity × Segment Impact × Evidence Strength × Evidence Diversity
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

BUILD_MIN_PRIORITY = 0.25
BUILD_MIN_STRENGTH = 0.70
BUILD_MIN_EVIDENCE = 20

INVESTIGATE_MIN_PRIORITY = 0.12
INVESTIGATE_MIN_STRENGTH = 0.60

MONITOR_MIN_PRIORITY = 0.05


def calculate_frequency_score(evidence_count: int, normalizer: float = 100.0) -> float:
    """
    Calculate normalized frequency score based on total cluster evidence volume.
    Formula: min(evidence_count / 100.0, 1.0)
    """
    if evidence_count < 0:
        raise ValueError("evidence_count cannot be negative")
    return float(min(evidence_count / normalizer, 1.0))


def calculate_severity_metrics(severities: Sequence[Union[int, float]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for cluster severity (1-5 scale) and normalized score.
    Formula: severity_score = mean_severity / 5.0
    """
    if not severities:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0,
            "severity_5_count": 0,
            "severity_4_or_5_count": 0,
            "severity_score": 0.0,
        }

    arr = np.array(severities, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    max_val = int(np.max(arr))
    sev5_cnt = int(np.sum(arr == 5))
    sev45_cnt = int(np.sum(arr >= 4))
    sev_score = float(np.clip(mean_val / 5.0, 0.0, 1.0))

    return {
        "mean": mean_val,
        "median": median_val,
        "max": max_val,
        "severity_5_count": sev5_cnt,
        "severity_4_or_5_count": sev45_cnt,
        "severity_score": sev_score,
    }


def calculate_segment_impact(
    segments: Sequence[Optional[str]], total_records: int
) -> Dict[str, Any]:
    """
    Calculate segment impact metrics excluding 'Unknown' segments.

    Formula:
      known_segment_count    = distinct non-Unknown customer segments
      known_segment_records  = records with non-Unknown customer segments
      known_segment_fraction = known_segment_records / total_records
      segment_coverage_score = min(known_segment_count / 4.0, 1.0)
      segment_impact_score   = 0.5 * segment_coverage_score + 0.5 * known_segment_fraction
    """
    if total_records <= 0:
        return {
            "known_segment_count": 0,
            "known_segment_records": 0,
            "known_segment_fraction": 0.0,
            "segment_coverage_score": 0.0,
            "segment_impact_score": 0.0,
        }

    known_segments = [
        str(s).strip()
        for s in segments
        if s is not None and str(s).strip() != "" and str(s).strip().lower() != "unknown"
    ]

    known_count = len(set(known_segments))
    known_records = len(known_segments)
    known_fraction = float(known_records / total_records)

    coverage_score = float(min(known_count / 4.0, 1.0))
    impact_score = float(np.clip(0.5 * coverage_score + 0.5 * known_fraction, 0.0, 1.0))

    return {
        "known_segment_count": int(known_count),
        "known_segment_records": int(known_records),
        "known_segment_fraction": known_fraction,
        "segment_coverage_score": coverage_score,
        "segment_impact_score": impact_score,
    }


def calculate_evidence_diversity_score(source_diversity_count: int, max_sources: float = 4.0) -> float:
    """
    Calculate evidence diversity score normalized across 4 possible source types.
    Formula: min(source_diversity_count / 4.0, 1.0)
    """
    if source_diversity_count < 0:
        raise ValueError("source_diversity_count cannot be negative")
    return float(min(source_diversity_count / max_sources, 1.0))


def calculate_priority_score(
    frequency_score: float,
    severity_score: float,
    segment_impact_score: float,
    evidence_strength: float,
    evidence_diversity_score: float,
) -> float:
    """
    Multiplicative priority score calculation.
    Priority = Frequency × Severity × Segment Impact × Evidence Strength × Evidence Diversity
    """
    raw_score = (
        float(frequency_score)
        * float(severity_score)
        * float(segment_impact_score)
        * float(evidence_strength)
        * float(evidence_diversity_score)
    )
    return float(np.clip(raw_score, 0.0, 1.0))


def classify_decision(
    priority_score: float,
    evidence_strength: float,
    evidence_count: int,
) -> str:
    """
    Deterministic hierarchical decision classification.

    Hierarchy:
      1. BUILD:        priority >= 0.25 AND evidence_strength >= 0.70 AND evidence_count >= 20
      2. INVESTIGATE:  priority >= 0.12 AND evidence_strength >= 0.60
      3. MONITOR:      priority >= 0.05
      4. DON'T BUILD YET: priority < 0.05
    """
    if (
        priority_score >= BUILD_MIN_PRIORITY
        and evidence_strength >= BUILD_MIN_STRENGTH
        and evidence_count >= BUILD_MIN_EVIDENCE
    ):
        return "BUILD"
    elif (
        priority_score >= INVESTIGATE_MIN_PRIORITY
        and evidence_strength >= INVESTIGATE_MIN_STRENGTH
    ):
        return "INVESTIGATE"
    elif priority_score >= MONITOR_MIN_PRIORITY:
        return "MONITOR"
    else:
        return "DON'T BUILD YET"


def evaluate_segment_limitation(
    mean_severity: float,
    evidence_count: int,
    known_segment_fraction: float,
) -> str:
    """Evaluate and explain segment limitations deterministically."""
    if mean_severity >= 2.5 and evidence_count >= 40 and known_segment_fraction < 0.35:
        return (
            f"High severity/frequency signal (mean severity {mean_severity:.2f}, {evidence_count} records), "
            f"but segment context is limited because {100.0 * (1.0 - known_segment_fraction):.1f}% of records have Unknown customer_segment."
        )
    elif known_segment_fraction < 0.20:
        return (
            f"Limited segment granularity ({100.0 * known_segment_fraction:.1f}% known segment coverage) due to predominance of public review channels."
        )
    else:
        return "Robust segment context with high coverage across known customer segments."


def assess_theme_priority(
    bundle: Dict[str, Any],
    cluster_severities: Sequence[Union[int, float]],
    cluster_segments: Sequence[Optional[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assess priority metrics, score, decision, and limitation for a single cluster bundle."""
    cid = int(bundle["cluster_id"])
    tname = bundle["theme_name"]
    tdesc = bundle.get("theme_definition") or bundle.get("theme_description", "")
    ev_count = int(bundle["evidence_count"])
    ev_strength = float(bundle["evidence_strength"])
    src_count = int(bundle["source_diversity_count"])
    contra_flag = bool(bundle["contradiction_flag"])

    # 1. Frequency
    freq_score = calculate_frequency_score(ev_count)

    # 2. Severity
    sev_dict = calculate_severity_metrics(cluster_severities)

    # 3. Segment Impact
    seg_dict = calculate_segment_impact(cluster_segments, total_records=ev_count)

    # 4. Evidence Diversity
    div_score = calculate_evidence_diversity_score(src_count)

    # 5. Priority Score
    priority_score = calculate_priority_score(
        frequency_score=freq_score,
        severity_score=sev_dict["severity_score"],
        segment_impact_score=seg_dict["segment_impact_score"],
        evidence_strength=ev_strength,
        evidence_diversity_score=div_score,
    )

    # 6. Decision
    decision = classify_decision(
        priority_score=priority_score,
        evidence_strength=ev_strength,
        evidence_count=ev_count,
    )

    # 7. Limitation
    limitation = evaluate_segment_limitation(
        mean_severity=sev_dict["mean"],
        evidence_count=ev_count,
        known_segment_fraction=seg_dict["known_segment_fraction"],
    )

    # Structured JSON item
    json_item: Dict[str, Any] = {
        "cluster_id": cid,
        "theme_name": tname,
        "theme_definition": tdesc,
        "evidence_count": ev_count,
        "frequency_score": round(freq_score, 4),
        "mean_severity": round(sev_dict["mean"], 4),
        "median_severity": round(sev_dict["median"], 4),
        "max_severity": int(sev_dict["max"]),
        "severity_5_count": int(sev_dict["severity_5_count"]),
        "severity_4_or_5_count": int(sev_dict["severity_4_or_5_count"]),
        "severity_score": round(sev_dict["severity_score"], 4),
        "known_segment_count": int(seg_dict["known_segment_count"]),
        "known_segment_records": int(seg_dict["known_segment_records"]),
        "known_segment_fraction": round(seg_dict["known_segment_fraction"], 4),
        "segment_coverage_score": round(seg_dict["segment_coverage_score"], 4),
        "segment_impact_score": round(seg_dict["segment_impact_score"], 4),
        "evidence_strength": round(ev_strength, 4),
        "source_diversity_count": int(src_count),
        "evidence_diversity_score": round(div_score, 4),
        "priority_score": round(priority_score, 4),
        "decision": decision,
        "segment_limitation": limitation,
        "contradiction_flag": contra_flag,
    }

    # Flat CSV row
    csv_row: Dict[str, Any] = {
        "cluster_id": cid,
        "theme_name": tname,
        "evidence_count": ev_count,
        "frequency_score": round(freq_score, 4),
        "mean_severity": round(sev_dict["mean"], 4),
        "median_severity": round(sev_dict["median"], 4),
        "max_severity": int(sev_dict["max"]),
        "severity_5_count": int(sev_dict["severity_5_count"]),
        "severity_4_or_5_count": int(sev_dict["severity_4_or_5_count"]),
        "severity_score": round(sev_dict["severity_score"], 4),
        "known_segment_count": int(seg_dict["known_segment_count"]),
        "known_segment_records": int(seg_dict["known_segment_records"]),
        "known_segment_fraction": round(seg_dict["known_segment_fraction"], 4),
        "segment_coverage_score": round(seg_dict["segment_coverage_score"], 4),
        "segment_impact_score": round(seg_dict["segment_impact_score"], 4),
        "evidence_strength": round(ev_strength, 4),
        "source_diversity_count": int(src_count),
        "evidence_diversity_score": round(div_score, 4),
        "priority_score": round(priority_score, 4),
        "decision": decision,
        "segment_limitation": limitation,
    }

    return json_item, csv_row


def assess_all_priorities(
    canonical_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    evidence_bundles: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    """Execute priority assessment across all 8 discovered clusters."""
    if len(canonical_df) != 800 or len(clusters_df) != 800:
        raise ValueError("canonical_df and clusters_df must both have 800 rows")

    json_assessments: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    traceability_items: List[Dict[str, Any]] = []

    for bundle in evidence_bundles:
        cid = bundle["cluster_id"]
        mask = (clusters_df["cluster_id"] == cid).to_numpy()
        indices = np.where(mask)[0]

        severities = canonical_df.iloc[indices]["severity"].tolist()
        segments = canonical_df.iloc[indices]["customer_segment"].tolist()

        j_item, c_row = assess_theme_priority(
            bundle=bundle,
            cluster_severities=severities,
            cluster_segments=segments,
        )

        json_assessments.append(j_item)
        csv_rows.append(c_row)
        traceability_items.append(j_item)

    assessments_df = pd.DataFrame(csv_rows)
    traceability_data = {
        "pipeline_version": "1.0",
        "num_clusters_assessed": len(json_assessments),
        "total_records": len(canonical_df),
        "assessments": traceability_items,
    }

    return json_assessments, assessments_df, traceability_data


