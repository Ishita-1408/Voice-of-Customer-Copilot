"""
Unit tests for deterministic prioritization and decision classification.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.ai.prioritization import (
    BUILD_MIN_EVIDENCE,
    BUILD_MIN_PRIORITY,
    BUILD_MIN_STRENGTH,
    calculate_evidence_diversity_score,
    calculate_frequency_score,
    calculate_priority_score,
    calculate_segment_impact,
    calculate_severity_metrics,
    classify_decision,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
BUNDLES_PATH = PROCESSED_DIR / "evidence_bundles.json"
PRIORITY_JSON_PATH = PROCESSED_DIR / "priority_assessments.json"
PRIORITY_CSV_PATH = PROCESSED_DIR / "priority_assessments.csv"
TRACEABILITY_JSON_PATH = EVAL_DIR / "prioritization_traceability.json"


@pytest.fixture(scope="module")
def voc_df():
    assert VOC_PATH.exists()
    return pd.read_csv(VOC_PATH)


@pytest.fixture(scope="module")
def clusters_df():
    assert CLUSTERS_PATH.exists()
    return pd.read_csv(CLUSTERS_PATH)


@pytest.fixture(scope="module")
def bundles():
    assert BUNDLES_PATH.exists()
    with open(BUNDLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def priority_json():
    assert PRIORITY_JSON_PATH.exists()
    with open(PRIORITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def priority_csv():
    assert PRIORITY_CSV_PATH.exists()
    return pd.read_csv(PRIORITY_CSV_PATH)


@pytest.fixture(scope="module")
def traceability_json():
    assert TRACEABILITY_JSON_PATH.exists()
    with open(TRACEABILITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_priority_assessments_count(priority_json, priority_csv, clusters_df):
    """Verify exactly 8 cluster assessments."""
    assert len(priority_json) == 8
    assert len(priority_csv) == 8
    unique_clusters = sorted(clusters_df["cluster_id"].unique())
    assert [p["cluster_id"] for p in priority_json] == unique_clusters


def test_priority_score_range_and_decisions(priority_json):
    """Verify score ranges [0, 1] and valid decisions."""
    valid_decisions = {"BUILD", "INVESTIGATE", "MONITOR", "DON'T BUILD YET", "DE-PRIORITIZE"}
    for p in priority_json:
        assert 0.0 <= p["priority_score"] <= 1.0
        assert p["decision"] in valid_decisions
        assert p["evidence_count"] > 0
        assert 0.0 <= p["evidence_strength"] <= 1.0
        assert 0.0 <= p["frequency_score"] <= 1.0


def test_deterministic_classification_logic():
    """Verify classification thresholds."""
    # High score and strength -> BUILD
    dec = classify_decision(
        priority_score=0.75,
        evidence_strength=0.7,
        evidence_count=50,
    )
    assert dec == "BUILD"

    # Low score -> DON'T BUILD YET
    dec_low = classify_decision(
        priority_score=0.03,
        evidence_strength=0.1,
        evidence_count=10,
    )
    assert dec_low == "DON'T BUILD YET"

