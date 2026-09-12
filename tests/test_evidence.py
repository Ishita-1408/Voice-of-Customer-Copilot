"""
Unit tests for evidence retrieval and bundles.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.ai.evidence import (
    calculate_contradiction,
    calculate_evidence_confidence,
    calculate_evidence_strength,
    generate_limitations,
    retrieve_cluster_evidence,
    validate_evidence_inputs,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
THEME_DEFS_PATH = PROCESSED_DIR / "theme_definitions.json"
BUNDLES_JSON_PATH = PROCESSED_DIR / "evidence_bundles.json"
BUNDLES_CSV_PATH = PROCESSED_DIR / "evidence_bundles.csv"
TRACEABILITY_JSON_PATH = EVAL_DIR / "evidence_traceability.json"


@pytest.fixture(scope="module")
def voc_df():
    assert VOC_PATH.exists()
    return pd.read_csv(VOC_PATH)


@pytest.fixture(scope="module")
def clusters_df():
    assert CLUSTERS_PATH.exists()
    return pd.read_csv(CLUSTERS_PATH)


@pytest.fixture(scope="module")
def theme_defs():
    assert THEME_DEFS_PATH.exists()
    with open(THEME_DEFS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bundles_json():
    assert BUNDLES_JSON_PATH.exists()
    with open(BUNDLES_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def bundles_csv():
    assert BUNDLES_CSV_PATH.exists()
    return pd.read_csv(BUNDLES_CSV_PATH)


@pytest.fixture(scope="module")
def traceability_json():
    assert TRACEABILITY_JSON_PATH.exists()
    with open(TRACEABILITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_bundles_count_and_cluster_ids(bundles_json, bundles_csv, clusters_df):
    """Verify exactly 8 bundles matching clusters 0..7."""
    assert len(bundles_json) == 8, f"Expected 8 bundles, got {len(bundles_json)}"
    unique_clusters = sorted(clusters_df["cluster_id"].unique())
    assert [b["cluster_id"] for b in bundles_json] == unique_clusters
    assert sorted(bundles_csv["cluster_id"].unique().tolist()) == unique_clusters


def test_bundle_evidence_structure(bundles_json):
    """Verify each bundle contains valid evidence items."""
    for b in bundles_json:
        assert b["evidence_count"] > 0
        assert b["selected_evidence_count"] >= min(5, b["evidence_count"])
        assert len(b["evidence_items"]) == b["selected_evidence_count"]
        assert 0.0 <= b["evidence_strength"] <= 1.0
        assert 0.0 <= b["confidence"] <= 1.0
        assert isinstance(b["contradiction_flag"], bool)
        assert isinstance(b["contradiction_summary"], str)
        assert isinstance(b["limitations"], str)

        for item in b["evidence_items"]:
            assert "feedback_id" in item
            assert "similarity_score" in item
            assert -1.0 <= item["similarity_score"] <= 1.0


def test_evidence_strength_formula():
    """Verify deterministic evidence strength calculation."""
    score = calculate_evidence_strength(
        evidence_count=100,
        mean_similarity=0.8,
        source_diversity_count=4,
    )
    assert 0.0 <= score <= 1.0


def test_contradiction_detection():
    """Verify sentiment contradiction logic."""
    sentiments_mixed = ["positive", "positive", "negative", "negative", "neutral"]
    _, contra, summary = calculate_contradiction(sentiments_mixed)
    assert contra is True
    assert "Mixed sentiment" in summary

    sentiments_neg = ["negative", "negative", "negative", "negative", "neutral"]
    _, contra_neg, _ = calculate_contradiction(sentiments_neg)
    assert contra_neg is False

