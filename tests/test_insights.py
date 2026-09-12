"""
Unit tests for grounded product insights.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from app.ai.insights import (
    build_fallback_insight,
    build_insight_prompt,
    parse_and_validate_insight_json,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
BUNDLES_PATH = PROCESSED_DIR / "evidence_bundles.json"
PRIORITY_PATH = PROCESSED_DIR / "priority_assessments.json"
INSIGHTS_JSON_PATH = PROCESSED_DIR / "product_insights.json"
INSIGHTS_CSV_PATH = PROCESSED_DIR / "product_insights.csv"
TRACEABILITY_JSON_PATH = EVAL_DIR / "insight_traceability.json"


@pytest.fixture(scope="module")
def voc_df():
    assert VOC_PATH.exists()
    return pd.read_csv(VOC_PATH)


@pytest.fixture(scope="module")
def bundles():
    assert BUNDLES_PATH.exists()
    with open(BUNDLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def priority():
    assert PRIORITY_PATH.exists()
    with open(PRIORITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def insights_json():
    assert INSIGHTS_JSON_PATH.exists()
    with open(INSIGHTS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def insights_csv():
    assert INSIGHTS_CSV_PATH.exists()
    return pd.read_csv(INSIGHTS_CSV_PATH)


@pytest.fixture(scope="module")
def traceability_json():
    assert TRACEABILITY_JSON_PATH.exists()
    with open(TRACEABILITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_insights_count_and_schema(insights_json, insights_csv):
    """Verify exactly 8 insights matching schema."""
    assert len(insights_json) == 8
    assert len(insights_csv) == 8

    required_fields = {
        "cluster_id",
        "theme_name",
        "insight",
        "customer_problem",
        "recommended_action",
        "decision",
        "priority_score",
        "evidence_count",
        "evidence_strength",
        "supporting_feedback_ids",
        "confidence",
        "limitations",
    }
    for item in insights_json:
        assert required_fields.issubset(set(item.keys()))
        assert isinstance(item["insight"], str) and len(item["insight"]) > 0
        assert isinstance(item["customer_problem"], str) and len(item["customer_problem"]) > 0
        assert isinstance(item["recommended_action"], str) and len(item["recommended_action"]) > 0
        assert isinstance(item["supporting_feedback_ids"], list)
        assert len(item["supporting_feedback_ids"]) > 0


def test_decision_and_priority_preservation(insights_json, priority):
    """Verify insight generator did NOT override deterministic decisions or priority scores."""
    priority_map = {p["cluster_id"]: p for p in priority}
    for item in insights_json:
        cid = item["cluster_id"]
        p = priority_map[cid]
        assert item["decision"] == p["decision"], f"Decision mismatch for cluster {cid}"
        assert item["priority_score"] == p["priority_score"], f"Priority score mismatch for cluster {cid}"


def test_evidence_id_traceability(insights_json, voc_df):
    """Verify all supporting feedback IDs exist in canonical dataset."""
    valid_ids = set(voc_df["feedback_id"])
    for item in insights_json:
        for fid in item["supporting_feedback_ids"]:
            assert fid in valid_ids, f"Invalid feedback ID {fid} cited in cluster {item['cluster_id']}"
