"""
Unit tests for theme naming and definitions.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from app.ai.theme_naming import (
    FALLBACK_THEME_DEF,
    build_theme_naming_prompt,
    parse_and_validate_theme_json,
    select_representative_feedback,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

THEME_DEFS_CSV_PATH = PROCESSED_DIR / "theme_definitions.csv"
THEME_DEFS_JSON_PATH = PROCESSED_DIR / "theme_definitions.json"
TRACEABILITY_JSON_PATH = EVAL_DIR / "theme_naming_traceability.json"
VOC_FEEDBACK_PATH = PROCESSED_DIR / "voc_feedback.csv"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"


@pytest.fixture(scope="module")
def voc_df():
    assert VOC_FEEDBACK_PATH.exists(), f"Missing {VOC_FEEDBACK_PATH}"
    return pd.read_csv(VOC_FEEDBACK_PATH)


@pytest.fixture(scope="module")
def clusters_df():
    assert CLUSTERS_PATH.exists(), f"Missing {CLUSTERS_PATH}"
    return pd.read_csv(CLUSTERS_PATH)


@pytest.fixture(scope="module")
def theme_defs_df():
    assert THEME_DEFS_CSV_PATH.exists(), f"Missing {THEME_DEFS_CSV_PATH}"
    return pd.read_csv(THEME_DEFS_CSV_PATH)


@pytest.fixture(scope="module")
def theme_defs_json():
    assert THEME_DEFS_JSON_PATH.exists(), f"Missing {THEME_DEFS_JSON_PATH}"
    with open(THEME_DEFS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def traceability_json():
    assert TRACEABILITY_JSON_PATH.exists(), f"Missing {TRACEABILITY_JSON_PATH}"
    with open(TRACEABILITY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_theme_definitions_count_and_ids(theme_defs_df, theme_defs_json, clusters_df):
    """Verify exactly 8 cluster definitions matching clusters 0..7."""
    assert len(theme_defs_df) == 8, f"Expected 8 theme definitions in CSV, got {len(theme_defs_df)}"
    assert len(theme_defs_json) == 8, f"Expected 8 theme definitions in JSON, got {len(theme_defs_json)}"
    
    unique_clusters = sorted(clusters_df["cluster_id"].unique())
    assert sorted(theme_defs_df["cluster_id"].tolist()) == unique_clusters
    assert [item["cluster_id"] for item in theme_defs_json] == unique_clusters


def test_theme_definitions_schema(theme_defs_df, theme_defs_json):
    """Verify required keys and non-empty values."""
    required_keys = {
        "cluster_id",
        "theme_name",
        "theme_description",
        "problem_summary",
        "confidence",
        "cluster_size",
        "representative_feedback_ids",
    }
    for item in theme_defs_json:
        assert required_keys.issubset(set(item.keys())), f"Missing keys in {item}"
        assert isinstance(item["theme_name"], str) and len(item["theme_name"].strip()) > 0
        assert isinstance(item["problem_summary"], str) and len(item["problem_summary"].strip()) > 0
        assert 0.0 <= item["confidence"] <= 1.0

    csv_cols = set(theme_defs_df.columns)
    assert {"cluster_id", "theme_name", "theme_description", "problem_summary"}.issubset(csv_cols)


def test_prompt_builder_structure():
    """Verify prompt builder formats representative feedback without leaking ground-truth themes."""
    sample_texts = [
        "Payment failed during checkout.",
        "Card was charged twice.",
    ]
    prompt = build_theme_naming_prompt(representative_texts=sample_texts)
    assert "Payment failed during checkout." in prompt
    assert "Card was charged twice." in prompt
    assert "theme_origin" not in prompt


def test_parse_and_validate_theme_json():
    """Verify robust parsing of JSON response."""
    raw_response = '''```json
    {
        "theme_name": "Checkout Payment Failures",
        "theme_description": "Recurring transaction timeouts at payment gateway.",
        "problem_summary": "Users experience timeout errors when submitting payments.",
        "confidence": 0.95
    }
    ```'''
    res = parse_and_validate_theme_json(raw_response)
    assert res["theme_name"] == "Checkout Payment Failures"
    assert res["confidence"] == 0.95
