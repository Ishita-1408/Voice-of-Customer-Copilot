"""
Unit tests for the canonical VoC dataset structure and integrity.
"""

from pathlib import Path
import pandas as pd
import pytest

from app.data.build_canonical_dataset import CANONICAL_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VOC_FEEDBACK_PATH = PROCESSED_DIR / "voc_feedback.csv"


@pytest.fixture(scope="module")
def voc_df():
    assert VOC_FEEDBACK_PATH.exists(), f"Missing {VOC_FEEDBACK_PATH}"
    return pd.read_csv(VOC_FEEDBACK_PATH)


def test_canonical_dataset_schema_and_counts(voc_df):
    """Verify exact 800 rows, 17 columns, and correct partition counts."""
    # 1. Dimensions & Schema
    assert len(voc_df) == 800
    assert list(voc_df.columns) == CANONICAL_COLUMNS

    # 2. Public vs Synthetic counts
    assert (voc_df["data_type"] == "public").sum() == 350
    assert (voc_df["data_type"] == "synthetic").sum() == 450

    # 3. Unique IDs
    assert voc_df["feedback_id"].is_unique
    assert len(set(voc_df["feedback_id"])) == 800


def test_source_and_theme_metadata_integrity(voc_df):
    """Verify source distribution and theme ground-truth labels."""
    # Source counts
    src_map = voc_df["source_type"].value_counts().to_dict()
    assert src_map["Public Product Review"] == 350
    assert src_map["Support Ticket"] >= 50
    assert src_map["Interview"] >= 30
    assert src_map["Survey"] >= 30

    # Public theme rules
    pub_records = voc_df[voc_df["data_type"] == "public"]
    assert (pub_records["theme"] == "Unknown").all()
    assert (pub_records["theme_origin"] == "not_available").all()

    # Synthetic theme rules
    syn_records = voc_df[voc_df["data_type"] == "synthetic"]
    assert (syn_records["theme_origin"] == "synthetic_ground_truth").all()
    assert (syn_records["theme"] != "Unknown").all()
