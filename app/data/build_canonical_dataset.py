"""Canonical VoC dataset builder.

Merges the 350 validated public reviews and 150 controlled synthetic records into
a single canonical 500-record dataset (data/processed/voc_feedback.csv) with ground-truth
evaluation theme metadata.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from app.data.generate_synthetic_feedback import assemble_all_records

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "feedback_id",
    "source_type",
    "source_name",
    "feedback_text",
    "created_at",
    "customer_segment",
    "product_area",
    "sentiment",
    "severity",
    "language",
    "data_type",
    "metadata_origin",
    "rating",
    "product_name",
    "product_price",
    "theme",
    "theme_origin",
]


def compute_file_hash(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file to verify read-only integrity."""
    p = Path(filepath)
    if not p.exists():
        return ""
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_synthetic_theme_map() -> Dict[str, str]:
    """Retrieves ground-truth theme mapping for synthetic feedback IDs."""
    records = assemble_all_records()
    return {r["feedback_id"]: r["theme"] for r in records}


def build_canonical_dataset(
    public_path: Union[str, Path] = "data/processed/voc_public_reviews.csv",
    synthetic_path: Union[str, Path] = "data/processed/voc_synthetic_feedback.csv",
    output_path: Union[str, Path] = "data/processed/voc_feedback.csv"
) -> pd.DataFrame:
    """Merges public reviews and synthetic feedback into the canonical 500-row dataset."""
    pub_p = Path(public_path)
    syn_p = Path(synthetic_path)
    out_p = Path(output_path)

    if not pub_p.exists():
        raise FileNotFoundError(f"Public review dataset not found at: {pub_p}")
    if not syn_p.exists():
        raise FileNotFoundError(f"Synthetic feedback dataset not found at: {syn_p}")

    # Track hashes before processing to ensure read-only preservation
    pub_hash_before = compute_file_hash(pub_p)
    syn_hash_before = compute_file_hash(syn_p)

    # 1. Load source datasets
    df_pub = pd.read_csv(pub_p)
    df_syn = pd.read_csv(syn_p)

    assert len(df_pub) == 350, f"Expected 350 public records, got {len(df_pub)}"
    assert len(df_syn) == 150, f"Expected 150 synthetic records, got {len(df_syn)}"

    # 2. Add ground-truth evaluation metadata to Public records
    df_pub_augmented = df_pub.copy()
    df_pub_augmented["theme"] = "Unknown"
    df_pub_augmented["theme_origin"] = "not_available"

    # 3. Add ground-truth evaluation metadata to Synthetic records
    theme_map = get_synthetic_theme_map()
    df_syn_augmented = df_syn.copy()
    df_syn_augmented["theme"] = df_syn_augmented["feedback_id"].map(theme_map)
    df_syn_augmented["theme_origin"] = "synthetic_ground_truth"

    assert df_syn_augmented["theme"].notnull().all(), "Some synthetic records lack ground-truth theme mappings"

    # 4. Vertical Merge
    df_canonical = pd.concat([df_pub_augmented, df_syn_augmented], ignore_index=True)
    df_canonical = df_canonical[CANONICAL_COLUMNS].copy()

    # 5. Validation Assertions
    # 5.1 Row Count & Data types
    assert len(df_canonical) == 500, f"Expected exactly 500 rows, got {len(df_canonical)}"
    assert (df_canonical["data_type"] == "public").sum() == 350, "Expected exactly 350 public records"
    assert (df_canonical["data_type"] == "synthetic").sum() == 150, "Expected exactly 150 synthetic records"

    # 5.2 Unique IDs
    assert df_canonical["feedback_id"].is_unique, "Duplicate feedback_id values detected in canonical dataset"
    assert len(set(df_canonical["feedback_id"])) == 500

    # 5.3 Source Type Breakdown
    src_counts = df_canonical["source_type"].value_counts()
    assert src_counts.get("Public Product Review", 0) == 350, "Expected 350 Public Product Reviews"
    assert src_counts.get("Support Ticket", 0) == 75, "Expected 75 Support Tickets"
    assert src_counts.get("Interview", 0) == 40, "Expected 40 Interviews"
    assert src_counts.get("Survey", 0) == 35, "Expected 35 Surveys"

    # 5.4 Theme Metadata Integrity
    pub_mask = df_canonical["data_type"] == "public"
    syn_mask = df_canonical["data_type"] == "synthetic"

    assert (df_canonical.loc[pub_mask, "theme"] == "Unknown").all(), "All public records must have theme='Unknown'"
    assert (df_canonical.loc[pub_mask, "theme_origin"] == "not_available").all(), "Public records must have theme_origin='not_available'"
    assert (df_canonical.loc[syn_mask, "theme_origin"] == "synthetic_ground_truth").all(), "Synthetic records must have theme_origin='synthetic_ground_truth'"
    assert (df_canonical.loc[syn_mask, "theme"] != "Unknown").all(), "Synthetic records must have identified themes"

    # 5.5 Critical Field Non-Null Assertions
    assert df_canonical["feedback_text"].notnull().all(), "feedback_text contains null values"
    assert (df_canonical["feedback_text"].str.strip() != "").all(), "feedback_text contains empty strings"
    assert df_canonical["sentiment"].isin(["positive", "negative", "neutral"]).all()
    assert df_canonical["severity"].isin([1, 2, 3, 4, 5]).all()

    # 5.6 Date Range Assertion for Synthetic Records
    syn_dates = df_canonical.loc[syn_mask, "created_at"]
    assert syn_dates.min() >= "2026-06-01", f"Synthetic date {syn_dates.min()} before 2026-06-01"
    assert syn_dates.max() <= "2026-09-10", f"Synthetic date {syn_dates.max()} after 2026-09-10"
    assert df_canonical.loc[pub_mask, "created_at"].isnull().all(), "Public records must have null created_at"

    # 6. Save Canonical Dataset
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_canonical.to_csv(out_p, index=False)

    # Verify source datasets remained untouched
    pub_hash_after = compute_file_hash(pub_p)
    syn_hash_after = compute_file_hash(syn_p)
    assert pub_hash_before == pub_hash_after, "Error: voc_public_reviews.csv was modified during processing"
    assert syn_hash_before == syn_hash_after, "Error: voc_synthetic_feedback.csv was modified during processing"

    # 7. Print Final Report
    print("=" * 60)
    print("CANONICAL VoC DATASET")
    print("=" * 60)
    print(f"Total records:                {len(df_canonical):,}")
    print(f"Public records:               {(df_canonical['data_type'] == 'public').sum():,}")
    print(f"Synthetic records:            {(df_canonical['data_type'] == 'synthetic').sum():,}")

    print("\nSource distribution:")
    for stype, cnt in df_canonical["source_type"].value_counts().items():
        print(f"  - {stype:26s}: {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nData type distribution:")
    for dtype, cnt in df_canonical["data_type"].value_counts().items():
        print(f"  - {dtype:26s}: {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nTheme distribution:")
    print("  [Public Unlabeled]")
    print(f"  - {'Unknown (Public)':26s}: {(df_canonical['theme'] == 'Unknown').sum():3d} ({(df_canonical['theme'] == 'Unknown').sum()/len(df_canonical)*100:.1f}%)")
    print("  [Synthetic Ground Truth]")
    for thm, cnt in df_canonical.loc[syn_mask, "theme"].value_counts().items():
        print(f"  - {thm:42s}: {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nSentiment distribution:")
    for sent, cnt in df_canonical["sentiment"].value_counts().items():
        print(f"  - {sent:26s}: {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nSeverity distribution:")
    for sev, cnt in df_canonical["severity"].value_counts().sort_index().items():
        print(f"  - Level {sev:1d}:                   {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nCustomer segment distribution:")
    for seg, cnt in df_canonical["customer_segment"].value_counts().items():
        print(f"  - {seg:26s}: {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nProduct area distribution:")
    for pa, cnt in df_canonical["product_area"].value_counts().items():
        print(f"  - {pa:26s}: {cnt:3d} ({cnt/len(df_canonical)*100:.1f}%)")

    print("\nMissing values:")
    for col, cnt in df_canonical.isnull().sum().items():
        if cnt > 0:
            print(f"  - {col:26s}: {cnt:3d} (Expected for {col})")
    if (df_canonical.isnull().sum() == 0).all():
        print("  - None")

    dup_ids = df_canonical["feedback_id"].duplicated().sum()
    print(f"\nDuplicate feedback IDs:       {dup_ids}")
    print(f"Date range:                   {syn_dates.min()} to {syn_dates.max()} (Synthetic records; Public dates null)")
    print(f"\nOutput:\n{out_p}")
    print("=" * 60)

    return df_canonical


if __name__ == "__main__":
    build_canonical_dataset()
