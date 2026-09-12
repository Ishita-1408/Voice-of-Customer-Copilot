"""
Validation module for the Enhanced Dataset (Step 12F).

Verifies schema, integrity, record counts, metadata origins, and data distributions
for data/processed/voc_feedback_enhanced.csv.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CANONICAL_PATH = PROJECT_ROOT / "data" / "processed" / "voc_feedback.csv"
ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "voc_feedback_enhanced.csv"

VALID_SYNTHETIC_THEMES = {
    "Payment & Checkout Reliability",
    "Delivery-Date Uncertainty",
    "Onboarding Confusion",
    "Search Relevance",
    "Wishlist / AI Assistant Feature Requests",
    "Returns / Refund Friction",
    "Product Quality Issues",
    "Positive Checkout/Shopping Experience",
}

VALID_SENTIMENTS = {"positive", "neutral", "negative"}
VALID_SEVERITIES = {1, 2, 3, 4, 5}
VALID_CUSTOMER_SEGMENTS = {
    "New User - Mobile",
    "New User - Desktop",
    "Returning User - Mobile",
    "Returning User - Desktop",
    "Power User - Mobile",
    "Power User - Desktop",
    "Unknown",
}
VALID_SOURCE_TYPES = {
    "Public Product Review",
    "Support Ticket",
    "Interview",
    "Survey",
    "App Review",
}


def validate_enhanced_dataset(
    enhanced_csv_path: Path = ENHANCED_PATH,
    canonical_csv_path: Path = CANONICAL_PATH,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Perform exhaustive data integrity and schema validation on the enhanced dataset.

    Returns:
        is_valid: bool indicating if all checks passed
        metadata: dictionary with distribution stats
        errors: list of error descriptions if any
    """
    errors: List[str] = []

    if not enhanced_csv_path.exists():
        return False, {}, [f"Enhanced dataset file not found: {enhanced_csv_path}"]

    enhanced_df = pd.read_csv(enhanced_csv_path)

    # 1. Row count check
    total_records = len(enhanced_df)
    if total_records != 800:
        errors.append(f"Expected exactly 800 records, found {total_records}")

    # 2. Canonical preservation check
    if canonical_csv_path.exists():
        canonical_df = pd.read_csv(canonical_csv_path)
        if len(canonical_df) != 500:
            errors.append(f"Canonical dataset expected 500 records, found {len(canonical_df)}")

        # Check first 500 rows match canonical
        first_500 = enhanced_df.iloc[:500].copy()
        if not first_500["feedback_id"].equals(canonical_df["feedback_id"]):
            errors.append("First 500 feedback IDs do not match canonical dataset in order")
        if not first_500["feedback_text"].equals(canonical_df["feedback_text"]):
            errors.append("First 500 feedback texts do not match canonical dataset")
    else:
        errors.append(f"Canonical dataset file not found: {canonical_csv_path}")

    # 3. Columns / Schema check
    expected_cols = [
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
    if list(enhanced_df.columns) != expected_cols:
        errors.append(f"Schema mismatch. Expected {expected_cols}, got {list(enhanced_df.columns)}")

    # 4. Feedback ID uniqueness
    if enhanced_df["feedback_id"].nunique() != total_records:
        dups = enhanced_df["feedback_id"][enhanced_df["feedback_id"].duplicated()].tolist()
        errors.append(f"Duplicate feedback IDs found: {dups[:10]}")

    # 5. Missing / Empty feedback text
    null_texts = enhanced_df["feedback_text"].isna().sum()
    empty_texts = (enhanced_df["feedback_text"].str.strip() == "").sum()
    if null_texts > 0 or empty_texts > 0:
        errors.append(f"Missing or empty feedback_text found: {null_texts} nulls, {empty_texts} empties")

    # 6. Duplicate feedback text among NEW records (rows 500..800)
    new_records = enhanced_df.iloc[500:].copy() if total_records >= 500 else pd.DataFrame()
    if len(new_records) > 0:
        new_text_dups = new_records["feedback_text"][new_records["feedback_text"].duplicated()].tolist()
        if new_text_dups:
            errors.append(f"Duplicate feedback text found among new records: {new_text_dups[:5]}")

    # 7. Data type split
    type_counts = enhanced_df["data_type"].value_counts().to_dict()
    public_count = type_counts.get("public", 0)
    synthetic_count = type_counts.get("synthetic", 0)
    if public_count != 350:
        errors.append(f"Expected exactly 350 public records, found {public_count}")
    if synthetic_count != 450:
        errors.append(f"Expected exactly 450 synthetic records, found {synthetic_count}")

    # 8. Sentiment and Severity validation
    invalid_sentiments = set(enhanced_df["sentiment"].dropna()) - VALID_SENTIMENTS
    if invalid_sentiments:
        errors.append(f"Invalid sentiment values: {invalid_sentiments}")

    invalid_severities = set(enhanced_df["severity"].dropna()) - VALID_SEVERITIES
    if invalid_severities:
        errors.append(f"Invalid severity values: {invalid_severities}")

    # 9. Date validity for new records
    if len(new_records) > 0:
        for idx, date_str in enumerate(new_records["created_at"]):
            if pd.isna(date_str) or not str(date_str).strip():
                errors.append(f"New synthetic record at row {500 + idx} has missing created_at")
                break
            try:
                dt = datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
                if not (datetime(2026, 6, 5) <= dt <= datetime(2026, 9, 10)):
                    errors.append(f"Date out of range for new synthetic record ({date_str}) at row {500 + idx}")
                    break
            except ValueError:
                errors.append(f"Invalid date format ({date_str}) at row {500 + idx}")
                break

    # 10. Synthetic Theme validation
    synth_df = enhanced_df[enhanced_df["theme_origin"] == "synthetic_ground_truth"]
    synth_themes = set(synth_df["theme"].dropna())
    unexpected_themes = synth_themes - VALID_SYNTHETIC_THEMES
    if unexpected_themes:
        errors.append(f"Unexpected synthetic theme labels found: {unexpected_themes}")

    missing_themes = VALID_SYNTHETIC_THEMES - synth_themes
    if missing_themes:
        errors.append(f"Missing ground truth synthetic themes: {missing_themes}")

    # Check that public records have theme_origin == "not_available"
    pub_df = enhanced_df[enhanced_df["data_type"] == "public"]
    if not (pub_df["theme_origin"] == "not_available").all():
        errors.append("Some public records have non-not_available theme_origin")

    # Construct metadata summary
    metadata = {
        "total_records": int(total_records),
        "public_records": int(public_count),
        "synthetic_records": int(synthetic_count),
        "new_synthetic_records": int(len(new_records)),
        "theme_distribution": {str(k): int(v) for k, v in synth_df["theme"].value_counts().items()},
        "source_distribution": {str(k): int(v) for k, v in enhanced_df["source_type"].value_counts().items()},
        "segment_distribution": {str(k): int(v) for k, v in enhanced_df["customer_segment"].value_counts().items()},
        "sentiment_distribution": {str(k): int(v) for k, v in enhanced_df["sentiment"].value_counts().items()},
        "severity_distribution": {str(k): int(v) for k, v in enhanced_df["severity"].value_counts().items()},
        "date_range": {
            "min_created_at": str(enhanced_df["created_at"].dropna().min()),
            "max_created_at": str(enhanced_df["created_at"].dropna().max()),
        },
        "validation_status": "PASSED" if not errors else "FAILED",
        "validation_errors": errors,
        "notes": (
            "The synthetic ground-truth 'theme' and 'theme_origin' fields are strictly "
            "reserved for offline diagnostic evaluation and must NOT be used as input features "
            "or model selection signals for any AI or clustering pipeline."
        ),
    }

    is_valid = len(errors) == 0
    return is_valid, metadata, errors


if __name__ == "__main__":
    valid, meta, errs = validate_enhanced_dataset()
    print("Validation status:", "PASSED" if valid else "FAILED")
    if errs:
        print("Errors:")
        for e in errs:
            print(f"  - {e}")
    else:
        print("Metadata summary:")
        print(json.dumps(meta, indent=2))
