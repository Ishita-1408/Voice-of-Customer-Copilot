"""Public customer review preprocessing pipeline.

Loads, cleans, normalizes, and samples the public review dataset to generate
the canonical VoC public review schema without modifying the raw source dataset.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Schema constants
TARGET_COLUMNS = [
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
]

SOURCE_TYPE_DEFAULT = "Public Product Review"
SOURCE_NAME_DEFAULT = "Flipkart Public Review Dataset"
DATA_TYPE_DEFAULT = "public"
CUSTOMER_SEGMENT_DEFAULT = "Unknown"

METADATA_ORIGIN_SPEC = {
    "source_type": "public_source",
    "source_name": "public_source",
    "feedback_text": "public_source",
    "sentiment": "public_source",
    "rating": "public_source",
    "product_name": "public_source",
    "product_price": "public_source",
    "data_type": "public_source",
    "product_area": "derived",
    "severity": "derived",
    "language": "derived",
    "customer_segment": "unknown",
    "created_at": "unknown",
}


def clean_text(review: Any, summary: Any) -> str:
    """Combines and cleans Review (headline) and Summary (body) into feedback_text.

    Preserves customer wording without losing detail or hallucinating text.
    """
    rev = ""
    if pd.notnull(review):
        r_str = str(review).strip()
        if r_str and r_str.lower() != "nan":
            rev = r_str

    summ = ""
    if pd.notnull(summary):
        s_str = str(summary).strip()
        if s_str and s_str.lower() != "nan":
            summ = s_str

    if not rev and not summ:
        return ""
    if not rev:
        return summ
    if not summ:
        return rev

    # If both exist and one is a substring of the other (case-insensitive)
    if rev.lower() == summ.lower() or rev.lower() in summ.lower():
        return summ
    if summ.lower() in rev.lower():
        return rev

    # Merge headline and body cleanly
    if rev[-1] in ".!?":
        return f"{rev} {summ}"
    return f"{rev}. {summ}"


def normalize_sentiment(sentiment_raw: Any) -> str:
    """Normalizes sentiment label into standard classes: 'positive', 'negative', 'neutral'."""
    if not isinstance(sentiment_raw, str):
        return "neutral"
    s = sentiment_raw.strip().lower()
    if "pos" in s:
        return "positive"
    if "neg" in s:
        return "negative"
    if "neu" in s:
        return "neutral"
    return "neutral"


def parse_rating(rate_raw: Any) -> Optional[int]:
    """Parses rating into integer 1 to 5; returns None if unparseable."""
    if pd.isnull(rate_raw):
        return None
    try:
        val_str = str(rate_raw).strip()
        match = re.search(r"\b([1-5])\b", val_str)
        if match:
            return int(match.group(1))
        val_float = float(val_str)
        if 1.0 <= val_float <= 5.0:
            return int(round(val_float))
    except (ValueError, TypeError):
        pass
    return None


def extract_product_area(product_name: Any) -> str:
    """Deterministic keyword extraction mapping product_name to broad product areas.

    Categories:
    - computing
    - mobile
    - audio
    - personal_care
    - kitchen
    - home_appliances
    - accessories
    - electronics
    - other
    """
    if not isinstance(product_name, str):
        return "other"

    text = f" {product_name.lower()} "

    # Computing
    if any(k in text for k in [
        "laptop", "desktop", "monitor", "keyboard", "mouse", "printer", "router",
        "pen drive", "hard drive", "hard disk", "hdd", "ssd", "tablet", "ipad",
        "computer", "pendrive", "flash drive", "usb hub"
    ]):
        return "computing"

    # Mobile
    if any(k in text for k in [
        "smartphone", "mobile", "iphone", "galaxy phone", "redmi", "realme",
        "poco", "vivo", "oppo", "oneplus phone", "cellular"
    ]):
        return "mobile"

    # Audio
    if any(k in text for k in [
        "headset", "headphone", "earphone", "neckband", "earbuds", "earbud",
        "speaker", "soundbar", "home theatre", "sound blast", "subwoofer", "tws"
    ]) or re.search(r"\baudio\b", text):
        return "audio"

    # Personal care
    if any(k in text for k in [
        "trimmer", "shaver", "hair dryer", "straightener", "curler", "grooming",
        "epilator", "toothbrush", "clipper", "shaving"
    ]):
        return "personal_care"

    # Kitchen
    if any(k in text for k in [
        "dishwasher", "chimney", "kettle", "toaster", "induction", "mixer",
        "grinder", "blender", "juicer", "cookware", "gas stove", "cooktop",
        "water purifier", "air fryer", "microwave", "oven", "cooker",
        "flask", "chopper", "food processor", "bottle"
    ]) or re.search(r"\b(pan|pot|knife)\b", text):
        return "kitchen"

    # Home appliances (use word boundary for 'iron', 'ac', 'fan')
    if any(k in text for k in [
        "air cooler", "cooler", "ceiling fan", "exhaust fan", "table fan",
        "heater", "geyser", "water heater", "dry iron", "steam iron",
        "washing machine", "vacuum", "air conditioner", "refrigerator", "fridge"
    ]) or re.search(r"\b(iron|fan|ac)\b", text):
        return "home_appliances"

    # Accessories
    if any(k in text for k in [
        "wrist support", "support", "cricket", "bat", "ball", "net", "stumps",
        "case", "cover", "strap", "cable", "charger", "adapter", "power bank",
        "stand", "bag", "backpack", "pouch", "holder", "sleeve", "remote",
        "screen guard", "tempered glass"
    ]):
        return "accessories"

    # Electronics
    if any(k in text for k in [
        "camera", "dslr", "lens", "television", "smart tv", "projector",
        "inverter", "led display", "smartwatch", "smart watch", "drone"
    ]) or re.search(r"\btv\b", text):
        return "electronics"

    return "other"


def derive_severity(rating: Optional[int], sentiment: str) -> int:
    """Derives a demo severity signal (1-5) based on rating and sentiment.

    - negative 1-star -> 5
    - negative 2-star -> 4
    - negative 3-star -> 3
    - neutral 3-star  -> 3
    - positive 4-star -> 2
    - positive 5-star -> 1
    Handles missing/unusual combinations conservatively.
    """
    norm_sent = normalize_sentiment(sentiment)

    if norm_sent == "negative":
        if rating == 1:
            return 5
        if rating == 2:
            return 4
        if rating == 3:
            return 3
        if rating in (4, 5):
            return 3
        return 4  # Default for unrated negative

    if norm_sent == "neutral":
        if rating in (1, 2):
            return 4
        if rating == 3:
            return 3
        if rating in (4, 5):
            return 2
        return 3  # Default for unrated neutral

    # positive
    if rating == 5:
        return 1
    if rating == 4:
        return 2
    if rating == 3:
        return 2
    if rating in (1, 2):
        return 3
    return 2  # Default for unrated positive


def detect_language(text: str) -> str:
    """Deterministic check to verify if feedback text is English without external APIs."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    cleaned = text.strip()
    ascii_count = sum(1 for c in cleaned if c.isascii())
    has_letters = any(c.isalpha() for c in cleaned)
    if has_letters and (ascii_count / len(cleaned)) >= 0.8:
        return "en"
    return "unknown"


def generate_feedback_id(index: int) -> str:
    """Generate deterministic feedback ID like PUB-000001."""
    return f"PUB-{index:06d}"


def sample_stratified_sentiment(
    df: pd.DataFrame,
    sample_size: int = 350,
    random_state: int = 42
) -> pd.DataFrame:
    """Deterministically samples records stratified by sentiment to avoid overwhelming positive bias.

    Target allocation for 350 items:
    - negative: 140 (40%)
    - positive: 130 (37.1%)
    - neutral: 80 (22.9%)
    """
    if len(df) <= sample_size:
        return df.copy().reset_index(drop=True)

    neg_n = int(round(sample_size * 0.40))
    pos_n = int(round(sample_size * 0.371))
    neu_n = max(0, sample_size - neg_n - pos_n)

    target_alloc = {
        "negative": neg_n,
        "positive": pos_n,
        "neutral": neu_n,
    }

    subsets = []
    for sentiment, target_n in target_alloc.items():
        subset = df[df["sentiment"] == sentiment]
        if len(subset) == 0:
            continue
        n_to_take = min(target_n, len(subset))
        if n_to_take > 0:
            sampled = subset.sample(n=n_to_take, random_state=random_state)
            subsets.append(sampled)

    combined = pd.concat(subsets, ignore_index=True)

    # In case any category lacked records to hit sample_size, fill from remaining
    if len(combined) < sample_size:
        remaining_needed = sample_size - len(combined)
        remaining_indices = df.index.difference(combined.index)
        if len(remaining_indices) > 0 and remaining_needed > 0:
            extra_sample = df.loc[remaining_indices].sample(
                n=min(remaining_needed, len(remaining_indices)),
                random_state=random_state
            )
            combined = pd.concat([combined, extra_sample], ignore_index=True)

    # Deterministic shuffle
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return combined


def prepare_public_reviews(
    input_path: Union[str, Path] = "data/raw/public_reviews.csv",
    output_path: Union[str, Path] = "data/processed/voc_public_reviews.csv",
    sample_size: int = 350,
    random_state: int = 42
) -> pd.DataFrame:
    """Preprocesses raw public review dataset into canonical VoC schema.

    Returns the cleaned, sampled dataframe and saves it to output_path.
    """
    in_p = Path(input_path)
    out_p = Path(output_path)

    if not in_p.exists():
        raise FileNotFoundError(f"Source dataset not found at {in_p}")

    # 1. Load raw dataset (Read-only)
    df_raw = pd.read_csv(in_p)
    original_rows = len(df_raw)

    # 2. Deduplication
    df_dedup = df_raw.drop_duplicates(keep="first").copy()
    rows_after_dedup = len(df_dedup)
    dup_rows_removed = original_rows - rows_after_dedup

    # 3. Clean and construct feedback text
    df_dedup["feedback_text"] = [
        clean_text(r, s) for r, s in zip(df_dedup.get("Review"), df_dedup.get("Summary"))
    ]

    # Remove rows with no usable feedback text
    df_valid = df_dedup[df_dedup["feedback_text"].str.strip() != ""].copy()
    rows_unusable_removed = rows_after_dedup - len(df_valid)

    # 4. Standardize columns
    df_valid["sentiment"] = df_valid["Sentiment"].apply(normalize_sentiment)
    df_valid["rating"] = df_valid["Rate"].apply(parse_rating)
    df_valid["product_area"] = df_valid["product_name"].apply(extract_product_area)
    df_valid["severity"] = [
        derive_severity(r, s) for r, s in zip(df_valid["rating"], df_valid["sentiment"])
    ]
    df_valid["language"] = df_valid["feedback_text"].apply(detect_language)

    # 5. Fixed constants
    df_valid["source_type"] = SOURCE_TYPE_DEFAULT
    df_valid["source_name"] = SOURCE_NAME_DEFAULT
    df_valid["data_type"] = DATA_TYPE_DEFAULT
    df_valid["customer_segment"] = CUSTOMER_SEGMENT_DEFAULT
    df_valid["created_at"] = None
    
    metadata_origin_json = json.dumps(METADATA_ORIGIN_SPEC)
    df_valid["metadata_origin"] = metadata_origin_json

    # 6. Stratified Sampling
    df_sampled = sample_stratified_sentiment(
        df_valid,
        sample_size=sample_size,
        random_state=random_state
    )

    # 7. Generate deterministic IDs
    df_sampled["feedback_id"] = [generate_feedback_id(i + 1) for i in range(len(df_sampled))]

    # 8. Reorder and enforce schema
    df_final = df_sampled[TARGET_COLUMNS].copy()

    # 9. Assertions
    assert len(df_final) == sample_size, f"Expected {sample_size} rows, got {len(df_final)}"
    assert df_final["feedback_id"].is_unique, "feedback_id is not unique"
    assert df_final["feedback_text"].notnull().all(), "feedback_text contains null values"
    assert (df_final["feedback_text"].str.strip() != "").all(), "feedback_text contains empty strings"
    assert (df_final["data_type"] == DATA_TYPE_DEFAULT).all(), f"data_type must be '{DATA_TYPE_DEFAULT}'"
    assert df_final["source_type"].notnull().all(), "source_type contains null values"
    assert df_final["source_name"].notnull().all(), "source_name contains null values"

    # 10. Save to target location
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_p, index=False)

    # 11. Final Report
    print("=" * 65)
    print("PUBLIC REVIEWS PREPROCESSING REPORT")
    print("=" * 65)
    print(f"Original rows:                       {original_rows:,}")
    print(f"Duplicate rows removed:              {dup_rows_removed:,}")
    print(f"Rows after deduplication:            {rows_after_dedup:,}")
    print(f"Rows with unusable feedback removed: {rows_unusable_removed:,}")
    print(f"Final public-review rows:            {len(df_final):,}")
    print(f"Missing feedback_text:               {df_final['feedback_text'].isnull().sum()}")
    print(f"Missing customer_segment:            {df_final['customer_segment'].isnull().sum()} (Set to '{CUSTOMER_SEGMENT_DEFAULT}')")
    print(f"Missing created_at:                  {df_final['created_at'].isnull().sum()} (Set to null)")
    print("\nSentiment distribution:")
    for sent, cnt in df_final["sentiment"].value_counts().items():
        print(f"  - {sent:8s}: {cnt:4d} ({cnt/len(df_final)*100:.1f}%)")
    print("\nProduct-area distribution:")
    for area, cnt in df_final["product_area"].value_counts().items():
        print(f"  - {area:16s}: {cnt:4d} ({cnt/len(df_final)*100:.1f}%)")
    print("\nSeverity distribution:")
    for sev, cnt in df_final["severity"].value_counts().sort_index().items():
        print(f"  - Level {sev}: {cnt:4d} ({cnt/len(df_final)*100:.1f}%)")
    print("=" * 65)
    print(f"Output saved to: {out_p}")

    return df_final


if __name__ == "__main__":
    prepare_public_reviews()
