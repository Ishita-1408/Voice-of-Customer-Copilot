"""
Script to generate and export Product Insights.

Outputs:
1. data/processed/product_insights.json
2. data/processed/product_insights.csv
3. data/evaluation/insight_traceability.json
"""

import json
import logging
from pathlib import Path
import pandas as pd
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.insights import generate_all_product_insights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

PRIORITY_PATH = PROCESSED_DIR / "priority_assessments.json"
BUNDLES_PATH = PROCESSED_DIR / "evidence_bundles.json"

OUTPUT_INSIGHTS_JSON = PROCESSED_DIR / "product_insights.json"
OUTPUT_INSIGHTS_CSV = PROCESSED_DIR / "product_insights.csv"
OUTPUT_TRACEABILITY_JSON = EVAL_DIR / "insight_traceability.json"


def main():
    print("=" * 70)
    print("GENERATING PRODUCT INSIGHTS & RECOMMENDATIONS")
    print("=" * 70)

    # 1. Load inputs
    with open(PRIORITY_PATH, "r", encoding="utf-8") as f:
        priority_assessments = json.load(f)
    with open(BUNDLES_PATH, "r", encoding="utf-8") as f:
        evidence_bundles = json.load(f)

    print(f"Loaded {len(priority_assessments)} priority assessments and {len(evidence_bundles)} evidence bundles.")

    # 2. Generate insights
    insights_json, insights_df, trace_data, stats = generate_all_product_insights(
        priority_assessments=priority_assessments,
        evidence_bundles=evidence_bundles,
    )

    print(f"\nGeneration Stats: {stats}")

    # 3. Save artifacts
    OUTPUT_INSIGHTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_INSIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(insights_json, f, indent=2)

    insights_df.to_csv(OUTPUT_INSIGHTS_CSV, index=False)

    with open(OUTPUT_TRACEABILITY_JSON, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    print("\nSaved product insights:")
    print(f"  - {OUTPUT_INSIGHTS_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_INSIGHTS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_TRACEABILITY_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
