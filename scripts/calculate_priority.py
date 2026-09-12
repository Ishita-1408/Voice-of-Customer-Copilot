"""
Script to calculate and export Prioritization Assessments.

Outputs:
1. data/processed/priority_assessments.json
2. data/processed/priority_assessments.csv
3. data/evaluation/prioritization_traceability.json
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

from app.ai.prioritization import assess_all_priorities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
BUNDLES_PATH = PROCESSED_DIR / "evidence_bundles.json"

OUTPUT_PRIORITY_JSON = PROCESSED_DIR / "priority_assessments.json"
OUTPUT_PRIORITY_CSV = PROCESSED_DIR / "priority_assessments.csv"
OUTPUT_TRACEABILITY_JSON = EVAL_DIR / "prioritization_traceability.json"


def main():
    print("=" * 70)
    print("CALCULATING PRIORITIZATION ASSESSMENTS")
    print("=" * 70)

    # 1. Load inputs
    df_voc = pd.read_csv(VOC_PATH)
    df_clusters = pd.read_csv(CLUSTERS_PATH)
    with open(BUNDLES_PATH, "r", encoding="utf-8") as f:
        bundles = json.load(f)

    print(f"Loaded {len(df_voc)} feedback records and {len(bundles)} evidence bundles.")

    # 2. Assess priorities
    json_assessments, assessments_df, trace_data = assess_all_priorities(
        canonical_df=df_voc,
        clusters_df=df_clusters,
        evidence_bundles=bundles,
    )

    # 3. Save artifacts
    OUTPUT_PRIORITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PRIORITY_JSON, "w", encoding="utf-8") as f:
        json.dump(json_assessments, f, indent=2)

    assessments_df.to_csv(OUTPUT_PRIORITY_CSV, index=False)

    with open(OUTPUT_TRACEABILITY_JSON, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    print("Saved priority assessments:")
    print(f"  - {OUTPUT_PRIORITY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_PRIORITY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_TRACEABILITY_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
