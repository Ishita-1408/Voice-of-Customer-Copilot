"""
Script to build and export Evidence Bundles.

Outputs:
1. data/processed/evidence_bundles.json
2. data/processed/evidence_bundles.csv
3. data/evaluation/evidence_traceability.json
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.evidence import build_all_evidence_bundles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "feedback_embeddings.npy"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
THEME_DEFS_PATH = PROCESSED_DIR / "theme_definitions.json"

OUTPUT_BUNDLES_JSON = PROCESSED_DIR / "evidence_bundles.json"
OUTPUT_BUNDLES_CSV = PROCESSED_DIR / "evidence_bundles.csv"
OUTPUT_TRACEABILITY_JSON = EVAL_DIR / "evidence_traceability.json"


def main():
    print("=" * 70)
    print("BUILDING EVIDENCE BUNDLES")
    print("=" * 70)

    # 1. Load inputs
    df_voc = pd.read_csv(VOC_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    df_clusters = pd.read_csv(CLUSTERS_PATH)
    with open(THEME_DEFS_PATH, "r", encoding="utf-8") as f:
        theme_defs = json.load(f)

    print(f"Loaded {len(df_voc)} feedback records, embeddings {embeddings.shape}, and {len(theme_defs)} theme definitions.")

    # 2. Build bundles
    bundles, evidence_df, trace_data = build_all_evidence_bundles(
        canonical_df=df_voc,
        clusters_df=df_clusters,
        embeddings=embeddings,
        theme_definitions=theme_defs,
    )

    # 3. Save artifacts
    with open(OUTPUT_BUNDLES_JSON, "w", encoding="utf-8") as f:
        json.dump(bundles, f, indent=2)

    evidence_df.to_csv(OUTPUT_BUNDLES_CSV, index=False)

    with open(OUTPUT_TRACEABILITY_JSON, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)

    print("Saved evidence bundles:")
    print(f"  - {OUTPUT_BUNDLES_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_BUNDLES_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_TRACEABILITY_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
