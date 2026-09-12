"""
Script to execute semantic theme discovery and clustering.

Outputs:
1. data/processed/theme_clusters.csv
2. data/processed/clustering_metadata.json
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

from app.ai.clustering import discover_semantic_themes, save_clustering_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "feedback_embeddings.npy"
EMBEDDING_META_PATH = PROCESSED_DIR / "embedding_metadata.json"

OUTPUT_CLUSTERS_CSV = PROCESSED_DIR / "theme_clusters.csv"
OUTPUT_CLUSTERING_META = PROCESSED_DIR / "clustering_metadata.json"


def main():
    print("=" * 70)
    print("RUNNING SEMANTIC THEME DISCOVERY & CLUSTERING")
    print("=" * 70)

    # 1. Load inputs
    df_voc = pd.read_csv(VOC_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    embedding_meta = {}
    if EMBEDDING_META_PATH.exists():
        with open(EMBEDDING_META_PATH, "r", encoding="utf-8") as f:
            embedding_meta = json.load(f)

    print(f"Loaded {len(df_voc)} feedback records with embedding shape {embeddings.shape}.")

    # 2. Discover semantic themes
    clusters_df, clustering_meta = discover_semantic_themes(
        embeddings=embeddings,
        feedback_ids=df_voc["feedback_id"].tolist(),
        k_range=[8],
        random_state=42,
        embedding_metadata=embedding_meta,
    )

    # 3. Save artifacts
    save_clustering_results(
        clusters_df=clusters_df,
        metadata=clustering_meta,
        output_csv_path=OUTPUT_CLUSTERS_CSV,
        output_json_path=OUTPUT_CLUSTERING_META,
    )

    print("\nSaved clustering results:")
    print(f"  - {OUTPUT_CLUSTERS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  - {OUTPUT_CLUSTERING_META.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
