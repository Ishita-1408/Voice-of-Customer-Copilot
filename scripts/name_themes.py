"""
Theme Naming Module & Execution Script

Generates human-readable names, definitions, and problem summaries for the 8 clusters
using Google Gemini LLM (gemini-3.6-flash).

CRITICAL DATA LEAKAGE RULE:
Gemini receives ONLY:
- representative feedback_text values (selected deterministically by cosine proximity to the cluster centroid)
No ground truth or metadata fields are ever passed to the LLM.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.theme_naming import (
    DEFAULT_NAMING_MODEL,
    DEFAULT_REPRESENTATIVES_PER_CLUSTER,
    get_gemini_client,
    name_cluster_with_gemini,
    select_representative_feedback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# File Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"

VOC_PATH = PROCESSED_DIR / "voc_feedback.csv"
EMBEDDINGS_PATH = PROCESSED_DIR / "feedback_embeddings.npy"
CLUSTERS_PATH = PROCESSED_DIR / "theme_clusters.csv"
METADATA_PATH = PROCESSED_DIR / "clustering_metadata.json"

OUTPUT_THEME_DEFS_JSON = PROCESSED_DIR / "theme_definitions.json"
OUTPUT_THEME_DEFS_CSV = PROCESSED_DIR / "theme_definitions.csv"
OUTPUT_TRACEABILITY_JSON = EVAL_DIR / "theme_naming_traceability.json"


def generate_theme_definitions(
    feedback_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    embeddings: np.ndarray,
    client: Optional[Any] = None,
    model: str = DEFAULT_NAMING_MODEL,
    n_representatives: int = DEFAULT_REPRESENTATIVES_PER_CLUSTER,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute semantic theme naming for all discovered clusters.
    """
    merged = feedback_df[["feedback_id", "feedback_text"]].merge(
        clusters_df[["feedback_id", "cluster_id"]], on="feedback_id", how="inner"
    )
    if len(merged) != len(feedback_df):
        raise ValueError(
            f"Row count mismatch after merging: feedback has {len(feedback_df)}, clusters has {len(clusters_df)}"
        )

    unique_clusters = sorted(clusters_df["cluster_id"].unique())
    theme_records = []
    json_definitions = []
    traceability_records = []

    for cluster_id in unique_clusters:
        cluster_mask = (merged["cluster_id"] == cluster_id).to_numpy()
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            logger.warning(f"Cluster {cluster_id} is empty.")
            continue

        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Representative feedback selection
        rep_indices_rel, rep_sims = select_representative_feedback(
            cluster_embeddings=cluster_embeddings,
            centroid=centroid,
            n_representatives=min(n_representatives, len(cluster_indices)),
        )

        rep_global_indices = cluster_indices[rep_indices_rel]
        rep_rows = merged.iloc[rep_global_indices]
        rep_ids = rep_rows["feedback_id"].tolist()
        rep_texts = rep_rows["feedback_text"].tolist()

        logger.info(
            f"Naming Cluster {cluster_id} (Size: {len(cluster_indices)}, Reps: {len(rep_texts)})..."
        )

        # Generate theme name with Gemini LLM
        naming_result, used_fallback = name_cluster_with_gemini(
            representative_texts=rep_texts,
            client=client,
            model=model,
        )

        theme_name = naming_result["theme_name"]
        theme_desc = naming_result["theme_description"]
        prob_summary = naming_result["problem_summary"]
        confidence = float(naming_result.get("confidence", 0.85))

        # Build CSV Record
        theme_records.append({
            "cluster_id": cluster_id,
            "theme_name": theme_name,
            "theme_description": theme_desc,
            "problem_summary": prob_summary,
            "confidence": confidence,
            "cluster_size": len(cluster_indices),
            "representative_feedback_ids": ";".join(rep_ids),
        })

        # Build JSON Record
        json_definitions.append({
            "cluster_id": int(cluster_id),
            "theme_name": theme_name,
            "theme_definition": theme_desc,
            "theme_description": theme_desc,
            "customer_problem": prob_summary,
            "problem_summary": prob_summary,
            "confidence": confidence,
            "cluster_size": len(cluster_indices),
            "representative_feedback_ids": rep_ids,
            "representative_feedback_texts": rep_texts,
        })

        # Build Traceability record
        traceability_records.append({
            "cluster_id": int(cluster_id),
            "cluster_size": len(cluster_indices),
            "theme_name": theme_name,
            "theme_definition": theme_desc,
            "customer_problem": prob_summary,
            "confidence": confidence,
            "representative_feedback": [
                {"feedback_id": fid, "feedback_text": txt}
                for fid, txt in zip(rep_ids, rep_texts)
            ],
        })

    theme_df = pd.DataFrame(theme_records)
    traceability_data = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "num_clusters": len(unique_clusters),
        "total_feedback_records": len(merged),
        "clusters": traceability_records,
    }

    return theme_df, json_definitions, traceability_data


def main():
    print("=" * 70)
    print("THEME NAMING EXECUTION")
    print("=" * 70)

    # Load input artifacts
    df_voc = pd.read_csv(VOC_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    df_clusters = pd.read_csv(CLUSTERS_PATH)

    theme_df, json_defs, trace_data = generate_theme_definitions(
        feedback_df=df_voc,
        clusters_df=df_clusters,
        embeddings=embeddings,
    )

    # Save outputs
    OUTPUT_THEME_DEFS_CSV.parent.mkdir(parents=True, exist_ok=True)
    theme_df.to_csv(OUTPUT_THEME_DEFS_CSV, index=False)
    print(f"Saved theme definitions CSV to {OUTPUT_THEME_DEFS_CSV}")

    OUTPUT_THEME_DEFS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_THEME_DEFS_JSON, "w", encoding="utf-8") as f:
        json.dump(json_defs, f, indent=2)
    print(f"Saved theme definitions JSON to {OUTPUT_THEME_DEFS_JSON}")

    OUTPUT_TRACEABILITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRACEABILITY_JSON, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
    print(f"Saved traceability JSON to {OUTPUT_TRACEABILITY_JSON}")


if __name__ == "__main__":
    main()
