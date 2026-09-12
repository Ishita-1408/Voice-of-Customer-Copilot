"""
Semantic theme discovery and clustering module for VoC feedback embeddings.

This module clusters customer feedback embeddings using scikit-learn KMeans
on L2-normalized vectors (cosine distance geometry) to discover latent customer
topics without using any metadata or synthetic ground truth.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

DEFAULT_K_RANGE = range(6, 13)
DEFAULT_RANDOM_STATE = 42


def validate_embeddings(
    embeddings: np.ndarray,
    expected_rows: Optional[int] = None,
    expected_dim: Optional[int] = None,
) -> None:
    """Validate embedding matrix structure and numerical validity."""
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Embeddings must be a numpy.ndarray, got {type(embeddings)}")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D matrix of shape (n_samples, n_features), got ndim={embeddings.ndim}"
        )

    if embeddings.size == 0:
        raise ValueError("Embeddings matrix is empty.")

    if np.isnan(embeddings).any():
        raise ValueError("Embeddings matrix contains NaN values.")

    if np.isinf(embeddings).any():
        raise ValueError("Embeddings matrix contains Infinite values.")

    if expected_rows is not None and embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} embedding rows, but got {embeddings.shape[0]}"
        )

    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Expected embedding dimension {expected_dim}, but got {embeddings.shape[1]}"
        )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Apply L2 normalization to embedding vectors so Euclidean distance is
    monotonically equivalent to cosine distance.
    """
    validate_embeddings(embeddings)
    normalized = normalize(embeddings, norm="l2", axis=1)
    return normalized


def evaluate_k_range(
    normalized_embeddings: np.ndarray,
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[int, Dict[int, float], Dict[int, Dict[int, int]], Dict[int, np.ndarray]]:
    """
    Evaluate multiple K values for KMeans on normalized embeddings.

    Returns:
        best_k: K with the highest silhouette score.
        silhouette_scores: Mapping of K -> silhouette score.
        cluster_size_distributions: Mapping of K -> (cluster_id -> count).
        labels_by_k: Mapping of K -> cluster labels array.
    """
    validate_embeddings(normalized_embeddings)

    if not k_range:
        raise ValueError("k_range must contain at least one integer value.")

    n_samples = normalized_embeddings.shape[0]
    silhouette_scores: Dict[int, float] = {}
    cluster_size_distributions: Dict[int, Dict[int, int]] = {}
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in k_range:
        if k < 2:
            raise ValueError(f"K must be >= 2 for clustering, got {k}")
        if k >= n_samples:
            raise ValueError(
                f"K ({k}) must be strictly less than number of samples ({n_samples})"
            )

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(normalized_embeddings)
        score = float(silhouette_score(normalized_embeddings, labels, metric="cosine"))

        silhouette_scores[k] = round(score, 4)
        labels_by_k[k] = labels

        # Record cluster size counts
        unique, counts = np.unique(labels, return_counts=True)
        cluster_size_distributions[k] = {
            int(cid): int(cnt) for cid, cnt in zip(unique, counts)
        }

    # Best K is selected based on highest silhouette score
    best_k = max(silhouette_scores.keys(), key=lambda k: silhouette_scores[k])
    return best_k, silhouette_scores, cluster_size_distributions, labels_by_k


def discover_semantic_themes(
    embeddings: np.ndarray,
    feedback_ids: Sequence[str],
    k_range: Sequence[int] = DEFAULT_K_RANGE,
    random_state: int = DEFAULT_RANDOM_STATE,
    embedding_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end theme discovery on embedding vectors only.

    Args:
        embeddings: 2D numpy array of shape (N, D).
        feedback_ids: Sequence of N unique feedback identifier strings.
        k_range: Range of candidate K values to evaluate.
        random_state: Seed for deterministic KMeans execution.
        embedding_metadata: Optional dictionary with embedding provenance.

    Returns:
        clusters_df: DataFrame with ['feedback_id', 'cluster_id'] preserving canonical order.
        metadata: Comprehensive metadata dictionary regarding clustering evaluation.
    """
    validate_embeddings(embeddings)

    if len(feedback_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Row count mismatch: {len(feedback_ids)} feedback_ids vs {embeddings.shape[0]} embeddings"
        )

    if len(set(feedback_ids)) != len(feedback_ids):
        raise ValueError("feedback_ids must be strictly unique.")

    # 1. L2 Normalization
    normalized_embeddings = normalize_embeddings(embeddings)

    # 2. Evaluate K range
    best_k, scores, size_dists, labels_by_k = evaluate_k_range(
        normalized_embeddings=normalized_embeddings,
        k_range=k_range,
        random_state=random_state,
    )

    best_labels = labels_by_k[best_k]

    # 3. Create DataFrame preserving exact input ordering
    clusters_df = pd.DataFrame(
        {
            "feedback_id": list(feedback_ids),
            "cluster_id": best_labels.astype(int),
        }
    )

    # 4. Construct metadata
    meta = embedding_metadata or {}
    metadata: Dict[str, Any] = {
        "selected_k": int(best_k),
        "tested_k_values": [int(k) for k in k_range],
        "silhouette_scores": {str(k): float(v) for k, v in scores.items()},
        "cluster_sizes": {str(cid): int(cnt) for cid, cnt in size_dists[best_k].items()},
        "algorithm": "KMeans (L2-normalized cosine distance)",
        "random_state": int(random_state),
        "embedding_model": meta.get("model", "gemini-embedding-001"),
        "embedding_dimension": int(embeddings.shape[1]),
        "dataset_rows": int(embeddings.shape[0]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return clusters_df, metadata


def save_clustering_results(
    clusters_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_csv_path: Union[str, Path] = "data/processed/theme_clusters.csv",
    output_metadata_path: Union[str, Path] = "data/processed/clustering_metadata.json",
) -> None:
    """Save cluster assignments and metadata to disk."""
    csv_path = Path(output_csv_path)
    meta_path = Path(output_metadata_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    clusters_df.to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved clusters to %s", csv_path)
    logger.info("Saved clustering metadata to %s", meta_path)
