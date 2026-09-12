"""
Unit tests for semantic theme discovery and clustering module.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from app.ai.clustering import (
    DEFAULT_K_RANGE,
    DEFAULT_RANDOM_STATE,
    discover_semantic_themes,
    evaluate_k_range,
    normalize_embeddings,
    save_clustering_results,
    validate_embeddings,
)


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Generate deterministic synthetic embedding matrix for testing."""
    np.random.seed(42)
    # 50 records, 16 dimensions
    return np.random.randn(50, 16).astype(np.float32)


@pytest.fixture
def sample_feedback_ids() -> list[str]:
    """Generate 50 sample feedback IDs."""
    return [f"fb_{i:04d}" for i in range(50)]


def test_validate_embeddings_success(sample_embeddings: np.ndarray):
    """Test valid embedding validation."""
    validate_embeddings(sample_embeddings, expected_rows=50, expected_dim=16)


def test_validate_embeddings_non_array():
    """Test non-numpy input raises TypeError."""
    with pytest.raises(TypeError, match="numpy.ndarray"):
        validate_embeddings([[1.0, 2.0], [3.0, 4.0]])  # type: ignore


def test_validate_embeddings_wrong_dim():
    """Test 1D or 3D arrays raise ValueError."""
    with pytest.raises(ValueError, match="2D matrix"):
        validate_embeddings(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="2D matrix"):
        validate_embeddings(np.zeros((10, 5, 2)))


def test_validate_embeddings_empty():
    """Test empty matrix raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validate_embeddings(np.empty((0, 10)))


def test_validate_embeddings_nan_and_inf():
    """Test NaN and Inf detection."""
    arr_nan = np.ones((10, 5))
    arr_nan[2, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_embeddings(arr_nan)

    arr_inf = np.ones((10, 5))
    arr_inf[4, 1] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        validate_embeddings(arr_inf)


def test_validate_embeddings_expected_rows_dim():
    """Test row and dimension mismatch errors."""
    arr = np.ones((10, 5))
    with pytest.raises(ValueError, match="Expected 20 embedding rows"):
        validate_embeddings(arr, expected_rows=20)

    with pytest.raises(ValueError, match="Expected embedding dimension 8"):
        validate_embeddings(arr, expected_dim=8)


def test_normalize_embeddings(sample_embeddings: np.ndarray):
    """Test L2 normalization ensures unit length vectors."""
    normalized = normalize_embeddings(sample_embeddings)
    assert normalized.shape == sample_embeddings.shape
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), rtol=1e-5)


def test_evaluate_k_range(sample_embeddings: np.ndarray):
    """Test evaluation across a range of K values."""
    norm_emb = normalize_embeddings(sample_embeddings)
    k_range = [3, 4, 5]
    best_k, scores, size_dists, labels_by_k = evaluate_k_range(
        norm_emb, k_range=k_range, random_state=42
    )

    assert best_k in k_range
    assert set(scores.keys()) == {3, 4, 5}
    for k in k_range:
        assert isinstance(scores[k], float)
        assert len(labels_by_k[k]) == 50
        assert sum(size_dists[k].values()) == 50


def test_discover_semantic_themes_determinism(
    sample_embeddings: np.ndarray, sample_feedback_ids: list[str]
):
    """Test that fixed random_state produces identical clusters every time."""
    df1, meta1 = discover_semantic_themes(
        embeddings=sample_embeddings,
        feedback_ids=sample_feedback_ids,
        k_range=[3, 4, 5],
        random_state=42,
    )
    df2, meta2 = discover_semantic_themes(
        embeddings=sample_embeddings,
        feedback_ids=sample_feedback_ids,
        k_range=[3, 4, 5],
        random_state=42,
    )

    pd.testing.assert_frame_equal(df1, df2)
    assert meta1["selected_k"] == meta2["selected_k"]
    assert meta1["silhouette_scores"] == meta2["silhouette_scores"]


def test_discover_semantic_themes_row_ordering_and_completeness(
    sample_embeddings: np.ndarray, sample_feedback_ids: list[str]
):
    """Test that all records get cluster assignments and row ordering is preserved."""
    df, meta = discover_semantic_themes(
        embeddings=sample_embeddings,
        feedback_ids=sample_feedback_ids,
        k_range=[3, 4, 5],
        random_state=42,
    )

    assert list(df["feedback_id"]) == sample_feedback_ids
    assert len(df) == len(sample_feedback_ids)
    assert set(df.columns) == {"feedback_id", "cluster_id"}
    assert df["cluster_id"].nunique() == meta["selected_k"]
    assert not df["cluster_id"].isna().any()


def test_metadata_structure(
    sample_embeddings: np.ndarray, sample_feedback_ids: list[str]
):
    """Test metadata keys and content."""
    df, meta = discover_semantic_themes(
        embeddings=sample_embeddings,
        feedback_ids=sample_feedback_ids,
        k_range=[3, 4, 5],
        random_state=42,
        embedding_metadata={"model": "gemini-embedding-001"},
    )

    expected_keys = {
        "selected_k",
        "tested_k_values",
        "silhouette_scores",
        "cluster_sizes",
        "algorithm",
        "random_state",
        "embedding_model",
        "embedding_dimension",
        "dataset_rows",
        "generated_at",
    }
    assert expected_keys.issubset(set(meta.keys()))
    assert meta["dataset_rows"] == 50
    assert meta["embedding_dimension"] == 16
    assert meta["embedding_model"] == "gemini-embedding-001"
    assert meta["tested_k_values"] == [3, 4, 5]


def test_metadata_input_isolation(
    sample_embeddings: np.ndarray, sample_feedback_ids: list[str]
):
    """
    Verify theme/theme_origin/sentiment are NOT part of function inputs,
    guaranteeing no feature leakage.
    """
    # Attempting to pass non-existent metadata arguments will fail signature inspection
    import inspect

    sig = inspect.signature(discover_semantic_themes)
    allowed_params = set(sig.parameters.keys())
    assert "theme" not in allowed_params
    assert "theme_origin" not in allowed_params
    assert "sentiment" not in allowed_params
    assert "severity" not in allowed_params
    assert "rating" not in allowed_params


def test_save_clustering_results(
    sample_embeddings: np.ndarray, sample_feedback_ids: list[str]
):
    """Test file persistence for clusters CSV and JSON metadata."""
    df, meta = discover_semantic_themes(
        embeddings=sample_embeddings,
        feedback_ids=sample_feedback_ids,
        k_range=[3, 4],
        random_state=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_out = Path(tmpdir) / "theme_clusters.csv"
        meta_out = Path(tmpdir) / "clustering_metadata.json"

        save_clustering_results(df, meta, csv_out, meta_out)

        assert csv_out.exists()
        assert meta_out.exists()

        loaded_df = pd.read_csv(csv_out)
        pd.testing.assert_frame_equal(df, loaded_df)

        with open(meta_out, "r", encoding="utf-8") as f:
            loaded_meta = json.load(f)
        assert loaded_meta["selected_k"] == meta["selected_k"]
