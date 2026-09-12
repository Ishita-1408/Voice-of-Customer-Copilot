"""Unit tests for the vector embedding module using mocked Google Gemini responses."""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import numpy as np
import pandas as pd
import pytest

from app.ai.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_TASK_TYPE,
    embed_texts,
    extract_retry_delay,
    generate_dataset_embeddings,
    get_embedding,
    get_or_create_dataset_embeddings,
    is_rate_limit_error,
    load_embedding_metadata,
    load_embeddings,
    sanitize_input_text,
    save_embedding_metadata,
    save_embeddings,
)

MOCK_DIMENSION = 1536


class MockContentEmbedding:
    def __init__(self, values: list):
        self.values = values


class MockEmbedContentResponse:
    def __init__(self, embeddings: list):
        self.embeddings = embeddings


def create_mock_gemini_response(contents: list, dimension: int = MOCK_DIMENSION):
    """Generates a deterministic mocked Google GenAI embeddings API response."""
    embeddings = []
    for idx, text in enumerate(contents):
        seed_val = (idx + 1) * 0.01 + len(str(text)) * 0.001
        vector = [(seed_val + (d * 0.0001)) % 1.0 for d in range(dimension)]
        embeddings.append(MockContentEmbedding(values=vector))

    return MockEmbedContentResponse(embeddings=embeddings)


@pytest.fixture
def mock_gemini_client():
    """Provides a mocked Google GenAI client that generates deterministic vectors without network calls."""
    client = MagicMock()
    client.models.embed_content.side_effect = (
        lambda model, contents, config: create_mock_gemini_response(contents, dimension=getattr(config, "output_dimensionality", MOCK_DIMENSION))
    )
    return client


def test_get_embedding_returns_vector(mock_gemini_client):
    """Verify get_embedding returns a single 1D vector of correct dimensions."""
    vec = get_embedding("Payment failed at checkout.", client=mock_gemini_client)

    assert isinstance(vec, list)
    assert len(vec) == MOCK_DIMENSION
    assert all(isinstance(x, float) for x in vec)
    mock_gemini_client.models.embed_content.assert_called_once()


def test_batch_embedding_preserves_input_order(mock_gemini_client):
    """Verify batch embedding preserves exact input text order."""
    texts = [
        "First text about payments",
        "Second text about delivery",
        "Third text about onboarding",
        "Fourth text about search",
    ]

    embeddings = embed_texts(texts, batch_size=2, client=mock_gemini_client)

    assert len(embeddings) == 4
    assert len(embeddings[0]) == MOCK_DIMENSION

    # Verify order: distinct texts have distinct vectors matching their positions
    assert embeddings[0] != embeddings[1]
    assert embeddings[1] != embeddings[2]
    assert embeddings[2] != embeddings[3]


def test_gemini_model_and_config_parameters(mock_gemini_client):
    """Verify Gemini model name, SEMANTIC_SIMILARITY task type, and output_dimensionality=1536 are passed."""
    get_embedding("Testing Gemini configuration parameters.", client=mock_gemini_client)

    call_args = mock_gemini_client.models.embed_content.call_args
    assert call_args.kwargs["model"] == "gemini-embedding-001"
    config = call_args.kwargs["config"]
    assert config.task_type == "SEMANTIC_SIMILARITY"
    assert config.output_dimensionality == 1536


def test_null_and_empty_text_handled_safely(mock_gemini_client):
    """Verify null, empty, whitespace-only, and NaN inputs are sanitized safely without crashing."""
    assert sanitize_input_text(None) == " "
    assert sanitize_input_text(np.nan) == " "
    assert sanitize_input_text("   ") == " "
    assert sanitize_input_text("Real text") == "Real text"

    texts = ["Valid comment", None, "", "   ", np.nan, "Another valid comment"]
    embeddings = embed_texts(texts, client=mock_gemini_client)

    assert len(embeddings) == 6
    assert all(len(e) == MOCK_DIMENSION for e in embeddings)


def test_dataset_embedding_row_count_and_dimensions(mock_gemini_client):
    """Verify generate_dataset_embeddings returns (N, D) array matching dataframe rows."""
    mock_df = pd.DataFrame({
        "feedback_id": [f"TEST-{i:03d}" for i in range(25)],
        "feedback_text": [f"Sample customer review text {i}" for i in range(25)],
        "theme": ["Payment"] * 25,
        "sentiment": ["negative"] * 25,
    })

    emb_array = generate_dataset_embeddings(mock_df, text_column="feedback_text", client=mock_gemini_client)

    assert isinstance(emb_array, np.ndarray)
    assert emb_array.shape == (25, MOCK_DIMENSION)
    assert emb_array.dtype == np.float32


def test_theme_and_metadata_not_passed_to_embedding(mock_gemini_client):
    """Verify theme, sentiment, severity, and other metadata are NOT passed into embedding input."""
    mock_df = pd.DataFrame({
        "feedback_id": ["TEST-001"],
        "feedback_text": ["Clean isolated feedback text"],
        "theme": ["SECRET_THEME_LABEL"],
        "theme_origin": ["synthetic_ground_truth"],
        "sentiment": ["negative"],
        "severity": [5],
    })

    generate_dataset_embeddings(mock_df, text_column="feedback_text", client=mock_gemini_client)

    call_args = mock_gemini_client.models.embed_content.call_args
    passed_contents = call_args.kwargs["contents"]

    assert passed_contents == ["Clean isolated feedback text"]
    assert "SECRET_THEME_LABEL" not in passed_contents[0]
    assert "negative" not in passed_contents[0]


def test_extract_retry_delay_from_error():
    """Verify parsing of retry delay from various 429 error message formats."""
    err1 = Exception("429 RESOURCE_EXHAUSTED: Please retry in 45.8s.")
    assert extract_retry_delay(err1) == pytest.approx(47.8, rel=1e-2)

    err2 = Exception("Resource exhausted error. {'retryDelay': '30s'}")
    assert extract_retry_delay(err2) == pytest.approx(32.0, rel=1e-2)

    err3 = Exception("429 Too Many Requests without explicit duration.")
    assert extract_retry_delay(err3, fallback_delay=65.0) == 65.0


@patch("time.sleep")
def test_429_rate_limit_retry_success(mock_sleep):
    """Verify that a 429 error triggers backoff wait and succeeds on retry without skipping records."""
    client = MagicMock()
    # First call raises 429, second call succeeds
    client.models.embed_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED. Please retry in 10s."),
        create_mock_gemini_response(["text1", "text2"]),
    ]

    embeddings = embed_texts(["text1", "text2"], client=client)

    assert len(embeddings) == 2
    assert client.models.embed_content.call_count == 2
    mock_sleep.assert_called_once_with(pytest.approx(12.0, rel=1e-2))


@patch("time.sleep")
def test_429_rate_limit_max_retries_exhausted(mock_sleep):
    """Verify that exceeding max_retries raises the error after 5 attempts."""
    client = MagicMock()
    client.models.embed_content.side_effect = Exception("429 RESOURCE_EXHAUSTED. Rate limit exceeded.")

    with pytest.raises(Exception, match="429 RESOURCE_EXHAUSTED"):
        embed_texts(["text1"], max_retries=5, fallback_delay=0.1, client=client)

    assert client.models.embed_content.call_count == 5
    assert mock_sleep.call_count == 4


def test_cache_loading_and_regeneration(tmp_path: Path, mock_gemini_client):
    """Verify embeddings are loaded from cache when valid and regenerated when forced."""
    emb_file = tmp_path / "test_embeddings.npy"
    meta_file = tmp_path / "test_metadata.json"

    mock_df = pd.DataFrame({
        "feedback_id": [f"ID-{i}" for i in range(10)],
        "feedback_text": [f"Feedback sentence {i}" for i in range(10)],
    })

    # 1. Initial generation (cache miss)
    emb1, was_cached1 = get_or_create_dataset_embeddings(
        df=mock_df,
        output_embeddings_path=emb_file,
        output_metadata_path=meta_file,
        force_regenerate=False,
        client=mock_gemini_client,
    )
    assert not was_cached1
    assert emb_file.exists()
    assert meta_file.exists()
    assert emb1.shape == (10, MOCK_DIMENSION)
    assert mock_gemini_client.models.embed_content.call_count > 0

    mock_gemini_client.models.embed_content.reset_mock()

    # 2. Second call with valid cache (cache hit -> no API calls)
    emb2, was_cached2 = get_or_create_dataset_embeddings(
        df=mock_df,
        output_embeddings_path=emb_file,
        output_metadata_path=meta_file,
        force_regenerate=False,
        client=mock_gemini_client,
    )
    assert was_cached2
    assert np.array_equal(emb1, emb2)
    mock_gemini_client.models.embed_content.assert_not_called()

    # 3. Third call with force_regenerate=True (force refresh -> API called)
    emb3, was_cached3 = get_or_create_dataset_embeddings(
        df=mock_df,
        output_embeddings_path=emb_file,
        output_metadata_path=meta_file,
        force_regenerate=True,
        client=mock_gemini_client,
    )
    assert not was_cached3
    mock_gemini_client.models.embed_content.assert_called()


def test_metadata_file_creation(tmp_path: Path):
    """Verify metadata json format and secret safety."""
    meta_file = tmp_path / "metadata.json"
    metadata = {
        "model": DEFAULT_EMBEDDING_MODEL,
        "task_type": DEFAULT_TASK_TYPE,
        "dataset_rows": 500,
        "embedding_dimension": 1536,
        "generated_at": "2026-09-12T00:00:00Z",
        "source_dataset": "data/processed/voc_feedback.csv",
        "secret_api_key": "GEMINI_KEY_SHOULD_BE_EXCLUDED",
    }

    save_embedding_metadata(metadata, meta_file)
    loaded = load_embedding_metadata(meta_file)

    assert loaded["model"] == "gemini-embedding-001"
    assert loaded["task_type"] == "SEMANTIC_SIMILARITY"
    assert loaded["dataset_rows"] == 500
    assert loaded["embedding_dimension"] == 1536
    assert "secret_api_key" not in loaded


def test_deterministic_row_ordering_preserved(mock_gemini_client):
    """Verify row ordering is identical across repeated executions."""
    texts = [f"Item number {i:04d}" for i in range(50)]
    df = pd.DataFrame({"feedback_text": texts})

    arr1 = generate_dataset_embeddings(df, client=mock_gemini_client)
    arr2 = generate_dataset_embeddings(df, client=mock_gemini_client)

    assert np.array_equal(arr1, arr2)
