"""Vector embedding generation, caching, and management module using Google Gemini API.

Uses Google GenAI SDK (gemini-embedding-001 with SEMANTIC_SIMILARITY task type and
output_dimensionality=1536) to generate embeddings strictly from customer feedback text.
Guarantees deterministic row ordering, prevents metadata leakage, handles 429 rate limits,
and supports caching.
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_TASK_TYPE = "SEMANTIC_SIMILARITY"
DEFAULT_OUTPUT_DIMENSIONALITY = 1536
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_RETRIES = 5
FALLBACK_RETRY_DELAY_SECONDS = 65.0


def get_gemini_client(api_key: Optional[str] = None) -> Any:
    """Instantiates Google GenAI Client using GEMINI_API_KEY environment variable or provided key."""
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai package is required. Install with `pip install google-genai`.")

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. Please set it in .env or pass a client."
        )
    return genai.Client(api_key=key.strip())


def sanitize_input_text(text: Any) -> str:
    """Sanitizes text to prevent API errors on null/empty inputs without altering meaning."""
    if text is None:
        return " "
    if isinstance(text, float) and np.isnan(text):
        return " "
    s = str(text).strip()
    return s if s else " "


def is_rate_limit_error(error: Exception) -> bool:
    """Detects whether an exception corresponds to a 429 / RESOURCE_EXHAUSTED rate limit."""
    err_str = str(error).upper()
    return any(marker in err_str for marker in ["429", "RESOURCE_EXHAUSTED", "RATE_LIMIT", "QUOTA"])


def extract_retry_delay(error: Exception, fallback_delay: float = FALLBACK_RETRY_DELAY_SECONDS) -> float:
    """Extracts suggested retry duration in seconds from rate limit error message or details."""
    error_str = str(error)

    # Match "Please retry in 45.8s" or "retry in 45s"
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s?", error_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 2.0  # +2.0s buffer
        except (ValueError, TypeError):
            pass

    # Match "'retryDelay': '45s'"
    match_delay = re.search(
        r"['\"]?retryDelay['\"]?\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)?)\s*s?['\"]?",
        error_str,
        re.IGNORECASE,
    )
    if match_delay:
        try:
            return float(match_delay.group(1)) + 2.0
        except (ValueError, TypeError):
            pass

    return fallback_delay


def embed_texts(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = MAX_BATCH_RETRIES,
    fallback_delay: float = FALLBACK_RETRY_DELAY_SECONDS,
    client: Optional[Any] = None,
) -> List[List[float]]:
    """Generates embedding vectors for a list of texts in batches using Google GenAI SDK.

    Guarantees strict input order preservation, handles 429 RESOURCE_EXHAUSTED rate limits
    gracefully with dynamic backoff, and avoids unneeded sleep between successful batches.
    """
    if not texts:
        return []

    if client is None:
        client = get_gemini_client()

    from google.genai import types

    sanitized = [sanitize_input_text(t) for t in texts]
    all_embeddings: List[List[float]] = []

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )

    total_batches = (len(sanitized) + batch_size - 1) // batch_size

    for i in range(0, len(sanitized), batch_size):
        batch = sanitized[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=config,
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    batch_vectors = [item.values for item in response.embeddings]
                elif hasattr(response, "embedding") and response.embedding:
                    batch_vectors = [response.embedding.values]
                else:
                    raise ValueError("Unexpected response format from Gemini embed_content API.")

                all_embeddings.extend(batch_vectors)
                logger.info(f"Successfully embedded batch {batch_num}/{total_batches} ({len(batch)} records)")
                break
            except Exception as e:
                if is_rate_limit_error(e) and attempt < max_retries:
                    wait_seconds = extract_retry_delay(e, fallback_delay=fallback_delay)
                    logger.warning(
                        f"Rate limit 429 encountered on batch {batch_num}/{total_batches} (attempt {attempt}/{max_retries}). "
                        f"Waiting {wait_seconds:.1f}s before retrying..."
                    )
                    time.sleep(wait_seconds)
                else:
                    logger.error(f"Failed to generate Gemini embeddings for batch {batch_num} on attempt {attempt}: {e}")
                    raise

    return all_embeddings


def get_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    client: Optional[Any] = None,
) -> List[float]:
    """Generates an embedding vector for a single text."""
    vectors = embed_texts(
        texts=[text],
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        batch_size=1,
        client=client,
    )
    return vectors[0]


def generate_dataset_embeddings(
    df: pd.DataFrame,
    text_column: str = "feedback_text",
    model: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = MAX_BATCH_RETRIES,
    fallback_delay: float = FALLBACK_RETRY_DELAY_SECONDS,
    client: Optional[Any] = None,
) -> np.ndarray:
    """Generates an (N, D) NumPy array of embeddings strictly from df[text_column].

    CRITICAL: Does NOT access or leak theme, sentiment, or other metadata columns.
    """
    if text_column not in df.columns:
        raise ValueError(f"Required text column '{text_column}' not found in DataFrame.")

    # Strictly extract only the feedback text column
    texts = df[text_column].tolist()
    embeddings_list = embed_texts(
        texts=texts,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        batch_size=batch_size,
        max_retries=max_retries,
        fallback_delay=fallback_delay,
        client=client,
    )

    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    assert embeddings_array.shape[0] == len(df), f"Row mismatch: {embeddings_array.shape[0]} != {len(df)}"
    assert embeddings_array.shape[1] == output_dimensionality, (
        f"Dimension mismatch: expected {output_dimensionality}, got {embeddings_array.shape[1]}"
    )
    return embeddings_array


def save_embeddings(embeddings: np.ndarray, output_path: Union[str, Path]) -> None:
    """Saves embedding array to .npy file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, embeddings)
    logger.info(f"Saved embeddings {embeddings.shape} to {p}")


def load_embeddings(input_path: Union[str, Path]) -> np.ndarray:
    """Loads embedding array from .npy file."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")
    return np.load(p)


def save_embedding_metadata(metadata: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Saves embedding metadata to JSON without exposing secrets or API keys."""
    safe_metadata = {k: v for k, v in metadata.items() if "key" not in k.lower() and "secret" not in k.lower()}
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(safe_metadata, f, indent=2)
    logger.info(f"Saved embedding metadata to {p}")


def load_embedding_metadata(input_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads embedding metadata JSON."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding metadata file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_create_dataset_embeddings(
    df: pd.DataFrame,
    text_column: str = "feedback_text",
    model: str = DEFAULT_EMBEDDING_MODEL,
    task_type: str = DEFAULT_TASK_TYPE,
    output_dimensionality: int = DEFAULT_OUTPUT_DIMENSIONALITY,
    output_embeddings_path: Union[str, Path] = "data/processed/feedback_embeddings.npy",
    output_metadata_path: Union[str, Path] = "data/processed/embedding_metadata.json",
    source_dataset_path: str = "data/processed/voc_feedback.csv",
    force_regenerate: bool = False,
    max_retries: int = MAX_BATCH_RETRIES,
    fallback_delay: float = FALLBACK_RETRY_DELAY_SECONDS,
    client: Optional[Any] = None,
) -> Tuple[np.ndarray, bool]:
    """Caches and returns dataset embeddings.

    Reuses cached embeddings if present and valid unless force_regenerate is True.
    """
    emb_p = Path(output_embeddings_path)
    meta_p = Path(output_metadata_path)

    if not force_regenerate and emb_p.exists():
        try:
            cached = load_embeddings(emb_p)
            if cached.shape[0] == len(df) and cached.shape[1] == output_dimensionality:
                logger.info(f"Cache hit: Loaded {cached.shape[0]} embeddings from {emb_p}")
                return cached, True
            else:
                logger.warning(
                    f"Cache mismatch ({cached.shape} cached vs expected ({len(df)}, {output_dimensionality})). Regenerating..."
                )
        except Exception as e:
            logger.warning(f"Failed to read cache file ({e}). Regenerating...")

    # Generate fresh embeddings
    embeddings = generate_dataset_embeddings(
        df=df,
        text_column=text_column,
        model=model,
        task_type=task_type,
        output_dimensionality=output_dimensionality,
        max_retries=max_retries,
        fallback_delay=fallback_delay,
        client=client,
    )

    save_embeddings(embeddings, emb_p)

    metadata = {
        "model": model,
        "task_type": task_type,
        "dataset_rows": int(len(df)),
        "embedding_dimension": int(embeddings.shape[1]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dataset_path),
    }
    save_embedding_metadata(metadata, meta_p)

    return embeddings, False
