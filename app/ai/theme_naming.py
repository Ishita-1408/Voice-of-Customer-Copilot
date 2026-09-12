"""
Semantic cluster theme naming module using Google Gemini LLM.

Generates human-readable, actionable product theme names and descriptions
from discovered embedding clusters using representative feedback items
selected via embedding centroid proximity.
"""

from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import normalize

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.clustering import validate_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_NAMING_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.6-flash")
DEFAULT_REPRESENTATIVES_PER_CLUSTER = 10
MAX_NAMING_RETRIES = 3
FALLBACK_NAMING_RETRY_DELAY = 15.0

FALLBACK_THEME_DEF = {
    "theme_name": "Mixed Uncategorized Feedback",
    "theme_description": "Diverse customer feedback requiring manual inspection.",
    "problem_summary": "A collection of customer feedback items with varied topics without a single dominant pattern.",
    "confidence": 0.3,
}


def get_gemini_client(api_key: Optional[str] = None) -> Any:
    """Instantiate Google GenAI client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package is required. Install with `pip install google-genai`."
        )

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. Please set it in .env or pass a client."
        )
    return genai.Client(api_key=key.strip())


def extract_retry_delay_from_error(
    error: Exception, fallback_delay: float = FALLBACK_NAMING_RETRY_DELAY
) -> float:
    """Extract retry delay from 429 RESOURCE_EXHAUSTED errors."""
    error_str = str(error)
    match = re.search(
        r"Please retry in ([0-9]+(?:\.[0-9]+)?)\s*s", error_str, re.IGNORECASE
    )
    if match:
        try:
            return float(match.group(1)) + 2.0
        except (ValueError, TypeError):
            pass

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


def select_representative_feedback(
    feedback_ids: Sequence[str],
    feedback_texts: Sequence[str],
    embeddings: np.ndarray,
    n_representatives: int = DEFAULT_REPRESENTATIVES_PER_CLUSTER,
) -> Tuple[List[str], List[str]]:
    """
    Select representative feedback records closest to the cluster centroid.

    Deterministic ranking:
    1. L2 normalize cluster embeddings.
    2. Compute mean vector (centroid) and L2 normalize it.
    3. Calculate cosine similarities between each record and centroid.
    4. Sort descending by similarity, tie-breaking by feedback_id ascending.
    5. Return top N feedback_ids and feedback_texts.
    """
    validate_embeddings(embeddings)
    n_records = len(feedback_ids)
    if n_records != len(feedback_texts) or n_records != embeddings.shape[0]:
        raise ValueError(
            f"Mismatched input lengths: {len(feedback_ids)} ids, {len(feedback_texts)} texts, {embeddings.shape[0]} embeddings"
        )

    if n_records == 0:
        return [], []

    # If cluster has fewer records than requested sample, return all deterministically sorted
    k_samples = min(n_representatives, n_records)

    # 1. Normalize embeddings
    norm_embeddings = normalize(embeddings, norm="l2", axis=1)

    # 2. Centroid
    centroid = np.mean(norm_embeddings, axis=0, keepdims=True)
    norm_centroid = normalize(centroid, norm="l2", axis=1)

    # 3. Cosine similarity
    similarities = np.dot(norm_embeddings, norm_centroid.T).flatten()

    # 4. Deterministic sort: similarity descending, feedback_id ascending
    indexed_data = [
        (float(sim), str(fid), str(txt))
        for sim, fid, txt in zip(similarities, feedback_ids, feedback_texts)
    ]
    # Python sort is stable: sort by fid first, then sort by similarity descending
    indexed_data.sort(key=lambda x: x[1])
    indexed_data.sort(key=lambda x: x[0], reverse=True)

    selected = indexed_data[:k_samples]
    selected_ids = [item[1] for item in selected]
    selected_texts = [item[2] for item in selected]

    return selected_ids, selected_texts


def build_theme_naming_prompt(representative_texts: List[str]) -> str:
    """Construct structured prompt for LLM theme naming."""
    formatted_examples = "\n".join(
        [f"[{i + 1}] \"{text.strip()}\"" for i, text in enumerate(representative_texts)]
    )

    prompt = f"""You are an expert Voice-of-Customer product analytics assistant.
Analyze the following representative customer feedback samples from a single semantic cluster and identify the underlying product problem/topic.

REPRESENTATIVE CUSTOMER FEEDBACK SAMPLES:
{formatted_examples}

INSTRUCTIONS:
1. "theme_name": A concise, descriptive product theme name (2 to 6 words). Must describe the common customer issue, topic, or need.
2. "theme_description": Exactly one concise sentence explaining what this theme represents.
3. "problem_summary": A short paragraph explaining the core customer friction point or pain point.
4. "confidence": A float between 0.0 and 1.0 indicating how strongly and consistently the samples support this coherent theme.
5. Strict constraints:
   - Do NOT invent facts or features not mentioned in the feedback.
   - Do NOT mention cluster numbers or IDs (e.g. do NOT write "Cluster 0").
   - Avoid overly generic names like "Customer Issues", "Product Feedback", or "User Problems".
   - If the feedback is mixed or contains unrelated topics, set theme_name to "Mixed Customer Feedback" and confidence <= 0.4.

Respond with ONLY a valid JSON object in this exact schema:
{{
  "theme_name": "...",
  "theme_description": "...",
  "problem_summary": "...",
  "confidence": 0.95
}}
"""
    return prompt


def parse_and_validate_theme_json(raw_text: str) -> Dict[str, Any]:
    """Parse and validate JSON response from Gemini, handling markdown code fences."""
    cleaned = raw_text.strip()
    # Strip markdown ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object.")

    required_keys = ["theme_name", "theme_description", "problem_summary", "confidence"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in theme JSON: {key}")

    theme_name = str(data["theme_name"]).strip()
    if not theme_name:
        raise ValueError("theme_name must be a non-empty string.")

    theme_description = str(data["theme_description"]).strip()
    problem_summary = str(data["problem_summary"]).strip()

    try:
        confidence = float(data["confidence"])
    except (ValueError, TypeError):
        confidence = 0.5

    # Clamp confidence to [0.0, 1.0]
    confidence = max(0.0, min(1.0, round(confidence, 2)))

    return {
        "theme_name": theme_name,
        "theme_description": theme_description,
        "problem_summary": problem_summary,
        "confidence": confidence,
    }


def name_cluster_with_gemini(
    representative_texts: List[str],
    client: Optional[Any] = None,
    model: str = DEFAULT_NAMING_MODEL,
    max_retries: int = MAX_NAMING_RETRIES,
) -> Dict[str, Any]:
    """
    Call Gemini LLM to name a single cluster from representative feedback.
    Handles rate limits, retries once on JSON malformation, and falls back gracefully.
    """
    if not representative_texts:
        return dict(FALLBACK_THEME_DEF)

    if client is None:
        client = get_gemini_client()

    prompt = build_theme_naming_prompt(representative_texts)

    # Attempt LLM generation with rate-limit and parsing retries
    json_parse_attempts = 2
    for parse_attempt in range(json_parse_attempts):
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Use Google GenAI generate_content
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                raw_text = response.text if hasattr(response, "text") else str(response)
                parsed = parse_and_validate_theme_json(raw_text)
                return parsed

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = extract_retry_delay_from_error(e)
                    logger.warning(
                        "Rate limit encountered while naming cluster (attempt %d/%d). Waiting %.1fs...",
                        retry_count + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    retry_count += 1
                elif isinstance(e, (json.JSONDecodeError, ValueError)):
                    logger.warning(
                        "JSON parsing error on cluster naming (parse attempt %d/%d): %s",
                        parse_attempt + 1,
                        json_parse_attempts,
                        e,
                    )
                    break  # Break retry loop to trigger next parse attempt
                else:
                    logger.error("Unexpected error during Gemini theme naming: %s", e)
                    retry_count += 1
                    time.sleep(2.0)

    logger.error("Failed to parse valid theme JSON after retries; using fallback theme.")
    return dict(FALLBACK_THEME_DEF)


def generate_theme_definitions(
    feedback_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    embeddings: np.ndarray,
    client: Optional[Any] = None,
    model: str = DEFAULT_NAMING_MODEL,
    n_representatives: int = DEFAULT_REPRESENTATIVES_PER_CLUSTER,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generate human-readable theme names for all clusters.

    CRITICAL DATA ISOLATION:
    Only feedback_id, feedback_text, and cluster_id are utilized.
    No metadata or synthetic theme columns are ever inspected or passed to the LLM.
    """
    validate_embeddings(embeddings, expected_rows=len(feedback_df))

    # Verify input columns strictly
    if "feedback_id" not in feedback_df.columns or "feedback_text" not in feedback_df.columns:
        raise ValueError("feedback_df must contain 'feedback_id' and 'feedback_text'")

    if "feedback_id" not in clusters_df.columns or "cluster_id" not in clusters_df.columns:
        raise ValueError("clusters_df must contain 'feedback_id' and 'cluster_id'")

    # Align data by index
    merged = pd.merge(
        feedback_df[["feedback_id", "feedback_text"]],
        clusters_df[["feedback_id", "cluster_id"]],
        on="feedback_id",
        how="inner",
    )

    if len(merged) != len(feedback_df):
        raise ValueError("Mismatch between feedback_df and clusters_df row alignment")

    unique_clusters = sorted(merged["cluster_id"].unique())
    logger.info("Naming %d semantic clusters using model %s...", len(unique_clusters), model)

    theme_definitions_records: List[Dict[str, Any]] = []
    json_definitions: List[Dict[str, Any]] = []

    for cluster_id in unique_clusters:
        # Find row indices for this cluster
        cluster_mask = (merged["cluster_id"] == cluster_id).to_numpy()
        cluster_indices = np.where(cluster_mask)[0]

        cluster_fids = merged.iloc[cluster_indices]["feedback_id"].tolist()
        cluster_texts = merged.iloc[cluster_indices]["feedback_text"].tolist()
        cluster_embs = embeddings[cluster_indices]

        # Select representative feedback
        rep_ids, rep_texts = select_representative_feedback(
            feedback_ids=cluster_fids,
            feedback_texts=cluster_texts,
            embeddings=cluster_embs,
            n_representatives=n_representatives,
        )

        logger.info(
            "Cluster %d (%d records): Selected %d representative feedback items.",
            cluster_id,
            len(cluster_indices),
            len(rep_ids),
        )

        # Call Gemini LLM (1 call per cluster)
        theme_info = name_cluster_with_gemini(
            representative_texts=rep_texts,
            client=client,
            model=model,
        )

        # CSV record
        theme_definitions_records.append(
            {
                "cluster_id": int(cluster_id),
                "theme_name": theme_info["theme_name"],
                "theme_description": theme_info["theme_description"],
                "problem_summary": theme_info["problem_summary"],
                "confidence": float(theme_info["confidence"]),
                "representative_feedback_ids": ";".join(rep_ids),
            }
        )

        # JSON record
        json_definitions.append(
            {
                "cluster_id": int(cluster_id),
                "theme_name": theme_info["theme_name"],
                "theme_description": theme_info["theme_description"],
                "problem_summary": theme_info["problem_summary"],
                "confidence": float(theme_info["confidence"]),
                "representative_feedback_ids": rep_ids,
                "representative_feedback_texts": rep_texts,
            }
        )

    theme_definitions_df = pd.DataFrame(theme_definitions_records)
    return theme_definitions_df, json_definitions


def save_theme_definitions(
    theme_df: pd.DataFrame,
    theme_json: List[Dict[str, Any]],
    csv_path: Union[str, Path] = "data/processed/theme_definitions.csv",
    json_path: Union[str, Path] = "data/processed/theme_definitions.json",
) -> None:
    """Save theme definitions to CSV and JSON."""
    csv_p = Path(csv_path)
    json_p = Path(json_path)

    csv_p.parent.mkdir(parents=True, exist_ok=True)
    json_p.parent.mkdir(parents=True, exist_ok=True)

    theme_df.to_csv(csv_p, index=False)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(theme_json, f, indent=2)

    logger.info("Saved theme definitions CSV to %s", csv_p)
    logger.info("Saved theme definitions JSON to %s", json_p)
