"""CLI script to generate or load vector embeddings for the canonical VoC dataset using Google Gemini API.

Usage:
    python scripts/generate_embeddings.py [--force]
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from app.ai.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OUTPUT_DIMENSIONALITY,
    DEFAULT_TASK_TYPE,
    get_or_create_dataset_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for VoC feedback dataset using Google Gemini.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of embeddings even if valid cache exists.",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/voc_feedback.csv",
        help="Path to the canonical dataset CSV.",
    )
    parser.add_argument(
        "--output-embeddings",
        default="data/processed/feedback_embeddings.npy",
        help="Target path for embeddings .npy file.",
    )
    parser.add_argument(
        "--output-metadata",
        default="data/processed/embedding_metadata.json",
        help="Target path for embeddings metadata .json file.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at '{dataset_path}'")
        sys.exit(1)

    # 1. Load dataset
    df = pd.read_csv(dataset_path)

    # 2. Validate dataset invariants
    if len(df) == 0:
        print("Error: Dataset is empty")
        sys.exit(1)

    if not df["feedback_id"].is_unique:
        print("Error: feedback_id column contains duplicates")
        sys.exit(1)

    if df["feedback_text"].isnull().any() or (df["feedback_text"].str.strip() == "").any():
        print("Error: feedback_text column contains null or empty values")
        sys.exit(1)

    # 3. Generate or load embeddings
    try:
        embeddings, was_cached = get_or_create_dataset_embeddings(
            df=df,
            text_column="feedback_text",
            model=DEFAULT_EMBEDDING_MODEL,
            task_type=DEFAULT_TASK_TYPE,
            output_dimensionality=DEFAULT_OUTPUT_DIMENSIONALITY,
            output_embeddings_path=args.output_embeddings,
            output_metadata_path=args.output_metadata,
            source_dataset_path=str(dataset_path),
            force_regenerate=args.force,
        )
    except ValueError as ve:
        print(f"\n[Notice] {ve}")
        print("To generate real Gemini embeddings, please set GEMINI_API_KEY in .env.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during Gemini embedding generation: {e}")
        sys.exit(1)

    # 4. Format and print report
    print("=" * 40)
    print("EMBEDDING GENERATION")
    print("=" * 40)
    print(f"Dataset rows:         {len(df):,}")
    print(f"Embedding model:      {DEFAULT_EMBEDDING_MODEL}")
    print(f"Task type:            {DEFAULT_TASK_TYPE}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embeddings generated: {embeddings.shape[0]} ({'Loaded from cache' if was_cached else 'Freshly computed'})")
    print(f"Output file:          {args.output_embeddings}")
    print("=" * 40)


if __name__ == "__main__":
    main()
