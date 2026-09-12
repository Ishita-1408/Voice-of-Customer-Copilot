"""
V4 Evidence Retrieval and Grounded Insight Support Module.

Extracts representative customer feedback evidence for each V4 semantic cluster
using strictly embedding cosine proximity to the cluster centroid.
Calculates evidence volume, centroid similarity, source diversity,
interpretable evidence strength scores, deterministic contradiction analysis,
and confidence scores.

STRICT DATA ISOLATION:
No ground-truth theme or metadata fields influence evidence selection.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.clustering import validate_embeddings

logger = logging.getLogger(__name__)

DEFAULT_TOP_EVIDENCE_COUNT = 10


def validate_evidence_inputs(
    canonical_df: pd.DataFrame,
    embeddings: np.ndarray,
    clusters_df: pd.DataFrame,
    expected_rows: int = 800,
    expected_dim: int = 1536,
    expected_clusters: int = 8,
) -> None:
    """
    Validate input datasets for row counts, embedding shapes, feedback_id alignment,
    and cluster completeness.
    """
    if len(canonical_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} canonical feedback rows, got {len(canonical_df)}"
        )
    if len(clusters_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} cluster assignment rows, got {len(clusters_df)}"
        )

    validate_embeddings(embeddings, expected_rows=expected_rows, expected_dim=expected_dim)

    required_canonical_cols = {"feedback_id", "feedback_text", "source_type", "sentiment"}
    missing_canonical = required_canonical_cols - set(canonical_df.columns)
    if missing_canonical:
        raise ValueError(f"canonical_df is missing required columns: {missing_canonical}")

    if "feedback_id" not in clusters_df.columns or "cluster_id" not in clusters_df.columns:
        raise ValueError("clusters_df must contain 'feedback_id' and 'cluster_id'")

    canonical_fids = canonical_df["feedback_id"].tolist()
    cluster_fids = clusters_df["feedback_id"].tolist()

    if len(set(canonical_fids)) != expected_rows:
        raise ValueError("canonical_df contains duplicate feedback_ids")
    if len(set(cluster_fids)) != expected_rows:
        raise ValueError("clusters_df contains duplicate feedback_ids")

    if canonical_fids != cluster_fids:
        raise ValueError(
            "Row ordering mismatch: feedback_ids in clusters_df do not match canonical_df exactly"
        )

    unique_clusters = set(clusters_df["cluster_id"].dropna().unique())
    if len(unique_clusters) != expected_clusters:
        raise ValueError(
            f"Expected exactly {expected_clusters} clusters, but found {len(unique_clusters)}"
        )


def calculate_evidence_strength(
    evidence_count: int,
    mean_similarity: float,
    source_diversity_count: int,
) -> float:
    """
    Calculate an interpretable evidence strength score bounded between 0.0 and 1.0.

    Formula:
      volume_score     = min(evidence_count / 50.0, 1.0)
      similarity_score = max(0.0, min(1.0, mean_similarity))
      diversity_score  = min(source_diversity_count / 4.0, 1.0)
      evidence_strength = 0.40 * volume_score + 0.40 * similarity_score + 0.20 * diversity_score
    """
    volume_score = min(float(evidence_count) / 50.0, 1.0)
    similarity_score = max(0.0, min(1.0, float(mean_similarity)))
    diversity_score = min(float(source_diversity_count) / 4.0, 1.0)

    raw_score = 0.40 * volume_score + 0.40 * similarity_score + 0.20 * diversity_score
    return round(float(np.clip(raw_score, 0.0, 1.0)), 4)


def calculate_evidence_confidence(
    evidence_strength: float,
    source_diversity_count: int,
) -> float:
    """
    Calculate deterministic confidence score based on evidence strength and source diversity.

    Formula:
      diversity_score = min(source_diversity_count / 4.0, 1.0)
      confidence      = 0.70 * evidence_strength + 0.30 * diversity_score
    """
    diversity_score = min(float(source_diversity_count) / 4.0, 1.0)
    raw_confidence = 0.70 * float(evidence_strength) + 0.30 * diversity_score
    return round(float(np.clip(raw_confidence, 0.0, 1.0)), 4)


def calculate_contradiction(
    sentiments: Sequence[str],
) -> Tuple[Dict[str, int], bool, str]:
    """
    Evaluate contradiction in customer sentiment using deterministic rule.

    Contradiction rule:
      contradiction_flag = TRUE when (positive_count >= 2 AND negative_count >= 2),
      otherwise FALSE.
    """
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for s in sentiments:
        s_clean = str(s).strip().lower()
        if s_clean in counts:
            counts[s_clean] += 1
        elif "pos" in s_clean:
            counts["positive"] += 1
        elif "neg" in s_clean:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

    contradiction_flag = bool(counts["positive"] >= 2 and counts["negative"] >= 2)
    
    if contradiction_flag:
        summary = (
            f"Mixed sentiment observed with {counts['positive']} positive and "
            f"{counts['negative']} negative feedback items ({counts['neutral']} neutral)."
        )
    else:
        dominant_sentiment = max(counts, key=counts.get)
        summary = (
            f"Consistent sentiment profile dominated by {dominant_sentiment} feedback "
            f"({counts['positive']} pos, {counts['neutral']} neu, {counts['negative']} neg)."
        )

    return counts, contradiction_flag, summary


def generate_limitations(
    evidence_count: int,
    source_diversity_count: int,
    confidence: float,
    contradiction_flag: bool,
) -> str:
    """Generate concise deterministic limitations notes for the evidence bundle."""
    limitations_parts = []
    if evidence_count < 25:
        limitations_parts.append(f"Relatively small cluster sample size ({evidence_count} items).")
    if source_diversity_count <= 2:
        limitations_parts.append("Evidence is concentrated in limited source channels.")
    if contradiction_flag:
        limitations_parts.append("Contradictory sentiment signals detected across customer records.")
    if not limitations_parts:
        limitations_parts.append(f"Strong evidence base with high confidence ({confidence:.2f}) across diverse channels.")
        
    return " ".join(limitations_parts)


def retrieve_cluster_evidence(
    cluster_id: int,
    member_fids: List[str],
    member_texts: List[str],
    member_embs: np.ndarray,
    member_sources: List[str],
    member_sentiments: List[str],
    max_representatives: int = DEFAULT_TOP_EVIDENCE_COUNT,
    theme_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Retrieve top supporting evidence records for a single cluster.
    """
    validate_embeddings(member_embs)
    n_members = len(member_fids)

    if n_members != len(member_texts) or n_members != member_embs.shape[0]:
        raise ValueError("Input length mismatch within cluster evidence retrieval")

    # 1. Centroid & cosine similarity
    norm_embs = normalize(member_embs, norm="l2", axis=1)
    centroid = np.mean(norm_embs, axis=0, keepdims=True)
    norm_centroid = normalize(centroid, norm="l2", axis=1)
    sims = np.dot(norm_embs, norm_centroid.T).flatten()

    # 2. Deterministic ranking: similarity descending, feedback_id ascending
    ranked_records = [
        (float(sim), str(fid), str(txt))
        for sim, fid, txt in zip(sims, member_fids, member_texts)
    ]
    ranked_records.sort(key=lambda x: x[1])  # Tie-breaker: feedback_id ascending
    ranked_records.sort(key=lambda x: x[0], reverse=True)  # Primary: similarity descending

    # 3. Select top K supporting records
    k_selected = min(max_representatives, n_members)
    top_records = ranked_records[:k_selected]

    evidence_items: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    theme_meta = theme_info or {}
    theme_name = theme_meta.get("theme_name", f"Theme {cluster_id}")
    theme_def = theme_meta.get("theme_definition") or theme_meta.get("theme_description", "")

    for rank_idx, (sim_score, fid, txt) in enumerate(top_records, start=1):
        evidence_items.append(
            {
                "feedback_id": fid,
                "feedback_text": txt,
                "similarity_score": round(sim_score, 4),
                "evidence_rank": rank_idx,
            }
        )
        csv_rows.append(
            {
                "cluster_id": int(cluster_id),
                "theme_name": theme_name,
                "feedback_id": fid,
                "feedback_text": txt,
                "similarity_score": round(sim_score, 4),
                "evidence_rank": rank_idx,
            }
        )

    # 4. Source diversity
    source_counts: Dict[str, int] = {}
    for st in member_sources:
        st_clean = str(st).strip()
        source_counts[st_clean] = source_counts.get(st_clean, 0) + 1
    source_diversity_count = len(source_counts)

    # 5. Evidence strength
    mean_selected_sim = (
        float(np.mean([r["similarity_score"] for r in evidence_items]))
        if evidence_items
        else 0.0
    )
    strength = calculate_evidence_strength(
        evidence_count=n_members,
        mean_similarity=mean_selected_sim,
        source_diversity_count=source_diversity_count,
    )

    # 6. Confidence
    confidence = calculate_evidence_confidence(
        evidence_strength=strength,
        source_diversity_count=source_diversity_count,
    )

    # 7. Contradiction analysis
    sentiment_counts, contradiction_flag, contradiction_summary = calculate_contradiction(member_sentiments)

    # 8. Limitations
    limitations = generate_limitations(
        evidence_count=n_members,
        source_diversity_count=source_diversity_count,
        confidence=confidence,
        contradiction_flag=contradiction_flag,
    )

    bundle = {
        "cluster_id": int(cluster_id),
        "theme_name": theme_name,
        "theme_definition": theme_def,
        "theme_description": theme_def,
        "evidence_count": int(n_members),
        "selected_evidence_count": int(len(evidence_items)),
        "source_diversity_count": int(source_diversity_count),
        "source_type_count": int(source_diversity_count),
        "source_type_breakdown": source_counts,
        "source_type_counts": source_counts,
        "mean_selected_similarity": round(mean_selected_sim, 4),
        "evidence_strength": float(strength),
        "contradiction_flag": bool(contradiction_flag),
        "contradiction_summary": contradiction_summary,
        "sentiment_counts": sentiment_counts,
        "confidence": float(confidence),
        "limitations": limitations,
        "evidence_items": evidence_items,
        "supporting_feedback": evidence_items,
    }

    return bundle, csv_rows


def build_all_evidence_bundles(
    canonical_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    embeddings: np.ndarray,
    theme_definitions: Optional[List[Dict[str, Any]]] = None,
    max_representatives: int = DEFAULT_TOP_EVIDENCE_COUNT,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    """
    Build evidence bundles for all V4 clusters.
    """
    validate_evidence_inputs(
        canonical_df=canonical_df,
        embeddings=embeddings,
        clusters_df=clusters_df,
        expected_rows=800,
        expected_dim=1536,
        expected_clusters=8,
    )

    theme_lookup: Dict[int, Dict[str, Any]] = {}
    if theme_definitions:
        for tdef in theme_definitions:
            if "cluster_id" in tdef:
                theme_lookup[int(tdef["cluster_id"])] = tdef

    unique_clusters = sorted(clusters_df["cluster_id"].unique())
    all_bundles: List[Dict[str, Any]] = []
    all_csv_rows: List[Dict[str, Any]] = []
    traceability_clusters: List[Dict[str, Any]] = []

    for cid in unique_clusters:
        mask = (clusters_df["cluster_id"] == cid).to_numpy()
        indices = np.where(mask)[0]

        member_fids = canonical_df.iloc[indices]["feedback_id"].tolist()
        member_texts = canonical_df.iloc[indices]["feedback_text"].tolist()
        member_sources = canonical_df.iloc[indices]["source_type"].tolist()
        member_sentiments = canonical_df.iloc[indices]["sentiment"].tolist()
        member_embs = embeddings[indices]

        theme_info = theme_lookup.get(int(cid))

        bundle, rows = retrieve_cluster_evidence(
            cluster_id=int(cid),
            member_fids=member_fids,
            member_texts=member_texts,
            member_embs=member_embs,
            member_sources=member_sources,
            member_sentiments=member_sentiments,
            max_representatives=max_representatives,
            theme_info=theme_info,
        )

        all_bundles.append(bundle)
        all_csv_rows.extend(rows)

        traceability_clusters.append({
            "cluster_id": int(cid),
            "theme_name": bundle["theme_name"],
            "theme_definition": bundle["theme_definition"],
            "evidence_count": bundle["evidence_count"],
            "selected_evidence_count": bundle["selected_evidence_count"],
            "source_diversity_count": bundle["source_diversity_count"],
            "source_type_breakdown": bundle["source_type_breakdown"],
            "evidence_strength": bundle["evidence_strength"],
            "confidence": bundle["confidence"],
            "contradiction_flag": bundle["contradiction_flag"],
            "contradiction_summary": bundle["contradiction_summary"],
            "selected_evidence": bundle["evidence_items"],
        })

    evidence_csv_df = pd.DataFrame(all_csv_rows)
    traceability_data = {
        "pipeline_version": "V4",
        "num_clusters": len(unique_clusters),
        "total_records": len(canonical_df),
        "clusters": traceability_clusters,
    }

    return all_bundles, evidence_csv_df, traceability_data
