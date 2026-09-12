"""
V4 Evidence-Grounded Product Insight and Recommendation Generation Module.

Transforms authoritative priority metrics, deterministic decision classifications,
and retrieved customer evidence into actionable, evidence-traceable PM insights
using Google Gemini LLM with strict grounding, citation guardrails, and deterministic
metric preservation.
"""

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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.theme_naming import extract_retry_delay_from_error, get_gemini_client

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_INSIGHT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.6-flash")
MAX_INSIGHT_RETRIES = 3


def build_insight_prompt(
    theme_assessment: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
) -> str:
    """
    Construct a strictly grounded prompt providing ONLY authoritative priority metrics,
    deterministic decision, and retrieved customer evidence examples.
    """
    cid = theme_assessment["cluster_id"]
    tname = theme_assessment["theme_name"]
    tdef = theme_assessment.get("theme_definition") or theme_assessment.get("theme_description", "")
    decision = theme_assessment["decision"]
    p_score = theme_assessment["priority_score"]
    ev_count = theme_assessment["evidence_count"]
    m_sev = theme_assessment["mean_severity"]
    sev_score = theme_assessment["severity_score"]
    seg_imp = theme_assessment["segment_impact_score"]
    known_seg_cnt = theme_assessment["known_segment_count"]
    known_seg_frac = theme_assessment["known_segment_fraction"]
    ev_str = theme_assessment["evidence_strength"]
    ev_div = theme_assessment["evidence_diversity_score"]
    src_cnt = theme_assessment["source_diversity_count"]
    contra_flag = theme_assessment["contradiction_flag"]
    seg_limitation = theme_assessment.get("segment_limitation", "")
    src_dist = evidence_bundle.get("source_type_breakdown", {})
    sent_dist = evidence_bundle.get("sentiment_counts", {})
    contra_summary = evidence_bundle.get("contradiction_summary", "")

    supporting_samples = evidence_bundle.get("evidence_items") or evidence_bundle.get("supporting_feedback", [])
    formatted_evidence = "\n".join(
        [
            f"- [{item['feedback_id']}] (Cosine Sim: {item['similarity_score']:.4f}) \"{item['feedback_text']}\""
            for item in supporting_samples
        ]
    )

    special_instructions = ""
    if cid == 6 or "Mixed" in tname:
        special_instructions = """
SPECIAL INSTRUCTION FOR HETEROGENEOUS/MIXED CLUSTER:
This cluster represents "Mixed Customer Feedback" containing disparate functional issues (e.g. search relevance, account onboarding, notification links).
1. Explicitly acknowledge that this cluster is heterogeneous with multiple sub-problems.
2. Recommend decomposing the cluster into targeted modular investigations rather than proposing a single monolithic product intervention.
3. Note that the BUILD decision is driven by the aggregated mathematical priority score, but low semantic homogeneity limits a single engineering fix.
"""

    prompt = f"""You are an expert Principal Product Manager creating an evidence-grounded product insight for a Voice-of-Customer analytics system.
You are given verified customer feedback evidence, deterministic priority metrics, and an AUTHORITATIVE product decision.

THEME OVERVIEW:
- Cluster ID: {cid}
- Theme Name: {tname}
- Theme Definition: {tdef}

AUTHORITATIVE DECISION & METRICS:
- AUTHORITATIVE DECISION: {decision}
  (CRITICAL RULE: You must NOT change this decision. Formulate recommendations strictly aligned with '{decision}')
- Priority Score: {p_score:.4f}
- Total Evidence Volume: {ev_count} feedback records
- Mean Severity: {m_sev:.2f} / 5.0 (Normalized: {sev_score:.4f})
- Segment Impact Score: {seg_imp:.2f} (Known customer segments: {known_seg_cnt}, Known record fraction: {known_seg_frac:.1%})
- Segment Limitation Note: {seg_limitation}
- Evidence Strength: {ev_str:.4f}
- Evidence Diversity Score: {ev_div:.2f} ({src_cnt} source types: {src_dist})
- Sentiment Breakdown: {sent_dist}
- Contradiction Flag: {contra_flag} ({contra_summary})

VERIFIED SUPPORTING EVIDENCE EXAMPLES (TOP 10 CLOSEST TO CENTROID):
{formatted_evidence}
{special_instructions}
GUARDRAILS & CONSTRAINTS:
1. CITATION REQUIREMENT: In "supporting_feedback_ids", select 5 to 10 feedback IDs from the verified supporting evidence list above. Do NOT invent IDs.
2. NO SPECULATION: Do NOT invent metrics, revenue impacts, churn rates, or claims not present in the evidence. Public reviews are public feedback, synthetic records are synthetic demonstration data.
3. CONTRADICTION HANDLING: If contradiction_flag is True, explicitly mention in the insight that feedback contains opposing sentiments.
4. UNKNOWN SEGMENT HANDLING: Preserve and incorporate the segment limitation notes (especially if public review data dominates).
5. DECISION ALIGNMENT:
   - BUILD: Recommend concrete product improvements and interventions.
   - INVESTIGATE: Recommend targeted user research, analytics instrumentation, or exploratory discovery.
   - MONITOR: Recommend telemetry metrics to track and thresholds to watch.
   - DON'T BUILD YET: Recommend deferring resources and explain why evidence/severity is currently insufficient.

Respond with ONLY a valid JSON object matching this schema:
{{
  "insight": "Concise, evidence-grounded executive summary of what customer feedback reveals.",
  "customer_problem": "Core underlying customer pain point directly supported by the supplied feedback.",
  "recommended_action": "Actionable product next step directly consistent with decision '{decision}'.",
  "supporting_feedback_ids": ["ID1", "ID2", "..."],
  "confidence": 0.90,
  "limitations": "Known data gaps, unknown segment caveats, or semantic mixture notes."
}}
"""
    return prompt


def parse_and_validate_insight_json(
    raw_text: str,
    valid_feedback_ids: Sequence[str],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Parse and validate LLM JSON response for insights.
    Filters supporting_feedback_ids so only valid IDs provided in the evidence bundle remain.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object.")

    required_keys = [
        "insight",
        "customer_problem",
        "recommended_action",
        "supporting_feedback_ids",
        "confidence",
        "limitations",
    ]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in insight JSON: {key}")

    insight = str(data["insight"]).strip()
    cust_prob = str(data["customer_problem"]).strip()
    rec_action = str(data["recommended_action"]).strip()
    limitations = str(data["limitations"]).strip()

    try:
        conf = float(data["confidence"])
    except (ValueError, TypeError):
        conf = 0.5
    conf = float(np.clip(round(conf, 2), 0.0, 1.0))

    # Validate citations
    raw_cited_ids = data["supporting_feedback_ids"]
    if isinstance(raw_cited_ids, str):
        raw_cited_ids = [raw_cited_ids]

    valid_id_set = set(valid_feedback_ids)
    valid_cited: List[str] = []
    invalid_cited: List[str] = []

    for fid in raw_cited_ids:
        fid_clean = str(fid).strip()
        if fid_clean in valid_id_set:
            if fid_clean not in valid_cited:
                valid_cited.append(fid_clean)
        else:
            invalid_cited.append(fid_clean)

    # Fallback to top 5 valid IDs if insufficient citations
    if len(valid_cited) < 3 and valid_feedback_ids:
        for fid in valid_feedback_ids[:5]:
            if fid not in valid_cited:
                valid_cited.append(fid)

    result = {
        "insight": insight,
        "customer_problem": cust_prob,
        "recommended_action": rec_action,
        "supporting_feedback_ids": valid_cited,
        "confidence": conf,
        "limitations": limitations,
    }

    return result, valid_cited, invalid_cited


def build_fallback_insight(
    theme_assessment: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a deterministic fallback insight if LLM calls fail."""
    cid = theme_assessment["cluster_id"]
    tname = theme_assessment["theme_name"]
    tdef = theme_assessment.get("theme_definition", "")
    decision = theme_assessment["decision"]
    ev_count = theme_assessment["evidence_count"]
    m_sev = theme_assessment["mean_severity"]
    contra_flag = theme_assessment["contradiction_flag"]
    seg_lim = theme_assessment.get("segment_limitation", "")

    top_items = evidence_bundle.get("evidence_items") or evidence_bundle.get("supporting_feedback", [])
    top_ids = [item["feedback_id"] for item in top_items[:5]]

    if cid == 6:
        action = "Decompose the cluster into separate functional tracks (search relevance, onboarding, notification delivery) for modular investigation."
        insight = "Evidence indicates multiple disparate UX issues across navigation and onboarding."
    elif decision == "BUILD":
        action = f"Prioritize engineering development for {tname} addressing root cause friction points."
        insight = f"Critical customer friction identified for {tname} with high severity ({m_sev:.2f}/5) across {ev_count} records."
    elif decision == "INVESTIGATE":
        action = f"Conduct targeted customer interviews and diagnostic telemetry for {tname}."
        insight = f"Substantial feedback for {tname} merits exploratory scoping prior to engineering allocation."
    elif decision == "MONITOR":
        action = f"Establish telemetry tracking for incoming feedback regarding {tname}."
        insight = f"Positive experience and low-severity telemetry observed for {tname}."
    else:
        action = f"Defer active development on {tname}; re-evaluate if negative feedback volume increases."
        insight = f"Feedback for {tname} reflects high satisfaction or sub-threshold friction."

    return {
        "insight": insight,
        "customer_problem": tdef or f"Customer feedback regarding {tname}.",
        "recommended_action": action,
        "supporting_feedback_ids": top_ids,
        "confidence": 0.50,
        "limitations": seg_lim or "Deterministic fallback generated.",
    }


def generate_single_insight(
    theme_assessment: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
    client: Optional[Any] = None,
    model: str = DEFAULT_INSIGHT_MODEL,
    max_retries: int = MAX_INSIGHT_RETRIES,
) -> Tuple[Dict[str, Any], List[str], List[str], bool]:
    """
    Generate an insight for a single cluster.
    """
    top_items = evidence_bundle.get("evidence_items") or evidence_bundle.get("supporting_feedback", [])
    valid_ids = [item["feedback_id"] for item in top_items]

    if client is None:
        client = get_gemini_client()

    prompt = build_insight_prompt(theme_assessment, evidence_bundle)

    json_parse_attempts = 2
    for parse_attempt in range(json_parse_attempts):
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                raw_text = response.text if hasattr(response, "text") else str(response)
                parsed, val_ids, inval_ids = parse_and_validate_insight_json(
                    raw_text, valid_feedback_ids=valid_ids
                )
                return parsed, val_ids, inval_ids, False

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = extract_retry_delay_from_error(e)
                    logger.warning(
                        "Rate limit on cluster %d insight generation (attempt %d/%d). Waiting %.1fs...",
                        theme_assessment["cluster_id"],
                        retry_count + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    retry_count += 1
                elif isinstance(e, (json.JSONDecodeError, ValueError)):
                    logger.warning(
                        "JSON parse issue on cluster %d (attempt %d/%d): %s",
                        theme_assessment["cluster_id"],
                        parse_attempt + 1,
                        json_parse_attempts,
                        e,
                    )
                    break
                else:
                    logger.error(
                        "Unexpected error generating insight for cluster %d: %s",
                        theme_assessment["cluster_id"],
                        e,
                    )
                    retry_count += 1
                    time.sleep(2.0)

    logger.warning("Using deterministic fallback insight for cluster %d.", theme_assessment["cluster_id"])
    fallback = build_fallback_insight(theme_assessment, evidence_bundle)
    return fallback, fallback["supporting_feedback_ids"], [], True


def generate_all_product_insights(
    priority_assessments: List[Dict[str, Any]],
    evidence_bundles: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model: str = DEFAULT_INSIGHT_MODEL,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any], Dict[str, int]]:
    """
    Generate grounded product insights for all 8 clusters.
    """
    bundle_map = {b["cluster_id"]: b for b in evidence_bundles}

    insights_json: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    traceability_items: List[Dict[str, Any]] = []

    stats = {
        "gemini_calls": 0,
        "successful_generations": 0,
        "fallback_generations": 0,
        "invalid_ids_returned": 0,
    }

    for assessment in priority_assessments:
        cid = assessment["cluster_id"]
        bundle = bundle_map.get(cid, {})

        stats["gemini_calls"] += 1
        insight_info, val_ids, inval_ids, was_fallback = generate_single_insight(
            theme_assessment=assessment,
            evidence_bundle=bundle,
            client=client,
            model=model,
        )

        if was_fallback:
            stats["fallback_generations"] += 1
        else:
            stats["successful_generations"] += 1

        stats["invalid_ids_returned"] += len(inval_ids)

        # STRICT DETERMINISTIC PRESERVATION:
        # LLM output cannot modify priority_score, decision, evidence_count, strength, diversity, etc.
        record = {
            "cluster_id": int(cid),
            "theme_name": assessment["theme_name"],
            "insight": insight_info["insight"],
            "customer_problem": insight_info["customer_problem"],
            "recommended_action": insight_info["recommended_action"],
            "decision": assessment["decision"],
            "priority_score": float(assessment["priority_score"]),
            "evidence_count": int(assessment["evidence_count"]),
            "evidence_strength": float(assessment["evidence_strength"]),
            "evidence_diversity_score": float(assessment["evidence_diversity_score"]),
            "mean_severity": float(assessment["mean_severity"]),
            "segment_impact_score": float(assessment["segment_impact_score"]),
            "supporting_feedback_ids": insight_info["supporting_feedback_ids"],
            "confidence": float(insight_info["confidence"]),
            "contradiction_flag": bool(assessment["contradiction_flag"]),
            "limitations": insight_info["limitations"],
        }
        insights_json.append(record)

        csv_rows.append({
            "cluster_id": int(cid),
            "theme_name": assessment["theme_name"],
            "insight": insight_info["insight"],
            "customer_problem": insight_info["customer_problem"],
            "recommended_action": insight_info["recommended_action"],
            "decision": assessment["decision"],
            "priority_score": float(assessment["priority_score"]),
            "evidence_count": int(assessment["evidence_count"]),
            "evidence_strength": float(assessment["evidence_strength"]),
            "evidence_diversity_score": float(assessment["evidence_diversity_score"]),
            "mean_severity": float(assessment["mean_severity"]),
            "segment_impact_score": float(assessment["segment_impact_score"]),
            "supporting_feedback_ids": ";".join(insight_info["supporting_feedback_ids"]),
            "confidence": float(insight_info["confidence"]),
            "contradiction_flag": bool(assessment["contradiction_flag"]),
            "limitations": insight_info["limitations"],
        })

        top_items = bundle.get("evidence_items") or bundle.get("supporting_feedback", [])
        prompt_ids = [item["feedback_id"] for item in top_items]

        traceability_items.append({
            "cluster_id": int(cid),
            "theme_name": assessment["theme_name"],
            "prompt_input_evidence_ids": prompt_ids,
            "generated_insight": insight_info["insight"],
            "recommended_action": insight_info["recommended_action"],
            "supporting_feedback_ids": insight_info["supporting_feedback_ids"],
            "deterministic_priority_score": float(assessment["priority_score"]),
            "deterministic_decision": assessment["decision"],
            "validation_status": "VALID" if not inval_ids else "FILTERED",
            "invalid_ids_filtered": inval_ids,
        })

    insights_df = pd.DataFrame(csv_rows)
    traceability_data = {
        "pipeline_version": "1.0",
        "num_insights_generated": len(insights_json),
        "model": model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "insights_traceability": traceability_items,
    }

    return insights_json, insights_df, traceability_data, stats


